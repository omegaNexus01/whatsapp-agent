import logging
import os
from io import BytesIO
from typing import Dict

import httpx
from fastapi import APIRouter, Request, Response, BackgroundTasks
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ai_companion.graph import graph_builder
from ai_companion.modules.image import ImageToText
from ai_companion.modules.speech import SpeechToText, TextToSpeech
from ai_companion.settings import settings

logger = logging.getLogger(__name__)

# Global module instances
speech_to_text = SpeechToText()
text_to_speech = TextToSpeech()
image_to_text = ImageToText()

# Router for WhatsApp responses
whatsapp_router = APIRouter()

# WhatsApp API credentials
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")


async def _process_info_point_async(
    to_number: str,
    info_point_type: str,
    info_point_param: Dict[str, any],
):
    """
    Función en segundo plano para llamar a la API externa de info_point.
    No retorna nada al endpoint original.
    """
    try:
        url = f"https://cgg3hg4s-7001.uks1.devtunnels.ms/chatbot/whatsapp/{info_point_type}"
        payload = {"to": f"+{to_number}", **info_point_param}

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            logger.info(f"Info point {info_point_type} enviado correctamente a {to_number}")
    except Exception as e:
        logger.error(f"Error en info_point ({info_point_type}) para {to_number}: {e}")


@whatsapp_router.api_route("/whatsapp_response", methods=["GET", "POST"])
async def whatsapp_handler(
    request: Request, background_tasks: BackgroundTasks
) -> Response:
    """Handles incoming messages and status updates from the WhatsApp Cloud API."""

    if request.method == "GET":
        params = request.query_params
        if params.get("hub.verify_token") == os.getenv("WHATSAPP_VERIFY_TOKEN"):
            return Response(content=params.get("hub.challenge"), status_code=200)
        return Response(content="Verification token mismatch", status_code=403)

    try:
        data = await request.json()
        change_value = data["entry"][0]["changes"][0]["value"]

        if "messages" in change_value:
            message = change_value["messages"][0]
            from_number = message["from"]
            session_id = from_number

            # Obtener el contenido del usuario
            if message["type"] == "audio":
                content = await process_audio_message(message)
            elif message["type"] == "image":
                content = message.get("image", {}).get("caption", "")
                image_bytes = await download_media(message["image"]["id"])
                try:
                    description = await image_to_text.analyze_image(
                        image_bytes,
                        "Please describe what you see in this image in the context of our conversation.",
                    )
                    content += f"\n[Image Analysis: {description}]"
                except Exception as e:
                    logger.warning(f"Failed to analyze image: {e}")
            else:
                content = message["text"]["body"]

            # Invocar el grafo
            async with AsyncSqliteSaver.from_conn_string(
                settings.SHORT_TERM_MEMORY_DB_PATH
            ) as short_term_memory:
                graph = graph_builder.compile(checkpointer=short_term_memory)
                await graph.ainvoke(
                    {"messages": [HumanMessage(content=content)]},
                    {"configurable": {"thread_id": session_id}},
                )
                output_state = await graph.aget_state(
                    config={"configurable": {"thread_id": session_id}}
                )

            workflow = output_state.values.get("workflow", "conversation")
            response_message = output_state.values["messages"][-1].content
            print(f"workflow: {workflow}")

            # 1) Si es info_point: agendar en background y devolver ya 200
            if workflow == "info_point":
                info_point_type = output_state.values.get("info_point")
                info_point_param = output_state.values.get("info_point_param", {})

                # Encolar la llamada en segundo plano
                background_tasks.add_task(
                    _process_info_point_async,
                    from_number,
                    info_point_type,
                    info_point_param,
                )

                # Confirmamos recibo inmediato a Meta
                return Response(content="Received", status_code=200)

            # 2) Otros workflows: audio, image o texto
            if workflow == "audio":
                audio_buffer = output_state.values["audio_buffer"]
                success = await send_response(
                    from_number, response_message, "audio", audio_buffer
                )
            elif workflow == "image":
                image_path = output_state.values["image_path"]
                with open(image_path, "rb") as f:
                    img_data = f.read()
                success = await send_response(
                    from_number, response_message, "image", img_data
                )
            else:
                success = await send_response(from_number, response_message, "text")

            if not success:
                return Response(content="Failed to send message", status_code=500)

            return Response(content="Message processed", status_code=200)

        elif "statuses" in change_value:
            return Response(content="Status update received", status_code=200)
        else:
            return Response(content="Unknown event type", status_code=400)

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        return Response(content="Internal server error", status_code=500)


async def download_media(media_id: str) -> bytes:
    """Download media from WhatsApp."""
    media_metadata_url = f"https://graph.facebook.com/v21.0/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    async with httpx.AsyncClient() as client:
        meta = await client.get(media_metadata_url, headers=headers)
        meta.raise_for_status()
        url = meta.json().get("url")
        media = await client.get(url, headers=headers)
        media.raise_for_status()
        return media.content


async def process_audio_message(message: Dict) -> str:
    """Download and transcribe audio message."""
    audio_id = message["audio"]["id"]
    media_metadata_url = f"https://graph.facebook.com/v21.0/{audio_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

    async with httpx.AsyncClient() as client:
        meta = await client.get(media_metadata_url, headers=headers)
        meta.raise_for_status()
        url = meta.json().get("url")
    async with httpx.AsyncClient() as client:
        audio = await client.get(url, headers=headers)
        audio.raise_for_status()
        buffer = BytesIO(audio.content)
        return await speech_to_text.transcribe(buffer.read())


async def send_response(
    from_number: str,
    response_text: str,
    message_type: str = "text",
    media_content: bytes = None,
) -> bool:
    """Send response to user via WhatsApp API."""
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    if message_type in ["audio", "image"]:
        try:
            mime_type = "audio/mpeg" if message_type == "audio" else "image/png"
            media_id = await upload_media(BytesIO(media_content), mime_type)
            payload = {
                "messaging_product": "whatsapp",
                "to": from_number,
                "type": message_type,
                message_type: {"id": media_id},
            }
            if message_type == "image":
                payload["image"]["caption"] = response_text
        except Exception as e:
            logger.error(f"Media upload failed, falling back to text: {e}")
            message_type = "text"

    if message_type == "text":
        payload = {
            "messaging_product": "whatsapp",
            "to": from_number,
            "type": "text",
            "text": {"body": response_text},
        }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/messages",
            headers=headers,
            json=payload,
        )
    return resp.status_code == 200


async def upload_media(media_buffer: BytesIO, mime_type: str) -> str:
    """Upload media to WhatsApp servers."""
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    files = {"file": ("file", media_buffer, mime_type)}
    data = {"messaging_product": "whatsapp", "type": mime_type}

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/media",
            headers=headers,
            files=files,
            data=data,
        )
        resp.raise_for_status()
        result = resp.json()
    return result["id"]
