from fastapi import FastAPI

from ai_companion.interfaces.whatsapp.whatsapp_response import whatsapp_router

app = FastAPI()
app.include_router(whatsapp_router)

@app.get("/")
async def healthcheck():
    """
    Endpoint de healthcheck para verificar que la API está funcionando.
    Retorna 200 OK con información básica del servicio.
    """
    return {
        "status": "ok",
        "service": "WhatsApp Agent API",
        "version": "1.0.0",
    }