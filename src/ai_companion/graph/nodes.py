import os
from uuid import uuid4
import logging

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from ai_companion.graph.state import AICompanionState
from ai_companion.graph.utils.chains import (
    get_character_response_chain,
    get_router_chain,
)
from ai_companion.graph.utils.helpers import (
    get_chat_model,
    get_text_to_speech_module,
)
from ai_companion.modules.memory.long_term.memory_manager import get_memory_manager
from ai_companion.modules.schedules.context_generation import ScheduleContextGenerator
from ai_companion.settings import settings

logger = logging.getLogger(__name__)


async def router_node(state: AICompanionState):
    chain = get_router_chain()
    response = await chain.ainvoke({"messages": state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE :]})
    return {"workflow": response.response_type}


def context_injection_node(state: AICompanionState):
    schedule_context = ScheduleContextGenerator.get_current_activity()
    if schedule_context != state.get("current_activity", ""):
        apply_activity = True
    else:
        apply_activity = False
    return {"apply_activity": apply_activity, "current_activity": schedule_context}


# En nodes.py - Siguiendo el patrón del ejemplo
async def conversation_node(state: AICompanionState, config: RunnableConfig):
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    from langchain.prompts import ChatPromptTemplate
    from ai_companion.graph.tools import send_project_card, current_thread_id,search_projects,send_unit_card
    import json
    
    # Obtener thread_id del config como en el ejemplo
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    current_thread_id.set(thread_id)
    
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")
    messages = state.get("messages", [])
    
    tools = [send_project_card,search_projects, send_unit_card]
    
    system_prompt = f"""
    Eres un asistente especializado en bienes raíces que ayuda a encontrar la propiedad perfecta.

    CONTEXTO DE MEMORIA:
    {memory_context}
    
    ACTIVIDAD ACTUAL:
    {current_activity}

    CONVERSACIÓN COMPLETA:
    {chr(10).join([f"{msg.type}: {msg.content}" for msg in messages[-10:]])}

    HERRAMIENTAS DISPONIBLES:
    1. search_projects: Para buscar proyectos inmobiliarios
    2. send_project_card: Para enviar tarjetas detalladas de proyectos específicos
    3. send_unit_card: Para enviar tarjetas detalladas de unidades específicas

    FLUJO DE TRABAJO INTELIGENTE:
    
    CUÁNDO BUSCAR (usar search_projects):
    - Usuario pregunta sobre proyectos disponibles: "¿Qué proyectos hay en Miami?"
    - Usuario busca por características: "Apartamentos de 3 habitaciones"
    - Usuario pregunta por ubicación: "¿Qué hay cerca de Brickell?"
    - Usuario pregunta por precios: "Apartamentos bajo $500,000"
    - Usuario pregunta por unidades específicas: "¿Qué unidades hay en el proyecto Torre Platinum?"

    CUÁNDO ENVIAR TARJETA DE PROYECTO (usar send_project_card):
    - Usuario muestra interés específico en UN PROYECTO: "Quiero más información sobre Torre Platinum"
    - Usuario pide detalles del proyecto: "Envíame los detalles del proyecto 1428 Brickell"
    - Usar project_id (numérico) de los resultados de búsqueda

    CUÁNDO ENVIAR TARJETA DE UNIDAD (usar send_unit_card):
    - Usuario quiere una unidad de un proyecto específico: "Envíame info de la unidad en el proyecto 1428 Brickell"
    - Usuario pide detalles de una unidad específica: "Envíame info de una unidad de 3 habitaciones en ese proyecto"
    - Usar unit_id (UUID) de los resultados de búsqueda

    CUÁNDO SOLO CONVERSAR (sin herramientas):
    - Saludos: "Hola", "¿Cómo estás?"
    - Preguntas generales sobre ti
    - Conversación casual

    ESTRATEGIA INTELIGENTE:
    1. Si el usuario pregunta sobre proyectos y no tienes información reciente → BUSCAR PRIMERO
    2. Presentar resultados de forma amigable
    3. Si el usuario muestra interés en UN proyecto específico → ENVIAR TARJETA
    4. Si el usuario pregunta por una unidad específica → ENVIAR TARJETA DE UNIDAD
    5. Puedes usar múltiples herramientas en secuencia según sea necesario
    
    

    EJEMPLOS DE FLUJOS:

    EJEMPLO 1 - Búsqueda nueva:
    Usuario: "¿Qué apartamentos hay en Miami?"
    → Usar search_real_estate_projects(semantic_query="apartamentos en Miami", search_in=["projects"])
    → Presentar resultados amigablemente
    → Esperar respuesta del usuario

    EJEMPLO 2 - Interés específico después de búsqueda:
    Usuario: "Me interesa Torre Platinum"
    → Usar send_project_card(project_id=XXX, project_name="Torre Platinum")
    (project_id debe venir de resultados de búsqueda previos)

    EJEMPLO 3 - Flujo completo:
    Usuario: "Busco apartamentos de 3 habitaciones en Brickell"
    → Usar search_real_estate_projects(semantic_query="apartamentos Brickell", bedrooms=3)
    → Presentar opciones encontradas
    Usuario: "Quiero más información sobre 1428 Brickell"
    → Usar send_project_card(project_id=395, project_name="1428 Brickell Residences")
    
    EJEMPLO 4 - Búsqueda y tarjeta de unidad:
    Usuario: "Busco apartamentos de 3 habitaciones"
    → Usar search_projects(bedrooms=3)
    → Presentar resultados con unidades disponibles
    Usuario: "Quiero información de una unidad con 3 habitaciones en el proyecto Torre Platinum"
    → Usar send_unit_card(unit_id="46831b47-c4e5-451d-8b74-6058ebbf639b", unit_name="3 Beds - North 05")
    
    ⚠️ REGLA CRÍTICA PARA UNIT IDs:
    - NUNCA inventes unit_ids
    - NUNCA uses placeholders como "unit_id_de_la_unidad_XXX"
    - SIEMPRE usa el EXACT unitId de los resultados de búsqueda
    - Los unit_ids reales se ven así: "46831b47-c4e5-451d-8b74-6058ebbf639b" o "4196"

    EJEMPLO CORRECTO:
    Si en búsqueda ves:
    • Unit ID: 46831b47-c4e5-451d-8b74-6058ebbf639b
    • Unit Name: 1 Bed - 01

    Entonces usar: send_unit_card(unit_id="46831b47-c4e5-451d-8b74-6058ebbf639b", unit_name="1 Bed - 01")

    EJEMPLO INCORRECTO (NO HACER):
    send_unit_card(unit_id="unit_id_de_la_unidad_1_Bed_01", unit_name="1 Bed - 01") ❌

    IMPORTANTE: 
    - SIEMPRE busca primero si no tienes información sobre los proyectos
    - Extrae project_id de los resultados de búsqueda para send_project_card
    - Sé natural y conversacional
    - Si hay dudas, es mejor preguntar al usuario que asumir
    - Los project_id son numéricos (ej: 395)
    - Los unit_id son UUIDs (ej: "46831b47-c4e5-451d-8b74-6058ebbf639b")
    - Extrae los IDs correctos de los resultados de búsqueda
    - Puedes usar múltiples herramientas en secuencia
    
    NOTA: No tienes información previa de búsqueda. Si el usuario pregunta sobre proyectos, 
    DEBES usar search_real_estate_projects para obtener información actualizada.
    """
    
    llm = get_chat_model()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        max_iterations=3, 
        early_stopping_method="force",
        return_intermediate_steps=True,
    )
    
    current_input = messages[-1].content if messages else ""
    
    try:
        result = await agent_executor.ainvoke({"input": current_input})
        
        if "intermediate_steps" in result:
            print(f"🔧 Herramientas ejecutadas:")
            for i, (action, observation) in enumerate(result["intermediate_steps"], 1):
                print(f"  {i}. Herramienta: {action.tool}")
                print(f"     Parámetros: {action.tool_input}")
                
        return {"messages": AIMessage(content=result["output"])}
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error: {str(e)}")
        
        # Fallback como en el ejemplo
        chain = get_character_response_chain(state.get("summary", ""))
        response = await chain.ainvoke({
            "messages": messages,
            "current_activity": current_activity,
            "memory_context": memory_context,
        }, config)
        return {"messages": AIMessage(content=response)}




async def audio_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_speech_module = get_text_to_speech_module()

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    output_audio = await text_to_speech_module.synthesize(response)

    return {"messages": response, "audio_buffer": output_audio}


async def summarize_conversation_node(state: AICompanionState):
    model = get_chat_model()
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is summary of the conversation to date between Ava and the user: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = (
            "Create a summary of the conversation above between Ava and the user. "
            "The summary must be a short description of the conversation so far, "
            "but that captures all the relevant information shared between Ava and the user:"
        )

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]]
    return {"summary": response.content, "messages": delete_messages}


async def memory_extraction_node(state: AICompanionState):
    """Extract and store important information from the last message."""
    if not state["messages"]:
        return {}

    memory_manager = get_memory_manager()
    await memory_manager.extract_and_store_memories(state["messages"][-1])
    return {}


def memory_injection_node(state: AICompanionState):
    """Retrieve and inject relevant memories into the character card."""
    memory_manager = get_memory_manager()

    # Get relevant memories based on recent conversation
    recent_context = " ".join([m.content for m in state["messages"][-3:]])
    memories = memory_manager.get_relevant_memories(recent_context)

    # Format memories for the character card
    memory_context = memory_manager.format_memories_for_prompt(memories)

    return {"memory_context": memory_context}

async def search_node(state: AICompanionState, config: RunnableConfig):
    """
    Evaluates if real estate API information is needed, generates appropriate query parameters 
    and processes the results in a single step.
    """
    model = get_chat_model()
    memory_context = state.get("memory_context", "")
    recent_messages = state["messages"][-3:]
    query = state["messages"][-1].content
    
    # Unified prompt to determine search need and generate parameters (in English)
    unified_prompt = [
        HumanMessage(content=f"""
        Analyze this user query: "{query}"
        
        Recent conversation context:
        {[m.content for m in recent_messages]}
        
        Information in memory:
        {memory_context}
        
        Does this query require looking up projects information? If not, respond only with: "NO_SEARCH_NEEDED".
        
        If YES, the query requires projects information, generate a JSON with exactly this format:
        ```json
        {{
            "nameQuery": string | null,       // Specific text to search by name (e.g., "Retiro")
            "semanticQuery": string | null,   // Natural language query (e.g., "apartment near downtown")
            "searchIn": string[],             // Array with at least one of: "zones", "projects", "developers", "pois" 
            "params": {{                      // Optional parameters to filter results
                "bedrooms": number | null,    // Number of bedrooms
                "minPrice": number | null,    // Minimum price
                "maxPrice": number | null,    // Maximum price
                "propertyType": string | null // Property type (e.g., "apartment", "house")
            }},
            "flexibleSearch": boolean,        // Whether to allow flexible search
            "useNameGuesser": boolean        // Whether to use the NameGuesser for better results every true
        }}
        ```
        
        Only include parameters that are explicitly mentioned or implied in the user's query.
        Respond only with "NO_SEARCH_NEEDED" or the valid, parseable JSON, without additional explanations.
        """)
    ]
    
    # Evaluate need and generate parameters
    response = await model.ainvoke(unified_prompt)
    print(f"Search node response: {response.content}")
    
    # If no search is needed, continue the flow
    if "NO_SEARCH_NEEDED" in response.content:
        return {"needs_api": False}
    
    try:
        # Extract JSON from the response
        import json
        import re
        
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*?})', response.content)
        print(f"JSON match: {json_match}")
        if not json_match:
            # No valid JSON found
            return {"needs_api": False, "api_info": "Error: Invalid parameter format"}
        
        # Get the JSON content (from group 1 or 2, depending on the format)
        json_content = json_match.group(1) if json_match.group(1) else json_match.group(2)
        search_params = json.loads(json_content)
        print(f"Search parameters: {search_params}")
        
        # Import the API client
        from ai_companion.modules.api import APIClient
        
        api_client = APIClient()
        
        # Query the API with the generated parameters
        search_results = await api_client.search(search_params)
        print(f"Search results: {search_results}")
        
        # Process and format results for context
        formatted_results = await format_search_results(search_results, search_params)
        
        # Create a message with the retrieved information
        search_message = HumanMessage(content=formatted_results)
        
        return {
            "needs_api": True, 
            "api_info": search_results, 
            "messages": search_message, 
            "api_params": search_params
        }
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in search_node: {str(e)}")
        return {"needs_api": False, "api_info": f"Error processing search: {str(e)}"}
    
async def format_search_results(search_results, search_params):
    """
    Formats search results to present them to the user in a readable way.
    """
    model = get_chat_model()
    
    prompt = [
        HumanMessage(content=f"""
        I need you to convert these real estate search results into clear and concise text for the user.
        
        The original query was:
        {search_params.get("nameQuery") or search_params.get("semanticQuery") or "Real estate search"}
        
        API Results:
        ```json
        {search_results}
        ```
        
        Please format these results in natural language, highlighting:
        1. Total number of results found
        2. For each zone/region, mention number of relevant projects and units
        3. If there are property examples, mention 2-3 notable ones with their main features
        4. If flexible search was used, briefly explain what criteria were made flexible
        
        The format should be conversational friendly, avoiding unnecessary technical terms.
        Don't mention internally that this data comes from an API.
        Don't use markdown or special formatting, just plain text.
        """)
    ]
    
    response = await model.ainvoke(prompt)
    return response.content

async def project_card_node(state: AICompanionState, config: RunnableConfig):
    """
    Sends a project card when a specific project has been identified.
    No additional response is generated as the card serves as the response.
    """
    # Get project ID from state (set by should_send_project_card)
    project_id = state.get("project_id")
    print(f"Project ID in project_card_node: {project_id}")
    # Extract user's phone number from config
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    
    if not project_id:
        # Fallback to conversation if no project ID found
        logger.warning("Project card requested but no project ID found")
        return await conversation_node(state, config)
    
    # Send the project card via API client
    try:
        # Use the APIClient to send the card
        from ai_companion.modules.api import APIClient
        api_client = APIClient()
        
        card_result = await api_client.send_project_card(int(project_id), thread_id)
        print(f"Card result: {card_result}")
        
        if card_result.get("success", True):
            logger.info(f"Project card sent for project ID {project_id} to {thread_id}")
            
            # Return empty message - no text response needed as the card is the response
            return {"project_id": project_id}
        else:
            logger.error(f"Failed to send project card: {card_result.get('message')}")
            # Fallback to conversation if card sending failed
            return await conversation_node(state, config)
            
    except Exception as e:
        logger.error(f"Error sending project card: {str(e)}")
        # Fallback to conversation if an exception occurred
        return await conversation_node(state, config)

