import os
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from ai_companion.graph.state import AICompanionState
from ai_companion.graph.utils.chains import (
    get_character_response_chain,
    get_router_chain,
)
from ai_companion.graph.utils.helpers import (
    get_chat_model,
    get_text_to_image_module,
    get_text_to_speech_module,
)
from ai_companion.modules.memory.long_term.memory_manager import get_memory_manager
from ai_companion.modules.schedules.context_generation import ScheduleContextGenerator
from ai_companion.settings import settings


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


async def conversation_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    return {"messages": AIMessage(content=response)}


async def image_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_image_module = get_text_to_image_module()

    scenario = await text_to_image_module.create_scenario(state["messages"][-5:])
    os.makedirs("generated_images", exist_ok=True)
    img_path = f"generated_images/image_{str(uuid4())}.png"
    await text_to_image_module.generate_image(scenario.image_prompt, img_path)

    # Inject the image prompt information as an AI message
    scenario_message = HumanMessage(content=f"<image attached by Ava generated from prompt: {scenario.image_prompt}>")
    updated_messages = state["messages"] + [scenario_message]

    response = await chain.ainvoke(
        {
            "messages": updated_messages,
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )

    return {"messages": AIMessage(content=response), "image_path": img_path}


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
        
        Does this query require looking up real estate information? If not, respond only with: "NO_SEARCH_NEEDED".
        
        If YES, the query requires real estate information, generate a JSON with exactly this format:
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
            "includeExamples": boolean        // Whether to include examples in the response
        }}
        ```
        
        Only include parameters that are explicitly mentioned or implied in the user's query.
        Respond only with "NO_SEARCH_NEEDED" or the valid, parseable JSON, without additional explanations.
        """)
    ]
    
    # Evaluate need and generate parameters
    response = await model.ainvoke(unified_prompt)
    
    # If no search is needed, continue the flow
    if "NO_SEARCH_NEEDED" in response.content:
        return {"needs_api": False}
    
    try:
        # Extract JSON from the response
        import json
        import re
        
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*?})', response.content)
        if not json_match:
            # No valid JSON found
            return {"needs_api": False, "api_info": "Error: Invalid parameter format"}
        
        # Get the JSON content (from group 1 or 2, depending on the format)
        json_content = json_match.group(1) if json_match.group(1) else json_match.group(2)
        search_params = json.loads(json_content)
        
        # Import the API client
        from ai_companion.modules.api import get_api_client
        
        api_client = get_api_client()
        
        # Query the API with the generated parameters
        search_results = await api_client.search(search_params)
        
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
        

