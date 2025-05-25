from langgraph.graph import END
from typing_extensions import Literal

from ai_companion.graph.state import AICompanionState
from ai_companion.settings import settings
from ai_companion.modules.schedules.context_generation import ScheduleContextGenerator


def should_summarize_conversation(
    state: AICompanionState,
) -> Literal["summarize_conversation_node", "__end__"]:
    messages = state["messages"]

    if len(messages) > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "summarize_conversation_node"

    return END


def select_workflow(
    state: AICompanionState,
) -> Literal["conversation_node", "audio_node"]:
    workflow = state["workflow"]

    if workflow == "audio":
        return "audio_node"

    else:
        return "conversation_node"
    
def should_send_project_card(
    state: AICompanionState,
) -> Literal["project_card_node", "continue_conversation_node"]:
    """
    Uses AI to determine if a project card should be sent or if normal conversation should continue.
    
    Returns:
        project_card_node: If a specific project is identified
        continue_conversation_node: If no specific project is found
    """
    api_info = state.get("api_info", {})
    messages = state.get("messages", [])
    current_activity = state.get("current_activity", ScheduleContextGenerator.get_current_activity())
    memory_context = state.get("memory_context", "")
    
    # Import here to avoid circular imports
    from ai_companion.graph.utils.helpers import get_chat_model
    from langchain_core.messages import HumanMessage
    import json
    
    # Check if we have enough information to make a decision
    if not messages:
        return "continue_conversation_node"
    
    model = get_chat_model()
    
    prompt = [
        HumanMessage(content=f"""
        Analyze the following information to determine if the user is interested in ONE SPECIFIC REAL ESTATE PROJECT.

        LATEST USER MESSAGE:
        {messages[-1].content if messages else ""}
        
        MEMORY CONTEXT:
        {memory_context}
        
        CURRENT ACTIVITY:
        {current_activity}
        
        API SEARCH RESULTS:
        {json.dumps(api_info, indent=2)[:1000] + "..." if api_info and len(json.dumps(api_info)) > 1000 else json.dumps(api_info) if api_info else "No API results available"}
        
        TASK:
        Determine if the user has expressed interest in ONE specific project or is still considering multiple options.
        
        Conditions for sending a project card:
        1. Information about a specific project exists (either in API results, memory or conversation)
        2. The user has shown particular interest in ONE specific project (by name or characteristics)
        3. There is no ambiguity about which project the user is interested in
        
        Respond ONLY in JSON format with this structure:
        {{
          "should_send_card": boolean,  // true if card should be sent, false if not
          "project_id": number | null,  // ID of the project if there's a specific one, null if not
          "reasoning": string           // Brief explanation of why you made this decision
        }}
        """)
    ]
    print(f"Prompt for should_send_project_card: {prompt[0].content}")
    try:
        response = model.invoke(prompt)
        result = json.loads(response.content)
        
        # Only send card if confidence is high or medium
        if (result.get("should_send_card", False) and 
            result.get("project_id")):
            
            # Save project ID in state for project_card node to use
            state["project_id"] = result["project_id"]
            return "project_card_node"
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error in should_send_project_card: {str(e)}")
    
    return "continue_conversation_node"
