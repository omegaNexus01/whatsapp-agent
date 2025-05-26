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
) -> Literal["conversation_node", "audio_node", "project_card_node"]:
    """
    Selects the appropriate workflow based on message type and content analysis.
    
    Returns:
        audio_node: If an audio response is requested
        conversation_node: For all other standard text conversations
    """
    workflow = state["workflow"]
    

    # First check for audio workflow
    if workflow == "audio":
        return "audio_node"
    
    else:
        return "conversation_node"
    
