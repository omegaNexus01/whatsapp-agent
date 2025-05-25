from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from ai_companion.graph.edges import (
    select_workflow,
    should_summarize_conversation,
    should_send_project_card,
)
from ai_companion.graph.nodes import (
    audio_node,
    context_injection_node,
    conversation_node,
    project_card_node,
    memory_extraction_node,
    memory_injection_node,
    router_node,
    summarize_conversation_node,
    search_node,
    
)
from ai_companion.graph.state import AICompanionState


@lru_cache(maxsize=1)
def create_workflow_graph():
    graph_builder = StateGraph(AICompanionState)

    # Add all nodes
    graph_builder.add_node("memory_extraction_node", memory_extraction_node)
    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("context_injection_node", context_injection_node)
    graph_builder.add_node("memory_injection_node", memory_injection_node)
    graph_builder.add_node("conversation_node", conversation_node)
    graph_builder.add_node("continue_conversation_node", conversation_node)  # Alias para conversation_node
    graph_builder.add_node("project_card_node", project_card_node)
    graph_builder.add_node("audio_node", audio_node)
    graph_builder.add_node("summarize_conversation_node", summarize_conversation_node)
    graph_builder.add_node("search_node", search_node)
    
   

    # Define the flow
    # First extract memories from user message
    graph_builder.add_edge(START, "memory_extraction_node")

    # Then determine response type
    graph_builder.add_edge("memory_extraction_node", "router_node")

    # Then inject both context and memories
    graph_builder.add_edge("router_node", "context_injection_node")
    graph_builder.add_edge("context_injection_node", "memory_injection_node")

    # After memory injection, check if search is needed
    graph_builder.add_edge("memory_injection_node", "search_node")
    
    # After search, proceed to appropriate response node based on workflow type
    graph_builder.add_conditional_edges("search_node", select_workflow)

    graph_builder.add_conditional_edges("conversation_node", should_send_project_card)
    
    # Check for summarization after any response
    graph_builder.add_conditional_edges("continue_conversation_node", should_summarize_conversation)
    graph_builder.add_conditional_edges("project_card_node", should_summarize_conversation)
    graph_builder.add_conditional_edges("audio_node", should_summarize_conversation)
    graph_builder.add_edge("summarize_conversation_node", END)

    return graph_builder


# Compiled without a checkpointer. Used for LangGraph Studio
graph = create_workflow_graph().compile()
