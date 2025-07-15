from langgraph.graph import StateGraph, END
from .state import InvestigationState
from .agents import web_search_agent, entity_extraction_agent

# Define the graph
workflow = StateGraph(InvestigationState)

# the nodes (agents)
workflow.add_node("web_search", web_search_agent)

workflow.add_node("entity_extraction", entity_extraction_agent)

# the entry point
workflow.set_entry_point("web_search")

# edges to define the flow
workflow.add_edge("web_search", "entity_extraction")
workflow.add_edge("entity_extraction", END)

# Compile the graph
app = workflow.compile()