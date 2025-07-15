from langgraph.graph import StateGraph, END

from .state import InvestigationState
# Make sure you are importing the correct agent function
from .agents import web_search_agent 

# Define the graph
workflow = StateGraph(InvestigationState)

# Add the nodes (agents)
# This line now adds our Gemini agent because we updated the imported function
workflow.add_node("web_search", web_search_agent)

# Set the entry point
workflow.set_entry_point("web_search")

# Add the edges (for now, a simple end)
workflow.add_edge("web_search", END)

# Compile the graph
app = workflow.compile()