"""
Example LangGraph agent implementation
This is a template showing how to integrate LangGraph with FastAPI
"""
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

from agents.sample.state import State
from agents.sample.nodes import our_first_node

# Load environment variables
load_dotenv()


class LangGraphAgent:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        # Initialize the StateGraph with our State
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("first_node", our_first_node)
        
        # Add edges
        graph_builder.add_edge(START, "first_node")
        graph_builder.add_edge("first_node", END)
        
        # Compile the graph
        return graph_builder.compile()

# Create a global agent instance
agent = LangGraphAgent()