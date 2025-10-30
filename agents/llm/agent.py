"""
Example LangGraph agent implementation
This is a template showing how to integrate LangGraph with FastAPI
"""
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

from agents.llm.state import State
from agents.llm.nodes import chatbot_node

class LangGraphAgent:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        # Initialize the StateGraph with our State
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("chatbot_node", chatbot_node)
        
        # Add edges
        graph_builder.add_edge(START, "chatbot_node")
        graph_builder.add_edge("chatbot_node", END)
        
        # Compile the graph
        return graph_builder.compile()

# Create a global agent instance
agent = LangGraphAgent()