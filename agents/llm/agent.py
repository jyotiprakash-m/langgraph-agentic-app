"""
Example LangGraph agent implementation
This is a template showing how to integrate LangGraph with FastAPI
"""
from langgraph.graph import StateGraph, START, END

from agents.llm.state import State
from agents.llm.nodes import chatbot_node, tools

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
# In memory checkpointing for simplicity
memory = MemorySaver()

db_path = "memory.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
sql_memory = SqliteSaver(conn)

class LangGraphAgent:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        
        # Initialize the StateGraph with our State
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("chatbot", chatbot_node)
        graph_builder.add_node("tools", ToolNode(tools=tools))
        
        # Add edges
        graph_builder.add_conditional_edges("chatbot", tools_condition, ["tools", END])
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")

        # Compile the graph
        return graph_builder.compile(checkpointer=sql_memory)

# Create a global agent instance
agent = LangGraphAgent()