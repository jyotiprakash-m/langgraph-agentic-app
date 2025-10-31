"""
Example LangGraph agent implementation
This is a template showing how to integrate LangGraph with FastAPI
"""
from langgraph.graph import StateGraph, START, END

from agents.llm.state import State
from agents.llm.nodes import chatbot_node, tools

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

db_path = "memory.db"

class LangGraphAgent:
    def __init__(self, graph):
        self.graph = graph

    @classmethod
    async def create(cls):
        # Initialize the StateGraph with our State
        graph_builder = StateGraph(State)
        # Add nodes
        graph_builder.add_node("chatbot", chatbot_node)
        graph_builder.add_node("tools", ToolNode(tools=tools))
        # Add edges
        graph_builder.add_conditional_edges("chatbot", tools_condition, ["tools", END])
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")
        # Compile the graph with async checkpointer
        graph = graph_builder.compile(checkpointer=AsyncSqliteSaver(db_path))  # type: ignore
        return cls(graph)

# Usage:
# agent = await LangGraphAgent.create()