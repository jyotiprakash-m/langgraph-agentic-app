from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field
from agents.sidekick.tools import playwright_tools, other_tools
from agents.sidekick.nodes import worker, worker_router, evaluator, route_based_on_evaluation
import uuid
import asyncio
from datetime import datetime
from agents.sidekick.state import State
import functools

load_dotenv(override=True)

class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )

class Sidekick:
    def __init__(self):
        self.worker_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.tools = None
        self.llm_with_tools = None
        self.graph = None
        self.sidekick_id = str(uuid.uuid4())
        self.memory = MemorySaver()
        self.browser = None
        self.playwright = None

    # --- Wrapper methods for tracing ---
    def worker_node(self, state):
        return worker(self, state)

    def evaluator_node(self, state):
        return evaluator(self, state)

    def worker_router_node(self, state):
        return worker_router(self, state)

    def route_based_on_evaluation_node(self, state):
        return route_based_on_evaluation(self, state)

    async def setup(self):
        self.tools, self.browser, self.playwright = await playwright_tools()
        self.tools += await other_tools()
        worker_llm = ChatOpenAI(model="gpt-4o-mini")
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)
        evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)
        await self.build_graph()

    async def build_graph(self):
        # Set up Graph Builder with State
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("worker", self.worker_node)
        graph_builder.add_node("tools", ToolNode(tools=self.tools or []))
        graph_builder.add_node("evaluator", self.evaluator_node)

        # Add edges
        graph_builder.add_conditional_edges(
            "worker", self.worker_router_node, {"tools": "tools", "evaluator": "evaluator"}
        )
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges(
            "evaluator", self.route_based_on_evaluation_node, {"worker": "worker", "END": END}
        )
        graph_builder.add_edge(START, "worker")

        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=self.memory)

    async def run_superstep(self, message, success_criteria, history):
        config = {"configurable": {"thread_id": self.sidekick_id}}

        if self.graph is None:
            raise RuntimeError("Sidekick.setup() must be called and awaited before run_superstep().")

        state = State(
            messages=message,
            success_criteria=success_criteria or "The answer should be clear and accurate",
            feedback_on_work=None,
            success_criteria_met=False,
            user_input_needed=False,
        )
        result = await self.graph.ainvoke(state, config=config) # type: ignore
        user = {"role": "user", "content": message}
        reply = {"role": "assistant", "content": result["messages"][-2].content}
        feedback = {"role": "assistant", "content": result["messages"][-1].content}
        return history + [user, reply, feedback]

    def cleanup(self):
        if self.browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except RuntimeError:
                # If no loop is running, do a direct run
                asyncio.run(self.browser.close())
                if self.playwright:
                    asyncio.run(self.playwright.stop())