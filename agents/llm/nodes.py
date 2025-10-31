import random
import requests
from agents.llm.state import State
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
import os
import nest_asyncio
import asyncio

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser

nest_asyncio.apply()
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# --- Tool definitions ---

# tool: web search
serper = GoogleSerperAPIWrapper()
tool_search = Tool(
    name="search",
    func=serper.run,
    description="Useful for when you need more information from an online search"
)

# tool: push notification
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"
def push(text: str):
    """Send a push notification to the user"""
    try:
        response = requests.post(
            pushover_url,
            data={"token": pushover_token, "user": pushover_user, "message": text},
            timeout=10
        )
        response.raise_for_status()
        return {"role": "tool", "content": "Push notification sent successfully."}
    except requests.RequestException as e:
        return {"role": "tool", "content": f"Failed to send push notification: {str(e)}"}

tool_push = Tool(
    name="send_push_notification",
    func=push,
    description="useful for when you want to send a push notification"
)

# --- NEW: Async setup for Playwright and tools ---
def wrap_tool(tool):
    async def wrapped(*args, **kwargs):
        result = await tool.invoke(*args, **kwargs) if asyncio.iscoroutinefunction(tool.invoke) else tool.invoke(*args, **kwargs)
        if isinstance(result, str):
            return {"role": "tool", "content": result}
        return result
    return wrapped

async def setup_agent():
    browser_instance = create_async_playwright_browser(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser_instance)
    wrapped_playwright_tools = [
        Tool(
            name=t.name,
            func=wrap_tool(t),
            description=t.description
        )
        for t in toolkit.get_tools()
    ]
    global tools
    tools = [tool_push]
    global llm_with_tools
    llm_with_tools = llm.bind_tools(tools)
    return browser_instance

# Run the async setup at module load
browser_to_close = asyncio.run(setup_agent())

# --- Node definitions ---

async def chatbot_node(state: State) -> State:
    """
    Chatbot node that processes user messages and generates AI responses.
    Args:
        state: Current state containing conversation messages
    Returns:
        Updated state with AI response message
    """
    try:
        response = await llm_with_tools.ainvoke(state["messages"])
        if isinstance(response, str):
            response = {"role": "assistant", "content": response}
        new_state = State(messages=state["messages"] + [response])
        return new_state
    except Exception as e:
        error_message = {
            "role": "assistant",
            "content": f"I encountered an error processing your request: {str(e)}"
        }
        return State(messages=state["messages"] + [error_message])

# --- (Optional) Cleanup step for Playwright browser ---
# When your app is shutting down, you should close the browser:
# asyncio.run(browser_to_close.stop())

