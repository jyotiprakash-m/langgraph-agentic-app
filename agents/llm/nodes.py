import random

import requests
from agents.llm.state import State
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
import os


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

#==============================
# Tool definitions
#=============================

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
        return "Push notification sent successfully."
    except requests.RequestException as e:
        return f"Failed to send push notification: {str(e)}"
    
tool_push = Tool(
        name="send_push_notification",
        func=push,
        description="useful for when you want to send a push notification"
    )

# =============================
# Bind tools to LLM
# =============================

tools = [tool_search, tool_push]
llm_with_tools = llm.bind_tools(tools)


# ==============================
# Node definitions
# ==============================

def chatbot_node(state: State) -> State:
    """
    Chatbot node that processes user messages and generates AI responses.
    
    Args:
        state: Current state containing conversation messages
        
    Returns:
        Updated state with AI response message
    """
    try:
        # Invoke the LLM with the current messages
        response = llm_with_tools.invoke(state["messages"])
        
        # Create new state with the AI response
        new_state = State(messages=state["messages"] + [response])
        
        return new_state
        
    except Exception as e:
        error_message = {
            "role": "assistant",
            "content": f"I encountered an error processing your request: {str(e)}"
        }
        return State(messages=[error_message])