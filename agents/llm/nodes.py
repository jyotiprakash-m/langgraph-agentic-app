import random

import requests
from agents.llm.state import State
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
import os
from langchain_community.agent_toolkits import FileManagementToolkit
from fpdf import FPDF
from langchain_core.messages import AIMessage
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# ===============================
# Tool definitions
# ===============================

def safe_tool(func):
    """Wrapper to ensure tool functions always return a string, even on errors."""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return str(result) if result is not None else "Tool executed successfully."
        except Exception as e:
            return f"Tool error: {str(e)}"
    return wrapper

# tool: web search
serper = GoogleSerperAPIWrapper()
tool_search = Tool(
    name="search",
    func=safe_tool(serper.run),
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
    func=safe_tool(push),
    description="useful for when you want to send a push notification"
)

def get_file_tools():
    toolkit = FileManagementToolkit(root_dir="sandbox")
    tools = toolkit.get_tools()
    # Wrap each tool function to be safe
    for tool in tools:
        tool = safe_tool(tool)
    return tools

# Custom tool : Get the file link
def get_file_link(file_name: str) -> str:
    """Generate a public link for a file in the sandbox directory"""
    # Get root address from environment or default
    base_url = os.getenv("BASE_URL", "http://localhost:8000/public/")
    file_path = os.path.join("sandbox", file_name)
    if os.path.exists(file_path):
        return base_url + file_name
    else:
        return "File not found."
    
file_link_tool = Tool(name="get_file_link", func=safe_tool(get_file_link), description="Use this tool to get a public link for a file")

# Custom tool : save file in pdf
def save_file_pdf(file_name: str, content: str) -> str:
    """
    Save the provided content as a PDF file in the sandbox directory using ReportLab.

    Args:
        file_name (str): The name of the PDF file to be created (e.g., 'output.pdf').
        content (str): The text content to be written into the PDF file.

    Returns:
        str: A message indicating the result, including the file path if successful.
    """
    if not file_name or not isinstance(file_name, str):
        return "Error: file_name (str) is required to save as PDF. Please provide a valid file name (e.g., 'output.pdf')."
    if not content or not isinstance(content, str):
        return "Error: content (str) is required to save as PDF. Please provide the content to save."

    try:
        file_path = os.path.join("sandbox", file_name)
        c = canvas.Canvas(file_path, pagesize=letter)
        width, height = letter
        
        # Split content into lines and draw each line
        lines = content.split('\n')
        y_position = height - 100
        line_height = 15
        max_lines = 50  # Limit to prevent overflow
        max_line_length = 80  # Limit line length
        
        for line in lines[:max_lines]:
            if y_position < 50:  # Prevent drawing off the page
                break
            truncated_line = line[:max_line_length]
            c.drawString(100, y_position, truncated_line)
            y_position -= line_height
        
        c.save()
        return f"File saved as {file_path}"
    except Exception as e:
        return f"Error saving PDF: {str(e)}"
    
save_pdf_tool = Tool(name="save_file_pdf", func=safe_tool(save_file_pdf), description="Use this tool to save content as a PDF file")

# =============================
# Bind tools to LLM
# =============================

tools = get_file_tools() + [tool_search, tool_push, file_link_tool, save_pdf_tool]
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
        # Create an error message as an AIMessage object
        error_message = AIMessage(content=f"I encountered an error processing your request: {str(e)}")
        # Append the error message to the existing conversation, so the thread can continue
        return State(messages=state["messages"] + [error_message])