from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from dotenv import load_dotenv
import os
import requests
from langchain_core.tools import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


load_dotenv(override=True)

pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"
serper = GoogleSerperAPIWrapper()

def safe_tool(func):
    """Wrapper to ensure tool functions always return a string, even on errors."""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return str(result) if result is not None else "Tool executed successfully."
        except Exception as e:
            return f"Tool error: {str(e)}"
    return wrapper

async def playwright_tools():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), browser, playwright

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


async def other_tools():
    push_tool = Tool(name="send_push_notification", func=safe_tool(push), description="Use this tool when you want to send a push notification")
    file_tools = get_file_tools()
    file_link_tool = Tool(name="get_file_link", func=safe_tool(get_file_link), description="Use this tool to get a public link for a file")
    save_pdf_tool = Tool(name="save_file_pdf", func=safe_tool(save_file_pdf), description="Use this tool to save content as a PDF file")

    tool_search = Tool(
        name="search",
        func=safe_tool(serper.run),
        description="Use this tool when you want to get the results of an online web search"
    )

    wikipedia = WikipediaAPIWrapper(wiki_client=None)
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)


    return file_tools + [push_tool, tool_search, wiki_tool, file_link_tool, save_pdf_tool]