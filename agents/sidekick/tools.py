import json
from typing import List, Optional
# from playwright.async_api import async_playwright
# from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
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
from pypdf import PdfReader
import logging

from twilio.rest import Client


from langchain_core.tools import StructuredTool
from langchain_community.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper

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

# async def playwright_tools():
#     playwright = await async_playwright().start()
#     browser = await playwright.chromium.launch(headless=False)
#     toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
#     return toolkit.get_tools(), browser, playwright

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


def extract_text_from_file(
    file_path: str,
    task_type: str = "default",
    max_tokens: int = 16000,
    temperature: float = 0.1,
    top_p: float = 0.6,
    repetition_penalty: float = 1.2,
    pages: Optional[List[int]] = None
) -> str:
    """
    Extracts text from a PDF or image file using the OpenTyphoon OCR API.
    """
    url = os.getenv('OPEN_ROUTER_OCR_BASE_URL')
    
    if not url:
        return "Error: OPEN_ROUTER_OCR_BASE_URL is not set in environment variables."

    # Determine pages to extract if not specified
    if pages is None:
        if file_path.lower().endswith('.pdf'):
            try:
                reader = PdfReader(file_path)
                num_pages = len(reader.pages)
                pages = list(range(1, num_pages + 1))  # 1-based indexing
            except Exception as e:
                return f"Error reading PDF: {str(e)}"
        else:
            pages = [1]  # Assume single page for images

    try:
        with open(file_path, 'rb') as file:
            files = {'file': file}
            data = {
                'task_type': task_type,
                'max_tokens': str(max_tokens),
                'temperature': str(temperature),
                'top_p': str(top_p),
                'repetition_penalty': str(repetition_penalty),
                'pages': json.dumps(pages)  # Always send pages
            }
            
            api_key = os.getenv('OPENTYPHOON_API_KEY')
            if not api_key:
                return "Error: OPENTYPHOON_API_KEY is not set in environment variables."

            headers = {'Authorization': f'Bearer {api_key}'}
            response = requests.post(url, files=files, data=data, headers=headers)

            if response.status_code == 200:
                try:
                    result = response.json()
                    extracted_texts = []
                    for page_result in result.get('results', []):
                        if page_result.get('success') and page_result.get('message'):
                            content = page_result['message']['choices'][0]['message']['content']
                            try:
                                parsed_content = json.loads(content)
                                text = parsed_content.get('natural_text', content)
                            except json.JSONDecodeError:
                                text = content
                            extracted_texts.append(text)
                        elif not page_result.get('success'):
                            logging.error(f"Error processing {page_result.get('filename', 'unknown')}: {page_result.get('error', 'Unknown error')}")
                    logging.info("OCR extraction completed. %d pages processed.", len(extracted_texts))
                    return '\n'.join(extracted_texts)
                except json.JSONDecodeError:
                    # If not JSON, try to decode as text, ignoring errors
                    try:
                        text_response = response.content.decode('utf-8', errors='ignore')[:500]
                        return f"Error parsing API response: {text_response}"
                    except Exception:
                        return "Error: API returned non-text data."
            else:
                try:
                    error_text = response.content.decode('utf-8', errors='ignore')[:500]
                    return f"Error: {response.status_code} - {error_text}"
                except Exception:
                    return f"Error: {response.status_code} - Non-text error response."
    except Exception as e:
        return f"Error during OCR extraction: {str(e)}"
        
def send_telegram_message( text: str) -> str:
    """
    Send a message to a Telegram user or group.
    Args:
        text (str): The message to send.
    Returns:
        str: Result message.
    """

    
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    if not TELEGRAM_BOT_TOKEN:
        return "Error: TELEGRAM_BOT_TOKEN not set."
    payload = {"chat_id": 1206152577, "text": text}
    try:
        resp = requests.post(TELEGRAM_API_URL, data=payload, timeout=10)
        if resp.status_code == 200:
            return "Message sent to Telegram."
        else:
            return f"Failed to send: {resp.text}"
    except Exception as e:
        return f"Telegram send error: {str(e)}"
    
def send_whatapp_message(to_number: str, message: str, message_type: str = "sms") -> str:
    """
    Send SMS or WhatsApp message using Twilio API.
    to_number: Recipient's number in format +91xxxxxxxxxx
    message: The message to send
    message_type: "sms" for SMS or "whatsapp" for WhatsApp (default: "sms")
    """
    import os
    

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_PHONE_NUMBER")
    sms_number = os.getenv("TWILIO_PHONE_SMS_NUMBER")

    if not all([account_sid, auth_token, from_number, sms_number]):
        return "❌ Twilio credentials not found in environment variables."
    
    client = Client(account_sid, auth_token)
    
    try:
        if message_type.lower() == "whatsapp":
            message_obj = client.messages.create(
                body=message,
                to="whatsapp:" + to_number,
                from_="whatsapp:" + str(from_number),
            )
        else:  # SMS
            message_obj = client.messages.create(
                body=message,
                to=to_number,
                from_=str(sms_number),
            )
        return f"✅ {message_type.upper()} sent! SID: {message_obj.sid}"
    except Exception as e:
        return f"❌ Failed to send {message_type}: {str(e)}"
    
def other_tools():
    push_tool = Tool(name="send_push_notification", func=safe_tool(push), description="Use this tool when you want to send a push notification")
    file_tools = get_file_tools()
    file_link_tool = Tool(name="get_file_link", func=safe_tool(get_file_link), description="Use this tool to get a public link for a file")
    save_pdf_tool = Tool(name="save_file_pdf", func=safe_tool(save_file_pdf), description="Use this tool to save content as a PDF file")

    tool_search = Tool(
        name="search",
        func=safe_tool(serper.run),
        description="Use this tool when you want to get the results of an online web search"
    )
    ocr_tool = Tool(
        name="extract_text_from_file",
        func=safe_tool(extract_text_from_file),  # Wrap with safe_tool for error handling
        description="Extract text from a PDF or image file using OCR. Provide the file path in the sandbox directory (e.g., 'uploaded.pdf')."
    )
    
    telegram_tool = StructuredTool.from_function(
    name="send_telegram_message",
    func=send_telegram_message,
    description="Send a message to a Telegram bot."
    )
    whatsapp_tool = StructuredTool.from_function(
    func=send_whatapp_message,
    name="send_whatapp_message",
    description="Send SMS or WhatsApp message using Twilio. Requires to_number (recipient), message (text), and optional message_type ('sms' or 'whatsapp')."
)

    wikipedia = WikipediaAPIWrapper(wiki_client=None)
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)
    
    wolfram_api_id = os.getenv("WOLFRAMA_APP_ID")
    wolfram_wrapper = WolframAlphaAPIWrapper(wolfram_alpha_appid=wolfram_api_id)
    wolfram_tool = WolframAlphaQueryRun(api_wrapper=wolfram_wrapper)


    return file_tools + [push_tool, tool_search, wiki_tool, file_link_tool, save_pdf_tool, ocr_tool, telegram_tool, whatsapp_tool, wolfram_tool]