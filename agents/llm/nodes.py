import random
from agents.llm.state import State
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

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
        response = llm.invoke(state["messages"])
        
        # Create new state with the AI response
        new_state = State(messages=[response])
        
        return new_state
        
    except Exception as e:
        error_message = {
            "role": "assistant",
            "content": f"I encountered an error processing your request: {str(e)}"
        }
        return State(messages=[error_message])