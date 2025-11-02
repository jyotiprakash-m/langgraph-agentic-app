from fastapi import APIRouter, Depends, HTTPException, Query, Form
from pydantic import BaseModel
from agents.llm.agent import agent
from agents.llm.state import State
from agents.sidekick.agent import Sidekick  # Import the Sidekick agent
import aiosqlite
from langchain_core.messages import AIMessage, HumanMessage
from typing import Optional
from fastapi import UploadFile
from pathlib import Path

sidekick_agent = Sidekick()
# Initialize the API router for agent functionality
router = APIRouter(prefix="/agent", tags=["Agent Endpoints"])

# Pydantic models for request/response
class AgentRequest(BaseModel):
    message: str
    username: str
    chat_id: str
    file: Optional[UploadFile] = None

@router.post("/run")
async def run_agent(request: AgentRequest):
    """
    Endpoint to run the LangGraph agent with the provided message and context.
    """
    try:
        # Create the initial state with the user's message

        config = {"configurable": {"thread_id": f"{request.username}_{request.chat_id}"}}
        initial_state = State(
            messages=[HumanMessage(content=request.message)]  # Use HumanMessage object, not dict
        )
        
        # Run the LangGraph agent
        result = agent.graph.invoke(initial_state, config=config)  # type: ignore
        
        # Extract the agent's response from the last message
        agent_response = result["messages"][-1].content
        
        response = {
            "agent_response": agent_response,
            "user_message": request.message,
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
    
# Endpoint to run the Sidekick agent with the provided message and context
@router.post("/sidekick/run")
async def run_sidekick_agent(
    message: str = Form(...),
    username: str = Form(...),
    chat_id: str = Form(...),
    file: Optional[UploadFile] = None
):
    """
    Endpoint to run the Sidekick agent with the provided message and context.
    
    Supports file upload via Swagger UI for tasks like OCR.
    """
    try:
        # Ensure the agent is set up (tools, graph, etc.)
        if sidekick_agent.graph is None:
            await sidekick_agent.setup()
            
        # Handle file upload if present
        file_path = None
        if file and file.filename:
            # Save the uploaded file to the sandbox directory
            sandbox_dir = Path("sandbox")
            sandbox_dir.mkdir(exist_ok=True)
            file_path = sandbox_dir / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            file_path = str(file_path)  # Convert to string for the message
        elif file:
            raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

        # Prepare the message content, including file path if uploaded
        message_content = message
        if file_path:
            message_content += f" File uploaded: {file_path}"
            
        
        
        # Prepare the state for Sidekick
        state = {
            "messages": [HumanMessage(content=message_content)],
            "success_criteria": "The answer should be clear and accurate",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }

        # Run the Sidekick agent
        result = await sidekick_agent.graph.ainvoke(state, config={"configurable": {"thread_id": f"{username}_{chat_id}"}}) # type: ignore
        agent_response = result["messages"][-2].content  # Get the agent's response, not the evaluator feedback

        response = {
            "agent_response": agent_response,
            "user_message": message,
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sidekick Agent error: {str(e)}")


@router.get("/threads/{username}")
async def get_user_threads(username: str):
    """
    Get all thread IDs for a specific user.
    """
    try:
        conn = await aiosqlite.connect("memory.db")
        cursor = await conn.execute("SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE ?", (f"{username}_%",))
        rows = await cursor.fetchall()
        await conn.close()
        threads = [row[0] for row in rows]
        return {"threads": threads}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving threads: {str(e)}")

@router.get("/thread/{username}/{chat_id}/messages")
async def get_thread_messages(username: str, chat_id: str):
    """
    Get all messages for a specific thread.
    """
    try:
        thread_id = f"{username}_{chat_id}"
        # Ensure the agent is set up
        if sidekick_agent.graph is None:
            await sidekick_agent.setup()
        
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = await sidekick_agent.graph.aget_state(config) # type: ignore
        
        if state_snapshot and state_snapshot.values:
            messages = state_snapshot.values.get("messages", [])
            # Filter to show only human messages and AI responses with response_metadata
            filtered_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    filtered_messages.append(msg)
                elif isinstance(msg, AIMessage) and hasattr(msg, 'response_metadata') and msg.response_metadata:
                    filtered_messages.append(msg)
        else:
            filtered_messages = []
        
        return {"messages": filtered_messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving messages: {str(e)}")