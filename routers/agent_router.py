from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from agents.llm.agent import agent
from agents.llm.state import State
from agents.sidekick.agent import Sidekick  # Import the Sidekick agent
import aiosqlite

sidekick_agent = Sidekick()
# Initialize the API router for agent functionality
router = APIRouter(prefix="/agent", tags=["Agent Endpoints"])

# Pydantic models for request/response
class AgentRequest(BaseModel):
    message: str
    username: str
    chat_id: str

@router.post("/run")
async def run_agent(request: AgentRequest):
    """
    Endpoint to run the LangGraph agent with the provided message and context.
    """
    try:
        # Create the initial state with the user's message

        config = {"configurable": {"thread_id": f"{request.username}_{request.chat_id}"}}
        initial_state = State(
            messages=[{"role": "user", "content": request.message}]
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
async def run_sidekick_agent(request: AgentRequest):
    """
    Endpoint to run the Sidekick agent with the provided message and context.
    """
    try:
        # Ensure the agent is set up (tools, graph, etc.)
        if sidekick_agent.graph is None:
            await sidekick_agent.setup()

        # Prepare the state for Sidekick
        state = {
            "messages": [{"role": "user", "content": request.message}],
            "success_criteria": "The answer should be clear and accurate",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }

        # Run the Sidekick agent
        result = await sidekick_agent.graph.ainvoke(state, config={"configurable": {"thread_id": f"{request.username}_{request.chat_id}"}}) # type: ignore
        agent_response = result["messages"][-2].content  # Get the agent's response, not the evaluator feedback

        response = {
            "agent_response": agent_response,
            "user_message": request.message,
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