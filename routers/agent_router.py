
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from agents.llm.agent import agent
from agents.llm.state import State

# Initialize the API router for agent functionality
router = APIRouter(prefix="/agent", tags=["Agent Endpoints"])

# Pydantic models for request/response
class AgentRequest(BaseModel):
    message: str

@router.post("/run")
async def run_agent(request: AgentRequest):
    """
    Endpoint to run the LangGraph agent with the provided message and context.
    """
    try:
        # Create the initial state with the user's message
        initial_state = State(
            messages=[{"role": "user", "content": request.message}]
        )
        
        # Run the LangGraph agent
        result = agent.graph.invoke(initial_state)
        
        # Extract the agent's response from the last message
        agent_response = result["messages"][-1].content
        
        response = {
            "agent_response": agent_response,
            "user_message": request.message,
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")