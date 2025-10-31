from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
from agents.llm.state import State

router = APIRouter(prefix="/agent", tags=["Agent Endpoints"])

class AgentRequest(BaseModel):
    message: str
    thread_id: str = Query(default="1")

@router.post("/run")
async def run_agent(request: Request, body: AgentRequest):
    """
    Endpoint to run the LangGraph agent with the provided message and context.
    """
    try:
        config = {"configurable": {"thread_id": body.thread_id}}
        initial_state = State(
            messages=[{"role": "user", "content": body.message}]
        )
        agent = request.app.state.agent
        result = await agent.graph.ainvoke(initial_state, config=config)  # type: ignore

        # Robust extraction of agent response
        messages = result.get("messages", [])
        if not messages or not hasattr(messages[-1], "content"):
            agent_response = "No response generated."
        else:
            agent_response = messages[-1].content

        response = {
            "agent_response": agent_response,
            "user_message": body.message,
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")