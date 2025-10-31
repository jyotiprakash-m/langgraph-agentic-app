from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from agents.llm.agent import LangGraphAgent
from routers.agent_router import router as agent_router
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Startup: initialize agent
    app.state.agent = await LangGraphAgent.create()
    yield
    # Shutdown: (optional) cleanup here

app = FastAPI(
    title="LangGraph Agentic App",
    description="A FastAPI application powered by LangGraph agents",
    version="0.1.0",
    lifespan=lifespan
)

# =========================================
# 🌐 CORS Middleware (for Next.js frontend)
# =========================================
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "LangGraph Agentic App is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "langgraph-agentic-app",
        "version": "0.1.0"
    }

# Include routes
app.include_router(agent_router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
