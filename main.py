from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from routers.agent_router import router as agent_router
from routers.user_router import router as user_router
from fastapi.staticfiles import StaticFiles

from utils.database import init_db


# ✅ Modern lifespan event system
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up db...")
    init_db()   # Initialize the database
    yield
    print("🛑 Shutting down db...")

app = FastAPI(
    title="LangGraph Agentic App",
    description="A FastAPI application powered by LangGraph agents",
    version="0.1.0",
    lifespan=lifespan
)


# =========================================
# 🌐 CORS Middleware (for Next.js frontend)
# =========================================
# You can update `origins` later to your frontend’s actual domain
origins = [
    "http://localhost:3000",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # or ["*"] for all (less secure)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving zipped projects
app.mount("/public", StaticFiles(directory="sandbox"), name="public")


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
app.include_router(user_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
