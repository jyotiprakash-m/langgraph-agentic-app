# LangGraph Agentic App

A modern FastAPI application powered by LangGraph agents, built with uv for fast and reliable dependency management. Demonstrates advanced AI agent architecture with proper state management and modular design.

## 🚀 Features

- **FastAPI Backend**: High-performance async web framework
- **LangGraph Integration**: Advanced agent workflow management with state graphs
- **UV Package Management**: Lightning-fast dependency resolution and virtual environment management
- **Modular Architecture**: Clean separation of concerns with agents, nodes, and state
- **Type Safety**: Full Pydantic model validation and TypedDict state management
- **CORS Support**: Ready for frontend integration (Next.js compatible)
- **Health Checks**: Built-in monitoring and diagnostics endpoints

## 📋 Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

## 🛠️ Installation

### Quick Start with uv

```bash
# Clone the repository
git clone <your-repo-url>
cd langgraph-agentic-app

# Install all dependencies and create virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### Development Dependencies

```bash
# Add new dependencies
uv add fastapi uvicorn langgraph

# Add development dependencies
uv add --dev pytest black isort mypy

# Remove dependencies
uv remove package-name
```

## 🔧 Configuration

Create a `.env` file in the project root:

```env
# AI Model Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# LangChain Configuration
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=langgraph-agentic-app

# Application Configuration
ENVIRONMENT=development
LOG_LEVEL=info
```

## 🚀 Running the Application

### Development Mode (Recommended)

```bash
# Using uv to run the application
uv run python main.py

# Or activate environment and run directly
source .venv/bin/activate
python main.py

# Using uvicorn directly with hot reload
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
# Single worker
uv run uvicorn main:app --host 0.0.0.0 --port 8000

# Multiple workers for production
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at: `http://localhost:8000`

## 📖 API Documentation

Once the server is running, access the interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## 🛡️ API Endpoints

### Health Monitoring

```http
GET /
Response: {"message": "LangGraph Agentic App is running!", "status": "healthy"}

GET /health
Response: {"status": "healthy", "service": "langgraph-agentic-app", "version": "0.1.0"}
```

### Agent Interaction

```http
POST /agent
Content-Type: application/json

{
  "message": "Your message to the agent"
}
```

**Response:**

```json
{
  "agent_response": "Agent's contextual response",
  "user_message": "Your original message",
  "total_messages": 2
}
```

**Example:**

```bash
curl -X POST "http://localhost:8000/agent" \
     -H "Content-Type: application/json" \
     -d '{"message": "Tell me something interesting!"}'
```

## 🏗️ Project Structure

```
langgraph-agentic-app/
├── agents/
│   ├── __init__.py
│   ├── agent.py          # LangGraph agent with StateGraph implementation
│   ├── nodes.py          # Agent workflow nodes and business logic
│   └── state.py          # Centralized state definitions (TypedDict)
├── routers/
│   ├── __init__.py
│   └── agent_router.py   # Agent-related API routes
├── main.py               # FastAPI application entry point
├── pyproject.toml        # UV project configuration and dependencies
├── uv.lock              # Lockfile for reproducible builds
├── .env                 # Environment variables (create from .env.example)
├── .env.example         # Environment variables template
└── README.md
```

## 🔄 LangGraph Agent Architecture

### State Management Pattern

```python
# agents/state.py - Centralized state definition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # Automatic message aggregation
```

### Node-Based Workflow

```python
# agents/nodes.py - Individual processing nodes
def our_first_node(state: State) -> State:
    # Node processes current state and returns updated state
    # Follows functional programming principles
    return new_state
```

### Graph Compilation

```python
# agents/agent.py - StateGraph definition
graph_builder = StateGraph(State)
graph_builder.add_node("first_node", our_first_node)
graph_builder.add_edge(START, "first_node")
graph_builder.add_edge("first_node", END)
compiled_graph = graph_builder.compile()
```

### Current Workflow

```
START → first_node → END
```

## 🧪 Testing the Agent

### Interactive Testing

```python
import requests

# Test the agent endpoint
response = requests.post(
    "http://localhost:8000/agent",
    json={"message": "What can you tell me about unicorns?"}
)

print(response.json())
# Expected: Random fun response about unicorns being adjective
```

### Health Check Testing

```bash
# Quick health check
curl http://localhost:8000/health

# Root endpoint
curl http://localhost:8000/
```

## 🔧 Development with uv

### Adding Dependencies

```bash
# Production dependencies
uv add requests httpx pydantic-settings

# Development dependencies
uv add --dev pytest pytest-asyncio httpx black isort mypy

# Optional dependencies
uv add --optional langchain-openai  # For OpenAI integration
```

### Managing Virtual Environment

```bash
# Show current environment info
uv python list

# Create new environment with specific Python version
uv python install 3.11

# Remove virtual environment
rm -rf .venv
uv sync  # Recreates environment
```

### Project Commands

```bash
# Update all dependencies
uv sync --upgrade

# Install project in editable mode
uv pip install -e .

# Export requirements (if needed for other tools)
uv export --format requirements-txt > requirements.txt
```

## 🎯 Extending the Agent

### Adding New Nodes

1. Create node function in `agents/nodes.py`:

```python
def data_processing_node(state: State) -> State:
    # Process incoming data
    processed_data = process_user_input(state["messages"][-1].content)

    # Return updated state
    return State(
        messages=[{"role": "assistant", "content": processed_data}]
    )
```

2. Register in `agents/agent.py`:

```python
graph_builder.add_node("data_processor", data_processing_node)
graph_builder.add_edge("first_node", "data_processor")
graph_builder.add_edge("data_processor", END)
```

### Adding Conditional Logic

```python
def should_continue(state: State) -> str:
    # Route based on state conditions
    last_message = state["messages"][-1].content
    if "urgent" in last_message.lower():
        return "priority_handler"
    return "standard_handler"

# Add conditional edge
graph_builder.add_conditional_edges(
    "decision_node",
    should_continue,
    {"priority_handler": "priority_node", "standard_handler": "standard_node"}
)
```

## 🐛 Troubleshooting

### Common uv Issues

```bash
# Cache issues - clear uv cache
uv cache clean

# Lock file conflicts
rm uv.lock
uv sync

# Python version issues
uv python list  # Check available versions
uv python install 3.11  # Install specific version
```

### Application Issues

```bash
# Port conflicts
lsof -i :8000  # Find process using port 8000
kill -9 <PID>  # Kill the process

# Environment variables not loading
source .venv/bin/activate
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"

# Import errors - check Python path
uv run python -c "import sys; print(sys.path)"
```

### Debug Mode

```python
# Enable debug logging in main.py
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via uvicorn
uv run uvicorn main:app --log-level debug --reload
```

## 📊 Performance & Monitoring

### Built-in Monitoring

- Health check endpoints (`/` and `/health`)
- Request/response logging via FastAPI
- LangGraph state inspection via `print(result)`

### Recommended Monitoring Stack

```bash
# Add monitoring dependencies
uv add prometheus-client structlog

# For production
uv add gunicorn  # WSGI server for production
```

### Development Setup

```bash
# Fork and clone
git clone <your-fork-url>
cd langgraph-agentic-app

# Install with dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black .
uv run isort .

# Type checking
uv run mypy .
```
