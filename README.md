# LMStack

LLM Deployment Management Platform - Deploy and manage Large Language Models on distributed GPU workers.

## Features

- Web UI for managing workers, models, and deployments
- Docker-based worker agents for GPU nodes
- vLLM inference backend support
- Real-time deployment status monitoring
- Container logs viewing

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Web Frontend  │────▶│   API Server    │
│   (React)       │     │   (FastAPI)     │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
            ┌──────────────┐          ┌──────────────┐
            │ Worker Agent │          │ Worker Agent │
            │  (GPU Node)  │          │  (GPU Node)  │
            └──────────────┘          └──────────────┘
                    │                         │
                    ▼                         ▼
            ┌──────────────┐          ┌──────────────┐
            │ vLLM Docker  │          │ vLLM Docker  │
            │  Container   │          │  Container   │
            └──────────────┘          └──────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker
- NVIDIA GPU with CUDA support (for workers)

### 1. Start the Server

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Visit http://localhost:3000 to access the UI.

### 3. Start a Worker Agent

On each GPU node:

```bash
cd worker

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start worker agent
python agent.py --name gpu-worker-01 --server-url http://SERVER_IP:8000 --port 8080
```

### 4. Deploy a Model

1. Go to **Workers** page and verify your worker is online
2. Go to **Models** page and add a model (e.g., Qwen3-0.6B)
3. Go to **Deployments** page and create a new deployment
4. Select the model and worker, then click Deploy

## Development

### Backend

```bash
cd backend
pip install -e ".[dev]"

# Run with auto-reload
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Using Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Configuration

### Server Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| LMSTACK_DATABASE_URL | sqlite+aiosqlite:///./lmstack.db | Database connection URL |
| LMSTACK_DEBUG | false | Enable debug mode |
| LMSTACK_HOST | 0.0.0.0 | Server host |
| LMSTACK_PORT | 8000 | Server port |
| LMSTACK_VLLM_DEFAULT_IMAGE | vllm/vllm-openai:latest | Default vLLM Docker image |

### Worker Agent Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| --name | Yes | Worker name (must be unique) |
| --server-url | Yes | LMStack server URL |
| --host | No | Agent listen host (default: 0.0.0.0) |
| --port | No | Agent listen port (default: 8080) |

## Project Structure

```
lmstack/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── models/         # SQLAlchemy models
│   │   ├── schemas/        # Pydantic schemas
│   │   ├── services/       # Business logic
│   │   └── main.py         # Application entry
│   └── requirements.txt
├── frontend/               # React frontend
│   ├── src/
│   │   ├── pages/         # Page components
│   │   ├── services/      # API client
│   │   └── types/         # TypeScript types
│   └── package.json
├── worker/                 # Worker agent
│   ├── agent.py           # Agent entry point
│   ├── docker_runner.py   # Docker management
│   └── requirements.txt
└── docker-compose.yml
```

## License

Apache-2.0
