# Architecture

This document describes the architecture of LMStack.

## System Overview

```mermaid
graph TB
    subgraph "Clients"
        UI[Web Dashboard]
        SDK[OpenAI SDK]
        CLI[CLI Tools]
    end

    subgraph "Backend Server"
        API[FastAPI Application]
        Auth[Authentication]
        DB[(PostgreSQL)]
    end

    subgraph "Worker Nodes"
        W1[Worker Agent 1]
        W2[Worker Agent 2]
        WN[Worker Agent N]
    end

    subgraph "Inference Containers"
        V[vLLM]
        O[Ollama]
        S[SGLang]
    end

    UI --> API
    SDK --> API
    CLI --> API
    API --> Auth
    API --> DB
    API --> W1
    API --> W2
    API --> WN
    W1 --> V
    W2 --> O
    WN --> S
```

## Components

### Backend Server

The backend is a FastAPI application that provides:

- **REST API**: Deployment management, worker management, user management
- **OpenAI-Compatible API**: `/v1/chat/completions`, `/v1/models`
- **WebSocket**: Real-time log streaming
- **Authentication**: JWT tokens and API keys

#### Directory Structure

```
backend/app/
├── api/                 # API endpoints
│   ├── apps/            # Application routes (modular)
│   │   ├── deployments.py
│   │   ├── workers.py
│   │   └── users.py
│   └── openai.py        # OpenAI-compatible endpoints
├── core/                # Core configuration
│   ├── config.py        # Settings
│   ├── security.py      # Auth utilities
│   └── database.py      # DB connection
├── models/              # SQLAlchemy models
├── schemas/             # Pydantic schemas
└── services/            # Business logic
```

#### Request Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI
    participant Auth as Auth Middleware
    participant Service as Service Layer
    participant DB as Database
    participant Worker as Worker Agent

    C->>API: POST /api/deployments
    API->>Auth: Validate token
    Auth-->>API: User context
    API->>Service: create_deployment()
    Service->>DB: Insert deployment
    Service->>Worker: Send deploy command
    Worker-->>Service: Acknowledge
    Service-->>API: Deployment created
    API-->>C: 201 Created
```

### Frontend

React-based single-page application:

- **Ant Design**: UI component library
- **React Query**: Data fetching and caching
- **React Router**: Client-side routing

#### Directory Structure

```
frontend/src/
├── components/          # Reusable components
│   ├── chat/            # Chat interface
│   ├── logos/           # Brand logos
│   └── common/          # Shared components
├── pages/               # Page components
├── hooks/               # Custom hooks
├── utils/               # Utility functions
├── api/                 # API client
└── types/               # TypeScript types
```

### Worker Agent

Lightweight Python agent running on GPU nodes:

- **Docker Management**: Container lifecycle management
- **GPU Detection**: NVIDIA GPU enumeration
- **Heartbeat**: Health reporting to backend
- **Model Deployment**: Container orchestration

#### Directory Structure

```
worker/
├── docker_ops/          # Docker operations
│   ├── gpu.py           # GPU detection
│   ├── system.py        # System info
│   ├── runner.py        # Model deployment
│   ├── images.py        # Image management
│   └── containers.py    # Container management
├── routes/              # API endpoints
│   ├── deployment.py    # Deploy/stop operations
│   ├── images.py        # Image operations
│   ├── containers.py    # Container operations
│   └── storage.py       # Storage operations
├── models.py            # Request/response models
└── agent.py             # Main entry point
```

## Data Flow

### Deployment Creation

```mermaid
flowchart TD
    A[User creates deployment] --> B[Backend validates request]
    B --> C[Store in database]
    C --> D[Queue deployment task]
    D --> E[Worker receives command]
    E --> F{Image available?}
    F -->|No| G[Pull image]
    F -->|Yes| H[Start container]
    G --> H
    H --> I[Report status]
    I --> J[Update database]
    J --> K[Notify frontend]
```

### Inference Request

```mermaid
flowchart TD
    A[Client sends request] --> B[Backend authenticates]
    B --> C[Find deployment]
    C --> D[Route to worker]
    D --> E[Forward to container]
    E --> F[Model inference]
    F --> G[Return response]
    G --> H[Stream to client]
```

## Database Schema

### Core Tables

```sql
-- Deployments
CREATE TABLE deployments (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    backend VARCHAR(50) NOT NULL,
    model VARCHAR(500) NOT NULL,
    status VARCHAR(50) NOT NULL,
    worker_id UUID REFERENCES workers(id),
    config JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Workers
CREATE TABLE workers (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    gpu_info JSONB,
    system_info JSONB,
    last_heartbeat TIMESTAMP
);

-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE
);

-- API Keys
CREATE TABLE api_keys (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMP,
    last_used TIMESTAMP
);
```

## Security

### Authentication Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant B as Backend
    participant DB as Database

    U->>F: Login with credentials
    F->>B: POST /auth/login
    B->>DB: Verify credentials
    DB-->>B: User data
    B->>B: Generate JWT
    B-->>F: JWT token
    F->>F: Store token
    F->>B: API request + token
    B->>B: Validate JWT
    B-->>F: Response
```

### API Key Authentication

```mermaid
sequenceDiagram
    participant C as Client
    participant B as Backend
    participant DB as Database

    C->>B: Request + API Key
    B->>DB: Lookup key hash
    DB-->>B: Key metadata
    B->>B: Validate key
    B->>DB: Update last_used
    B-->>C: Response
```

## Scalability

### Horizontal Scaling

```
                    ┌─────────────┐
                    │Load Balancer│
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      ┌─────────┐    ┌─────────┐    ┌─────────┐
      │Backend 1│    │Backend 2│    │Backend 3│
      └────┬────┘    └────┬────┘    └────┬────┘
           │              │              │
           └──────────────┼──────────────┘
                          │
                    ┌─────┴─────┐
                    │ PostgreSQL │
                    │  (Primary) │
                    └─────┬─────┘
                          │
                    ┌─────┴─────┐
                    │ PostgreSQL │
                    │ (Replica)  │
                    └───────────┘
```

### Worker Distribution

Workers can be deployed across multiple GPU nodes:

- Each worker registers independently
- Backend distributes deployments across workers
- Workers handle local container orchestration
- Fault tolerance through redundancy

## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend API | FastAPI (Python) |
| Frontend | React + TypeScript |
| UI Components | Ant Design |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Worker Agent | Python + Docker SDK |
| Inference | vLLM, Ollama, SGLang |
| Container Runtime | Docker |
| GPU Support | NVIDIA Container Toolkit |
