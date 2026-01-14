# Deployment

This section covers various deployment options for LMStack in production environments.

## Deployment Options

### Docker Compose

The simplest way to deploy LMStack with all components in containers.

- **Best for**: Small to medium deployments, single-node setups
- **Complexity**: Low
- **Guide**: [Docker Compose Deployment](docker-compose.md)

### Kubernetes

Scalable deployment for large-scale production environments.

- **Best for**: Large deployments, high availability requirements
- **Complexity**: High
- **Guide**: [Kubernetes Deployment](kubernetes.md)

### Manual Deployment

Deploy each component manually for maximum control.

- **Best for**: Custom infrastructure, specific requirements
- **Complexity**: Medium
- **Guide**: [Production Guide](production.md)

## Architecture Considerations

### Single-Node Deployment

```
┌─────────────────────────────────────────┐
│             Single Server               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ Backend │  │Frontend │  │ Worker  │ │
│  └─────────┘  └─────────┘  └─────────┘ │
│        │           │            │       │
│        └───────────┼────────────┘       │
│                    │                    │
│              ┌─────────┐                │
│              │PostgreSQL│               │
│              └─────────┘                │
└─────────────────────────────────────────┘
```

### Multi-Node Deployment

```
┌──────────────┐     ┌──────────────┐
│ Load Balancer│     │   Database   │
│   (nginx)    │     │ (PostgreSQL) │
└──────┬───────┘     └──────────────┘
       │                    │
┌──────┴───────┐           │
│              │           │
▼              ▼           │
┌────────┐  ┌────────┐     │
│Backend │  │Backend │◄────┘
│   #1   │  │   #2   │
└────────┘  └────────┘
     │           │
     └─────┬─────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────┐   ┌────────┐
│Worker 1│   │Worker 2│
│ (GPU)  │   │ (GPU)  │
└────────┘   └────────┘
```

## Resource Planning

### Backend Server

| Scale | CPU | RAM | Storage |
|-------|-----|-----|---------|
| Small (< 10 workers) | 2 cores | 4 GB | 20 GB |
| Medium (10-50 workers) | 4 cores | 8 GB | 50 GB |
| Large (50+ workers) | 8+ cores | 16+ GB | 100+ GB |

### Worker Nodes

| Model Size | GPU VRAM | System RAM | Storage |
|------------|----------|------------|---------|
| 7B | 16 GB | 32 GB | 100 GB |
| 13B | 24 GB | 64 GB | 200 GB |
| 70B | 80 GB (multi-GPU) | 128 GB | 500 GB |

## Next Steps

Choose your deployment method:

1. [Docker Compose](docker-compose.md) - Recommended for getting started
2. [Kubernetes](kubernetes.md) - For production at scale
3. [Production Guide](production.md) - Best practices and optimization
