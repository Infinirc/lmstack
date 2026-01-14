# Docker Compose Deployment

Deploy LMStack using Docker Compose for a simple, containerized setup.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose v2+
- NVIDIA Container Toolkit (for GPU workers)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/lmstack/lmstack.git
cd lmstack
```

### 2. Configure Environment

Create `.env` file in the project root:

```bash
# Database
POSTGRES_USER=lmstack
POSTGRES_PASSWORD=your-secure-password
POSTGRES_DB=lmstack

# Backend
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://lmstack:your-secure-password@db:5432/lmstack

# Frontend
VITE_API_URL=http://localhost:52000
```

### 3. Start Services

```bash
docker compose up -d
```

## Docker Compose Configuration

### Basic Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 5s
      timeout: 5s
      retries: 5

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: ${DATABASE_URL}
      SECRET_KEY: ${SECRET_KEY}
    ports:
      - "8000:52000"
    depends_on:
      db:
        condition: service_healthy

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "80:80"
    depends_on:
      - backend

  worker:
    build:
      context: ./worker
      dockerfile: Dockerfile
    environment:
      BACKEND_URL: http://backend:52000
      WORKER_TOKEN: ${WORKER_TOKEN}
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    depends_on:
      - backend

volumes:
  postgres_data:
```

### With GPU Support

```yaml
# docker-compose.gpu.yml
version: '3.8'

services:
  worker:
    extends:
      file: docker-compose.yml
      service: worker
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Run with GPU support:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## Service Management

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
```

### Restart Services

```bash
# Restart all
docker compose restart

# Restart specific service
docker compose restart backend
```

### Stop Services

```bash
docker compose down

# Stop and remove volumes
docker compose down -v
```

### Update Services

```bash
# Pull latest images
docker compose pull

# Rebuild and restart
docker compose up -d --build
```

## Scaling

### Scale Backend

```bash
docker compose up -d --scale backend=3
```

### Multiple Workers

For multiple GPU workers, use a separate compose file for each worker node:

```yaml
# docker-compose.worker.yml
version: '3.8'

services:
  worker:
    build:
      context: ./worker
      dockerfile: Dockerfile
    environment:
      BACKEND_URL: http://backend-server:52000
      WORKER_TOKEN: ${WORKER_TOKEN}
      WORKER_NAME: ${WORKER_NAME:-worker-1}
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Deploy on worker nodes:

```bash
WORKER_NAME=gpu-node-1 docker compose -f docker-compose.worker.yml up -d
```

## Networking

### Custom Network

```yaml
networks:
  lmstack:
    driver: bridge

services:
  backend:
    networks:
      - lmstack
```

### External Access

For external worker nodes, expose the backend API:

```yaml
services:
  backend:
    ports:
      - "0.0.0.0:52000:52000"
```

## Persistence

### Data Volumes

```yaml
volumes:
  postgres_data:
    driver: local
  model_cache:
    driver: local

services:
  worker:
    volumes:
      - model_cache:/root/.cache/huggingface
```

### Backup Database

```bash
docker compose exec db pg_dump -U lmstack lmstack > backup.sql
```

### Restore Database

```bash
cat backup.sql | docker compose exec -T db psql -U lmstack lmstack
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs backend

# Check container status
docker compose ps
```

### GPU Not Detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Check worker logs
docker compose logs worker
```

### Database Connection Issues

```bash
# Check database health
docker compose exec db pg_isready -U lmstack

# View database logs
docker compose logs db
```

## Next Steps

- [Kubernetes Deployment](kubernetes.md) - For larger scale
- [Production Guide](production.md) - Security and optimization
