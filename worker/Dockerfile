# LMStack Worker Agent Dockerfile
# For GPU nodes that run model inference containers

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker application
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV BACKEND_URL=http://backend:8000
ENV WORKER_NAME=worker
ENV AGENT_PORT=52001

# Health check (uses AGENT_PORT env var)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${AGENT_PORT}/health || exit 1

EXPOSE ${AGENT_PORT}

CMD ["python", "agent.py"]
