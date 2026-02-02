# LMStack Backend Dockerfile
# Multi-stage build for optimized production image

FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies to /usr/local (system-wide)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production image
FROM python:3.11-slim

WORKDIR /app

# Install docker CLI for local worker spawn feature
# Using Docker 27.x for API version 1.47 compatibility
# Auto-detect architecture (x86_64 or aarch64)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && ARCH=$(uname -m) \
    && if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then DOCKER_ARCH="aarch64"; else DOCKER_ARCH="x86_64"; fi \
    && curl -fsSL "https://download.docker.com/linux/static/stable/${DOCKER_ARCH}/docker-27.4.1.tgz" | tar xz --strip-components=1 -C /usr/local/bin docker/docker \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/
COPY migrations/ ./migrations/

# Create data directory
RUN mkdir -p /app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://127.0.0.1:52000/health'); exit(0 if r.status_code==200 else 1)"

EXPOSE 52000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "52000", "--workers", "4"]
