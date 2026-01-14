# Configuration

This guide covers all configuration options for LMStack components.

## Backend Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | - | Yes |
| `SECRET_KEY` | JWT signing key | - | Yes |
| `REDIS_URL` | Redis connection string | - | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `CORS_ORIGINS` | Allowed CORS origins | `*` | No |
| `API_PREFIX` | API route prefix | `/api` | No |

### Database Configuration

```bash
# PostgreSQL connection
DATABASE_URL=postgresql://user:password@host:5432/dbname

# With SSL
DATABASE_URL=postgresql://user:password@host:5432/dbname?sslmode=require
```

### Security Configuration

```bash
# Generate a secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set in environment
SECRET_KEY=your-generated-secret-key

# JWT token expiration (in minutes)
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## Frontend Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | `http://localhost:8000` |
| `VITE_WS_URL` | WebSocket URL | Auto-derived from API URL |

### Build Configuration

Edit `vite.config.ts` for custom build settings:

```typescript
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
```

## Worker Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `BACKEND_URL` | Backend server URL | - | Yes |
| `WORKER_TOKEN` | Authentication token | - | Yes |
| `WORKER_NAME` | Worker identifier | hostname | No |
| `HEARTBEAT_INTERVAL` | Heartbeat frequency (seconds) | `30` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |

### Docker Configuration

Configure Docker settings in the worker:

```bash
# Docker socket path (default)
DOCKER_SOCKET=/var/run/docker.sock

# Custom Docker host
DOCKER_HOST=tcp://localhost:2375
```

### GPU Configuration

```bash
# Specify visible GPUs
CUDA_VISIBLE_DEVICES=0,1

# GPU memory fraction limit
GPU_MEMORY_FRACTION=0.9
```

## Backend-Specific Configuration

### vLLM Settings

```json
{
  "backend": "vllm",
  "config": {
    "gpu_memory_utilization": 0.9,
    "max_model_len": 4096,
    "tensor_parallel_size": 1,
    "dtype": "auto",
    "quantization": null,
    "enforce_eager": false
  }
}
```

| Option | Description | Default |
|--------|-------------|---------|
| `gpu_memory_utilization` | Fraction of GPU memory to use | 0.9 |
| `max_model_len` | Maximum context length | Model default |
| `tensor_parallel_size` | Number of GPUs for tensor parallelism | 1 |
| `dtype` | Data type (auto, float16, bfloat16) | auto |
| `quantization` | Quantization method (awq, gptq) | None |

### Ollama Settings

```json
{
  "backend": "ollama",
  "config": {
    "num_gpu": -1,
    "num_ctx": 4096,
    "num_batch": 512
  }
}
```

| Option | Description | Default |
|--------|-------------|---------|
| `num_gpu` | Number of GPU layers (-1 for all) | -1 |
| `num_ctx` | Context window size | 4096 |
| `num_batch` | Batch size for processing | 512 |

### SGLang Settings

```json
{
  "backend": "sglang",
  "config": {
    "tp_size": 1,
    "mem_fraction_static": 0.9,
    "max_total_tokens": 4096
  }
}
```

## Logging Configuration

### Backend Logging

```python
# In app/core/config.py or via environment
LOG_LEVEL=DEBUG
LOG_FORMAT=json  # or 'text'
```

### Worker Logging

```bash
# Set log level
LOG_LEVEL=DEBUG

# Log to file
LOG_FILE=/var/log/lmstack/worker.log
```

## Production Configuration

### Recommended Settings

```bash
# Backend
LOG_LEVEL=WARNING
WORKERS=4
DATABASE_POOL_SIZE=20

# Frontend (build-time)
VITE_API_URL=https://api.yourdomain.com

# Worker
HEARTBEAT_INTERVAL=60
```

### Security Hardening

```bash
# Restrict CORS
CORS_ORIGINS=https://yourdomain.com

# Use secure cookies
SECURE_COOKIES=true

# Enable HTTPS redirect
FORCE_HTTPS=true
```

## Next Steps

- [Docker Compose Deployment](../deployment/docker-compose.md)
- [Kubernetes Deployment](../deployment/kubernetes.md)
- [Production Guide](../deployment/production.md)
