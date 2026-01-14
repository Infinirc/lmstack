# Quick Start

This guide will help you deploy your first LLM model using LMStack.

## Prerequisites

Ensure you have completed the [installation](installation.md) and have:

- Backend server running
- At least one worker registered
- Frontend accessible (optional, for UI-based deployment)

## Deploy via Web UI

### 1. Access the Dashboard

Open your browser and navigate to `http://localhost:5173` (development) or your production URL.

### 2. Navigate to Deploy Apps

Click on "Deploy Apps" in the sidebar to see available applications.

### 3. Select a Model

Choose from available options:

- **vLLM**: High-performance inference for large models
- **Ollama**: Easy-to-use with GGUF quantization support
- **SGLang**: Optimized for structured generation

### 4. Configure Deployment

Fill in the deployment form:

- **Model Name**: HuggingFace model ID (e.g., `meta-llama/Llama-2-7b-chat-hf`)
- **Worker**: Select target worker node
- **GPU Memory**: Fraction of GPU memory to use
- **Max Model Length**: Maximum context length

### 5. Deploy

Click "Deploy" and monitor the progress in the Deployments page.

## Deploy via API

### 1. Get API Key

```bash
# Create an API key via the UI or API
curl -X POST http://localhost:8000/api/api-keys \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-api-key"}'
```

### 2. Create Deployment

```bash
curl -X POST http://localhost:8000/api/deployments \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "llama2-7b",
    "backend": "vllm",
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "worker_id": "worker-uuid",
    "config": {
      "gpu_memory_utilization": 0.9,
      "max_model_len": 4096
    }
  }'
```

### 3. Check Deployment Status

```bash
curl http://localhost:8000/api/deployments/DEPLOYMENT_ID \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Using the Model

Once deployed, you can interact with your model using the OpenAI-compatible API.

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2-7b",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

### Streaming Response

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2-7b",
    "messages": [
      {"role": "user", "content": "Write a poem about AI"}
    ],
    "stream": true
  }'
```

### Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="llama2-7b",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

print(response.choices[0].message.content)
```

## Using the Chat Interface

LMStack includes a built-in chat interface for testing models:

1. Navigate to "Chat" in the sidebar
2. Select your deployed model from the dropdown
3. Start chatting!

## Monitoring

### View Deployment Logs

```bash
curl http://localhost:8000/api/deployments/DEPLOYMENT_ID/logs \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Check GPU Usage

Monitor GPU utilization through the Workers page or via API:

```bash
curl http://localhost:8000/api/workers/WORKER_ID \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Next Steps

- [Configuration Guide](configuration.md) - Fine-tune your setup
- [API Reference](../api_reference/index.md) - Complete API documentation
- [Deployment Options](../deployment/index.md) - Production deployment guides
