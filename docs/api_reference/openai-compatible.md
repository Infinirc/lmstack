# OpenAI Compatible API

LMStack provides an OpenAI-compatible API, allowing you to use existing OpenAI SDKs and tools.

## Overview

The `/v1` endpoints mirror the OpenAI API specification, enabling seamless integration with:

- OpenAI Python SDK
- OpenAI Node.js SDK
- LangChain
- LlamaIndex
- Any OpenAI-compatible client

## Base URL

```
http://localhost:8000/v1
```

## Authentication

Use your LMStack API key as the OpenAI API key:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-lmstack-api-key",
    base_url="http://localhost:8000/v1"
)
```

## Endpoints

### List Models

```http
GET /v1/models
```

Returns available deployed models.

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama2-7b",
      "object": "model",
      "created": 1704067200,
      "owned_by": "lmstack"
    }
  ]
}
```

### Chat Completions

```http
POST /v1/chat/completions
```

Generate chat completions.

**Request:**

```json
{
  "model": "llama2-7b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

**Response:**

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "llama2-7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

### Streaming

Set `stream: true` for streaming responses:

```python
stream = client.chat.completions.create(
    model="llama2-7b",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

**Streaming Response Format:**

```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":" there"}}]}

data: [DONE]
```

### Completions (Legacy)

```http
POST /v1/completions
```

Text completion endpoint (legacy).

**Request:**

```json
{
  "model": "llama2-7b",
  "prompt": "The quick brown fox",
  "max_tokens": 100,
  "temperature": 0.7
}
```

## Parameters

### Chat Completion Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model` | string | Model name (required) | - |
| `messages` | array | Conversation messages (required) | - |
| `temperature` | float | Sampling temperature (0-2) | 1.0 |
| `top_p` | float | Nucleus sampling parameter | 1.0 |
| `max_tokens` | integer | Maximum tokens to generate | Model default |
| `stream` | boolean | Enable streaming | false |
| `stop` | string/array | Stop sequences | null |
| `presence_penalty` | float | Presence penalty (-2 to 2) | 0 |
| `frequency_penalty` | float | Frequency penalty (-2 to 2) | 0 |

### Message Format

```json
{
  "role": "user|assistant|system",
  "content": "Message content"
}
```

## Code Examples

### Python

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8000/v1"
)

# Chat completion
response = client.chat.completions.create(
    model="llama2-7b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### JavaScript/TypeScript

```typescript
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: 'your-api-key',
  baseURL: 'http://localhost:8000/v1'
});

const response = await openai.chat.completions.create({
  model: 'llama2-7b',
  messages: [
    { role: 'user', content: 'Hello!' }
  ]
});

console.log(response.choices[0].message.content);
```

### cURL

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2-7b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="llama2-7b",
    openai_api_key="your-api-key",
    openai_api_base="http://localhost:8000/v1"
)

response = llm.invoke("Explain machine learning")
print(response.content)
```

### LlamaIndex

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="llama2-7b",
    api_key="your-api-key",
    api_base="http://localhost:8000/v1"
)

response = llm.complete("The capital of France is")
print(response.text)
```

## Differences from OpenAI API

### Supported Features

| Feature | Status |
|---------|--------|
| Chat Completions | ✅ Supported |
| Streaming | ✅ Supported |
| Completions (legacy) | ✅ Supported |
| Function Calling | ⚠️ Backend dependent |
| Vision | ⚠️ Backend dependent |
| Embeddings | ❌ Not supported |
| Audio | ❌ Not supported |
| Images | ❌ Not supported |

### Notes

1. **Model Names**: Use your deployment names instead of OpenAI model names
2. **Rate Limits**: May differ from OpenAI's limits
3. **Token Counting**: Varies by model tokenizer
4. **Function Calling**: Only supported with compatible backends (vLLM)

## Error Handling

Errors follow OpenAI's format:

```json
{
  "error": {
    "message": "Model not found: invalid-model",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

## Best Practices

1. **Handle Streaming Properly**: Always handle the `[DONE]` signal
2. **Set Timeouts**: Large models may have longer response times
3. **Retry Logic**: Implement exponential backoff for errors
4. **Monitor Usage**: Track token usage for cost management
