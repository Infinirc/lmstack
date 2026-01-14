# API Reference

LMStack provides a comprehensive REST API for managing LLM deployments, workers, and inference.

## API Overview

The API is organized into the following sections:

| Section | Description |
|---------|-------------|
| [REST API](rest-api.md) | Core API for deployments, workers, and management |
| [Worker API](worker-api.md) | Internal API for worker-backend communication |
| [OpenAI Compatible](openai-compatible.md) | Drop-in replacement for OpenAI API |

## Authentication

All API endpoints (except health checks) require authentication using API keys.

### Using API Keys

Include the API key in the `Authorization` header:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8088/api/deployments
```

### Creating API Keys

API keys can be created via the web UI or API:

```bash
curl -X POST http://localhost:8088/api/api-keys \
  -H "Authorization: Bearer ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-api-key"}'
```

## Base URL

- Development: `http://localhost:8088`
- Production: `https://api.yourdomain.com`

## Response Format

All responses are JSON formatted:

```json
{
  "data": { ... },
  "message": "Success",
  "status": "ok"
}
```

### Error Responses

```json
{
  "detail": "Error message",
  "status_code": 400
}
```

## Rate Limiting

API requests may be rate-limited in production environments. Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1609459200
```

## Pagination

List endpoints support pagination:

```bash
GET /api/deployments?skip=0&limit=20
```

Response includes pagination metadata:

```json
{
  "items": [...],
  "total": 100,
  "skip": 0,
  "limit": 20
}
```

## WebSocket Endpoints

Real-time updates are available via WebSocket:

- `/ws/logs/{deployment_id}` - Stream deployment logs
- `/ws/status` - Stream status updates

## SDK Support

### Python

```python
import httpx

client = httpx.Client(
    base_url="http://localhost:8088",
    headers={"Authorization": f"Bearer {api_key}"}
)

# List deployments
response = client.get("/api/deployments")
deployments = response.json()
```

### JavaScript/TypeScript

```typescript
const response = await fetch('http://localhost:8088/api/deployments', {
  headers: {
    'Authorization': `Bearer ${apiKey}`
  }
});
const deployments = await response.json();
```

## Next Steps

- [REST API Reference](rest-api.md) - Complete endpoint documentation
- [OpenAI Compatible API](openai-compatible.md) - Use with OpenAI SDKs
