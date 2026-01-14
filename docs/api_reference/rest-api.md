# REST API

Complete reference for the LMStack REST API.

## Deployments

### List Deployments

```http
GET /api/deployments
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `skip` | integer | Number of records to skip (default: 0) |
| `limit` | integer | Maximum records to return (default: 100) |
| `status` | string | Filter by status |
| `worker_id` | string | Filter by worker |

**Response:**

```json
{
  "items": [
    {
      "id": "uuid",
      "name": "llama2-7b",
      "backend": "vllm",
      "model": "meta-llama/Llama-2-7b-chat-hf",
      "status": "running",
      "worker_id": "worker-uuid",
      "created_at": "2024-01-01T00:00:00Z",
      "config": {
        "gpu_memory_utilization": 0.9,
        "max_model_len": 4096
      }
    }
  ],
  "total": 1
}
```

### Get Deployment

```http
GET /api/deployments/{deployment_id}
```

**Response:**

```json
{
  "id": "uuid",
  "name": "llama2-7b",
  "backend": "vllm",
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "status": "running",
  "worker_id": "worker-uuid",
  "container_id": "container-hash",
  "port": 8001,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z",
  "config": {}
}
```

### Create Deployment

```http
POST /api/deployments
```

**Request Body:**

```json
{
  "name": "llama2-7b",
  "backend": "vllm",
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "worker_id": "worker-uuid",
  "config": {
    "gpu_memory_utilization": 0.9,
    "max_model_len": 4096
  }
}
```

**Response:** `201 Created`

```json
{
  "id": "uuid",
  "name": "llama2-7b",
  "status": "pending"
}
```

### Delete Deployment

```http
DELETE /api/deployments/{deployment_id}
```

**Response:** `204 No Content`

### Get Deployment Logs

```http
GET /api/deployments/{deployment_id}/logs
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `lines` | integer | Number of log lines (default: 100) |

**Response:**

```json
{
  "logs": "Container log output..."
}
```

## Workers

### List Workers

```http
GET /api/workers
```

**Response:**

```json
{
  "items": [
    {
      "id": "uuid",
      "name": "gpu-worker-01",
      "status": "online",
      "gpu_info": {
        "count": 2,
        "devices": [
          {"name": "NVIDIA A100", "memory": 81920}
        ]
      },
      "last_heartbeat": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### Get Worker

```http
GET /api/workers/{worker_id}
```

**Response:**

```json
{
  "id": "uuid",
  "name": "gpu-worker-01",
  "status": "online",
  "gpu_info": {},
  "system_info": {
    "cpu_count": 32,
    "memory_total": 128000000000
  },
  "deployments": []
}
```

### Get Worker Containers

```http
GET /api/workers/{worker_id}/containers
```

**Response:**

```json
{
  "containers": [
    {
      "id": "container-hash",
      "name": "lmstack-llama2-7b",
      "status": "running",
      "image": "vllm/vllm-openai:latest"
    }
  ]
}
```

## API Keys

### List API Keys

```http
GET /api/api-keys
```

**Response:**

```json
{
  "items": [
    {
      "id": "uuid",
      "name": "my-api-key",
      "prefix": "lmsk_",
      "created_at": "2024-01-01T00:00:00Z",
      "last_used": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### Create API Key

```http
POST /api/api-keys
```

**Request Body:**

```json
{
  "name": "my-api-key"
}
```

**Response:** `201 Created`

```json
{
  "id": "uuid",
  "name": "my-api-key",
  "key": "lmsk_xxxxxxxxxxxxx"
}
```

!!! warning
    The full API key is only returned once at creation time. Store it securely.

### Delete API Key

```http
DELETE /api/api-keys/{key_id}
```

**Response:** `204 No Content`

## Users

### List Users

```http
GET /api/users
```

**Response:**

```json
{
  "items": [
    {
      "id": "uuid",
      "username": "admin",
      "email": "admin@example.com",
      "is_admin": true,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### Create User

```http
POST /api/users
```

**Request Body:**

```json
{
  "username": "newuser",
  "email": "user@example.com",
  "password": "secure-password",
  "is_admin": false
}
```

### Update User

```http
PATCH /api/users/{user_id}
```

**Request Body:**

```json
{
  "email": "newemail@example.com"
}
```

### Delete User

```http
DELETE /api/users/{user_id}
```

## Health

### Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy"
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid or missing API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource doesn't exist |
| 409 | Conflict - Resource already exists |
| 422 | Validation Error - Invalid request body |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Worker offline |
