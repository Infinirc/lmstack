# Testing

This guide covers testing practices for LMStack.

## Overview

LMStack uses:

- **pytest** for backend testing
- **vitest** for frontend testing

## Backend Testing

### Setup

```bash
cd backend
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_deployments.py

# Run specific test
pytest tests/test_deployments.py::test_create_deployment

# Run with verbose output
pytest -v

# Run and stop on first failure
pytest -x
```

### Test Structure

```
backend/tests/
├── conftest.py          # Shared fixtures
├── test_deployments.py  # Deployment tests
├── test_workers.py      # Worker tests
├── test_users.py        # User tests
├── test_api_keys.py     # API key tests
└── test_openai.py       # OpenAI API tests
```

### Writing Tests

#### Unit Tests

```python
# tests/test_deployments.py
import pytest
from app.services.deployments import create_deployment, get_deployment

def test_create_deployment():
    """Test deployment creation."""
    deployment = create_deployment(
        name="test-model",
        backend="vllm",
        model="meta-llama/Llama-2-7b-chat-hf"
    )

    assert deployment.name == "test-model"
    assert deployment.backend == "vllm"
    assert deployment.status == "pending"

def test_create_deployment_invalid_backend():
    """Test deployment with invalid backend."""
    with pytest.raises(ValueError) as exc_info:
        create_deployment(
            name="test",
            backend="invalid",
            model="some-model"
        )

    assert "Invalid backend" in str(exc_info.value)
```

#### Async Tests

```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_list_deployments(async_client: AsyncClient):
    """Test listing deployments."""
    response = await async_client.get("/api/deployments")

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
```

#### Using Fixtures

```python
# tests/conftest.py
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

@pytest.fixture
async def db_session() -> AsyncSession:
    """Create a test database session."""
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()

@pytest.fixture
async def async_client(db_session: AsyncSession) -> AsyncClient:
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def sample_deployment(db_session: AsyncSession):
    """Create a sample deployment for testing."""
    deployment = Deployment(
        name="test-deployment",
        backend="vllm",
        model="test-model",
        status="running"
    )
    db_session.add(deployment)
    db_session.commit()
    return deployment
```

#### Mocking

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_deploy_to_worker():
    """Test deployment to worker with mocked Docker."""
    mock_docker = AsyncMock()
    mock_docker.containers.run.return_value.id = "container-123"

    with patch("app.services.docker_client", mock_docker):
        result = await deploy_model(config)

        assert result.container_id == "container-123"
        mock_docker.containers.run.assert_called_once()
```

### API Testing

```python
@pytest.mark.asyncio
async def test_create_deployment_api(
    async_client: AsyncClient,
    auth_headers: dict
):
    """Test deployment creation via API."""
    response = await async_client.post(
        "/api/deployments",
        json={
            "name": "api-test",
            "backend": "vllm",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "worker_id": "worker-123"
        },
        headers=auth_headers
    )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "api-test"
    assert data["status"] == "pending"
```

## Frontend Testing

### Setup

```bash
cd frontend
npm install
```

### Running Tests

```bash
# Run all tests
npm run test

# Run with UI
npm run test:ui

# Run with coverage
npm run test:coverage

# Run specific file
npm run test -- src/components/DeploymentCard.test.tsx

# Watch mode
npm run test -- --watch
```

### Test Structure

```
frontend/src/
├── components/
│   ├── DeploymentCard.tsx
│   └── DeploymentCard.test.tsx
├── pages/
│   ├── Deployments.tsx
│   └── Deployments.test.tsx
└── hooks/
    ├── useDeployments.ts
    └── useDeployments.test.ts
```

### Writing Tests

#### Component Tests

```typescript
// src/components/DeploymentCard.test.tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { DeploymentCard } from './DeploymentCard';

describe('DeploymentCard', () => {
  const mockDeployment = {
    id: '1',
    name: 'llama2-7b',
    backend: 'vllm',
    status: 'running',
    model: 'meta-llama/Llama-2-7b-chat-hf'
  };

  it('renders deployment name', () => {
    render(<DeploymentCard deployment={mockDeployment} />);

    expect(screen.getByText('llama2-7b')).toBeInTheDocument();
  });

  it('shows running status', () => {
    render(<DeploymentCard deployment={mockDeployment} />);

    expect(screen.getByText('running')).toBeInTheDocument();
  });

  it('displays backend type', () => {
    render(<DeploymentCard deployment={mockDeployment} />);

    expect(screen.getByText('vLLM')).toBeInTheDocument();
  });
});
```

#### Testing User Interactions

```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

describe('DeploymentActions', () => {
  it('calls onStop when stop button clicked', async () => {
    const onStop = vi.fn();
    const user = userEvent.setup();

    render(<DeploymentActions onStop={onStop} />);

    await user.click(screen.getByRole('button', { name: /stop/i }));

    expect(onStop).toHaveBeenCalledTimes(1);
  });
});
```

#### Hook Tests

```typescript
// src/hooks/useDeployments.test.ts
import { renderHook, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { useDeployments } from './useDeployments';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  });

  return ({ children }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

describe('useDeployments', () => {
  it('fetches deployments', async () => {
    const mockData = [{ id: '1', name: 'test' }];
    vi.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ items: mockData })
    });

    const { result } = renderHook(() => useDeployments(), {
      wrapper: createWrapper()
    });

    await waitFor(() => {
      expect(result.current.data).toEqual(mockData);
    });
  });
});
```

#### Mocking API Calls

```typescript
import { vi } from 'vitest';
import { rest } from 'msw';
import { setupServer } from 'msw/node';

const server = setupServer(
  rest.get('/api/deployments', (req, res, ctx) => {
    return res(ctx.json({ items: [{ id: '1', name: 'test' }] }));
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
```

## Test Coverage

### Backend Coverage

```bash
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

### Frontend Coverage

```bash
npm run test:coverage
```

### Coverage Requirements

- Aim for >80% coverage on critical paths
- All API endpoints should have tests
- Business logic should be thoroughly tested

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          cd backend
          pytest --cov=app

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: |
          cd frontend
          npm install
      - name: Run tests
        run: |
          cd frontend
          npm run test:coverage
```

## Best Practices

1. **Write tests alongside code**: Don't leave testing for later
2. **Test behavior, not implementation**: Focus on what code does
3. **Use descriptive test names**: Make failures easy to understand
4. **Keep tests independent**: Each test should run in isolation
5. **Mock external dependencies**: Database, APIs, file system
6. **Test edge cases**: Empty inputs, errors, boundary conditions
