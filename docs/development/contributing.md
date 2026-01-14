# Contributing

Thank you for your interest in contributing to LMStack! This guide will help you get started.

## Code of Conduct

Please be respectful and constructive in all interactions. We're building an inclusive community.

## How to Contribute

### Reporting Issues

1. Search existing issues first
2. Use the issue template
3. Provide clear reproduction steps
4. Include system information

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write/update tests
5. Submit a pull request

## Development Workflow

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/lmstack.git
cd lmstack
git remote add upstream https://github.com/lmstack/lmstack.git
```

### 2. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Follow code style guidelines
- Add tests for new functionality
- Update documentation if needed

### 4. Test Your Changes

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm run test
npm run lint
```

### 5. Commit Changes

Follow conventional commits:

```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug in deployment"
git commit -m "docs: update API documentation"
```

**Commit Types:**

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style changes |
| `refactor` | Code refactoring |
| `test` | Adding tests |
| `chore` | Maintenance tasks |

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Pull Request Guidelines

### Title

Use conventional commit format:
- `feat: add deployment status webhook`
- `fix: resolve memory leak in worker`

### Description

Include:
- What changes were made
- Why the changes are needed
- How to test the changes
- Screenshots (for UI changes)

### Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changelog updated (for significant changes)

## Code Review

All PRs require review before merging:

1. Address reviewer comments
2. Keep discussions constructive
3. Squash commits if requested

## Development Guidelines

### Backend (Python)

```python
# Use type hints
async def get_deployment(deployment_id: str) -> Deployment:
    ...

# Use async/await for I/O operations
async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Handle errors gracefully
try:
    result = await risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

### Frontend (TypeScript)

```typescript
// Use TypeScript interfaces
interface Deployment {
  id: string;
  name: string;
  status: DeploymentStatus;
}

// Use functional components
function DeploymentCard({ deployment }: { deployment: Deployment }) {
  return (
    <Card>
      <h3>{deployment.name}</h3>
      <Status value={deployment.status} />
    </Card>
  );
}

// Use custom hooks for logic reuse
function useDeployments() {
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  // ...
  return { deployments, isLoading, error };
}
```

### Worker (Python)

```python
# Handle Docker operations safely
async def run_container(config: ContainerConfig) -> str:
    try:
        container = await docker.containers.run(
            image=config.image,
            detach=True,
            ...
        )
        return container.id
    except docker.errors.ImageNotFound:
        raise DeploymentError(f"Image not found: {config.image}")
```

## Testing Guidelines

### Unit Tests

Test individual functions/components:

```python
# backend/tests/test_deployments.py
def test_create_deployment():
    deployment = create_deployment(name="test", model="llama2")
    assert deployment.name == "test"
    assert deployment.status == "pending"
```

### Integration Tests

Test component interactions:

```python
async def test_deployment_workflow():
    # Create deployment
    response = await client.post("/api/deployments", json={...})
    assert response.status_code == 201

    # Check status
    response = await client.get(f"/api/deployments/{id}")
    assert response.json()["status"] == "pending"
```

### Frontend Tests

```typescript
// Test component rendering
test('renders deployment card', () => {
  render(<DeploymentCard deployment={mockDeployment} />);
  expect(screen.getByText('llama2-7b')).toBeInTheDocument();
});
```

## Getting Help

- Open a discussion on GitHub
- Join our community chat
- Check existing issues and documentation

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to LMStack!
