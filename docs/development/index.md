# Development

Welcome to the LMStack development guide. This section covers everything you need to contribute to the project.

## Getting Started

1. [Contributing Guide](contributing.md) - How to contribute to LMStack
2. [Architecture Overview](architecture.md) - Understand the system design
3. [Testing Guide](testing.md) - Writing and running tests

## Development Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker
- PostgreSQL (or use Docker)
- Git

### Clone the Repository

```bash
git clone https://github.com/lmstack/lmstack.git
cd lmstack
```

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Start development server
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Worker Setup

```bash
cd worker
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start worker agent
python agent.py
```

## Project Structure

```
lmstack/
├── backend/           # FastAPI backend
│   ├── app/
│   │   ├── api/       # API routes
│   │   ├── core/      # Core configuration
│   │   ├── models/    # Database models
│   │   └── services/  # Business logic
│   ├── tests/         # Backend tests
│   └── alembic/       # Database migrations
│
├── frontend/          # React frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   └── utils/
│   └── tests/         # Frontend tests
│
├── worker/            # Worker agent
│   ├── docker_ops/    # Docker operations
│   ├── routes/        # API routes
│   └── tests/         # Worker tests
│
└── docs/              # Documentation
```

## Code Style

### Python

- Follow PEP 8 style guide
- Use type hints
- Format with Black
- Sort imports with isort
- Lint with ruff

### TypeScript

- Use TypeScript strict mode
- Follow ESLint configuration
- Format with Prettier
- Use functional components with hooks

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
pip install pre-commit
pre-commit install
```

This will automatically run:

- Code formatting (Black, Prettier)
- Linting (ruff, ESLint)
- Type checking (mypy, TypeScript)

## Running Tests

### Backend Tests

```bash
cd backend
pytest
pytest --cov=app  # With coverage
```

### Frontend Tests

```bash
cd frontend
npm run test
npm run test:coverage  # With coverage
```

## Next Steps

- [Contributing Guide](contributing.md) - Learn how to submit changes
- [Architecture](architecture.md) - Understand the codebase
- [Testing](testing.md) - Write effective tests
