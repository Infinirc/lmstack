"""LMStack Worker Agent package.

This package provides the worker agent that runs on GPU nodes to manage
model deployments and container operations.

Module Structure:
- agent.py: Main agent entry point and FastAPI app
- models.py: Pydantic request/response models
- docker_ops/: Docker management operations
- routes/: FastAPI route handlers
- docker_runner.py: Deprecated compatibility shim
"""
