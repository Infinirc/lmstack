"""API routes for LMStack Worker.

This package contains the FastAPI route handlers organized by domain:
- deployment.py: Model deployment endpoints
- images.py: Docker image management endpoints
- containers.py: Docker container management endpoints
- storage.py: Storage and volume management endpoints
- native.py: Native deployment endpoints (Mac without Docker)
"""

from .containers import router as containers_router
from .deployment import router as deployment_router
from .images import router as images_router
from .native import router as native_router
from .storage import router as storage_router

__all__ = [
    "deployment_router",
    "images_router",
    "containers_router",
    "storage_router",
    "native_router",
]
