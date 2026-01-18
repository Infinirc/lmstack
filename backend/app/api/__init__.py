"""API routes"""

from fastapi import APIRouter

from app.api import (
    api_keys,
    apps,
    auth,
    containers,
    conversations,
    dashboard,
    deployments,
    headscale,
    huggingface,
    images,
    model_files,
    models,
    ollama,
    semantic_router,
    storage,
    system,
    workers,
)

api_router = APIRouter()

# Authentication
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])

# Infrastructure management
api_router.include_router(workers.router, prefix="/workers", tags=["workers"])
api_router.include_router(images.router, prefix="/images", tags=["images"])
api_router.include_router(containers.router, prefix="/containers", tags=["containers"])
api_router.include_router(storage.router, prefix="/storage", tags=["storage"])

# AI/ML management
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(deployments.router, prefix="/deployments", tags=["deployments"])
api_router.include_router(model_files.router, prefix="/model-files", tags=["model-files"])

# Dashboard and access control
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(api_keys.router, prefix="/api-keys", tags=["api-keys"])

# System management
api_router.include_router(system.router, prefix="/system", tags=["system"])

# HuggingFace integration
api_router.include_router(huggingface.router, prefix="/huggingface", tags=["huggingface"])

# Ollama library integration
api_router.include_router(ollama.router, prefix="/ollama", tags=["ollama"])

# Chat conversations
api_router.include_router(conversations.router, prefix="/conversations", tags=["conversations"])

# Deploy Apps
api_router.include_router(apps.router, prefix="/apps", tags=["apps"])

# Headscale VPN
api_router.include_router(headscale.router, prefix="/headscale", tags=["headscale"])

# Semantic Router
api_router.include_router(semantic_router.router)
