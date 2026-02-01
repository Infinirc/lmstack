"""Deployer service package - handles model deployment on workers.

This package provides the DeployerService class for deploying models
to workers using various backends (vLLM, SGLang, Ollama) and deployment
methods (Docker, native).
"""

from .service import DeployerService, _update_semantic_router_config_background

__all__ = ["DeployerService", "_update_semantic_router_config_background"]
