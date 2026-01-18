"""Semantic Router Configuration Service

Generates and manages config.yaml for the Semantic Router app.
Supports hot-reload by updating the config file when models change.
"""

import logging
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.app import App, AppStatus, AppType
from app.models.deployment import Deployment, DeploymentStatus

logger = logging.getLogger(__name__)


class SemanticRouterService:
    """Service for managing Semantic Router configuration."""

    # Default categories for semantic routing
    DEFAULT_CATEGORIES = [
        {"name": "math", "description": "Mathematics and quantitative reasoning"},
        {"name": "coding", "description": "Programming and software development"},
        {"name": "science", "description": "Scientific questions and research"},
        {"name": "creative", "description": "Creative writing and brainstorming"},
        {"name": "general", "description": "General knowledge and conversation"},
    ]

    async def generate_config(
        self,
        db: AsyncSession,
        lmstack_api_url: str,
    ) -> dict[str, Any]:
        """Generate semantic router config.yaml content.

        Args:
            db: Database session
            lmstack_api_url: LMStack API URL (e.g., http://host.docker.internal:52000)

        Returns:
            Config dictionary ready to be serialized to YAML
        """
        # Get all running deployments with their models
        result = await db.execute(
            select(Deployment)
            .where(Deployment.status == DeploymentStatus.RUNNING.value)
            .options(selectinload(Deployment.model), selectinload(Deployment.worker))
        )
        deployments = result.scalars().all()

        # Build vllm_endpoints from deployments
        vllm_endpoints = []
        model_configs = {}

        for deployment in deployments:
            if not deployment.model or not deployment.worker:
                continue

            # Use LMStack gateway as the endpoint (semantic router will call LMStack API)
            # Parse host and port from lmstack_api_url
            endpoint_name = f"lmstack-{deployment.model.name}".replace("/", "-").replace(":", "-")

            # Extract host and port from URL
            url_parts = lmstack_api_url.replace("http://", "").replace("https://", "").split(":")
            host = url_parts[0]
            port = int(url_parts[1].split("/")[0]) if len(url_parts) > 1 else 52000

            vllm_endpoints.append(
                {
                    "name": endpoint_name,
                    "address": host,
                    "port": port,
                    "weight": 1,
                }
            )

            # Map model name to endpoint
            model_configs[deployment.model.name] = {
                "preferred_endpoints": [endpoint_name],
            }

        # If no deployments, add a placeholder
        if not vllm_endpoints:
            vllm_endpoints.append(
                {
                    "name": "placeholder",
                    "address": "localhost",
                    "port": 8000,
                    "weight": 1,
                }
            )

        # Build config
        config = {
            # Response API
            "response_api": {
                "enabled": True,
                "store_backend": "memory",
                "ttl_seconds": 86400,
                "max_responses": 1000,
            },
            # Semantic cache
            "semantic_cache": {
                "enabled": True,
                "backend_type": "memory",
                "similarity_threshold": 0.85,
                "max_entries": 1000,
                "ttl_seconds": 3600,
                "embedding_model": "qwen3",
            },
            # Prompt guard (jailbreak protection)
            "prompt_guard": {
                "enabled": True,
                "threshold": 0.7,
                "use_cpu": True,
            },
            # Classifier
            "classifier": {
                "category_model": {
                    "model_id": "models/mom-domain-classifier",
                    "threshold": 0.6,
                    "use_cpu": True,
                },
            },
            # vLLM endpoints (pointing to LMStack)
            "vllm_endpoints": vllm_endpoints,
            # Model configs
            "model_config": model_configs,
            # Categories
            "categories": self.DEFAULT_CATEGORIES,
            # Routing strategy
            "strategy": "priority",
            # Default model (use first available)
            "default_model": list(model_configs.keys())[0] if model_configs else "default",
            # Decisions (routing rules)
            "decisions": self._generate_decisions(list(model_configs.keys())),
            # Embedding models
            "embedding_models": {
                "qwen3_model_path": "models/mom-embedding-pro",
                "use_cpu": True,
            },
            # Observability
            "observability": {
                "metrics": {"enabled": True},
                "tracing": {"enabled": False},
            },
        }

        return config

    def _generate_decisions(self, model_names: list[str]) -> list[dict]:
        """Generate routing decisions based on available models.

        For now, creates a simple default decision that routes all requests
        to the first available model. Users can customize this later.
        """
        if not model_names:
            return []

        default_model = model_names[0]

        return [
            {
                "name": "default_decision",
                "description": "Default routing for all queries",
                "priority": 50,
                "rules": {
                    "operator": "AND",
                    "conditions": [{"type": "domain", "name": "general"}],
                },
                "modelRefs": [{"model": default_model, "use_reasoning": False}],
                "plugins": [
                    {
                        "type": "system_prompt",
                        "configuration": {"system_prompt": "You are a helpful assistant."},
                    },
                    {
                        "type": "semantic-cache",
                        "configuration": {"enabled": True, "similarity_threshold": 0.85},
                    },
                ],
            }
        ]

    def config_to_yaml(self, config: dict) -> str:
        """Convert config dict to YAML string."""
        return yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)

    async def get_semantic_router_app(self, db: AsyncSession) -> App | None:
        """Get the deployed Semantic Router app if exists."""
        result = await db.execute(
            select(App)
            .where(
                App.app_type == AppType.SEMANTIC_ROUTER.value,
                App.status == AppStatus.RUNNING.value,
            )
            .options(selectinload(App.worker))
        )
        return result.scalar_one_or_none()

    async def is_semantic_router_deployed(self, db: AsyncSession) -> bool:
        """Check if Semantic Router is deployed and running."""
        app = await self.get_semantic_router_app(db)
        return app is not None

    async def get_semantic_router_url(self, db: AsyncSession) -> str | None:
        """Get the Semantic Router API URL if deployed."""
        app = await self.get_semantic_router_app(db)
        if not app or not app.worker:
            return None

        # Return the worker address with semantic router port
        worker_address = app.worker.address.split(":")[0]
        return f"http://{worker_address}:{app.port}"


# Global instance
semantic_router_service = SemanticRouterService()
