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

    # Default domains for semantic routing
    DEFAULT_DOMAINS = [
        {
            "name": "math",
            "description": "Mathematics and quantitative reasoning",
            "mmlu_categories": ["math"],
        },
        {
            "name": "coding",
            "description": "Programming and software development",
            "mmlu_categories": ["computer_science"],
        },
        {
            "name": "science",
            "description": "Scientific questions and research",
            "mmlu_categories": ["physics", "chemistry", "biology"],
        },
        {
            "name": "creative",
            "description": "Creative writing and brainstorming",
            "mmlu_categories": ["other"],
        },
        {
            "name": "general",
            "description": "General knowledge and conversation",
            "mmlu_categories": ["other"],
        },
    ]

    async def generate_config(
        self,
        db: AsyncSession,
        lmstack_api_url: str,
    ) -> dict[str, Any]:
        """Generate semantic router config.yaml content.

        Uses the new v0.1 config format with version, listeners, providers, etc.

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

        # Extract host and port from lmstack_api_url
        url_parts = lmstack_api_url.replace("http://", "").replace("https://", "").split(":")
        host = url_parts[0]
        port = int(url_parts[1].split("/")[0]) if len(url_parts) > 1 else 52000

        # Build models list for providers
        models = []
        model_names = []

        for deployment in deployments:
            if not deployment.model or not deployment.worker:
                continue

            model_name = deployment.model.name.replace("/", "-").replace(":", "-")
            endpoint_name = f"lmstack-{model_name}"
            model_names.append(model_name)

            models.append(
                {
                    "name": model_name,
                    "endpoints": [
                        {
                            "name": endpoint_name,
                            "weight": 1,
                            "endpoint": f"{host}:{port}",
                            "protocol": "http",
                        }
                    ],
                }
            )

        # If no deployments, add a placeholder model
        if not models:
            models.append(
                {
                    "name": "default",
                    "endpoints": [
                        {
                            "name": "placeholder",
                            "weight": 1,
                            "endpoint": f"{host}:{port}",
                            "protocol": "http",
                        }
                    ],
                }
            )
            model_names.append("default")

        default_model = model_names[0]

        # Build config in new v0.1 format
        config = {
            "version": "v0.1",
            # Listener configuration
            "listeners": [
                {
                    "name": "http-8888",
                    "address": "0.0.0.0",
                    "port": 8888,
                    "timeout": "300s",
                }
            ],
            # Response API
            "response_api": {
                "enabled": True,
                "store_backend": "memory",
                "ttl_seconds": 86400,
                "max_responses": 1000,
            },
            # Semantic cache - disabled by default (requires embedding model)
            # Enable and configure embedding_model if you have HF_TOKEN for gated models
            "semantic_cache": {
                "enabled": False,
            },
            # Prompt guard (jailbreak protection)
            "prompt_guard": {
                "enabled": True,
                "model_id": "models/mom-jailbreak-classifier",
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
                "pii_model": {
                    "model_id": "models/mom-pii-classifier",
                    "threshold": 0.9,
                    "use_cpu": True,
                },
            },
            # Hallucination mitigation (disabled by default)
            "hallucination_mitigation": {
                "enabled": False,
            },
            # Signals (domains for routing)
            "signals": {
                "domains": self.DEFAULT_DOMAINS,
            },
            # Decisions (routing rules)
            "decisions": self._generate_decisions(model_names, default_model),
            # Providers (models and endpoints)
            "providers": {
                "models": models,
                "default_model": default_model,
                "reasoning_families": {},
                "default_reasoning_effort": "high",
            },
        }

        return config

    def _generate_decisions(self, model_names: list[str], default_model: str) -> list[dict]:
        """Generate routing decisions based on available models.

        Creates decisions for each domain that route to available models.
        """
        if not model_names:
            return []

        decisions = []

        # Math decision - use reasoning if available
        decisions.append(
            {
                "name": "math_decision",
                "description": "Mathematics and quantitative reasoning",
                "priority": 100,
                "rules": {
                    "operator": "AND",
                    "conditions": [{"type": "domain", "name": "math"}],
                },
                "modelRefs": [{"model": default_model, "use_reasoning": True}],
                "plugins": [
                    {
                        "type": "system_prompt",
                        "configuration": {
                            "system_prompt": "You are a mathematics expert. Provide step-by-step solutions with clear reasoning."
                        },
                    },
                ],
            }
        )

        # Coding decision
        decisions.append(
            {
                "name": "coding_decision",
                "description": "Programming and software development",
                "priority": 100,
                "rules": {
                    "operator": "AND",
                    "conditions": [{"type": "domain", "name": "coding"}],
                },
                "modelRefs": [{"model": default_model, "use_reasoning": False}],
                "plugins": [
                    {
                        "type": "system_prompt",
                        "configuration": {
                            "system_prompt": "You are a programming expert. Provide clean, well-documented code with explanations."
                        },
                    },
                ],
            }
        )

        # Science decision
        decisions.append(
            {
                "name": "science_decision",
                "description": "Scientific questions and research",
                "priority": 100,
                "rules": {
                    "operator": "AND",
                    "conditions": [{"type": "domain", "name": "science"}],
                },
                "modelRefs": [{"model": default_model, "use_reasoning": True}],
                "plugins": [
                    {
                        "type": "system_prompt",
                        "configuration": {
                            "system_prompt": "You are a science expert. Explain concepts clearly with scientific accuracy."
                        },
                    },
                ],
            }
        )

        # General/default decision (lowest priority)
        decisions.append(
            {
                "name": "general_decision",
                "description": "General knowledge and miscellaneous",
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
        )

        return decisions

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
