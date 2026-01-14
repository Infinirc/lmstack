"""API Gateway service for authentication, routing, and usage tracking"""
import logging
from datetime import datetime, date
from typing import Optional, Tuple

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.api_key import ApiKey, Usage
from app.models.deployment import Deployment, DeploymentStatus
from app.models.llm_model import LLMModel
from app.api.api_keys import verify_secret, API_KEY_PREFIX

logger = logging.getLogger(__name__)


class GatewayService:
    """Service for API Gateway operations"""

    @staticmethod
    def parse_api_key(authorization: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse API key from Authorization header.

        Expected format: Bearer lmsk_<access_key>_<secret_key>
        Returns: (access_key, secret_key) or (None, None) if invalid
        """
        if not authorization:
            return None, None

        parts = authorization.split(" ")
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None, None

        token = parts[1]
        if not token.startswith(f"{API_KEY_PREFIX}_"):
            return None, None

        # Remove prefix and split
        key_parts = token[len(API_KEY_PREFIX) + 1:].split("_", 1)
        if len(key_parts) != 2:
            return None, None

        return key_parts[0], key_parts[1]

    @staticmethod
    async def validate_api_key(
        db: AsyncSession,
        access_key: str,
        secret_key: str,
    ) -> Optional[ApiKey]:
        """Validate API key and return the key record if valid."""
        result = await db.execute(
            select(ApiKey).where(ApiKey.access_key == access_key)
        )
        api_key = result.scalar_one_or_none()

        if not api_key:
            return None

        # Verify secret
        if not verify_secret(secret_key, api_key.hashed_secret):
            return None

        # Check expiration
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return None

        # Update last used
        api_key.last_used_at = datetime.utcnow()
        await db.commit()

        return api_key

    @staticmethod
    async def check_model_access(
        api_key: ApiKey,
        model_id: int,
    ) -> bool:
        """Check if API key has access to the specified model."""
        if not api_key.allowed_model_ids:
            # No restrictions, allow all
            return True

        return model_id in api_key.allowed_model_ids

    @staticmethod
    async def find_deployment_for_model(
        db: AsyncSession,
        model_name: str,
    ) -> Optional[Tuple[Deployment, LLMModel]]:
        """Find a running deployment for the given model name.

        Searches by:
        1. Model name (LLMModel.name)
        2. Deployment name (Deployment.name)
        3. Model ID (LLMModel.model_id) - e.g., "Qwen/Qwen2.5-7B-Instruct"

        Returns the deployment and model if found, None otherwise.
        """
        # Try to find by model name first
        result = await db.execute(
            select(LLMModel).where(LLMModel.name == model_name)
        )
        model = result.scalar_one_or_none()

        if model:
            # Find a running deployment for this model
            result = await db.execute(
                select(Deployment)
                .where(
                    and_(
                        Deployment.model_id == model.id,
                        Deployment.status == DeploymentStatus.RUNNING.value,
                    )
                )
                .limit(1)
            )
            deployment = result.scalar_one_or_none()
            if deployment:
                return deployment, model

        # Try to find by deployment name
        result = await db.execute(
            select(Deployment, LLMModel)
            .join(LLMModel, Deployment.model_id == LLMModel.id)
            .where(
                and_(
                    Deployment.name == model_name,
                    Deployment.status == DeploymentStatus.RUNNING.value,
                )
            )
            .limit(1)
        )
        row = result.first()
        if row:
            return row[0], row[1]

        # Try to find by model_id (HuggingFace ID or Ollama model name)
        result = await db.execute(
            select(LLMModel).where(LLMModel.model_id == model_name)
        )
        model = result.scalar_one_or_none()

        if model:
            result = await db.execute(
                select(Deployment)
                .where(
                    and_(
                        Deployment.model_id == model.id,
                        Deployment.status == DeploymentStatus.RUNNING.value,
                    )
                )
                .limit(1)
            )
            deployment = result.scalar_one_or_none()
            if deployment:
                return deployment, model

        return None

    @staticmethod
    async def get_available_models(
        db: AsyncSession,
        api_key: Optional[ApiKey] = None,
    ) -> list[dict]:
        """Get list of available models (with running deployments).

        If api_key is provided, filters by allowed models.
        """
        # Get all running deployments with their models
        query = (
            select(Deployment, LLMModel)
            .join(LLMModel, Deployment.model_id == LLMModel.id)
            .where(Deployment.status == DeploymentStatus.RUNNING.value)
        )

        result = await db.execute(query)
        deployments = result.all()

        models = []
        seen_model_ids = set()

        for deployment, model in deployments:
            # Skip if already seen (multiple deployments of same model)
            if model.id in seen_model_ids:
                continue

            # Check access if api_key provided
            if api_key and api_key.allowed_model_ids:
                if model.id not in api_key.allowed_model_ids:
                    continue

            seen_model_ids.add(model.id)
            created_timestamp = int(model.created_at.timestamp()) if model.created_at else 0
            models.append({
                "id": model.name,
                "object": "model",
                "created": created_timestamp,
                "owned_by": "lmstack",
                "root": model.model_id,
                "parent": None,
                "permission": [
                    {
                        "id": f"modelperm-{model.id}",
                        "object": "model_permission",
                        "created": created_timestamp,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False,
                    }
                ],
            })

        return models

    @staticmethod
    async def record_usage(
        db: AsyncSession,
        api_key_id: Optional[int],
        model_id: Optional[int],
        deployment_id: Optional[int],
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Record API usage for tracking."""
        today = date.today()
        today_datetime = datetime.combine(today, datetime.min.time())

        # Try to find existing usage record for today
        result = await db.execute(
            select(Usage).where(
                and_(
                    Usage.api_key_id == api_key_id,
                    Usage.model_id == model_id,
                    Usage.deployment_id == deployment_id,
                    Usage.date == today_datetime,
                )
            )
        )
        usage = result.scalar_one_or_none()

        if usage:
            # Update existing record
            usage.request_count += 1
            usage.prompt_tokens += prompt_tokens
            usage.completion_tokens += completion_tokens
            usage.updated_at = datetime.utcnow()
        else:
            # Create new record
            usage = Usage(
                api_key_id=api_key_id,
                model_id=model_id,
                deployment_id=deployment_id,
                date=today_datetime,
                request_count=1,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            db.add(usage)

        await db.commit()

    @staticmethod
    def build_upstream_url(worker_address: str, port: int) -> str:
        """Build the upstream URL for the deployment.

        worker_address may include port (e.g., "192.168.1.1:8080"),
        we only need the host part.
        """
        # Extract host from worker address (remove agent port if present)
        host = worker_address.split(":")[0]
        return f"http://{host}:{port}"


gateway_service = GatewayService()
