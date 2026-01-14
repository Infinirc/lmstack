"""
Tests for deployment management.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.deployment import Deployment, DeploymentStatus
from app.models.worker import Worker, WorkerStatus
from app.models.llm_model import LLMModel, BackendType


@pytest.fixture
async def sample_worker(db_session: AsyncSession) -> Worker:
    """Create a sample worker for testing."""
    worker = Worker(
        name="deploy-test-worker",
        hostname="deploy-test-host",
        status=WorkerStatus.ONLINE.value,
        agent_port=8080,
        gpu_count=1,
    )
    db_session.add(worker)
    await db_session.commit()
    await db_session.refresh(worker)
    return worker


@pytest.fixture
async def sample_model(db_session: AsyncSession) -> LLMModel:
    """Create a sample LLM model for testing."""
    model = LLMModel(
        name="test-llama",
        model_id="meta-llama/Llama-2-7b-chat-hf",
        backend=BackendType.VLLM.value,
    )
    db_session.add(model)
    await db_session.commit()
    await db_session.refresh(model)
    return model


class TestDeploymentModel:
    """Test Deployment model functionality."""

    @pytest.mark.asyncio
    async def test_deployment_creation(
        self, db_session: AsyncSession, sample_worker: Worker, sample_model: LLMModel
    ):
        """Test creating a deployment."""
        deployment = Deployment(
            name="test-deployment",
            model_id=sample_model.id,
            worker_id=sample_worker.id,
            backend=BackendType.VLLM.value,
            status=DeploymentStatus.PENDING.value,
        )
        db_session.add(deployment)
        await db_session.commit()

        assert deployment.id is not None
        assert deployment.name == "test-deployment"
        assert deployment.status == "pending"

    @pytest.mark.asyncio
    async def test_deployment_status_transitions(
        self, db_session: AsyncSession, sample_worker: Worker, sample_model: LLMModel
    ):
        """Test deployment status transitions."""
        deployment = Deployment(
            name="status-test-deployment",
            model_id=sample_model.id,
            worker_id=sample_worker.id,
            backend=BackendType.VLLM.value,
            status=DeploymentStatus.PENDING.value,
        )
        db_session.add(deployment)
        await db_session.commit()

        # Transition to downloading
        deployment.status = DeploymentStatus.DOWNLOADING.value
        await db_session.commit()
        assert deployment.status == "downloading"

        # Transition to starting
        deployment.status = DeploymentStatus.STARTING.value
        await db_session.commit()
        assert deployment.status == "starting"

        # Transition to running
        deployment.status = DeploymentStatus.RUNNING.value
        deployment.container_id = "abc123"
        deployment.port = 8001
        await db_session.commit()
        assert deployment.status == "running"
        assert deployment.container_id == "abc123"

    @pytest.mark.asyncio
    async def test_deployment_with_extra_params(
        self, db_session: AsyncSession, sample_worker: Worker, sample_model: LLMModel
    ):
        """Test deployment with extra parameters."""
        extra_params = {
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
            "tensor_parallel_size": 1,
        }

        deployment = Deployment(
            name="params-test-deployment",
            model_id=sample_model.id,
            worker_id=sample_worker.id,
            backend=BackendType.VLLM.value,
            extra_params=extra_params,
        )
        db_session.add(deployment)
        await db_session.commit()
        await db_session.refresh(deployment)

        assert deployment.extra_params == extra_params
        assert deployment.extra_params["gpu_memory_utilization"] == 0.9


class TestDeploymentStatus:
    """Test deployment status enumeration."""

    def test_deployment_status_values(self):
        """Test all deployment status values."""
        assert DeploymentStatus.PENDING.value == "pending"
        assert DeploymentStatus.DOWNLOADING.value == "downloading"
        assert DeploymentStatus.STARTING.value == "starting"
        assert DeploymentStatus.RUNNING.value == "running"
        assert DeploymentStatus.STOPPING.value == "stopping"
        assert DeploymentStatus.STOPPED.value == "stopped"
        assert DeploymentStatus.ERROR.value == "error"

    def test_deployment_status_is_string_enum(self):
        """Test that DeploymentStatus values are strings."""
        for status in DeploymentStatus:
            assert isinstance(status.value, str)


class TestBackendType:
    """Test backend type enumeration."""

    def test_backend_type_values(self):
        """Test all backend type values."""
        assert BackendType.VLLM.value == "vllm"
        assert BackendType.SGLANG.value == "sglang"
        assert BackendType.OLLAMA.value == "ollama"
