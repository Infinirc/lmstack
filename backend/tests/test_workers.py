"""
Tests for worker management endpoints.
"""
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.worker import Worker, WorkerStatus


@pytest.fixture
async def sample_worker(db_session: AsyncSession) -> Worker:
    """Create a sample worker for testing."""
    worker = Worker(
        name="test-worker",
        hostname="test-host",
        status=WorkerStatus.ONLINE.value,
        agent_port=8080,
        gpu_count=2,
        gpu_info=[
            {"index": 0, "name": "NVIDIA A100", "memory_total": 81920},
            {"index": 1, "name": "NVIDIA A100", "memory_total": 81920},
        ],
    )
    db_session.add(worker)
    await db_session.commit()
    await db_session.refresh(worker)
    return worker


class TestWorkerEndpoints:
    """Test worker management API endpoints."""

    @pytest.mark.asyncio
    async def test_list_workers_empty(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test listing workers when none exist."""
        response = await async_client.get(
            "/api/workers",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    @pytest.mark.asyncio
    async def test_list_workers(
        self, async_client: AsyncClient, auth_headers: dict, sample_worker: Worker
    ):
        """Test listing workers."""
        response = await async_client.get(
            "/api/workers",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "test-worker"
        assert data[0]["status"] == "online"

    @pytest.mark.asyncio
    async def test_get_worker(
        self, async_client: AsyncClient, auth_headers: dict, sample_worker: Worker
    ):
        """Test getting a specific worker."""
        response = await async_client.get(
            f"/api/workers/{sample_worker.id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_worker.id
        assert data["name"] == "test-worker"
        assert data["gpu_count"] == 2

    @pytest.mark.asyncio
    async def test_get_worker_not_found(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test getting a nonexistent worker."""
        response = await async_client.get(
            "/api/workers/99999",
            headers=auth_headers
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_workers_unauthorized(self, async_client: AsyncClient):
        """Test listing workers without authentication."""
        response = await async_client.get("/api/workers")

        assert response.status_code == 401


class TestWorkerModel:
    """Test Worker model functionality."""

    @pytest.mark.asyncio
    async def test_worker_creation(self, db_session: AsyncSession):
        """Test creating a worker."""
        worker = Worker(
            name="model-test-worker",
            hostname="model-test-host",
            status=WorkerStatus.ONLINE.value,
            agent_port=8080,
        )
        db_session.add(worker)
        await db_session.commit()

        assert worker.id is not None
        assert worker.name == "model-test-worker"
        assert worker.status == "online"

    @pytest.mark.asyncio
    async def test_worker_status_enum(self):
        """Test worker status enumeration values."""
        assert WorkerStatus.ONLINE.value == "online"
        assert WorkerStatus.OFFLINE.value == "offline"
        assert WorkerStatus.ERROR.value == "error"
