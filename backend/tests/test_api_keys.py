"""
Tests for API key management.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.api_key import ApiKey
from app.models.user import User


class TestApiKeyModel:
    """Test ApiKey model functionality."""

    @pytest.mark.asyncio
    async def test_api_key_creation(self, db_session: AsyncSession, test_user: User):
        """Test creating an API key."""
        api_key = ApiKey(
            name="test-api-key",
            user_id=test_user.id,
        )
        # Generate key and set hash
        raw_key = api_key.generate_key()

        db_session.add(api_key)
        await db_session.commit()

        assert api_key.id is not None
        assert api_key.name == "test-api-key"
        assert api_key.key_hash is not None
        assert raw_key.startswith("lmsk_")

    @pytest.mark.asyncio
    async def test_api_key_verification(self, db_session: AsyncSession, test_user: User):
        """Test API key verification."""
        api_key = ApiKey(
            name="verify-test-key",
            user_id=test_user.id,
        )
        raw_key = api_key.generate_key()

        db_session.add(api_key)
        await db_session.commit()

        # Verify correct key
        assert api_key.verify_key(raw_key)

        # Verify wrong key
        assert not api_key.verify_key("wrong-key")
        assert not api_key.verify_key("lmsk_wrongprefix")


class TestApiKeyEndpoints:
    """Test API key management endpoints."""

    @pytest.mark.asyncio
    async def test_list_api_keys_empty(self, async_client: AsyncClient, auth_headers: dict):
        """Test listing API keys when none exist."""
        response = await async_client.get("/api/api-keys", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_create_api_key(self, async_client: AsyncClient, auth_headers: dict):
        """Test creating an API key."""
        response = await async_client.post(
            "/api/api-keys", json={"name": "new-api-key"}, headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "key" in data
        assert data["key"].startswith("lmsk_")
        assert data["name"] == "new-api-key"

    @pytest.mark.asyncio
    async def test_create_api_key_unauthorized(self, async_client: AsyncClient):
        """Test creating API key without authentication."""
        response = await async_client.post("/api/api-keys", json={"name": "unauthorized-key"})

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_delete_api_key(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
        test_user: User,
    ):
        """Test deleting an API key."""
        # Create a key first
        api_key = ApiKey(
            name="delete-test-key",
            user_id=test_user.id,
        )
        api_key.generate_key()
        db_session.add(api_key)
        await db_session.commit()
        await db_session.refresh(api_key)

        # Delete the key
        response = await async_client.delete(f"/api/api-keys/{api_key.id}", headers=auth_headers)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_nonexistent_api_key(self, async_client: AsyncClient, auth_headers: dict):
        """Test deleting a nonexistent API key."""
        response = await async_client.delete("/api/api-keys/99999", headers=auth_headers)

        assert response.status_code == 404
