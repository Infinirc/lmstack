"""
Tests for authentication endpoints and services.
"""
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.services.auth import auth_service


class TestAuthService:
    """Test auth service functions."""

    def test_create_access_token(self):
        """Test JWT token creation."""
        token = auth_service.create_access_token(
            data={"sub": "1", "username": "testuser"}
        )

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_decode_token_valid(self):
        """Test decoding a valid JWT token."""
        token = auth_service.create_access_token(
            data={"sub": "1", "username": "testuser"}
        )

        payload = auth_service.decode_token(token)

        assert payload is not None
        assert payload["sub"] == "1"
        assert payload["username"] == "testuser"

    def test_decode_token_invalid(self):
        """Test decoding an invalid token returns None."""
        payload = auth_service.decode_token("invalid-token")

        assert payload is None

    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "securepassword123"
        hashed = auth_service.get_password_hash(password)

        assert hashed != password
        assert auth_service.verify_password(password, hashed)
        assert not auth_service.verify_password("wrongpassword", hashed)


class TestUserModel:
    """Test User model methods."""

    @pytest.mark.asyncio
    async def test_user_password(self, db_session: AsyncSession):
        """Test user password setting and verification."""
        user = User(
            username="passtest",
            email="pass@test.com",
            role="viewer",
        )
        user.set_password("mypassword")

        assert user.password_hash is not None
        assert user.password_hash != "mypassword"
        assert user.check_password("mypassword")
        assert not user.check_password("wrongpassword")


class TestAuthEndpoints:
    """Test authentication API endpoints."""

    @pytest.mark.asyncio
    async def test_login_success(
        self, async_client: AsyncClient, test_user: User
    ):
        """Test successful login."""
        response = await async_client.post(
            "/api/auth/login",
            json={
                "username": "testuser",
                "password": "testpassword123"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_wrong_password(
        self, async_client: AsyncClient, test_user: User
    ):
        """Test login with wrong password."""
        response = await async_client.post(
            "/api/auth/login",
            json={
                "username": "testuser",
                "password": "wrongpassword"
            }
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, async_client: AsyncClient):
        """Test login with nonexistent user."""
        response = await async_client.post(
            "/api/auth/login",
            json={
                "username": "nonexistent",
                "password": "somepassword"
            }
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user(
        self, async_client: AsyncClient, auth_headers: dict, test_user: User
    ):
        """Test getting current user info."""
        response = await async_client.get(
            "/api/auth/me",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user.username
        assert data["email"] == test_user.email

    @pytest.mark.asyncio
    async def test_get_current_user_no_token(self, async_client: AsyncClient):
        """Test getting current user without token."""
        response = await async_client.get("/api/auth/me")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(
        self, async_client: AsyncClient
    ):
        """Test getting current user with invalid token."""
        response = await async_client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer invalid-token"}
        )

        assert response.status_code == 401
