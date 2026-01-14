"""Authentication service for user login and JWT management"""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta

from jose import JWTError, jwt
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.user import User, UserRole

logger = logging.getLogger(__name__)
settings = get_settings()

# JWT settings
SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes


class AuthService:
    """Service for authentication operations"""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using PBKDF2-SHA256."""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
        ).hex()
        return f"{salt}${pwd_hash}"

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt, pwd_hash = hashed_password.split("$")
            new_hash = hashlib.pbkdf2_hmac(
                "sha256", plain_password.encode("utf-8"), salt.encode("utf-8"), 100000
            ).hex()
            return secrets.compare_digest(pwd_hash, new_hash)
        except (ValueError, AttributeError) as e:
            # Invalid hash format or missing password
            logger.debug(f"Password verification failed: {e}")
            return False

    @staticmethod
    def create_access_token(
        user_id: int,
        username: str,
        role: str,
        expires_delta: timedelta | None = None,
    ) -> str:
        """Create a JWT access token."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode = {
            "sub": str(user_id),
            "username": username,
            "role": role,
            "exp": expire,
        }
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def decode_token(token: str) -> dict | None:
        """Decode and validate a JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            logger.warning(f"JWT decode error: {e}")
            return None

    @staticmethod
    async def authenticate_user(
        db: AsyncSession,
        username: str,
        password: str,
    ) -> User | None:
        """Authenticate a user with username and password."""
        result = await db.execute(select(User).where(User.username == username))
        user = result.scalar_one_or_none()

        if not user:
            return None

        if not user.is_active:
            return None

        if not AuthService.verify_password(password, user.hashed_password):
            return None

        # Update last login
        user.last_login_at = datetime.utcnow()
        await db.commit()

        return user

    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: int) -> User | None:
        """Get a user by ID."""
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
        """Get a user by username."""
        result = await db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    @staticmethod
    async def create_user(
        db: AsyncSession,
        username: str,
        password: str,
        role: str = UserRole.VIEWER.value,
        email: str | None = None,
        display_name: str | None = None,
    ) -> User:
        """Create a new user."""
        user = User(
            username=username,
            hashed_password=AuthService.hash_password(password),
            role=role,
            email=email,
            display_name=display_name or username,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user

    @staticmethod
    async def is_initialized(db: AsyncSession) -> bool:
        """Check if the system has been initialized (has at least one admin)."""
        count = await db.scalar(
            select(func.count())
            .select_from(User)
            .where(User.role == UserRole.ADMIN.value)
        )
        return count > 0

    @staticmethod
    async def initialize_admin(
        db: AsyncSession,
        username: str,
        password: str,
        email: str | None = None,
    ) -> User:
        """Initialize the first admin user."""
        # Check if already initialized
        if await AuthService.is_initialized(db):
            raise ValueError("System is already initialized")

        return await AuthService.create_user(
            db=db,
            username=username,
            password=password,
            role=UserRole.ADMIN.value,
            email=email,
            display_name=username,
        )


auth_service = AuthService()
