"""User database model"""

from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class UserRole(str, Enum):
    """User role enum"""

    ADMIN = "admin"  # Full access - can manage users, settings, everything
    OPERATOR = "operator"  # Can manage deployments, models, workers
    VIEWER = "viewer"  # Read-only access


class User(Base):
    """User account for authentication and authorization"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)

    # User info
    display_name: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Role and status
    role: Mapped[str] = mapped_column(String(20), default=UserRole.VIEWER.value)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"

    def has_permission(self, required_role: UserRole) -> bool:
        """Check if user has at least the required role level."""
        role_hierarchy = {
            UserRole.VIEWER.value: 0,
            UserRole.OPERATOR.value: 1,
            UserRole.ADMIN.value: 2,
        }
        user_level = role_hierarchy.get(self.role, 0)
        required_level = role_hierarchy.get(required_role.value, 0)
        return user_level >= required_level
