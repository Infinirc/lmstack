"""Registration token model for worker authentication"""

import secrets
from datetime import datetime, timedelta

from sqlalchemy import Boolean, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def generate_token() -> str:
    """Generate a secure registration token"""
    return secrets.token_urlsafe(32)


class RegistrationToken(Base):
    """Registration token for worker authentication"""

    __tablename__ = "registration_tokens"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    token: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, default=generate_token
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)  # Suggested worker name
    is_used: Mapped[bool] = mapped_column(Boolean, default=False)
    used_by_worker_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @classmethod
    def create(cls, name: str, expires_in_hours: int = 24) -> "RegistrationToken":
        """Create a new registration token"""
        return cls(
            token=generate_token(),
            name=name,
            expires_at=datetime.utcnow() + timedelta(hours=expires_in_hours),
        )

    @property
    def is_valid(self) -> bool:
        """Check if token is still valid"""
        return not self.is_used and datetime.utcnow() < self.expires_at

    def __repr__(self) -> str:
        return f"<RegistrationToken(id={self.id}, name='{self.name}', is_used={self.is_used})>"
