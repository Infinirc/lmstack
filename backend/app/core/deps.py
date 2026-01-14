"""Shared dependencies for API endpoints"""

from typing import Optional

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User, UserRole
from app.services.auth import auth_service


async def get_current_user(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Dependency to get current authenticated user from JWT token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    token = parts[1]
    payload = auth_service.decode_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = int(payload.get("sub", 0))
    user = await auth_service.get_user_by_id(db, user_id)

    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user


async def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Dependency to optionally get current user (returns None if not authenticated)."""
    if not authorization:
        return None

    try:
        return await get_current_user(authorization, db)
    except HTTPException:
        return None


async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Dependency to require admin role."""
    if not current_user.has_permission(UserRole.ADMIN):
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


async def require_operator(current_user: User = Depends(get_current_user)) -> User:
    """Dependency to require operator role or higher."""
    if not current_user.has_permission(UserRole.OPERATOR):
        raise HTTPException(status_code=403, detail="Operator access required")
    return current_user


async def require_viewer(current_user: User = Depends(get_current_user)) -> User:
    """Dependency to require at least viewer role (any authenticated user)."""
    return current_user
