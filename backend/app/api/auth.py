"""Authentication and user management API routes"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User, UserRole
from app.schemas.user import (
    UserCreate,
    UserUpdate,
    UserResponse,
    UserListResponse,
    LoginRequest,
    TokenResponse,
    SetupRequest,
    SetupStatusResponse,
    PasswordChange,
)
from app.services.auth import auth_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== Dependencies ====================

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


# ==================== Setup Endpoints ====================

@router.get("/setup/status", response_model=SetupStatusResponse)
async def get_setup_status(db: AsyncSession = Depends(get_db)):
    """Check if the system has been initialized."""
    initialized = await auth_service.is_initialized(db)
    return SetupStatusResponse(initialized=initialized)


@router.post("/setup", response_model=TokenResponse)
async def setup_admin(
    request: SetupRequest,
    db: AsyncSession = Depends(get_db),
):
    """Initialize the system with the first admin user."""
    # Check if already initialized
    if await auth_service.is_initialized(db):
        raise HTTPException(status_code=400, detail="System is already initialized")

    # Check for duplicate username
    existing = await auth_service.get_user_by_username(db, request.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Create admin user
    user = await auth_service.initialize_admin(
        db=db,
        username=request.username,
        password=request.password,
        email=request.email,
    )

    # Generate token
    access_token = auth_service.create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role,
    )

    return TokenResponse(
        access_token=access_token,
        user=UserResponse.model_validate(user),
    )


# ==================== Auth Endpoints ====================

@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Login with username and password."""
    user = await auth_service.authenticate_user(db, request.username, request.password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token = auth_service.create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role,
    )

    return TokenResponse(
        access_token=access_token,
        user=UserResponse.model_validate(user),
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse.model_validate(current_user)


@router.post("/me/password")
async def change_password(
    request: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Change current user's password."""
    # Verify current password
    if not auth_service.verify_password(request.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    # Update password
    current_user.hashed_password = auth_service.hash_password(request.new_password)
    await db.commit()

    return {"message": "Password changed successfully"}


# ==================== User Management Endpoints (Admin only) ====================

@router.get("/users", response_model=UserListResponse)
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all users (admin only)."""
    # Count total
    total = await db.scalar(select(func.count()).select_from(User))

    # Get paginated results
    result = await db.execute(
        select(User)
        .offset(skip)
        .limit(limit)
        .order_by(User.created_at.desc())
    )
    users = result.scalars().all()

    return UserListResponse(
        items=[UserResponse.model_validate(u) for u in users],
        total=total or 0,
    )


@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(
    user_in: UserCreate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a new user (admin only)."""
    # Check for duplicate username
    existing = await auth_service.get_user_by_username(db, user_in.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Validate role
    valid_roles = [r.value for r in UserRole]
    if user_in.role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Invalid role. Valid roles: {valid_roles}")

    # Create user
    user = await auth_service.create_user(
        db=db,
        username=user_in.username,
        password=user_in.password,
        role=user_in.role,
        email=user_in.email,
        display_name=user_in.display_name,
    )

    return UserResponse.model_validate(user)


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get a user by ID (admin only)."""
    user = await auth_service.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse.model_validate(user)


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_in: UserUpdate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update a user (admin only)."""
    user = await auth_service.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Prevent self-demotion from admin
    if user.id == current_user.id and user_in.role and user_in.role != UserRole.ADMIN.value:
        raise HTTPException(status_code=400, detail="Cannot demote yourself from admin")

    # Validate role if provided
    if user_in.role:
        valid_roles = [r.value for r in UserRole]
        if user_in.role not in valid_roles:
            raise HTTPException(status_code=400, detail=f"Invalid role. Valid roles: {valid_roles}")

    # Update fields
    update_data = user_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(user, field, value)

    await db.commit()
    await db.refresh(user)

    return UserResponse.model_validate(user)


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete a user (admin only)."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    user = await auth_service.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    await db.delete(user)
    await db.commit()
