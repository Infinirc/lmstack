"""User schemas for API requests/responses"""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """Base user schema"""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr | None = None
    display_name: str | None = Field(None, max_length=100)


class UserCreate(UserBase):
    """Schema for creating a new user"""

    password: str = Field(..., min_length=6, max_length=100)
    role: str = Field(default="viewer")


class UserUpdate(BaseModel):
    """Schema for updating a user"""

    email: EmailStr | None = None
    display_name: str | None = Field(None, max_length=100)
    role: str | None = None
    is_active: bool | None = None


class PasswordChange(BaseModel):
    """Schema for password change"""

    current_password: str
    new_password: str = Field(..., min_length=6, max_length=100)


class UserResponse(BaseModel):
    """User response schema (without password)"""

    id: int
    username: str
    email: str | None
    display_name: str | None
    role: str
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None

    model_config = {"from_attributes": True}


class UserListResponse(BaseModel):
    """Response for user list"""

    items: list[UserResponse]
    total: int


# Authentication schemas
class LoginRequest(BaseModel):
    """Login request schema"""

    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response after login"""

    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class SetupRequest(BaseModel):
    """Initial setup request (first admin user)"""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)
    email: EmailStr | None = None


class SetupStatusResponse(BaseModel):
    """Setup status response"""

    initialized: bool
