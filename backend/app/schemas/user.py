"""User schemas for API requests/responses"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """Base user schema"""

    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    display_name: Optional[str] = Field(None, max_length=100)


class UserCreate(UserBase):
    """Schema for creating a new user"""

    password: str = Field(..., min_length=6, max_length=100)
    role: str = Field(default="viewer")


class UserUpdate(BaseModel):
    """Schema for updating a user"""

    email: Optional[EmailStr] = None
    display_name: Optional[str] = Field(None, max_length=100)
    role: Optional[str] = None
    is_active: Optional[bool] = None


class PasswordChange(BaseModel):
    """Schema for password change"""

    current_password: str
    new_password: str = Field(..., min_length=6, max_length=100)


class UserResponse(BaseModel):
    """User response schema (without password)"""

    id: int
    username: str
    email: Optional[str]
    display_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    last_login_at: Optional[datetime]

    model_config = {"from_attributes": True}


class UserListResponse(BaseModel):
    """Response for user list"""

    items: List[UserResponse]
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
    email: Optional[EmailStr] = None


class SetupStatusResponse(BaseModel):
    """Setup status response"""

    initialized: bool
