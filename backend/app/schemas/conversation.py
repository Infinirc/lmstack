"""Conversation and Message schemas for API requests/responses"""

from datetime import datetime

from pydantic import BaseModel, Field


# Message schemas
class MessageBase(BaseModel):
    """Base message schema"""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class MessageCreate(MessageBase):
    """Schema for creating a message"""

    thinking: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class MessageResponse(BaseModel):
    """Message response schema"""

    id: int
    role: str
    content: str
    thinking: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


# Conversation schemas
class ConversationBase(BaseModel):
    """Base conversation schema"""

    title: str = Field(..., max_length=255)


class ConversationCreate(ConversationBase):
    """Schema for creating a conversation"""

    deployment_id: int | None = None
    messages: list[MessageCreate] | None = None


class ConversationUpdate(BaseModel):
    """Schema for updating a conversation"""

    title: str | None = Field(None, max_length=255)


class ConversationResponse(BaseModel):
    """Conversation response schema (without messages)"""

    id: int
    title: str
    deployment_id: int | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetailResponse(ConversationResponse):
    """Conversation response with messages"""

    messages: list[MessageResponse] = []


class ConversationListResponse(BaseModel):
    """Response for conversation list"""

    items: list[ConversationResponse]
    total: int


# Add messages to existing conversation
class AddMessagesRequest(BaseModel):
    """Schema for adding messages to a conversation"""

    messages: list[MessageCreate]
