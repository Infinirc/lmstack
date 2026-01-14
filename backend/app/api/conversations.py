"""Conversation API routes"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.conversation import Conversation, Message
from app.models.user import User
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    MessageResponse,
    AddMessagesRequest,
)
from app.api.auth import get_current_user

router = APIRouter()


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all conversations for the current user"""
    query = select(Conversation).where(Conversation.user_id == current_user.id)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # Get paginated results ordered by updated_at descending
    query = query.offset(skip).limit(limit).order_by(Conversation.updated_at.desc())
    result = await db.execute(query)
    conversations = result.scalars().all()

    return ConversationListResponse(
        items=[ConversationResponse.model_validate(c) for c in conversations],
        total=total or 0,
    )


@router.post("", response_model=ConversationDetailResponse, status_code=201)
async def create_conversation(
    conversation_in: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new conversation"""
    conversation = Conversation(
        user_id=current_user.id,
        title=conversation_in.title,
        deployment_id=conversation_in.deployment_id,
    )

    db.add(conversation)
    await db.flush()  # Get the conversation ID

    # Add initial messages if provided
    if conversation_in.messages:
        for msg_data in conversation_in.messages:
            message = Message(
                conversation_id=conversation.id,
                role=msg_data.role,
                content=msg_data.content,
                thinking=msg_data.thinking,
                prompt_tokens=msg_data.prompt_tokens,
                completion_tokens=msg_data.completion_tokens,
            )
            db.add(message)

    await db.commit()
    await db.refresh(conversation)

    # Load messages
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation.id)
        .options(selectinload(Conversation.messages))
    )
    conversation = result.scalar_one()

    return ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title,
        deployment_id=conversation.deployment_id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[MessageResponse.model_validate(m) for m in conversation.messages],
    )


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a conversation with all messages"""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .options(selectinload(Conversation.messages))
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title,
        deployment_id=conversation.deployment_id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[MessageResponse.model_validate(m) for m in conversation.messages],
    )


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: int,
    conversation_in: ConversationUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a conversation (e.g., title)"""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    update_data = conversation_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(conversation, field, value)

    await db.commit()
    await db.refresh(conversation)

    return ConversationResponse.model_validate(conversation)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation and all its messages"""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await db.delete(conversation)
    await db.commit()


@router.post("/{conversation_id}/messages", response_model=ConversationDetailResponse)
async def add_messages(
    conversation_id: int,
    request: AddMessagesRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Add messages to an existing conversation"""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Add new messages
    for msg_data in request.messages:
        message = Message(
            conversation_id=conversation.id,
            role=msg_data.role,
            content=msg_data.content,
            thinking=msg_data.thinking,
            prompt_tokens=msg_data.prompt_tokens,
            completion_tokens=msg_data.completion_tokens,
        )
        db.add(message)

    await db.commit()

    # Reload conversation with messages
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )
    conversation = result.scalar_one()

    return ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title,
        deployment_id=conversation.deployment_id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[MessageResponse.model_validate(m) for m in conversation.messages],
    )


@router.delete("", status_code=204)
async def clear_all_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete all conversations for the current user"""
    await db.execute(delete(Conversation).where(Conversation.user_id == current_user.id))
    await db.commit()
