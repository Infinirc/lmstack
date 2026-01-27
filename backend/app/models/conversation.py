"""Conversation and Message database models"""

from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ConversationType(str, Enum):
    """Type of conversation"""

    CHAT = "chat"  # Traditional chat with deployment
    AGENT = "agent"  # MCP-based agent chat


class Conversation(Base):
    """Chat conversation"""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)

    # Conversation type: "chat" or "agent"
    conversation_type: Mapped[str] = mapped_column(
        String(20), default=ConversationType.CHAT.value, nullable=False
    )

    # Optional: link to deployment used (for chat type)
    deployment_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("deployments.id"), nullable=True
    )

    # Agent configuration (for agent type)
    agent_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title='{self.title}')>"


class MessageRole(str, Enum):
    """Role of message sender"""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"  # For agent tool results


class MessageStepType(str, Enum):
    """Type of agent execution step"""

    THINKING = "thinking"
    PLANNING = "planning"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MESSAGE = "message"


class Message(Base):
    """Chat message within a conversation"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Message content
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user', 'assistant', or 'tool'
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Optional: thinking content for assistant messages
    thinking: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Token usage (optional)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Agent-specific fields
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)  # List of tool calls
    tool_call_id: Mapped[str | None] = mapped_column(String(100), nullable=True)  # For tool results
    step_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # thinking, tool_call, etc.
    execution_time_ms: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )  # Tool execution time

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role='{self.role}')>"
