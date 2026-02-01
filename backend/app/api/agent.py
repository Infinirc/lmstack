"""
Agent Chat API

Provides SSE streaming endpoints for the MCP-based AI agent.
This enables Claude Code-style interaction where users can see
the agent's thinking process and tool executions in real-time.

Endpoints:
    POST /agent/chat - Stream agent chat with SSE
    POST /agent/chat/simple - Simple request-response chat
    GET /agent/tools - List available tools
    GET /agent/conversations - List user's Agent conversations
    GET /agent/conversations/{id} - Get conversation details
    DELETE /agent/conversations/{id} - Delete conversation
    DELETE /agent/conversation - Clear conversation history (legacy)
"""

import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.deps import require_viewer
from app.database import get_db
from app.models.conversation import Conversation, ConversationType, Message, MessageRole
from app.models.deployment import Deployment, DeploymentStatus
from app.models.user import User
from app.services.mcp import AgentService, EventType

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class LLMConfig(BaseModel):
    """LLM configuration for the agent."""

    provider: str = Field(
        default="system",
        description="LLM provider: 'system' (local deployment), 'openai', or 'custom'",
    )
    deployment_id: int | None = Field(
        default=None, description="Deployment ID when using 'system' provider"
    )
    api_key: str | None = Field(default=None, description="API key for external providers")
    base_url: str | None = Field(default=None, description="Base URL for custom provider")
    model: str | None = Field(default=None, description="Model name to use")


class AgentChatRequest(BaseModel):
    """Request body for agent chat."""

    message: str = Field(..., description="User message to send to the agent")
    llm_config: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    conversation_id: int | None = Field(
        default=None, description="Database conversation ID to continue"
    )


class AgentChatSimpleRequest(BaseModel):
    """Request body for simple (non-streaming) agent chat."""

    message: str = Field(..., description="User message to send to the agent")
    llm_config: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")


class AgentChatSimpleResponse(BaseModel):
    """Response for simple agent chat."""

    response: str = Field(..., description="Agent's response")
    conversation_history: list[dict] = Field(
        default_factory=list, description="Full conversation history"
    )


class ToolInfo(BaseModel):
    """Information about an available tool."""

    name: str
    description: str
    parameters: list[dict]


class ToolsResponse(BaseModel):
    """Response for listing tools."""

    tools: list[ToolInfo]
    total: int


class ConversationSummary(BaseModel):
    """Summary of a conversation for list view."""

    id: int
    title: str
    conversation_type: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationListResponse(BaseModel):
    """Response for listing conversations."""

    conversations: list[ConversationSummary]
    total: int


class MessageDetail(BaseModel):
    """Detail of a message in a conversation."""

    id: int
    role: str
    content: str
    thinking: str | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None
    step_type: str | None = None
    execution_time_ms: float | None = None
    created_at: datetime


class ConversationDetail(BaseModel):
    """Full conversation with messages."""

    id: int
    title: str
    conversation_type: str
    agent_config: dict | None = None
    created_at: datetime
    updated_at: datetime
    messages: list[MessageDetail]


# ============================================================================
# Active Agent Sessions
# ============================================================================

# Store active agent sessions (in production, use Redis or similar)
_active_sessions: dict[str, AgentService] = {}
_session_lock = asyncio.Lock()


async def get_or_create_agent(
    session_id: str,
    llm_config: LLMConfig,
    db: AsyncSession,
    api_token: str | None = None,
) -> AgentService:
    """Get existing agent session or create a new one."""
    async with _session_lock:
        if session_id in _active_sessions:
            return _active_sessions[session_id]

        # Resolve deployment if using system provider
        llm_base_url = None
        llm_api_key = llm_config.api_key
        llm_model = llm_config.model

        # Track if tool calling is supported
        supports_tool_calling = True

        if llm_config.provider == "system" and llm_config.deployment_id:
            # Look up the deployment with model info
            result = await db.execute(
                select(Deployment)
                .where(Deployment.id == llm_config.deployment_id)
                .options(selectinload(Deployment.worker), selectinload(Deployment.model))
            )
            deployment = result.scalar_one_or_none()

            if not deployment:
                raise HTTPException(status_code=404, detail="Deployment not found")
            if deployment.status != DeploymentStatus.RUNNING.value:
                raise HTTPException(status_code=400, detail="Deployment is not running")

            worker = deployment.worker
            # Extract IP from worker address (format: IP:Port)
            worker_ip = worker.address.split(":")[0]
            llm_base_url = f"http://{worker_ip}:{deployment.port}/v1"
            llm_api_key = "dummy"
            # Use the actual model_id from the LLMModel (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
            llm_model = deployment.model.model_id

            # Check if deployment has tool calling enabled
            extra_params = deployment.extra_params or {}
            supports_tool_calling = extra_params.get("enable-auto-tool-choice", False)

        elif llm_config.provider == "openai":
            llm_base_url = "https://api.openai.com/v1"
            llm_model = llm_config.model or "gpt-4o"

        elif llm_config.provider == "custom":
            llm_base_url = llm_config.base_url
            llm_model = llm_config.model or "default"

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider: {llm_config.provider}. "
                "Use 'system' with deployment_id, 'openai' with api_key, or 'custom' with base_url",
            )

        # Create agent with MCP configuration
        # The MCP server needs to call back to the LMStack API
        from app.config import get_settings

        settings = get_settings()
        mcp_api_url = f"http://localhost:{settings.port}/api"

        agent = AgentService(
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            mcp_api_url=mcp_api_url,
            mcp_api_token=api_token,
            supports_tool_calling=supports_tool_calling,
        )
        await agent.initialize()

        _active_sessions[session_id] = agent
        return agent


async def cleanup_session(session_id: str) -> None:
    """Cleanup an agent session."""
    async with _session_lock:
        agent = _active_sessions.pop(session_id, None)
        if agent:
            await agent.cleanup()


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/chat")
async def agent_chat_stream(
    request: AgentChatRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """
    Stream agent chat with Server-Sent Events (SSE).

    The agent will process the user's message and stream back events
    including thinking process, tool executions, and responses.

    Messages are persisted to the database for conversation continuity.

    Event types:
    - thinking: Agent is processing
    - planning: Agent is planning actions
    - message: Agent message to user
    - tool_start: Starting tool execution
    - tool_progress: Tool execution progress
    - tool_result: Tool execution completed
    - tool_error: Tool execution failed
    - done: Agent finished
    - error: Agent error
    - cancelled: User cancelled
    """
    from app.database import async_session_maker

    # Extract auth token from request to pass to MCP server
    auth_header = http_request.headers.get("Authorization", "")
    api_token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else None

    # Create or load conversation from database
    conversation_id = request.conversation_id
    if conversation_id:
        # Load existing conversation
        result = await db.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .where(Conversation.user_id == current_user.id)
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Create new conversation
        title = request.message[:50] + "..." if len(request.message) > 50 else request.message
        conversation = Conversation(
            user_id=current_user.id,
            title=title,
            conversation_type=ConversationType.AGENT.value,
            agent_config={
                "llm_provider": request.llm_config.provider,
                "llm_model": request.llm_config.model,
                "deployment_id": request.llm_config.deployment_id,
            },
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        conversation_id = conversation.id

    # Save user message to database
    user_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.USER.value,
        content=request.message,
    )
    db.add(user_message)
    await db.commit()

    # Use conversation_id as session key
    session_id = f"agent_conv_{conversation_id}"

    async def event_generator():
        agent = None
        accumulated_content = ""
        accumulated_thinking = ""
        tool_calls_list = []

        try:
            agent = await get_or_create_agent(session_id, request.llm_config, db, api_token)

            # First, send conversation_id to client
            init_event = json.dumps(
                {
                    "type": "init",
                    "data": {"conversation_id": conversation_id},
                }
            )
            yield f"data: {init_event}\n\n"

            async for event in agent.chat(request.message):
                # Format as SSE
                event_data = json.dumps(event.to_dict(), ensure_ascii=False)
                yield f"data: {event_data}\n\n"

                # Accumulate content for database
                if event.type == EventType.MESSAGE and event.content:
                    accumulated_content += event.content
                elif event.type == EventType.THINKING and event.content:
                    accumulated_thinking += event.content
                elif event.type == EventType.TOOL_START:
                    tool_calls_list.append(
                        {
                            "tool_name": event.data.get("tool_name") if event.data else None,
                            "arguments": event.data.get("arguments") if event.data else None,
                            "status": "running",
                        }
                    )
                elif event.type == EventType.TOOL_RESULT:
                    if tool_calls_list:
                        tool_calls_list[-1]["status"] = "completed"
                        tool_calls_list[-1]["result"] = (
                            event.data.get("result") if event.data else None
                        )
                        tool_calls_list[-1]["execution_time_ms"] = (
                            event.data.get("execution_time_ms") if event.data else None
                        )
                elif event.type == EventType.TOOL_ERROR:
                    if tool_calls_list:
                        tool_calls_list[-1]["status"] = "error"
                        tool_calls_list[-1]["error"] = (
                            event.data.get("error") if event.data else event.content
                        )

                # Check for disconnect
                if await http_request.is_disconnected():
                    agent.cancel()
                    break

                # If done or error, save to database
                if event.type in (EventType.DONE, EventType.ERROR, EventType.CANCELLED):
                    break

        except Exception as e:
            logger.exception(f"Agent chat error: {e}")
            error_event = json.dumps(
                {
                    "type": "error",
                    "content": str(e),
                }
            )
            yield f"data: {error_event}\n\n"
            accumulated_content = f"Error: {str(e)}"

        finally:
            # Save assistant message to database using a fresh session
            if accumulated_content or tool_calls_list:
                try:
                    async with async_session_maker() as save_db:
                        assistant_message = Message(
                            conversation_id=conversation_id,
                            role=MessageRole.ASSISTANT.value,
                            content=accumulated_content or "No response",
                            thinking=accumulated_thinking if accumulated_thinking else None,
                            tool_calls=tool_calls_list if tool_calls_list else None,
                        )
                        save_db.add(assistant_message)
                        await save_db.commit()
                except Exception as save_error:
                    logger.error(f"Failed to save assistant message: {save_error}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/simple", response_model=AgentChatSimpleResponse)
async def agent_chat_simple(
    request: AgentChatSimpleRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """
    Simple (non-streaming) agent chat.

    Sends a message to the agent and waits for the complete response.
    Useful for programmatic access or when streaming is not needed.
    """
    # Extract auth token from request to pass to MCP server
    auth_header = http_request.headers.get("Authorization", "")
    api_token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else None

    # Get MCP configuration
    from app.config import get_settings

    settings = get_settings()
    mcp_api_url = f"http://localhost:{settings.port}/api"

    # Create a temporary agent for this request
    llm_base_url = None
    llm_api_key = request.llm_config.api_key
    llm_model = request.llm_config.model

    if request.llm_config.provider == "system" and request.llm_config.deployment_id:
        result = await db.execute(
            select(Deployment)
            .where(Deployment.id == request.llm_config.deployment_id)
            .options(selectinload(Deployment.worker), selectinload(Deployment.model))
        )
        deployment = result.scalar_one_or_none()

        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")
        if deployment.status != DeploymentStatus.RUNNING.value:
            raise HTTPException(status_code=400, detail="Deployment is not running")

        worker = deployment.worker
        # Extract IP from worker address (format: IP:Port)
        worker_ip = worker.address.split(":")[0]
        llm_base_url = f"http://{worker_ip}:{deployment.port}/v1"
        llm_api_key = "dummy"
        # Use the actual model_id from the LLMModel (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        llm_model = deployment.model.model_id

    elif request.llm_config.provider == "openai":
        llm_base_url = "https://api.openai.com/v1"
        llm_model = request.llm_config.model or "gpt-4o"

    elif request.llm_config.provider == "custom":
        llm_base_url = request.llm_config.base_url
        llm_model = request.llm_config.model or "default"

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider: {request.llm_config.provider}",
        )

    try:
        async with AgentService(
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            mcp_api_url=mcp_api_url,
            mcp_api_token=api_token,
        ) as agent:
            response = await agent.chat_simple(request.message)
            history = agent.get_conversation_history()

            return AgentChatSimpleResponse(
                response=response,
                conversation_history=history,
            )

    except Exception as e:
        logger.exception(f"Agent chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools", response_model=ToolsResponse)
async def list_agent_tools(
    current_user: User = Depends(require_viewer),
):
    """
    List all tools available to the agent.

    Returns the list of MCP tools that the agent can use.
    """
    try:
        from app.services.mcp import MCPClient

        async with MCPClient() as client:
            tools = await client.list_tools()
            tool_infos = [
                ToolInfo(
                    name=t.name,
                    description=t.description,
                    parameters=[
                        {
                            "name": p.name,
                            "type": p.type,
                            "description": p.description,
                            "required": p.required,
                        }
                        for p in t.parameters
                    ],
                )
                for t in tools
            ]
            return ToolsResponse(tools=tool_infos, total=len(tool_infos))

    except Exception as e:
        logger.exception(f"Failed to list tools: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {e}")


@router.delete("/conversation")
async def clear_conversation(
    session_id: str = Query(..., description="Session ID to clear"),
    current_user: User = Depends(require_viewer),
):
    """
    Clear the conversation history for a session.

    This will reset the agent's memory for the specified session.
    """
    async with _session_lock:
        agent = _active_sessions.get(session_id)
        if agent:
            agent.reset()
            return {"success": True, "message": "Conversation cleared"}
        else:
            return {"success": False, "message": "Session not found"}


@router.post("/cancel")
async def cancel_operation(
    session_id: str = Query(..., description="Session ID to cancel"),
    current_user: User = Depends(require_viewer),
):
    """
    Cancel an ongoing agent operation.

    This will stop the current tool execution or LLM call.
    """
    async with _session_lock:
        agent = _active_sessions.get(session_id)
        if agent:
            agent.cancel()
            return {"success": True, "message": "Operation cancelled"}
        else:
            return {"success": False, "message": "Session not found"}


# ============================================================================
# Conversation Management Endpoints
# ============================================================================


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    conversation_type: str = Query(
        default="agent",
        description="Filter by conversation type: 'agent' or 'chat'",
    ),
    limit: int = Query(
        default=50, ge=1, le=100, description="Maximum number of conversations to return"
    ),
    offset: int = Query(default=0, ge=0, description="Number of conversations to skip"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """
    List user's conversations.

    Returns a paginated list of conversations with summary info.
    """
    # Count total
    count_query = (
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .where(Conversation.conversation_type == conversation_type)
    )
    count_result = await db.execute(count_query)
    total = len(count_result.all())

    # Get conversations with message count
    query = (
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .where(Conversation.conversation_type == conversation_type)
        .order_by(Conversation.updated_at.desc())
        .offset(offset)
        .limit(limit)
        .options(selectinload(Conversation.messages))
    )
    result = await db.execute(query)
    conversations = result.scalars().all()

    summaries = [
        ConversationSummary(
            id=conv.id,
            title=conv.title,
            conversation_type=conv.conversation_type,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=len(conv.messages),
        )
        for conv in conversations
    ]

    return ConversationListResponse(conversations=summaries, total=total)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """
    Get a conversation with all its messages.

    Returns the full conversation history including tool calls and thinking.
    """
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.user_id == current_user.id)
        .options(selectinload(Conversation.messages))
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = [
        MessageDetail(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            thinking=msg.thinking,
            tool_calls=msg.tool_calls,
            tool_call_id=msg.tool_call_id,
            step_type=msg.step_type,
            execution_time_ms=msg.execution_time_ms,
            created_at=msg.created_at,
        )
        for msg in conversation.messages
    ]

    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        conversation_type=conversation.conversation_type,
        agent_config=conversation.agent_config,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=messages,
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """
    Delete a conversation and all its messages.

    This will permanently remove the conversation history.
    """
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.user_id == current_user.id)
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Delete conversation (messages will be cascade deleted)
    await db.delete(conversation)
    await db.commit()

    # Also cleanup any active session
    session_id = f"agent_conv_{conversation_id}"
    await cleanup_session(session_id)

    return {"success": True, "message": "Conversation deleted"}


@router.patch("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: int,
    title: str = Query(..., description="New title for the conversation"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """
    Update a conversation's title.
    """
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.user_id == current_user.id)
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation.title = title
    await db.commit()

    return {"success": True, "message": "Conversation updated"}
