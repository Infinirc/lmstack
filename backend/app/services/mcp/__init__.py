"""
MCP (Model Context Protocol) Module

This module provides:
1. MCPClient - A Python client for communicating with MCP servers
2. AgentService - An AI agent that uses MCP for system interaction

Usage:
    # Direct MCP client usage
    from app.services.mcp import MCPClient

    async with MCPClient() as client:
        result = await client.call_tool("get_hardware_info", {"worker_id": 1})

    # Agent service usage
    from app.services.mcp import AgentService

    async with AgentService(...) as agent:
        async for event in agent.chat("Deploy Qwen-7B on Worker 1"):
            print(event)
"""

from .agent import (
    AGENT_SYSTEM_PROMPT,
    AgentEvent,
    AgentService,
    ConversationMessage,
    EventType,
    create_agent,
)
from .client import MCPClient, MCPClientPool
from .types import (
    MCPConnectionError,
    MCPError,
    MCPResource,
    MCPTimeoutError,
    MCPTool,
    MCPToolError,
    ToolCallResult,
    ToolCallStatus,
)

__all__ = [
    # Client
    "MCPClient",
    "MCPClientPool",
    # Types
    "MCPError",
    "MCPConnectionError",
    "MCPToolError",
    "MCPTimeoutError",
    "ToolCallResult",
    "ToolCallStatus",
    "MCPResource",
    "MCPTool",
    # Agent
    "AgentService",
    "AgentEvent",
    "EventType",
    "ConversationMessage",
    "create_agent",
    "AGENT_SYSTEM_PROMPT",
]
