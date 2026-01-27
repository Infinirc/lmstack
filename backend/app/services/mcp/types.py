"""
MCP Type Definitions

This module contains all type definitions for the MCP client.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MCPError(Exception):
    """Base exception for MCP-related errors."""

    pass


class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""

    pass


class MCPToolError(MCPError):
    """Raised when a tool execution fails."""

    def __init__(self, tool_name: str, message: str, details: dict | None = None):
        self.tool_name = tool_name
        self.details = details or {}
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class MCPTimeoutError(MCPError):
    """Raised when an MCP operation times out."""

    pass


class MCPProtocolError(MCPError):
    """Raised when there's a protocol-level error."""

    pass


class ToolCallStatus(str, Enum):
    """Status of a tool call."""

    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ToolCallResult:
    """Result of a tool call."""

    tool_name: str
    status: ToolCallStatus
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == ToolCallStatus.SUCCESS

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class MCPResource:
    """Represents an MCP resource."""

    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"

    @classmethod
    def from_dict(cls, data: dict) -> "MCPResource":
        return cls(
            uri=data.get("uri", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            mime_type=data.get("mimeType", "text/plain"),
        )


@dataclass
class MCPToolParameter:
    """Represents a parameter for an MCP tool."""

    name: str
    type: str
    description: str
    required: bool = False
    enum: list[str] | None = None
    default: Any = None


@dataclass
class MCPTool:
    """Represents an MCP tool."""

    name: str
    description: str
    parameters: list[MCPToolParameter] = field(default_factory=list)
    requires_confirmation: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "MCPTool":
        params = []
        input_schema = data.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        for param_name, param_info in properties.items():
            params.append(
                MCPToolParameter(
                    name=param_name,
                    type=param_info.get("type", "string"),
                    description=param_info.get("description", ""),
                    required=param_name in required,
                    enum=param_info.get("enum"),
                    default=param_info.get("default"),
                )
            )

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            parameters=params,
        )


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 request."""

    method: str
    params: dict | None = None
    id: int | str | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict:
        result = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.params is not None:
            result["params"] = self.params
        if self.id is not None:
            result["id"] = self.id
        return result


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 response."""

    id: int | str | None
    result: Any = None
    error: dict | None = None
    jsonrpc: str = "2.0"

    @classmethod
    def from_dict(cls, data: dict) -> "JSONRPCResponse":
        return cls(
            id=data.get("id"),
            result=data.get("result"),
            error=data.get("error"),
            jsonrpc=data.get("jsonrpc", "2.0"),
        )

    @property
    def success(self) -> bool:
        return self.error is None
