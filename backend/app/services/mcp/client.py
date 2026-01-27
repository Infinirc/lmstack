"""
MCP Client Implementation

A Python client for communicating with MCP (Model Context Protocol) servers.
Implements JSON-RPC 2.0 over stdio transport.

This client can connect to any MCP-compliant server and execute tools,
read resources, and manage the connection lifecycle.

Example:
    async with MCPClient() as client:
        # List available tools
        tools = await client.list_tools()

        # Call a tool
        result = await client.call_tool("deploy_model", {
            "model_id": 1,
            "worker_id": 1,
        })
"""

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

from .types import (
    JSONRPCRequest,
    JSONRPCResponse,
    MCPConnectionError,
    MCPProtocolError,
    MCPResource,
    MCPTimeoutError,
    MCPTool,
    ToolCallResult,
    ToolCallStatus,
)

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Async MCP Client for communicating with MCP servers.

    This client manages a subprocess running the MCP server and communicates
    with it using JSON-RPC 2.0 over stdin/stdout.

    Attributes:
        server_path: Path to the MCP server executable
        env: Environment variables to pass to the server process
        timeout: Default timeout for operations in seconds
    """

    # Default path to the LMStack MCP server
    # Path: backend/app/services/mcp/client.py -> 5 parents up = lmstack/ -> mcp-server/dist/index.js
    DEFAULT_SERVER_PATH = (
        Path(__file__).parent.parent.parent.parent.parent / "mcp-server" / "dist" / "index.js"
    )

    def __init__(
        self,
        server_path: str | Path | None = None,
        api_url: str | None = None,
        api_token: str | None = None,
        timeout: float = 60.0,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
    ):
        """
        Initialize the MCP client.

        Args:
            server_path: Path to the MCP server. Defaults to LMStack MCP server.
            api_url: LMStack API URL. Defaults to environment variable.
            api_token: LMStack API token. Defaults to environment variable.
            timeout: Default timeout for operations in seconds.
            llm_base_url: LLM base URL to pass to MCP server for auto-tuning.
            llm_api_key: LLM API key to pass to MCP server for auto-tuning.
            llm_model: LLM model name to pass to MCP server for auto-tuning.
        """
        self.server_path = Path(server_path) if server_path else self.DEFAULT_SERVER_PATH
        self.timeout = timeout

        # Build environment for the MCP server
        self.env = os.environ.copy()
        if api_url:
            self.env["LMSTACK_API_URL"] = api_url
        if api_token:
            self.env["LMSTACK_API_TOKEN"] = api_token

        # Pass LLM configuration for auto-tuning agent
        if llm_base_url:
            self.env["AGENT_LLM_BASE_URL"] = llm_base_url
        if llm_api_key:
            self.env["AGENT_LLM_API_KEY"] = llm_api_key
        if llm_model:
            self.env["AGENT_LLM_MODEL"] = llm_model

        # Connection state
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._read_task: asyncio.Task | None = None
        self._connected = False
        self._lock = asyncio.Lock()

        # Cached data
        self._tools: list[MCPTool] | None = None
        self._resources: list[MCPResource] | None = None

    @property
    def connected(self) -> bool:
        """Check if the client is connected to the MCP server."""
        return self._connected and self._process is not None

    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """
        Connect to the MCP server.

        Starts the MCP server subprocess and initializes the connection.

        Raises:
            MCPConnectionError: If connection fails.
        """
        if self._connected:
            return

        async with self._lock:
            if self._connected:
                return

            try:
                # Validate server path
                if not self.server_path.exists():
                    raise MCPConnectionError(
                        f"MCP server not found at {self.server_path}. "
                        "Please run 'npm run build' in the mcp-server directory."
                    )

                logger.info(f"Starting MCP server: {self.server_path}")

                # Start the MCP server process
                self._process = await asyncio.create_subprocess_exec(
                    "node",
                    str(self.server_path),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=self.env,
                )

                # Start the reader task
                self._read_task = asyncio.create_task(self._read_responses())

                # Initialize the connection
                await self._initialize()

                self._connected = True
                logger.info("MCP client connected successfully")

            except Exception as e:
                await self._cleanup()
                raise MCPConnectionError(f"Failed to connect to MCP server: {e}") from e

    async def disconnect(self) -> None:
        """
        Disconnect from the MCP server.

        Cleanly shuts down the subprocess and cleans up resources.
        """
        async with self._lock:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self._connected = False

        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        # Cancel read task
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None

        # Terminate process
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception as e:
                logger.warning(f"Error terminating MCP server: {e}")
            self._process = None

        # Clear cache
        self._tools = None
        self._resources = None

    async def _initialize(self) -> None:
        """Initialize the MCP connection with handshake."""
        # Send initialize request
        response = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "lmstack-python-client",
                    "version": "1.0.0",
                },
            },
        )

        if not response.success:
            raise MCPProtocolError(f"Initialize failed: {response.error}")

        # Send initialized notification
        await self._send_notification("notifications/initialized", {})

        logger.debug(f"MCP initialized: {response.result}")

    async def _send_request(self, method: str, params: dict | None = None) -> JSONRPCResponse:
        """
        Send a JSON-RPC request and wait for the response.

        Args:
            method: The method name to call.
            params: Optional parameters for the method.

        Returns:
            The JSON-RPC response.

        Raises:
            MCPTimeoutError: If the request times out.
            MCPConnectionError: If not connected.
        """
        if not self._process or not self._process.stdin:
            raise MCPConnectionError("Not connected to MCP server")

        self._request_id += 1
        request_id = self._request_id

        request = JSONRPCRequest(method=method, params=params, id=request_id)

        # Create future for response
        future: asyncio.Future[JSONRPCResponse] = asyncio.Future()
        self._pending_requests[request_id] = future

        try:
            # Send request
            message = json.dumps(request.to_dict()) + "\n"
            self._process.stdin.write(message.encode())
            await self._process.stdin.drain()

            logger.debug(f"Sent request: {method} (id={request_id})")

            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=self.timeout)
            return response

        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise MCPTimeoutError(f"Request timed out: {method}")
        except Exception:
            self._pending_requests.pop(request_id, None)
            raise

    async def _send_notification(self, method: str, params: dict | None = None) -> None:
        """
        Send a JSON-RPC notification (no response expected).

        Args:
            method: The method name.
            params: Optional parameters.
        """
        if not self._process or not self._process.stdin:
            raise MCPConnectionError("Not connected to MCP server")

        request = JSONRPCRequest(method=method, params=params, id=None)
        message = json.dumps(request.to_dict()) + "\n"
        self._process.stdin.write(message.encode())
        await self._process.stdin.drain()

    async def _read_responses(self) -> None:
        """Background task to read responses from the MCP server."""
        if not self._process or not self._process.stdout:
            return

        buffer = ""
        try:
            while True:
                chunk = await self._process.stdout.read(4096)
                if not chunk:
                    break

                buffer += chunk.decode()

                # Process complete lines
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        response = JSONRPCResponse.from_dict(data)

                        # Match response to pending request
                        if response.id is not None:
                            future = self._pending_requests.pop(response.id, None)
                            if future and not future.done():
                                future.set_result(response)
                            else:
                                logger.warning(f"No pending request for id: {response.id}")
                        else:
                            # This is a notification from the server
                            logger.debug(f"Received notification: {data}")

                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON from server: {e}")
                    except Exception as e:
                        logger.error(f"Error processing response: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error reading from MCP server: {e}")

    async def list_tools(self, use_cache: bool = True) -> list[MCPTool]:
        """
        List all available tools from the MCP server.

        Args:
            use_cache: Whether to use cached tool list.

        Returns:
            List of available tools.
        """
        if use_cache and self._tools is not None:
            return self._tools

        response = await self._send_request("tools/list", {})
        if not response.success:
            raise MCPProtocolError(f"Failed to list tools: {response.error}")

        tools = [MCPTool.from_dict(t) for t in response.result.get("tools", [])]
        self._tools = tools
        return tools

    async def list_resources(self, use_cache: bool = True) -> list[MCPResource]:
        """
        List all available resources from the MCP server.

        Args:
            use_cache: Whether to use cached resource list.

        Returns:
            List of available resources.
        """
        if use_cache and self._resources is not None:
            return self._resources

        response = await self._send_request("resources/list", {})
        if not response.success:
            raise MCPProtocolError(f"Failed to list resources: {response.error}")

        resources = [MCPResource.from_dict(r) for r in response.result.get("resources", [])]
        self._resources = resources
        return resources

    async def read_resource(self, uri: str) -> str:
        """
        Read the content of a resource.

        Args:
            uri: The resource URI (e.g., "lmstack://workers").

        Returns:
            The resource content as a string.
        """
        response = await self._send_request("resources/read", {"uri": uri})
        if not response.success:
            raise MCPProtocolError(f"Failed to read resource {uri}: {response.error}")

        contents = response.result.get("contents", [])
        if contents:
            return contents[0].get("text", "")
        return ""

    async def call_tool(self, name: str, arguments: dict | None = None) -> ToolCallResult:
        """
        Call a tool on the MCP server.

        Args:
            name: The tool name.
            arguments: Optional arguments for the tool.

        Returns:
            The tool call result.

        Raises:
            MCPToolError: If the tool execution fails.
        """
        start_time = time.time()

        try:
            response = await self._send_request(
                "tools/call",
                {
                    "name": name,
                    "arguments": arguments or {},
                },
            )

            execution_time = (time.time() - start_time) * 1000

            if not response.success:
                error_msg = (
                    response.error.get("message", "Unknown error")
                    if response.error
                    else "Unknown error"
                )
                return ToolCallResult(
                    tool_name=name,
                    status=ToolCallStatus.ERROR,
                    error=error_msg,
                    execution_time_ms=execution_time,
                )

            # Parse the result
            content = response.result.get("content", [])
            result_text = ""
            for item in content:
                if item.get("type") == "text":
                    result_text += item.get("text", "")

            # Check if the result indicates an error
            is_error = response.result.get("isError", False)
            if is_error:
                return ToolCallResult(
                    tool_name=name,
                    status=ToolCallStatus.ERROR,
                    error=result_text,
                    execution_time_ms=execution_time,
                )

            return ToolCallResult(
                tool_name=name,
                status=ToolCallStatus.SUCCESS,
                result=result_text,
                execution_time_ms=execution_time,
            )

        except MCPTimeoutError:
            return ToolCallResult(
                tool_name=name,
                status=ToolCallStatus.ERROR,
                error="Tool execution timed out",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return ToolCallResult(
                tool_name=name,
                status=ToolCallStatus.ERROR,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def call_tool_streaming(
        self, name: str, arguments: dict | None = None
    ) -> AsyncIterator[dict]:
        """
        Call a tool with streaming progress updates.

        This method yields progress updates while the tool is executing,
        useful for long-running operations like deployments or benchmarks.

        Args:
            name: The tool name.
            arguments: Optional arguments for the tool.

        Yields:
            Progress updates with status information.
        """
        start_time = time.time()

        yield {
            "type": "status",
            "status": "executing",
            "tool_name": name,
            "arguments": arguments,
        }

        try:
            result = await self.call_tool(name, arguments)

            if result.success:
                yield {
                    "type": "result",
                    "status": "success",
                    "tool_name": name,
                    "result": result.result,
                    "execution_time_ms": result.execution_time_ms,
                }
            else:
                yield {
                    "type": "result",
                    "status": "error",
                    "tool_name": name,
                    "error": result.error,
                    "execution_time_ms": result.execution_time_ms,
                }

        except Exception as e:
            yield {
                "type": "result",
                "status": "error",
                "tool_name": name,
                "error": str(e),
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

    def get_tool_by_name(self, name: str) -> MCPTool | None:
        """
        Get a tool by its name from the cached tool list.

        Args:
            name: The tool name.

        Returns:
            The tool if found, None otherwise.
        """
        if self._tools is None:
            return None
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None


class MCPClientPool:
    """
    Pool of MCP clients for concurrent access.

    Useful when you need multiple simultaneous connections to the MCP server.
    """

    def __init__(self, size: int = 3, **client_kwargs):
        """
        Initialize the client pool.

        Args:
            size: Number of clients in the pool.
            **client_kwargs: Arguments to pass to MCPClient constructor.
        """
        self.size = size
        self.client_kwargs = client_kwargs
        self._clients: list[MCPClient] = []
        self._semaphore = asyncio.Semaphore(size)
        self._index = 0
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "MCPClientPool":
        """Initialize all clients."""
        for _ in range(self.size):
            client = MCPClient(**self.client_kwargs)
            await client.connect()
            self._clients.append(client)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Disconnect all clients."""
        for client in self._clients:
            await client.disconnect()
        self._clients.clear()

    async def acquire(self) -> MCPClient:
        """
        Acquire a client from the pool.

        Returns:
            An available MCP client.
        """
        await self._semaphore.acquire()
        async with self._lock:
            client = self._clients[self._index % len(self._clients)]
            self._index += 1
            return client

    def release(self) -> None:
        """Release a client back to the pool."""
        self._semaphore.release()
