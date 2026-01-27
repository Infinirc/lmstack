"""
MCP-based Agent Service

This module implements an AI agent that uses MCP to interact with the LMStack platform.
The agent can autonomously execute tasks like model deployment, benchmarking, and
configuration optimization through natural language interaction.

The agent follows a Claude Code-style interaction pattern:
1. User provides a natural language request
2. Agent analyzes the request and plans actions
3. Agent executes actions via MCP tools, streaming progress
4. Agent provides results and recommendations

Example:
    async with AgentService() as agent:
        async for event in agent.chat("Deploy Qwen-7B on Worker 1"):
            print(event)
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .client import MCPClient
from .types import MCPTool

logger = logging.getLogger(__name__)


# ============================================================================
# Type Definitions
# ============================================================================


class EventType(str, Enum):
    """Types of events streamed from the agent."""

    # Agent status events
    THINKING = "thinking"  # Agent is processing
    PLANNING = "planning"  # Agent is planning actions
    REASONING = "reasoning"  # Model's reasoning/thinking process
    MESSAGE = "message"  # Agent message to user

    # Tool execution events
    TOOL_START = "tool_start"  # Starting tool execution
    TOOL_PROGRESS = "tool_progress"  # Tool execution progress
    TOOL_RESULT = "tool_result"  # Tool execution completed
    TOOL_ERROR = "tool_error"  # Tool execution failed

    # UI events
    PAGE_REFERENCE = "page_reference"  # Reference to a LMStack page
    ACTION_SUGGESTIONS = "action_suggestions"  # Suggested actions user can click

    # Control events
    DONE = "done"  # Agent finished
    ERROR = "error"  # Agent error
    CANCELLED = "cancelled"  # User cancelled


@dataclass
class AgentEvent:
    """An event emitted by the agent during execution."""

    type: EventType
    content: str | None = None
    data: dict | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "content": self.content,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConversationMessage:
    """A message in the conversation history."""

    role: str  # "user", "assistant", "tool"
    content: str
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        result = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    def to_api_format(self) -> dict:
        """Convert to OpenAI API message format."""
        result = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


# ============================================================================
# System Prompt
# ============================================================================


AGENT_SYSTEM_PROMPT = """You are LMStack AI Assistant, an expert in deploying and optimizing Large Language Model inference services.

## Your Capabilities

You can help users with:
1. **Model Deployment**: Deploy LLM models to GPU workers
2. **Performance Optimization**: Find optimal configurations for throughput or latency
3. **System Management**: Monitor workers, containers, and deployments
4. **Troubleshooting**: Diagnose and fix deployment issues

## Available Tools

You have access to MCP tools that let you:
- Query hardware information (GPUs, memory, workers)
- Deploy and manage models
- Run performance benchmarks
- Query the performance knowledge base
- Manage containers and storage

## Workflow Guidelines

### For Deployment Requests:
1. First check available workers with `list_workers`
2. Verify the model exists with `list_models`
3. Check GPU availability and memory
4. Deploy with appropriate configuration
5. Wait for deployment to complete
6. Verify the deployment is running

### For Performance Tuning:
1. Analyze hardware with `get_gpu_status`
2. Query knowledge base for similar configurations
3. Consider optimization target (throughput vs latency)
4. Test configurations with benchmarks
5. Compare results and recommend best config

### Engine Selection Guidelines:
- **vLLM**: Best for high throughput, supports tensor parallelism
- **SGLang**: Good for multi-turn conversations, prefix caching
- **Ollama**: Simple deployment, good for smaller models

### Quantization Notes:
- FP16: Full precision, requires more VRAM
- FP8: Half the memory, requires Hopper+ GPUs (H100, H200)
- AWQ/GPTQ: 4-bit quantization, needs pre-quantized models

## Response Style

- **IMPORTANT: Always respond in the EXACT same language and script as the user's message.**
  - If the user writes in Traditional Chinese (繁體中文), you MUST respond in Traditional Chinese (繁體中文).
  - If the user writes in Simplified Chinese (简体中文), respond in Simplified Chinese.
  - If the user writes in English, respond in English.
  - Never mix scripts - if the user uses 「」for quotes, use 「」not "".
- Be concise but informative
- Show your reasoning when making decisions
- Report progress during long operations
- Provide clear recommendations with rationale
- Use markdown formatting for readability

## Important Notes

- Always verify operations completed successfully
- Check logs if deployments fail
- Consider VRAM constraints when selecting configurations
- Report any errors or issues clearly
"""


# ============================================================================
# Agent Service
# ============================================================================


class AgentService:
    """
    AI Agent service that uses MCP for system interaction.

    This agent can:
    - Process natural language requests
    - Execute operations via MCP tools
    - Stream progress updates to the client
    - Maintain conversation context
    """

    def __init__(
        self,
        llm_client: Any = None,
        llm_model: str = "gpt-4o",
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        mcp_api_url: str | None = None,
        mcp_api_token: str | None = None,
        max_iterations: int = 20,
    ):
        """
        Initialize the Agent Service.

        Args:
            llm_client: Pre-configured LLM client (OpenAI compatible).
            llm_model: Model to use for the agent.
            llm_base_url: Base URL for LLM API.
            llm_api_key: API key for LLM.
            mcp_api_url: LMStack API URL for MCP server.
            mcp_api_token: LMStack API token for MCP server.
            max_iterations: Maximum tool call iterations.
        """
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.max_iterations = max_iterations

        # MCP configuration
        self.mcp_api_url = mcp_api_url
        self.mcp_api_token = mcp_api_token

        # State
        self.mcp_client: MCPClient | None = None
        self.conversation: list[ConversationMessage] = []
        self.tools: list[MCPTool] = []
        self._cancelled = False
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "AgentService":
        """Initialize the agent."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup the agent."""
        await self.cleanup()

    async def initialize(self) -> None:
        """Initialize the agent and MCP connection."""
        # Initialize MCP client with LLM config for auto-tuning
        self.mcp_client = MCPClient(
            api_url=self.mcp_api_url,
            api_token=self.mcp_api_token,
            llm_base_url=self.llm_base_url,
            llm_api_key=self.llm_api_key,
            llm_model=self.llm_model,
        )
        await self.mcp_client.connect()

        # Load available tools
        self.tools = await self.mcp_client.list_tools()
        logger.info(f"Agent initialized with {len(self.tools)} tools")

        # Initialize LLM client if not provided
        if self.llm_client is None:
            from openai import AsyncOpenAI

            self.llm_client = AsyncOpenAI(
                base_url=self.llm_base_url,
                api_key=self.llm_api_key or "dummy",
            )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.mcp_client:
            await self.mcp_client.disconnect()
            self.mcp_client = None

    def cancel(self) -> None:
        """Cancel the current operation."""
        self._cancelled = True

    def reset(self) -> None:
        """Reset the agent state for a new conversation."""
        self.conversation = []
        self._cancelled = False

    def _build_tools_schema(self) -> list[dict]:
        """Build OpenAI-compatible tools schema from MCP tools."""
        tools = []
        for tool in self.tools:
            # Build parameters schema
            properties = {}
            required = []
            for param in tool.parameters:
                prop = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.enum:
                    prop["enum"] = param.enum
                properties[param.name] = prop
                if param.required:
                    required.append(param.name)

            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return tools

    def _get_page_reference(self, tool_name: str, tool_args: dict) -> dict | None:
        """Get page reference for a tool execution."""
        # Map tools to their relevant pages
        page_mappings = {
            "list_workers": {
                "path": "/workers",
                "title": "Workers",
                "icon": "cluster",
                "description": "查看所有 GPU Worker 節點",
            },
            "get_gpu_status": {
                "path": "/workers",
                "title": "Workers",
                "icon": "cluster",
                "description": "查看 GPU 狀態與記憶體使用",
            },
            "list_deployments": {
                "path": "/deployments",
                "title": "Deployments",
                "icon": "rocket",
                "description": "查看所有模型部署",
            },
            "deploy_model": {
                "path": "/deployments",
                "title": "Deployments",
                "icon": "rocket",
                "description": "查看部署狀態",
            },
            "list_models": {
                "path": "/models",
                "title": "Models",
                "icon": "database",
                "description": "查看所有可用模型",
            },
            "add_model": {
                "path": "/models",
                "title": "Models",
                "icon": "database",
                "description": "查看新增的模型",
            },
            "list_containers": {
                "path": "/containers",
                "title": "Containers",
                "icon": "container",
                "description": "查看 Docker 容器",
            },
            "get_system_status": {
                "path": "/",
                "title": "Dashboard",
                "icon": "dashboard",
                "description": "查看系統總覽",
            },
            "list_api_keys": {
                "path": "/api-keys",
                "title": "API Keys",
                "icon": "key",
                "description": "管理 API 金鑰",
            },
        }
        return page_mappings.get(tool_name)

    def _get_action_suggestions(
        self, tool_name: str, tool_result: str, tool_args: dict
    ) -> list[dict] | None:
        """Generate action suggestions based on tool results."""
        import re

        suggestions = []

        # After listing models, suggest deployment actions and Auto-Tune
        if tool_name == "list_models":
            # Parse model IDs and names from the result
            # Look for patterns like "**ModelName** (ID: N)"
            model_matches = re.findall(r"\*\*([^*]+)\*\* \(ID: (\d+)\)", tool_result)
            for model_name, model_id in model_matches[:2]:  # Limit to 2 deploy suggestions
                suggestions.append(
                    {
                        "label": f"部署 {model_name}",
                        "message": f"部署模型 {model_name} (ID: {model_id}) 到 worker 1，使用 vllm",
                        "icon": "rocket",
                        "type": "primary",
                    }
                )

            # Add Auto-Tune suggestion
            if model_matches:
                suggestions.append(
                    {
                        "label": "Auto-Tune 最佳配置",
                        "message": "為這些模型找出最佳部署配置，請先查詢 GPU 狀態，然後搜尋網路上的最佳配置建議",
                        "icon": "experiment",
                        "type": "default",
                    }
                )

        # After listing deployments, suggest stop actions for running ones
        elif tool_name == "list_deployments":
            # Look for running deployments
            running_matches = re.findall(
                r"## ([^\[]+) \[running\][\s\S]*?- \*\*ID:\*\* (\d+)", tool_result
            )
            for dep_name, dep_id in running_matches[:2]:
                suggestions.append(
                    {
                        "label": f"停止 {dep_name.strip()}",
                        "message": f"停止部署 ID {dep_id}",
                        "icon": "stop",
                        "type": "danger",
                    }
                )

        # After listing workers, suggest checking GPU status
        elif tool_name == "list_workers":
            suggestions.append(
                {
                    "label": "查看 GPU 詳細狀態",
                    "message": "顯示所有 GPU 的詳細狀態和記憶體使用情況",
                    "icon": "monitor",
                    "type": "default",
                }
            )

        # After get_gpu_status, suggest searching for optimal configs
        elif tool_name == "get_gpu_status":
            # Extract GPU model from result
            gpu_model_match = re.search(
                r"GPU \d+: (NVIDIA[^,\n]+|RTX[^,\n]+|A\d+[^,\n]+|H\d+[^,\n]+)", tool_result
            )
            if gpu_model_match:
                gpu_model = gpu_model_match.group(1).strip()
                suggestions.append(
                    {
                        "label": f"搜尋 {gpu_model} 最佳配置",
                        "message": f"搜尋網路上 {gpu_model} 部署 LLM 的最佳配置和效能基準",
                        "icon": "search",
                        "type": "primary",
                    }
                )
            suggestions.append(
                {
                    "label": "列出可部署的模型",
                    "message": "列出所有可以部署的模型",
                    "icon": "unordered-list",
                    "type": "default",
                }
            )

        # After web search, suggest deploying with found config
        elif tool_name == "web_search" or tool_name == "search_llm_config":
            suggestions.append(
                {
                    "label": "使用此配置部署",
                    "message": "根據搜尋結果中的最佳配置部署模型",
                    "icon": "rocket",
                    "type": "primary",
                }
            )
            suggestions.append(
                {
                    "label": "列出現有模型",
                    "message": "列出所有已註冊的模型",
                    "icon": "unordered-list",
                    "type": "default",
                }
            )

        # After deploy_model, suggest checking status
        elif tool_name == "deploy_model":
            suggestions.append(
                {
                    "label": "查看部署狀態",
                    "message": "列出所有部署狀態",
                    "icon": "check-circle",
                    "type": "primary",
                }
            )

        return suggestions if suggestions else None

    def _build_messages(self, user_message: str) -> list[dict]:
        """Build messages array for LLM API call."""
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

        # Add conversation history
        for msg in self.conversation:
            messages.append(msg.to_api_format())

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    async def chat(
        self,
        message: str,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """
        Process a user message and stream agent responses.

        Args:
            message: The user's message.
            on_event: Optional callback for each event.

        Yields:
            AgentEvent objects representing the agent's progress.
        """
        self._cancelled = False

        # Add user message to conversation
        user_msg = ConversationMessage(role="user", content=message)
        self.conversation.append(user_msg)

        try:
            # Initial thinking event
            yield AgentEvent(type=EventType.THINKING, content="Analyzing your request...")

            iteration = 0
            while iteration < self.max_iterations:
                if self._cancelled:
                    yield AgentEvent(type=EventType.CANCELLED, content="Operation cancelled")
                    return

                iteration += 1
                logger.debug(f"Agent iteration {iteration}")

                # Build messages and call LLM
                messages = self._build_messages(message if iteration == 1 else "")
                if iteration > 1:
                    # For continuation, don't add the user message again
                    messages = messages[:-1]

                tools_schema = self._build_tools_schema()

                try:
                    # Use streaming for better UX
                    stream = await self.llm_client.chat.completions.create(
                        model=self.llm_model,
                        messages=messages,
                        tools=tools_schema if tools_schema else None,
                        tool_choice="auto" if tools_schema else None,
                        stream=True,
                    )

                    # Accumulate streaming response
                    accumulated_content = ""
                    accumulated_reasoning = ""
                    accumulated_tool_calls: dict[int, dict] = {}
                    finish_reason = None

                    async for chunk in stream:
                        if self._cancelled:
                            yield AgentEvent(
                                type=EventType.CANCELLED, content="Operation cancelled"
                            )
                            return

                        delta = chunk.choices[0].delta if chunk.choices else None
                        if not delta:
                            continue

                        # Handle reasoning/thinking content (e.g., DeepSeek-R1)
                        reasoning_content = getattr(delta, "reasoning_content", None)
                        if reasoning_content:
                            accumulated_reasoning += reasoning_content
                            yield AgentEvent(
                                type=EventType.REASONING,
                                content=reasoning_content,
                            )

                        # Handle content streaming
                        if delta.content:
                            accumulated_content += delta.content
                            # Emit streaming content
                            yield AgentEvent(
                                type=EventType.MESSAGE,
                                content=delta.content,
                            )

                        # Handle tool calls (accumulate deltas)
                        if delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index
                                if idx not in accumulated_tool_calls:
                                    accumulated_tool_calls[idx] = {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                if tc_delta.id:
                                    accumulated_tool_calls[idx]["id"] = tc_delta.id
                                if tc_delta.function:
                                    if tc_delta.function.name:
                                        accumulated_tool_calls[idx]["function"][
                                            "name"
                                        ] = tc_delta.function.name
                                    if tc_delta.function.arguments:
                                        accumulated_tool_calls[idx]["function"][
                                            "arguments"
                                        ] += tc_delta.function.arguments

                        # Track finish reason
                        if chunk.choices and chunk.choices[0].finish_reason:
                            finish_reason = chunk.choices[0].finish_reason

                except Exception as e:
                    logger.error(f"LLM API error: {e}")
                    yield AgentEvent(
                        type=EventType.ERROR,
                        content=f"Failed to communicate with LLM: {str(e)}",
                    )
                    return

                # Convert accumulated tool calls to list
                tool_calls_list = [
                    accumulated_tool_calls[i] for i in sorted(accumulated_tool_calls.keys())
                ]

                # Check for tool calls
                if tool_calls_list:
                    # Add assistant message with tool calls to conversation
                    self.conversation.append(
                        ConversationMessage(
                            role="assistant",
                            content=accumulated_content,
                            tool_calls=tool_calls_list,
                        )
                    )

                    # Content was already streamed above, no need to yield again

                    # Execute tool calls
                    for tool_call in tool_calls_list:
                        if self._cancelled:
                            yield AgentEvent(
                                type=EventType.CANCELLED, content="Operation cancelled"
                            )
                            return

                        tool_name = tool_call["function"]["name"]
                        try:
                            tool_args = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError:
                            tool_args = {}

                        # Emit tool start event
                        yield AgentEvent(
                            type=EventType.TOOL_START,
                            content=f"Executing {tool_name}",
                            data={
                                "tool_name": tool_name,
                                "arguments": tool_args,
                            },
                        )

                        # Execute the tool via MCP
                        result = await self.mcp_client.call_tool(tool_name, tool_args)

                        # Emit tool result event
                        if result.success:
                            yield AgentEvent(
                                type=EventType.TOOL_RESULT,
                                content=f"Completed {tool_name}",
                                data={
                                    "tool_name": tool_name,
                                    "result": result.result,
                                    "execution_time_ms": result.execution_time_ms,
                                },
                            )

                            # Emit page reference for relevant tools
                            page_ref = self._get_page_reference(tool_name, tool_args)
                            if page_ref:
                                yield AgentEvent(
                                    type=EventType.PAGE_REFERENCE,
                                    data=page_ref,
                                )

                            # Emit action suggestions based on tool results
                            suggestions = self._get_action_suggestions(
                                tool_name, result.result, tool_args
                            )
                            if suggestions:
                                yield AgentEvent(
                                    type=EventType.ACTION_SUGGESTIONS,
                                    data={"suggestions": suggestions},
                                )
                        else:
                            yield AgentEvent(
                                type=EventType.TOOL_ERROR,
                                content=f"Failed {tool_name}: {result.error}",
                                data={
                                    "tool_name": tool_name,
                                    "error": result.error,
                                },
                            )

                        # Add tool result to conversation
                        self.conversation.append(
                            ConversationMessage(
                                role="tool",
                                content=(
                                    result.result if result.success else f"Error: {result.error}"
                                ),
                                tool_call_id=tool_call["id"],
                            )
                        )

                    # Continue to next iteration to process tool results
                    continue

                else:
                    # No tool calls, this is the final response
                    # Content was already streamed above
                    self.conversation.append(
                        ConversationMessage(
                            role="assistant",
                            content=accumulated_content,
                        )
                    )

                    # Check if done
                    if finish_reason == "stop":
                        yield AgentEvent(type=EventType.DONE)
                        return

            # Max iterations reached
            yield AgentEvent(
                type=EventType.ERROR,
                content="Maximum iterations reached. Please try a simpler request.",
            )

        except Exception as e:
            logger.exception(f"Agent error: {e}")
            yield AgentEvent(
                type=EventType.ERROR,
                content=f"An error occurred: {str(e)}",
            )

    async def chat_simple(self, message: str) -> str:
        """
        Simple chat interface that returns the final response.

        Args:
            message: The user's message.

        Returns:
            The agent's final response.
        """
        response_parts = []
        async for event in self.chat(message):
            if event.type == EventType.MESSAGE:
                response_parts.append(event.content or "")
            elif event.type == EventType.ERROR:
                return f"Error: {event.content}"
            elif event.type == EventType.CANCELLED:
                return "Operation cancelled"

        return "\n".join(response_parts)

    def get_conversation_history(self) -> list[dict]:
        """Get the conversation history as a list of dicts."""
        return [msg.to_dict() for msg in self.conversation]


# ============================================================================
# Factory Functions
# ============================================================================


async def create_agent(
    deployment_id: int | None = None,
    provider: str = "system",
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    mcp_api_url: str | None = None,
    mcp_api_token: str | None = None,
) -> AgentService:
    """
    Factory function to create an AgentService with appropriate configuration.

    Args:
        deployment_id: LMStack deployment ID to use as LLM backend.
        provider: LLM provider ("system", "openai", "custom").
        api_key: API key for the LLM provider.
        base_url: Base URL for the LLM API.
        model: Model name to use.
        mcp_api_url: LMStack API URL for MCP.
        mcp_api_token: LMStack API token for MCP.

    Returns:
        Configured AgentService instance.
    """
    # Determine LLM configuration based on provider
    if provider == "system" and deployment_id:
        # Use a local LMStack deployment
        # Need to look up the deployment to get the endpoint
        # For now, assume localhost with standard port
        llm_base_url = f"http://localhost:8000/api/deployments/{deployment_id}/v1"
        llm_api_key = "dummy"
        llm_model = model or "default"
    elif provider == "openai":
        llm_base_url = "https://api.openai.com/v1"
        llm_api_key = api_key
        llm_model = model or "gpt-4o"
    elif provider == "custom":
        llm_base_url = base_url
        llm_api_key = api_key or "dummy"
        llm_model = model or "default"
    else:
        raise ValueError(f"Invalid provider: {provider}")

    agent = AgentService(
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        mcp_api_url=mcp_api_url,
        mcp_api_token=mcp_api_token,
    )

    await agent.initialize()
    return agent
