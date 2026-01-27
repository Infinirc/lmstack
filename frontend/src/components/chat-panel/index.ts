/**
 * Chat Panel Components
 *
 * Global chat panel for AI conversations accessible from any page.
 * Supports two modes:
 * - Agent Mode: MCP-based agent with Claude Code-style interaction
 * - Chat Mode: Traditional tool calling via LLM API
 */

// Components
export { ChatPanel } from "./ChatPanel";
export { ModelSelector } from "./ModelSelector";
export { ToolConfirmModal } from "./ToolConfirmModal";
export { AgentChatView } from "./AgentChatView";

// Hooks
export { useChat } from "./useChat";
export { useAgentChat } from "./useAgentChat";

// Types
export type {
  ChatModelConfig,
  CustomEndpoint,
  ChatPanelState,
  ModelSourceType,
} from "./types";

export type {
  ToolDefinition,
  ToolCall,
  ToolResult,
  ToolMeta,
  PendingToolExecution,
} from "./tools";

export type {
  AgentEventType,
  AgentEvent,
  ExecutionStep,
  AgentChatMessage,
  AgentLLMConfig,
} from "./useAgentChat";

export {
  DEFAULT_PANEL_WIDTH,
  MIN_PANEL_WIDTH,
  MAX_PANEL_WIDTH,
  CHAT_PANEL_STORAGE_KEY,
} from "./types";

export {
  CHAT_TOOLS,
  TOOL_META,
  requiresConfirmation,
  getToolMeta,
  executeTool,
} from "./tools";
