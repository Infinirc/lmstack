/**
 * Chat Panel Components
 *
 * Global chat panel for AI conversations accessible from any page.
 * Uses MCP-based agent with Claude Code-style interaction.
 */

// Components
export { ChatPanel } from "./ChatPanel";
export { ModelSelector } from "./ModelSelector";
export { AgentChatView } from "./AgentChatView";

// Hooks
export { useAgentChat } from "./useAgentChat";

// Types
export type {
  ChatModelConfig,
  CustomEndpoint,
  ChatPanelState,
  ModelSourceType,
} from "./types";

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
