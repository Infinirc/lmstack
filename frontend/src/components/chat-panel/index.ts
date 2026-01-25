/**
 * Chat Panel Components
 *
 * Global chat panel for AI conversations accessible from any page.
 */

// Components
export { ChatPanel, TUNING_JOB_EVENT_KEY } from "./ChatPanel";
export { ModelSelector } from "./ModelSelector";
export { ToolConfirmModal } from "./ToolConfirmModal";

// Hooks
export { useChat } from "./useChat";

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
