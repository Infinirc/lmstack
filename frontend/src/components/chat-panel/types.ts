/**
 * Chat Panel Type Definitions
 *
 * Types for the global chat panel component.
 */

/**
 * Model source types
 */
export type ModelSourceType = "deployment" | "semantic-router" | "custom";

/**
 * Model configuration for chat
 */
export interface ChatModelConfig {
  type: ModelSourceType;
  /** Deployment ID (when type is "deployment") */
  deploymentId?: number;
  /** Display name */
  name: string;
  /** Model ID for API requests (when type is "custom") */
  modelId?: string;
  /** Custom endpoint URL (when type is "custom") */
  endpoint?: string;
  /** Custom API key (when type is "custom") */
  apiKey?: string;
}

/**
 * Custom endpoint configuration
 */
export interface CustomEndpoint {
  id: string;
  name: string;
  endpoint: string;
  apiKey?: string;
  /** Model ID for API requests */
  modelId?: string;
}

/**
 * Chat panel state stored in localStorage
 */
export interface ChatPanelState {
  isOpen: boolean;
  width: number;
  selectedModel: ChatModelConfig | null;
  customEndpoints: CustomEndpoint[];
}

/**
 * Default panel width
 */
export const DEFAULT_PANEL_WIDTH = 420;

/**
 * Min/max panel width
 */
export const MIN_PANEL_WIDTH = 320;
export const MAX_PANEL_WIDTH = 600;

/**
 * Storage key for chat panel state
 */
export const CHAT_PANEL_STORAGE_KEY = "lmstack-chat-panel";
