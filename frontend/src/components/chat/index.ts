/**
 * Chat Components
 *
 * Modular chat UI components for the LMStack chat interface.
 * Designed for maintainability and reusability.
 */

// Types and utilities
export type { ThemeColors, ChatMessage, ParsedContent } from "./types";
export {
  getThemeColors,
  parseThinkingContent,
  generateMessageId,
} from "./types";

// Hooks
export { useTheme } from "./useTheme";

// Components
export { ChatInput } from "./ChatInput";
export { CodeBlock, InlineCode } from "./CodeBlock";
export { MessageContent } from "./MessageContent";
export { ThinkingBlock, ThinkingIndicator } from "./ThinkingBlock";

// Styles
export { getChatStyles } from "./styles";
