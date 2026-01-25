/**
 * Chat Component Type Definitions
 *
 * Shared types for chat-related components.
 */

/** Tool call from LLM response */
export interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  /** Model name that generated this response (for MoM/Semantic Router) */
  model?: string;
  /** Extended thinking content */
  thinking?: string;
  /** Tool calls made by the assistant */
  toolCalls?: ToolCall[];
}

export interface ParsedContent {
  thinking: string | null;
  response: string;
  isThinkingComplete: boolean;
}

export interface ThemeColors {
  bg: string;
  chatBg: string;
  userBubble: string;
  text: string;
  textSecondary: string;
  textMuted: string;
  border: string;
  inputBg: string;
  inputBorder: string;
  hoverBg: string;
  sendBtnActive: string;
  sendBtnActiveText: string;
  sendBtnDisabled: string;
  sendBtnDisabledText: string;
  codeBg: string;
  codeHeaderBg: string;
}

/**
 * Get theme colors based on dark mode state
 */
export function getThemeColors(isDark: boolean): ThemeColors {
  if (isDark) {
    return {
      bg: "#0d0d0d",
      chatBg: "#171717",
      userBubble: "#262626",
      text: "#fafafa",
      textSecondary: "#a1a1aa",
      textMuted: "#71717a",
      border: "#262626",
      inputBg: "rgba(75, 75, 75, 0.05)",
      inputBorder: "rgba(133, 133, 133, 0.3)",
      hoverBg: "rgba(255,255,255,0.05)",
      sendBtnActive: "#ffffff",
      sendBtnActiveText: "#000000",
      sendBtnDisabled: "#3f3f46",
      sendBtnDisabledText: "#71717a",
      codeBg: "#1e1e1e",
      codeHeaderBg: "#2d2d2d",
    };
  }

  return {
    bg: "#ffffff",
    chatBg: "#f9f9f9",
    userBubble: "#f4f4f5",
    text: "#09090b",
    textSecondary: "#52525b",
    textMuted: "#a1a1aa",
    border: "#e4e4e7",
    inputBg: "rgba(255, 255, 255, 0.05)",
    inputBorder: "rgba(228, 228, 231, 0.3)",
    hoverBg: "rgba(0,0,0,0.03)",
    sendBtnActive: "#000000",
    sendBtnActiveText: "#ffffff",
    sendBtnDisabled: "#e4e4e7",
    sendBtnDisabledText: "#a1a1aa",
    codeBg: "#f8f8f8",
    codeHeaderBg: "#e8e8e8",
  };
}

/**
 * Parse content to extract thinking blocks
 */
export function parseThinkingContent(content: string): ParsedContent {
  if (!content) {
    return { thinking: null, response: "", isThinkingComplete: false };
  }

  const thinkStartMatch = content.match(/<think>/i);
  const thinkEndMatch = content.match(/<\/think>/i);

  if (!thinkStartMatch) {
    return { thinking: null, response: content, isThinkingComplete: true };
  }

  const thinkStartIndex = thinkStartMatch.index!;

  if (!thinkEndMatch) {
    const thinkingContent = content.slice(thinkStartIndex + 7);
    return {
      thinking: thinkingContent,
      response: "",
      isThinkingComplete: false,
    };
  }

  const thinkEndIndex = thinkEndMatch.index!;
  const thinkingContent = content.slice(thinkStartIndex + 7, thinkEndIndex);
  const responseContent = content.slice(thinkEndIndex + 8).trim();

  return {
    thinking: thinkingContent,
    response: responseContent,
    isThinkingComplete: true,
  };
}

/**
 * Generate unique message ID
 */
export function generateMessageId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}
