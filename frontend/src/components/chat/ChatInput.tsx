/**
 * ChatInput Component
 *
 * Message input area with auto-resize textarea and send/stop button.
 * Supports keyboard shortcuts and disabled states.
 */
import { useRef, useCallback, useEffect } from "react";
import type { ThemeColors } from "./types";

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled: boolean;
  isDark: boolean;
  colors: ThemeColors;
}

/**
 * Chat message input with send/stop button
 */
export function ChatInput({
  value,
  onChange,
  onSend,
  onStop,
  isStreaming,
  disabled,
  isDark,
  colors,
}: ChatInputProps) {
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      // Don't trigger send during IME composition (Chinese/Japanese/Korean input)
      if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
        e.preventDefault();
        onSend();
      }
    },
    [onSend],
  );

  // Auto-resize textarea
  useEffect(() => {
    const textarea = inputRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      const newHeight = Math.min(textarea.scrollHeight, 200); // max 8 rows approx
      textarea.style.height = `${newHeight}px`;
    }
  }, [value]);

  const canSend = value.trim() && !disabled && !isStreaming;

  return (
    <div
      className="chat-input-container"
      style={{
        display: "flex",
        alignItems: "flex-end",
        gap: 0,
        padding: "8px 8px 8px 16px",
        borderRadius: 24,
        border: `1px solid ${colors.inputBorder}`,
        background: colors.inputBg,
        backdropFilter: "blur(8px)",
        boxShadow: isDark
          ? "0 4px 24px rgba(0, 0, 0, 0.3)"
          : "0 4px 24px rgba(0, 0, 0, 0.08)",
        transition: "border-color 0.2s ease, box-shadow 0.2s ease",
      }}
    >
      <textarea
        ref={inputRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={
          disabled
            ? "Select a model first"
            : isStreaming
              ? "Waiting for response..."
              : "Send a Message"
        }
        disabled={disabled}
        rows={1}
        style={{
          flex: 1,
          resize: "none",
          fontSize: 15,
          padding: "8px 0",
          background: "transparent",
          color: colors.text,
          lineHeight: 1.5,
          outline: "none",
          boxShadow: "none",
          border: "none",
          fontFamily: "inherit",
          overflow: "hidden",
        }}
      />
      <SendButton
        onClick={isStreaming ? onStop : onSend}
        isStreaming={isStreaming}
        disabled={!isStreaming && !canSend}
        colors={colors}
      />
    </div>
  );
}

interface SendButtonProps {
  onClick: () => void;
  isStreaming: boolean;
  disabled: boolean;
  colors: ThemeColors;
}

/**
 * Send/Stop button component
 */
function SendButton({
  onClick,
  isStreaming,
  disabled,
  colors,
}: SendButtonProps) {
  // Always show active style when streaming (for stop button)
  const isActive = isStreaming || (!disabled && !isStreaming);
  const showStopIcon = isStreaming;

  return (
    <button
      onClick={onClick}
      disabled={disabled && !isStreaming}
      className="chat-send-btn"
      style={{
        width: 36,
        height: 36,
        borderRadius: "50%",
        background: isActive ? colors.sendBtnActive : colors.sendBtnDisabled,
        border: "none",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: disabled && !isStreaming ? "not-allowed" : "pointer",
        transition: "all 0.15s ease",
        color: isActive ? colors.sendBtnActiveText : colors.sendBtnDisabledText,
        flexShrink: 0,
      }}
      aria-label={showStopIcon ? "Stop generation" : "Send message"}
    >
      {showStopIcon ? <StopIcon /> : <ArrowUpIcon />}
    </button>
  );
}

/**
 * Arrow up icon for send button
 */
function ArrowUpIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 16 16"
      fill="currentColor"
      style={{ width: 18, height: 18 }}
    >
      <path
        fillRule="evenodd"
        d="M8 14a.75.75 0 0 1-.75-.75V4.56L4.03 7.78a.75.75 0 0 1-1.06-1.06l4.5-4.5a.75.75 0 0 1 1.06 0l4.5 4.5a.75.75 0 0 1-1.06 1.06L8.75 4.56v8.69A.75.75 0 0 1 8 14Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

/**
 * Stop icon for streaming state
 */
function StopIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 16 16"
      fill="currentColor"
      style={{ width: 14, height: 14 }}
    >
      <rect x="3" y="3" width="10" height="10" rx="1" />
    </svg>
  );
}
