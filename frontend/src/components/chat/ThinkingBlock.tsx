/**
 * ThinkingBlock Component
 *
 * Displays AI thinking process in a collapsible block.
 * Shows animated indicators during active thinking.
 */
import { useState, useEffect, useRef } from "react";
import type { ThemeColors } from "./types";

interface ThinkingIndicatorProps {
  isDark: boolean;
}

/**
 * Animated dots indicator for thinking state
 */
export function ThinkingIndicator({ isDark }: ThinkingIndicatorProps) {
  const dotColor = isDark ? "#71717a" : "#a1a1aa";

  return (
    <div className="thinking-indicator">
      <div className="thinking-dots">
        <span style={{ background: dotColor }} />
        <span style={{ background: dotColor }} />
        <span style={{ background: dotColor }} />
      </div>
    </div>
  );
}

interface ThinkingBlockProps {
  content: string;
  isComplete: boolean;
  isDark: boolean;
  colors: ThemeColors;
}

/**
 * Collapsible thinking block component
 */
export function ThinkingBlock({
  content,
  isComplete,
  isDark,
  colors,
}: ThinkingBlockProps) {
  const [isExpanded, setIsExpanded] = useState(!isComplete);
  const contentRef = useRef<HTMLDivElement>(null);

  // Auto-collapse when thinking completes
  useEffect(() => {
    if (isComplete) {
      setIsExpanded(false);
    }
  }, [isComplete]);

  // Auto-scroll thinking content to bottom
  useEffect(() => {
    if (isExpanded && !isComplete && contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [content, isExpanded, isComplete]);

  // Show indicator when no content yet
  if (!content && !isComplete) {
    return <ThinkingIndicator isDark={isDark} />;
  }

  const arrowColor = isDark ? "#71717a" : "#a1a1aa";

  return (
    <div
      className="thinking-block"
      style={{
        marginBottom: 12,
        padding: "8px 12px",
        borderRadius: 8,
        background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)",
      }}
    >
      {/* Header toggle */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          width: "100%",
          padding: 0,
          background: "none",
          border: "none",
          cursor: "pointer",
          textAlign: "left",
        }}
      >
        <svg
          width="12"
          height="12"
          viewBox="0 0 12 12"
          fill="none"
          style={{
            transform: isExpanded ? "rotate(90deg)" : "rotate(0deg)",
            transition: "transform 0.15s ease",
            flexShrink: 0,
          }}
        >
          <path
            d="M4.5 2.5L8 6L4.5 9.5"
            stroke={arrowColor}
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span style={{ fontSize: 13, color: colors.textMuted }}>
          {isComplete ? "Thought process" : "Thinking..."}
        </span>
        {!isComplete && <ThinkingPulse isDark={isDark} />}
      </button>

      {/* Expandable content */}
      {isExpanded && (
        <div
          ref={contentRef}
          style={{
            fontSize: 13,
            lineHeight: 1.6,
            color: colors.textSecondary,
            whiteSpace: "pre-wrap",
            maxHeight: 200,
            overflow: "auto",
            marginTop: 8,
            paddingLeft: 20,
          }}
        >
          {content}
        </div>
      )}
    </div>
  );
}

interface ThinkingPulseProps {
  isDark: boolean;
}

/**
 * Pulsing dot indicator
 */
function ThinkingPulse({ isDark }: ThinkingPulseProps) {
  return (
    <div
      className="thinking-pulse"
      style={{
        width: 6,
        height: 6,
        borderRadius: "50%",
        background: isDark ? "#71717a" : "#a1a1aa",
        marginLeft: 8,
      }}
    />
  );
}
