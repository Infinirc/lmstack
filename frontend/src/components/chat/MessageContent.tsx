/**
 * MessageContent Component
 *
 * Renders message content with Markdown support, code highlighting,
 * and thinking block visualization.
 */
import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CodeBlock, InlineCode } from "./CodeBlock";
import { ThinkingBlock, ThinkingIndicator } from "./ThinkingBlock";
import { parseThinkingContent, type ThemeColors } from "./types";

interface MessageContentProps {
  content: string;
  isStreaming: boolean;
  isDark: boolean;
  colors: ThemeColors;
}

/**
 * Main message content renderer
 */
export function MessageContent({
  content,
  isStreaming,
  isDark,
  colors,
}: MessageContentProps) {
  const parsed = useMemo(() => parseThinkingContent(content), [content]);

  // Create markdown components with current theme
  const markdownComponents = useMemo(
    () => createMarkdownComponents(isDark, colors),
    [isDark, colors],
  );

  // Show indicator when streaming with no content
  if (isStreaming && !content) {
    return <ThinkingIndicator isDark={isDark} />;
  }

  return (
    <div className="message-content">
      {/* Thinking block if present */}
      {parsed.thinking !== null && (
        <ThinkingBlock
          content={parsed.thinking}
          isComplete={parsed.isThinkingComplete}
          isDark={isDark}
          colors={colors}
        />
      )}

      {/* Main response content */}
      {parsed.response ? (
        <div className="markdown-body">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={markdownComponents}
          >
            {parsed.response}
          </ReactMarkdown>
        </div>
      ) : (
        // Show cursor when thinking but no response yet
        parsed.thinking &&
        !parsed.isThinkingComplete && (
          <div style={{ display: "flex", alignItems: "center", marginTop: 8 }}>
            <BlinkingCursor color={colors.text} />
          </div>
        )
      )}

      {/* Blinking cursor during streaming */}
      {isStreaming && parsed.response && (
        <BlinkingCursor color={colors.text} style={{ marginLeft: 2 }} />
      )}
    </div>
  );
}

interface BlinkingCursorProps {
  color: string;
  style?: React.CSSProperties;
}

/**
 * Animated blinking cursor
 */
function BlinkingCursor({ color, style }: BlinkingCursorProps) {
  return (
    <span
      className="cursor-blink"
      style={{
        display: "inline-block",
        width: 2,
        height: 18,
        background: color,
        verticalAlign: "text-bottom",
        ...style,
      }}
    />
  );
}

/**
 * Create markdown component overrides
 */
function createMarkdownComponents(isDark: boolean, colors: ThemeColors) {
  return {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    code({ inline, className, children, ...props }: any) {
      const match = /language-(\w+)/.exec(className || "");
      const code = String(children).replace(/\n$/, "");

      // Block code with syntax highlighting
      if (!inline && match) {
        return (
          <CodeBlock
            code={code}
            language={match[1]}
            isDark={isDark}
            colors={colors}
          />
        );
      }

      // Inline code
      return (
        <InlineCode isDark={isDark} {...props}>
          {children}
        </InlineCode>
      );
    },
  };
}
