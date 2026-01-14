/**
 * CodeBlock Component
 *
 * Renders syntax-highlighted code blocks with copy functionality.
 * Supports multiple programming languages via Prism.
 */
import { useState, useCallback } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import {
  oneDark,
  oneLight,
} from "react-syntax-highlighter/dist/esm/styles/prism";
import { CheckOutlined, CopyOutlined } from "@ant-design/icons";
import type { ThemeColors } from "./types";

interface CodeBlockProps {
  code: string;
  language: string;
  isDark: boolean;
  colors: ThemeColors;
}

/**
 * Code block with syntax highlighting and copy button
 */
export function CodeBlock({ code, language, isDark, colors }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      // Try modern clipboard API first
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
        return;
      }

      // Fallback for non-secure contexts (HTTP)
      const textArea = document.createElement("textarea");
      textArea.value = code;
      textArea.style.position = "fixed";
      textArea.style.left = "-999999px";
      textArea.style.top = "-999999px";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();

      const successful = document.execCommand("copy");
      document.body.removeChild(textArea);

      if (successful) {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }
    } catch (err) {
      console.error("Failed to copy code:", err);
    }
  }, [code]);

  const codeStyle = isDark ? oneDark : oneLight;

  return (
    <div
      style={{
        margin: "12px 0",
        borderRadius: 10,
        overflow: "hidden",
        border: `1px solid ${colors.border}`,
      }}
    >
      {/* Header with language and copy button */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "8px 12px",
          background: colors.codeHeaderBg,
          borderBottom: `1px solid ${colors.border}`,
        }}
      >
        <span
          style={{
            fontSize: 12,
            color: colors.textMuted,
            textTransform: "lowercase",
          }}
        >
          {language}
        </span>
        <button
          onClick={handleCopy}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 4,
            padding: "4px 8px",
            fontSize: 12,
            color: copied ? "#22c55e" : colors.textSecondary,
            background: "transparent",
            border: "none",
            borderRadius: 4,
            cursor: "pointer",
            transition: "all 0.15s ease",
          }}
          onMouseOver={(e) => {
            if (!copied) {
              e.currentTarget.style.color = colors.text;
            }
          }}
          onMouseOut={(e) => {
            if (!copied) {
              e.currentTarget.style.color = colors.textSecondary;
            }
          }}
        >
          {copied ? (
            <>
              <CheckOutlined style={{ fontSize: 12 }} />
              <span>Copied!</span>
            </>
          ) : (
            <>
              <CopyOutlined style={{ fontSize: 12 }} />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code content */}
      <SyntaxHighlighter
        style={codeStyle}
        language={language}
        PreTag="div"
        customStyle={{
          margin: 0,
          padding: "12px 16px",
          fontSize: 13,
          background: colors.codeBg,
          borderRadius: 0,
        }}
        codeTagProps={{
          style: {
            fontFamily: "'SF Mono', 'Consolas', 'Monaco', monospace",
          },
        }}
        lineProps={{
          style: {
            background: "transparent",
            borderBottom: "none",
            display: "block",
          },
        }}
        wrapLines={true}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

interface InlineCodeProps {
  children: React.ReactNode;
  isDark: boolean;
}

/**
 * Inline code styling
 */
export function InlineCode({ children, isDark }: InlineCodeProps) {
  return (
    <code
      style={{
        background: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.05)",
        padding: "2px 6px",
        borderRadius: 4,
        fontFamily: "'SF Mono', 'Consolas', 'Monaco', monospace",
        fontSize: "0.9em",
      }}
    >
      {children}
    </code>
  );
}
