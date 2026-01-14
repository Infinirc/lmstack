/**
 * Chat Component Styles
 *
 * CSS-in-JS styles for chat components.
 * Organized by component for maintainability.
 */
import type { ThemeColors } from './types'

interface StyleConfig {
  isDark: boolean
  colors: ThemeColors
}

/**
 * Generate dynamic CSS styles for chat components
 */
export function getChatStyles({ isDark, colors }: StyleConfig): string {
  return `
    /* Scrollbar styles */
    .chat-page .messages-container::-webkit-scrollbar {
      width: 6px;
    }
    .chat-page .messages-container::-webkit-scrollbar-track {
      background: transparent;
    }
    .chat-page .messages-container::-webkit-scrollbar-thumb {
      background: ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'};
      border-radius: 3px;
    }
    .chat-page .messages-container::-webkit-scrollbar-thumb:hover {
      background: ${isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)'};
    }

    /* Input container focus state */
    .chat-page .chat-input-container:focus-within {
      border-color: ${isDark ? 'rgba(200, 200, 200, 0.4)' : 'rgba(200, 200, 200, 0.6)'};
      box-shadow: ${isDark ? '0 4px 32px rgba(0, 0, 0, 0.4)' : '0 4px 32px rgba(0, 0, 0, 0.12)'};
    }

    /* Send button hover */
    .chat-page .chat-send-btn:hover:not(:disabled) {
      opacity: 0.85;
    }

    /* Message row - no animation to avoid flash on streaming complete */
    .chat-page .message-row {
      opacity: 1;
    }

    /* Suggestion button - simple fade in */
    .chat-page .suggestion-btn {
      opacity: 1;
    }

    .chat-page .suggestion-btn:hover {
      background: ${colors.hoverBg} !important;
      border-color: ${isDark ? '#3f3f46' : '#d4d4d8'} !important;
      color: ${colors.text} !important;
    }

    /* Thinking indicator */
    .thinking-indicator {
      display: flex;
      align-items: center;
      padding: 8px 0;
    }

    .thinking-dots {
      display: flex;
      gap: 4px;
    }

    .thinking-dots span {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      animation: thinkingBounce 1.4s ease-in-out infinite;
    }

    .thinking-dots span:nth-child(1) { animation-delay: 0s; }
    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes thinkingBounce {
      0%, 80%, 100% {
        transform: scale(0.8);
        opacity: 0.5;
      }
      40% {
        transform: scale(1);
        opacity: 1;
      }
    }

    /* Thinking pulse */
    .thinking-pulse {
      animation: thinkingPulse 1.5s ease-in-out infinite;
    }

    @keyframes thinkingPulse {
      0%, 100% { opacity: 0.4; }
      50% { opacity: 1; }
    }

    /* Blinking cursor */
    .cursor-blink {
      animation: cursorBlink 1s step-end infinite;
    }

    @keyframes cursorBlink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0; }
    }

    /* Markdown body styles */
    .markdown-body {
      font-size: 15px;
      line-height: 1.7;
      color: ${colors.text};
    }

    .markdown-body p {
      margin: 0 0 12px 0;
    }

    .markdown-body p:last-child {
      margin-bottom: 0;
    }

    .markdown-body ul,
    .markdown-body ol {
      margin: 12px 0;
      padding-left: 24px;
    }

    .markdown-body li {
      margin-bottom: 6px;
    }

    .markdown-body pre {
      margin: 12px 0;
      border-radius: 10px;
      overflow: auto;
    }

    .markdown-body table {
      border-collapse: collapse;
      margin: 12px 0;
      width: 100%;
      border-radius: 8px;
      overflow: hidden;
    }

    .markdown-body th,
    .markdown-body td {
      border: 1px solid ${colors.border};
      padding: 10px 14px;
      text-align: left;
    }

    .markdown-body th {
      background: ${isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)'};
      font-weight: 600;
    }

    .markdown-body blockquote {
      margin: 12px 0;
      padding: 8px 16px;
      border-left: 3px solid ${isDark ? '#3f3f46' : '#d4d4d8'};
      background: ${isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)'};
      border-radius: 0 8px 8px 0;
      color: ${colors.textSecondary};
    }

    .markdown-body h1,
    .markdown-body h2,
    .markdown-body h3 {
      margin: 20px 0 12px 0;
      font-weight: 600;
      color: ${colors.text};
    }

    .markdown-body h1 { font-size: 1.4em; }
    .markdown-body h2 { font-size: 1.2em; }
    .markdown-body h3 { font-size: 1.1em; }

    .markdown-body hr {
      border: none;
      border-top: 1px solid ${colors.border};
      margin: 20px 0;
    }

    .markdown-body a {
      color: ${isDark ? '#60a5fa' : '#2563eb'};
      text-decoration: none;
    }

    .markdown-body a:hover {
      text-decoration: underline;
    }

    .markdown-body strong {
      font-weight: 600;
    }
  `
}
