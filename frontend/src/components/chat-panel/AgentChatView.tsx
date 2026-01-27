/**
 * Agent Chat View Component
 *
 * Claude Code-style chat interface that shows:
 * - Real-time thinking process
 * - Step-by-step tool execution
 * - Expandable tool results
 * - Streaming message content
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { Button, Tooltip } from "antd";
import {
  RobotOutlined,
  UserOutlined,
  LoadingOutlined,
  CheckCircleFilled,
  CloseCircleFilled,
  CaretRightOutlined,
  ThunderboltOutlined,
  ToolOutlined,
  BulbOutlined,
  CopyOutlined,
  CheckOutlined,
  ClusterOutlined,
  RocketOutlined,
  DatabaseOutlined,
  ContainerOutlined,
  DashboardOutlined,
  KeyOutlined,
  RightOutlined,
} from "@ant-design/icons";
import { MessageContent, getThemeColors } from "../chat";
import type { ThemeColors } from "../chat";
import type {
  AgentChatMessage,
  ExecutionStep,
  PageReference,
  ActionSuggestion,
} from "./useAgentChat";
import type { AppColors } from "../../hooks/useTheme";

interface AgentChatViewProps {
  messages: AgentChatMessage[];
  currentSteps: ExecutionStep[];
  isStreaming: boolean;
  isThinking: boolean;
  currentTool: string | null;
  onToggleStep: (stepId: string) => void;
  onNavigate?: (path: string) => void;
  onSendMessage?: (message: string) => void;
  isDark: boolean;
  colors: AppColors;
  /** Whether user has scrolled up - if true, don't auto-scroll */
  userScrolledUp?: boolean;
}

/**
 * Agent chat view with Claude Code-style execution display
 */
export function AgentChatView({
  messages,
  currentSteps,
  isStreaming,
  isThinking,
  currentTool: _currentTool,
  onToggleStep,
  onNavigate,
  onSendMessage,
  isDark,
  colors: _colors,
  userScrolledUp = false,
}: AgentChatViewProps) {
  void _colors; // Used for future customization
  void _currentTool; // Reserved for future tool highlighting
  const chatColors = getThemeColors(isDark);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom - only if user hasn't scrolled up
  useEffect(() => {
    if (!userScrolledUp) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, currentSteps, userScrolledUp]);

  if (messages.length === 0 && currentSteps.length === 0) {
    return <EmptyState colors={chatColors} isDark={isDark} />;
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 20,
        padding: "8px 0",
      }}
    >
      {messages.map((msg, index) => {
        const isLast = index === messages.length - 1;
        const showCurrentSteps =
          isLast && msg.role === "assistant" && isStreaming;

        return (
          <MessageBlock
            key={msg.id}
            message={msg}
            steps={showCurrentSteps ? currentSteps : msg.steps}
            pageReferences={msg.pageReferences || []}
            actionSuggestions={msg.actionSuggestions || []}
            isStreaming={msg.isStreaming}
            isThinking={isLast && isThinking}
            onToggleStep={onToggleStep}
            onNavigate={onNavigate}
            onSendMessage={onSendMessage}
            isDark={isDark}
            colors={chatColors}
          />
        );
      })}
      <div ref={messagesEndRef} />
    </div>
  );
}

/**
 * Empty state component
 */
function EmptyState({
  colors,
  isDark: _isDark,
}: {
  colors: ThemeColors;
  isDark: boolean;
}) {
  const isDark = _isDark;
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        textAlign: "center",
        padding: "40px 20px",
      }}
    >
      <div
        style={{
          width: 56,
          height: 56,
          borderRadius: 16,
          background: `linear-gradient(135deg, #3b82f6, #8b5cf6)`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: 20,
          boxShadow: "0 4px 20px rgba(59, 130, 246, 0.3)",
        }}
      >
        <ThunderboltOutlined style={{ fontSize: 28, color: "#fff" }} />
      </div>
      <div
        style={{
          fontSize: 17,
          fontWeight: 600,
          color: colors.text,
          marginBottom: 8,
        }}
      >
        LMStack AI Agent
      </div>
      <div
        style={{
          fontSize: 14,
          color: colors.textMuted,
          marginBottom: 24,
          maxWidth: 280,
          lineHeight: 1.5,
        }}
      >
        Powered by MCP. Deploy models, run benchmarks, and optimize
        configurations through natural language.
      </div>

      {/* Capability cards */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 10,
          width: "100%",
          maxWidth: 320,
        }}
      >
        {[
          { icon: <RobotOutlined />, text: "Deploy LLM models to GPU workers" },
          { icon: <ToolOutlined />, text: "Run performance benchmarks" },
          {
            icon: <BulbOutlined />,
            text: "Query knowledge base for optimal configs",
          },
        ].map((item, idx) => (
          <div
            key={idx}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              padding: "12px 16px",
              borderRadius: 10,
              background: isDark
                ? "rgba(255,255,255,0.03)"
                : "rgba(0,0,0,0.02)",
              border: `1px solid ${isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"}`,
            }}
          >
            <div style={{ color: "#3b82f6", fontSize: 16 }}>{item.icon}</div>
            <div style={{ fontSize: 13, color: colors.textSecondary }}>
              {item.text}
            </div>
          </div>
        ))}
      </div>

      <div
        style={{
          marginTop: 24,
          fontSize: 12,
          color: colors.textMuted,
        }}
      >
        Try: "Deploy Qwen-7B on Worker 1" or "What's the GPU memory status?"
      </div>
    </div>
  );
}

/**
 * Message block component
 */
interface MessageBlockProps {
  message: AgentChatMessage;
  steps: ExecutionStep[];
  pageReferences: PageReference[];
  actionSuggestions: ActionSuggestion[];
  isStreaming?: boolean;
  isThinking?: boolean;
  onToggleStep: (stepId: string) => void;
  onNavigate?: (path: string) => void;
  onSendMessage?: (message: string) => void;
  isDark: boolean;
  colors: ThemeColors;
}

function MessageBlock({
  message,
  steps,
  pageReferences,
  actionSuggestions,
  isStreaming,
  isThinking,
  onToggleStep,
  onNavigate,
  onSendMessage,
  isDark,
  colors,
}: MessageBlockProps) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div style={{ display: "flex", justifyContent: "flex-end", gap: 10 }}>
        <div
          style={{
            maxWidth: "85%",
            padding: "12px 16px",
            borderRadius: 18,
            background: colors.userBubble,
            color: colors.text,
            fontSize: 14,
            lineHeight: 1.5,
            whiteSpace: "pre-wrap",
          }}
        >
          {message.content}
        </div>
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: "50%",
            background: isDark ? "#3f3f46" : "#e4e4e7",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          <UserOutlined style={{ fontSize: 14, color: colors.textSecondary }} />
        </div>
      </div>
    );
  }

  // Assistant message with steps
  return (
    <div style={{ display: "flex", gap: 10 }}>
      {/* Avatar */}
      <div
        style={{
          width: 32,
          height: 32,
          borderRadius: 10,
          background: `linear-gradient(135deg, #3b82f6, #8b5cf6)`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        <ThunderboltOutlined style={{ fontSize: 16, color: "#fff" }} />
      </div>

      {/* Content */}
      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Execution steps */}
        {steps.length > 0 && (
          <div style={{ marginBottom: 12 }}>
            <StepList
              steps={steps}
              onToggleStep={onToggleStep}
              isDark={isDark}
              colors={colors}
            />
          </div>
        )}

        {/* Thinking indicator (when no steps yet) */}
        {isThinking && steps.length === 0 && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "12px 14px",
              borderRadius: 12,
              background: isDark
                ? "rgba(59, 130, 246, 0.1)"
                : "rgba(59, 130, 246, 0.06)",
              border: `1px solid ${isDark ? "rgba(59, 130, 246, 0.2)" : "rgba(59, 130, 246, 0.15)"}`,
              marginBottom: 12,
            }}
          >
            <LoadingOutlined style={{ color: "#3b82f6", fontSize: 14 }} spin />
            <span style={{ fontSize: 13, color: "#3b82f6" }}>
              Analyzing your request...
            </span>
          </div>
        )}

        {/* Message content */}
        {message.content && (
          <div style={{ color: colors.text }}>
            <MessageContent
              content={message.content}
              isStreaming={isStreaming ?? false}
              isDark={isDark}
              colors={colors}
            />
          </div>
        )}

        {/* Page reference cards */}
        {pageReferences.length > 0 && (
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 8,
              marginTop: 12,
            }}
          >
            {pageReferences.map((ref) => (
              <PageReferenceCard
                key={ref.path}
                reference={ref}
                onClick={() => onNavigate?.(ref.path)}
                isDark={isDark}
                colors={colors}
              />
            ))}
          </div>
        )}

        {/* Action suggestion buttons */}
        {actionSuggestions.length > 0 && !isStreaming && (
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 8,
              marginTop: 12,
            }}
          >
            {actionSuggestions.map((suggestion, idx) => (
              <ActionSuggestionButton
                key={`${suggestion.message}-${idx}`}
                suggestion={suggestion}
                onClick={() => onSendMessage?.(suggestion.message)}
                isDark={isDark}
                colors={colors}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Page reference card component
 */
interface PageReferenceCardProps {
  reference: PageReference;
  onClick?: () => void;
  isDark: boolean;
  colors: ThemeColors;
}

function PageReferenceCard({
  reference,
  onClick,
  isDark,
  colors,
}: PageReferenceCardProps) {
  const getIcon = () => {
    const iconStyle = { fontSize: 16, color: "#3b82f6" };
    switch (reference.icon) {
      case "cluster":
        return <ClusterOutlined style={iconStyle} />;
      case "rocket":
        return <RocketOutlined style={iconStyle} />;
      case "database":
        return <DatabaseOutlined style={iconStyle} />;
      case "container":
        return <ContainerOutlined style={iconStyle} />;
      case "dashboard":
        return <DashboardOutlined style={iconStyle} />;
      case "key":
        return <KeyOutlined style={iconStyle} />;
      default:
        return <RightOutlined style={iconStyle} />;
    }
  };

  return (
    <div
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "10px 14px",
        borderRadius: 10,
        background: isDark
          ? "rgba(59, 130, 246, 0.1)"
          : "rgba(59, 130, 246, 0.06)",
        border: `1px solid ${
          isDark ? "rgba(59, 130, 246, 0.25)" : "rgba(59, 130, 246, 0.2)"
        }`,
        cursor: "pointer",
        transition: "all 0.2s ease",
        minWidth: 180,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = isDark
          ? "rgba(59, 130, 246, 0.15)"
          : "rgba(59, 130, 246, 0.1)";
        e.currentTarget.style.borderColor = isDark
          ? "rgba(59, 130, 246, 0.4)"
          : "rgba(59, 130, 246, 0.35)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = isDark
          ? "rgba(59, 130, 246, 0.1)"
          : "rgba(59, 130, 246, 0.06)";
        e.currentTarget.style.borderColor = isDark
          ? "rgba(59, 130, 246, 0.25)"
          : "rgba(59, 130, 246, 0.2)";
      }}
    >
      <div
        style={{
          width: 32,
          height: 32,
          borderRadius: 8,
          background: isDark
            ? "rgba(59, 130, 246, 0.15)"
            : "rgba(59, 130, 246, 0.1)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {getIcon()}
      </div>
      <div style={{ flex: 1 }}>
        <div
          style={{
            fontSize: 13,
            fontWeight: 500,
            color: colors.text,
            marginBottom: 2,
          }}
        >
          {reference.title}
        </div>
        <div style={{ fontSize: 11, color: colors.textMuted }}>
          {reference.description}
        </div>
      </div>
      <RightOutlined style={{ fontSize: 12, color: colors.textMuted }} />
    </div>
  );
}

/**
 * Action suggestion button component
 */
interface ActionSuggestionButtonProps {
  suggestion: ActionSuggestion;
  onClick?: () => void;
  isDark: boolean;
  colors: ThemeColors;
}

function ActionSuggestionButton({
  suggestion,
  onClick,
  isDark,
  colors,
}: ActionSuggestionButtonProps) {
  const getIcon = () => {
    const iconStyle = { fontSize: 14 };
    switch (suggestion.icon) {
      case "rocket":
        return <RocketOutlined style={iconStyle} />;
      case "cluster":
        return <ClusterOutlined style={iconStyle} />;
      case "database":
        return <DatabaseOutlined style={iconStyle} />;
      case "container":
        return <ContainerOutlined style={iconStyle} />;
      case "dashboard":
        return <DashboardOutlined style={iconStyle} />;
      case "tool":
        return <ToolOutlined style={iconStyle} />;
      default:
        return <ThunderboltOutlined style={iconStyle} />;
    }
  };

  const getButtonStyle = () => {
    switch (suggestion.type) {
      case "primary":
        return {
          background: isDark
            ? "rgba(59, 130, 246, 0.15)"
            : "rgba(59, 130, 246, 0.1)",
          border: `1px solid ${
            isDark ? "rgba(59, 130, 246, 0.4)" : "rgba(59, 130, 246, 0.3)"
          }`,
          color: "#3b82f6",
          hoverBg: isDark
            ? "rgba(59, 130, 246, 0.25)"
            : "rgba(59, 130, 246, 0.15)",
          hoverBorder: isDark
            ? "rgba(59, 130, 246, 0.6)"
            : "rgba(59, 130, 246, 0.5)",
        };
      case "danger":
        return {
          background: isDark
            ? "rgba(239, 68, 68, 0.1)"
            : "rgba(239, 68, 68, 0.06)",
          border: `1px solid ${
            isDark ? "rgba(239, 68, 68, 0.3)" : "rgba(239, 68, 68, 0.2)"
          }`,
          color: "#ef4444",
          hoverBg: isDark ? "rgba(239, 68, 68, 0.2)" : "rgba(239, 68, 68, 0.1)",
          hoverBorder: isDark
            ? "rgba(239, 68, 68, 0.5)"
            : "rgba(239, 68, 68, 0.4)",
        };
      default:
        return {
          background: isDark
            ? "rgba(255, 255, 255, 0.05)"
            : "rgba(0, 0, 0, 0.04)",
          border: `1px solid ${
            isDark ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.1)"
          }`,
          color: colors.textSecondary,
          hoverBg: isDark ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.08)",
          hoverBorder: isDark
            ? "rgba(255, 255, 255, 0.2)"
            : "rgba(0, 0, 0, 0.2)",
        };
    }
  };

  const style = getButtonStyle();

  return (
    <div
      onClick={onClick}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        padding: "8px 14px",
        borderRadius: 8,
        background: style.background,
        border: style.border,
        color: style.color,
        cursor: "pointer",
        transition: "all 0.2s ease",
        fontSize: 13,
        fontWeight: 500,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = style.hoverBg;
        e.currentTarget.style.borderColor = style.hoverBorder;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = style.background;
        e.currentTarget.style.borderColor = style.border.replace(
          "1px solid ",
          "",
        );
      }}
    >
      {getIcon()}
      <span>{suggestion.label}</span>
    </div>
  );
}

/**
 * Step list component
 */
interface StepListProps {
  steps: ExecutionStep[];
  onToggleStep: (stepId: string) => void;
  isDark: boolean;
  colors: ThemeColors;
}

function StepList({ steps, onToggleStep, isDark, colors }: StepListProps) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 6,
      }}
    >
      {steps.map((step) => (
        <StepItem
          key={step.id}
          step={step}
          onToggle={() => onToggleStep(step.id)}
          isDark={isDark}
          colors={colors}
        />
      ))}
    </div>
  );
}

/**
 * Step item component
 */
interface StepItemProps {
  step: ExecutionStep;
  onToggle: () => void;
  isDark: boolean;
  colors: ThemeColors;
}

function StepItem({ step, onToggle, isDark, colors }: StepItemProps) {
  const [copied, setCopied] = useState(false);

  const getStepIcon = () => {
    if (step.status === "running") {
      return (
        <LoadingOutlined style={{ fontSize: 12, color: "#3b82f6" }} spin />
      );
    }
    if (step.status === "error") {
      return <CloseCircleFilled style={{ fontSize: 12, color: "#ef4444" }} />;
    }
    if (step.status === "completed") {
      return <CheckCircleFilled style={{ fontSize: 12, color: "#22c55e" }} />;
    }
    return (
      <CaretRightOutlined style={{ fontSize: 10, color: colors.textMuted }} />
    );
  };

  const getStepColor = () => {
    if (step.status === "running") return "#3b82f6";
    if (step.status === "error") return "#ef4444";
    if (step.status === "completed") return "#22c55e";
    return colors.textMuted;
  };

  const handleCopyResult = useCallback(() => {
    if (step.toolResult) {
      navigator.clipboard.writeText(step.toolResult);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [step.toolResult]);

  // For thinking/planning steps (simple display)
  if (step.type === "thinking" || step.type === "planning") {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "8px 12px",
          borderRadius: 8,
          background: isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.02)",
          fontSize: 13,
          color: colors.textSecondary,
        }}
      >
        {getStepIcon()}
        <span>{step.title}</span>
      </div>
    );
  }

  // For reasoning steps (expandable with content)
  if (step.type === "reasoning") {
    return (
      <div
        style={{
          borderRadius: 10,
          border: `1px solid ${isDark ? "rgba(139,92,246,0.3)" : "rgba(139,92,246,0.2)"}`,
          overflow: "hidden",
          background: isDark
            ? "rgba(139,92,246,0.05)"
            : "rgba(139,92,246,0.03)",
        }}
      >
        {/* Header */}
        <div
          onClick={onToggle}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "10px 12px",
            cursor: "pointer",
            background: isDark
              ? "rgba(139,92,246,0.08)"
              : "rgba(139,92,246,0.05)",
            borderBottom: step.expanded
              ? `1px solid ${isDark ? "rgba(139,92,246,0.2)" : "rgba(139,92,246,0.15)"}`
              : "none",
          }}
        >
          <BulbOutlined style={{ color: "#8b5cf6", fontSize: 14 }} />
          <span
            style={{
              flex: 1,
              fontSize: 13,
              fontWeight: 500,
              color: "#8b5cf6",
            }}
          >
            {step.title}
          </span>
          {step.status === "running" && (
            <LoadingOutlined style={{ fontSize: 12, color: "#8b5cf6" }} />
          )}
          <CaretRightOutlined
            style={{
              fontSize: 10,
              color: colors.textMuted,
              transform: step.expanded ? "rotate(90deg)" : "rotate(0deg)",
              transition: "transform 0.2s",
            }}
          />
        </div>
        {/* Reasoning content */}
        {step.expanded && step.content && (
          <div
            style={{
              padding: "12px",
              fontSize: 13,
              color: colors.textSecondary,
              lineHeight: 1.6,
              whiteSpace: "pre-wrap",
              maxHeight: 300,
              overflow: "auto",
              fontFamily: "inherit",
            }}
          >
            {step.content}
          </div>
        )}
      </div>
    );
  }

  // For tool steps
  return (
    <div
      style={{
        borderRadius: 10,
        border: `1px solid ${isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"}`,
        overflow: "hidden",
        background: isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.01)",
      }}
    >
      {/* Header */}
      <div
        onClick={onToggle}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "10px 12px",
          cursor: "pointer",
          background: isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.02)",
          borderBottom:
            step.expanded === true
              ? `1px solid ${isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"}`
              : "none",
        }}
      >
        {getStepIcon()}
        <span
          style={{
            flex: 1,
            fontSize: 13,
            fontWeight: 500,
            color: getStepColor(),
          }}
        >
          {step.title}
        </span>
        {step.executionTimeMs && (
          <span
            style={{
              fontSize: 11,
              color: colors.textMuted,
              background: isDark
                ? "rgba(255,255,255,0.05)"
                : "rgba(0,0,0,0.05)",
              padding: "2px 6px",
              borderRadius: 4,
            }}
          >
            {step.executionTimeMs.toFixed(0)}ms
          </span>
        )}
        <CaretRightOutlined
          style={{
            fontSize: 10,
            color: colors.textMuted,
            transform: step.expanded ? "rotate(90deg)" : "rotate(0deg)",
            transition: "transform 0.2s",
          }}
        />
      </div>

      {/* Expanded content */}
      {step.expanded && (
        <div style={{ padding: "12px" }}>
          {/* Tool result */}
          {step.toolResult && (
            <div>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: 6,
                }}
              >
                <div
                  style={{
                    fontSize: 11,
                    color: colors.textMuted,
                    textTransform: "uppercase",
                    letterSpacing: "0.5px",
                  }}
                >
                  Result
                </div>
                <Tooltip title={copied ? "Copied!" : "Copy result"}>
                  <Button
                    type="text"
                    size="small"
                    icon={copied ? <CheckOutlined /> : <CopyOutlined />}
                    onClick={handleCopyResult}
                    style={{
                      color: copied ? "#22c55e" : colors.textMuted,
                      fontSize: 12,
                      padding: "2px 6px",
                    }}
                  />
                </Tooltip>
              </div>
              <ToolResultDisplay
                result={step.toolResult}
                isDark={isDark}
                colors={colors}
              />
            </div>
          )}

          {/* Error */}
          {step.error && (
            <div
              style={{
                padding: "10px 12px",
                borderRadius: 6,
                background: isDark
                  ? "rgba(239, 68, 68, 0.1)"
                  : "rgba(239, 68, 68, 0.05)",
                border: `1px solid ${isDark ? "rgba(239, 68, 68, 0.3)" : "rgba(239, 68, 68, 0.2)"}`,
                color: "#ef4444",
                fontSize: 13,
              }}
            >
              {step.error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Tool result display component
 */
interface ToolResultDisplayProps {
  result: string;
  isDark: boolean;
  colors: ThemeColors;
}

function ToolResultDisplay({ result, isDark, colors }: ToolResultDisplayProps) {
  const [expanded, setExpanded] = useState(false);

  // Try to parse as JSON for better formatting
  let parsedResult: any = null;
  let isJson = false;

  try {
    parsedResult = JSON.parse(result);
    isJson = true;
  } catch {
    parsedResult = result;
  }

  // Generate summary for JSON results
  let summary = "";
  if (isJson && typeof parsedResult === "object") {
    if (parsedResult.success !== undefined) {
      summary = parsedResult.success
        ? parsedResult.message || "Success"
        : `Error: ${parsedResult.error || parsedResult.message || "Failed"}`;
    } else if (Array.isArray(parsedResult)) {
      summary = `${parsedResult.length} items`;
    } else if (parsedResult.total !== undefined) {
      summary = `${parsedResult.total} results`;
    } else {
      summary = "View details";
    }
  } else {
    summary = result.length > 100 ? result.slice(0, 100) + "..." : result;
  }

  const isError =
    isJson &&
    typeof parsedResult === "object" &&
    (parsedResult.success === false || parsedResult.error);

  return (
    <div
      style={{
        borderRadius: 6,
        overflow: "hidden",
        border: `1px solid ${
          isError
            ? isDark
              ? "rgba(239, 68, 68, 0.3)"
              : "rgba(239, 68, 68, 0.2)"
            : isDark
              ? "rgba(255,255,255,0.06)"
              : "rgba(0,0,0,0.06)"
        }`,
      }}
    >
      {/* Summary header */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "8px 10px",
          cursor: "pointer",
          background: isError
            ? isDark
              ? "rgba(239, 68, 68, 0.1)"
              : "rgba(239, 68, 68, 0.05)"
            : isDark
              ? "rgba(255,255,255,0.03)"
              : "rgba(0,0,0,0.02)",
        }}
      >
        {isError ? (
          <CloseCircleFilled style={{ fontSize: 12, color: "#ef4444" }} />
        ) : (
          <CheckCircleFilled style={{ fontSize: 12, color: "#22c55e" }} />
        )}
        <span
          style={{
            flex: 1,
            fontSize: 12,
            color: isError ? "#ef4444" : colors.textSecondary,
          }}
        >
          {summary}
        </span>
        <CaretRightOutlined
          style={{
            fontSize: 10,
            color: colors.textMuted,
            transform: expanded ? "rotate(90deg)" : "rotate(0deg)",
            transition: "transform 0.2s",
          }}
        />
      </div>

      {/* Expanded content */}
      {expanded && (
        <div
          style={{
            padding: "10px 12px",
            borderTop: `1px solid ${isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"}`,
            background: isDark ? "rgba(0,0,0,0.3)" : "rgba(0,0,0,0.02)",
          }}
        >
          <pre
            style={{
              margin: 0,
              fontFamily:
                "'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace",
              fontSize: 11,
              lineHeight: 1.6,
              color: colors.textSecondary,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              maxHeight: 300,
              overflow: "auto",
            }}
          >
            {isJson ? JSON.stringify(parsedResult, null, 2) : result}
          </pre>
        </div>
      )}
    </div>
  );
}
