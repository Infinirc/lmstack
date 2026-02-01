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
  UserOutlined,
  LoadingOutlined,
  CheckCircleFilled,
  CloseCircleFilled,
  CaretRightOutlined,
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
    return (
      <EmptyState
        colors={chatColors}
        isDark={isDark}
        onSendMessage={onSendMessage}
      />
    );
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
  onSendMessage,
}: {
  colors: ThemeColors;
  isDark: boolean;
  onSendMessage?: (message: string) => void;
}) {
  const isDark = _isDark;

  const quickActions = [
    { icon: <ClusterOutlined />, text: "列出所有 Worker 狀態" },
    { icon: <DashboardOutlined />, text: "GPU 記憶體使用狀況" },
    { icon: <ContainerOutlined />, text: "列出所有容器" },
  ];

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
          width: 44,
          height: 44,
          borderRadius: 10,
          background: isDark ? "#27272a" : "#f4f4f5",
          border: `1px solid ${isDark ? "#3f3f46" : "#e4e4e7"}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: 16,
        }}
      >
        <ToolOutlined style={{ fontSize: 20, color: colors.textSecondary }} />
      </div>
      <div
        style={{
          fontSize: 15,
          fontWeight: 500,
          color: colors.text,
          marginBottom: 6,
        }}
      >
        Agent
      </div>
      <div
        style={{
          fontSize: 13,
          color: colors.textMuted,
          marginBottom: 20,
          maxWidth: 260,
          lineHeight: 1.5,
        }}
      >
        部署模型、執行基準測試、查詢系統狀態
      </div>

      {/* Quick actions */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 8,
          width: "100%",
          maxWidth: 280,
        }}
      >
        {quickActions.map((item, idx) => (
          <div
            key={idx}
            onClick={() => onSendMessage?.(item.text)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "10px 14px",
              borderRadius: 8,
              background: isDark
                ? "rgba(255,255,255,0.03)"
                : "rgba(0,0,0,0.02)",
              border: `1px solid ${isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"}`,
              cursor: "pointer",
              transition: "background 0.15s",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = isDark
                ? "rgba(255,255,255,0.06)"
                : "rgba(0,0,0,0.04)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = isDark
                ? "rgba(255,255,255,0.03)"
                : "rgba(0,0,0,0.02)";
            }}
          >
            <div style={{ color: colors.textMuted, fontSize: 14 }}>
              {item.icon}
            </div>
            <div style={{ fontSize: 13, color: colors.textSecondary }}>
              {item.text}
            </div>
          </div>
        ))}
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
          width: 28,
          height: 28,
          borderRadius: 6,
          background: isDark ? "#27272a" : "#f4f4f5",
          border: `1px solid ${isDark ? "#3f3f46" : "#e4e4e7"}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
          marginTop: 2,
        }}
      >
        <ToolOutlined style={{ fontSize: 13, color: colors.textSecondary }} />
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
              gap: 8,
              padding: "10px 12px",
              borderRadius: 8,
              background: isDark
                ? "rgba(255,255,255,0.03)"
                : "rgba(0,0,0,0.02)",
              border: `1px solid ${isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"}`,
              marginBottom: 12,
            }}
          >
            <LoadingOutlined
              style={{ color: colors.textMuted, fontSize: 12 }}
              spin
            />
            <span style={{ fontSize: 13, color: colors.textMuted }}>
              處理中...
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
    const iconStyle = { fontSize: 14, color: colors.textMuted };
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
        padding: "8px 12px",
        borderRadius: 6,
        background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)",
        border: `1px solid ${
          isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"
        }`,
        cursor: "pointer",
        transition: "background 0.15s",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = isDark
          ? "rgba(255,255,255,0.06)"
          : "rgba(0,0,0,0.04)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = isDark
          ? "rgba(255,255,255,0.03)"
          : "rgba(0,0,0,0.02)";
      }}
    >
      {getIcon()}
      <div style={{ flex: 1 }}>
        <div
          style={{
            fontSize: 13,
            fontWeight: 500,
            color: colors.text,
          }}
        >
          {reference.title}
        </div>
        {reference.description && (
          <div style={{ fontSize: 11, color: colors.textMuted, marginTop: 1 }}>
            {reference.description}
          </div>
        )}
      </div>
      <RightOutlined style={{ fontSize: 10, color: colors.textMuted }} />
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
    const iconStyle = { fontSize: 12 };
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
        return <RightOutlined style={iconStyle} />;
    }
  };

  const isDanger = suggestion.type === "danger";

  return (
    <div
      onClick={onClick}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "6px 12px",
        borderRadius: 6,
        background: isDanger
          ? isDark
            ? "rgba(239, 68, 68, 0.1)"
            : "rgba(239, 68, 68, 0.06)"
          : isDark
            ? "rgba(255, 255, 255, 0.04)"
            : "rgba(0, 0, 0, 0.03)",
        border: `1px solid ${
          isDanger
            ? isDark
              ? "rgba(239, 68, 68, 0.25)"
              : "rgba(239, 68, 68, 0.2)"
            : isDark
              ? "rgba(255, 255, 255, 0.08)"
              : "rgba(0, 0, 0, 0.08)"
        }`,
        color: isDanger ? "#ef4444" : colors.textSecondary,
        cursor: "pointer",
        transition: "background 0.15s",
        fontSize: 12,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = isDanger
          ? isDark
            ? "rgba(239, 68, 68, 0.15)"
            : "rgba(239, 68, 68, 0.1)"
          : isDark
            ? "rgba(255, 255, 255, 0.08)"
            : "rgba(0, 0, 0, 0.06)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = isDanger
          ? isDark
            ? "rgba(239, 68, 68, 0.1)"
            : "rgba(239, 68, 68, 0.06)"
          : isDark
            ? "rgba(255, 255, 255, 0.04)"
            : "rgba(0, 0, 0, 0.03)";
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
          borderRadius: 8,
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
            borderBottom: step.expanded
              ? `1px solid ${isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"}`
              : "none",
          }}
        >
          <BulbOutlined style={{ color: colors.textMuted, fontSize: 13 }} />
          <span
            style={{
              flex: 1,
              fontSize: 13,
              fontWeight: 500,
              color: colors.textSecondary,
            }}
          >
            {step.title}
          </span>
          {step.status === "running" && (
            <LoadingOutlined
              style={{ fontSize: 12, color: colors.textMuted }}
            />
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
