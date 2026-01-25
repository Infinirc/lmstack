/**
 * Chat Panel Component
 *
 * A slide-out chat panel that can be used from any page.
 * Similar to Cursor's AI chat panel on the right side.
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { Button, Tooltip, Collapse } from "antd";
import {
  CloseOutlined,
  ClearOutlined,
  DownOutlined,
  RobotOutlined,
  UserOutlined,
  BulbOutlined,
  CheckCircleFilled,
  ToolOutlined,
  LoadingOutlined,
} from "@ant-design/icons";
import {
  ChatInput,
  MessageContent,
  getThemeColors,
  getChatStyles,
} from "../chat";
import type { ThemeColors, ChatMessage } from "../chat";
import { ModelSelector } from "./ModelSelector";
import { useChat } from "./useChat";
import { ToolConfirmModal } from "./ToolConfirmModal";
import { TuningJobView } from "./TuningJobView";
import type { ChatModelConfig, CustomEndpoint, ChatPanelState } from "./types";
import {
  DEFAULT_PANEL_WIDTH,
  MIN_PANEL_WIDTH,
  MAX_PANEL_WIDTH,
  CHAT_PANEL_STORAGE_KEY,
} from "./types";
import type { AppColors } from "../../hooks/useTheme";

// Event key for tuning job notifications
export const TUNING_JOB_EVENT_KEY = "lmstack-active-tuning-job";

interface ChatPanelProps {
  isOpen: boolean;
  onClose: () => void;
  onWidthChange?: (width: number) => void;
  isDark: boolean;
  colors: AppColors;
}

/**
 * Load panel state from localStorage
 */
function loadPanelState(): Partial<ChatPanelState> {
  try {
    const saved = localStorage.getItem(CHAT_PANEL_STORAGE_KEY);
    if (saved) {
      return JSON.parse(saved);
    }
  } catch {
    // Ignore parse errors
  }
  return {};
}

/**
 * Save panel state to localStorage
 */
function savePanelState(state: Partial<ChatPanelState>) {
  try {
    const current = loadPanelState();
    localStorage.setItem(
      CHAT_PANEL_STORAGE_KEY,
      JSON.stringify({ ...current, ...state }),
    );
  } catch {
    // Ignore save errors
  }
}

/**
 * Global chat panel component
 */
export function ChatPanel({
  isOpen,
  onClose,
  onWidthChange,
  isDark,
  colors,
}: ChatPanelProps) {
  const chatColors = getThemeColors(isDark);

  // Panel state
  const [width, setWidth] = useState(
    () => loadPanelState().width || DEFAULT_PANEL_WIDTH,
  );
  const [selectedModel, setSelectedModel] = useState<ChatModelConfig | null>(
    () => loadPanelState().selectedModel || null,
  );
  const [customEndpoints, setCustomEndpoints] = useState<CustomEndpoint[]>(
    () => loadPanelState().customEndpoints || [],
  );

  // Chat state
  const [inputValue, setInputValue] = useState("");
  const {
    messages,
    isStreaming,
    isExecutingTool,
    currentToolName,
    pendingTools,
    showConfirmModal,
    systemContext,
    refreshContext,
    sendMessage,
    stopStreaming,
    clearMessages,
    confirmToolExecution,
    cancelToolExecution,
  } = useChat();

  // Active tuning job (shows TuningJobView instead of chat)
  const [activeTuningJobId, setActiveTuningJobId] = useState<number | null>(
    null,
  );

  // Listen for tuning job events
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === TUNING_JOB_EVENT_KEY && e.newValue) {
        try {
          const data = JSON.parse(e.newValue);
          if (data.jobId) {
            setActiveTuningJobId(data.jobId);
          }
        } catch {
          // Ignore
        }
      }
    };

    // Also check on mount
    const checkInitial = () => {
      const stored = localStorage.getItem(TUNING_JOB_EVENT_KEY);
      if (stored) {
        try {
          const data = JSON.parse(stored);
          if (
            data.jobId &&
            data.timestamp &&
            Date.now() - data.timestamp < 5000
          ) {
            setActiveTuningJobId(data.jobId);
            // Clear it after reading
            localStorage.removeItem(TUNING_JOB_EVENT_KEY);
          }
        } catch {
          // Ignore
        }
      }
    };

    checkInitial();
    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, []);

  // Refs
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const resizeHandleRef = useRef<HTMLDivElement>(null);
  const [isResizing, setIsResizing] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const userScrolledUpRef = useRef(false);

  // Persist state changes
  useEffect(() => {
    savePanelState({ selectedModel });
  }, [selectedModel]);

  useEffect(() => {
    savePanelState({ customEndpoints });
  }, [customEndpoints]);

  useEffect(() => {
    savePanelState({ width });
    onWidthChange?.(width);
  }, [width, onWidthChange]);

  // Handle resize
  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      const newWidth = window.innerWidth - e.clientX;
      setWidth(Math.max(MIN_PANEL_WIDTH, Math.min(MAX_PANEL_WIDTH, newWidth)));
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing]);

  // Scroll handling
  const handleScroll = useCallback(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const { scrollTop, scrollHeight, clientHeight } = container;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;

    setShowScrollButton(!isNearBottom);
    if (isNearBottom) {
      userScrolledUpRef.current = false;
    } else if (isStreaming) {
      userScrolledUpRef.current = true;
    }
  }, [isStreaming]);

  const scrollToBottom = useCallback(() => {
    const container = messagesContainerRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
    userScrolledUpRef.current = false;
    setShowScrollButton(false);
  }, []);

  // Auto-scroll during streaming
  useEffect(() => {
    if (!isStreaming) return;

    if (!userScrolledUpRef.current) {
      scrollToBottom();
    }

    const interval = setInterval(() => {
      if (!userScrolledUpRef.current) {
        const container = messagesContainerRef.current;
        if (container) {
          container.scrollTop = container.scrollHeight;
        }
      }
    }, 50);

    return () => clearInterval(interval);
  }, [isStreaming, scrollToBottom]);

  // Scroll on new messages
  useEffect(() => {
    if (!isStreaming && messages.length > 0) {
      scrollToBottom();
    }
  }, [messages.length, isStreaming, scrollToBottom]);

  // Send message handler
  const handleSend = useCallback(() => {
    if (!inputValue.trim() || !selectedModel) return;
    sendMessage(inputValue, selectedModel);
    setInputValue("");
  }, [inputValue, selectedModel, sendMessage]);

  // Handle clear
  const handleClear = useCallback(() => {
    clearMessages();
  }, [clearMessages]);

  if (!isOpen) return null;

  return (
    <>
      {/* Dynamic styles for markdown */}
      <style>{getChatStyles({ isDark, colors: chatColors })}</style>

      {/* Backdrop for mobile */}
      <div
        onClick={onClose}
        style={{
          position: "fixed",
          inset: 0,
          background: "rgba(0, 0, 0, 0.3)",
          zIndex: 999,
          display: "none", // Enable for mobile
        }}
      />

      {/* Panel */}
      <div
        style={{
          position: "fixed",
          top: 0,
          right: 0,
          bottom: 0,
          width,
          background: isDark ? "#0d0d0d" : "#ffffff",
          borderLeft: `1px solid ${colors.border}`,
          zIndex: 1000,
          display: "flex",
          flexDirection: "column",
          boxShadow: isDark
            ? "-4px 0 24px rgba(0, 0, 0, 0.4)"
            : "-4px 0 24px rgba(0, 0, 0, 0.1)",
        }}
      >
        {/* Resize handle */}
        <div
          ref={resizeHandleRef}
          onMouseDown={() => setIsResizing(true)}
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            bottom: 0,
            width: 4,
            cursor: "ew-resize",
            background: isResizing ? colors.accent : "transparent",
            transition: "background 0.15s",
          }}
          onMouseEnter={(e) => {
            if (!isResizing) {
              e.currentTarget.style.background = colors.border;
            }
          }}
          onMouseLeave={(e) => {
            if (!isResizing) {
              e.currentTarget.style.background = "transparent";
            }
          }}
        />

        {/* Header - different for tuning view vs chat */}
        {activeTuningJobId ? (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "12px 16px",
              borderBottom: `1px solid ${colors.border}`,
              flexShrink: 0,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontWeight: 500, color: chatColors.text }}>
                Auto-Tuning Agent
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <Tooltip title="Back to Chat">
                <Button
                  type="text"
                  size="small"
                  onClick={() => setActiveTuningJobId(null)}
                  style={{ color: colors.textMuted, fontSize: 12 }}
                >
                  Back to Chat
                </Button>
              </Tooltip>
              <Tooltip title="Close">
                <Button
                  type="text"
                  size="small"
                  icon={<CloseOutlined />}
                  onClick={onClose}
                  style={{ color: colors.textMuted }}
                />
              </Tooltip>
            </div>
          </div>
        ) : (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "12px 16px",
              borderBottom: `1px solid ${colors.border}`,
              flexShrink: 0,
            }}
          >
            <ModelSelector
              value={selectedModel}
              onChange={setSelectedModel}
              customEndpoints={customEndpoints}
              onCustomEndpointsChange={setCustomEndpoints}
              isDark={isDark}
              colors={colors}
              compact
            />

            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              {/* System status indicator */}
              {systemContext && (
                <Tooltip
                  title={
                    <div style={{ fontSize: 12 }}>
                      <div>Workers: {systemContext.workers.length}</div>
                      <div>
                        Deployments:{" "}
                        {
                          systemContext.deployments.filter(
                            (d) => d.status === "running",
                          ).length
                        }
                      </div>
                      <div>Models: {systemContext.models.length}</div>
                      <div style={{ marginTop: 4, color: "#8c8c8c" }}>
                        Click to refresh
                      </div>
                    </div>
                  }
                >
                  <Button
                    type="text"
                    size="small"
                    icon={
                      <CheckCircleFilled
                        style={{ color: "#52c41a", fontSize: 12 }}
                      />
                    }
                    onClick={refreshContext}
                    style={{ color: colors.textMuted, padding: "0 4px" }}
                  >
                    <span style={{ fontSize: 11, marginLeft: 4 }}>
                      {systemContext.workers.length}W /{" "}
                      {
                        systemContext.deployments.filter(
                          (d) => d.status === "running",
                        ).length
                      }
                      D
                    </span>
                  </Button>
                </Tooltip>
              )}
              {messages.length > 0 && (
                <Tooltip title="Clear chat">
                  <Button
                    type="text"
                    size="small"
                    icon={<ClearOutlined />}
                    onClick={handleClear}
                    style={{ color: colors.textMuted }}
                  />
                </Tooltip>
              )}
              <Tooltip title="Close">
                <Button
                  type="text"
                  size="small"
                  icon={<CloseOutlined />}
                  onClick={onClose}
                  style={{ color: colors.textMuted }}
                />
              </Tooltip>
            </div>
          </div>
        )}

        {/* Content - either TuningJobView or normal chat */}
        {activeTuningJobId ? (
          <TuningJobView jobId={activeTuningJobId} isDark={isDark} />
        ) : (
          <>
            {/* Messages */}
            <div
              ref={messagesContainerRef}
              onScroll={handleScroll}
              style={{
                flex: 1,
                overflow: "auto",
                padding: "16px",
              }}
            >
              {messages.length === 0 ? (
                <EmptyState
                  selectedModel={selectedModel}
                  systemContext={systemContext}
                  colors={chatColors}
                />
              ) : (
                <div
                  style={{ display: "flex", flexDirection: "column", gap: 20 }}
                >
                  {messages.map((msg, index) => {
                    const isLast = index === messages.length - 1;
                    const showStreaming =
                      isLast && isStreaming && msg.role === "assistant";
                    const showToolExecution =
                      isLast && isExecutingTool && msg.role === "assistant";

                    return (
                      <MessageBubble
                        key={msg.id}
                        message={msg}
                        isStreaming={showStreaming}
                        isExecutingTool={showToolExecution}
                        currentToolName={
                          showToolExecution ? currentToolName : null
                        }
                        isDark={isDark}
                        colors={chatColors}
                      />
                    );
                  })}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Scroll to bottom button */}
            {showScrollButton && (
              <div
                style={{
                  position: "absolute",
                  bottom: 100,
                  left: "50%",
                  transform: "translateX(-50%)",
                  zIndex: 10,
                }}
              >
                <Button
                  type="default"
                  shape="circle"
                  size="small"
                  icon={<DownOutlined />}
                  onClick={scrollToBottom}
                  style={{
                    boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
                    background: isDark ? "#27272a" : "#ffffff",
                    borderColor: isDark ? "#3f3f46" : "#e4e4e7",
                  }}
                />
              </div>
            )}

            {/* Input */}
            <div
              style={{
                padding: "12px 16px",
                borderTop: `1px solid ${colors.border}`,
              }}
            >
              <ChatInput
                value={inputValue}
                onChange={setInputValue}
                onSend={handleSend}
                onStop={stopStreaming}
                isStreaming={isStreaming}
                disabled={!selectedModel}
                isDark={isDark}
                colors={chatColors}
              />
            </div>
          </>
        )}
      </div>

      {/* Tool Confirmation Modal */}
      <ToolConfirmModal
        visible={showConfirmModal}
        pendingTools={pendingTools}
        onConfirm={confirmToolExecution}
        onCancel={cancelToolExecution}
        isDark={isDark}
      />
    </>
  );
}

/**
 * Empty state component
 */
interface EmptyStateProps {
  selectedModel: ChatModelConfig | null;
  systemContext: import("./systemContext").SystemContext | null;
  colors: ThemeColors;
}

function EmptyState({ selectedModel, systemContext, colors }: EmptyStateProps) {
  const activeDeployments =
    systemContext?.deployments.filter((d) => d.status === "running") || [];
  const runningContainers =
    systemContext?.containers.filter(
      (c) =>
        c.status.toLowerCase().includes("running") ||
        c.status.toLowerCase().includes("up"),
    ) || [];

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        textAlign: "center",
        padding: "20px",
      }}
    >
      <div
        style={{
          width: 48,
          height: 48,
          borderRadius: "50%",
          background: colors.userBubble,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: 16,
        }}
      >
        <RobotOutlined style={{ fontSize: 24, color: colors.textSecondary }} />
      </div>
      <div
        style={{
          fontSize: 15,
          fontWeight: 500,
          color: colors.text,
          marginBottom: 8,
        }}
      >
        {selectedModel ? "LMStack AI Assistant" : "Select a Model"}
      </div>
      <div style={{ fontSize: 13, color: colors.textMuted, marginBottom: 16 }}>
        {selectedModel
          ? "I can help you manage your LLM infrastructure"
          : "Choose a model from the dropdown above"}
      </div>

      {/* System summary */}
      {selectedModel && systemContext && (
        <div
          style={{
            width: "100%",
            maxWidth: 300,
            padding: 12,
            borderRadius: 8,
            background: colors.userBubble,
            textAlign: "left",
            fontSize: 12,
          }}
        >
          <div style={{ fontWeight: 500, marginBottom: 8, color: colors.text }}>
            System Overview
          </div>
          <div style={{ color: colors.textMuted, lineHeight: 1.8 }}>
            <div>
              • {systemContext.workers.length} Worker
              {systemContext.workers.length !== 1 ? "s" : ""}
            </div>
            <div>
              • {runningContainers.length}/{systemContext.containers.length}{" "}
              Container{systemContext.containers.length !== 1 ? "s" : ""}{" "}
              running
            </div>
            <div>
              • {activeDeployments.length} Model deployment
              {activeDeployments.length !== 1 ? "s" : ""} active
            </div>
            <div>
              • {systemContext.models.length} Model
              {systemContext.models.length !== 1 ? "s" : ""} available
            </div>
            <div>
              • {systemContext.images.length} Docker image
              {systemContext.images.length !== 1 ? "s" : ""}
            </div>
          </div>
          <div style={{ marginTop: 12, fontSize: 11, color: colors.textMuted }}>
            Try: "有幾個容器在運行？" or "GPU 記憶體剩多少？"
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Format tool name for display
 */
function formatToolName(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Message bubble component
 */
interface MessageBubbleProps {
  message: ChatMessage;
  isStreaming: boolean;
  isExecutingTool: boolean;
  currentToolName: string | null;
  isDark: boolean;
  colors: ThemeColors;
}

function MessageBubble({
  message,
  isStreaming,
  isExecutingTool,
  currentToolName,
  isDark,
  colors,
}: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        gap: 8,
      }}
    >
      {/* Avatar */}
      {!isUser && (
        <div
          style={{
            width: 28,
            height: 28,
            borderRadius: "50%",
            background: isDark ? "#3f3f46" : "#e4e4e7",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
            marginTop: 2,
          }}
        >
          <RobotOutlined
            style={{ fontSize: 12, color: colors.textSecondary }}
          />
        </div>
      )}

      {/* Content */}
      <div
        style={{
          maxWidth: "85%",
          padding: isUser ? "10px 14px" : "0",
          borderRadius: isUser ? 18 : 0,
          background: isUser ? colors.userBubble : "transparent",
        }}
      >
        {isUser ? (
          <div
            style={{
              fontSize: 14,
              lineHeight: 1.5,
              color: colors.text,
              whiteSpace: "pre-wrap",
            }}
          >
            {message.content}
          </div>
        ) : (
          <>
            {/* Thinking section for reasoning models */}
            {message.thinking && (
              <Collapse
                size="small"
                ghost
                style={{ marginBottom: 8 }}
                items={[
                  {
                    key: "thinking",
                    label: (
                      <span style={{ fontSize: 12, color: colors.textMuted }}>
                        <BulbOutlined style={{ marginRight: 6 }} />
                        Thinking Process
                      </span>
                    ),
                    children: (
                      <div
                        style={{
                          fontSize: 13,
                          color: colors.textSecondary,
                          whiteSpace: "pre-wrap",
                          lineHeight: 1.6,
                          maxHeight: 300,
                          overflow: "auto",
                          padding: "8px 0",
                        }}
                      >
                        {message.thinking}
                      </div>
                    ),
                  },
                ]}
              />
            )}

            {/* Tool execution indicator - show when executing */}
            {isExecutingTool && currentToolName && (
              <div
                style={{
                  marginBottom: 12,
                  padding: "10px 12px",
                  borderRadius: 8,
                  background: isDark
                    ? "rgba(24, 144, 255, 0.1)"
                    : "rgba(24, 144, 255, 0.08)",
                  border: `1px solid ${isDark ? "rgba(24, 144, 255, 0.3)" : "rgba(24, 144, 255, 0.2)"}`,
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <LoadingOutlined
                  style={{ color: "#1890ff", fontSize: 14 }}
                  spin
                />
                <span
                  style={{ fontSize: 13, color: "#1890ff", fontWeight: 500 }}
                >
                  Executing: {formatToolName(currentToolName)}
                </span>
              </div>
            )}

            {/* Tool calls display - always show in history */}
            {message.toolCalls &&
              message.toolCalls.length > 0 &&
              !isExecutingTool && (
                <div
                  style={{
                    marginBottom: 12,
                    padding: "10px 12px",
                    borderRadius: 8,
                    background: isDark
                      ? "rgba(82, 196, 26, 0.1)"
                      : "rgba(82, 196, 26, 0.08)",
                    border: `1px solid ${isDark ? "rgba(82, 196, 26, 0.3)" : "rgba(82, 196, 26, 0.2)"}`,
                  }}
                >
                  <div
                    style={{
                      fontSize: 12,
                      color: "#52c41a",
                      fontWeight: 500,
                      marginBottom: 8,
                    }}
                  >
                    <ToolOutlined style={{ marginRight: 6 }} />
                    Tool Calls Executed
                  </div>
                  {message.toolCalls.map((tc) => {
                    let args: Record<string, any> = {};
                    try {
                      args = JSON.parse(tc.function.arguments);
                    } catch {
                      args = {};
                    }
                    return (
                      <div
                        key={tc.id}
                        style={{
                          fontSize: 12,
                          color: colors.textSecondary,
                          marginTop: 6,
                          padding: "6px 8px",
                          borderRadius: 4,
                          background: isDark
                            ? "rgba(255,255,255,0.03)"
                            : "rgba(0,0,0,0.02)",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 6,
                          }}
                        >
                          <CheckCircleFilled
                            style={{ color: "#52c41a", fontSize: 12 }}
                          />
                          <span style={{ fontWeight: 500, color: colors.text }}>
                            {formatToolName(tc.function.name)}
                          </span>
                        </div>
                        {Object.keys(args).length > 0 && (
                          <div
                            style={{
                              marginTop: 4,
                              marginLeft: 18,
                              fontSize: 11,
                              color: colors.textMuted,
                            }}
                          >
                            {Object.entries(args).map(([key, value]) => (
                              <div key={key}>
                                {key}:{" "}
                                {typeof value === "object"
                                  ? JSON.stringify(value)
                                  : String(value)}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

            <MessageContent
              content={message.content}
              isStreaming={isStreaming}
              isDark={isDark}
              colors={colors}
            />
            {message.model && (
              <div
                style={{ marginTop: 6, fontSize: 11, color: colors.textMuted }}
              >
                via {message.model}
              </div>
            )}
          </>
        )}
      </div>

      {/* User avatar */}
      {isUser && (
        <div
          style={{
            width: 28,
            height: 28,
            borderRadius: "50%",
            background: isDark ? "#3f3f46" : "#e4e4e7",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
            marginTop: 2,
          }}
        >
          <UserOutlined style={{ fontSize: 12, color: colors.textSecondary }} />
        </div>
      )}
    </div>
  );
}
