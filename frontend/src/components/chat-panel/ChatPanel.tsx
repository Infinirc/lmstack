/**
 * Chat Panel Component
 *
 * A slide-out chat panel that can be used from any page.
 * Uses MCP-based agent with Claude Code-style interaction.
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Button, Tooltip } from "antd";
import { CloseOutlined, ClearOutlined, DownOutlined } from "@ant-design/icons";
import { ChatInput, getThemeColors, getChatStyles } from "../chat";
import { ModelSelector } from "./ModelSelector";
import { useAgentChat } from "./useAgentChat";
import { AgentChatView } from "./AgentChatView";
import type { ChatModelConfig, CustomEndpoint, ChatPanelState } from "./types";
import {
  DEFAULT_PANEL_WIDTH,
  MIN_PANEL_WIDTH,
  MAX_PANEL_WIDTH,
  CHAT_PANEL_STORAGE_KEY,
} from "./types";
import type { AppColors } from "../../hooks/useTheme";
import { useChatPanel } from "../../contexts/ChatPanelContext";

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
  const navigate = useNavigate();

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

  // Input state
  const [inputValue, setInputValue] = useState("");

  // Agent chat state
  const {
    messages: agentMessages,
    currentSteps,
    isStreaming: agentIsStreaming,
    isThinking,
    currentTool,
    sendMessage: agentSendMessage,
    stopStreaming: agentStopStreaming,
    clearMessages: agentClearMessages,
    toggleStepExpanded,
    startNewConversation,
  } = useAgentChat();

  // Chat panel context - for external access
  const { _registerSendFunction } = useChatPanel();

  // Register send function with context so other pages can send messages
  useEffect(() => {
    _registerSendFunction(agentSendMessage, selectedModel);
    return () => {
      _registerSendFunction(null, null);
    };
  }, [_registerSendFunction, agentSendMessage, selectedModel]);

  // Derived state
  const isStreaming = agentIsStreaming;
  const hasMessages = agentMessages.length > 0;

  // Handle send message
  const handleSendMessage = useCallback(() => {
    if (!inputValue.trim() || !selectedModel) return;
    agentSendMessage(inputValue, selectedModel);
    setInputValue("");
  }, [inputValue, selectedModel, agentSendMessage]);

  // Handle stop streaming
  const handleStopStreaming = useCallback(() => {
    agentStopStreaming();
  }, [agentStopStreaming]);

  // Handle clear messages
  const handleClearMessages = useCallback(() => {
    agentClearMessages();
  }, [agentClearMessages]);

  // Handle sending message from action suggestions
  const handleAgentSendSuggestion = useCallback(
    (message: string) => {
      if (!selectedModel || agentIsStreaming) return;
      agentSendMessage(message, selectedModel);
    },
    [selectedModel, agentIsStreaming, agentSendMessage],
  );

  // Handle model change - start new conversation when model changes
  const handleModelChange = useCallback(
    (model: ChatModelConfig | null) => {
      // If model type or identity changes, start a new conversation
      const modelChanged =
        selectedModel?.type !== model?.type ||
        selectedModel?.deploymentId !== model?.deploymentId ||
        selectedModel?.endpoint !== model?.endpoint;

      if (modelChanged && agentMessages.length > 0) {
        startNewConversation();
      }
      setSelectedModel(model);
    },
    [selectedModel, agentMessages.length, startNewConversation],
  );

  // Refs
  const messagesContainerRef = useRef<HTMLDivElement>(null);
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

  // Scroll handling - track user scroll intent
  const handleScroll = useCallback(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const { scrollTop, scrollHeight, clientHeight } = container;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;

    setShowScrollButton(!isNearBottom);

    // If user scrolls up during streaming, mark it
    if (!isNearBottom) {
      userScrolledUpRef.current = true;
    } else {
      userScrolledUpRef.current = false;
    }
  }, []);

  const scrollToBottom = useCallback(() => {
    const container = messagesContainerRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
    userScrolledUpRef.current = false;
    setShowScrollButton(false);
  }, []);

  // Auto-scroll during streaming - only scroll if user hasn't scrolled up
  useEffect(() => {
    if (!isStreaming) return;

    // Only auto-scroll if user hasn't manually scrolled up
    if (!userScrolledUpRef.current) {
      scrollToBottom();
    }

    // Use a longer interval and check user intent
    const interval = setInterval(() => {
      if (!userScrolledUpRef.current) {
        const container = messagesContainerRef.current;
        if (container) {
          const { scrollTop, scrollHeight, clientHeight } = container;
          const isNearBottom = scrollHeight - scrollTop - clientHeight < 150;
          // Only scroll if already near bottom
          if (isNearBottom) {
            container.scrollTop = container.scrollHeight;
          }
        }
      }
    }, 200); // Slower interval for less aggressive scrolling

    return () => clearInterval(interval);
  }, [isStreaming, scrollToBottom]);

  // Scroll on new messages
  useEffect(() => {
    if (!isStreaming && agentMessages.length > 0) {
      scrollToBottom();
    }
  }, [agentMessages.length, isStreaming, scrollToBottom]);

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

        {/* Header with mode toggle */}
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
          {/* Left side: Model selector */}
          <ModelSelector
            value={selectedModel}
            onChange={handleModelChange}
            customEndpoints={customEndpoints}
            onCustomEndpointsChange={setCustomEndpoints}
            isDark={isDark}
            colors={colors}
            compact
          />

          {/* Right side: Actions */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {/* Clear button */}
            {hasMessages && (
              <Tooltip title="Clear chat">
                <Button
                  type="text"
                  size="small"
                  icon={<ClearOutlined />}
                  onClick={handleClearMessages}
                  style={{ color: colors.textMuted }}
                />
              </Tooltip>
            )}

            {/* Close button */}
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

        {/* Content - Agent mode or Chat mode */}
        <>
          {/* Messages area */}
          <div
            ref={messagesContainerRef}
            onScroll={handleScroll}
            style={{
              flex: 1,
              overflow: "auto",
              overflowY: "scroll",
              padding: "16px",
              WebkitOverflowScrolling: "touch",
              position: "relative",
            }}
          >
            <AgentChatView
              messages={agentMessages}
              currentSteps={currentSteps}
              isStreaming={agentIsStreaming}
              isThinking={isThinking}
              currentTool={currentTool}
              onToggleStep={toggleStepExpanded}
              onNavigate={navigate}
              onSendMessage={handleAgentSendSuggestion}
              isDark={isDark}
              colors={colors}
              userScrolledUp={showScrollButton}
            />
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
              onSend={handleSendMessage}
              onStop={handleStopStreaming}
              isStreaming={isStreaming}
              disabled={!selectedModel}
              isDark={isDark}
              colors={chatColors}
            />
          </div>
        </>
      </div>
    </>
  );
}
