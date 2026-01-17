/**
 * Chat Page
 *
 * Modern chat interface with streaming responses, thinking state visualization,
 * and markdown rendering with code highlighting.
 *
 * @module pages/Chat
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { Button, message, Tooltip, Dropdown, Popconfirm } from "antd";
import Loading from "../components/Loading";
import {
  DeleteOutlined,
  RobotOutlined,
  UserOutlined,
  DownOutlined,
  MessageOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
} from "@ant-design/icons";
import {
  useTheme,
  getChatStyles,
  ChatInput,
  MessageContent,
  generateMessageId,
} from "../components/chat";
import { deploymentsApi, conversationsApi } from "../services/api";
import type { Deployment, ChatMessage } from "../types";
import type {
  Conversation as ApiConversation,
  ConversationMessage,
} from "../services/api";
import { useResponsive } from "../hooks";

// Conversation type for UI (maps from API type)
interface Conversation {
  id: number;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

// Convert API conversation to UI conversation
const convertApiConversation = (apiConv: ApiConversation): Conversation => ({
  id: apiConv.id,
  title: apiConv.title,
  createdAt: new Date(apiConv.created_at),
  updatedAt: new Date(apiConv.updated_at),
  messages: (apiConv.messages || []).map((m: ConversationMessage) => ({
    id: `msg_${m.id}`,
    role: m.role,
    content: m.content,
    thinking: m.thinking,
    timestamp: new Date(m.created_at),
  })),
});

/**
 * Suggestions for empty chat state
 */
const SUGGESTIONS = [
  "Explain what machine learning is",
  "Write a Python sorting algorithm",
  "How to design a RESTful API?",
  "Analyze the time complexity of this code",
];

/**
 * Chat page component
 */
export default function Chat() {
  const { isDark, colors } = useTheme();
  const { isMobile } = useResponsive();

  // Conversation state
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<
    number | null
  >(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true); // Default collapsed
  const [_conversationsLoading, setConversationsLoading] = useState(true);

  // Auto-collapse sidebar on mobile
  useEffect(() => {
    setSidebarCollapsed(isMobile);
  }, [isMobile]);

  // State
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [selectedDeployment, setSelectedDeployment] =
    useState<Deployment | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [loading, setLoading] = useState(true);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const userScrolledUpRef = useRef(false);
  const [showScrollButton, setShowScrollButton] = useState(false);

  // Load conversations from API on mount
  useEffect(() => {
    const loadConversations = async () => {
      try {
        const response = await conversationsApi.list({ limit: 100 });
        setConversations(response.items.map(convertApiConversation));
      } catch (error) {
        console.error("Failed to load conversations:", error);
      } finally {
        setConversationsLoading(false);
      }
    };
    loadConversations();
  }, []);

  // Load conversation messages when switching conversations
  useEffect(() => {
    if (isStreaming) return; // Don't sync during streaming
    if (currentConversationId) {
      const loadMessages = async () => {
        try {
          const conv = await conversationsApi.get(currentConversationId);
          const converted = convertApiConversation(conv);
          setMessages(converted.messages);
          // Update the conversation in the list with full message data
          setConversations((prev) =>
            prev.map((c) => (c.id === currentConversationId ? converted : c)),
          );
        } catch (error) {
          console.error("Failed to load conversation:", error);
          setMessages([]);
        }
      };
      loadMessages();
    } else {
      setMessages([]);
    }
  }, [currentConversationId, isStreaming]);

  // Delete conversation
  const deleteConversation = useCallback(
    async (id: number) => {
      try {
        await conversationsApi.delete(id);
        setConversations((prev) => prev.filter((c) => c.id !== id));
        if (currentConversationId === id) {
          setCurrentConversationId(null);
          setMessages([]);
        }
      } catch (error) {
        console.error("Failed to delete conversation:", error);
        message.error("Failed to delete conversation");
      }
    },
    [currentConversationId],
  );

  // Clear all conversations
  const clearAllConversations = useCallback(async () => {
    try {
      await conversationsApi.clearAll();
      setConversations([]);
      setCurrentConversationId(null);
      setMessages([]);
    } catch (error) {
      console.error("Failed to clear conversations:", error);
      message.error("Failed to clear conversations");
    }
  }, []);

  /**
   * Fetch available deployments
   */
  const fetchDeployments = useCallback(async () => {
    try {
      const response = await deploymentsApi.list({ status: "running" });
      setDeployments(response.items);

      // Update or clear selectedDeployment based on current running deployments
      if (selectedDeployment) {
        // Check if selected deployment is still running and update its data (port may have changed)
        const updated = response.items.find(
          (d) => d.id === selectedDeployment.id,
        );
        if (updated) {
          // Update with latest data (port, status, etc.)
          if (updated.port !== selectedDeployment.port) {
            setSelectedDeployment(updated);
          }
        } else {
          // Selected deployment is no longer running
          setSelectedDeployment(
            response.items.length > 0 ? response.items[0] : null,
          );
        }
      } else if (response.items.length > 0) {
        setSelectedDeployment(response.items[0]);
      }
    } catch (error) {
      console.error("Failed to fetch deployments:", error);
    } finally {
      setLoading(false);
    }
  }, [selectedDeployment]);

  useEffect(() => {
    fetchDeployments();
    const interval = setInterval(fetchDeployments, 10000);
    return () => clearInterval(interval);
  }, [fetchDeployments]);

  // Handle scroll detection - check if user scrolled up
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

  // Scroll to bottom function
  const scrollToBottom = useCallback(() => {
    const container = messagesContainerRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
    userScrolledUpRef.current = false;
    setShowScrollButton(false);
  }, []);

  // Auto-scroll during streaming - use interval for consistent scrolling
  useEffect(() => {
    if (!isStreaming) return;

    // Scroll immediately when streaming starts
    if (!userScrolledUpRef.current) {
      scrollToBottom();
    }

    // Keep scrolling during streaming
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

  // Auto-scroll when new message is added (not during streaming)
  useEffect(() => {
    if (!isStreaming && messages.length > 0) {
      scrollToBottom();
    }
  }, [messages.length, isStreaming, scrollToBottom]);

  // Reset scroll flag when streaming stops
  useEffect(() => {
    if (!isStreaming) {
      userScrolledUpRef.current = false;
    }
  }, [isStreaming]);

  /**
   * Get endpoint URL for deployment (uses backend proxy to handle Docker networking)
   */
  const getEndpointUrl = (deployment: Deployment): string | null => {
    if (deployment.status !== "running") {
      return null;
    }
    // Use backend proxy endpoint instead of direct model URL
    // This handles Docker internal networking correctly (especially on Windows)
    return `/api/deployments/${deployment.id}/chat`;
  };

  /**
   * Send message to API
   */
  const handleSend = useCallback(
    async (content?: string) => {
      const messageContent = content || inputValue.trim();
      if (!messageContent || !selectedDeployment || isStreaming) return;

      const endpoint = getEndpointUrl(selectedDeployment);
      if (!endpoint) {
        message.error(
          "Deployment is not ready. Please wait for it to be running.",
        );
        return;
      }

      // Set streaming first to prevent useEffect from clearing messages
      setIsStreaming(true);
      setInputValue("");

      // Create new conversation if none exists
      let convId = currentConversationId;
      if (!convId) {
        try {
          const title =
            messageContent.slice(0, 30) +
            (messageContent.length > 30 ? "..." : "");
          const newConv = await conversationsApi.create({
            title,
            deployment_id: selectedDeployment.id,
          });
          const converted = convertApiConversation(newConv);
          setConversations((prev) => [converted, ...prev]);
          setCurrentConversationId(converted.id);
          convId = converted.id;
        } catch (error) {
          console.error("Failed to create conversation:", error);
          message.error("Failed to create conversation");
          setIsStreaming(false);
          return;
        }
      }

      const userMessage: ChatMessage = {
        id: generateMessageId(),
        role: "user",
        content: messageContent,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);

      const assistantMessage: ChatMessage = {
        id: generateMessageId(),
        role: "assistant",
        content: "",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);

      try {
        abortControllerRef.current = new AbortController();

        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: selectedDeployment.model?.model_id || "default",
            messages: [
              ...messages.map((m) => ({ role: m.role, content: m.content })),
              { role: "user", content: messageContent },
            ],
            stream: true,
            temperature: 0.7,
            // Don't specify max_tokens - let the server decide based on available context
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(
            `API error: ${response.status} ${response.statusText}`,
          );
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) throw new Error("No response body");

        let accumulatedContent = "";

        // eslint-disable-next-line no-constant-condition
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk
            .split("\n")
            .filter((line) => line.trim().startsWith("data:"));

          for (const line of lines) {
            const data = line.replace("data: ", "").trim();
            if (data === "[DONE]") continue;

            try {
              const parsed = JSON.parse(data);
              const deltaContent = parsed.choices?.[0]?.delta?.content || "";
              accumulatedContent += deltaContent;

              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMessage.id
                    ? { ...m, content: accumulatedContent }
                    : m,
                ),
              );
            } catch {
              // Skip invalid JSON
            }
          }
        }

        // Save messages to database after streaming completes
        if (convId && accumulatedContent) {
          try {
            await conversationsApi.addMessages(convId, {
              messages: [
                { role: "user", content: messageContent },
                { role: "assistant", content: accumulatedContent },
              ],
            });
            // Update local state
            setConversations((prev) =>
              prev.map((c) =>
                c.id === convId ? { ...c, updatedAt: new Date() } : c,
              ),
            );
          } catch (error) {
            console.error("Failed to save messages:", error);
          }
        }
      } catch (error: unknown) {
        const err = error as Error;
        if (err.name === "AbortError") {
          message.info("Generation stopped");
          // Still save the user message if we have a conversation
          if (convId) {
            try {
              await conversationsApi.addMessages(convId, {
                messages: [{ role: "user", content: messageContent }],
              });
            } catch (saveError) {
              console.error("Failed to save user message:", saveError);
            }
          }
        } else {
          message.error(`Error: ${err.message}`);
          setMessages((prev) =>
            prev.filter((m) => m.id !== assistantMessage.id),
          );
        }
      } finally {
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [
      inputValue,
      selectedDeployment,
      isStreaming,
      messages,
      currentConversationId,
    ],
  );

  /**
   * Stop streaming generation
   */
  const handleStop = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

  /**
   * Start new chat (clear current and create new)
   */
  const handleNewChat = useCallback(() => {
    handleStop();
    setCurrentConversationId(null);
    setMessages([]);
  }, [handleStop]);

  // Loading state
  if (loading) {
    return (
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "60vh",
        }}
      >
        <Loading size="large" />
      </div>
    );
  }

  // Model dropdown menu items
  const modelMenuItems = deployments.map((d) => ({
    key: d.id.toString(),
    label: (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "4px 0",
        }}
      >
        <RobotOutlined />
        <span>{d.model?.name || d.name}</span>
        <span style={{ color: colors.textMuted, fontSize: 12 }}>
          @{d.worker?.name}
        </span>
      </div>
    ),
    onClick: () => {
      const deployment = deployments.find((dep) => dep.id === d.id);
      setSelectedDeployment(deployment || null);
    },
  }));

  return (
    <div
      className="chat-page"
      style={{
        display: "flex",
        height: "calc(100vh - 100px)",
        width: "calc(100% + 48px)",
        margin: "0 -24px -24px -24px",
        padding: "0",
      }}
    >
      {/* Sidebar */}
      <ChatSidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={setCurrentConversationId}
        onNewChat={handleNewChat}
        onDeleteConversation={deleteConversation}
        onClearAll={clearAllConversations}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        isDark={isDark}
        colors={colors}
        isMobile={isMobile}
      />

      {/* Main Chat Area */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
        }}
      >
        {/* Header */}
        <ChatHeader
          selectedDeployment={selectedDeployment}
          modelMenuItems={modelMenuItems}
          onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
          sidebarCollapsed={sidebarCollapsed}
          colors={colors}
        />

        {/* Messages Area */}
        <div
          ref={messagesContainerRef}
          className="messages-container"
          onScroll={handleScroll}
          style={{
            flex: 1,
            overflow: "auto",
            paddingBottom: 8,
            width: "100%",
          }}
        >
          {messages.length === 0 ? (
            <EmptyState
              selectedDeployment={selectedDeployment}
              onSend={handleSend}
              colors={colors}
              isMobile={isMobile}
            />
          ) : (
            <MessageList
              messages={messages}
              isStreaming={isStreaming}
              isDark={isDark}
              colors={colors}
              messagesEndRef={messagesEndRef}
            />
          )}
        </div>

        {/* Input Area */}
        <div style={{ padding: "0", width: "100%", position: "relative" }}>
          {/* Scroll to bottom button */}
          {showScrollButton && (
            <div
              style={{
                position: "absolute",
                top: -48,
                left: "50%",
                transform: "translateX(-50%)",
                zIndex: 10,
              }}
            >
              <Button
                type="default"
                shape="circle"
                icon={<DownOutlined />}
                onClick={scrollToBottom}
                style={{
                  width: 36,
                  height: 36,
                  boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
                  background: isDark ? "#27272a" : "#ffffff",
                  borderColor: isDark ? "#3f3f46" : "#e4e4e7",
                }}
              />
            </div>
          )}
          <div style={{ maxWidth: 900, margin: "0 auto", padding: "0 16px" }}>
            <ChatInput
              value={inputValue}
              onChange={setInputValue}
              onSend={() => handleSend()}
              onStop={handleStop}
              isStreaming={isStreaming}
              disabled={!selectedDeployment}
              isDark={isDark}
              colors={colors}
            />
          </div>
        </div>
      </div>

      {/* Dynamic Styles */}
      <style>{getChatStyles({ isDark, colors })}</style>
    </div>
  );
}

/**
 * Chat header with model selector and action buttons
 */
interface ChatHeaderProps {
  selectedDeployment: Deployment | null;
  modelMenuItems: {
    key: string;
    label: React.ReactNode;
    onClick: () => void;
  }[];
  onToggleSidebar: () => void;
  sidebarCollapsed: boolean;
  colors: ReturnType<typeof useTheme>["colors"];
}

function ChatHeader({
  selectedDeployment,
  modelMenuItems,
  onToggleSidebar,
  sidebarCollapsed,
  colors,
}: ChatHeaderProps) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 16px 8px 16px",
        width: "100%",
      }}
    >
      {/* Left side */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {/* Toggle sidebar button */}
        <Tooltip title={sidebarCollapsed ? "Show sidebar" : "Hide sidebar"}>
          <Button
            type="text"
            icon={
              sidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />
            }
            onClick={onToggleSidebar}
            style={{
              width: 36,
              height: 36,
              borderRadius: 10,
              color: colors.textSecondary,
            }}
          />
        </Tooltip>

        {/* Model Selector */}
        <Dropdown menu={{ items: modelMenuItems }} trigger={["click"]}>
          <Button
            type="text"
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              height: 40,
              padding: "0 12px",
              borderRadius: 10,
              background: colors.hoverBg,
              border: "none",
              fontSize: 14,
              fontWeight: 500,
              color: colors.text,
            }}
          >
            <RobotOutlined style={{ fontSize: 16 }} />
            <span>{selectedDeployment?.model?.name || "Select Model"}</span>
            <DownOutlined style={{ fontSize: 10, color: colors.textMuted }} />
          </Button>
        </Dropdown>
      </div>
    </div>
  );
}

/**
 * Empty state with suggestions
 */
interface EmptyStateProps {
  selectedDeployment: Deployment | null;
  onSend: (content: string) => void;
  colors: ReturnType<typeof useTheme>["colors"];
  isMobile?: boolean;
}

function EmptyState({
  selectedDeployment,
  onSend,
  colors,
  isMobile,
}: EmptyStateProps) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        padding: isMobile ? "20px 16px" : "40px 20px",
        maxWidth: 900,
        margin: "0 auto",
      }}
    >
      <h2
        style={{
          fontSize: isMobile ? 22 : 28,
          fontWeight: 500,
          color: colors.text,
          margin: "0 0 8px 0",
          letterSpacing: "-0.02em",
          textAlign: "center",
        }}
      >
        {selectedDeployment
          ? `Chat with ${selectedDeployment.model?.name || "AI"}`
          : "Select a model to start"}
      </h2>
      <p
        style={{
          fontSize: isMobile ? 13 : 15,
          color: colors.textMuted,
          margin: isMobile ? "0 0 24px 0" : "0 0 40px 0",
          textAlign: "center",
        }}
      >
        {selectedDeployment
          ? "Ask me anything"
          : "Choose a deployed model from the dropdown above"}
      </p>

      {selectedDeployment && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: isMobile ? "1fr" : "repeat(2, 1fr)",
            gap: isMobile ? 8 : 12,
            width: "100%",
            maxWidth: 700,
          }}
        >
          {SUGGESTIONS.map((suggestion, idx) => (
            <button
              key={idx}
              onClick={() => onSend(suggestion)}
              className="suggestion-btn"
              style={{
                padding: isMobile ? "12px 16px" : "16px 20px",
                borderRadius: isMobile ? 12 : 16,
                border: `1px solid ${colors.border}`,
                background: "transparent",
                color: colors.textSecondary,
                fontSize: isMobile ? 13 : 14,
                textAlign: "left",
                cursor: "pointer",
                transition: "all 0.15s ease",
                lineHeight: 1.5,
                animationDelay: `${idx * 50}ms`,
              }}
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Message list component
 */
interface MessageListProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  isDark: boolean;
  colors: ReturnType<typeof useTheme>["colors"];
  messagesEndRef: React.RefObject<HTMLDivElement>;
}

function MessageList({
  messages,
  isStreaming,
  isDark,
  colors,
  messagesEndRef,
}: MessageListProps) {
  return (
    <div
      style={{
        maxWidth: 900,
        margin: "0 auto",
        display: "flex",
        flexDirection: "column",
        gap: 24,
      }}
    >
      {messages.map((msg, index) => {
        const isLast = index === messages.length - 1;
        const showStreaming = isLast && isStreaming && msg.role === "assistant";
        const isUser = msg.role === "user";

        return (
          <MessageRow
            key={msg.id}
            message={msg}
            isUser={isUser}
            showStreaming={showStreaming}
            isDark={isDark}
            colors={colors}
            animationDelay={index * 30}
          />
        );
      })}
      <div ref={messagesEndRef} style={{ height: 1 }} />
    </div>
  );
}

/**
 * Individual message row
 */
interface MessageRowProps {
  message: ChatMessage;
  isUser: boolean;
  showStreaming: boolean;
  isDark: boolean;
  colors: ReturnType<typeof useTheme>["colors"];
  animationDelay: number;
}

function MessageRow({
  message,
  isUser,
  showStreaming,
  isDark,
  colors,
  animationDelay,
}: MessageRowProps) {
  return (
    <div
      className="message-row"
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        animationDelay: `${animationDelay}ms`,
      }}
    >
      {/* Assistant Avatar */}
      {!isUser && (
        <Avatar
          icon={<RobotOutlined />}
          isDark={isDark}
          colors={colors}
          position="left"
        />
      )}

      {/* Message Content */}
      <div
        style={{
          maxWidth: isUser ? "75%" : "85%",
          padding: isUser ? "12px 16px" : "0",
          borderRadius: isUser ? 24 : 0,
          background: isUser ? colors.userBubble : "transparent",
        }}
      >
        {isUser ? (
          <div
            style={{
              whiteSpace: "pre-wrap",
              lineHeight: 1.6,
              fontSize: 15,
              color: colors.text,
            }}
          >
            {message.content}
          </div>
        ) : (
          <MessageContent
            content={message.content}
            isStreaming={showStreaming}
            isDark={isDark}
            colors={colors}
          />
        )}
      </div>

      {/* User Avatar */}
      {isUser && (
        <Avatar
          icon={<UserOutlined />}
          isDark={isDark}
          colors={colors}
          position="right"
        />
      )}
    </div>
  );
}

/**
 * Avatar component
 */
interface AvatarProps {
  icon: React.ReactNode;
  isDark: boolean;
  colors: ReturnType<typeof useTheme>["colors"];
  position: "left" | "right";
}

function Avatar({ icon, isDark, colors, position }: AvatarProps) {
  return (
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
        marginRight: position === "left" ? 12 : 0,
        marginLeft: position === "right" ? 12 : 0,
        marginTop: 4,
        color: colors.textSecondary,
        fontSize: 14,
      }}
    >
      {icon}
    </div>
  );
}

/**
 * Chat sidebar for conversation history
 */
interface ChatSidebarProps {
  conversations: Conversation[];
  currentConversationId: number | null;
  onSelectConversation: (id: number) => void;
  onNewChat: () => void;
  onDeleteConversation: (id: number) => void;
  onClearAll: () => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  isDark: boolean;
  colors: ReturnType<typeof useTheme>["colors"];
  isMobile: boolean;
}

function ChatSidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
  onClearAll,
  collapsed,
  onToggleCollapse,
  isDark,
  colors,
  isMobile,
}: ChatSidebarProps) {
  const [hoveredId, setHoveredId] = useState<number | null>(null);

  if (collapsed) {
    return null;
  }

  return (
    <>
      {/* Mobile overlay */}
      {isMobile && (
        <div
          onClick={onToggleCollapse}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.4)",
            zIndex: 99,
          }}
        />
      )}

      <div
        style={{
          width: isMobile ? 280 : 240,
          height: isMobile ? "auto" : "100%",
          background: "transparent",
          display: "flex",
          flexDirection: "column",
          position: isMobile ? "fixed" : "relative",
          left: 0,
          top: isMobile ? 64 : 0,
          bottom: isMobile ? 0 : "auto",
          zIndex: isMobile ? 100 : 1,
          boxShadow: isMobile ? "4px 0 12px rgba(0,0,0,0.15)" : "none",
        }}
      >
        {/* Header */}
        <div style={{ padding: "8px 8px 4px 8px" }}>
          <Button
            onClick={() => {
              onNewChat();
              if (isMobile) onToggleCollapse();
            }}
            style={{
              width: "100%",
              borderRadius: 8,
              height: 34,
              background: isDark
                ? "rgba(255,255,255,0.06)"
                : "rgba(0,0,0,0.03)",
              border: "none",
              color: colors.textSecondary,
              fontWeight: 500,
              fontSize: 13,
            }}
          >
            New Chat
          </Button>
        </div>

        {/* Conversation list */}
        <div
          style={{
            flex: 1,
            overflow: "auto",
            padding: "8px 8px",
          }}
        >
          {conversations.length === 0 ? (
            <div
              style={{
                padding: "40px 16px",
                textAlign: "center",
                color: colors.textMuted,
                fontSize: 13,
              }}
            >
              <MessageOutlined
                style={{
                  fontSize: 32,
                  opacity: 0.3,
                  marginBottom: 12,
                  display: "block",
                }}
              />
              No conversations
            </div>
          ) : (
            conversations.map((conv) => {
              const isActive = conv.id === currentConversationId;
              const isHovered = hoveredId === conv.id;

              return (
                <div
                  key={conv.id}
                  onClick={() => {
                    onSelectConversation(conv.id);
                    if (isMobile) onToggleCollapse();
                  }}
                  onMouseEnter={() => setHoveredId(conv.id)}
                  onMouseLeave={() => setHoveredId(null)}
                  style={{
                    padding: "10px 10px",
                    borderRadius: 8,
                    cursor: "pointer",
                    marginBottom: 2,
                    background: isActive
                      ? isDark
                        ? "rgba(255,255,255,0.1)"
                        : "rgba(0,0,0,0.06)"
                      : isHovered
                        ? isDark
                          ? "rgba(255,255,255,0.05)"
                          : "rgba(0,0,0,0.03)"
                        : "transparent",
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    transition: "background 0.15s ease",
                    position: "relative",
                  }}
                >
                  <MessageOutlined
                    style={{
                      fontSize: 13,
                      color: isActive ? colors.textSecondary : colors.textMuted,
                      flexShrink: 0,
                    }}
                  />
                  <div
                    style={{
                      flex: 1,
                      minWidth: 0,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      fontSize: 13,
                      color: isActive ? colors.text : colors.textSecondary,
                    }}
                  >
                    {conv.title}
                  </div>
                  {(isHovered || isActive) && (
                    <Button
                      type="text"
                      size="small"
                      icon={<DeleteOutlined style={{ fontSize: 12 }} />}
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteConversation(conv.id);
                      }}
                      style={{
                        width: 22,
                        height: 22,
                        minWidth: 22,
                        padding: 0,
                        color: colors.textMuted,
                        borderRadius: 4,
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.color = "#ef4444";
                        e.currentTarget.style.background = isDark
                          ? "rgba(239,68,68,0.1)"
                          : "rgba(239,68,68,0.08)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.color = colors.textMuted;
                        e.currentTarget.style.background = "transparent";
                      }}
                    />
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Footer with clear all */}
        {conversations.length > 0 && (
          <div
            style={{
              padding: "8px 12px",
              flexShrink: 0,
            }}
          >
            <Popconfirm
              title="Clear all?"
              onConfirm={onClearAll}
              okText="Clear"
              cancelText="Cancel"
              okButtonProps={{ danger: true, size: "small" }}
              cancelButtonProps={{ size: "small" }}
            >
              <span
                style={{
                  color: colors.textMuted,
                  fontSize: 12,
                  cursor: "pointer",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 4,
                }}
                onMouseEnter={(e) => (e.currentTarget.style.color = "#ef4444")}
                onMouseLeave={(e) =>
                  (e.currentTarget.style.color = colors.textMuted)
                }
              >
                <DeleteOutlined style={{ fontSize: 11 }} />
                Clear all
              </span>
            </Popconfirm>
          </div>
        )}
      </div>
    </>
  );
}
