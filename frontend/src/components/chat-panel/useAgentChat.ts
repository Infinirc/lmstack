/**
 * useAgentChat Hook
 *
 * Connects to the MCP-based Agent API using Server-Sent Events (SSE).
 * Provides Claude Code-style interaction with real-time streaming of
 * agent thinking, planning, and tool execution.
 */
import { useState, useRef, useCallback, useEffect } from "react";
import { message } from "antd";
import { STORAGE_KEYS } from "../../constants";
import type { ChatModelConfig } from "./types";

/**
 * Agent event types from the backend
 */
export type AgentEventType =
  | "init"
  | "thinking"
  | "planning"
  | "reasoning"
  | "message"
  | "tool_start"
  | "tool_progress"
  | "tool_result"
  | "tool_error"
  | "page_reference"
  | "action_suggestions"
  | "done"
  | "error"
  | "cancelled";

/**
 * Page reference for navigation cards
 */
export interface PageReference {
  path: string;
  title: string;
  icon: string;
  description: string;
}

/**
 * Action suggestion for quick actions
 */
export interface ActionSuggestion {
  label: string;
  message: string;
  icon: string;
  type: "primary" | "default" | "danger";
}

/**
 * Agent event from SSE stream
 */
export interface AgentEvent {
  type: AgentEventType;
  content: string | null;
  data: Record<string, any> | null;
  timestamp: string;
}

/**
 * Execution step in the agent workflow
 */
export interface ExecutionStep {
  id: string;
  type: "thinking" | "planning" | "reasoning" | "tool" | "message";
  status: "pending" | "running" | "completed" | "error";
  title: string;
  content?: string;
  toolName?: string;
  toolArgs?: Record<string, any>;
  toolResult?: string;
  executionTimeMs?: number;
  error?: string;
  timestamp: Date;
  expanded?: boolean;
}

/**
 * Agent chat message (conversation turn)
 */
export interface AgentChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  steps: ExecutionStep[];
  pageReferences: PageReference[];
  actionSuggestions: ActionSuggestion[];
  timestamp: Date;
  isStreaming?: boolean;
}

/**
 * LLM configuration for agent
 */
export interface AgentLLMConfig {
  provider: "system" | "openai" | "custom";
  deployment_id?: number;
  api_key?: string;
  base_url?: string;
  model?: string;
}

interface UseAgentChatOptions {
  onComplete?: () => void;
  onError?: (error: string) => void;
  onConversationCreated?: (conversationId: number) => void;
}

interface UseAgentChatReturn {
  messages: AgentChatMessage[];
  currentSteps: ExecutionStep[];
  isStreaming: boolean;
  isThinking: boolean;
  currentTool: string | null;
  conversationId: number | null;
  sendMessage: (content: string, model: ChatModelConfig) => Promise<void>;
  stopStreaming: () => void;
  clearMessages: () => void;
  toggleStepExpanded: (stepId: string) => void;
  loadConversation: (conversationId: number) => Promise<void>;
  startNewConversation: () => void;
}

let stepIdCounter = 0;
let messageIdCounter = 0;

function generateStepId(): string {
  return `step-${Date.now()}-${++stepIdCounter}`;
}

function generateMessageId(): string {
  return `msg-${Date.now()}-${++messageIdCounter}`;
}

/**
 * Convert ChatModelConfig to AgentLLMConfig
 */
function toAgentLLMConfig(model: ChatModelConfig): AgentLLMConfig {
  switch (model.type) {
    case "deployment":
      return {
        provider: "system",
        deployment_id: model.deploymentId,
        model: model.modelId || model.name,
      };
    case "custom":
      return {
        provider: "custom",
        base_url: model.endpoint,
        api_key: model.apiKey,
        model: model.modelId || model.name,
      };
    case "semantic-router":
      // For semantic router, use it as a custom endpoint
      return {
        provider: "custom",
        base_url: "/api/semantic-router/chat",
        model: model.name,
      };
    default:
      return {
        provider: "system",
        model: model.name,
      };
  }
}

/**
 * Hook for Claude Code-style agent chat
 */
export function useAgentChat(
  options: UseAgentChatOptions = {},
): UseAgentChatReturn {
  const { onComplete, onError, onConversationCreated } = options;

  const [messages, setMessages] = useState<AgentChatMessage[]>([]);
  const [currentSteps, setCurrentSteps] = useState<ExecutionStep[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [currentTool, setCurrentTool] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<number | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  /**
   * Get auth headers
   */
  const getHeaders = useCallback((): HeadersInit => {
    const headers: HeadersInit = {
      "Content-Type": "application/json",
    };
    const token = localStorage.getItem(STORAGE_KEYS.TOKEN);
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    return headers;
  }, []);

  /**
   * Send a message to the agent
   */
  const sendMessage = useCallback(
    async (content: string, model: ChatModelConfig) => {
      if (!content.trim() || isStreaming) return;

      setIsStreaming(true);
      setIsThinking(true);
      setCurrentSteps([]);

      // Create user message
      const userMessage: AgentChatMessage = {
        id: generateMessageId(),
        role: "user",
        content: content.trim(),
        steps: [],
        pageReferences: [],
        actionSuggestions: [],
        timestamp: new Date(),
      };

      // Create assistant message placeholder
      const assistantMessage: AgentChatMessage = {
        id: generateMessageId(),
        role: "assistant",
        content: "",
        steps: [],
        pageReferences: [],
        actionSuggestions: [],
        timestamp: new Date(),
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);

      const assistantMessageId = assistantMessage.id;
      const llmConfig = toAgentLLMConfig(model);

      try {
        abortControllerRef.current = new AbortController();

        const response = await fetch("/api/agent/chat", {
          method: "POST",
          headers: getHeaders(),
          body: JSON.stringify({
            message: content.trim(),
            llm_config: llmConfig,
            conversation_id: conversationId,
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(
            `API error: ${response.status} ${response.statusText}`,
          );
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let buffer = "";
        let accumulatedContent = "";
        const steps: ExecutionStep[] = [];
        const pageRefs: PageReference[] = [];
        const actionSuggs: ActionSuggestion[] = [];

        // Track current step for updates
        let currentStepId: string | null = null;

        let done = false;
        while (!done) {
          const result = await reader.read();
          done = result.done;
          if (done) break;
          const value = result.value;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmedLine = line.trim();
            if (!trimmedLine || !trimmedLine.startsWith("data:")) continue;

            const data = trimmedLine.slice(5).trim();
            if (!data || data === "[DONE]") continue;

            try {
              const event: AgentEvent = JSON.parse(data);

              switch (event.type) {
                case "init":
                  // Backend sends the conversation_id when a new conversation is created
                  if (event.data?.conversation_id) {
                    const newConversationId = event.data.conversation_id;
                    setConversationId(newConversationId);
                    onConversationCreated?.(newConversationId);
                  }
                  break;

                case "thinking":
                  setIsThinking(true);
                  {
                    const thinkingStep: ExecutionStep = {
                      id: generateStepId(),
                      type: "thinking",
                      status: "running",
                      title: "Analyzing request...",
                      content: event.content || undefined,
                      timestamp: new Date(),
                    };
                    steps.push(thinkingStep);
                    setCurrentSteps([...steps]);
                  }
                  break;

                case "planning":
                  {
                    const planningStep: ExecutionStep = {
                      id: generateStepId(),
                      type: "planning",
                      status: "running",
                      title: "Planning actions...",
                      content: event.content || undefined,
                      timestamp: new Date(),
                    };
                    steps.push(planningStep);
                    setCurrentSteps([...steps]);
                  }
                  break;

                case "reasoning":
                  // Handle model's reasoning/thinking process (e.g., DeepSeek-R1)
                  {
                    // Find existing reasoning step or create new one
                    let reasoningStep = steps.find(
                      (s) => s.type === "reasoning" && s.status === "running",
                    );
                    if (!reasoningStep) {
                      reasoningStep = {
                        id: generateStepId(),
                        type: "reasoning",
                        status: "running",
                        title: "Thinking...",
                        content: event.content || "",
                        timestamp: new Date(),
                        expanded: true,
                      };
                      steps.push(reasoningStep);
                    } else {
                      // Append to existing reasoning content
                      reasoningStep.content =
                        (reasoningStep.content || "") + (event.content || "");
                    }
                    setCurrentSteps([...steps]);
                  }
                  break;

                case "tool_start":
                  setIsThinking(false);
                  {
                    const toolName = event.data?.tool_name || "Unknown tool";
                    const toolArgs = event.data?.arguments || {};
                    setCurrentTool(toolName);

                    const toolStep: ExecutionStep = {
                      id: generateStepId(),
                      type: "tool",
                      status: "running",
                      title: formatToolName(toolName),
                      toolName,
                      toolArgs,
                      timestamp: new Date(),
                      expanded: true,
                    };
                    currentStepId = toolStep.id;
                    steps.push(toolStep);
                    setCurrentSteps([...steps]);

                    // Mark any previous thinking/planning/reasoning as completed
                    for (const step of steps) {
                      if (
                        (step.type === "thinking" ||
                          step.type === "planning" ||
                          step.type === "reasoning") &&
                        step.status === "running"
                      ) {
                        step.status = "completed";
                        // Auto-collapse reasoning when completed
                        if (step.type === "reasoning") {
                          step.expanded = false;
                        }
                      }
                    }
                  }
                  break;

                case "tool_progress":
                  // Update current tool step with progress
                  if (currentStepId) {
                    const stepIndex = steps.findIndex(
                      (s) => s.id === currentStepId,
                    );
                    if (stepIndex >= 0) {
                      steps[stepIndex] = {
                        ...steps[stepIndex],
                        content: event.content || steps[stepIndex].content,
                      };
                      setCurrentSteps([...steps]);
                    }
                  }
                  break;

                case "tool_result":
                  setCurrentTool(null);
                  if (currentStepId) {
                    const stepIndex = steps.findIndex(
                      (s) => s.id === currentStepId,
                    );
                    if (stepIndex >= 0) {
                      steps[stepIndex] = {
                        ...steps[stepIndex],
                        status: "completed",
                        toolResult: event.data?.result,
                        executionTimeMs: event.data?.execution_time_ms,
                      };
                      setCurrentSteps([...steps]);
                    }
                    currentStepId = null;
                  }
                  break;

                case "tool_error":
                  setCurrentTool(null);
                  if (currentStepId) {
                    const stepIndex = steps.findIndex(
                      (s) => s.id === currentStepId,
                    );
                    if (stepIndex >= 0) {
                      steps[stepIndex] = {
                        ...steps[stepIndex],
                        status: "error",
                        error:
                          event.data?.error || event.content || "Unknown error",
                      };
                      setCurrentSteps([...steps]);
                    }
                    currentStepId = null;
                  }
                  break;

                case "message":
                  setIsThinking(false);
                  if (event.content) {
                    accumulatedContent += event.content;
                    setMessages((prev) =>
                      prev.map((m) =>
                        m.id === assistantMessageId
                          ? { ...m, content: accumulatedContent }
                          : m,
                      ),
                    );
                  }
                  break;

                case "page_reference":
                  // Add page reference for navigation card
                  if (event.data) {
                    const ref = event.data as PageReference;
                    // Avoid duplicates
                    if (!pageRefs.some((r) => r.path === ref.path)) {
                      pageRefs.push(ref);
                    }
                  }
                  break;

                case "action_suggestions":
                  // Add action suggestions for quick actions
                  if (event.data?.suggestions) {
                    for (const suggestion of event.data.suggestions) {
                      // Avoid duplicates by message
                      if (
                        !actionSuggs.some(
                          (s) => s.message === suggestion.message,
                        )
                      ) {
                        actionSuggs.push(suggestion as ActionSuggestion);
                      }
                    }
                  }
                  break;

                case "done":
                  // Mark all running steps as completed
                  for (const step of steps) {
                    if (step.status === "running") {
                      step.status = "completed";
                      // Auto-collapse reasoning when completed
                      if (step.type === "reasoning") {
                        step.expanded = false;
                      }
                    }
                  }
                  setCurrentSteps([...steps]);

                  // Finalize the assistant message
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMessageId
                        ? {
                            ...m,
                            content: accumulatedContent,
                            steps: [...steps],
                            pageReferences: [...pageRefs],
                            actionSuggestions: [...actionSuggs],
                            isStreaming: false,
                          }
                        : m,
                    ),
                  );

                  onComplete?.();
                  break;

                case "error":
                  message.error(event.content || "Agent error");
                  onError?.(event.content || "Agent error");

                  // Mark message as error
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMessageId
                        ? {
                            ...m,
                            content: event.content || "An error occurred",
                            steps: [...steps],
                            isStreaming: false,
                          }
                        : m,
                    ),
                  );
                  break;

                case "cancelled":
                  message.info("Operation cancelled");
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMessageId
                        ? {
                            ...m,
                            content:
                              accumulatedContent || "Operation cancelled",
                            steps: [...steps],
                            isStreaming: false,
                          }
                        : m,
                    ),
                  );
                  break;
              }
            } catch {
              // Skip invalid JSON
            }
          }
        }
      } catch (error: unknown) {
        const err = error as Error;
        if (err.name === "AbortError") {
          message.info("Generation stopped");
        } else {
          message.error(`Error: ${err.message}`);
          onError?.(err.message);
        }
      } finally {
        setIsStreaming(false);
        setIsThinking(false);
        setCurrentTool(null);
        setCurrentSteps([]);
        abortControllerRef.current = null;
      }
    },
    [
      isStreaming,
      getHeaders,
      onComplete,
      onError,
      conversationId,
      onConversationCreated,
    ],
  );

  /**
   * Stop the current streaming operation
   */
  const stopStreaming = useCallback(() => {
    abortControllerRef.current?.abort();
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  /**
   * Clear all messages
   */
  const clearMessages = useCallback(() => {
    setMessages([]);
    setCurrentSteps([]);
  }, []);

  /**
   * Start a new conversation (clear state)
   */
  const startNewConversation = useCallback(() => {
    setMessages([]);
    setCurrentSteps([]);
    setConversationId(null);
  }, []);

  /**
   * Load an existing conversation from the database
   */
  const loadConversation = useCallback(
    async (convId: number) => {
      try {
        const response = await fetch(`/api/agent/conversations/${convId}`, {
          headers: getHeaders(),
        });

        if (!response.ok) {
          throw new Error(
            `Failed to load conversation: ${response.statusText}`,
          );
        }

        const data = await response.json();

        // Convert API messages to AgentChatMessage format
        const loadedMessages: AgentChatMessage[] = data.messages.map(
          (msg: {
            id: number;
            role: string;
            content: string;
            thinking?: string;
            tool_calls?: Array<{
              tool_name: string;
              arguments?: Record<string, unknown>;
              status: string;
              result?: string;
              error?: string;
              execution_time_ms?: number;
            }>;
            created_at: string;
          }) => {
            // Convert tool_calls to ExecutionSteps
            const steps: ExecutionStep[] = [];
            if (msg.tool_calls) {
              for (const tc of msg.tool_calls) {
                steps.push({
                  id: generateStepId(),
                  type: "tool",
                  status:
                    tc.status === "completed"
                      ? "completed"
                      : tc.status === "error"
                        ? "error"
                        : "pending",
                  title: formatToolName(tc.tool_name || "Unknown"),
                  toolName: tc.tool_name,
                  toolArgs: tc.arguments,
                  toolResult: tc.result,
                  error: tc.error,
                  executionTimeMs: tc.execution_time_ms,
                  timestamp: new Date(msg.created_at),
                  expanded: false,
                });
              }
            }

            return {
              id: `db-${msg.id}`,
              role: msg.role as "user" | "assistant",
              content: msg.content,
              steps,
              pageReferences: [],
              actionSuggestions: [],
              timestamp: new Date(msg.created_at),
              isStreaming: false,
            };
          },
        );

        setMessages(loadedMessages);
        setConversationId(convId);
        setCurrentSteps([]);
      } catch (error) {
        const err = error as Error;
        message.error(`Failed to load conversation: ${err.message}`);
        throw error;
      }
    },
    [getHeaders],
  );

  /**
   * Toggle step expansion
   */
  const toggleStepExpanded = useCallback((stepId: string) => {
    setCurrentSteps((prev) =>
      prev.map((step) =>
        step.id === stepId ? { ...step, expanded: !step.expanded } : step,
      ),
    );

    // Also update in messages
    setMessages((prev) =>
      prev.map((msg) => ({
        ...msg,
        steps: msg.steps.map((step) =>
          step.id === stepId ? { ...step, expanded: !step.expanded } : step,
        ),
      })),
    );
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  return {
    messages,
    currentSteps,
    isStreaming,
    isThinking,
    currentTool,
    conversationId,
    sendMessage,
    stopStreaming,
    clearMessages,
    toggleStepExpanded,
    loadConversation,
    startNewConversation,
  };
}

/**
 * Format tool name for display
 */
function formatToolName(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
