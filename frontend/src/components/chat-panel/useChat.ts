/**
 * useChat Hook
 *
 * Encapsulates chat logic for streaming conversations with LLM endpoints.
 * Supports deployments, Semantic Router, and custom OpenAI-compatible endpoints.
 * Includes system context injection for AI assistant capabilities.
 * Supports Tool Calling for LLM to interact with LMStack system.
 */
import { useState, useRef, useCallback, useEffect } from "react";
import { message } from "antd";
import { generateMessageId } from "../chat";
import type { ChatMessage } from "../chat";
import type { ChatModelConfig } from "./types";
import { STORAGE_KEYS } from "../../constants";
import {
  fetchSystemContext,
  formatSystemPrompt,
  type SystemContext,
} from "./systemContext";
import {
  CHAT_TOOLS,
  executeTool,
  requiresConfirmation,
  getToolMeta,
  type ToolCall,
  type ToolResult,
  type PendingToolExecution,
} from "./tools";

interface UseChatOptions {
  /** Called when a new message is added */
  onMessageAdded?: (message: ChatMessage) => void;
  /** Called when streaming completes */
  onStreamComplete?: (userMsg: ChatMessage, assistantMsg: ChatMessage) => void;
}

interface UseChatReturn {
  messages: ChatMessage[];
  isStreaming: boolean;
  isExecutingTool: boolean;
  currentToolName: string | null;
  pendingTools: PendingToolExecution[];
  showConfirmModal: boolean;
  systemContext: SystemContext | null;
  refreshContext: () => Promise<void>;
  sendMessage: (content: string, model: ChatModelConfig) => Promise<void>;
  stopStreaming: () => void;
  clearMessages: () => void;
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  confirmToolExecution: () => void;
  cancelToolExecution: () => void;
}

/**
 * Hook for managing chat state and streaming
 */
export function useChat(options: UseChatOptions = {}): UseChatReturn {
  const { onMessageAdded, onStreamComplete } = options;

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isExecutingTool, setIsExecutingTool] = useState(false);
  const [currentToolName, setCurrentToolName] = useState<string | null>(null);
  const [systemContext, setSystemContext] = useState<SystemContext | null>(
    null,
  );

  // Tool confirmation state
  const [pendingTools, setPendingTools] = useState<PendingToolExecution[]>([]);
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const pendingToolResolveRef = useRef<((confirmed: boolean) => void) | null>(
    null,
  );

  const abortControllerRef = useRef<AbortController | null>(null);

  // Fetch system context on mount and periodically refresh
  const refreshContext = useCallback(async () => {
    const context = await fetchSystemContext();
    setSystemContext(context);
  }, []);

  useEffect(() => {
    refreshContext();
    const interval = setInterval(refreshContext, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [refreshContext]);

  /**
   * Check if model uses proxy endpoint
   */
  const isProxyRequest = useCallback((model: ChatModelConfig): boolean => {
    return model.type === "custom";
  }, []);

  /**
   * Get the chat endpoint URL based on model config
   */
  const getEndpointUrl = useCallback((model: ChatModelConfig): string => {
    switch (model.type) {
      case "deployment":
        return `/api/deployments/${model.deploymentId}/chat`;
      case "semantic-router":
        return `/api/semantic-router/chat`;
      case "custom":
        // Use backend proxy to avoid CORS issues
        return `/api/chat-proxy`;
      default:
        return "";
    }
  }, []);

  /**
   * Get headers for the request
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
   * Stream a chat completion request
   */
  const streamChatCompletion = useCallback(
    async (
      endpoint: string,
      requestBody: any,
      assistantMessageId: string,
      signal: AbortSignal,
    ): Promise<{
      content: string;
      thinking: string;
      model?: string;
      toolCalls?: ToolCall[];
      finishReason?: string;
    }> => {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: getHeaders(),
        body: JSON.stringify(requestBody),
        signal,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let accumulatedContent = "";
      let accumulatedThinking = "";
      let responseModel: string | undefined;
      let buffer = "";
      let finishReason: string | undefined;

      // Track tool calls being accumulated
      const toolCallsMap: Map<
        number,
        { id: string; name: string; arguments: string }
      > = new Map();

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine || !trimmedLine.startsWith("data:")) continue;

          const data = trimmedLine.slice(5).trim();
          if (data === "[DONE]") continue;

          try {
            const parsed = JSON.parse(data);

            if (parsed.error) {
              throw new Error(parsed.error.message || "API error");
            }

            const choice = parsed.choices?.[0];
            const delta = choice?.delta;

            // Track finish reason
            if (choice?.finish_reason) {
              finishReason = choice.finish_reason;
            }

            // Handle regular content
            const deltaContent = delta?.content || "";
            accumulatedContent += deltaContent;

            // Handle thinking/reasoning content (for models like DeepSeek-R1)
            const deltaThinking = delta?.reasoning_content || "";
            accumulatedThinking += deltaThinking;

            // Handle tool calls streaming
            if (delta?.tool_calls) {
              for (const tc of delta.tool_calls) {
                const index = tc.index ?? 0;
                if (!toolCallsMap.has(index)) {
                  toolCallsMap.set(index, {
                    id: tc.id || "",
                    name: "",
                    arguments: "",
                  });
                }
                const existing = toolCallsMap.get(index)!;
                if (tc.id) existing.id = tc.id;
                if (tc.function?.name) existing.name = tc.function.name;
                if (tc.function?.arguments)
                  existing.arguments += tc.function.arguments;
              }
            }

            if (!responseModel && parsed.model) {
              responseModel = parsed.model;
            }

            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId
                  ? {
                      ...m,
                      content: accumulatedContent,
                      thinking: accumulatedThinking || undefined,
                      model: responseModel,
                    }
                  : m,
              ),
            );
          } catch {
            // Skip invalid JSON
          }
        }
      }

      // Process remaining buffer
      if (buffer.trim().startsWith("data:")) {
        const data = buffer.trim().slice(5).trim();
        if (data !== "[DONE]") {
          try {
            const parsed = JSON.parse(data);
            const choice = parsed.choices?.[0];
            const delta = choice?.delta;
            if (choice?.finish_reason) finishReason = choice.finish_reason;
            const deltaContent = delta?.content || "";
            const deltaThinking = delta?.reasoning_content || "";
            accumulatedContent += deltaContent;
            accumulatedThinking += deltaThinking;
            if (!responseModel && parsed.model) {
              responseModel = parsed.model;
            }
          } catch {
            // Skip invalid JSON
          }
        }
      }

      // Convert tool calls map to array
      const toolCalls: ToolCall[] = [];
      for (const [, tc] of toolCallsMap) {
        if (tc.id && tc.name) {
          toolCalls.push({
            id: tc.id,
            type: "function",
            function: { name: tc.name, arguments: tc.arguments },
          });
        }
      }

      return {
        content: accumulatedContent,
        thinking: accumulatedThinking,
        model: responseModel,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        finishReason,
      };
    },
    [getHeaders],
  );

  /**
   * Send a message and stream the response with tool calling support
   */
  const sendMessage = useCallback(
    async (content: string, model: ChatModelConfig) => {
      if (!content.trim() || isStreaming) return;

      const endpoint = getEndpointUrl(model);
      if (!endpoint) {
        message.error("Invalid endpoint configuration");
        return;
      }

      setIsStreaming(true);

      const userMessage: ChatMessage = {
        id: generateMessageId(),
        role: "user",
        content: content.trim(),
        timestamp: new Date(),
      };

      const assistantMessage: ChatMessage = {
        id: generateMessageId(),
        role: "assistant",
        content: "",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      onMessageAdded?.(userMessage);

      try {
        abortControllerRef.current = new AbortController();

        const modelName = model.modelId || model.name;

        // Build messages array with system context
        type ChatMessagePayload = {
          role: string;
          content: string | null;
          tool_calls?: ToolCall[];
          tool_call_id?: string;
        };
        const chatMessages: ChatMessagePayload[] = [];

        // Add system prompt with current context
        if (systemContext) {
          chatMessages.push({
            role: "system",
            content: formatSystemPrompt(systemContext),
          });
        }

        // Add conversation history
        chatMessages.push(
          ...messages.map((m) => ({ role: m.role, content: m.content })),
          { role: "user", content: content.trim() },
        );

        // Build the chat payload with tools
        const chatPayload = {
          model: modelName,
          messages: chatMessages,
          stream: true,
          temperature: 0.7,
          tools: CHAT_TOOLS,
          tool_choice: "auto" as const,
        };

        // Build request body
        const requestBody = isProxyRequest(model)
          ? {
              endpoint: model.endpoint,
              api_key: model.apiKey || null,
              payload: chatPayload,
            }
          : chatPayload;

        // Track the current message being streamed to
        let currentMessageId = assistantMessage.id;

        // First streaming request
        let result = await streamChatCompletion(
          endpoint,
          requestBody,
          currentMessageId,
          abortControllerRef.current.signal,
        );

        // Tool calling loop - continue until no more tool calls
        let iterationCount = 0;
        const maxIterations = 10; // Prevent infinite loops

        while (
          result.toolCalls &&
          result.toolCalls.length > 0 &&
          iterationCount < maxIterations
        ) {
          iterationCount++;

          // Check if any tool requires confirmation
          const toolsNeedingConfirmation = result.toolCalls.filter((tc) =>
            requiresConfirmation(tc.function.name),
          );

          if (toolsNeedingConfirmation.length > 0) {
            // Prepare pending tools for confirmation
            const pending: PendingToolExecution[] =
              toolsNeedingConfirmation.map((tc) => {
                let parsedArgs: Record<string, any> = {};
                try {
                  parsedArgs = JSON.parse(tc.function.arguments);
                } catch {
                  parsedArgs = { raw: tc.function.arguments };
                }
                return {
                  toolCall: tc,
                  parsedArgs,
                  meta: getToolMeta(tc.function.name),
                };
              });

            setPendingTools(pending);
            setShowConfirmModal(true);

            // Update message to show waiting for confirmation
            setMessages((prev) =>
              prev.map((m) =>
                m.id === currentMessageId
                  ? {
                      ...m,
                      content: result.content || "",
                      toolCalls: result.toolCalls,
                    }
                  : m,
              ),
            );

            // Wait for user confirmation
            const confirmed = await new Promise<boolean>((resolve) => {
              pendingToolResolveRef.current = resolve;
            });

            setPendingTools([]);
            setShowConfirmModal(false);
            pendingToolResolveRef.current = null;

            if (!confirmed) {
              // User cancelled - stop tool execution and inform LLM
              const cancelledResults: ToolResult[] =
                toolsNeedingConfirmation.map((tc) => ({
                  tool_call_id: tc.id,
                  role: "tool" as const,
                  content: JSON.stringify({
                    success: false,
                    message: "User cancelled the operation",
                  }),
                }));

              // Execute query tools that don't need confirmation
              const queryTools = result.toolCalls.filter(
                (tc) => !requiresConfirmation(tc.function.name),
              );
              for (const tc of queryTools) {
                const queryResult = await executeTool(tc, model);
                cancelledResults.push(queryResult);
              }

              // Update message
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === currentMessageId
                    ? {
                        ...m,
                        content: result.content || "",
                        toolCalls: result.toolCalls,
                      }
                    : m,
                ),
              );

              // Build new messages with cancelled results
              const newMessages: ChatMessagePayload[] = [
                ...chatMessages,
                {
                  role: "assistant",
                  content: result.content || null,
                  tool_calls: result.toolCalls,
                },
                ...cancelledResults.map((tr) => ({
                  role: "tool",
                  content: tr.content,
                  tool_call_id: tr.tool_call_id,
                })),
              ];

              // Continue to let LLM know about cancellation
              const continuationMessage: ChatMessage = {
                id: generateMessageId(),
                role: "assistant",
                content: "",
                timestamp: new Date(),
              };

              setMessages((prev) =>
                prev
                  .map((m) =>
                    m.id === currentMessageId
                      ? {
                          ...m,
                          content: result.content || "",
                          toolCalls: result.toolCalls,
                        }
                      : m,
                  )
                  .concat(continuationMessage),
              );

              // Update current message ID to the continuation
              currentMessageId = continuationMessage.id;

              const continuationPayload = {
                model: modelName,
                messages: newMessages,
                stream: true,
                temperature: 0.7,
                tools: CHAT_TOOLS,
                tool_choice: "auto" as const,
              };

              const continuationRequestBody = isProxyRequest(model)
                ? {
                    endpoint: model.endpoint,
                    api_key: model.apiKey || null,
                    payload: continuationPayload,
                  }
                : continuationPayload;

              result = await streamChatCompletion(
                endpoint,
                continuationRequestBody,
                currentMessageId,
                abortControllerRef.current.signal,
              );

              chatMessages.length = 0;
              chatMessages.push(...newMessages);
              continue;
            }
          }

          // Execute all tool calls (confirmed or query-only)
          setIsExecutingTool(true);
          const toolResults: ToolResult[] = [];

          for (const toolCall of result.toolCalls) {
            setCurrentToolName(toolCall.function.name);

            // Update message to show tool execution
            setMessages((prev) =>
              prev.map((m) =>
                m.id === currentMessageId
                  ? {
                      ...m,
                      content: result.content || "",
                      toolCalls: result.toolCalls,
                    }
                  : m,
              ),
            );

            const toolResult = await executeTool(toolCall, model);
            toolResults.push(toolResult);

            // Refresh system context after tool execution (data may have changed)
            await refreshContext();
          }

          setIsExecutingTool(false);
          setCurrentToolName(null);

          // Build new messages array with tool calls and results
          const newMessages: ChatMessagePayload[] = [
            ...chatMessages,
            // Assistant message with tool calls
            {
              role: "assistant",
              content: result.content || null,
              tool_calls: result.toolCalls,
            },
            // Tool results
            ...toolResults.map((tr) => ({
              role: "tool",
              content: tr.content,
              tool_call_id: tr.tool_call_id,
            })),
          ];

          // Create new assistant message for continued response
          const continuationMessage: ChatMessage = {
            id: generateMessageId(),
            role: "assistant",
            content: "",
            timestamp: new Date(),
          };

          setMessages((prev) => {
            // Update the current assistant message and add continuation
            return prev
              .map((m) =>
                m.id === currentMessageId
                  ? {
                      ...m,
                      content: result.content || "",
                      toolCalls: result.toolCalls,
                    }
                  : m,
              )
              .concat(continuationMessage);
          });

          // Update current message ID to the continuation
          currentMessageId = continuationMessage.id;

          // Build new chat payload with tool results
          const continuationPayload = {
            model: modelName,
            messages: newMessages,
            stream: true,
            temperature: 0.7,
            tools: CHAT_TOOLS,
            tool_choice: "auto" as const,
          };

          const continuationRequestBody = isProxyRequest(model)
            ? {
                endpoint: model.endpoint,
                api_key: model.apiKey || null,
                payload: continuationPayload,
              }
            : continuationPayload;

          // Continue streaming
          result = await streamChatCompletion(
            endpoint,
            continuationRequestBody,
            currentMessageId,
            abortControllerRef.current.signal,
          );

          // Update chat messages for next iteration if needed
          chatMessages.length = 0;
          chatMessages.push(...newMessages);
        }

        // Final update - only update the current (last) message
        setMessages((prev) =>
          prev.map((m) => {
            if (m.id === currentMessageId) {
              return {
                ...m,
                content: result.content,
                thinking: result.thinking || undefined,
                model: result.model,
                // Only set toolCalls if there are any (don't overwrite with undefined)
                ...(result.toolCalls ? { toolCalls: result.toolCalls } : {}),
              };
            }
            return m;
          }),
        );

        // Get the final message for callback
        const finalAssistantMsg: ChatMessage = {
          id: currentMessageId,
          role: "assistant",
          content: result.content,
          thinking: result.thinking || undefined,
          model: result.model,
          toolCalls: result.toolCalls,
          timestamp: new Date(),
        };

        onStreamComplete?.(userMessage, finalAssistantMsg);
      } catch (error: unknown) {
        const err = error as Error;
        if (err.name === "AbortError") {
          message.info("Generation stopped");
        } else {
          message.error(`Error: ${err.message}`);
          // Remove the initial assistant message on error
          setMessages((prev) =>
            prev.filter((m) => m.id !== assistantMessage.id),
          );
        }
      } finally {
        setIsStreaming(false);
        setIsExecutingTool(false);
        setCurrentToolName(null);
        abortControllerRef.current = null;
      }
    },
    [
      messages,
      isStreaming,
      systemContext,
      getEndpointUrl,
      isProxyRequest,
      getHeaders,
      onMessageAdded,
      onStreamComplete,
      streamChatCompletion,
      refreshContext,
    ],
  );

  /**
   * Stop the current streaming response
   */
  const stopStreaming = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

  /**
   * Clear all messages
   */
  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  /**
   * Confirm pending tool execution
   */
  const confirmToolExecution = useCallback(() => {
    if (pendingToolResolveRef.current) {
      pendingToolResolveRef.current(true);
    }
  }, []);

  /**
   * Cancel pending tool execution
   */
  const cancelToolExecution = useCallback(() => {
    if (pendingToolResolveRef.current) {
      pendingToolResolveRef.current(false);
    }
  }, []);

  return {
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
    setMessages,
    confirmToolExecution,
    cancelToolExecution,
  };
}
