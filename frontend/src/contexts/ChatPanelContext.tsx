/**
 * Chat Panel Context
 *
 * Provides global access to the Chat Panel functionality.
 * Allows any page to open the panel and send messages to the agent.
 */
import {
  createContext,
  useContext,
  useCallback,
  useRef,
  type ReactNode,
} from "react";
import type { ChatModelConfig } from "../components/chat-panel/types";

/**
 * Internal send function that requires model
 */
type SendMessageFn = (message: string, model: ChatModelConfig) => Promise<void>;

/**
 * Context value interface
 */
interface ChatPanelContextValue {
  /**
   * Opens the chat panel
   */
  openPanel: () => void;

  /**
   * Closes the chat panel
   */
  closePanel: () => void;

  /**
   * Whether the panel is currently open
   */
  isOpen: boolean;

  /**
   * Send a message to the agent using the currently selected model.
   * If no model is selected, this will do nothing and return false.
   */
  sendAgentMessage: (message: string) => boolean;

  /**
   * Get the currently selected model configuration.
   * Returns null if no model is selected.
   */
  getSelectedModel: () => ChatModelConfig | null;

  /**
   * Internal: Register the ChatPanel's send function and model.
   * This should only be called by ChatPanel.
   */
  _registerSendFunction: (
    fn: SendMessageFn | null,
    model: ChatModelConfig | null,
  ) => void;
}

const ChatPanelContext = createContext<ChatPanelContextValue | null>(null);

interface ChatPanelProviderProps {
  children: ReactNode;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
}

/**
 * Chat Panel Context Provider
 */
export function ChatPanelProvider({
  children,
  isOpen,
  onOpenChange,
}: ChatPanelProviderProps) {
  // Store the send function and model from ChatPanel
  const sendFnRef = useRef<SendMessageFn | null>(null);
  const modelRef = useRef<ChatModelConfig | null>(null);

  const openPanel = useCallback(() => {
    onOpenChange(true);
  }, [onOpenChange]);

  const closePanel = useCallback(() => {
    onOpenChange(false);
  }, [onOpenChange]);

  const _registerSendFunction = useCallback(
    (fn: SendMessageFn | null, model: ChatModelConfig | null) => {
      sendFnRef.current = fn;
      modelRef.current = model;
    },
    [],
  );

  const sendAgentMessage = useCallback((message: string): boolean => {
    const fn = sendFnRef.current;
    const model = modelRef.current;

    if (!fn || !model) {
      console.warn(
        "ChatPanel: Cannot send message - no model selected or panel not ready",
      );
      return false;
    }

    // Send the message (async, but we don't wait)
    fn(message, model);
    return true;
  }, []);

  const getSelectedModel = useCallback((): ChatModelConfig | null => {
    return modelRef.current;
  }, []);

  const value: ChatPanelContextValue = {
    openPanel,
    closePanel,
    isOpen,
    sendAgentMessage,
    getSelectedModel,
    _registerSendFunction,
  };

  return (
    <ChatPanelContext.Provider value={value}>
      {children}
    </ChatPanelContext.Provider>
  );
}

/**
 * Hook to access the Chat Panel context
 */
export function useChatPanel(): ChatPanelContextValue {
  const context = useContext(ChatPanelContext);

  if (!context) {
    throw new Error("useChatPanel must be used within a ChatPanelProvider");
  }

  return context;
}
