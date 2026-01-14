/**
 * Conversations API
 */
import { api } from "./client";
import type { ListResponse } from "../types";

export interface ConversationMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  created_at: string;
}

export interface Conversation {
  id: number;
  title: string;
  deployment_id?: number;
  created_at: string;
  updated_at: string;
  messages?: ConversationMessage[];
}

export interface ConversationListParams {
  skip?: number;
  limit?: number;
}

export interface ConversationCreateData {
  title: string;
  deployment_id?: number;
  messages?: {
    role: "user" | "assistant";
    content: string;
    thinking?: string;
    prompt_tokens?: number;
    completion_tokens?: number;
  }[];
}

export interface AddMessagesData {
  messages: {
    role: "user" | "assistant";
    content: string;
    thinking?: string;
    prompt_tokens?: number;
    completion_tokens?: number;
  }[];
}

export const conversationsApi = {
  list: async (
    params?: ConversationListParams,
  ): Promise<ListResponse<Conversation>> => {
    const response = await api.get<ListResponse<Conversation>>(
      "/conversations",
      { params },
    );
    return response.data;
  },

  get: async (id: number): Promise<Conversation> => {
    const response = await api.get<Conversation>(`/conversations/${id}`);
    return response.data;
  },

  create: async (data: ConversationCreateData): Promise<Conversation> => {
    const response = await api.post<Conversation>("/conversations", data);
    return response.data;
  },

  update: async (
    id: number,
    data: { title?: string },
  ): Promise<Conversation> => {
    const response = await api.patch<Conversation>(
      `/conversations/${id}`,
      data,
    );
    return response.data;
  },

  delete: async (id: number): Promise<void> => {
    await api.delete(`/conversations/${id}`);
  },

  addMessages: async (
    id: number,
    data: AddMessagesData,
  ): Promise<Conversation> => {
    const response = await api.post<Conversation>(
      `/conversations/${id}/messages`,
      data,
    );
    return response.data;
  },

  clearAll: async (): Promise<void> => {
    await api.delete("/conversations");
  },
};
