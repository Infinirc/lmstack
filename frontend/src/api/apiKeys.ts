/**
 * API Keys API
 */
import { api } from "./client";
import type {
  ApiKey,
  ApiKeyCreate,
  ApiKeyCreateResponse,
  ListResponse,
} from "../types";

export interface ApiKeyListParams {
  skip?: number;
  limit?: number;
}

export interface ApiKeyStats {
  total_requests: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_tokens: number;
  per_key_stats: Record<number, { requests: number; tokens: number }>;
}

export const apiKeysApi = {
  list: async (params?: ApiKeyListParams): Promise<ListResponse<ApiKey>> => {
    const response = await api.get<ListResponse<ApiKey>>("/api-keys", {
      params,
    });
    return response.data;
  },

  get: async (id: number): Promise<ApiKey> => {
    const response = await api.get<ApiKey>(`/api-keys/${id}`);
    return response.data;
  },

  create: async (data: ApiKeyCreate): Promise<ApiKeyCreateResponse> => {
    const response = await api.post<ApiKeyCreateResponse>("/api-keys", data);
    return response.data;
  },

  update: async (id: number, data: Partial<ApiKeyCreate>): Promise<ApiKey> => {
    const response = await api.patch<ApiKey>(`/api-keys/${id}`, data);
    return response.data;
  },

  delete: async (id: number): Promise<void> => {
    await api.delete(`/api-keys/${id}`);
  },

  getStats: async (): Promise<ApiKeyStats> => {
    const response = await api.get<ApiKeyStats>("/api-keys/stats/summary");
    return response.data;
  },
};
