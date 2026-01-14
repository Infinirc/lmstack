/**
 * Models API
 */
import { api } from "./client";
import type { LLMModel, LLMModelCreate, ListResponse } from "../types";

export interface ModelListParams {
  skip?: number;
  limit?: number;
  backend?: string;
}

export const modelsApi = {
  list: async (params?: ModelListParams): Promise<ListResponse<LLMModel>> => {
    const response = await api.get<ListResponse<LLMModel>>("/models", {
      params,
    });
    return response.data;
  },

  get: async (id: number): Promise<LLMModel> => {
    const response = await api.get<LLMModel>(`/models/${id}`);
    return response.data;
  },

  create: async (data: LLMModelCreate): Promise<LLMModel> => {
    const response = await api.post<LLMModel>("/models", data);
    return response.data;
  },

  update: async (
    id: number,
    data: Partial<LLMModelCreate>,
  ): Promise<LLMModel> => {
    const response = await api.patch<LLMModel>(`/models/${id}`, data);
    return response.data;
  },

  delete: async (id: number): Promise<void> => {
    await api.delete(`/models/${id}`);
  },
};
