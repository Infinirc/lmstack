/**
 * Workers API
 */
import { api } from "./client";
import type {
  Worker,
  WorkerCreate,
  ListResponse,
  RegistrationToken,
  RegistrationTokenCreate,
} from "../types";

export interface WorkerListParams {
  skip?: number;
  limit?: number;
  status?: string;
}

export const workersApi = {
  list: async (params?: WorkerListParams): Promise<ListResponse<Worker>> => {
    const response = await api.get<ListResponse<Worker>>("/workers", {
      params,
    });
    return response.data;
  },

  get: async (id: number): Promise<Worker> => {
    const response = await api.get<Worker>(`/workers/${id}`);
    return response.data;
  },

  create: async (data: WorkerCreate): Promise<Worker> => {
    const response = await api.post<Worker>("/workers", data);
    return response.data;
  },

  update: async (id: number, data: Partial<WorkerCreate>): Promise<Worker> => {
    const response = await api.patch<Worker>(`/workers/${id}`, data);
    return response.data;
  },

  delete: async (id: number): Promise<void> => {
    await api.delete(`/workers/${id}`);
  },

  registerLocal: async (): Promise<Worker> => {
    const response = await api.post<Worker>("/workers/local");
    return response.data;
  },

  // Registration Token APIs
  createToken: async (
    data: RegistrationTokenCreate,
  ): Promise<RegistrationToken> => {
    const response = await api.post<RegistrationToken>("/workers/tokens", data);
    return response.data;
  },

  listTokens: async (
    includeUsed = false,
  ): Promise<ListResponse<RegistrationToken>> => {
    const response = await api.get<ListResponse<RegistrationToken>>(
      "/workers/tokens",
      { params: { include_used: includeUsed } },
    );
    return response.data;
  },

  getToken: async (id: number): Promise<RegistrationToken> => {
    const response = await api.get<RegistrationToken>(`/workers/tokens/${id}`);
    return response.data;
  },

  deleteToken: async (id: number): Promise<void> => {
    await api.delete(`/workers/tokens/${id}`);
  },
};
