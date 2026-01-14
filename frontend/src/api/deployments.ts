/**
 * Deployments API
 */
import { api } from "./client";
import type { Deployment, DeploymentCreate, ListResponse } from "../types";

export interface DeploymentListParams {
  skip?: number;
  limit?: number;
  status?: string;
  worker_id?: number;
  model_id?: number;
}

export interface DeploymentLogsResponse {
  deployment_id: number;
  logs: string;
}

export const deploymentsApi = {
  list: async (
    params?: DeploymentListParams,
  ): Promise<ListResponse<Deployment>> => {
    const response = await api.get<ListResponse<Deployment>>("/deployments", {
      params,
    });
    return response.data;
  },

  get: async (id: number): Promise<Deployment> => {
    const response = await api.get<Deployment>(`/deployments/${id}`);
    return response.data;
  },

  create: async (data: DeploymentCreate): Promise<Deployment> => {
    const response = await api.post<Deployment>("/deployments", data);
    return response.data;
  },

  delete: async (id: number): Promise<void> => {
    await api.delete(`/deployments/${id}`);
  },

  stop: async (id: number): Promise<Deployment> => {
    const response = await api.post<Deployment>(`/deployments/${id}/stop`);
    return response.data;
  },

  start: async (id: number): Promise<Deployment> => {
    const response = await api.post<Deployment>(`/deployments/${id}/start`);
    return response.data;
  },

  getLogs: async (
    id: number,
    tail: number = 100,
  ): Promise<DeploymentLogsResponse> => {
    const response = await api.get<DeploymentLogsResponse>(
      `/deployments/${id}/logs`,
      {
        params: { tail },
      },
    );
    return response.data;
  },

  update: async (
    id: number,
    data: Partial<DeploymentCreate>,
  ): Promise<Deployment> => {
    const response = await api.patch<Deployment>(`/deployments/${id}`, data);
    return response.data;
  },
};
