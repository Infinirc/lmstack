/**
 * Semantic Router API
 */
import { api } from "./client";

export interface SemanticRouterStatus {
  deployed: boolean;
  url?: string;
  dashboard_url?: string;
  message?: string;
}

export const semanticRouterApi = {
  getStatus: async (): Promise<SemanticRouterStatus> => {
    const response = await api.get<SemanticRouterStatus>(
      "/semantic-router/status",
    );
    return response.data;
  },

  updateConfig: async (): Promise<{ success: boolean; message: string }> => {
    const response = await api.post<{ success: boolean; message: string }>(
      "/semantic-router/update-config",
    );
    return response.data;
  },
};
