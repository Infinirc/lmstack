/**
 * Dashboard API
 */
import { api } from "./client";
import type { DashboardData } from "../types";

export const dashboardApi = {
  get: async (): Promise<DashboardData> => {
    const response = await api.get<DashboardData>("/dashboard");
    return response.data;
  },
};
