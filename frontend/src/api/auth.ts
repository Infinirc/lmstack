/**
 * Auth API
 */
import { api } from "./client";
import type {
  User,
  LoginRequest,
  TokenResponse,
  SetupRequest,
  SetupStatus,
} from "../types";

export interface ChangePasswordRequest {
  current_password: string;
  new_password: string;
}

export const authApi = {
  getSetupStatus: async (): Promise<SetupStatus> => {
    const response = await api.get<SetupStatus>("/auth/setup/status");
    return response.data;
  },

  setup: async (data: SetupRequest): Promise<TokenResponse> => {
    const response = await api.post<TokenResponse>("/auth/setup", data);
    return response.data;
  },

  login: async (data: LoginRequest): Promise<TokenResponse> => {
    const response = await api.post<TokenResponse>("/auth/login", data);
    return response.data;
  },

  getCurrentUser: async (): Promise<User> => {
    const response = await api.get<User>("/auth/me");
    return response.data;
  },

  changePassword: async (data: ChangePasswordRequest): Promise<void> => {
    await api.post("/auth/me/password", data);
  },
};
