/**
 * Model Files API
 */
import { api } from "./client";
import type { ModelFileView, ListResponse } from "../types";

export interface ModelFileListParams {
  skip?: number;
  limit?: number;
  worker_id?: number;
  model_id?: number;
}

export interface ModelFileDeleteResponse {
  message: string;
  deleted_count: number;
}

export const modelFilesApi = {
  list: async (
    params?: ModelFileListParams,
  ): Promise<ListResponse<ModelFileView>> => {
    const response = await api.get<ListResponse<ModelFileView>>(
      "/model-files",
      { params },
    );
    return response.data;
  },

  delete: async (
    modelId: number,
    workerId: number,
  ): Promise<ModelFileDeleteResponse> => {
    const response = await api.delete<ModelFileDeleteResponse>(
      `/model-files/${modelId}/workers/${workerId}`,
    );
    return response.data;
  },
};
