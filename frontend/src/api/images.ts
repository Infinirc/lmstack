/**
 * Images API
 */
import { api } from "./client";
import type {
  ContainerImage,
  ContainerImageDetail,
  ImagePullRequest,
  ImageBuildRequest,
  ImageOperationProgress,
  ListResponse,
} from "../types";

export interface ImageListParams {
  skip?: number;
  limit?: number;
  worker_id?: number;
  repository?: string;
}

export interface ImageSearchResult {
  name: string;
  description: string;
  star_count: number;
  is_official: boolean;
}

export const imagesApi = {
  list: async (
    params?: ImageListParams,
  ): Promise<ListResponse<ContainerImage>> => {
    const response = await api.get<ListResponse<ContainerImage>>("/images", {
      params,
    });
    return response.data;
  },

  get: async (
    imageId: string,
    workerId: number,
  ): Promise<ContainerImageDetail> => {
    const response = await api.get<ContainerImageDetail>(
      `/images/${encodeURIComponent(imageId)}`,
      { params: { worker_id: workerId } },
    );
    return response.data;
  },

  pull: async (data: ImagePullRequest): Promise<{ task_id: string }> => {
    const response = await api.post<{ task_id: string }>("/images/pull", data);
    return response.data;
  },

  getPullProgress: async (
    taskId: string,
  ): Promise<ImageOperationProgress[]> => {
    const response = await api.get<ImageOperationProgress[]>(
      `/images/pull/${taskId}/progress`,
    );
    return response.data;
  },

  build: async (data: ImageBuildRequest): Promise<{ task_id: string }> => {
    const response = await api.post<{ task_id: string }>("/images/build", data);
    return response.data;
  },

  getBuildProgress: async (
    taskId: string,
  ): Promise<ImageOperationProgress[]> => {
    const response = await api.get<ImageOperationProgress[]>(
      `/images/build/${taskId}/progress`,
    );
    return response.data;
  },

  delete: async (
    imageId: string,
    workerId: number,
    force?: boolean,
  ): Promise<void> => {
    await api.delete(`/images/${encodeURIComponent(imageId)}`, {
      params: { worker_id: workerId, force },
    });
  },

  search: async (
    query: string,
    limit?: number,
  ): Promise<ImageSearchResult[]> => {
    const response = await api.get("/images/search", {
      params: { q: query, limit },
    });
    return response.data;
  },
};
