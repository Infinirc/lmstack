/**
 * API Module
 *
 * Re-exports all API services and types for convenient imports.
 *
 * @example
 * import { workersApi, modelsApi } from '@/api'
 * import type { DiskUsage, Volume } from '@/api'
 */

// Client
export { api, default } from "./client";
export type { ApiErrorResponse } from "./client";

// API Services
export { workersApi } from "./workers";
export { modelsApi } from "./models";
export { deploymentsApi } from "./deployments";
export { authApi } from "./auth";
export { usersApi } from "./users";
export { apiKeysApi } from "./apiKeys";
export { imagesApi } from "./images";
export { containersApi } from "./containers";
export { storageApi } from "./storage";
export { dashboardApi } from "./dashboard";
export { modelFilesApi } from "./modelFiles";
export { systemApi } from "./system";
export { huggingfaceApi } from "./huggingface";
export { ollamaApi } from "./ollama";
export { conversationsApi } from "./conversations";
export { appsApi } from "./apps";
export { headscaleApi } from "./headscale";
export { semanticRouterApi } from "./semanticRouter";

// Types - Workers
export type { WorkerListParams } from "./workers";

// Types - Models
export type { ModelListParams } from "./models";

// Types - Deployments
export type {
  DeploymentListParams,
  DeploymentLogsResponse,
} from "./deployments";

// Types - Auth
export type { ChangePasswordRequest } from "./auth";

// Types - Users
export type { UserListParams } from "./users";

// Types - API Keys
export type {
  ApiKeyListParams,
  ApiKeyStats,
  ModelUsageStats,
  ModelStatsResponse,
} from "./apiKeys";

// Types - Images
export type { ImageListParams, ImageSearchResult } from "./images";

// Types - Containers
export type { ContainerListParams } from "./containers";

// Types - Storage
export type {
  StorageCategory,
  DiskUsage,
  Volume,
  PruneRequest,
  PruneResult,
} from "./storage";

// Types - Model Files
export type {
  ModelFileListParams,
  ModelFileDeleteResponse,
} from "./modelFiles";

// Types - System
export type { BackupInfo, BackupListResponse, MessageResponse } from "./system";

// Types - HuggingFace
export type {
  HFModelInfo,
  VRAMEstimate,
  HFModelFile,
  HFSearchResult,
  ModelFormatInfo,
} from "./huggingface";

// Types - Ollama
export type { OllamaModel, OllamaTagInfo } from "./ollama";

// Types - Conversations
export type {
  ConversationMessage,
  Conversation,
  ConversationListParams,
  ConversationCreateData,
  AddMessagesData,
} from "./conversations";

// Types - Apps
export type {
  AppDefinition,
  DeployedApp,
  AppDeployRequest,
  DeployProgress,
  MonitoringServiceStatus,
  MonitoringStatus,
} from "./apps";

// Types - Headscale
export type {
  HeadscaleStatus,
  HeadscaleNode,
  PreauthKeyResponse,
  HeadscaleProgress,
} from "./headscale";

// Types - Semantic Router
export type { SemanticRouterStatus } from "./semanticRouter";
