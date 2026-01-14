/**
 * Status Utilities
 *
 * Provides consistent status color mapping across the application.
 *
 * @module utils/status
 */

/**
 * Status color mapping for deployments
 */
export const DEPLOYMENT_STATUS_COLORS: Record<string, string> = {
  running: "green",
  pending: "blue",
  downloading: "cyan",
  starting: "blue",
  stopping: "orange",
  stopped: "default",
  error: "red",
};

/**
 * Status color mapping for workers
 */
export const WORKER_STATUS_COLORS: Record<string, string> = {
  online: "green",
  offline: "default",
  error: "red",
};

/**
 * Status color mapping for containers
 */
export const CONTAINER_STATUS_COLORS: Record<string, string> = {
  running: "green",
  created: "blue",
  restarting: "orange",
  paused: "orange",
  exited: "default",
  dead: "red",
};

/**
 * Get status color for deployment
 */
export function getDeploymentStatusColor(status: string): string {
  return DEPLOYMENT_STATUS_COLORS[status] || "default";
}

/**
 * Get status color for worker
 */
export function getWorkerStatusColor(status: string): string {
  return WORKER_STATUS_COLORS[status] || "default";
}

/**
 * Get status color for container
 */
export function getContainerStatusColor(status: string): string {
  return CONTAINER_STATUS_COLORS[status] || "default";
}
