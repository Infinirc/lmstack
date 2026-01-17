/**
 * Deploy Apps Page
 *
 * Deploy and manage applications like Open WebUI that integrate with LMStack.
 */
import { useEffect, useState, useCallback, useRef } from "react";
import {
  Button,
  Card,
  Modal,
  Tag,
  message,
  Popconfirm,
  Row,
  Col,
  Typography,
  Select,
  Empty,
  Tooltip,
  Progress,
  Switch,
  Space,
} from "antd";
import {
  RocketOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  LinkOutlined,
  CheckCircleOutlined,
  SyncOutlined,
  CloseCircleOutlined,
  LoadingOutlined,
  ExclamationCircleOutlined,
  CloudDownloadOutlined,
  FileTextOutlined,
  ReloadOutlined,
  FullscreenOutlined,
  FullscreenExitOutlined,
  VerticalAlignBottomOutlined,
} from "@ant-design/icons";
import { appsApi, workersApi } from "../services/api";
import type { Worker } from "../types";
import type {
  AppDefinition,
  DeployedApp,
  DeployProgress,
} from "../services/api";
import { useResponsive } from "../hooks";
import Loading from "../components/Loading";

// App logos
import openWebuiLogo from "../assets/apps/open-webui.webp";
import n8nLogo from "../assets/apps/n8n.png";
import flowiseLogo from "../assets/apps/flowise-icon.png";
import anythingllmLogo from "../assets/apps/anythingllm.jpeg";
import lobechatLogo from "../assets/apps/lobechat.webp";

const { Title, Text, Paragraph } = Typography;

// Map app type to logo
const appLogos: Record<string, string> = {
  "open-webui": openWebuiLogo,
  n8n: n8nLogo,
  flowise: flowiseLogo,
  anythingllm: anythingllmLogo,
  lobechat: lobechatLogo,
};

const REFRESH_INTERVAL = 5000;

const statusConfig: Record<
  string,
  { color: string; icon: React.ReactNode; label: string }
> = {
  running: {
    color: "success",
    icon: <CheckCircleOutlined />,
    label: "Running",
  },
  pending: { color: "default", icon: <SyncOutlined spin />, label: "Pending" },
  pulling: {
    color: "processing",
    icon: <CloudDownloadOutlined />,
    label: "Pulling Image",
  },
  starting: {
    color: "processing",
    icon: <LoadingOutlined />,
    label: "Starting",
  },
  stopping: {
    color: "warning",
    icon: <SyncOutlined spin />,
    label: "Stopping",
  },
  stopped: {
    color: "default",
    icon: <PauseCircleOutlined />,
    label: "Stopped",
  },
  error: { color: "error", icon: <CloseCircleOutlined />, label: "Error" },
};

export default function DeployApps() {
  const [availableApps, setAvailableApps] = useState<AppDefinition[]>([]);
  const [deployedApps, setDeployedApps] = useState<DeployedApp[]>([]);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [loading, setLoading] = useState(true);
  const [deployModalOpen, setDeployModalOpen] = useState(false);
  const [selectedApp, setSelectedApp] = useState<AppDefinition | null>(null);
  const [selectedWorker, setSelectedWorker] = useState<number | null>(null);
  const [useProxy, setUseProxy] = useState(true);
  const [deploying, setDeploying] = useState(false);
  const [progressMap, setProgressMap] = useState<
    Record<number, DeployProgress>
  >({});
  const [logsModal, setLogsModal] = useState<{
    id: number;
    name: string;
  } | null>(null);
  const [logs, setLogs] = useState<string>("");
  const [logsLoading, setLogsLoading] = useState(false);
  const [logsFullscreen, setLogsFullscreen] = useState(true);
  const [logsAutoRefresh, setLogsAutoRefresh] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  const logsRef = useRef<HTMLPreElement>(null);
  const { isMobile } = useResponsive();

  const fetchData = useCallback(async () => {
    try {
      const [availableRes, deployedRes, workersRes] = await Promise.all([
        appsApi.listAvailable(),
        appsApi.list(),
        workersApi.list({ status: "online" }),
      ]);
      setAvailableApps(availableRes.items);
      setDeployedApps(deployedRes.items);
      setWorkers(workersRes.items);
    } catch (error) {
      console.error("Failed to fetch data:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Poll progress for deploying apps
  useEffect(() => {
    const deployingApps = deployedApps.filter((app) =>
      ["pending", "pulling", "starting"].includes(app.status),
    );

    if (deployingApps.length === 0) return;

    const pollProgress = async () => {
      const updates: Record<number, DeployProgress> = {};
      for (const app of deployingApps) {
        try {
          const progress = await appsApi.getProgress(app.id);
          updates[app.id] = progress;
        } catch {
          // Ignore errors
        }
      }
      setProgressMap((prev) => ({ ...prev, ...updates }));
    };

    pollProgress();
    const interval = setInterval(pollProgress, 1000); // Poll every second during deployment
    return () => clearInterval(interval);
  }, [deployedApps]);

  // Auto-refresh logs when modal is open and auto-refresh is enabled
  useEffect(() => {
    if (!logsModal || !logsAutoRefresh) return;

    const interval = setInterval(async () => {
      try {
        const response = await appsApi.getLogs(logsModal.id, 500);
        setLogs(response.logs);
      } catch {
        // Silently fail on auto-refresh
      }
    }, 2000); // Refresh every 2 seconds

    return () => clearInterval(interval);
  }, [logsModal, logsAutoRefresh]);

  // Auto-scroll logs to bottom
  useEffect(() => {
    if (autoScroll && logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const handleViewLogs = async (app: DeployedApp) => {
    setLogsModal({ id: app.id, name: app.name });
    setLogsLoading(true);
    try {
      const response = await appsApi.getLogs(app.id, 500);
      setLogs(response.logs);
    } catch {
      setLogs("Failed to fetch logs");
    } finally {
      setLogsLoading(false);
    }
  };

  const refreshLogs = async () => {
    if (!logsModal) return;
    setLogsLoading(true);
    try {
      const response = await appsApi.getLogs(logsModal.id, 500);
      setLogs(response.logs);
    } catch {
      setLogs("Failed to fetch logs");
    } finally {
      setLogsLoading(false);
    }
  };

  const handleDeploy = async () => {
    if (!selectedApp || !selectedWorker) {
      message.error("Please select an app and worker");
      return;
    }

    setDeploying(true);
    try {
      await appsApi.deploy({
        app_type: selectedApp.type,
        worker_id: selectedWorker,
        use_proxy: useProxy,
      });
      message.success(`${selectedApp.name} deployment started`);
      setDeployModalOpen(false);
      setSelectedApp(null);
      setSelectedWorker(null);
      setUseProxy(true);
      fetchData();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to deploy app");
    } finally {
      setDeploying(false);
    }
  };

  const handleStop = async (app: DeployedApp) => {
    try {
      await appsApi.stop(app.id);
      message.success("App stopped");
      fetchData();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to stop app");
    }
  };

  const handleStart = async (app: DeployedApp) => {
    try {
      await appsApi.start(app.id);
      message.success("App started");
      fetchData();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to start app");
    }
  };

  const handleDelete = async (app: DeployedApp) => {
    try {
      await appsApi.delete(app.id);
      message.success("App deleted");
      fetchData();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to delete app");
    }
  };

  const openDeployModal = (app: AppDefinition) => {
    // Check if already deployed
    const existing = deployedApps.find((d) => d.app_type === app.type);
    if (existing) {
      message.warning(`${app.name} is already deployed`);
      return;
    }
    setSelectedApp(app);
    setDeployModalOpen(true);
  };

  const getAppUrl = (app: DeployedApp) => {
    // Build URL based on proxy setting
    if (!app.port) return null;

    if (app.use_proxy) {
      // Use current hostname (LMStack IP) + app port
      return `http://${window.location.hostname}:${app.port}`;
    } else {
      // Direct connection - for local workers, use current hostname
      // Check if worker_address is a private/internal IP (Docker network, etc.)
      if (app.worker_address) {
        const workerHost = app.worker_address.split(":")[0];
        const isInternalIp =
          workerHost.startsWith("172.") ||
          workerHost.startsWith("10.") ||
          workerHost.startsWith("192.168.") ||
          workerHost === "localhost" ||
          workerHost === "127.0.0.1";

        if (isInternalIp) {
          // Local worker - use current browser hostname
          return `http://${window.location.hostname}:${app.port}`;
        } else {
          // Remote worker - use worker's IP
          return `http://${workerHost}:${app.port}`;
        }
      }
      return null;
    }
  };

  if (loading) {
    return (
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "60vh",
        }}
      >
        <Loading size="large" />
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto" }}>
      <div style={{ marginBottom: 24 }}>
        <h1
          style={{
            margin: 0,
            fontSize: isMobile ? 22 : 28,
            fontWeight: 600,
            letterSpacing: "-0.02em",
          }}
        >
          Deploy Apps
        </h1>
        <Text type="secondary" style={{ fontSize: isMobile ? 13 : 14 }}>
          Deploy companion applications that integrate with LMStack
        </Text>
      </div>

      {/* Available Apps */}
      <Title level={5}>Available Apps</Title>
      <Row gutter={[16, 16]} style={{ marginBottom: 32 }}>
        {availableApps.map((app) => {
          const isDeployed = deployedApps.some((d) => d.app_type === app.type);
          return (
            <Col xs={24} sm={12} lg={8} key={app.type}>
              <Card
                hoverable={!isDeployed}
                style={{ height: "100%", opacity: isDeployed ? 0.7 : 1 }}
              >
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    height: "100%",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 12,
                      marginBottom: 12,
                    }}
                  >
                    {appLogos[app.type] && (
                      <img
                        src={appLogos[app.type]}
                        alt={app.name}
                        style={{
                          width: 32,
                          height: 32,
                          borderRadius: app.type === "flowise" ? 0 : 6,
                          objectFit: "contain",
                        }}
                      />
                    )}
                    <Title level={5} style={{ margin: 0 }}>
                      {app.name}
                    </Title>
                  </div>
                  <Paragraph
                    type="secondary"
                    style={{ flex: 1, marginBottom: 16 }}
                  >
                    {app.description}
                  </Paragraph>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {app.image.split("/").pop()?.split(":")[0]}
                    </Text>
                    {isDeployed ? (
                      <Tag color="green">Deployed</Tag>
                    ) : (
                      <Button
                        type="primary"
                        icon={<RocketOutlined />}
                        onClick={() => openDeployModal(app)}
                        disabled={workers.length === 0}
                      >
                        Deploy
                      </Button>
                    )}
                  </div>
                </div>
              </Card>
            </Col>
          );
        })}

        {availableApps.length === 0 && (
          <Col span={24}>
            <Empty description="No apps available" />
          </Col>
        )}
      </Row>

      {/* Deployed Apps */}
      {deployedApps.length > 0 && (
        <>
          <Title level={5}>Deployed Apps</Title>
          <Row gutter={[16, 16]}>
            {deployedApps.map((app) => {
              const status = statusConfig[app.status] || statusConfig.error;
              const appUrl = getAppUrl(app);
              const progress = progressMap[app.id];
              const isDeploying = ["pending", "pulling", "starting"].includes(
                app.status,
              );

              return (
                <Col xs={24} sm={12} lg={8} key={app.id}>
                  <Card>
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: 12,
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "flex-start",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 12,
                          }}
                        >
                          {appLogos[app.app_type] && (
                            <img
                              src={appLogos[app.app_type]}
                              alt={app.name}
                              style={{
                                width: 32,
                                height: 32,
                                borderRadius:
                                  app.app_type === "flowise" ? 0 : 6,
                                objectFit: "contain",
                              }}
                            />
                          )}
                          <div>
                            <Title level={5} style={{ margin: 0 }}>
                              {app.name}
                            </Title>
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              Worker: {app.worker_name || `#${app.worker_id}`}
                            </Text>
                          </div>
                        </div>
                        <Tag color={status.color} icon={status.icon}>
                          {status.label}
                        </Tag>
                      </div>

                      {/* Progress indicator during deployment */}
                      {isDeploying && (
                        <div style={{ marginTop: 8 }}>
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 8,
                              marginBottom: 4,
                            }}
                          >
                            {app.status === "pulling" && (
                              <CloudDownloadOutlined />
                            )}
                            {app.status === "starting" && <LoadingOutlined />}
                            {app.status === "pending" && <SyncOutlined spin />}
                            <Text type="secondary" style={{ fontSize: 13 }}>
                              {app.status === "starting"
                                ? "Starting app (first startup may take 1-3 minutes)..."
                                : progress?.stage === "unknown" ||
                                    !progress?.message
                                  ? app.status === "pulling"
                                    ? "Pulling image..."
                                    : "Preparing..."
                                  : progress.message}
                            </Text>
                          </div>
                          {/* Use indeterminate style for starting stage or when no real progress data */}
                          {app.status === "starting" ||
                          !progress ||
                          progress.stage === "unknown" ||
                          (app.status === "pulling" &&
                            progress.progress === 0) ? (
                            <Progress
                              percent={100}
                              size="small"
                              status="active"
                              showInfo={false}
                              strokeColor={{ from: "#108ee9", to: "#87d068" }}
                            />
                          ) : (
                            <Progress
                              percent={progress.progress}
                              size="small"
                              status={
                                progress.stage === "error"
                                  ? "exception"
                                  : "active"
                              }
                              showInfo={false}
                            />
                          )}
                        </div>
                      )}

                      {/* Only show status_message in red for error status */}
                      {app.status === "error" && app.status_message && (
                        <Tooltip title={app.status_message}>
                          <Text type="danger" ellipsis style={{ fontSize: 12 }}>
                            <ExclamationCircleOutlined
                              style={{ marginRight: 4 }}
                            />
                            {app.status_message}
                          </Text>
                        </Tooltip>
                      )}

                      {app.status === "running" && appUrl && (
                        <div>
                          <Button
                            type="link"
                            icon={<LinkOutlined />}
                            href={appUrl}
                            target="_blank"
                            style={{ padding: 0, height: "auto" }}
                          >
                            Open {app.name}
                          </Button>
                          {app.port && (
                            <Text
                              type="secondary"
                              style={{ fontSize: 12, marginLeft: 8 }}
                            >
                              Port: {app.port}
                            </Text>
                          )}
                        </div>
                      )}

                      <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                        <Button
                          icon={<FileTextOutlined />}
                          onClick={() => handleViewLogs(app)}
                          size="small"
                          disabled={!app.container_id}
                        >
                          Logs
                        </Button>
                        {app.status === "running" ? (
                          <Button
                            icon={<PauseCircleOutlined />}
                            onClick={() => handleStop(app)}
                            size="small"
                          >
                            Stop
                          </Button>
                        ) : app.status === "stopped" ||
                          app.status === "error" ? (
                          <Button
                            type="primary"
                            icon={<PlayCircleOutlined />}
                            onClick={() => handleStart(app)}
                            size="small"
                          >
                            Start
                          </Button>
                        ) : null}

                        <Popconfirm
                          title="Delete this app?"
                          description={
                            ["pulling", "starting"].includes(app.status)
                              ? "Deployment is in progress. Delete anyway?"
                              : "This will remove the container and associated API key."
                          }
                          onConfirm={() => handleDelete(app)}
                          okText="Delete"
                          okButtonProps={{ danger: true }}
                        >
                          <Button
                            danger
                            icon={<DeleteOutlined />}
                            size="small"
                            disabled={app.status === "stopping"}
                          >
                            Delete
                          </Button>
                        </Popconfirm>
                      </div>
                    </div>
                  </Card>
                </Col>
              );
            })}
          </Row>
        </>
      )}

      {/* Deploy Modal */}
      <Modal
        title={`Deploy ${selectedApp?.name || "App"}`}
        open={deployModalOpen}
        onOk={handleDeploy}
        onCancel={() => {
          setDeployModalOpen(false);
          setSelectedApp(null);
          setSelectedWorker(null);
          setUseProxy(true);
        }}
        okText="Deploy"
        okButtonProps={{ loading: deploying, disabled: !selectedWorker }}
      >
        <div style={{ marginBottom: 16 }}>
          <Text type="secondary">
            Select a worker to deploy {selectedApp?.name}. An API key will be
            automatically created.
          </Text>
        </div>

        <div style={{ marginBottom: 16 }}>
          <Text strong>Worker</Text>
          <Select
            style={{ width: "100%", marginTop: 8 }}
            placeholder="Select a worker"
            value={selectedWorker}
            onChange={(value) => {
              setSelectedWorker(value);
              // Auto-disable proxy for local workers (same machine as LMStack)
              const worker = workers.find((w) => w.id === value);
              if (worker) {
                const isLocalWorker = worker.labels?.type === "local";
                if (isLocalWorker) {
                  setUseProxy(false);
                } else {
                  setUseProxy(true);
                }
              }
            }}
            options={workers.map((w) => ({
              value: w.id,
              label: `${w.name} (${w.address})`,
            }))}
          />
        </div>

        {/* Hide proxy option for local workers (same machine as LMStack) */}
        {(() => {
          const worker = workers.find((w) => w.id === selectedWorker);
          const isLocalWorker = worker?.labels?.type === "local";
          if (isLocalWorker) return null;
          return (
            <div style={{ marginBottom: 16 }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <div>
                  <Text strong>Use LMStack Proxy</Text>
                  <div>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {useProxy
                        ? "Access app through LMStack server (recommended)"
                        : "Connect directly to worker (requires network access to worker)"}
                    </Text>
                  </div>
                </div>
                <Switch checked={useProxy} onChange={setUseProxy} />
              </div>
            </div>
          );
        })()}

        {workers.length === 0 && (
          <div style={{ marginTop: 16 }}>
            <Text type="danger">
              No online workers available. Please add a worker first.
            </Text>
          </div>
        )}
      </Modal>

      {/* Logs Modal */}
      <Modal
        title={
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              alignItems: "center",
              gap: 8,
            }}
          >
            <span>Logs: {logsModal?.name}</span>
            <Tag
              color={logsAutoRefresh ? "green" : "default"}
              style={{ cursor: "pointer" }}
              onClick={() => setLogsAutoRefresh(!logsAutoRefresh)}
            >
              Auto-refresh {logsAutoRefresh ? "ON" : "OFF"}
            </Tag>
          </div>
        }
        open={!!logsModal}
        onCancel={() => {
          setLogsModal(null);
          setLogsFullscreen(false);
          setLogsAutoRefresh(true);
          setAutoScroll(true);
        }}
        footer={
          <Space wrap>
            <Button
              icon={<ReloadOutlined spin={logsLoading} />}
              onClick={refreshLogs}
              size="small"
            >
              Refresh
            </Button>
            <Button
              type={autoScroll ? "primary" : "default"}
              icon={<VerticalAlignBottomOutlined />}
              onClick={() => setAutoScroll(!autoScroll)}
              size="small"
            >
              Auto-scroll
            </Button>
            <Button
              icon={
                logsFullscreen ? (
                  <FullscreenExitOutlined />
                ) : (
                  <FullscreenOutlined />
                )
              }
              onClick={() => setLogsFullscreen(!logsFullscreen)}
              size="small"
            >
              {logsFullscreen ? "Minimize" : "Fullscreen"}
            </Button>
          </Space>
        }
        width={logsFullscreen ? "95vw" : isMobile ? "100%" : 1000}
        style={
          logsFullscreen
            ? { top: 20 }
            : isMobile
              ? { top: 20, maxWidth: "100%", margin: "0 8px" }
              : undefined
        }
      >
        <pre
          ref={logsRef}
          style={{
            background: "#1e1e1e",
            color: "#d4d4d4",
            padding: isMobile ? 12 : 16,
            borderRadius: 4,
            height: logsFullscreen
              ? "calc(100vh - 200px)"
              : isMobile
                ? 300
                : 500,
            overflow: "auto",
            fontSize: isMobile ? 11 : 13,
            fontFamily: "'Fira Code', 'Consolas', monospace",
            lineHeight: 1.5,
            whiteSpace: "pre-wrap",
            wordBreak: "break-all",
          }}
        >
          {logsLoading && !logs ? "Loading..." : logs || "No logs available"}
        </pre>
      </Modal>
    </div>
  );
}
