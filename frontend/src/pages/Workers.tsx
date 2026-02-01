import { useEffect, useState, useCallback } from "react";
import {
  Button,
  Card,
  Collapse,
  Form,
  Input,
  InputNumber,
  Modal,
  Space,
  Table,
  Tag,
  message,
  Popconfirm,
  Descriptions,
  Progress,
  Row,
  Col,
  Statistic,
  Tooltip,
  Typography,
  Alert,
} from "antd";
import {
  PlusOutlined,
  DeleteOutlined,
  EyeOutlined,
  EditOutlined,
  ReloadOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  SyncOutlined,
  CloseCircleOutlined,
  DatabaseOutlined,
  DesktopOutlined,
  CopyOutlined,
} from "@ant-design/icons";
import { workersApi, modelFilesApi } from "../services/api";
import type {
  Worker,
  GPUInfo,
  SystemInfo,
  ModelFileView,
  RegistrationToken,
} from "../types";
import { useResponsive } from "../hooks";
import { useAuth } from "../contexts/AuthContext";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";

dayjs.extend(utc);

const { Text } = Typography;

const REFRESH_INTERVAL = 5000; // 5 seconds

export default function Workers() {
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [detailModal, setDetailModal] = useState<Worker | null>(null);
  const [modelFiles, setModelFiles] = useState<ModelFileView[]>([]);
  const [modelFilesLoading, setModelFilesLoading] = useState(false);
  const [modelFilesModal, setModelFilesModal] = useState<Worker | null>(null);
  const [editModal, setEditModal] = useState<Worker | null>(null);
  const [editForm] = Form.useForm();
  const [editing, setEditing] = useState(false);
  const [form] = Form.useForm();
  const { isMobile } = useResponsive();
  const { canEdit } = useAuth();

  // Registration token state
  const [generatedToken, setGeneratedToken] =
    useState<RegistrationToken | null>(null);
  const [generatingToken, setGeneratingToken] = useState(false);

  const fetchWorkers = useCallback(async () => {
    try {
      const response = await workersApi.list();
      setWorkers(response.items);
    } catch (error) {
      console.error("Failed to fetch workers:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWorkers();

    // Auto refresh
    const interval = setInterval(fetchWorkers, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchWorkers]);

  // Update detail modal data when workers update
  useEffect(() => {
    if (detailModal) {
      const updated = workers.find((w) => w.id === detailModal.id);
      if (updated) {
        setDetailModal(updated);
      }
    }
  }, [workers, detailModal]);

  const handleGenerateToken = async (values: {
    name: string;
    expires_in_hours: number;
  }) => {
    setGeneratingToken(true);
    try {
      const token = await workersApi.createToken({
        name: values.name,
        expires_in_hours: values.expires_in_hours,
      });
      setGeneratedToken(token);
      message.success("Registration token generated");
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to generate token");
    } finally {
      setGeneratingToken(false);
    }
  };

  const handleCopyCommand = async () => {
    if (!generatedToken?.docker_command) return;
    const text = generatedToken.docker_command;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        message.success("Command copied to clipboard");
        return;
      }
      // Fallback for non-secure contexts (HTTP)
      const textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.style.position = "fixed";
      textArea.style.left = "-999999px";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      const successful = document.execCommand("copy");
      document.body.removeChild(textArea);
      if (successful) {
        message.success("Command copied to clipboard");
      } else {
        message.error("Failed to copy");
      }
    } catch (err) {
      message.error("Failed to copy");
    }
  };

  const handleCloseAddModal = () => {
    setModalOpen(false);
    setGeneratedToken(null);
    form.resetFields();
  };

  const handleDelete = async (id: number) => {
    try {
      await workersApi.delete(id);
      message.success("Worker deleted successfully");
      fetchWorkers();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to delete worker");
    }
  };

  const openEditModal = (worker: Worker) => {
    setEditModal(worker);
    editForm.setFieldsValue({
      name: worker.name,
      description: worker.description || "",
    });
  };

  const handleEditWorker = async (values: {
    name: string;
    description?: string;
  }) => {
    if (!editModal) return;
    setEditing(true);
    try {
      await workersApi.update(editModal.id, {
        name: values.name,
        description: values.description,
      });
      message.success("Worker updated successfully");
      setEditModal(null);
      editForm.resetFields();
      fetchWorkers();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to update worker");
    } finally {
      setEditing(false);
    }
  };

  const [registeringLocal, setRegisteringLocal] = useState(false);

  const handleRegisterLocal = async () => {
    setRegisteringLocal(true);
    try {
      const result = await workersApi.registerLocal();
      message.success(
        `Docker worker "${result.worker_name}" started (${result.container_id}). It will appear shortly.`,
      );
      // Refresh workers list after a short delay to allow worker to register
      setTimeout(fetchWorkers, 3000);
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(
        err.response?.data?.detail || "Failed to spawn local Docker worker",
      );
    } finally {
      setRegisteringLocal(false);
    }
  };

  // Open model files modal for a worker
  const openModelFilesModal = (worker: Worker) => {
    setModelFilesModal(worker);
    fetchModelFilesForWorker(worker.id);
  };

  // Fetch model files for a specific worker (for modal)
  const fetchModelFilesForWorker = async (workerId: number) => {
    setModelFilesLoading(true);
    try {
      const response = await modelFilesApi.list({ worker_id: workerId });
      setModelFiles(response.items);
    } catch (error) {
      console.error("Failed to fetch model files:", error);
    } finally {
      setModelFilesLoading(false);
    }
  };

  const formatBytes = (bytes: number) => {
    const gb = bytes / (1024 * 1024 * 1024);
    return `${gb.toFixed(1)} GB`;
  };

  const getUtilizationColor = (util: number) => {
    if (util < 30) return "#52c41a";
    if (util < 70) return "#1677ff";
    return "#faad14";
  };

  const getTemperatureColor = (temp: number) => {
    if (temp < 50) return "#52c41a";
    if (temp < 70) return "#faad14";
    return "#ff4d4f";
  };

  // Delete model files from worker
  const handleDeleteModelFile = async (
    modelId: number,
    workerId: number,
    modelName: string,
  ) => {
    try {
      await modelFilesApi.delete(modelId, workerId);
      message.success(`Model "${modelName}" files deleted`);
      if (modelFilesModal) {
        fetchModelFilesForWorker(modelFilesModal.id);
      }
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(
        err.response?.data?.detail || "Failed to delete model files",
      );
    }
  };

  // Get status tag for model file
  const getModelStatusTag = (status: string) => {
    switch (status) {
      case "ready":
        return (
          <Tag icon={<CheckCircleOutlined />} color="success">
            Ready
          </Tag>
        );
      case "starting":
        return (
          <Tag icon={<SyncOutlined spin />} color="processing">
            Starting
          </Tag>
        );
      case "downloading":
        return (
          <Tag icon={<SyncOutlined spin />} color="warning">
            Downloading
          </Tag>
        );
      case "stopped":
        return (
          <Tag icon={<CloseCircleOutlined />} color="default">
            Stopped
          </Tag>
        );
      default:
        return <Tag>{status}</Tag>;
    }
  };

  // Mobile columns (simplified)
  const mobileColumns = [
    {
      title: "Worker",
      key: "worker",
      render: (_: unknown, record: Worker) => (
        <div>
          <div style={{ fontWeight: 500 }}>{record.name}</div>
          <div style={{ fontSize: 12, color: "#888" }}>{record.address}</div>
          <div
            style={{ marginTop: 4, display: "flex", flexWrap: "wrap", gap: 4 }}
          >
            <Tag color={record.status === "online" ? "green" : "default"}>
              {record.status.toUpperCase()}
            </Tag>
            {record.os_type === "darwin" && (
              <>
                <Tag color="blue" style={{ fontSize: 10 }}>
                  macOS
                </Tag>
                {record.status === "online" &&
                  !record.capabilities?.ollama_running && (
                    <Tag color="error" style={{ fontSize: 10 }}>
                      No Ollama
                    </Tag>
                  )}
              </>
            )}
          </div>
          <div style={{ fontSize: 12, color: "#888", marginTop: 4 }}>
            {record.status === "offline"
              ? "Offline"
              : `${record.gpu_info?.length || 0} GPU(s) · ${record.deployment_count} deployments`}
          </div>
        </div>
      ),
    },
    {
      title: "Actions",
      key: "actions",
      width: 100,
      render: (_: unknown, record: Worker) => (
        <Space size={4}>
          <Button
            type="text"
            size="small"
            icon={<DatabaseOutlined />}
            onClick={() => openModelFilesModal(record)}
          />
          <Button
            type="text"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => setDetailModal(record)}
          />
          {canEdit && (
            <>
              <Button
                type="text"
                size="small"
                icon={<EditOutlined />}
                onClick={() => openEditModal(record)}
              />
              <Popconfirm
                title="Delete this worker?"
                description="This action cannot be undone."
                onConfirm={() => handleDelete(record.id)}
                okText="Delete"
                okButtonProps={{ danger: true }}
              >
                <Button
                  type="text"
                  size="small"
                  danger
                  icon={<DeleteOutlined />}
                />
              </Popconfirm>
            </>
          )}
        </Space>
      ),
    },
  ];

  // Desktop columns (full)
  const desktopColumns = [
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
    },
    {
      title: "Address",
      dataIndex: "address",
      key: "address",
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      render: (status: string, record: Worker) => {
        const colorMap: Record<string, string> = {
          online: "green",
          offline: "default",
          error: "red",
        };
        return (
          <div>
            <Tag color={colorMap[status]}>{status.toUpperCase()}</Tag>
            {record.os_type === "darwin" && (
              <div style={{ marginTop: 4 }}>
                <Tag color="blue" style={{ fontSize: 10 }}>
                  macOS
                </Tag>
                {record.status === "online" &&
                  (record.capabilities?.ollama_running ? (
                    <Tag color="success" style={{ fontSize: 10 }}>
                      Ollama
                    </Tag>
                  ) : record.capabilities?.ollama ? (
                    <Tooltip title="Ollama installed but not running. Run: brew services start ollama">
                      <Tag
                        color="warning"
                        style={{ fontSize: 10, cursor: "pointer" }}
                      >
                        Ollama Stopped
                      </Tag>
                    </Tooltip>
                  ) : (
                    <Tooltip title="Ollama not installed. Run: brew install ollama && brew services start ollama">
                      <Tag
                        color="error"
                        style={{ fontSize: 10, cursor: "pointer" }}
                      >
                        No Ollama
                      </Tag>
                    </Tooltip>
                  ))}
              </div>
            )}
          </div>
        );
      },
    },
    {
      title: "GPUs",
      dataIndex: "gpu_info",
      key: "gpu_info",
      width: 350,
      render: (gpuInfo: GPUInfo[] | null, record: Worker) => {
        if (record.status === "offline") {
          return (
            <Text type="secondary" style={{ fontSize: 12 }}>
              Offline
            </Text>
          );
        }
        if (!gpuInfo || gpuInfo.length === 0) return "-";
        return (
          <div>
            {gpuInfo.map((gpu, idx) => {
              const memUsed = gpu.memory_total - gpu.memory_free;
              const memPercent = Math.round((memUsed / gpu.memory_total) * 100);
              return (
                <div
                  key={idx}
                  style={{ marginBottom: idx < gpuInfo.length - 1 ? 8 : 0 }}
                >
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <span style={{ fontSize: 12 }}>{gpu.name}</span>
                    <Space size={4}>
                      <Tooltip title="GPU Utilization">
                        <Tag
                          color={getUtilizationColor(gpu.utilization || 0)}
                          style={{ margin: 0 }}
                        >
                          <ThunderboltOutlined /> {gpu.utilization || 0}%
                        </Tag>
                      </Tooltip>
                      <Tooltip title="Temperature">
                        <Tag
                          color={getTemperatureColor(gpu.temperature || 0)}
                          style={{ margin: 0 }}
                        >
                          {gpu.temperature || 0}°C
                        </Tag>
                      </Tooltip>
                    </Space>
                  </div>
                  <Tooltip
                    title={`${formatBytes(memUsed)} / ${formatBytes(gpu.memory_total)}`}
                  >
                    <Progress
                      percent={memPercent}
                      size="small"
                      strokeColor={memPercent > 90 ? "#ff4d4f" : "#1677ff"}
                      showInfo={false}
                      style={{ marginTop: 4 }}
                    />
                  </Tooltip>
                </div>
              );
            })}
          </div>
        );
      },
    },
    {
      title: "System",
      dataIndex: "system_info",
      key: "system_info",
      width: 200,
      render: (systemInfo: SystemInfo | null, record: Worker) => {
        if (record.status === "offline") {
          return (
            <Text type="secondary" style={{ fontSize: 12 }}>
              Offline
            </Text>
          );
        }
        if (!systemInfo) return "-";
        return (
          <div style={{ fontSize: 12 }}>
            {systemInfo.cpu && (
              <Tooltip
                title={`${systemInfo.cpu.count} cores @ ${Math.round(systemInfo.cpu.freq_mhz)} MHz`}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    marginBottom: 4,
                  }}
                >
                  <span>CPU</span>
                  <Tag
                    color={getUtilizationColor(systemInfo.cpu.percent)}
                    style={{ margin: 0 }}
                  >
                    {Math.round(systemInfo.cpu.percent)}%
                  </Tag>
                </div>
              </Tooltip>
            )}
            {systemInfo.memory && (
              <Tooltip
                title={`${formatBytes(systemInfo.memory.used)} / ${formatBytes(systemInfo.memory.total)}`}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    marginBottom: 4,
                  }}
                >
                  <span>RAM</span>
                  <Tag
                    color={getUtilizationColor(systemInfo.memory.percent)}
                    style={{ margin: 0 }}
                  >
                    {Math.round(systemInfo.memory.percent)}%
                  </Tag>
                </div>
              </Tooltip>
            )}
            {systemInfo.disk && (
              <Tooltip
                title={`${formatBytes(systemInfo.disk.used)} / ${formatBytes(systemInfo.disk.total)}`}
              >
                <div
                  style={{ display: "flex", justifyContent: "space-between" }}
                >
                  <span>Disk</span>
                  <Tag
                    color={getUtilizationColor(systemInfo.disk.percent)}
                    style={{ margin: 0 }}
                  >
                    {Math.round(systemInfo.disk.percent)}%
                  </Tag>
                </div>
              </Tooltip>
            )}
          </div>
        );
      },
    },
    {
      title: "Deployments",
      dataIndex: "deployment_count",
      key: "deployment_count",
      width: 100,
    },
    {
      title: "Last Heartbeat",
      dataIndex: "last_heartbeat",
      key: "last_heartbeat",
      render: (time: string | null) =>
        time ? dayjs.utc(time).local().format("HH:mm:ss") : "-",
    },
    {
      title: "Actions",
      key: "actions",
      width: 150,
      render: (_: unknown, record: Worker) => (
        <Space size={4}>
          <Tooltip title="Model Files">
            <Button
              type="text"
              icon={<DatabaseOutlined />}
              onClick={() => openModelFilesModal(record)}
              size="small"
            />
          </Tooltip>
          <Tooltip title="Details">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => setDetailModal(record)}
              size="small"
            />
          </Tooltip>
          {canEdit && (
            <>
              <Tooltip title="Edit">
                <Button
                  type="text"
                  icon={<EditOutlined />}
                  onClick={() => openEditModal(record)}
                  size="small"
                />
              </Tooltip>
              <Popconfirm
                title="Delete this worker?"
                description="This action cannot be undone."
                onConfirm={() => handleDelete(record.id)}
                okText="Delete"
                okButtonProps={{ danger: true }}
              >
                <Button
                  type="text"
                  danger
                  icon={<DeleteOutlined />}
                  size="small"
                />
              </Popconfirm>
            </>
          )}
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Card
        style={{ borderRadius: 12 }}
        title={
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              alignItems: "center",
              gap: 8,
            }}
          >
            <span>Workers</span>
            <Tag color="processing" style={{ borderRadius: 6 }}>
              {workers.length}
            </Tag>
            <Tag color="success" style={{ borderRadius: 6 }}>
              {workers.filter((w) => w.status === "online").length} online
            </Tag>
          </div>
        }
        extra={
          <Space wrap>
            <Button
              icon={<ReloadOutlined />}
              onClick={fetchWorkers}
              size={isMobile ? "small" : "middle"}
            >
              {!isMobile && "Refresh"}
            </Button>
            {canEdit && (
              <>
                <Button
                  icon={<DesktopOutlined />}
                  onClick={handleRegisterLocal}
                  loading={registeringLocal}
                  size={isMobile ? "small" : "middle"}
                >
                  {isMobile ? "Local" : "Add Local"}
                </Button>
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={() => setModalOpen(true)}
                  size={isMobile ? "small" : "middle"}
                >
                  {isMobile ? "Add" : "Add Worker"}
                </Button>
              </>
            )}
          </Space>
        }
      >
        <Table
          dataSource={workers}
          columns={isMobile ? mobileColumns : desktopColumns}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 10 }}
          scroll={isMobile ? undefined : { x: 900 }}
          size={isMobile ? "small" : "middle"}
        />
      </Card>

      <Modal
        title="Add Worker"
        open={modalOpen}
        onCancel={handleCloseAddModal}
        footer={null}
        width={isMobile ? "100%" : 600}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        {!generatedToken ? (
          <Form
            form={form}
            layout="vertical"
            onFinish={handleGenerateToken}
            initialValues={{ expires_in_hours: 24 }}
          >
            <Alert
              message="Worker Registration"
              description="Generate a registration token and run the Docker command on your GPU machine to add a worker."
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
            />
            <Form.Item
              name="name"
              label="Worker Name"
              rules={[{ required: true, message: "Please enter worker name" }]}
            >
              <Input placeholder="e.g., gpu-worker-01" />
            </Form.Item>
            <Form.Item
              name="expires_in_hours"
              label="Token Validity (hours)"
              rules={[
                { required: true, message: "Please enter token validity" },
              ]}
            >
              <InputNumber min={1} max={168} style={{ width: "100%" }} />
            </Form.Item>
            <Form.Item>
              <Space>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={generatingToken}
                >
                  Generate Token
                </Button>
                <Button onClick={handleCloseAddModal}>Cancel</Button>
              </Space>
            </Form.Item>
          </Form>
        ) : (
          <div>
            <Alert
              message="Token Generated Successfully"
              description={
                <span>
                  Copy and run the command below on your GPU machine. Token
                  expires at{" "}
                  <strong>
                    {dayjs
                      .utc(generatedToken.expires_at)
                      .local()
                      .format("YYYY-MM-DD HH:mm")}
                  </strong>
                </span>
              }
              type="success"
              showIcon
              style={{ marginBottom: 16 }}
            />
            <div style={{ marginBottom: 16 }}>
              <Text strong>Docker Command:</Text>
              <div
                style={{
                  marginTop: 8,
                  padding: 12,
                  backgroundColor: "#1e1e1e",
                  borderRadius: 6,
                  position: "relative",
                }}
              >
                <pre
                  style={{
                    margin: 0,
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-all",
                    color: "#d4d4d4",
                    fontSize: 12,
                    fontFamily: "monospace",
                  }}
                >
                  {generatedToken.docker_command}
                </pre>
                <Button
                  type="primary"
                  icon={<CopyOutlined />}
                  size="small"
                  onClick={handleCopyCommand}
                  style={{ position: "absolute", top: 8, right: 8 }}
                >
                  Copy
                </Button>
              </div>
            </div>
            <Collapse
              size="small"
              style={{ marginBottom: 16 }}
              items={[
                {
                  key: "local-docker",
                  label: "Local Docker Build (after ./scripts/build-local.sh)",
                  children: (
                    <div
                      style={{
                        padding: 12,
                        backgroundColor: "#1e1e1e",
                        borderRadius: 6,
                        position: "relative",
                      }}
                    >
                      <pre
                        style={{
                          margin: 0,
                          whiteSpace: "pre-wrap",
                          wordBreak: "break-all",
                          color: "#d4d4d4",
                          fontSize: 12,
                          fontFamily: "monospace",
                        }}
                      >
                        {generatedToken.docker_command?.replace(
                          "infinirc/lmstack-worker:latest",
                          "infinirc/lmstack-worker:local",
                        )}
                      </pre>
                      <Button
                        type="primary"
                        icon={<CopyOutlined />}
                        size="small"
                        onClick={() => {
                          const localCmd =
                            generatedToken.docker_command?.replace(
                              "infinirc/lmstack-worker:latest",
                              "infinirc/lmstack-worker:local",
                            );
                          if (localCmd) {
                            if (navigator.clipboard && window.isSecureContext) {
                              navigator.clipboard.writeText(localCmd);
                              message.success("Command copied to clipboard");
                            } else {
                              const textArea =
                                document.createElement("textarea");
                              textArea.value = localCmd;
                              textArea.style.position = "fixed";
                              textArea.style.left = "-999999px";
                              document.body.appendChild(textArea);
                              textArea.focus();
                              textArea.select();
                              document.execCommand("copy");
                              document.body.removeChild(textArea);
                              message.success("Command copied to clipboard");
                            }
                          }
                        }}
                        style={{ position: "absolute", top: 8, right: 8 }}
                      >
                        Copy
                      </Button>
                    </div>
                  ),
                },
                {
                  key: "windows-local",
                  label: "Windows (Same Machine as Backend)",
                  children: (
                    <div
                      style={{
                        padding: 12,
                        backgroundColor: "#1e1e1e",
                        borderRadius: 6,
                        position: "relative",
                      }}
                    >
                      <pre
                        style={{
                          margin: 0,
                          whiteSpace: "pre-wrap",
                          wordBreak: "break-all",
                          color: "#d4d4d4",
                          fontSize: 12,
                          fontFamily: "monospace",
                        }}
                      >
                        {generatedToken.docker_command?.replace(
                          /BACKEND_URL=http:\/\/[^:]+:52000/,
                          "BACKEND_URL=http://host.docker.internal:52000",
                        )}
                      </pre>
                      <Button
                        type="primary"
                        icon={<CopyOutlined />}
                        size="small"
                        onClick={() => {
                          const windowsCmd =
                            generatedToken.docker_command?.replace(
                              /BACKEND_URL=http:\/\/[^:]+:52000/,
                              "BACKEND_URL=http://host.docker.internal:52000",
                            );
                          if (windowsCmd) {
                            if (navigator.clipboard && window.isSecureContext) {
                              navigator.clipboard.writeText(windowsCmd);
                              message.success("Command copied to clipboard");
                            } else {
                              const textArea =
                                document.createElement("textarea");
                              textArea.value = windowsCmd;
                              textArea.style.position = "fixed";
                              textArea.style.left = "-999999px";
                              document.body.appendChild(textArea);
                              textArea.focus();
                              textArea.select();
                              document.execCommand("copy");
                              document.body.removeChild(textArea);
                              message.success("Command copied to clipboard");
                            }
                          }
                        }}
                        style={{ position: "absolute", top: 8, right: 8 }}
                      >
                        Copy
                      </Button>
                    </div>
                  ),
                },
                {
                  key: "dev-python",
                  label: "Development Mode (Python)",
                  children: (
                    <div
                      style={{
                        padding: 12,
                        backgroundColor: "#1e1e1e",
                        borderRadius: 6,
                        position: "relative",
                      }}
                    >
                      <pre
                        style={{
                          margin: 0,
                          whiteSpace: "pre-wrap",
                          wordBreak: "break-all",
                          color: "#d4d4d4",
                          fontSize: 12,
                          fontFamily: "monospace",
                        }}
                      >
                        {`cd worker
pip install -r requirements.txt
python agent.py \\
  --name ${generatedToken.name} \\
  --server-url ${window.location.protocol}//${window.location.hostname}:52000 \\
  --registration-token ${generatedToken.token}`}
                      </pre>
                      <Button
                        type="primary"
                        icon={<CopyOutlined />}
                        size="small"
                        onClick={() => {
                          const devCommand = `cd worker
pip install -r requirements.txt
python agent.py \\
  --name ${generatedToken.name} \\
  --server-url ${window.location.protocol}//${window.location.hostname}:52000 \\
  --registration-token ${generatedToken.token}`;
                          if (navigator.clipboard && window.isSecureContext) {
                            navigator.clipboard.writeText(devCommand);
                            message.success("Command copied to clipboard");
                          } else {
                            const textArea = document.createElement("textarea");
                            textArea.value = devCommand;
                            textArea.style.position = "fixed";
                            textArea.style.left = "-999999px";
                            document.body.appendChild(textArea);
                            textArea.focus();
                            textArea.select();
                            document.execCommand("copy");
                            document.body.removeChild(textArea);
                            message.success("Command copied to clipboard");
                          }
                        }}
                        style={{ position: "absolute", top: 8, right: 8 }}
                      >
                        Copy
                      </Button>
                    </div>
                  ),
                },
                {
                  key: "macos",
                  label: "macOS (Apple Silicon)",
                  children: (
                    <div>
                      <Alert
                        message="macOS Requirements"
                        description={
                          <div>
                            <p style={{ margin: "8px 0" }}>
                              Before running the worker on macOS, install Ollama
                              for LLM inference:
                            </p>
                            <pre
                              style={{
                                background: "#1e1e1e",
                                color: "#d4d4d4",
                                padding: 8,
                                borderRadius: 4,
                                fontSize: 12,
                              }}
                            >
                              brew install ollama{"\n"}
                              brew services start ollama
                            </pre>
                          </div>
                        }
                        type="warning"
                        showIcon
                        style={{ marginBottom: 12 }}
                      />
                      <div
                        style={{
                          padding: 12,
                          backgroundColor: "#1e1e1e",
                          borderRadius: 6,
                          position: "relative",
                        }}
                      >
                        <pre
                          style={{
                            margin: 0,
                            whiteSpace: "pre-wrap",
                            wordBreak: "break-all",
                            color: "#d4d4d4",
                            fontSize: 12,
                            fontFamily: "monospace",
                          }}
                        >
                          {generatedToken.docker_command
                            ?.replace("--gpus all --privileged ", "")
                            .replace("--network host ", "-p 52001:52001 ")
                            .replace("-v /:/host:ro ", "")}
                        </pre>
                        <Button
                          type="primary"
                          icon={<CopyOutlined />}
                          size="small"
                          onClick={() => {
                            const macCmd = generatedToken.docker_command
                              ?.replace("--gpus all --privileged ", "")
                              .replace("--network host ", "-p 52001:52001 ")
                              .replace("-v /:/host:ro ", "");
                            if (macCmd) {
                              if (
                                navigator.clipboard &&
                                window.isSecureContext
                              ) {
                                navigator.clipboard.writeText(macCmd);
                                message.success("Command copied to clipboard");
                              } else {
                                const textArea =
                                  document.createElement("textarea");
                                textArea.value = macCmd;
                                textArea.style.position = "fixed";
                                textArea.style.left = "-999999px";
                                document.body.appendChild(textArea);
                                textArea.focus();
                                textArea.select();
                                document.execCommand("copy");
                                document.body.removeChild(textArea);
                                message.success("Command copied to clipboard");
                              }
                            }
                          }}
                          style={{ position: "absolute", top: 8, right: 8 }}
                        >
                          Copy
                        </Button>
                      </div>
                    </div>
                  ),
                },
              ]}
            />
            <div style={{ textAlign: "right" }}>
              <Button onClick={handleCloseAddModal}>Done</Button>
            </div>
          </div>
        )}
      </Modal>

      <Modal
        title={`Worker: ${detailModal?.name}`}
        open={!!detailModal}
        onCancel={() => setDetailModal(null)}
        footer={null}
        width={isMobile ? "100%" : 800}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        {detailModal && (
          <div>
            <Descriptions
              column={isMobile ? 1 : 2}
              bordered
              size="small"
              labelStyle={isMobile ? { width: 100 } : undefined}
            >
              <Descriptions.Item label="ID">{detailModal.id}</Descriptions.Item>
              <Descriptions.Item label="Status">
                <Tag
                  color={detailModal.status === "online" ? "green" : "default"}
                >
                  {detailModal.status.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Address">
                {detailModal.address}
              </Descriptions.Item>
              <Descriptions.Item label="Deployments">
                {detailModal.deployment_count}
              </Descriptions.Item>
              <Descriptions.Item label="OS">
                <Tag
                  color={detailModal.os_type === "darwin" ? "blue" : "default"}
                >
                  {detailModal.os_type === "darwin"
                    ? "macOS"
                    : detailModal.os_type === "windows"
                      ? "Windows"
                      : "Linux"}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Ollama">
                {detailModal.capabilities?.ollama ? (
                  detailModal.capabilities?.ollama_running ? (
                    <Tag icon={<CheckCircleOutlined />} color="success">
                      Running
                    </Tag>
                  ) : (
                    <Tag icon={<CloseCircleOutlined />} color="warning">
                      Installed (Not Running)
                    </Tag>
                  )
                ) : (
                  <Tag icon={<CloseCircleOutlined />} color="default">
                    Not Installed
                  </Tag>
                )}
              </Descriptions.Item>
              <Descriptions.Item label="Created">
                {dayjs
                  .utc(detailModal.created_at)
                  .local()
                  .format("YYYY-MM-DD HH:mm:ss")}
              </Descriptions.Item>
              <Descriptions.Item label="Last Heartbeat">
                {detailModal.last_heartbeat
                  ? dayjs
                      .utc(detailModal.last_heartbeat)
                      .local()
                      .format("YYYY-MM-DD HH:mm:ss")
                  : "-"}
              </Descriptions.Item>
            </Descriptions>

            {detailModal.system_info && (
              <div style={{ marginTop: 16 }}>
                <h4>System Information</h4>
                <Row gutter={[16, 16]}>
                  {detailModal.system_info.cpu && (
                    <Col xs={24} sm={8}>
                      <Card size="small" title="CPU">
                        <Statistic
                          title={`${detailModal.system_info.cpu.count} cores`}
                          value={Math.round(
                            detailModal.system_info.cpu.percent,
                          )}
                          suffix="%"
                          valueStyle={{
                            color: getUtilizationColor(
                              detailModal.system_info.cpu.percent,
                            ),
                            fontSize: isMobile ? 18 : 24,
                          }}
                        />
                        <div
                          style={{ fontSize: 12, color: "#888", marginTop: 4 }}
                        >
                          {Math.round(detailModal.system_info.cpu.freq_mhz)} MHz
                        </div>
                      </Card>
                    </Col>
                  )}
                  {detailModal.system_info.memory && (
                    <Col xs={24} sm={8}>
                      <Card size="small" title="Memory (RAM)">
                        <Statistic
                          title={`${formatBytes(detailModal.system_info.memory.used)} / ${formatBytes(detailModal.system_info.memory.total)}`}
                          value={Math.round(
                            detailModal.system_info.memory.percent,
                          )}
                          suffix="%"
                          valueStyle={{
                            color: getUtilizationColor(
                              detailModal.system_info.memory.percent,
                            ),
                            fontSize: isMobile ? 18 : 24,
                          }}
                        />
                        <Progress
                          percent={Math.round(
                            detailModal.system_info.memory.percent,
                          )}
                          strokeColor={
                            detailModal.system_info.memory.percent > 90
                              ? "#ff4d4f"
                              : "#1677ff"
                          }
                          showInfo={false}
                          style={{ marginTop: 8 }}
                        />
                      </Card>
                    </Col>
                  )}
                  {detailModal.system_info.disk && (
                    <Col xs={24} sm={8}>
                      <Card size="small" title="Disk">
                        <Statistic
                          title={`${formatBytes(detailModal.system_info.disk.used)} / ${formatBytes(detailModal.system_info.disk.total)}`}
                          value={Math.round(
                            detailModal.system_info.disk.percent,
                          )}
                          suffix="%"
                          valueStyle={{
                            color: getUtilizationColor(
                              detailModal.system_info.disk.percent,
                            ),
                            fontSize: isMobile ? 18 : 24,
                          }}
                        />
                        <Progress
                          percent={Math.round(
                            detailModal.system_info.disk.percent,
                          )}
                          strokeColor={
                            detailModal.system_info.disk.percent > 90
                              ? "#ff4d4f"
                              : "#1677ff"
                          }
                          showInfo={false}
                          style={{ marginTop: 8 }}
                        />
                      </Card>
                    </Col>
                  )}
                </Row>
              </div>
            )}

            {detailModal.gpu_info && detailModal.gpu_info.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <h4>GPU Information</h4>
                <Row gutter={[16, 16]}>
                  {detailModal.gpu_info.map((gpu, idx) => {
                    const memUsed = gpu.memory_total - gpu.memory_free;
                    const memPercent = Math.round(
                      (memUsed / gpu.memory_total) * 100,
                    );
                    return (
                      <Col xs={24} sm={12} key={idx}>
                        <Card
                          size="small"
                          title={`GPU ${gpu.index}: ${gpu.name}`}
                        >
                          <Row gutter={16}>
                            <Col span={12}>
                              <Statistic
                                title="Utilization"
                                value={gpu.utilization || 0}
                                suffix="%"
                                valueStyle={{
                                  color: getUtilizationColor(
                                    gpu.utilization || 0,
                                  ),
                                  fontSize: isMobile ? 18 : 24,
                                }}
                              />
                            </Col>
                            <Col span={12}>
                              <Statistic
                                title="Temperature"
                                value={gpu.temperature || 0}
                                suffix="°C"
                                valueStyle={{
                                  color: getTemperatureColor(
                                    gpu.temperature || 0,
                                  ),
                                  fontSize: isMobile ? 18 : 24,
                                }}
                              />
                            </Col>
                          </Row>
                          <div style={{ marginTop: 12 }}>
                            <div
                              style={{
                                marginBottom: 4,
                                fontSize: isMobile ? 12 : 14,
                              }}
                            >
                              Memory: {formatBytes(memUsed)} /{" "}
                              {formatBytes(gpu.memory_total)}
                            </div>
                            <Progress
                              percent={memPercent}
                              strokeColor={
                                memPercent > 90 ? "#ff4d4f" : "#1677ff"
                              }
                              status="active"
                            />
                          </div>
                        </Card>
                      </Col>
                    );
                  })}
                </Row>
              </div>
            )}
          </div>
        )}
      </Modal>

      {/* Model Files Modal */}
      <Modal
        title={
          <Space>
            <DatabaseOutlined />
            <span>Model Files: {modelFilesModal?.name}</span>
          </Space>
        }
        open={!!modelFilesModal}
        onCancel={() => setModelFilesModal(null)}
        footer={null}
        width={isMobile ? "100%" : 600}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        {modelFilesLoading ? (
          <div style={{ textAlign: "center", padding: 24 }}>Loading...</div>
        ) : modelFiles.length === 0 ? (
          <div style={{ textAlign: "center", padding: 24, color: "#888" }}>
            No model files on this worker
          </div>
        ) : (
          <Table
            dataSource={modelFiles}
            rowKey={(mf) => `${mf.model_id}-${mf.worker_id}`}
            size="small"
            pagination={false}
            columns={[
              {
                title: "Model",
                key: "model",
                render: (_: unknown, mf: ModelFileView) => (
                  <div>
                    <Text strong>{mf.model_name}</Text>
                    <br />
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      {mf.model_source}
                    </Text>
                  </div>
                ),
              },
              {
                title: "Status",
                key: "status",
                width: 100,
                render: (_: unknown, mf: ModelFileView) =>
                  getModelStatusTag(mf.status),
              },
              {
                title: "Deployments",
                key: "deployments",
                width: 100,
                render: (_: unknown, mf: ModelFileView) => (
                  <Tag color={mf.running_count > 0 ? "green" : "default"}>
                    {mf.running_count} / {mf.deployment_count}
                  </Tag>
                ),
              },
              {
                title: "",
                key: "actions",
                width: 60,
                render: (_: unknown, mf: ModelFileView) =>
                  canEdit && (
                    <Popconfirm
                      title="Delete model files?"
                      description={
                        mf.running_count > 0
                          ? "Warning: This model has running deployments!"
                          : "This will delete cached model files."
                      }
                      onConfirm={() =>
                        handleDeleteModelFile(
                          mf.model_id,
                          mf.worker_id,
                          mf.model_name,
                        )
                      }
                      okText="Delete"
                      okButtonProps={{ danger: true }}
                    >
                      <Button
                        type="text"
                        size="small"
                        danger
                        icon={<DeleteOutlined />}
                      />
                    </Popconfirm>
                  ),
              },
            ]}
          />
        )}
      </Modal>

      {/* Edit Worker Modal */}
      <Modal
        title={`Edit Worker: ${editModal?.name}`}
        open={!!editModal}
        onCancel={() => {
          setEditModal(null);
          editForm.resetFields();
        }}
        footer={null}
        width={isMobile ? "100%" : 400}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        <Form form={editForm} layout="vertical" onFinish={handleEditWorker}>
          <Form.Item
            name="name"
            label="Worker Name"
            rules={[{ required: true, message: "Please enter worker name" }]}
          >
            <Input placeholder="e.g., gpu-worker-01" />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <Input.TextArea placeholder="Optional description" rows={3} />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={editing}>
                Save
              </Button>
              <Button
                onClick={() => {
                  setEditModal(null);
                  editForm.resetFields();
                }}
              >
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
