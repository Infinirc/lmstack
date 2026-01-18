import { useEffect, useState, useCallback, useRef } from "react";
import {
  Button,
  Card,
  Form,
  Input,
  Modal,
  Space,
  Table,
  Tag,
  message,
  Popconfirm,
  Select,
  Tooltip,
  Switch,
  Tabs,
  Descriptions,
} from "antd";
import {
  PlusOutlined,
  DeleteOutlined,
  EyeOutlined,
  ReloadOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  CodeOutlined,
  FileTextOutlined,
  DesktopOutlined,
  FullscreenOutlined,
  FullscreenExitOutlined,
  VerticalAlignBottomOutlined,
} from "@ant-design/icons";
import { containersApi, workersApi, imagesApi } from "../services/api";
import type {
  Container,
  ContainerState,
  Worker,
  ContainerImage,
} from "../types";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import utc from "dayjs/plugin/utc";

dayjs.extend(relativeTime);
dayjs.extend(utc);

const REFRESH_INTERVAL = 5000;

import { useAuth } from "../contexts/AuthContext";

function useResponsive() {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return { isMobile };
}

const STATE_COLORS: Record<ContainerState, string> = {
  running: "green",
  created: "blue",
  paused: "orange",
  restarting: "cyan",
  removing: "orange",
  exited: "default",
  dead: "red",
};

export default function Containers() {
  const [containers, setContainers] = useState<Container[]>([]);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [images, setImages] = useState<ContainerImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [detailModal, setDetailModal] = useState<Container | null>(null);
  const [logsModal, setLogsModal] = useState<Container | null>(null);
  const [logs, setLogs] = useState<string>("");
  const [logsLoading, setLogsLoading] = useState(false);
  const [logsFullscreen, setLogsFullscreen] = useState(true);
  const [logsAutoRefresh, setLogsAutoRefresh] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  const logsRef = useRef<HTMLPreElement>(null);
  const [execModal, setExecModal] = useState<Container | null>(null);
  const [execCommand, setExecCommand] = useState("");
  const [execResult, setExecResult] = useState<string>("");
  const [execLoading, setExecLoading] = useState(false);
  const [workerFilter, setWorkerFilter] = useState<number | undefined>();
  const [stateFilter, setStateFilter] = useState<ContainerState | undefined>();
  const [showAll, setShowAll] = useState(true);
  const [managedOnly, setManagedOnly] = useState(false);
  const [createForm] = Form.useForm();
  const { isMobile } = useResponsive();
  const { canEdit } = useAuth();

  const fetchData = useCallback(async () => {
    try {
      const [containersRes, workersRes, imagesRes] = await Promise.all([
        containersApi.list({
          worker_id: workerFilter,
          state: stateFilter,
          all: showAll,
          managed_only: managedOnly,
        }),
        workersApi.list(),
        imagesApi.list(),
      ]);
      setContainers(containersRes.items);
      setWorkers(workersRes.items);
      setImages(imagesRes.items);
    } catch (error) {
      console.error("Failed to fetch containers:", error);
    } finally {
      setLoading(false);
    }
  }, [workerFilter, stateFilter, showAll, managedOnly]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Auto-refresh logs when modal is open and auto-refresh is enabled
  useEffect(() => {
    if (!logsModal || !logsAutoRefresh) return;

    const interval = setInterval(async () => {
      try {
        const response = await containersApi.getLogs(
          logsModal.id,
          logsModal.worker_id,
          { tail: 500 },
        );
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

  const handleStart = async (container: Container) => {
    try {
      await containersApi.start(container.id, container.worker_id);
      message.success("Container started");
      fetchData();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to start container");
    }
  };

  const handleStop = async (container: Container) => {
    try {
      await containersApi.stop(container.id, container.worker_id);
      message.success("Container stopped");
      fetchData();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to stop container");
    }
  };

  const handleRestart = async (container: Container) => {
    try {
      await containersApi.restart(container.id, container.worker_id);
      message.success("Container restarted");
      fetchData();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(
        err.response?.data?.detail || "Failed to restart container",
      );
    }
  };

  const handleDelete = async (container: Container) => {
    try {
      await containersApi.delete(container.id, container.worker_id, true);
      message.success("Container deleted");
      fetchData();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to delete container");
    }
  };

  const handleCreate = async (values: {
    worker_id: number;
    name: string;
    image: string;
    command?: string;
    ports?: string;
    gpu_ids?: number[];
  }) => {
    try {
      const ports = values.ports
        ? values.ports.split(",").map((p) => {
            const [host, container] = p.trim().split(":");
            return {
              container_port: parseInt(container || host),
              host_port: parseInt(host),
            };
          })
        : undefined;

      await containersApi.create({
        worker_id: values.worker_id,
        name: values.name,
        image: values.image,
        command: values.command ? values.command.split(" ") : undefined,
        ports,
        gpu_ids: values.gpu_ids,
      });
      message.success("Container created");
      setCreateModalOpen(false);
      createForm.resetFields();
      fetchData();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to create container");
    }
  };

  const handleViewLogs = async (container: Container) => {
    setLogsModal(container);
    setLogsLoading(true);
    try {
      const response = await containersApi.getLogs(
        container.id,
        container.worker_id,
        { tail: 500 },
      );
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
      const response = await containersApi.getLogs(
        logsModal.id,
        logsModal.worker_id,
        { tail: 500 },
      );
      setLogs(response.logs);
    } catch {
      setLogs("Failed to fetch logs");
    } finally {
      setLogsLoading(false);
    }
  };

  const handleExec = async () => {
    if (!execModal || !execCommand.trim()) return;
    setExecLoading(true);
    try {
      const result = await containersApi.exec(
        execModal.id,
        execModal.worker_id,
        {
          command: execCommand.split(" "),
        },
      );
      setExecResult(
        `Exit code: ${result.exit_code}\n\n--- stdout ---\n${result.stdout}\n\n--- stderr ---\n${result.stderr}`,
      );
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      setExecResult(
        `Error: ${err.response?.data?.detail || "Failed to execute command"}`,
      );
    } finally {
      setExecLoading(false);
    }
  };

  const mobileColumns = [
    {
      title: "Container",
      key: "container",
      render: (_: unknown, record: Container) => (
        <div>
          <div style={{ fontWeight: 500 }}>
            {record.name}
            {record.is_managed && (
              <Tag color="blue" style={{ marginLeft: 8, fontSize: 10 }}>
                Managed
              </Tag>
            )}
          </div>
          <code style={{ fontSize: 10, color: "#888" }}>{record.image}</code>
          <div style={{ marginTop: 4 }}>
            <Tag color={STATE_COLORS[record.state]}>
              {record.state.toUpperCase()}
            </Tag>
          </div>
          <div style={{ fontSize: 11, color: "#888", marginTop: 4 }}>
            <DesktopOutlined style={{ marginRight: 4 }} />
            {record.worker_name}
          </div>
          {record.deployment_name && (
            <div style={{ fontSize: 11, color: "#888" }}>
              Deployment: {record.deployment_name}
            </div>
          )}
        </div>
      ),
    },
    {
      title: "",
      key: "actions",
      width: 100,
      render: (_: unknown, record: Container) => (
        <Space direction="vertical" size={4}>
          <Space size={4}>
            {canEdit && record.state === "running" ? (
              <Button
                type="text"
                size="small"
                icon={<PauseCircleOutlined />}
                onClick={() => handleStop(record)}
              />
            ) : canEdit &&
              (record.state === "exited" || record.state === "created") ? (
              <Button
                type="text"
                size="small"
                icon={<PlayCircleOutlined />}
                onClick={() => handleStart(record)}
              />
            ) : null}
            <Button
              type="text"
              size="small"
              icon={<FileTextOutlined />}
              onClick={() => handleViewLogs(record)}
            />
          </Space>
          <Space size={4}>
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => setDetailModal(record)}
            />
            {canEdit && (
              <Popconfirm
                title="Delete container?"
                onConfirm={() => handleDelete(record)}
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
            )}
          </Space>
        </Space>
      ),
    },
  ];

  const desktopColumns = [
    {
      title: "Container",
      key: "container",
      render: (_: unknown, record: Container) => (
        <div>
          <div style={{ fontWeight: 500 }}>
            {record.name}
            {record.is_managed && (
              <Tag color="blue" style={{ marginLeft: 8 }}>
                Managed
              </Tag>
            )}
          </div>
          <code style={{ fontSize: 11, color: "#888" }}>{record.image}</code>
          {record.deployment_name && (
            <div style={{ fontSize: 11, color: "#888" }}>
              Deployment: {record.deployment_name}
            </div>
          )}
        </div>
      ),
    },
    {
      title: "Worker",
      key: "worker",
      render: (_: unknown, record: Container) => (
        <Space>
          <DesktopOutlined />
          {record.worker_name}
        </Space>
      ),
    },
    {
      title: "State",
      key: "state",
      render: (_: unknown, record: Container) => (
        <div>
          <Tag color={STATE_COLORS[record.state]}>
            {record.state.toUpperCase()}
          </Tag>
          <div style={{ fontSize: 11, color: "#888" }}>{record.status}</div>
        </div>
      ),
    },
    {
      title: "Ports",
      key: "ports",
      render: (_: unknown, record: Container) =>
        record.ports.length > 0
          ? record.ports
              .map((p) => `${p.host_port}:${p.container_port}`)
              .join(", ")
          : "-",
    },
    {
      title: "Actions",
      key: "actions",
      width: 200,
      render: (_: unknown, record: Container) => (
        <Space>
          {canEdit && record.state === "running" ? (
            <>
              <Tooltip title="Stop">
                <Button
                  type="text"
                  icon={<PauseCircleOutlined />}
                  onClick={() => handleStop(record)}
                />
              </Tooltip>
              <Tooltip title="Restart">
                <Button
                  type="text"
                  icon={<ReloadOutlined />}
                  onClick={() => handleRestart(record)}
                />
              </Tooltip>
            </>
          ) : canEdit &&
            (record.state === "exited" || record.state === "created") ? (
            <Tooltip title="Start">
              <Button
                type="text"
                icon={<PlayCircleOutlined />}
                onClick={() => handleStart(record)}
              />
            </Tooltip>
          ) : null}
          <Tooltip title="Logs">
            <Button
              type="text"
              icon={<FileTextOutlined />}
              onClick={() => handleViewLogs(record)}
            />
          </Tooltip>
          <Tooltip title="Details">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => setDetailModal(record)}
            />
          </Tooltip>
          {canEdit && record.state === "running" && (
            <Tooltip title="Terminal">
              <Button
                type="text"
                icon={<CodeOutlined />}
                onClick={() => setExecModal(record)}
              />
            </Tooltip>
          )}
          {canEdit && (
            <Popconfirm
              title="Delete container?"
              description="This will force remove the container."
              onConfirm={() => handleDelete(record)}
              okText="Delete"
              okButtonProps={{ danger: true }}
            >
              <Tooltip title="Delete">
                <Button type="text" danger icon={<DeleteOutlined />} />
              </Tooltip>
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ];

  const runningCount = containers.filter((c) => c.state === "running").length;

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
            <span>Containers</span>
            <Tag color="processing" style={{ borderRadius: 6 }}>
              {containers.length}
            </Tag>
            <Tag color="green" style={{ borderRadius: 6 }}>
              {runningCount} running
            </Tag>
          </div>
        }
        extra={
          <Space wrap>
            <Switch
              checkedChildren="Managed"
              unCheckedChildren="All"
              checked={managedOnly}
              onChange={setManagedOnly}
              size={isMobile ? "small" : "default"}
            />
            <Button
              icon={<ReloadOutlined />}
              onClick={fetchData}
              size={isMobile ? "small" : "middle"}
            >
              {!isMobile && "Refresh"}
            </Button>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setCreateModalOpen(true)}
              size={isMobile ? "small" : "middle"}
            >
              {isMobile ? "New" : "Create"}
            </Button>
          </Space>
        }
      >
        <Space wrap style={{ marginBottom: 16 }}>
          <Select
            placeholder="Worker"
            allowClear
            style={{ width: isMobile ? 100 : 150 }}
            size={isMobile ? "small" : "middle"}
            onChange={(value) => setWorkerFilter(value)}
            options={workers.map((w) => ({ label: w.name, value: w.id }))}
          />
          <Select
            placeholder="State"
            allowClear
            style={{ width: isMobile ? 100 : 120 }}
            size={isMobile ? "small" : "middle"}
            onChange={(value) => setStateFilter(value)}
            options={[
              { label: "Running", value: "running" },
              { label: "Exited", value: "exited" },
              { label: "Created", value: "created" },
              { label: "Paused", value: "paused" },
            ]}
          />
          <Switch
            checkedChildren="All"
            unCheckedChildren="Running"
            checked={showAll}
            onChange={setShowAll}
            size={isMobile ? "small" : "default"}
          />
        </Space>

        <Table
          dataSource={containers}
          columns={isMobile ? mobileColumns : desktopColumns}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 10 }}
          size={isMobile ? "small" : "middle"}
        />
      </Card>

      {/* Create Container Modal */}
      <Modal
        title="Create Container"
        open={createModalOpen}
        onCancel={() => {
          setCreateModalOpen(false);
          createForm.resetFields();
        }}
        footer={null}
        width={isMobile ? "100%" : 600}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        <Form form={createForm} layout="vertical" onFinish={handleCreate}>
          <Form.Item
            name="worker_id"
            label="Worker"
            rules={[{ required: true, message: "Please select a worker" }]}
          >
            <Select
              placeholder="Select worker"
              options={workers
                .filter((w) => w.status === "online")
                .map((w) => ({ label: w.name, value: w.id }))}
            />
          </Form.Item>
          <Form.Item
            name="name"
            label="Container Name"
            rules={[{ required: true, message: "Please enter container name" }]}
          >
            <Input placeholder="my-container" />
          </Form.Item>
          <Form.Item
            name="image"
            label="Image"
            rules={[{ required: true, message: "Please enter image name" }]}
          >
            <Select
              showSearch
              placeholder="Select or enter image"
              options={images.map((img) => ({
                label: img.full_name,
                value: img.full_name,
              }))}
              filterOption={(input, option) =>
                (option?.label ?? "")
                  .toLowerCase()
                  .includes(input.toLowerCase())
              }
            />
          </Form.Item>
          <Form.Item
            name="command"
            label="Command"
            extra="Optional command to run"
          >
            <Input placeholder="python main.py" />
          </Form.Item>
          <Form.Item
            name="ports"
            label="Port Mappings"
            extra="Format: host:container, comma separated (e.g., 8080:80, 443:443)"
          >
            <Input placeholder="8080:80" />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" icon={<PlusOutlined />}>
                Create
              </Button>
              <Button onClick={() => setCreateModalOpen(false)}>Cancel</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Container Detail Modal */}
      <Modal
        title={`Container: ${detailModal?.name}`}
        open={!!detailModal}
        onCancel={() => setDetailModal(null)}
        footer={null}
        width={isMobile ? "100%" : 700}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        {detailModal && (
          <Tabs
            defaultActiveKey="info"
            items={[
              {
                key: "info",
                label: "Info",
                children: (
                  <Descriptions column={isMobile ? 1 : 2} bordered size="small">
                    <Descriptions.Item label="ID">
                      <code>{detailModal.id.substring(0, 12)}</code>
                    </Descriptions.Item>
                    <Descriptions.Item label="State">
                      <Tag color={STATE_COLORS[detailModal.state]}>
                        {detailModal.state.toUpperCase()}
                      </Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="Image">
                      {detailModal.image}
                    </Descriptions.Item>
                    <Descriptions.Item label="Worker">
                      {detailModal.worker_name}
                    </Descriptions.Item>
                    <Descriptions.Item label="Status">
                      {detailModal.status}
                    </Descriptions.Item>
                    <Descriptions.Item label="Created">
                      {dayjs
                        .utc(detailModal.created_at)
                        .local()
                        .format("YYYY-MM-DD HH:mm:ss")}
                    </Descriptions.Item>
                    {detailModal.ports.length > 0 && (
                      <Descriptions.Item label="Ports" span={2}>
                        {detailModal.ports
                          .map(
                            (p) =>
                              `${p.host_port}:${p.container_port}/${p.protocol}`,
                          )
                          .join(", ")}
                      </Descriptions.Item>
                    )}
                    {detailModal.deployment_name && (
                      <Descriptions.Item label="Deployment" span={2}>
                        {detailModal.deployment_name}
                      </Descriptions.Item>
                    )}
                  </Descriptions>
                ),
              },
              {
                key: "env",
                label: "Environment",
                children: (
                  <div
                    style={{
                      background: "#f5f5f5",
                      padding: 12,
                      borderRadius: 8,
                      maxHeight: 300,
                      overflow: "auto",
                    }}
                  >
                    {detailModal.env && detailModal.env.length > 0 ? (
                      <pre style={{ margin: 0, fontSize: 12 }}>
                        {detailModal.env.join("\n")}
                      </pre>
                    ) : (
                      <span style={{ color: "#888" }}>
                        No environment variables
                      </span>
                    )}
                  </div>
                ),
              },
            ]}
          />
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
          setLogs("");
          setLogsFullscreen(true);
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

      {/* Exec Modal */}
      <Modal
        title={`Terminal: ${execModal?.name}`}
        open={!!execModal}
        onCancel={() => {
          setExecModal(null);
          setExecCommand("");
          setExecResult("");
        }}
        footer={null}
        width={isMobile ? "100%" : 800}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        <Space.Compact style={{ width: "100%", marginBottom: 16 }}>
          <Input
            placeholder="Enter command (e.g., ls -la)"
            value={execCommand}
            onChange={(e) => setExecCommand(e.target.value)}
            onPressEnter={handleExec}
            style={{ fontFamily: "monospace" }}
          />
          <Button type="primary" onClick={handleExec} loading={execLoading}>
            Run
          </Button>
        </Space.Compact>
        <div
          style={{
            background: "#1e1e1e",
            color: "#d4d4d4",
            padding: 16,
            borderRadius: 8,
            height: 300,
            overflow: "auto",
            fontFamily: "monospace",
            fontSize: 12,
            whiteSpace: "pre-wrap",
            wordBreak: "break-all",
          }}
        >
          {execResult || "Output will appear here..."}
        </div>
      </Modal>
    </div>
  );
}
