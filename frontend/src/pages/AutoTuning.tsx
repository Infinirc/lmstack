import { useEffect, useState, useCallback } from "react";
import {
  Button,
  Card,
  Form,
  Modal,
  Select,
  Space,
  Table,
  Tag,
  message,
  Progress,
  Typography,
  Empty,
  Tooltip,
  Radio,
  Tabs,
  Statistic,
  Row,
  Col,
  Input,
  Divider,
  Alert,
  Popconfirm,
} from "antd";
import {
  PlusOutlined,
  ReloadOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  LoadingOutlined,
  ExperimentOutlined,
  DatabaseOutlined,
  RocketOutlined,
  BarChartOutlined,
  HistoryOutlined,
  ApiOutlined,
  DeleteOutlined,
  CommentOutlined,
} from "@ant-design/icons";
import { useAppTheme } from "../hooks/useTheme";
import { workersApi, modelsApi } from "../services/api";
import { deploymentsApi } from "../api";
import { api } from "../api/client";
import type { Worker, LLMModel, Deployment } from "../types";
import { useResponsive } from "../hooks";
import { useAuth } from "../contexts/AuthContext";
import {
  CHAT_PANEL_STORAGE_KEY,
  TUNING_JOB_EVENT_KEY,
  type CustomEndpoint,
  type ChatPanelState,
} from "../components/chat-panel";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";

dayjs.extend(relativeTime);

const { Text, Paragraph } = Typography;

// Helper to load chat panel state (shared with Chat Panel)
function loadChatPanelState(): Partial<ChatPanelState> {
  try {
    const saved = localStorage.getItem(CHAT_PANEL_STORAGE_KEY);
    if (saved) {
      return JSON.parse(saved);
    }
  } catch {
    // Ignore load errors
  }
  return {};
}

// Helper to save chat panel state (shared with Chat Panel)
function saveChatPanelState(state: Partial<ChatPanelState>) {
  try {
    const current = loadChatPanelState();
    localStorage.setItem(
      CHAT_PANEL_STORAGE_KEY,
      JSON.stringify({ ...current, ...state }),
    );
  } catch {
    // Ignore save errors
  }
}

const REFRESH_INTERVAL = 3000;

// Types
interface TuningJobProgress {
  step: number;
  total_steps: number;
  step_name: string;
  step_description: string;
  configs_tested: number;
  configs_total: number;
  current_config?: Record<string, unknown>;
  best_config_so_far?: Record<string, unknown>;
  best_score_so_far?: number;
}

interface ConversationMessage {
  role: "user" | "assistant" | "tool";
  content: string;
  timestamp?: string;
  tool_calls?: Array<{
    id: string;
    name: string;
    arguments: string;
  }>;
  tool_call_id?: string;
  name?: string; // tool name
}

interface TuningJob {
  id: number;
  model_id: number;
  worker_id: number;
  optimization_target: string;
  status: string;
  status_message?: string;
  current_step: number;
  total_steps: number;
  progress?: TuningJobProgress;
  best_config?: Record<string, unknown>;
  all_results?: Record<string, unknown>[];
  conversation_log?: ConversationMessage[];
  created_at: string;
  updated_at: string;
  completed_at?: string;
  model_name?: string;
  worker_name?: string;
}

interface KnowledgeRecord {
  id: number;
  gpu_model: string;
  gpu_count: number;
  total_vram_gb: number;
  model_name: string;
  model_family: string;
  engine: string;
  quantization?: string;
  tensor_parallel: number;
  throughput_tps: number;
  ttft_ms: number;
  tpot_ms: number;
  score?: number;
  created_at: string;
}

// Helper functions
function getStatusColor(status: string): string {
  const colors: Record<string, string> = {
    pending: "default",
    analyzing: "processing",
    querying_kb: "processing",
    exploring: "processing",
    benchmarking: "processing",
    completed: "success",
    failed: "error",
    cancelled: "warning",
  };
  return colors[status] || "default";
}

function getStatusIcon(status: string) {
  const icons: Record<string, React.ReactNode> = {
    pending: <LoadingOutlined />,
    analyzing: <LoadingOutlined spin />,
    querying_kb: <DatabaseOutlined />,
    exploring: <ExperimentOutlined />,
    benchmarking: <BarChartOutlined />,
    completed: <CheckCircleOutlined />,
    failed: <CloseCircleOutlined />,
    cancelled: <CloseCircleOutlined />,
  };
  return icons[status] || <LoadingOutlined />;
}

function getOptimizationTargetLabel(target: string): string {
  const labels: Record<string, string> = {
    throughput: "Throughput (TPS)",
    latency: "Latency (TTFT/TPOT)",
    cost: "Cost (Min Resources)",
    balanced: "Balanced",
  };
  return labels[target] || target;
}

export default function AutoTuning() {
  const [jobs, setJobs] = useState<TuningJob[]>([]);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [models, setModels] = useState<LLMModel[]>([]);
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [knowledge, setKnowledge] = useState<KnowledgeRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [detailModal, setDetailModal] = useState<TuningJob | null>(null);
  const [form] = Form.useForm();
  const [addEndpointForm] = Form.useForm();
  const { isMobile } = useResponsive();
  const { isDark } = useAppTheme();
  const { canEdit } = useAuth();

  // Custom endpoints from shared localStorage (same as Chat Panel)
  const [customEndpoints, setCustomEndpoints] = useState<CustomEndpoint[]>(
    () => loadChatPanelState().customEndpoints || [],
  );

  // LLM source type for modal
  const [llmSourceType, setLlmSourceType] = useState<"deployment" | "custom">(
    "deployment",
  );
  const [showAddEndpoint, setShowAddEndpoint] = useState(false);

  // Save custom endpoints to shared localStorage (same as Chat Panel)
  useEffect(() => {
    saveChatPanelState({ customEndpoints });
  }, [customEndpoints]);

  // Fetch tuning jobs
  const fetchJobs = useCallback(async () => {
    try {
      const response = await api.get("/auto-tuning/jobs");
      setJobs(response.data.items || []);
    } catch (error) {
      console.error("Failed to fetch tuning jobs:", error);
    }
  }, []);

  // Fetch knowledge base
  const fetchKnowledge = useCallback(async () => {
    try {
      const response = await api.post("/auto-tuning/knowledge/query", {
        limit: 50,
      });
      setKnowledge(response.data.items || []);
    } catch (error) {
      console.error("Failed to fetch knowledge:", error);
    }
  }, []);

  // Fetch workers, models, and deployments
  const fetchResources = useCallback(async () => {
    try {
      const [workersRes, modelsRes, deploymentsRes] = await Promise.all([
        workersApi.list(),
        modelsApi.list(),
        deploymentsApi.list({ status: "running" }),
      ]);
      setWorkers(workersRes.items || []);
      setModels(modelsRes.items || []);
      setDeployments(deploymentsRes.items || []);
    } catch (error) {
      console.error("Failed to fetch resources:", error);
    }
  }, []);

  // Initial load
  useEffect(() => {
    const load = async () => {
      setLoading(true);
      await Promise.all([fetchJobs(), fetchResources(), fetchKnowledge()]);
      setLoading(false);
    };
    load();
  }, [fetchJobs, fetchResources, fetchKnowledge]);

  // Auto refresh for running jobs
  useEffect(() => {
    const hasRunningJobs = jobs.some((j) =>
      [
        "pending",
        "analyzing",
        "querying_kb",
        "exploring",
        "benchmarking",
      ].includes(j.status),
    );

    if (!hasRunningJobs) return;

    const interval = setInterval(fetchJobs, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [jobs, fetchJobs]);

  // Create new tuning job
  const handleCreate = async (values: {
    model_id: number;
    worker_id: number;
    optimization_target: string;
    llm_deployment_id?: number;
    llm_custom_endpoint?: string;
  }) => {
    try {
      // Build LLM config based on selection
      let llm_config: Record<string, unknown> | undefined;

      if (llmSourceType === "deployment" && values.llm_deployment_id) {
        llm_config = { deployment_id: values.llm_deployment_id };
      } else if (llmSourceType === "custom" && values.llm_custom_endpoint) {
        const endpoint = customEndpoints.find(
          (e) => e.id === values.llm_custom_endpoint,
        );
        if (endpoint) {
          llm_config = {
            base_url: endpoint.endpoint,
            api_key: endpoint.apiKey,
            model: endpoint.modelId,
          };
        }
      }

      const response = await api.post("/auto-tuning/jobs", {
        model_id: values.model_id,
        worker_id: values.worker_id,
        optimization_target: values.optimization_target,
        llm_config,
      });
      message.success("Auto-tuning job started");
      setModalOpen(false);
      form.resetFields();
      setLlmSourceType("deployment");
      fetchJobs();

      // Trigger Chat Panel to open with tuning job view
      const jobId = response.data.id;
      if (jobId) {
        localStorage.setItem(
          TUNING_JOB_EVENT_KEY,
          JSON.stringify({
            jobId,
            timestamp: Date.now(),
          }),
        );
        // Dispatch storage event for same-window listeners
        window.dispatchEvent(
          new StorageEvent("storage", {
            key: TUNING_JOB_EVENT_KEY,
            newValue: JSON.stringify({ jobId, timestamp: Date.now() }),
          }),
        );
      }
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to start tuning job");
    }
  };

  // Add custom endpoint
  const handleAddEndpoint = (values: {
    name: string;
    endpoint: string;
    apiKey?: string;
    modelId?: string;
  }) => {
    const newEndpoint: CustomEndpoint = {
      id: `custom-${Date.now()}`,
      name: values.name,
      endpoint: values.endpoint,
      apiKey: values.apiKey,
      modelId: values.modelId,
    };
    setCustomEndpoints((prev) => [...prev, newEndpoint]);
    addEndpointForm.resetFields();
    setShowAddEndpoint(false);
    message.success("Endpoint added");
  };

  // Delete custom endpoint
  const handleDeleteEndpoint = (id: string) => {
    setCustomEndpoints((prev) => prev.filter((e) => e.id !== id));
    message.success("Endpoint removed");
  };

  // Cancel job
  const handleCancel = async (jobId: number) => {
    try {
      await api.post(`/auto-tuning/jobs/${jobId}/cancel`);
      message.success("Tuning job cancelled");
      fetchJobs();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to cancel job");
    }
  };

  // Delete job
  const handleDelete = async (jobId: number) => {
    try {
      await api.delete(`/auto-tuning/jobs/${jobId}`);
      message.success("Tuning job deleted");
      fetchJobs();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to delete job");
    }
  };

  // View job details (fetch with conversation log)
  const [detailLoading, setDetailLoading] = useState(false);
  const handleViewDetail = async (job: TuningJob) => {
    setDetailModal(job); // Show modal immediately with basic info
    setDetailLoading(true);
    try {
      const response = await api.get(`/auto-tuning/jobs/${job.id}`);
      setDetailModal(response.data);
    } catch (error) {
      console.error("Failed to fetch job details:", error);
    } finally {
      setDetailLoading(false);
    }
  };

  // Auto-refresh detail modal for running jobs
  useEffect(() => {
    if (!detailModal) return;
    const isRunning = [
      "pending",
      "analyzing",
      "querying_kb",
      "exploring",
      "benchmarking",
    ].includes(detailModal.status);
    if (!isRunning) return;

    const interval = setInterval(async () => {
      try {
        const response = await api.get(`/auto-tuning/jobs/${detailModal.id}`);
        setDetailModal(response.data);
      } catch (error) {
        console.error("Failed to refresh job:", error);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [detailModal?.id, detailModal?.status]);

  // Stats
  const completedJobs = jobs.filter((j) => j.status === "completed").length;
  const runningJobs = jobs.filter((j) =>
    [
      "pending",
      "analyzing",
      "querying_kb",
      "exploring",
      "benchmarking",
    ].includes(j.status),
  ).length;

  // Table columns for jobs
  const jobColumns = [
    {
      title: "Model",
      dataIndex: "model_name",
      key: "model_name",
      render: (name: string) => <Text strong>{name || "Unknown"}</Text>,
    },
    {
      title: "Worker",
      dataIndex: "worker_name",
      key: "worker_name",
      responsive: ["md" as const],
    },
    {
      title: "Target",
      dataIndex: "optimization_target",
      key: "optimization_target",
      responsive: ["sm" as const],
      render: (target: string) => (
        <Tag color="blue">{getOptimizationTargetLabel(target)}</Tag>
      ),
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      render: (status: string, record: TuningJob) => (
        <Space size={4} wrap>
          <Tag icon={getStatusIcon(status)} color={getStatusColor(status)}>
            {status.toUpperCase()}
          </Tag>
          {record.progress && ["benchmarking"].includes(status) && (
            <Text type="secondary" style={{ fontSize: 12 }}>
              {record.progress.configs_tested}/{record.progress.configs_total}
            </Text>
          )}
        </Space>
      ),
    },
    {
      title: "Progress",
      key: "progress",
      width: 120,
      render: (_: unknown, record: TuningJob) => {
        if (!record.progress) return "-";
        const percent = Math.round(
          (record.progress.step / record.progress.total_steps) * 100,
        );
        return (
          <Tooltip title={record.progress.step_description}>
            <Progress
              percent={percent}
              size="small"
              status={
                record.status === "failed"
                  ? "exception"
                  : record.status === "completed"
                    ? "success"
                    : "active"
              }
              style={{ width: 100, minWidth: 80 }}
            />
          </Tooltip>
        );
      },
    },
    {
      title: "Score",
      key: "best_score",
      responsive: ["lg" as const],
      render: (_: unknown, record: TuningJob) => {
        const score =
          record.progress?.best_score_so_far ??
          (record.best_config?.score as number | undefined);
        return typeof score === "number" ? (
          <Text type="success">{score.toFixed(2)}</Text>
        ) : (
          "-"
        );
      },
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      responsive: ["md" as const],
      render: (date: string) => dayjs(date).fromNow(),
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: unknown, record: TuningJob) => {
        const isRunning = [
          "pending",
          "analyzing",
          "querying_kb",
          "exploring",
          "benchmarking",
        ].includes(record.status);
        return (
          <Space size={4} wrap>
            <Tooltip title="在 Chat Panel 中查看">
              <Button
                size="small"
                icon={<CommentOutlined />}
                onClick={() => {
                  // Trigger Chat Panel to open with tuning job view
                  localStorage.setItem(
                    TUNING_JOB_EVENT_KEY,
                    JSON.stringify({
                      jobId: record.id,
                      timestamp: Date.now(),
                    }),
                  );
                  window.dispatchEvent(
                    new StorageEvent("storage", {
                      key: TUNING_JOB_EVENT_KEY,
                      newValue: JSON.stringify({
                        jobId: record.id,
                        timestamp: Date.now(),
                      }),
                    }),
                  );
                }}
              >
                {isMobile ? "" : "Chat"}
              </Button>
            </Tooltip>
            <Button size="small" onClick={() => handleViewDetail(record)}>
              {isMobile ? "Log" : "Log"}
            </Button>
            {isRunning && canEdit && (
              <Button
                size="small"
                danger
                onClick={() => handleCancel(record.id)}
              >
                {isMobile ? "X" : "Cancel"}
              </Button>
            )}
            {!isRunning && canEdit && (
              <Popconfirm
                title="Delete this job?"
                description="This action cannot be undone."
                onConfirm={() => handleDelete(record.id)}
                okText="Delete"
                cancelText="Cancel"
                okButtonProps={{ danger: true }}
              >
                <Button size="small" danger icon={<DeleteOutlined />} />
              </Popconfirm>
            )}
          </Space>
        );
      },
    },
  ];

  // Table columns for knowledge base
  const knowledgeColumns = [
    {
      title: "Model",
      dataIndex: "model_name",
      key: "model_name",
      render: (name: string, record: KnowledgeRecord) => (
        <div>
          <Text strong>{name}</Text>
          {!isMobile && (
            <>
              <br />
              <Text type="secondary" style={{ fontSize: 12 }}>
                {record.model_family}
              </Text>
            </>
          )}
        </div>
      ),
    },
    {
      title: "GPU",
      key: "gpu",
      responsive: ["md" as const],
      render: (_: unknown, record: KnowledgeRecord) => (
        <div>
          <Text>
            {record.gpu_count}x {record.gpu_model}
          </Text>
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.total_vram_gb.toFixed(1)} GB
          </Text>
        </div>
      ),
    },
    {
      title: "Engine",
      dataIndex: "engine",
      key: "engine",
      render: (engine: string, record: KnowledgeRecord) => (
        <Space direction="vertical" size={0}>
          <Tag color="blue">{engine}</Tag>
          {record.quantization && (
            <Tag color="green" style={{ marginTop: 2 }}>
              {record.quantization}
            </Tag>
          )}
        </Space>
      ),
    },
    {
      title: "TP",
      dataIndex: "tensor_parallel",
      key: "tensor_parallel",
      responsive: ["lg" as const],
    },
    {
      title: "TPS",
      dataIndex: "throughput_tps",
      key: "throughput_tps",
      render: (v: number) => <Text type="success">{v.toFixed(1)}</Text>,
      sorter: (a: KnowledgeRecord, b: KnowledgeRecord) =>
        a.throughput_tps - b.throughput_tps,
    },
    {
      title: "TTFT",
      dataIndex: "ttft_ms",
      key: "ttft_ms",
      responsive: ["sm" as const],
      render: (v: number) => `${v.toFixed(0)} ms`,
      sorter: (a: KnowledgeRecord, b: KnowledgeRecord) => a.ttft_ms - b.ttft_ms,
    },
    {
      title: "TPOT",
      dataIndex: "tpot_ms",
      key: "tpot_ms",
      responsive: ["md" as const],
      render: (v: number) => `${v.toFixed(1)} ms`,
      sorter: (a: KnowledgeRecord, b: KnowledgeRecord) => a.tpot_ms - b.tpot_ms,
    },
    {
      title: "Score",
      dataIndex: "score",
      key: "score",
      responsive: ["lg" as const],
      render: (v: number | undefined) =>
        v ? <Text type="success">{v.toFixed(2)}</Text> : "-",
      sorter: (a: KnowledgeRecord, b: KnowledgeRecord) =>
        (a.score || 0) - (b.score || 0),
    },
  ];

  // Online workers with GPUs
  const availableWorkers = workers.filter(
    (w) => w.status === "online" && w.gpu_info && w.gpu_info.length > 0,
  );

  return (
    <div style={{ padding: isMobile ? 16 : 24 }}>
      {/* Stats Cards */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Completed Jobs"
              value={completedJobs}
              prefix={<CheckCircleOutlined style={{ color: "#52c41a" }} />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Running Jobs"
              value={runningJobs}
              prefix={
                <RocketOutlined
                  style={{ color: runningJobs > 0 ? "#1890ff" : "#d9d9d9" }}
                />
              }
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Knowledge Records"
              value={knowledge.length}
              prefix={<DatabaseOutlined style={{ color: "#722ed1" }} />}
            />
          </Card>
        </Col>
      </Row>

      {/* Main Content */}
      <Card
        title={
          <Space>
            <ThunderboltOutlined />
            <span>Auto-Tuning Agent</span>
          </Space>
        }
        extra={
          <Space>
            <Button
              icon={<ReloadOutlined />}
              onClick={() => {
                fetchJobs();
                fetchKnowledge();
              }}
            >
              Refresh
            </Button>
            {canEdit && (
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setModalOpen(true)}
              >
                New Tuning Job
              </Button>
            )}
          </Space>
        }
      >
        <Paragraph type="secondary" style={{ marginBottom: 16 }}>
          Auto-Tuning Agent automatically finds the best deployment
          configuration. Use the <strong>Chat Panel</strong> on the right to
          interact with the agent, or start a job directly below.
        </Paragraph>

        <Tabs
          defaultActiveKey="jobs"
          items={[
            {
              key: "jobs",
              label: (
                <span>
                  <HistoryOutlined /> Job History
                </span>
              ),
              children: (
                <Table
                  dataSource={jobs}
                  columns={jobColumns}
                  rowKey="id"
                  loading={loading}
                  pagination={{ pageSize: 10 }}
                  scroll={{ x: "max-content" }}
                  style={{ overflowX: "auto" }}
                  locale={{
                    emptyText: (
                      <Empty
                        image={Empty.PRESENTED_IMAGE_SIMPLE}
                        description="No tuning jobs yet"
                      >
                        {canEdit && (
                          <Button
                            type="primary"
                            icon={<RocketOutlined />}
                            onClick={() => setModalOpen(true)}
                          >
                            Start Auto-Tuning
                          </Button>
                        )}
                      </Empty>
                    ),
                  }}
                />
              ),
            },
            {
              key: "knowledge",
              label: (
                <span>
                  <DatabaseOutlined /> Knowledge Base
                </span>
              ),
              children: (
                <div>
                  <Paragraph type="secondary" style={{ marginBottom: 16 }}>
                    Historical benchmark results used for configuration
                    recommendations. The agent uses this data to suggest optimal
                    configs for similar setups.
                  </Paragraph>
                  <Table
                    dataSource={knowledge}
                    columns={knowledgeColumns}
                    rowKey="id"
                    loading={loading}
                    pagination={{ pageSize: 10 }}
                    scroll={{ x: "max-content" }}
                    style={{ overflowX: "auto" }}
                    locale={{
                      emptyText: (
                        <Empty
                          image={Empty.PRESENTED_IMAGE_SIMPLE}
                          description="No knowledge records yet. Run benchmarks to populate the knowledge base."
                        />
                      ),
                    }}
                  />
                </div>
              ),
            },
          ]}
        />
      </Card>

      {/* Create Modal */}
      <Modal
        title={
          <Space>
            <ThunderboltOutlined />
            <span>Start Auto-Tuning</span>
          </Space>
        }
        open={modalOpen}
        onCancel={() => {
          setModalOpen(false);
          form.resetFields();
          setLlmSourceType("deployment");
          setShowAddEndpoint(false);
        }}
        footer={null}
        width={600}
      >
        <Form form={form} layout="vertical" onFinish={handleCreate}>
          {/* Model to tune */}
          <Form.Item
            name="model_id"
            label="Model to Tune"
            rules={[{ required: true, message: "Please select a model" }]}
          >
            <Select
              placeholder="Select model to tune"
              showSearch
              optionFilterProp="children"
            >
              {models.map((model) => (
                <Select.Option key={model.id} value={model.id}>
                  {model.name}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>

          {/* Worker */}
          <Form.Item
            name="worker_id"
            label="Worker"
            rules={[{ required: true, message: "Please select a worker" }]}
          >
            <Select placeholder="Select worker">
              {availableWorkers.map((worker) => (
                <Select.Option key={worker.id} value={worker.id}>
                  {worker.name} ({worker.gpu_info?.length || 0} GPUs)
                </Select.Option>
              ))}
            </Select>
          </Form.Item>

          {/* Optimization Target */}
          <Form.Item
            name="optimization_target"
            label="Optimization Target"
            initialValue="balanced"
          >
            <Radio.Group>
              <Radio.Button value="throughput">Throughput</Radio.Button>
              <Radio.Button value="latency">Latency</Radio.Button>
              <Radio.Button value="balanced">Balanced</Radio.Button>
              <Radio.Button value="cost">Cost</Radio.Button>
            </Radio.Group>
          </Form.Item>

          <Divider>
            <Space>
              <ApiOutlined />
              <span>Agent LLM</span>
            </Space>
          </Divider>

          {/* Agent LLM Selection */}
          <Alert
            message="Select which LLM the agent will use for reasoning and decision-making"
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
          />

          <Form.Item label="LLM Source">
            <Radio.Group
              value={llmSourceType}
              onChange={(e) => setLlmSourceType(e.target.value)}
            >
              <Radio.Button value="deployment">Local Deployment</Radio.Button>
              <Radio.Button value="custom">Custom Endpoint</Radio.Button>
            </Radio.Group>
          </Form.Item>

          {llmSourceType === "deployment" && (
            <Form.Item
              name="llm_deployment_id"
              label="Select Deployment"
              rules={[
                { required: true, message: "Please select a deployment" },
              ]}
            >
              <Select placeholder="Select a running deployment">
                {deployments.length === 0 ? (
                  <Select.Option value="" disabled>
                    No running deployments available
                  </Select.Option>
                ) : (
                  deployments.map((d) => (
                    <Select.Option key={d.id} value={d.id}>
                      {d.name} ({d.model?.name || "Unknown model"})
                    </Select.Option>
                  ))
                )}
              </Select>
            </Form.Item>
          )}

          {llmSourceType === "custom" && (
            <>
              <Form.Item
                name="llm_custom_endpoint"
                label="Select Endpoint"
                rules={[
                  {
                    required: customEndpoints.length > 0,
                    message: "Please select an endpoint",
                  },
                ]}
              >
                <Select
                  placeholder={
                    customEndpoints.length === 0
                      ? "No endpoints - add one below"
                      : "Select an endpoint"
                  }
                  disabled={customEndpoints.length === 0}
                  dropdownRender={(menu) => (
                    <>
                      {menu}
                      <Divider style={{ margin: "8px 0" }} />
                      <Button
                        type="link"
                        icon={<PlusOutlined />}
                        onClick={() => setShowAddEndpoint(true)}
                        style={{ width: "100%", textAlign: "left" }}
                      >
                        Add New Endpoint
                      </Button>
                    </>
                  )}
                >
                  {customEndpoints.map((ep) => (
                    <Select.Option key={ep.id} value={ep.id}>
                      <Space
                        style={{
                          width: "100%",
                          justifyContent: "space-between",
                        }}
                      >
                        <span>{ep.name}</span>
                        <Button
                          type="text"
                          size="small"
                          danger
                          icon={<DeleteOutlined />}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteEndpoint(ep.id);
                          }}
                        />
                      </Space>
                    </Select.Option>
                  ))}
                </Select>
              </Form.Item>

              {customEndpoints.length === 0 && !showAddEndpoint && (
                <Button
                  type="dashed"
                  icon={<PlusOutlined />}
                  onClick={() => setShowAddEndpoint(true)}
                  block
                  style={{ marginBottom: 16 }}
                >
                  Add Custom Endpoint
                </Button>
              )}

              {showAddEndpoint && (
                <Card
                  size="small"
                  title="Add New Endpoint"
                  extra={
                    <Button
                      type="text"
                      size="small"
                      onClick={() => setShowAddEndpoint(false)}
                    >
                      Cancel
                    </Button>
                  }
                  style={{ marginBottom: 16 }}
                >
                  <Form
                    form={addEndpointForm}
                    layout="vertical"
                    size="small"
                    onFinish={handleAddEndpoint}
                  >
                    <Form.Item
                      name="name"
                      label="Name"
                      rules={[{ required: true, message: "Required" }]}
                    >
                      <Input placeholder="e.g., OpenAI GPT-4" />
                    </Form.Item>
                    <Form.Item
                      name="endpoint"
                      label="Endpoint URL"
                      rules={[{ required: true, message: "Required" }]}
                    >
                      <Input placeholder="https://api.openai.com/v1" />
                    </Form.Item>
                    <Form.Item name="apiKey" label="API Key">
                      <Input.Password placeholder="sk-..." />
                    </Form.Item>
                    <Form.Item name="modelId" label="Model ID">
                      <Input placeholder="gpt-4o" />
                    </Form.Item>
                    <Button type="primary" htmlType="submit" size="small">
                      Add Endpoint
                    </Button>
                  </Form>
                </Card>
              )}
            </>
          )}

          <Form.Item style={{ marginTop: 24 }}>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                icon={<RocketOutlined />}
              >
                Start Tuning
              </Button>
              <Button onClick={() => setModalOpen(false)}>Cancel</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Detail Modal - Docker-style Log View */}
      <Modal
        title={
          <Space>
            <ExperimentOutlined />
            <span>Tuning Log - {detailModal?.model_name}</span>
            <Tag
              icon={detailModal ? getStatusIcon(detailModal.status) : null}
              color={
                detailModal ? getStatusColor(detailModal.status) : "default"
              }
            >
              {detailModal?.status.toUpperCase()}
            </Tag>
            {detailLoading && <LoadingOutlined spin />}
          </Space>
        }
        open={!!detailModal}
        onCancel={() => setDetailModal(null)}
        footer={null}
        width={900}
        styles={{ body: { padding: 0 } }}
      >
        {detailModal && (
          <div>
            {/* Docker-style Log Container */}
            <div
              style={{
                background: "#1e1e1e",
                color: "#d4d4d4",
                fontFamily: "'Consolas', 'Monaco', 'Courier New', monospace",
                fontSize: 13,
                lineHeight: 1.5,
                padding: 16,
                maxHeight: "60vh",
                overflowY: "auto",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
              }}
            >
              {detailModal.conversation_log &&
              detailModal.conversation_log.length > 0 ? (
                detailModal.conversation_log.map((msg, idx) => {
                  const timestamp = msg.timestamp
                    ? dayjs(msg.timestamp).format("HH:mm:ss")
                    : "";

                  if (msg.role === "user") {
                    return (
                      <div key={idx} style={{ marginBottom: 12 }}>
                        <span style={{ color: "#569cd6" }}>[{timestamp}]</span>
                        <span style={{ color: "#4ec9b0" }}> [USER] </span>
                        <span style={{ color: "#ce9178" }}>{msg.content}</span>
                      </div>
                    );
                  }

                  if (msg.role === "assistant") {
                    return (
                      <div key={idx} style={{ marginBottom: 12 }}>
                        <span style={{ color: "#569cd6" }}>[{timestamp}]</span>
                        <span style={{ color: "#dcdcaa" }}> [AGENT] </span>
                        {msg.content && (
                          <span style={{ color: "#d4d4d4" }}>
                            {msg.content}
                          </span>
                        )}
                        {msg.tool_calls && msg.tool_calls.length > 0 && (
                          <div style={{ marginLeft: 20, marginTop: 4 }}>
                            {msg.tool_calls.map((tc, tcIdx) => (
                              <div key={tcIdx} style={{ color: "#9cdcfe" }}>
                                -&gt; Calling: {tc.name}(
                                {(() => {
                                  try {
                                    const args = JSON.parse(tc.arguments);
                                    return Object.entries(args)
                                      .map(
                                        ([k, v]) => `${k}=${JSON.stringify(v)}`,
                                      )
                                      .join(", ");
                                  } catch {
                                    return tc.arguments;
                                  }
                                })()}
                                )
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  }

                  if (msg.role === "tool") {
                    let content = msg.content;
                    try {
                      const parsed = JSON.parse(msg.content);
                      content = JSON.stringify(parsed, null, 2);
                    } catch {
                      // Keep original
                    }
                    return (
                      <div key={idx} style={{ marginBottom: 12 }}>
                        <span style={{ color: "#569cd6" }}>[{timestamp}]</span>
                        <span style={{ color: "#6a9955" }}>
                          {" "}
                          [TOOL:{msg.name}]{" "}
                        </span>
                        <div
                          style={{
                            marginLeft: 20,
                            marginTop: 4,
                            padding: 8,
                            background: "rgba(255,255,255,0.05)",
                            borderRadius: 4,
                            maxHeight: 200,
                            overflow: "auto",
                            color: "#b5cea8",
                          }}
                        >
                          {content}
                        </div>
                      </div>
                    );
                  }

                  return null;
                })
              ) : (
                <div style={{ color: "#808080" }}>
                  {detailLoading
                    ? "Loading logs..."
                    : detailModal.status === "pending"
                      ? "Waiting for agent to start..."
                      : "No logs available"}
                </div>
              )}

              {/* Running indicator */}
              {[
                "pending",
                "analyzing",
                "querying_kb",
                "exploring",
                "benchmarking",
              ].includes(detailModal.status) && (
                <div style={{ marginTop: 8 }}>
                  <span style={{ color: "#569cd6" }}>
                    [{dayjs().format("HH:mm:ss")}]
                  </span>
                  <span style={{ color: "#c586c0" }}> [STATUS] </span>
                  <span style={{ color: "#808080" }}>
                    {detailModal.status_message || "Processing..."}
                    <span
                      className="blink"
                      style={{ animation: "blink 1s infinite" }}
                    >
                      {" "}
                      _
                    </span>
                  </span>
                </div>
              )}
            </div>

            {/* Best Config Section */}
            {detailModal.best_config && (
              <div
                style={{
                  padding: 16,
                  borderTop: `1px solid ${isDark ? "#303030" : "#e8e8e8"}`,
                }}
              >
                <Text strong>Best Configuration:</Text>
                <pre
                  style={{
                    background: isDark ? "#1f1f1f" : "#f5f5f5",
                    padding: 12,
                    borderRadius: 6,
                    overflow: "auto",
                    margin: "8px 0 0 0",
                    fontSize: 12,
                  }}
                >
                  {JSON.stringify(detailModal.best_config, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}
