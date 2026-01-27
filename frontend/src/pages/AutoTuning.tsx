/**
 * Auto-Tuning Page
 *
 * Bayesian optimization-based hyperparameter tuning for LLM deployments.
 * Uses Optuna TPE (Tree-structured Parzen Estimator) for efficient search.
 */
import React, { useEffect, useState, useCallback } from "react";
import {
  Button,
  Card,
  Form,
  Select,
  Space,
  Table,
  Tag,
  message,
  Progress,
  Typography,
  Empty,
  Tooltip,
  Tabs,
  Statistic,
  Row,
  Col,
  Alert,
  Popconfirm,
  Modal,
  Descriptions,
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
  DeleteOutlined,
  PlayCircleOutlined,
  ClockCircleOutlined,
  AimOutlined,
  SettingOutlined,
  LineChartOutlined,
} from "@ant-design/icons";
import { workersApi, modelsApi } from "../services/api";
import { api } from "../api/client";
import type { Worker, LLMModel } from "../types";
import { useResponsive } from "../hooks";
import { useAuth } from "../contexts/AuthContext";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import duration from "dayjs/plugin/duration";

dayjs.extend(relativeTime);
dayjs.extend(duration);

const { Text } = Typography;

const REFRESH_INTERVAL = 3000;

// ============================================================================
// Types
// ============================================================================

interface TuningJobProgress {
  step: number;
  total_steps: number;
  step_name: string;
  step_description?: string;
  current_config?: Record<string, unknown>;
  completed_trials?: number;
  successful_trials?: number;
}

interface TrialResult {
  parameters?: Record<string, unknown>;
  metrics?: Record<string, number>;
  success?: boolean;
  error?: string;
}

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
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
  all_results?: TrialResult[];
  logs?: LogEntry[];
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

// ============================================================================
// Constants
// ============================================================================

const STATUS_CONFIG: Record<
  string,
  { color: string; icon: React.ReactNode; label: string }
> = {
  pending: {
    color: "default",
    icon: <ClockCircleOutlined />,
    label: "Pending",
  },
  analyzing: {
    color: "processing",
    icon: <LoadingOutlined spin />,
    label: "Analyzing",
  },
  querying_kb: {
    color: "processing",
    icon: <DatabaseOutlined />,
    label: "Querying KB",
  },
  exploring: {
    color: "processing",
    icon: <ExperimentOutlined />,
    label: "Exploring",
  },
  benchmarking: {
    color: "processing",
    icon: <LineChartOutlined />,
    label: "Benchmarking",
  },
  completed: {
    color: "success",
    icon: <CheckCircleOutlined />,
    label: "Completed",
  },
  failed: { color: "error", icon: <CloseCircleOutlined />, label: "Failed" },
  cancelled: {
    color: "warning",
    icon: <CloseCircleOutlined />,
    label: "Cancelled",
  },
};

const OPTIMIZATION_TARGETS = [
  {
    value: "throughput",
    label: "Throughput",
    description: "Maximize tokens per second (TPS)",
    icon: <ThunderboltOutlined />,
  },
  {
    value: "latency",
    label: "Latency",
    description: "Minimize time-to-first-token and response time",
    icon: <RocketOutlined />,
  },
  {
    value: "balanced",
    label: "Balanced",
    description: "Optimize for both throughput and latency",
    icon: <AimOutlined />,
  },
];

// ============================================================================
// Helper Components
// ============================================================================

function StatusTag({ status }: { status: string }) {
  const config = STATUS_CONFIG[status] || STATUS_CONFIG.pending;
  return (
    <Tag icon={config.icon} color={config.color}>
      {config.label}
    </Tag>
  );
}

function TargetTag({ target }: { target: string }) {
  const config = OPTIMIZATION_TARGETS.find((t) => t.value === target);
  const colors: Record<string, string> = {
    throughput: "green",
    latency: "blue",
    balanced: "purple",
  };
  return (
    <Tag color={colors[target] || "default"}>{config?.label || target}</Tag>
  );
}

function LogViewer({
  logs,
  maxHeight = 300,
}: {
  logs: LogEntry[];
  maxHeight?: number;
}) {
  const logContainerRef = React.useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new logs arrive
  React.useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const getLevelColor = (level: string) => {
    switch (level.toUpperCase()) {
      case "ERROR":
        return "#ff4d4f";
      case "WARNING":
        return "#faad14";
      case "INFO":
      default:
        return "#8c8c8c";
    }
  };

  return (
    <div
      ref={logContainerRef}
      style={{
        background: "#1e1e1e",
        borderRadius: 4,
        padding: 12,
        maxHeight,
        overflow: "auto",
        fontFamily: "'Fira Code', 'Consolas', monospace",
        fontSize: 12,
        lineHeight: 1.6,
      }}
    >
      {logs.length === 0 ? (
        <Text type="secondary" style={{ color: "#666" }}>
          Waiting for logs...
        </Text>
      ) : (
        logs.map((log, idx) => {
          const time = dayjs(log.timestamp).format("HH:mm:ss");
          return (
            <div key={idx} style={{ color: "#d4d4d4" }}>
              <span style={{ color: "#666" }}>[{time}]</span>{" "}
              <span style={{ color: getLevelColor(log.level) }}>
                {log.level.padEnd(7)}
              </span>{" "}
              <span>{log.message}</span>
            </div>
          );
        })
      )}
    </div>
  );
}

function ProgressDisplay({ job }: { job: TuningJob }) {
  const { progress, status } = job;

  if (!progress) {
    return <Text type="secondary">-</Text>;
  }

  const completed = progress.completed_trials ?? progress.step ?? 0;
  const total = progress.total_steps || 10;
  const percent = Math.round((completed / total) * 100);
  const successful = progress.successful_trials ?? 0;

  const isRunning = [
    "analyzing",
    "querying_kb",
    "exploring",
    "benchmarking",
  ].includes(status);

  return (
    <Tooltip
      title={
        <div>
          <div>
            Trial {completed} / {total}
          </div>
          <div>Successful: {successful}</div>
          {progress.step_name && <div>Current: {progress.step_name}</div>}
        </div>
      }
    >
      <Progress
        percent={percent}
        size="small"
        status={
          status === "failed"
            ? "exception"
            : status === "completed"
              ? "success"
              : isRunning
                ? "active"
                : "normal"
        }
        format={() => `${completed}/${total}`}
        style={{ width: 100, minWidth: 80 }}
      />
    </Tooltip>
  );
}

function JobDetailCard({
  job,
  onClose: _onClose,
}: {
  job: TuningJob;
  onClose: () => void;
}) {
  const bestMetrics = job.best_config?.metrics as
    | Record<string, number>
    | undefined;
  const trials = job.all_results || [];
  const successfulTrials = trials.filter((t) => t.success);
  const logs = job.logs || [];

  return (
    <div>
      {/* Best Configuration */}
      {job.best_config && (
        <Card
          size="small"
          title={
            <Space>
              <CheckCircleOutlined style={{ color: "#52c41a" }} />
              <span>Best Configuration</span>
            </Space>
          }
          style={{ marginBottom: 16 }}
        >
          <Row gutter={[16, 16]}>
            <Col span={6}>
              <Statistic
                title="Throughput"
                value={bestMetrics?.throughput_tps ?? 0}
                suffix="TPS"
                precision={1}
                valueStyle={{ color: "#52c41a" }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="TTFT"
                value={bestMetrics?.avg_ttft_ms ?? 0}
                suffix="ms"
                precision={0}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Engine"
                value={(job.best_config.engine as string) || "vllm"}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="GPU Memory"
                value={(
                  (job.best_config.gpu_memory_utilization as number) * 100 || 90
                ).toFixed(0)}
                suffix="%"
              />
            </Col>
          </Row>

          <Descriptions size="small" column={2} style={{ marginTop: 16 }}>
            <Descriptions.Item label="Max Sequences">
              {(job.best_config.max_num_seqs as number) || "-"}
            </Descriptions.Item>
            <Descriptions.Item label="Tensor Parallel">
              {(job.best_config.tensor_parallel_size as number) || 1}
            </Descriptions.Item>
            <Descriptions.Item label="Objective Value">
              {(job.best_config.objective_value as number)?.toFixed(2) || "-"}
            </Descriptions.Item>
          </Descriptions>
        </Card>
      )}

      {/* Logs */}
      {logs.length > 0 && (
        <Card
          size="small"
          title={
            <Space>
              <BarChartOutlined />
              <span>Execution Logs</span>
              <Tag>{logs.length} entries</Tag>
            </Space>
          }
          style={{ marginBottom: 16 }}
        >
          <LogViewer logs={logs} maxHeight={250} />
        </Card>
      )}

      {/* Trial Results Table */}
      {trials.length > 0 && (
        <Card size="small" title="Trial History">
          <Table
            dataSource={trials.map((t, idx) => ({
              ...t,
              key: idx,
              trial_num: idx + 1,
            }))}
            columns={[
              {
                title: "#",
                dataIndex: "trial_num",
                key: "trial_num",
                width: 50,
              },
              {
                title: "Engine",
                key: "engine",
                render: (_, record) => record.parameters?.engine || "-",
              },
              {
                title: "GPU Mem",
                key: "gpu_mem",
                render: (_, record) => {
                  const val = record.parameters
                    ?.gpu_memory_utilization as number;
                  return val ? `${(val * 100).toFixed(0)}%` : "-";
                },
              },
              {
                title: "TPS",
                key: "tps",
                render: (_, record) => {
                  const val = record.metrics?.throughput_tps;
                  return val ? (
                    <Text type="success">{val.toFixed(1)}</Text>
                  ) : (
                    "-"
                  );
                },
              },
              {
                title: "TTFT",
                key: "ttft",
                render: (_, record) => {
                  const val = record.metrics?.avg_ttft_ms;
                  return val ? `${val.toFixed(0)} ms` : "-";
                },
              },
              {
                title: "Status",
                key: "success",
                width: 80,
                render: (_, record) =>
                  record.success ? (
                    <CheckCircleOutlined style={{ color: "#52c41a" }} />
                  ) : (
                    <Tooltip title={record.error}>
                      <CloseCircleOutlined style={{ color: "#ff4d4f" }} />
                    </Tooltip>
                  ),
              },
            ]}
            size="small"
            pagination={false}
            scroll={{ y: 200 }}
          />
        </Card>
      )}

      {/* Summary */}
      <div style={{ marginTop: 16, textAlign: "center" }}>
        <Space split={<span style={{ color: "#d9d9d9" }}>|</span>}>
          <Text type="secondary">Total Trials: {trials.length}</Text>
          <Text type="secondary">Successful: {successfulTrials.length}</Text>
          {job.completed_at && (
            <Text type="secondary">
              Duration:{" "}
              {dayjs(job.completed_at).diff(dayjs(job.created_at), "minute")}{" "}
              min
            </Text>
          )}
        </Space>
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export default function AutoTuning() {
  const [jobs, setJobs] = useState<TuningJob[]>([]);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [models, setModels] = useState<LLMModel[]>([]);
  const [knowledge, setKnowledge] = useState<KnowledgeRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [detailModal, setDetailModal] = useState<TuningJob | null>(null);
  const [logModal, setLogModal] = useState<TuningJob | null>(null);
  const [form] = Form.useForm();
  const { isMobile } = useResponsive();
  const { canEdit } = useAuth();

  // --------------------------------------------------------------------------
  // Data Fetching
  // --------------------------------------------------------------------------

  const fetchJobs = useCallback(async () => {
    try {
      const response = await api.get("/auto-tuning/jobs");
      setJobs(response.data.items || []);
    } catch (error) {
      console.error("Failed to fetch tuning jobs:", error);
    }
  }, []);

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

  const fetchResources = useCallback(async () => {
    try {
      const [workersRes, modelsRes] = await Promise.all([
        workersApi.list(),
        modelsApi.list(),
      ]);
      setWorkers(workersRes.items || []);
      setModels(modelsRes.items || []);
    } catch (error) {
      console.error("Failed to fetch resources:", error);
    }
  }, []);

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
    if (!hasRunningJobs && !logModal) return;
    const interval = setInterval(fetchJobs, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [jobs, fetchJobs, logModal]);

  // Update log modal when jobs refresh
  useEffect(() => {
    if (logModal) {
      const updatedJob = jobs.find((j) => j.id === logModal.id);
      if (updatedJob) {
        setLogModal(updatedJob);
      }
    }
  }, [jobs, logModal?.id]);

  // --------------------------------------------------------------------------
  // Actions
  // --------------------------------------------------------------------------

  const handleCreate = async (values: {
    model_id: number;
    worker_id: number;
    optimization_target: string;
  }) => {
    try {
      const response = await api.post("/auto-tuning/jobs", {
        model_id: values.model_id,
        worker_id: values.worker_id,
        optimization_target: values.optimization_target,
      });

      message.success(`Tuning job #${response.data.id} created successfully`);
      setCreateModalOpen(false);
      form.resetFields();
      fetchJobs();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(
        err.response?.data?.detail || "Failed to create tuning job",
      );
    }
  };

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

  const handleDeployBestConfig = async (job: TuningJob) => {
    if (!job.best_config) {
      message.warning("No best configuration available");
      return;
    }

    try {
      const engine = (job.best_config.engine as string) || "vllm";
      const gpuMemUtil = job.best_config.gpu_memory_utilization as number;
      const maxNumSeqs = job.best_config.max_num_seqs as number;
      const tpSize = (job.best_config.tensor_parallel_size as number) || 1;

      const extraParams: Record<string, unknown> = {};
      if (gpuMemUtil) extraParams["gpu-memory-utilization"] = gpuMemUtil;
      if (maxNumSeqs) extraParams["max-num-seqs"] = maxNumSeqs;

      await api.post("/deployments", {
        model_id: job.model_id,
        worker_id: job.worker_id,
        name: `tuned-${job.model_name?.split("/").pop() || "model"}-${Date.now()}`,
        backend: engine,
        gpu_indexes: Array.from({ length: tpSize }, (_, i) => i),
        extra_params: extraParams,
      });

      message.success("Deployment created with optimized configuration");
      setDetailModal(null);
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(
        err.response?.data?.detail || "Failed to create deployment",
      );
    }
  };

  // --------------------------------------------------------------------------
  // Computed Values
  // --------------------------------------------------------------------------

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
  const availableWorkers = workers.filter(
    (w) => w.status === "online" && w.gpu_info && w.gpu_info.length > 0,
  );

  // --------------------------------------------------------------------------
  // Table Columns
  // --------------------------------------------------------------------------

  const jobColumns = [
    {
      title: "Model",
      dataIndex: "model_name",
      key: "model_name",
      render: (name: string, record: TuningJob) => (
        <Space direction="vertical" size={0}>
          <Text strong>{name || "Unknown"}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            Job #{record.id}
          </Text>
        </Space>
      ),
    },
    {
      title: "Target",
      dataIndex: "optimization_target",
      key: "optimization_target",
      width: 120,
      render: (target: string) => <TargetTag target={target} />,
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      width: 140,
      render: (status: string, record: TuningJob) => (
        <Space direction="vertical" size={0}>
          <StatusTag status={status} />
          {record.status_message && (
            <Text
              type="secondary"
              style={{ fontSize: 11 }}
              ellipsis={{ tooltip: record.status_message }}
            >
              {record.status_message.slice(0, 30)}
            </Text>
          )}
        </Space>
      ),
    },
    {
      title: "Progress",
      key: "progress",
      width: 130,
      render: (_: unknown, record: TuningJob) => (
        <ProgressDisplay job={record} />
      ),
    },
    {
      title: "Best TPS",
      key: "best_tps",
      width: 100,
      render: (_: unknown, record: TuningJob) => {
        const metrics = record.best_config?.metrics as
          | Record<string, number>
          | undefined;
        const tps = metrics?.throughput_tps;
        return tps ? (
          <Text type="success" strong>
            {tps.toFixed(1)}
          </Text>
        ) : (
          <Text type="secondary">-</Text>
        );
      },
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      width: 120,
      responsive: ["md" as const],
      render: (date: string) => (
        <Tooltip title={dayjs(date).format("YYYY-MM-DD HH:mm:ss")}>
          <Text type="secondary">{dayjs(date).fromNow()}</Text>
        </Tooltip>
      ),
    },
    {
      title: "Actions",
      key: "actions",
      width: 220,
      render: (_: unknown, record: TuningJob) => {
        const isRunning = [
          "pending",
          "analyzing",
          "querying_kb",
          "exploring",
          "benchmarking",
        ].includes(record.status);
        return (
          <Space size={4}>
            {isRunning && (
              <Button size="small" onClick={() => setLogModal(record)}>
                Logs
              </Button>
            )}
            {record.status === "completed" && (
              <>
                <Button
                  size="small"
                  type="primary"
                  icon={<RocketOutlined />}
                  onClick={() => handleDeployBestConfig(record)}
                >
                  Deploy
                </Button>
                <Button size="small" onClick={() => setDetailModal(record)}>
                  Details
                </Button>
              </>
            )}
            {record.status === "failed" && (
              <Button size="small" onClick={() => setDetailModal(record)}>
                Details
              </Button>
            )}
            {isRunning && canEdit && (
              <Button
                size="small"
                danger
                onClick={() => handleCancel(record.id)}
              >
                Cancel
              </Button>
            )}
            {!isRunning && canEdit && (
              <Popconfirm
                title="Delete this tuning job?"
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

  const knowledgeColumns = [
    {
      title: "Model",
      dataIndex: "model_name",
      key: "model_name",
      render: (name: string) => <Text strong>{name}</Text>,
    },
    {
      title: "GPU",
      key: "gpu",
      responsive: ["md" as const],
      render: (_: unknown, record: KnowledgeRecord) => (
        <Text>
          {record.gpu_count}x {record.gpu_model}
        </Text>
      ),
    },
    {
      title: "Engine",
      dataIndex: "engine",
      key: "engine",
      width: 100,
      render: (engine: string) => <Tag color="blue">{engine}</Tag>,
    },
    {
      title: "TPS",
      dataIndex: "throughput_tps",
      key: "throughput_tps",
      width: 80,
      render: (v: number) => <Text type="success">{v.toFixed(1)}</Text>,
      sorter: (a: KnowledgeRecord, b: KnowledgeRecord) =>
        a.throughput_tps - b.throughput_tps,
    },
    {
      title: "TTFT",
      dataIndex: "ttft_ms",
      key: "ttft_ms",
      width: 80,
      responsive: ["sm" as const],
      render: (v: number) => `${v.toFixed(0)} ms`,
    },
    {
      title: "TPOT",
      dataIndex: "tpot_ms",
      key: "tpot_ms",
      width: 80,
      responsive: ["md" as const],
      render: (v: number) => `${v.toFixed(1)} ms`,
    },
  ];

  // --------------------------------------------------------------------------
  // Render
  // --------------------------------------------------------------------------

  return (
    <div style={{ padding: isMobile ? 16 : 24 }}>
      {/* Statistics Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
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
                runningJobs > 0 ? (
                  <LoadingOutlined spin style={{ color: "#1890ff" }} />
                ) : (
                  <ExperimentOutlined style={{ color: "#d9d9d9" }} />
                )
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
            <span>Auto-Tuning</span>
            <Tag color="blue">Bayesian Optimization</Tag>
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
                onClick={() => setCreateModalOpen(true)}
              >
                New Job
              </Button>
            )}
          </Space>
        }
      >
        <Alert
          message="Bayesian Optimization for LLM Deployment"
          description="Automatically searches for optimal deployment parameters using TPE (Tree-structured Parzen Estimator). The system will test different configurations and find the best settings for your target metric."
          type="info"
          showIcon
          icon={<ExperimentOutlined />}
          style={{ marginBottom: 16 }}
        />

        <Tabs
          defaultActiveKey="jobs"
          items={[
            {
              key: "jobs",
              label: (
                <span>
                  <HistoryOutlined /> Tuning Jobs
                </span>
              ),
              children: (
                <Table
                  dataSource={jobs}
                  columns={jobColumns}
                  rowKey="id"
                  loading={loading}
                  pagination={{ pageSize: 10, showSizeChanger: false }}
                  scroll={{ x: "max-content" }}
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
                            onClick={() => setCreateModalOpen(true)}
                          >
                            Create First Job
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
                <Table
                  dataSource={knowledge}
                  columns={knowledgeColumns}
                  rowKey="id"
                  loading={loading}
                  pagination={{ pageSize: 10, showSizeChanger: false }}
                  scroll={{ x: "max-content" }}
                  locale={{
                    emptyText: (
                      <Empty
                        image={Empty.PRESENTED_IMAGE_SIMPLE}
                        description="No performance records. Complete tuning jobs to populate the knowledge base."
                      />
                    ),
                  }}
                />
              ),
            },
          ]}
        />
      </Card>

      {/* Create Job Modal */}
      <Modal
        title={
          <Space>
            <ThunderboltOutlined />
            <span>New Auto-Tuning Job</span>
          </Space>
        }
        open={createModalOpen}
        onCancel={() => {
          setCreateModalOpen(false);
          form.resetFields();
        }}
        footer={null}
        width={520}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreate}
          initialValues={{ optimization_target: "throughput" }}
        >
          <Form.Item
            name="model_id"
            label="Model"
            rules={[{ required: true, message: "Please select a model" }]}
          >
            <Select
              placeholder="Select a model to tune"
              showSearch
              optionFilterProp="children"
              size="large"
            >
              {models.map((model) => (
                <Select.Option key={model.id} value={model.id}>
                  {model.name}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="worker_id"
            label="Worker"
            rules={[{ required: true, message: "Please select a worker" }]}
          >
            <Select placeholder="Select a worker with GPU" size="large">
              {availableWorkers.length === 0 ? (
                <Select.Option disabled value="">
                  No workers with GPU available
                </Select.Option>
              ) : (
                availableWorkers.map((worker) => (
                  <Select.Option key={worker.id} value={worker.id}>
                    <Space>
                      <span>{worker.name}</span>
                      <Tag color="green">
                        {worker.gpu_info?.length || 0} GPU
                      </Tag>
                    </Space>
                  </Select.Option>
                ))
              )}
            </Select>
          </Form.Item>

          <Form.Item name="optimization_target" label="Optimization Target">
            <Select size="large">
              {OPTIMIZATION_TARGETS.map((target) => (
                <Select.Option key={target.value} value={target.value}>
                  <Space>
                    {target.icon}
                    <span>{target.label}</span>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      - {target.description}
                    </Text>
                  </Space>
                </Select.Option>
              ))}
            </Select>
          </Form.Item>

          <Alert
            type="info"
            showIcon
            icon={<SettingOutlined />}
            message="Parameters searched automatically"
            description={
              <ul style={{ margin: "8px 0 0 0", paddingLeft: 16 }}>
                <li>Inference engine (vLLM, SGLang)</li>
                <li>GPU memory utilization</li>
                <li>Maximum concurrent sequences</li>
                <li>Tensor parallelism (if multiple GPUs)</li>
              </ul>
            }
            style={{ marginBottom: 24 }}
          />

          <Form.Item style={{ marginBottom: 0 }}>
            <Space style={{ width: "100%", justifyContent: "flex-end" }}>
              <Button onClick={() => setCreateModalOpen(false)}>Cancel</Button>
              <Button
                type="primary"
                htmlType="submit"
                icon={<PlayCircleOutlined />}
              >
                Start Tuning
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Job Detail Modal */}
      <Modal
        title={
          <Space>
            <BarChartOutlined />
            <span>Tuning Results</span>
            {detailModal && <Tag>Job #{detailModal.id}</Tag>}
          </Space>
        }
        open={!!detailModal}
        onCancel={() => setDetailModal(null)}
        footer={
          <Space>
            {detailModal?.best_config && (
              <Button
                type="primary"
                icon={<RocketOutlined />}
                onClick={() =>
                  detailModal && handleDeployBestConfig(detailModal)
                }
              >
                Deploy Best Configuration
              </Button>
            )}
            <Button onClick={() => setDetailModal(null)}>Close</Button>
          </Space>
        }
        width={800}
      >
        {detailModal && (
          <JobDetailCard
            job={detailModal}
            onClose={() => setDetailModal(null)}
          />
        )}
      </Modal>

      {/* Live Log Modal */}
      <Modal
        title={
          <Space>
            <LoadingOutlined spin />
            <span>Live Logs</span>
            {logModal && (
              <>
                <Tag>Job #{logModal.id}</Tag>
                <StatusTag status={logModal.status} />
              </>
            )}
          </Space>
        }
        open={!!logModal}
        onCancel={() => setLogModal(null)}
        footer={
          <Space>
            <Button onClick={() => fetchJobs()}>
              <ReloadOutlined /> Refresh
            </Button>
            <Button onClick={() => setLogModal(null)}>Close</Button>
          </Space>
        }
        width={800}
      >
        {logModal && (
          <div>
            {/* Progress */}
            <Card size="small" style={{ marginBottom: 16 }}>
              <Row gutter={16} align="middle">
                <Col span={12}>
                  <Space direction="vertical" size={0}>
                    <Text strong>{logModal.model_name}</Text>
                    <Text type="secondary">{logModal.status_message}</Text>
                  </Space>
                </Col>
                <Col span={12}>
                  <ProgressDisplay job={logModal} />
                </Col>
              </Row>
            </Card>

            {/* Logs */}
            <LogViewer logs={logModal.logs || []} maxHeight={400} />

            <div style={{ marginTop: 12, textAlign: "center" }}>
              <Text type="secondary">
                Logs auto-refresh every {REFRESH_INTERVAL / 1000} seconds
              </Text>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
