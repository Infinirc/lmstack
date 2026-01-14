/**
 * Dashboard Page - Clean Professional Style
 *
 * System overview with minimal, data-focused design
 *
 * @module pages/Dashboard
 */
import { useEffect, useState, useCallback } from "react";
import { Col, Row, Empty } from "antd";
import Loading from "../components/Loading";
import { CheckCircleOutlined, CloseCircleOutlined } from "@ant-design/icons";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
} from "recharts";
import { dashboardApi, workersApi } from "../services/api";
import { TIMING } from "../constants";
import { useAppTheme } from "../hooks/useTheme";
import type { DashboardData, Worker, GPUInfo } from "../types";

// ============================================================================
// Color Constants
// ============================================================================

// Status colors for data-driven visualization
const STATUS_COLORS = {
  good: { light: "#22c55e", dark: "#4ade80" }, // Green
  warning: { light: "#f59e0b", dark: "#fbbf24" }, // Orange/Yellow
  critical: { light: "#ef4444", dark: "#f87171" }, // Red
  neutral: { light: "#3b82f6", dark: "#60a5fa" }, // Blue
};

/**
 * Get color based on percentage value
 * 0-50%: green (good)
 * 50-80%: yellow/orange (warning)
 * 80-100%: red (critical)
 */
function getStatusColor(percentage: number, isDark: boolean): string {
  if (percentage >= 80) {
    return isDark ? STATUS_COLORS.critical.dark : STATUS_COLORS.critical.light;
  } else if (percentage >= 50) {
    return isDark ? STATUS_COLORS.warning.dark : STATUS_COLORS.warning.light;
  } else {
    return isDark ? STATUS_COLORS.good.dark : STATUS_COLORS.good.light;
  }
}

/**
 * Get temperature color
 * <60°C: green
 * 60-80°C: yellow/orange
 * >80°C: red
 */
function getTemperatureColor(temp: number, isDark: boolean): string {
  if (temp >= 80) {
    return isDark ? STATUS_COLORS.critical.dark : STATUS_COLORS.critical.light;
  } else if (temp >= 60) {
    return isDark ? STATUS_COLORS.warning.dark : STATUS_COLORS.warning.light;
  } else {
    return isDark ? STATUS_COLORS.good.dark : STATUS_COLORS.good.light;
  }
}

// Chart accent colors
const CHART_COLORS = {
  light: "#3b82f6",
  dark: "#60a5fa",
};

// ============================================================================
// Metric Card Component
// ============================================================================

interface MetricCardProps {
  label: string;
  value: string | number;
  sublabel?: string;
  isDark: boolean;
}

function MetricCard({ label, value, sublabel, isDark }: MetricCardProps) {
  return (
    <div
      style={{
        padding: "24px",
        background: isDark ? "rgba(255, 255, 255, 0.03)" : "#ffffff",
        borderRadius: 12,
        border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.06)" : "#e2e8f0"}`,
        height: "100%",
        boxShadow: isDark ? "none" : "0 1px 3px rgba(0, 0, 0, 0.05)",
      }}
    >
      <div
        style={{
          fontSize: 12,
          fontWeight: 500,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          color: isDark ? "rgba(255, 255, 255, 0.5)" : "#64748b",
          marginBottom: 8,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 32,
          fontWeight: 600,
          color: isDark ? "#ffffff" : "#0f172a",
          lineHeight: 1.2,
          fontFamily:
            '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        }}
      >
        {typeof value === "number" ? value.toLocaleString() : value}
      </div>
      {sublabel && (
        <div
          style={{
            marginTop: 4,
            fontSize: 13,
            color: isDark ? "rgba(255, 255, 255, 0.4)" : "#64748b",
          }}
        >
          {sublabel}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Chart Card Component
// ============================================================================

interface ChartCardProps {
  title: string;
  children: React.ReactNode;
  isDark: boolean;
}

function ChartCard({ title, children, isDark }: ChartCardProps) {
  return (
    <div
      style={{
        padding: 24,
        background: isDark ? "rgba(255, 255, 255, 0.03)" : "#ffffff",
        borderRadius: 12,
        border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.06)" : "#e2e8f0"}`,
        height: "100%",
        boxShadow: isDark ? "none" : "0 1px 3px rgba(0, 0, 0, 0.05)",
      }}
    >
      <div
        style={{
          fontSize: 13,
          fontWeight: 500,
          color: isDark ? "rgba(255, 255, 255, 0.6)" : "#475569",
          marginBottom: 16,
        }}
      >
        {title}
      </div>
      <div style={{ height: 220 }}>{children}</div>
    </div>
  );
}

// ============================================================================
// Progress Bar Component
// ============================================================================

interface ProgressBarProps {
  label: string;
  value: number;
  max: number;
  unit: string;
  isDark: boolean;
  colorType?: "usage" | "temperature" | "neutral";
}

function ProgressBar({
  label,
  value,
  max,
  unit,
  isDark,
  colorType = "usage",
}: ProgressBarProps) {
  const percentage = max > 0 ? Math.round((value / max) * 100) : 0;

  // Get color based on colorType
  let barColor: string;
  let textColor: string;

  if (colorType === "temperature") {
    barColor = getTemperatureColor(value, isDark);
    textColor = barColor;
  } else if (colorType === "neutral") {
    barColor = isDark ? CHART_COLORS.dark : CHART_COLORS.light;
    textColor = isDark ? "#ffffff" : "#0f172a";
  } else {
    barColor = getStatusColor(percentage, isDark);
    textColor = barColor;
  }

  return (
    <div
      style={{
        padding: 20,
        background: isDark ? "rgba(255, 255, 255, 0.03)" : "#ffffff",
        borderRadius: 12,
        border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.06)" : "#e2e8f0"}`,
        boxShadow: isDark ? "none" : "0 1px 3px rgba(0, 0, 0, 0.05)",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 12,
        }}
      >
        <span
          style={{
            fontSize: 13,
            color: isDark ? "rgba(255, 255, 255, 0.6)" : "#475569",
          }}
        >
          {label}
        </span>
        <span
          style={{
            fontSize: 13,
            fontWeight: 500,
            color: isDark ? "#ffffff" : "#0f172a",
          }}
        >
          {value.toFixed(1)} / {max.toFixed(1)} {unit}
        </span>
      </div>
      <div
        style={{
          height: 8,
          background: isDark ? "rgba(255, 255, 255, 0.08)" : "#e2e8f0",
          borderRadius: 4,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${percentage}%`,
            background: barColor,
            borderRadius: 4,
            transition: "width 0.3s ease, background-color 0.3s ease",
          }}
        />
      </div>
      <div
        style={{
          marginTop: 8,
          fontSize: 24,
          fontWeight: 600,
          color: textColor,
        }}
      >
        {percentage}%
      </div>
    </div>
  );
}

// ============================================================================
// GPU Card Component
// ============================================================================

interface GPUCardProps {
  gpu: GPUInfo;
  workerName: string;
  isDark: boolean;
}

function GPUCard({ gpu, workerName, isDark }: GPUCardProps) {
  const memoryPercent =
    gpu.memory_total > 0
      ? Math.round((gpu.memory_used / gpu.memory_total) * 100)
      : 0;
  const memoryUsedGB = (gpu.memory_used / 1024 ** 3).toFixed(1);
  const memoryTotalGB = (gpu.memory_total / 1024 ** 3).toFixed(1);

  // Data-driven colors
  const vramColor = getStatusColor(memoryPercent, isDark);
  const utilizationColor = getStatusColor(gpu.utilization || 0, isDark);
  const temperatureColor = getTemperatureColor(gpu.temperature || 0, isDark);

  return (
    <div
      style={{
        padding: 20,
        background: isDark ? "rgba(255, 255, 255, 0.03)" : "#ffffff",
        borderRadius: 12,
        border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.06)" : "#e2e8f0"}`,
        boxShadow: isDark ? "none" : "0 1px 3px rgba(0, 0, 0, 0.05)",
      }}
    >
      <div style={{ marginBottom: 16 }}>
        <div
          style={{
            fontSize: 14,
            fontWeight: 500,
            color: isDark ? "#ffffff" : "#0f172a",
            marginBottom: 4,
          }}
        >
          {gpu.name}
        </div>
        <div
          style={{
            fontSize: 12,
            color: isDark ? "rgba(255, 255, 255, 0.4)" : "#64748b",
          }}
        >
          {workerName} · GPU {gpu.index}
        </div>
      </div>

      {/* VRAM Progress */}
      <div style={{ marginBottom: 16 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 6,
          }}
        >
          <span
            style={{
              fontSize: 12,
              color: isDark ? "rgba(255, 255, 255, 0.5)" : "#64748b",
            }}
          >
            VRAM
          </span>
          <span style={{ fontSize: 12, color: vramColor, fontWeight: 500 }}>
            {memoryUsedGB} / {memoryTotalGB} GB
          </span>
        </div>
        <div
          style={{
            height: 6,
            background: isDark ? "rgba(255, 255, 255, 0.08)" : "#e2e8f0",
            borderRadius: 3,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${memoryPercent}%`,
              background: vramColor,
              borderRadius: 3,
              transition: "width 0.3s ease, background-color 0.3s ease",
            }}
          />
        </div>
      </div>

      {/* Stats Row */}
      <div style={{ display: "flex", gap: 24 }}>
        <div>
          <div
            style={{
              fontSize: 11,
              color: isDark ? "rgba(255, 255, 255, 0.4)" : "#64748b",
              marginBottom: 2,
            }}
          >
            Utilization
          </div>
          <div
            style={{ fontSize: 18, fontWeight: 600, color: utilizationColor }}
          >
            {gpu.utilization || 0}%
          </div>
        </div>
        <div>
          <div
            style={{
              fontSize: 11,
              color: isDark ? "rgba(255, 255, 255, 0.4)" : "#64748b",
              marginBottom: 2,
            }}
          >
            Temperature
          </div>
          <div
            style={{ fontSize: 18, fontWeight: 600, color: temperatureColor }}
          >
            {gpu.temperature || 0}°C
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Worker Card Component
// ============================================================================

interface WorkerCardProps {
  worker: Worker;
  isDark: boolean;
}

function WorkerCard({ worker, isDark }: WorkerCardProps) {
  const isOnline = worker.status === "online";
  const gpuCount = worker.gpu_info?.length || 0;
  const totalVRAM =
    worker.gpu_info?.reduce((sum, gpu) => sum + gpu.memory_total, 0) || 0;
  const totalVRAMGB = (totalVRAM / 1024 ** 3).toFixed(0);

  return (
    <div
      style={{
        padding: 20,
        background: isDark ? "rgba(255, 255, 255, 0.03)" : "#ffffff",
        borderRadius: 12,
        border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.06)" : "#e2e8f0"}`,
        boxShadow: isDark ? "none" : "0 1px 3px rgba(0, 0, 0, 0.05)",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: isOnline ? "#22c55e" : "#ef4444",
          }}
        />
        <div>
          <div
            style={{
              fontSize: 14,
              fontWeight: 500,
              color: isDark ? "#ffffff" : "#0f172a",
              marginBottom: 2,
            }}
          >
            {worker.name}
          </div>
          <div
            style={{
              fontSize: 12,
              color: isDark ? "rgba(255, 255, 255, 0.4)" : "#64748b",
            }}
          >
            {worker.address}
          </div>
        </div>
      </div>

      <div style={{ textAlign: "right" }}>
        <div
          style={{
            fontSize: 12,
            color: isOnline ? "#22c55e" : "#ef4444",
            fontWeight: 500,
            display: "flex",
            alignItems: "center",
            gap: 4,
            justifyContent: "flex-end",
            marginBottom: 2,
          }}
        >
          {isOnline ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
          {worker.status.toUpperCase()}
        </div>
        <div
          style={{
            fontSize: 12,
            color: isDark ? "rgba(255, 255, 255, 0.5)" : "#64748b",
          }}
        >
          {gpuCount} GPU{gpuCount !== 1 ? "s" : ""} · {totalVRAMGB} GB
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Section Header Component
// ============================================================================

interface SectionHeaderProps {
  title: string;
  count?: number;
  isDark: boolean;
}

function SectionHeader({ title, count, isDark }: SectionHeaderProps) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        marginBottom: 16,
      }}
    >
      <h2
        style={{
          fontSize: 14,
          fontWeight: 600,
          color: isDark ? "rgba(255, 255, 255, 0.8)" : "#1e293b",
          margin: 0,
        }}
      >
        {title}
      </h2>
      {count !== undefined && (
        <span
          style={{
            fontSize: 12,
            padding: "2px 8px",
            background: isDark ? "rgba(255, 255, 255, 0.08)" : "#e2e8f0",
            borderRadius: 10,
            color: isDark ? "rgba(255, 255, 255, 0.6)" : "#475569",
            fontWeight: 500,
          }}
        >
          {count}
        </span>
      )}
    </div>
  );
}

// ============================================================================
// Deployment Card Component
// ============================================================================

interface DeploymentCardProps {
  model: { model_name: string; request_count: number; token_count: number };
  isDark: boolean;
}

function DeploymentCard({ model, isDark }: DeploymentCardProps) {
  return (
    <div
      style={{
        padding: 20,
        background: isDark ? "rgba(255, 255, 255, 0.03)" : "#ffffff",
        borderRadius: 12,
        border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.06)" : "#e2e8f0"}`,
        boxShadow: isDark ? "none" : "0 1px 3px rgba(0, 0, 0, 0.05)",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <div>
        <div
          style={{
            fontSize: 14,
            fontWeight: 500,
            color: isDark ? "#ffffff" : "#0f172a",
            marginBottom: 4,
          }}
        >
          {model.model_name}
        </div>
        <div
          style={{
            fontSize: 12,
            color: isDark ? "rgba(255, 255, 255, 0.4)" : "#64748b",
          }}
        >
          {model.request_count.toLocaleString()} requests
        </div>
      </div>
      <div style={{ textAlign: "right" }}>
        <div
          style={{
            fontSize: 20,
            fontWeight: 600,
            color: isDark ? CHART_COLORS.dark : CHART_COLORS.light,
          }}
        >
          {(model.token_count / 1000).toFixed(1)}K
        </div>
        <div
          style={{
            fontSize: 11,
            color: isDark ? "rgba(255, 255, 255, 0.4)" : "#64748b",
          }}
        >
          tokens
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Custom Tooltip Component
// ============================================================================

interface CustomTooltipProps {
  active?: boolean;
  payload?: { value: number }[];
  label?: string;
  isDark: boolean;
  formatter?: (value: number) => string;
}

function CustomTooltip({
  active,
  payload,
  label,
  isDark,
  formatter,
}: CustomTooltipProps) {
  if (!active || !payload?.length) return null;

  return (
    <div
      style={{
        background: isDark ? "#1a1a1a" : "#ffffff",
        border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.1)" : "#e2e8f0"}`,
        borderRadius: 8,
        padding: "10px 14px",
        boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
      }}
    >
      <div
        style={{
          fontSize: 11,
          color: isDark ? "rgba(255, 255, 255, 0.5)" : "#64748b",
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      {payload.map((item, idx) => (
        <div
          key={idx}
          style={{
            fontSize: 14,
            fontWeight: 600,
            color: isDark ? CHART_COLORS.dark : CHART_COLORS.light,
          }}
        >
          {formatter ? formatter(item.value) : item.value.toLocaleString()}
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export default function Dashboard() {
  const { isDark } = useAppTheme();
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const [dashboardRes, workersRes] = await Promise.all([
        dashboardApi.get(),
        workersApi.list(),
      ]);
      setDashboard(dashboardRes);
      setWorkers(workersRes.items);
    } catch (error) {
      console.error("Failed to fetch dashboard data:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, TIMING.DASHBOARD_REFRESH);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (loading) {
    return (
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "50vh",
        }}
      >
        <Loading size="large" />
      </div>
    );
  }

  if (!dashboard) {
    return <Empty description="Failed to load dashboard data" />;
  }

  const { resources, gpu_summary, usage } = dashboard;

  // Collect all GPUs from online workers
  const allGPUs: { gpu: GPUInfo; workerName: string }[] = [];
  workers.forEach((worker) => {
    if (worker.gpu_info && worker.status === "online") {
      worker.gpu_info.forEach((gpu) => {
        allGPUs.push({ gpu, workerName: worker.name });
      });
    }
  });

  // Chart data
  const requestChartData =
    usage.request_history?.map((point) => ({
      date: point.date,
      value: point.value,
    })) || [];

  const tokenChartData =
    usage.token_history?.map((point) => ({
      date: point.date,
      value: point.value,
    })) || [];

  const chartStroke = isDark ? CHART_COLORS.dark : CHART_COLORS.light;
  const gridColor = isDark ? "rgba(255, 255, 255, 0.06)" : "#e2e8f0";
  const tickColor = isDark ? "rgba(255, 255, 255, 0.4)" : "#64748b";

  return (
    <div style={{ maxWidth: 1400, margin: "0 auto" }}>
      {/* Key Metrics */}
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} lg={6}>
          <MetricCard
            label="Workers"
            value={resources.worker_count}
            sublabel={`${resources.worker_online_count} online`}
            isDark={isDark}
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <MetricCard
            label="GPUs"
            value={resources.gpu_count}
            sublabel={`${gpu_summary.total_memory_gb.toFixed(0)} GB VRAM`}
            isDark={isDark}
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <MetricCard
            label="Models"
            value={resources.model_count}
            sublabel={`${resources.deployment_running_count} deployed`}
            isDark={isDark}
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <MetricCard
            label="API Requests"
            value={usage.total_requests}
            sublabel="Last 30 days"
            isDark={isDark}
          />
        </Col>
      </Row>

      {/* Workers Section */}
      <div style={{ marginTop: 32 }}>
        <SectionHeader title="Workers" count={workers.length} isDark={isDark} />
        <Row gutter={[16, 16]}>
          {workers.map((worker) => (
            <Col xs={24} md={12} lg={8} key={worker.id}>
              <WorkerCard worker={worker} isDark={isDark} />
            </Col>
          ))}
          {workers.length === 0 && (
            <Col span={24}>
              <div
                style={{
                  padding: 40,
                  textAlign: "center",
                  background: isDark ? "rgba(255, 255, 255, 0.03)" : "#ffffff",
                  borderRadius: 12,
                  border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.06)" : "#e2e8f0"}`,
                  color: isDark ? "rgba(255, 255, 255, 0.4)" : "#64748b",
                  boxShadow: isDark ? "none" : "0 1px 3px rgba(0, 0, 0, 0.05)",
                }}
              >
                No workers registered
              </div>
            </Col>
          )}
        </Row>
      </div>

      {/* GPU Overview */}
      <div style={{ marginTop: 32 }}>
        <SectionHeader
          title="GPU Overview"
          count={allGPUs.length}
          isDark={isDark}
        />

        {/* GPU Summary */}
        <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
          <Col xs={24} md={8}>
            <ProgressBar
              label="VRAM Usage"
              value={gpu_summary.used_memory_gb}
              max={gpu_summary.total_memory_gb}
              unit="GB"
              isDark={isDark}
              colorType="usage"
            />
          </Col>
          <Col xs={24} md={8}>
            <ProgressBar
              label="Avg Utilization"
              value={gpu_summary.utilization_avg}
              max={100}
              unit="%"
              isDark={isDark}
              colorType="usage"
            />
          </Col>
          <Col xs={24} md={8}>
            <ProgressBar
              label="Avg Temperature"
              value={gpu_summary.temperature_avg || 0}
              max={100}
              unit="°C"
              isDark={isDark}
              colorType="temperature"
            />
          </Col>
        </Row>

        {/* GPU Cards */}
        <Row gutter={[16, 16]}>
          {allGPUs.map(({ gpu, workerName }, idx) => (
            <Col
              xs={24}
              sm={12}
              lg={8}
              xl={6}
              key={`${workerName}-${gpu.index}-${idx}`}
            >
              <GPUCard gpu={gpu} workerName={workerName} isDark={isDark} />
            </Col>
          ))}
          {allGPUs.length === 0 && (
            <Col span={24}>
              <div
                style={{
                  padding: 40,
                  textAlign: "center",
                  background: isDark ? "rgba(255, 255, 255, 0.03)" : "#ffffff",
                  borderRadius: 12,
                  border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.06)" : "#e2e8f0"}`,
                  color: isDark ? "rgba(255, 255, 255, 0.4)" : "#64748b",
                  boxShadow: isDark ? "none" : "0 1px 3px rgba(0, 0, 0, 0.05)",
                }}
              >
                No GPUs available
              </div>
            </Col>
          )}
        </Row>
      </div>

      {/* Active Deployments */}
      <div style={{ marginTop: 32 }}>
        <SectionHeader
          title="Active Deployments"
          count={resources.deployment_running_count}
          isDark={isDark}
        />
        <Row gutter={[16, 16]}>
          {dashboard.top_models.length > 0 ? (
            dashboard.top_models.map((model, idx) => (
              <Col xs={24} sm={12} lg={8} key={idx}>
                <DeploymentCard model={model} isDark={isDark} />
              </Col>
            ))
          ) : (
            <Col span={24}>
              <div
                style={{
                  padding: 40,
                  textAlign: "center",
                  background: isDark ? "rgba(255, 255, 255, 0.03)" : "#ffffff",
                  borderRadius: 12,
                  border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.06)" : "#e2e8f0"}`,
                  color: isDark ? "rgba(255, 255, 255, 0.4)" : "#64748b",
                  boxShadow: isDark ? "none" : "0 1px 3px rgba(0, 0, 0, 0.05)",
                }}
              >
                No deployments yet
              </div>
            </Col>
          )}
        </Row>
      </div>

      {/* Usage Charts */}
      <div style={{ marginTop: 32, marginBottom: 32 }}>
        <SectionHeader title="Usage Statistics" isDark={isDark} />
        <Row gutter={[16, 16]}>
          <Col xs={24} lg={12}>
            <ChartCard title="API Requests (30 Days)" isDark={isDark}>
              {requestChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={requestChartData}>
                    <defs>
                      <linearGradient
                        id="requestGradient"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="0%"
                          stopColor={chartStroke}
                          stopOpacity={0.15}
                        />
                        <stop
                          offset="100%"
                          stopColor={chartStroke}
                          stopOpacity={0}
                        />
                      </linearGradient>
                    </defs>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke={gridColor}
                      vertical={false}
                    />
                    <XAxis
                      dataKey="date"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: tickColor, fontSize: 11 }}
                      tickFormatter={(value) => {
                        const date = new Date(value);
                        return `${date.getMonth() + 1}/${date.getDate()}`;
                      }}
                    />
                    <YAxis
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: tickColor, fontSize: 11 }}
                      width={40}
                    />
                    <RechartsTooltip
                      content={<CustomTooltip isDark={isDark} />}
                    />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke={chartStroke}
                      strokeWidth={1.5}
                      fill="url(#requestGradient)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div
                  style={{
                    height: "100%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: isDark ? "rgba(255,255,255,0.3)" : "#64748b",
                  }}
                >
                  No data available
                </div>
              )}
            </ChartCard>
          </Col>
          <Col xs={24} lg={12}>
            <ChartCard title="Token Usage (30 Days)" isDark={isDark}>
              {tokenChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={tokenChartData}>
                    <defs>
                      <linearGradient
                        id="tokenGradient"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="0%"
                          stopColor={chartStroke}
                          stopOpacity={0.15}
                        />
                        <stop
                          offset="100%"
                          stopColor={chartStroke}
                          stopOpacity={0}
                        />
                      </linearGradient>
                    </defs>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke={gridColor}
                      vertical={false}
                    />
                    <XAxis
                      dataKey="date"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: tickColor, fontSize: 11 }}
                      tickFormatter={(value) => {
                        const date = new Date(value);
                        return `${date.getMonth() + 1}/${date.getDate()}`;
                      }}
                    />
                    <YAxis
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: tickColor, fontSize: 11 }}
                      width={50}
                      tickFormatter={(value) => `${(value / 1000).toFixed(0)}K`}
                    />
                    <RechartsTooltip
                      content={
                        <CustomTooltip
                          isDark={isDark}
                          formatter={(v) => `${v.toLocaleString()} tokens`}
                        />
                      }
                    />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke={chartStroke}
                      strokeWidth={1.5}
                      fill="url(#tokenGradient)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div
                  style={{
                    height: "100%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: isDark ? "rgba(255,255,255,0.3)" : "#64748b",
                  }}
                >
                  No data available
                </div>
              )}
            </ChartCard>
          </Col>
        </Row>
      </div>
    </div>
  );
}
