/**
 * Tuning Job View Component
 *
 * Displays a tuning job's conversation in the Chat Panel with a terminal-like UI.
 */

import { useEffect, useState, useRef, useCallback } from "react";
import { Tag, Spin } from "antd";
import {
  LoadingOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  RobotOutlined,
  ToolOutlined,
} from "@ant-design/icons";
import { api } from "../../api/client";
import dayjs from "dayjs";

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
  name?: string;
}

interface TuningJob {
  id: number;
  model_name?: string;
  worker_name?: string;
  optimization_target: string;
  status: string;
  status_message?: string;
  conversation_log?: ConversationMessage[];
  best_config?: Record<string, unknown>;
}

interface TuningJobViewProps {
  jobId: number;
  isDark: boolean;
}

function getStatusIcon(status: string) {
  switch (status) {
    case "completed":
      return <CheckCircleOutlined style={{ color: "#52c41a" }} />;
    case "failed":
    case "cancelled":
      return <CloseCircleOutlined style={{ color: "#ff4d4f" }} />;
    default:
      return <LoadingOutlined spin style={{ color: "#1677ff" }} />;
  }
}

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

// Tool result display component - shows results in a more readable format
interface ToolResultDisplayProps {
  name: string;
  content: string;
  timestamp?: string;
  isDark: boolean;
  colors: Record<string, string>;
}

function ToolResultDisplay({
  name,
  content,
  isDark,
  colors,
}: ToolResultDisplayProps) {
  const [expanded, setExpanded] = useState(false);

  // Parse and format the result
  let parsedResult: Record<string, unknown> | null = null;
  let isError = false;
  let summary = "";

  try {
    parsedResult = JSON.parse(content);
    isError = !!parsedResult?.error;

    // Generate summary based on tool name
    if (name === "get_hardware_info" && parsedResult) {
      const gpuCount = parsedResult.gpu_count || 0;
      const gpus =
        (parsedResult.gpus as Array<{
          name: string;
          memory_total_gb: number;
        }>) || [];
      const gpuName = gpus[0]?.name || "Unknown GPU";
      const totalVram = parsedResult.total_vram_gb || 0;
      summary = `${gpuCount}x ${gpuName}, ${totalVram}GB VRAM total`;
    } else if (name === "get_model_info" && parsedResult) {
      summary = `${parsedResult.name} (${parsedResult.model_family})`;
    } else if (name === "query_knowledge_base" && parsedResult) {
      const found = (parsedResult.found as number) || 0;
      summary =
        found > 0
          ? `Found ${found} historical record(s)`
          : "No historical data found";
    } else if (name === "deploy_model" && parsedResult) {
      summary = parsedResult.success
        ? `Deployment #${parsedResult.deployment_id} created`
        : `Failed: ${parsedResult.error}`;
    } else if (name === "wait_for_deployment" && parsedResult) {
      summary = parsedResult.success
        ? `Ready in ${parsedResult.wait_time_seconds}s`
        : `Failed: ${parsedResult.error}`;
    } else if (name === "run_benchmark" && parsedResult) {
      if (parsedResult.success && parsedResult.metrics) {
        const m = parsedResult.metrics as Record<string, number>;
        summary = `${m.throughput_tps} TPS, TTFT: ${m.avg_ttft_ms}ms, TPOT: ${m.avg_tpot_ms}ms`;
      } else {
        summary = `Failed: ${parsedResult.error}`;
      }
    } else if (name === "stop_deployment" && parsedResult) {
      summary = parsedResult.success
        ? "Deployment stopped"
        : `Failed: ${parsedResult.error}`;
    } else if (name === "finish_tuning" && parsedResult) {
      summary = "Tuning completed successfully";
    } else if (isError) {
      summary = `Error: ${parsedResult?.error}`;
    } else {
      summary = "Completed";
    }
  } catch {
    summary = content.length > 50 ? content.slice(0, 50) + "..." : content;
  }

  return (
    <div
      style={{
        borderRadius: 8,
        background: isError
          ? isDark
            ? "rgba(239, 68, 68, 0.1)"
            : "rgba(239, 68, 68, 0.05)"
          : colors.toolBg,
        border: `1px solid ${isError ? (isDark ? "#7f1d1d" : "#fecaca") : colors.toolBorder}`,
        overflow: "hidden",
      }}
    >
      {/* Header - clickable to expand */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          padding: "8px 12px",
          display: "flex",
          alignItems: "center",
          gap: 8,
          cursor: "pointer",
          background: isError
            ? isDark
              ? "rgba(239, 68, 68, 0.15)"
              : "rgba(239, 68, 68, 0.1)"
            : isDark
              ? "rgba(255,255,255,0.02)"
              : "rgba(0,0,0,0.02)",
        }}
      >
        {isError ? (
          <CloseCircleOutlined style={{ color: "#ef4444", fontSize: 12 }} />
        ) : (
          <CheckCircleOutlined style={{ color: "#22c55e", fontSize: 12 }} />
        )}
        <span style={{ fontSize: 12, color: colors.textSecondary, flex: 1 }}>
          {summary}
        </span>
        <span
          style={{
            fontSize: 10,
            color: colors.textMuted,
            transform: expanded ? "rotate(180deg)" : "rotate(0deg)",
            transition: "transform 0.2s",
          }}
        >
          ▼
        </span>
      </div>

      {/* Expanded content */}
      {expanded && (
        <div
          style={{
            padding: "8px 12px",
            borderTop: `1px solid ${colors.border}`,
          }}
        >
          <pre
            style={{
              margin: 0,
              fontSize: 11,
              fontFamily:
                "'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace",
              maxHeight: 300,
              overflow: "auto",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              color: colors.textSecondary,
              lineHeight: 1.5,
            }}
          >
            {parsedResult ? JSON.stringify(parsedResult, null, 2) : content}
          </pre>
        </div>
      )}
    </div>
  );
}

export function TuningJobView({ jobId, isDark }: TuningJobViewProps) {
  const [job, setJob] = useState<TuningJob | null>(null);
  const [loading, setLoading] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Theme colors
  const colors = {
    bg: isDark ? "#0d0d0d" : "#ffffff",
    cardBg: isDark ? "#1a1a1a" : "#f8f9fa",
    border: isDark ? "#2a2a2a" : "#e8e8e8",
    text: isDark ? "#e4e4e7" : "#18181b",
    textSecondary: isDark ? "#a1a1aa" : "#71717a",
    textMuted: isDark ? "#71717a" : "#a1a1aa",
    accent: "#3b82f6",
    agentBg: isDark ? "#1e293b" : "#f0f9ff",
    agentBorder: isDark ? "#334155" : "#bae6fd",
    toolBg: isDark ? "#18181b" : "#f4f4f5",
    toolBorder: isDark ? "#3f3f46" : "#d4d4d8",
    successBg: isDark ? "#14532d" : "#dcfce7",
    successBorder: isDark ? "#166534" : "#86efac",
  };

  // Fetch job data
  const fetchJob = useCallback(async () => {
    try {
      const response = await api.get(`/auto-tuning/jobs/${jobId}`);
      setJob(response.data);
      setLoading(false);
    } catch (error) {
      console.error("Failed to fetch tuning job:", error);
      setLoading(false);
    }
  }, [jobId]);

  // Initial fetch
  useEffect(() => {
    fetchJob();
  }, [fetchJob]);

  // Auto-refresh for running jobs
  useEffect(() => {
    if (!job) return;
    const isRunning = [
      "pending",
      "analyzing",
      "querying_kb",
      "exploring",
      "benchmarking",
    ].includes(job.status);
    if (!isRunning) return;

    const interval = setInterval(fetchJob, 2000);
    return () => clearInterval(interval);
  }, [job?.status, fetchJob]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [job?.conversation_log?.length]);

  const isRunning =
    job &&
    [
      "pending",
      "analyzing",
      "querying_kb",
      "exploring",
      "benchmarking",
    ].includes(job.status);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        flex: 1,
        background: colors.bg,
        overflow: "hidden",
      }}
    >
      {/* Status Bar */}
      <div
        style={{
          padding: "10px 16px",
          borderBottom: `1px solid ${colors.border}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: colors.cardBg,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {job && (
            <>
              <Tag
                color={getStatusColor(job.status)}
                icon={getStatusIcon(job.status)}
                style={{ margin: 0, fontWeight: 500 }}
              >
                {job.status.toUpperCase()}
              </Tag>
              <span style={{ fontSize: 13, color: colors.textSecondary }}>
                {job.model_name}
              </span>
            </>
          )}
        </div>
        {job && (
          <span style={{ fontSize: 12, color: colors.textMuted }}>
            {job.worker_name} · {job.optimization_target}
          </span>
        )}
      </div>

      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: 16,
          background: colors.bg,
        }}
      >
        {loading ? (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              gap: 12,
            }}
          >
            <Spin size="large" />
            <span style={{ color: colors.textMuted, fontSize: 14 }}>
              Loading tuning job...
            </span>
          </div>
        ) : job?.conversation_log && job.conversation_log.length > 0 ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {job.conversation_log.map((msg, idx) => (
              <div key={idx}>
                {/* Agent Message */}
                {msg.role === "assistant" && (
                  <div style={{ display: "flex", gap: 10 }}>
                    {/* Avatar */}
                    <div
                      style={{
                        width: 32,
                        height: 32,
                        borderRadius: 8,
                        background: `linear-gradient(135deg, ${colors.accent}, #8b5cf6)`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        flexShrink: 0,
                      }}
                    >
                      <RobotOutlined style={{ color: "#fff", fontSize: 16 }} />
                    </div>
                    {/* Content */}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div
                        style={{
                          fontSize: 12,
                          color: colors.textMuted,
                          marginBottom: 4,
                          display: "flex",
                          alignItems: "center",
                          gap: 8,
                        }}
                      >
                        <span
                          style={{
                            fontWeight: 500,
                            color: colors.textSecondary,
                          }}
                        >
                          Agent
                        </span>
                        {msg.timestamp && (
                          <span>{dayjs(msg.timestamp).format("HH:mm:ss")}</span>
                        )}
                      </div>
                      {/* Agent text content */}
                      {msg.content && (
                        <div
                          style={{
                            padding: "12px 14px",
                            borderRadius: 12,
                            background: colors.agentBg,
                            border: `1px solid ${colors.agentBorder}`,
                            color: colors.text,
                            fontSize: 14,
                            lineHeight: 1.6,
                            whiteSpace: "pre-wrap",
                          }}
                        >
                          {msg.content}
                        </div>
                      )}
                      {/* Tool Calls - show what actions the agent is taking */}
                      {msg.tool_calls && msg.tool_calls.length > 0 && (
                        <div
                          style={{
                            marginTop: msg.content ? 8 : 0,
                            padding: "10px 12px",
                            borderRadius: 8,
                            background: isDark ? "#1c1c1e" : "#f0f0f0",
                            border: `1px solid ${colors.border}`,
                          }}
                        >
                          <div
                            style={{
                              fontSize: 11,
                              color: colors.textMuted,
                              marginBottom: 6,
                              textTransform: "uppercase",
                              letterSpacing: "0.5px",
                            }}
                          >
                            Executing Actions
                          </div>
                          {msg.tool_calls.map((tc, tcIdx) => {
                            let argsPreview = "";
                            try {
                              const args = JSON.parse(tc.arguments);
                              argsPreview = Object.entries(args)
                                .map(
                                  ([k, v]) =>
                                    `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`,
                                )
                                .join(", ");
                            } catch {
                              argsPreview = tc.arguments;
                            }
                            return (
                              <div
                                key={tcIdx}
                                style={{
                                  display: "flex",
                                  alignItems: "flex-start",
                                  gap: 8,
                                  padding: "6px 0",
                                  borderTop:
                                    tcIdx > 0
                                      ? `1px solid ${colors.border}`
                                      : "none",
                                }}
                              >
                                <ToolOutlined
                                  style={{ color: colors.accent, marginTop: 2 }}
                                />
                                <div>
                                  <div
                                    style={{
                                      fontWeight: 500,
                                      color: colors.text,
                                      fontSize: 13,
                                    }}
                                  >
                                    {tc.name
                                      .replace(/_/g, " ")
                                      .replace(/\b\w/g, (c) => c.toUpperCase())}
                                  </div>
                                  {argsPreview && (
                                    <div
                                      style={{
                                        fontSize: 11,
                                        color: colors.textMuted,
                                        marginTop: 2,
                                        fontFamily: "monospace",
                                      }}
                                    >
                                      {argsPreview.length > 80
                                        ? argsPreview.slice(0, 80) + "..."
                                        : argsPreview}
                                    </div>
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Tool Response */}
                {msg.role === "tool" && (
                  <div style={{ marginLeft: 42, marginTop: 4 }}>
                    <ToolResultDisplay
                      name={msg.name || "unknown"}
                      content={msg.content}
                      timestamp={msg.timestamp}
                      isDark={isDark}
                      colors={colors}
                    />
                  </div>
                )}

                {/* User Message (system prompt) */}
                {msg.role === "user" && (
                  <div
                    style={{
                      padding: "8px 12px",
                      borderRadius: 8,
                      background: isDark ? "#1e1e1e" : "#fafafa",
                      border: `1px dashed ${colors.border}`,
                      fontSize: 12,
                      color: colors.textMuted,
                    }}
                  >
                    <span style={{ fontWeight: 500 }}>System: </span>
                    {msg.content.length > 100
                      ? msg.content.slice(0, 100) + "..."
                      : msg.content}
                  </div>
                )}
              </div>
            ))}

            {/* Running indicator */}
            {isRunning && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "12px 14px",
                  borderRadius: 12,
                  background: colors.agentBg,
                  border: `1px solid ${colors.agentBorder}`,
                  marginLeft: 42,
                }}
              >
                <LoadingOutlined
                  spin
                  style={{ color: colors.accent, fontSize: 16 }}
                />
                <span style={{ color: colors.textSecondary, fontSize: 14 }}>
                  {job.status_message || "Processing..."}
                </span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        ) : (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              gap: 12,
            }}
          >
            <div
              style={{
                width: 48,
                height: 48,
                borderRadius: 12,
                background: `linear-gradient(135deg, ${colors.accent}, #8b5cf6)`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <RobotOutlined style={{ color: "#fff", fontSize: 24 }} />
            </div>
            <span style={{ color: colors.textMuted, fontSize: 14 }}>
              {job?.status === "pending"
                ? "Waiting for agent to start..."
                : "No conversation yet"}
            </span>
          </div>
        )}
      </div>

      {/* Best Config */}
      {job?.best_config && (
        <div
          style={{
            padding: 16,
            borderTop: `1px solid ${colors.border}`,
            background: colors.successBg,
          }}
        >
          <div
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: isDark ? "#4ade80" : "#16a34a",
              marginBottom: 8,
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <CheckCircleOutlined />
            Best Configuration Found
          </div>
          <pre
            style={{
              margin: 0,
              padding: 12,
              background: colors.bg,
              borderRadius: 8,
              border: `1px solid ${colors.successBorder}`,
              fontSize: 12,
              fontFamily:
                "'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace",
              maxHeight: 120,
              overflow: "auto",
              color: colors.text,
            }}
          >
            {JSON.stringify(job.best_config, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
