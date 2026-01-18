/**
 * API Keys Page
 *
 * Manage API keys with usage statistics and code examples.
 * SpaceX/OpenAI inspired modern design.
 *
 * @module pages/ApiKeys
 */
import { useEffect, useState, useCallback, useMemo } from "react";
import {
  Button,
  Card,
  Form,
  Input,
  InputNumber,
  Modal,
  Select,
  Space,
  Table,
  Tag,
  message,
  Popconfirm,
  Typography,
  Tooltip,
  Row,
  Col,
  Segmented,
  Tabs,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import {
  PlusOutlined,
  DeleteOutlined,
  CopyOutlined,
  KeyOutlined,
  CodeOutlined,
  ApiOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  EyeOutlined,
  EyeInvisibleOutlined,
} from "@ant-design/icons";
import {
  apiKeysApi,
  deploymentsApi,
  modelsApi,
  type ApiKeyStats,
} from "../services/api";
import type { ApiKey, ApiKeyCreate, Deployment, LLMModel } from "../types";
import { useAppTheme, useResponsive } from "../hooks";
import { useAuth } from "../contexts/AuthContext";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";

dayjs.extend(utc);

const { Text } = Typography;

// ============================================================================
// Types
// ============================================================================

interface NewKeyModalData {
  name: string;
  key: string;
}

// ============================================================================
// Code Examples Component
// ============================================================================

interface CodeExamplesProps {
  apiKey: string;
  baseUrl: string;
  model?: string;
}

function CodeExamples({
  apiKey,
  baseUrl,
  model = "your-model-name",
}: CodeExamplesProps) {
  const [activeTab, setActiveTab] = useState("curl");

  const examples = useMemo(
    () => ({
      curl: `curl ${baseUrl}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${apiKey}" \\
  -d '{
    "model": "${model}",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'`,

      python: `from openai import OpenAI

client = OpenAI(
    base_url="${baseUrl}/v1",
    api_key="${apiKey}"
)

response = client.chat.completions.create(
    model="${model}",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)`,

      nodejs: `import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: '${baseUrl}/v1',
  apiKey: '${apiKey}',
});

const response = await client.chat.completions.create({
  model: '${model}',
  messages: [
    { role: 'user', content: 'Hello!' }
  ],
});

console.log(response.choices[0].message.content);`,
    }),
    [apiKey, baseUrl, model],
  );

  const copyCode = async () => {
    const text = examples[activeTab as keyof typeof examples];
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        message.success("Code copied to clipboard");
        return;
      }

      // Fallback for non-secure contexts (HTTP)
      const textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.style.position = "fixed";
      textArea.style.left = "-999999px";
      textArea.style.top = "-999999px";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();

      const successful = document.execCommand("copy");
      document.body.removeChild(textArea);

      if (successful) {
        message.success("Code copied to clipboard");
      } else {
        message.error("Failed to copy");
      }
    } catch (err) {
      console.error("Copy failed:", err);
      message.error("Failed to copy");
    }
  };

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <Segmented
          value={activeTab}
          onChange={(v) => setActiveTab(v as string)}
          options={[
            { label: "cURL", value: "curl" },
            { label: "Python", value: "python" },
            { label: "Node.js", value: "nodejs" },
          ]}
        />
        <Button icon={<CopyOutlined />} onClick={copyCode} type="text">
          Copy
        </Button>
      </div>
      <pre
        style={{
          background: "#0a0a0a",
          color: "#e5e5e5",
          padding: 16,
          borderRadius: 8,
          fontSize: 13,
          lineHeight: 1.6,
          overflow: "auto",
          maxHeight: 400,
          border: "1px solid #27272a",
        }}
      >
        <code>{examples[activeTab as keyof typeof examples]}</code>
      </pre>
    </div>
  );
}

// ============================================================================
// Stats Card Component
// ============================================================================

interface StatsCardProps {
  title: string;
  value: number | string;
  icon: React.ReactNode;
  suffix?: string;
  isDark: boolean;
}

function StatsCard({ title, value, icon, suffix, isDark }: StatsCardProps) {
  return (
    <Card
      style={{
        background: isDark
          ? "linear-gradient(135deg, #18181b 0%, #09090b 100%)"
          : "linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)",
        border: `1px solid ${isDark ? "#27272a" : "#e2e8f0"}`,
        borderRadius: 12,
      }}
      styles={{ body: { padding: "20px 24px" } }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
        }}
      >
        <div>
          <Text
            style={{
              color: isDark ? "#71717a" : "#64748b",
              fontSize: 13,
              fontWeight: 500,
            }}
          >
            {title}
          </Text>
          <div style={{ marginTop: 8 }}>
            <Text
              style={{
                color: isDark ? "#fafafa" : "#0f172a",
                fontSize: 28,
                fontWeight: 600,
                letterSpacing: "-0.02em",
              }}
            >
              {typeof value === "number" ? value.toLocaleString() : value}
            </Text>
            {suffix && (
              <Text
                style={{
                  color: isDark ? "#71717a" : "#64748b",
                  fontSize: 14,
                  marginLeft: 4,
                }}
              >
                {suffix}
              </Text>
            )}
          </div>
        </div>
        <div
          style={{
            width: 40,
            height: 40,
            borderRadius: 10,
            background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: isDark ? "#a1a1aa" : "#64748b",
          }}
        >
          {icon}
        </div>
      </div>
    </Card>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export default function ApiKeys() {
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [models, setModels] = useState<LLMModel[]>([]);
  const [runningDeployments, setRunningDeployments] = useState<Deployment[]>(
    [],
  );
  const [stats, setStats] = useState<ApiKeyStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [codeModalKey, setCodeModalKey] = useState<ApiKey | null>(null);
  const [newKeyModal, setNewKeyModal] = useState<NewKeyModalData | null>(null);
  const [revealedKeys, setRevealedKeys] = useState<Set<number>>(new Set());
  const [createForm] = Form.useForm();
  const { isMobile } = useResponsive();
  const { isDark } = useAppTheme();
  const { canEdit } = useAuth();

  // Get base URL for API Gateway (always port 52000)
  const baseUrl = useMemo(() => {
    const { protocol, hostname } = window.location;
    return `${protocol}//${hostname}:52000`;
  }, []);

  const fetchData = useCallback(async () => {
    try {
      const [keysRes, modelsRes, deploymentsRes, statsRes] = await Promise.all([
        apiKeysApi.list(),
        modelsApi.list(),
        deploymentsApi.list(),
        apiKeysApi.getStats().catch(() => null),
      ]);
      setApiKeys(keysRes.items);
      setModels(modelsRes.items);
      // Filter to running deployments only
      setRunningDeployments(
        deploymentsRes.items.filter((d) => d.status === "running"),
      );
      setStats(statsRes);
    } catch (error) {
      console.error("Failed to fetch data:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleCreate = async (values: ApiKeyCreate) => {
    try {
      const response = await apiKeysApi.create(values);
      message.success("API key created");
      setCreateModalOpen(false);
      createForm.resetFields();
      setNewKeyModal({ name: response.name, key: response.api_key });
      fetchData();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to create API key");
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await apiKeysApi.delete(id);
      message.success("API key deleted");
      fetchData();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to delete API key");
    }
  };

  const copyToClipboard = async (text: string, label = "Copied") => {
    try {
      // Try modern clipboard API first
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        message.success(label);
        return;
      }

      // Fallback for non-secure contexts (HTTP)
      const textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.style.position = "fixed";
      textArea.style.left = "-999999px";
      textArea.style.top = "-999999px";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();

      const successful = document.execCommand("copy");
      document.body.removeChild(textArea);

      if (successful) {
        message.success(label);
      } else {
        message.error("Failed to copy");
      }
    } catch (err) {
      console.error("Copy failed:", err);
      message.error("Failed to copy");
    }
  };

  const toggleReveal = (id: number) => {
    setRevealedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  // Mobile columns
  const mobileColumns: ColumnsType<ApiKey> = [
    {
      title: "API Key",
      key: "apikey",
      render: (_: unknown, record: ApiKey) => {
        const isExpired =
          record.expires_at && dayjs(record.expires_at).isBefore(dayjs());
        const keyStats = stats?.per_key_stats?.[record.id];
        return (
          <div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 4,
              }}
            >
              <Text strong style={{ fontSize: 14 }}>
                {record.name}
              </Text>
              <Tag
                color={isExpired ? "error" : "success"}
                style={{ margin: 0 }}
              >
                {isExpired ? "Expired" : "Active"}
              </Tag>
            </div>
            <code
              style={{
                background: "#18181b",
                padding: "2px 6px",
                borderRadius: 4,
                fontSize: 11,
                color: "#a1a1aa",
                border: "1px solid #27272a",
              }}
            >
              lmsk_{record.access_key.slice(0, 4)}••••
            </code>
            {keyStats && (
              <div style={{ fontSize: 11, color: "#888", marginTop: 4 }}>
                {keyStats.requests.toLocaleString()} req ·{" "}
                {keyStats.tokens.toLocaleString()} tokens
              </div>
            )}
          </div>
        );
      },
    },
    {
      title: "",
      key: "actions",
      width: 80,
      render: (_: unknown, record: ApiKey) => (
        <Space direction="vertical" size={4}>
          <Button
            type="text"
            size="small"
            icon={<CodeOutlined />}
            onClick={() => setCodeModalKey(record)}
          />
          {canEdit && (
            <Popconfirm
              title="Delete API key?"
              description="Applications using this key will stop working."
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
          )}
        </Space>
      ),
    },
  ];

  // Desktop columns
  const desktopColumns: ColumnsType<ApiKey> = [
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
      render: (name: string, record: ApiKey) => (
        <div>
          <Text strong style={{ fontSize: 14 }}>
            {name}
          </Text>
          {record.description && (
            <div>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {record.description}
              </Text>
            </div>
          )}
        </div>
      ),
    },
    {
      title: "Key",
      dataIndex: "access_key",
      key: "access_key",
      width: 280,
      render: (accessKey: string, record: ApiKey) => {
        const isRevealed = revealedKeys.has(record.id);
        const displayKey = isRevealed
          ? `lmsk_${accessKey}_••••••••`
          : `lmsk_${accessKey.slice(0, 4)}••••••••••••`;

        return (
          <Space>
            <code
              style={{
                background: "#18181b",
                padding: "4px 8px",
                borderRadius: 4,
                fontSize: 12,
                color: "#a1a1aa",
                border: "1px solid #27272a",
              }}
            >
              {displayKey}
            </code>
            <Tooltip title={isRevealed ? "Hide" : "Reveal"}>
              <Button
                type="text"
                size="small"
                icon={isRevealed ? <EyeInvisibleOutlined /> : <EyeOutlined />}
                onClick={() => toggleReveal(record.id)}
              />
            </Tooltip>
            <Tooltip title="Copy key prefix">
              <Button
                type="text"
                size="small"
                icon={<CopyOutlined />}
                onClick={() =>
                  copyToClipboard(`lmsk_${accessKey}`, "Key prefix copied")
                }
              />
            </Tooltip>
          </Space>
        );
      },
    },
    {
      title: "Usage (30d)",
      key: "usage",
      width: 160,
      render: (_: unknown, record: ApiKey) => {
        const keyStats = stats?.per_key_stats?.[record.id];
        const usedTokens = keyStats?.tokens || 0;
        const limit = record.monthly_token_limit;
        const isOverLimit = limit && usedTokens > limit;

        return (
          <div>
            <div>
              <Text style={{ fontSize: 13 }}>
                {(keyStats?.requests || 0).toLocaleString()} requests
              </Text>
            </div>
            <div>
              <Text
                style={{ fontSize: 12 }}
                type={isOverLimit ? "danger" : "secondary"}
              >
                {usedTokens.toLocaleString()}
                {limit ? ` / ${limit.toLocaleString()}` : ""}
                {" tokens"}
              </Text>
            </div>
          </div>
        );
      },
    },
    {
      title: "Status",
      key: "status",
      width: 100,
      render: (_: unknown, record: ApiKey) => {
        if (record.expires_at && dayjs(record.expires_at).isBefore(dayjs())) {
          return <Tag color="error">Expired</Tag>;
        }
        return <Tag color="success">Active</Tag>;
      },
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      width: 100,
      render: (date: string) => (
        <Text type="secondary" style={{ fontSize: 12 }}>
          {dayjs.utc(date).local().format("MMM D, YYYY")}
        </Text>
      ),
    },
    {
      title: "",
      key: "actions",
      width: 120,
      render: (_: unknown, record: ApiKey) => (
        <Space>
          <Tooltip title="View Code">
            <Button
              type="text"
              icon={<CodeOutlined />}
              onClick={() => setCodeModalKey(record)}
            />
          </Tooltip>
          {canEdit && (
            <Popconfirm
              title="Delete API key?"
              description="Applications using this key will stop working."
              onConfirm={() => handleDelete(record.id)}
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

  // Only show models that have running deployments
  const runningModelIds = new Set(runningDeployments.map((d) => d.model_id));
  const availableModels = models.filter((m) => runningModelIds.has(m.id));

  const modelOptions = availableModels.map((m) => ({
    label: m.name,
    value: m.id,
  }));

  // Use first running model name for code examples
  const defaultModel =
    runningDeployments.length > 0
      ? runningDeployments[0].model?.name || runningDeployments[0].name
      : availableModels.length > 0
        ? availableModels[0].name
        : "your-model-name";

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          flexDirection: isMobile ? "column" : "row",
          justifyContent: "space-between",
          alignItems: isMobile ? "flex-start" : "center",
          gap: isMobile ? 16 : 0,
          marginBottom: isMobile ? 20 : 32,
        }}
      >
        <div>
          <h1
            style={{
              margin: 0,
              fontSize: isMobile ? 22 : 28,
              fontWeight: 600,
              letterSpacing: "-0.02em",
            }}
          >
            API Keys
          </h1>
          <Text type="secondary" style={{ fontSize: isMobile ? 13 : 14 }}>
            Manage your API keys for programmatic access
          </Text>
        </div>
        {canEdit && (
          <Button
            type="primary"
            icon={<PlusOutlined />}
            size={isMobile ? "middle" : "large"}
            onClick={() => setCreateModalOpen(true)}
            style={{ borderRadius: 8 }}
          >
            {isMobile ? "Create" : "Create Key"}
          </Button>
        )}
      </div>

      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <StatsCard
            title="Total Keys"
            value={apiKeys.length}
            icon={<KeyOutlined style={{ fontSize: 18 }} />}
            isDark={isDark}
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatsCard
            title="Total Requests"
            value={stats?.total_requests || 0}
            icon={<ApiOutlined style={{ fontSize: 18 }} />}
            suffix="30d"
            isDark={isDark}
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatsCard
            title="Total Tokens"
            value={stats?.total_tokens || 0}
            icon={<ThunderboltOutlined style={{ fontSize: 18 }} />}
            suffix="30d"
            isDark={isDark}
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatsCard
            title="Active Keys"
            value={
              apiKeys.filter(
                (k) => !k.expires_at || dayjs(k.expires_at).isAfter(dayjs()),
              ).length
            }
            icon={<CheckCircleOutlined style={{ fontSize: 18 }} />}
            isDark={isDark}
          />
        </Col>
      </Row>

      {/* Quick Start - Hidden on mobile */}
      {!isMobile && (
        <Card
          style={{ marginBottom: 24, borderRadius: 12 }}
          bodyStyle={{ padding: 0 }}
        >
          <div
            style={{ padding: "16px 24px", borderBottom: "1px solid #27272a" }}
          >
            <Space>
              <CodeOutlined />
              <Text strong>Quick Start</Text>
            </Space>
          </div>
          <div style={{ padding: 24 }}>
            <Text
              type="secondary"
              style={{ display: "block", marginBottom: 8 }}
            >
              Use the OpenAI SDK with your LMStack endpoint. Base URL:{" "}
              <code>{baseUrl}/v1</code>
            </Text>
            <Text
              type="secondary"
              style={{ display: "block", marginBottom: 16, fontSize: 12 }}
            >
              For Docker containers (e.g., Open WebUI, n8n), use:{" "}
              <code>http://172.17.0.1:52000/v1</code>
            </Text>
            <Tabs
              items={[
                {
                  key: "python",
                  label: "Python",
                  children: (
                    <pre
                      style={{
                        background: "#0a0a0a",
                        color: "#e5e5e5",
                        padding: 16,
                        borderRadius: 8,
                        fontSize: 13,
                        margin: 0,
                        border: "1px solid #27272a",
                      }}
                    >
                      {`pip install openai

from openai import OpenAI
client = OpenAI(base_url="${baseUrl}/v1", api_key="YOUR_API_KEY")`}
                    </pre>
                  ),
                },
                {
                  key: "nodejs",
                  label: "Node.js",
                  children: (
                    <pre
                      style={{
                        background: "#0a0a0a",
                        color: "#e5e5e5",
                        padding: 16,
                        borderRadius: 8,
                        fontSize: 13,
                        margin: 0,
                        border: "1px solid #27272a",
                      }}
                    >
                      {`npm install openai

import OpenAI from 'openai';
const client = new OpenAI({ baseURL: '${baseUrl}/v1', apiKey: 'YOUR_API_KEY' });`}
                    </pre>
                  ),
                },
                {
                  key: "curl",
                  label: "cURL",
                  children: (
                    <pre
                      style={{
                        background: "#0a0a0a",
                        color: "#e5e5e5",
                        padding: 16,
                        borderRadius: 8,
                        fontSize: 13,
                        margin: 0,
                        border: "1px solid #27272a",
                      }}
                    >
                      {`curl ${baseUrl}/v1/chat/completions \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${defaultModel}", "messages": [{"role": "user", "content": "Hello"}]}'`}
                    </pre>
                  ),
                },
              ]}
            />
          </div>
        </Card>
      )}

      {/* API Keys Table */}
      <Card style={{ borderRadius: 12 }}>
        <Table
          dataSource={apiKeys}
          columns={isMobile ? mobileColumns : desktopColumns}
          rowKey="id"
          loading={loading}
          pagination={false}
          size={isMobile ? "small" : "middle"}
          locale={{ emptyText: "No API keys yet. Create one to get started." }}
        />
      </Card>

      {/* Create Modal */}
      <Modal
        title={null}
        open={createModalOpen}
        onCancel={() => {
          setCreateModalOpen(false);
          createForm.resetFields();
        }}
        footer={null}
        width={isMobile ? "100%" : 480}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        <div style={{ marginBottom: 24 }}>
          <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600 }}>
            Create API Key
          </h2>
          <Text type="secondary">Create a new key for API access</Text>
        </div>

        <Form form={createForm} layout="vertical" onFinish={handleCreate}>
          <Form.Item
            name="name"
            label="Name"
            rules={[{ required: true, message: "Enter a name" }]}
          >
            <Input placeholder="e.g., Production API" size="large" />
          </Form.Item>

          <Form.Item name="description" label="Description">
            <Input.TextArea placeholder="Optional description" rows={2} />
          </Form.Item>

          <Form.Item
            name="allowed_model_ids"
            label="Model Access"
            extra={
              availableModels.length > 0
                ? "Leave empty for all running models"
                : "No models are currently running"
            }
          >
            <Select
              mode="multiple"
              placeholder={
                availableModels.length > 0
                  ? "All running models"
                  : "No running models"
              }
              options={modelOptions}
              allowClear
              size="large"
              disabled={availableModels.length === 0}
            />
          </Form.Item>

          <Form.Item
            name="monthly_token_limit"
            label="Monthly Token Limit (tokens)"
            extra="Leave empty for unlimited"
          >
            <InputNumber<number>
              min={1000}
              placeholder="Unlimited"
              style={{ width: "100%" }}
              size="large"
              formatter={(value) =>
                value ? `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ",") : ""
              }
              parser={(value) => (value ? Number(value.replace(/,/g, "")) : 0)}
            />
          </Form.Item>

          <Form.Item
            name="expires_in_days"
            label="Expiration (days)"
            extra="Leave empty for no expiration"
          >
            <InputNumber
              min={1}
              max={365}
              placeholder="Never expires"
              style={{ width: "100%" }}
              size="large"
            />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0, marginTop: 24 }}>
            <Space style={{ width: "100%", justifyContent: "flex-end" }}>
              <Button size="large" onClick={() => setCreateModalOpen(false)}>
                Cancel
              </Button>
              <Button type="primary" htmlType="submit" size="large">
                Create Key
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* New Key Modal */}
      <Modal
        title={null}
        open={!!newKeyModal}
        onCancel={() => setNewKeyModal(null)}
        footer={
          <Button
            type="primary"
            size={isMobile ? "middle" : "large"}
            block
            onClick={() => setNewKeyModal(null)}
          >
            Done
          </Button>
        }
        width={isMobile ? "100%" : 560}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        {newKeyModal && (
          <div>
            <div style={{ textAlign: "center", marginBottom: 24 }}>
              <div
                style={{
                  width: 64,
                  height: 64,
                  borderRadius: 16,
                  background:
                    "linear-gradient(135deg, #22c55e 0%, #16a34a 100%)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  margin: "0 auto 16px",
                }}
              >
                <KeyOutlined style={{ fontSize: 28, color: "#fff" }} />
              </div>
              <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600 }}>
                API Key Created
              </h2>
              <Text type="secondary">
                Save this key now. You won't be able to see it again.
              </Text>
            </div>

            <div
              style={{
                background: "#0a0a0a",
                border: "1px solid #27272a",
                borderRadius: 8,
                padding: 16,
                marginBottom: 24,
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 8,
                }}
              >
                <Text type="secondary" style={{ fontSize: 12 }}>
                  API KEY
                </Text>
                <Button
                  type="text"
                  size="small"
                  icon={<CopyOutlined />}
                  onClick={() =>
                    copyToClipboard(newKeyModal.key, "API key copied")
                  }
                >
                  Copy
                </Button>
              </div>
              <code
                style={{
                  display: "block",
                  fontSize: 14,
                  color: "#22c55e",
                  wordBreak: "break-all",
                  lineHeight: 1.6,
                }}
              >
                {newKeyModal.key}
              </code>
            </div>

            <CodeExamples
              apiKey={newKeyModal.key}
              baseUrl={baseUrl}
              model={defaultModel}
            />
          </div>
        )}
      </Modal>

      {/* View Code Modal */}
      <Modal
        title={null}
        open={!!codeModalKey}
        onCancel={() => setCodeModalKey(null)}
        footer={null}
        width={isMobile ? "100%" : 640}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        {codeModalKey && (
          <div>
            <div style={{ marginBottom: 24 }}>
              <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600 }}>
                Use API Key: {codeModalKey.name}
              </h2>
              <Text type="secondary">
                Copy these code examples to integrate with your application
              </Text>
            </div>

            <div
              style={{
                background: "#18181b",
                border: "1px solid #27272a",
                borderRadius: 8,
                padding: 12,
                marginBottom: 24,
              }}
            >
              <Text
                type="secondary"
                style={{ fontSize: 12, display: "block", marginBottom: 4 }}
              >
                BASE URL
              </Text>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <code style={{ color: "#fafafa", flex: 1 }}>{baseUrl}/v1</code>
                <Button
                  type="text"
                  size="small"
                  icon={<CopyOutlined />}
                  onClick={() =>
                    copyToClipboard(`${baseUrl}/v1`, "Base URL copied")
                  }
                />
              </div>
            </div>

            <CodeExamples
              apiKey={`lmsk_${codeModalKey.access_key}_YOUR_SECRET`}
              baseUrl={baseUrl}
              model={defaultModel}
            />
          </div>
        )}
      </Modal>
    </div>
  );
}
