/**
 * Model Selector Component
 *
 * Dropdown for selecting chat models from deployments, Semantic Router,
 * or custom OpenAI-compatible endpoints.
 */
import { useState, useEffect, useCallback } from "react";
import {
  Dropdown,
  Button,
  Modal,
  Form,
  Input,
  Space,
  message,
  Tooltip,
  Select,
  Spin,
} from "antd";
import {
  RobotOutlined,
  ThunderboltOutlined,
  PlusOutlined,
  ApiOutlined,
  DeleteOutlined,
  DownOutlined,
  EditOutlined,
  SyncOutlined,
} from "@ant-design/icons";
import { deploymentsApi, semanticRouterApi } from "../../services/api";
import { api } from "../../api/client";
import type { Deployment } from "../../types";
import type { SemanticRouterStatus } from "../../services/api";
import type { ChatModelConfig, CustomEndpoint } from "./types";
import type { AppColors } from "../../hooks/useTheme";

interface RemoteModel {
  id: string;
  owned_by: string;
}

interface ModelSelectorProps {
  value: ChatModelConfig | null;
  onChange: (config: ChatModelConfig | null) => void;
  customEndpoints: CustomEndpoint[];
  onCustomEndpointsChange: (endpoints: CustomEndpoint[]) => void;
  isDark: boolean;
  colors: AppColors;
  compact?: boolean;
}

/**
 * Model selector with support for multiple model sources
 */
export function ModelSelector({
  value,
  onChange,
  customEndpoints,
  onCustomEndpointsChange,
  isDark,
  colors,
  compact = false,
}: ModelSelectorProps) {
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [semanticRouterStatus, setSemanticRouterStatus] =
    useState<SemanticRouterStatus | null>(null);
  const [customModalOpen, setCustomModalOpen] = useState(false);
  const [editingEndpoint, setEditingEndpoint] = useState<CustomEndpoint | null>(
    null,
  );
  const [form] = Form.useForm();

  // Remote models state
  const [remoteModels, setRemoteModels] = useState<RemoteModel[]>([]);
  const [fetchingModels, setFetchingModels] = useState(false);

  // Fetch models from remote endpoint
  const fetchRemoteModels = useCallback(async () => {
    const endpoint = form.getFieldValue("endpoint");
    if (!endpoint) {
      message.warning("Please enter a Base URL first");
      return;
    }

    setFetchingModels(true);
    setRemoteModels([]);

    try {
      const response = await api.post<{ models: RemoteModel[] }>(
        "/fetch-models",
        {
          endpoint,
          api_key: form.getFieldValue("apiKey") || null,
        },
      );

      if (response.data.models.length === 0) {
        message.info("No models found at this endpoint");
      } else {
        setRemoteModels(response.data.models);
        // Auto-select if only one model
        if (response.data.models.length === 1) {
          form.setFieldValue("modelId", response.data.models[0].id);
        }
        message.success(`Found ${response.data.models.length} model(s)`);
      }
    } catch (error) {
      console.error("Failed to fetch models:", error);
      message.error("Failed to fetch models from endpoint");
    } finally {
      setFetchingModels(false);
    }
  }, [form]);

  // Fetch deployments
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [deploymentsRes, srStatus] = await Promise.all([
          deploymentsApi.list({ status: "running" }),
          semanticRouterApi.getStatus().catch(() => null),
        ]);
        setDeployments(deploymentsRes.items);
        setSemanticRouterStatus(srStatus);
      } catch (error) {
        console.error("Failed to fetch models:", error);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  // Handle custom endpoint save
  const handleSaveCustomEndpoint = useCallback(async () => {
    try {
      const values = await form.validateFields();
      const endpoint: CustomEndpoint = {
        id: editingEndpoint?.id || `custom_${Date.now()}`,
        name: values.name,
        endpoint: values.endpoint,
        apiKey: values.apiKey || undefined,
        modelId: values.modelId || undefined,
      };

      if (editingEndpoint) {
        onCustomEndpointsChange(
          customEndpoints.map((e) =>
            e.id === editingEndpoint.id ? endpoint : e,
          ),
        );
      } else {
        onCustomEndpointsChange([...customEndpoints, endpoint]);
      }

      // Auto-select the new endpoint
      onChange({
        type: "custom",
        name: endpoint.name,
        modelId: endpoint.modelId,
        endpoint: endpoint.endpoint,
        apiKey: endpoint.apiKey,
      });

      setCustomModalOpen(false);
      setEditingEndpoint(null);
      form.resetFields();
      message.success(editingEndpoint ? "Endpoint updated" : "Endpoint added");
    } catch {
      // Validation failed
    }
  }, [
    form,
    editingEndpoint,
    customEndpoints,
    onCustomEndpointsChange,
    onChange,
  ]);

  // Handle custom endpoint delete
  const handleDeleteCustomEndpoint = useCallback(
    (id: string) => {
      onCustomEndpointsChange(customEndpoints.filter((e) => e.id !== id));
      if (
        value?.type === "custom" &&
        value.endpoint === customEndpoints.find((e) => e.id === id)?.endpoint
      ) {
        onChange(null);
      }
    },
    [customEndpoints, onCustomEndpointsChange, value, onChange],
  );

  // Build menu items
  const menuItems: any[] = [];

  // Semantic Router option
  if (semanticRouterStatus?.deployed) {
    menuItems.push({
      key: "semantic-router",
      label: (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "4px 0",
          }}
        >
          <ThunderboltOutlined style={{ color: "#faad14" }} />
          <div>
            <div>Semantic Router</div>
            <div style={{ fontSize: 11, color: colors.textMuted }}>
              Auto-select best model
            </div>
          </div>
        </div>
      ),
      onClick: () =>
        onChange({
          type: "semantic-router",
          name: "Semantic Router",
        }),
    });
  }

  // Deployments
  if (deployments.length > 0) {
    if (menuItems.length > 0) {
      menuItems.push({ type: "divider" });
    }
    menuItems.push({
      key: "deployments-header",
      label: (
        <span style={{ fontSize: 11, color: colors.textMuted }}>
          DEPLOYED MODELS
        </span>
      ),
      disabled: true,
    });

    deployments.forEach((d) => {
      menuItems.push({
        key: `deployment-${d.id}`,
        label: (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "4px 0",
            }}
          >
            <RobotOutlined />
            <div>
              <div>{d.model?.name || d.name}</div>
              <div style={{ fontSize: 11, color: colors.textMuted }}>
                @{d.worker?.name}
              </div>
            </div>
          </div>
        ),
        onClick: () =>
          onChange({
            type: "deployment",
            deploymentId: d.id,
            name: d.model?.name || d.name,
          }),
      });
    });
  }

  // Custom endpoints section (always shown to allow adding new endpoints)
  if (menuItems.length > 0) {
    menuItems.push({ type: "divider" });
  }
  menuItems.push({
    key: "custom-header",
    label: (
      <span style={{ fontSize: 11, color: colors.textMuted }}>
        CUSTOM ENDPOINTS
      </span>
    ),
    disabled: true,
  });

  customEndpoints.forEach((ep) => {
    menuItems.push({
      key: `custom-${ep.id}`,
      label: (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "4px 0",
            width: "100%",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <ApiOutlined />
            <div>
              <div>{ep.name}</div>
              <div
                style={{
                  fontSize: 11,
                  color: colors.textMuted,
                  maxWidth: 180,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {ep.endpoint}
              </div>
            </div>
          </div>
          <Space size={4}>
            <Tooltip title="Edit">
              <Button
                type="text"
                size="small"
                icon={<EditOutlined style={{ fontSize: 12 }} />}
                onClick={(e) => {
                  e.stopPropagation();
                  setEditingEndpoint(ep);
                  form.setFieldsValue(ep);
                  setRemoteModels([]);
                  setCustomModalOpen(true);
                }}
                style={{ width: 24, height: 24, minWidth: 24 }}
              />
            </Tooltip>
            <Tooltip title="Delete">
              <Button
                type="text"
                size="small"
                danger
                icon={<DeleteOutlined style={{ fontSize: 12 }} />}
                onClick={(e) => {
                  e.stopPropagation();
                  handleDeleteCustomEndpoint(ep.id);
                }}
                style={{ width: 24, height: 24, minWidth: 24 }}
              />
            </Tooltip>
          </Space>
        </div>
      ),
      onClick: () =>
        onChange({
          type: "custom",
          name: ep.name,
          modelId: ep.modelId,
          endpoint: ep.endpoint,
          apiKey: ep.apiKey,
        }),
    });
  });

  // Add custom endpoint button
  menuItems.push({
    key: "add-custom",
    label: (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "4px 0",
          color: colors.textSecondary,
        }}
      >
        <PlusOutlined />
        <span>Add Custom Endpoint</span>
      </div>
    ),
    onClick: () => {
      setEditingEndpoint(null);
      form.resetFields();
      setRemoteModels([]);
      setCustomModalOpen(true);
    },
  });

  // Get display label
  const getDisplayLabel = () => {
    if (!value) return "Select Model";
    if (value.type === "semantic-router") return "Semantic Router";
    return value.name;
  };

  // Get display icon
  const getDisplayIcon = () => {
    if (!value) return <RobotOutlined />;
    if (value.type === "semantic-router")
      return <ThunderboltOutlined style={{ color: "#faad14" }} />;
    if (value.type === "custom") return <ApiOutlined />;
    return <RobotOutlined />;
  };

  return (
    <>
      <Dropdown
        menu={{
          items: menuItems,
          style: {
            background: isDark ? "#1f1f1f" : "#ffffff",
            borderRadius: 12,
            maxHeight: 400,
            overflow: "auto",
            boxShadow: isDark
              ? "0 6px 16px rgba(0, 0, 0, 0.4)"
              : "0 6px 16px rgba(0, 0, 0, 0.12)",
          },
        }}
        trigger={["click"]}
        placement="bottomLeft"
      >
        <Button
          type="text"
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            height: compact ? 32 : 36,
            padding: compact ? "0 8px" : "0 12px",
            borderRadius: 8,
            background: colors.menuItemHover,
            border: "none",
            fontSize: compact ? 13 : 14,
            fontWeight: 500,
            color: colors.text,
            maxWidth: compact ? 160 : 220,
          }}
        >
          {getDisplayIcon()}
          <span
            style={{
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {getDisplayLabel()}
          </span>
          <DownOutlined
            style={{ fontSize: 10, color: colors.textMuted, flexShrink: 0 }}
          />
        </Button>
      </Dropdown>

      {/* Custom endpoint modal */}
      <Modal
        title={editingEndpoint ? "Edit Endpoint" : "Add Custom Endpoint"}
        open={customModalOpen}
        onOk={handleSaveCustomEndpoint}
        onCancel={() => {
          setCustomModalOpen(false);
          setEditingEndpoint(null);
          form.resetFields();
          setRemoteModels([]);
        }}
        okText={editingEndpoint ? "Save" : "Add"}
        width={480}
      >
        <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
          <Form.Item
            name="name"
            label="Display Name"
            rules={[{ required: true, message: "Please enter a name" }]}
          >
            <Input placeholder="e.g., Local Ollama" />
          </Form.Item>
          <Form.Item
            name="endpoint"
            label="Base URL"
            rules={[{ required: true, message: "Please enter the base URL" }]}
            extra="OpenAI-compatible API base URL (e.g., http://ip:port/v1/)"
          >
            <Input placeholder="http://192.168.201.17:30000/v1/" />
          </Form.Item>
          <Form.Item label="Model" extra="Select a model or enter manually">
            <Space.Compact style={{ width: "100%" }}>
              <Form.Item name="modelId" noStyle>
                <Select
                  placeholder="Click refresh to detect models"
                  allowClear
                  showSearch
                  style={{ width: "100%" }}
                  loading={fetchingModels}
                  options={remoteModels.map((m) => ({
                    label: m.id,
                    value: m.id,
                  }))}
                  dropdownRender={(menu) => (
                    <>
                      {menu}
                      {remoteModels.length === 0 && !fetchingModels && (
                        <div
                          style={{
                            padding: 8,
                            textAlign: "center",
                            color: colors.textMuted,
                          }}
                        >
                          No models loaded
                        </div>
                      )}
                    </>
                  )}
                  notFoundContent={
                    fetchingModels ? (
                      <div style={{ padding: 8, textAlign: "center" }}>
                        <Spin size="small" /> Loading...
                      </div>
                    ) : null
                  }
                />
              </Form.Item>
              <Tooltip title="Detect available models">
                <Button
                  icon={<SyncOutlined spin={fetchingModels} />}
                  onClick={fetchRemoteModels}
                  loading={fetchingModels}
                />
              </Tooltip>
            </Space.Compact>
          </Form.Item>
          <Form.Item
            name="apiKey"
            label="API Key (Optional)"
            extra="Leave empty if not required"
          >
            <Input.Password placeholder="sk-..." />
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
}
