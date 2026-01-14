import { useEffect, useState } from "react";
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
} from "antd";
import { PlusOutlined, DeleteOutlined, EditOutlined } from "@ant-design/icons";
import { modelsApi } from "../services/api";
import type { LLMModel, LLMModelCreate } from "../types";
import dayjs from "dayjs";
import { useAppTheme } from "../hooks/useTheme";
import ModelCompatibilityCheck from "../components/ModelCompatibilityCheck";
import HuggingFaceModelPicker from "../components/HuggingFaceModelPicker";
import OllamaModelPicker from "../components/OllamaModelPicker";
import type { HFModelInfo } from "../services/api";

// Logo imports
import ollamaLogoDark from "../assets/ollama-dark.png";
import ollamaLogoLight from "../assets/ollama-light.png";
import huggingfaceLogo from "../assets/huggingface-2.svg";

// HuggingFace logo component
const HuggingFaceLogo = ({ height = 16 }: { height?: number }) => (
  <img
    src={huggingfaceLogo}
    alt="HuggingFace"
    style={{ height, width: "auto", objectFit: "contain" }}
  />
);

const OllamaLogo = ({
  height = 16,
  isDark,
}: {
  height?: number;
  isDark: boolean;
}) => (
  <img
    src={isDark ? ollamaLogoDark : ollamaLogoLight}
    alt="Ollama"
    style={{ height, width: "auto", objectFit: "contain" }}
  />
);

// Source configuration
const getSourceConfig = (isDark: boolean) => {
  // Use white text on dark mode, black text on light mode
  const tagColor = isDark ? "#ffffff" : "#000000";
  return {
    huggingface: {
      label: "HuggingFace",
      icon: <HuggingFaceLogo height={14} />,
      color: tagColor,
      placeholder: "e.g., Qwen/Qwen2.5-7B-Instruct",
      helpText: "HuggingFace model ID (org/model-name format)",
    },
    ollama: {
      label: "Ollama",
      icon: <OllamaLogo height={14} isDark={isDark} />,
      color: tagColor,
      placeholder: "e.g., qwen2.5:7b",
      helpText: "Ollama model name (name:tag format)",
    },
  };
};

// Auto-detect source from model_id
function detectSource(modelId: string): "huggingface" | "ollama" {
  // Ollama models: simple name or name:tag format (no slash)
  // HuggingFace models: org/model-name format (has slash)
  if (modelId.includes("/")) {
    return "huggingface";
  }
  return "ollama";
}

// Hook for responsive detection
function useResponsive() {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return { isMobile };
}

export default function Models() {
  const [models, setModels] = useState<LLMModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [editingModel, setEditingModel] = useState<LLMModel | null>(null);
  const [currentModelId, setCurrentModelId] = useState<string>("");
  const [currentSource, setCurrentSource] = useState<"huggingface" | "ollama">(
    "huggingface",
  );
  const [hfPickerOpen, setHfPickerOpen] = useState(false);
  const [ollamaPickerOpen, setOllamaPickerOpen] = useState(false);
  const [form] = Form.useForm();
  const [editForm] = Form.useForm();
  const { isMobile } = useResponsive();
  const { isDark } = useAppTheme();

  const SOURCE_CONFIG = getSourceConfig(isDark);

  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await modelsApi.list();
      setModels(response.items);
    } catch (error) {
      message.error("Failed to fetch models");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleCreate = async (values: LLMModelCreate) => {
    try {
      // Add source based on model_id
      const source = detectSource(values.model_id);
      await modelsApi.create({ ...values, source });
      message.success("Model created successfully");
      resetModalState();
      fetchModels();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to create model");
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await modelsApi.delete(id);
      message.success("Model deleted successfully");
      fetchModels();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to delete model");
    }
  };

  const handleEdit = (model: LLMModel) => {
    setEditingModel(model);
    editForm.setFieldsValue({
      name: model.name,
      description: model.description || "",
    });
    setEditModalOpen(true);
  };

  const handleUpdate = async (values: {
    name: string;
    description?: string;
  }) => {
    if (!editingModel) return;
    try {
      await modelsApi.update(editingModel.id, values);
      message.success("Model updated successfully");
      setEditModalOpen(false);
      setEditingModel(null);
      editForm.resetFields();
      fetchModels();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to update model");
    }
  };

  const resetModalState = () => {
    setModalOpen(false);
    form.resetFields();
    setCurrentModelId("");
    setCurrentSource("huggingface");
    setHfPickerOpen(false);
    setOllamaPickerOpen(false);
  };

  const handleHfModelSelect = (modelId: string, _modelInfo?: HFModelInfo) => {
    setCurrentModelId(modelId);
    setCurrentSource("huggingface");
    form.setFieldValue("model_id", modelId);
    // Always update name from model ID when selecting a new model
    const modelName = modelId.split("/").pop() || modelId;
    form.setFieldValue("name", modelName);
  };

  const handleOllamaModelSelect = (modelName: string, _tag?: string) => {
    setCurrentModelId(modelName);
    setCurrentSource("ollama");
    form.setFieldValue("model_id", modelName);
    // Always update display name from model name when selecting a new model
    const displayName = modelName.split(":")[0];
    form.setFieldValue("name", displayName);
  };

  // Helper to render source tag with icon
  const renderSourceTag = (source: string) => {
    const config = SOURCE_CONFIG[source as keyof typeof SOURCE_CONFIG];
    if (!config) return <Tag>{source}</Tag>;
    return (
      <Tag
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 4,
          background: isDark
            ? "rgba(255, 255, 255, 0.1)"
            : "rgba(0, 0, 0, 0.06)",
          borderColor: isDark
            ? "rgba(255, 255, 255, 0.2)"
            : "rgba(0, 0, 0, 0.15)",
          color: config.color,
        }}
      >
        {config.icon}
        {config.label}
      </Tag>
    );
  };

  // Mobile columns
  const mobileColumns = [
    {
      title: "Model",
      key: "model",
      render: (_: unknown, record: LLMModel) => (
        <div>
          <div style={{ fontWeight: 500 }}>{record.name}</div>
          <code style={{ fontSize: 11, color: "#888" }}>{record.model_id}</code>
          <div
            style={{
              marginTop: 4,
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            {renderSourceTag(record.source)}
            <span style={{ fontSize: 12, color: "#888" }}>
              {record.deployment_count} deployments
            </span>
          </div>
        </div>
      ),
    },
    {
      title: "",
      key: "actions",
      width: 80,
      render: (_: unknown, record: LLMModel) => (
        <Space>
          <Button
            type="text"
            size="small"
            icon={<EditOutlined />}
            onClick={() => handleEdit(record)}
          />
          <Popconfirm
            title="Delete this model?"
            description="This action cannot be undone."
            onConfirm={() => handleDelete(record.id)}
            okText="Delete"
            okButtonProps={{ danger: true }}
          >
            <Button type="text" size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  // Desktop columns
  const desktopColumns = [
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
      render: (name: string, record: LLMModel) => (
        <div>
          <div style={{ fontWeight: 500 }}>{name}</div>
          <code style={{ fontSize: 11, color: "#888" }}>{record.model_id}</code>
        </div>
      ),
    },
    {
      title: "Source",
      dataIndex: "source",
      key: "source",
      width: 140,
      render: (source: string) => renderSourceTag(source),
    },
    {
      title: "Deployments",
      dataIndex: "deployment_count",
      key: "deployment_count",
      width: 100,
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      width: 160,
      render: (time: string) => dayjs(time).format("YYYY-MM-DD HH:mm"),
    },
    {
      title: "Actions",
      key: "actions",
      width: 100,
      render: (_: unknown, record: LLMModel) => (
        <Space>
          <Button
            type="text"
            icon={<EditOutlined />}
            onClick={() => handleEdit(record)}
          />
          <Popconfirm
            title="Delete this model?"
            description="This action cannot be undone."
            onConfirm={() => handleDelete(record.id)}
            okText="Delete"
            okButtonProps={{ danger: true }}
          >
            <Button type="text" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Card
        style={{ borderRadius: 12 }}
        title="Models"
        extra={
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => setModalOpen(true)}
            size={isMobile ? "small" : "middle"}
          >
            {isMobile ? "Add" : "Add Model"}
          </Button>
        }
      >
        <Table
          dataSource={models}
          columns={isMobile ? mobileColumns : desktopColumns}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 10 }}
          scroll={isMobile ? undefined : { x: 800 }}
          size={isMobile ? "small" : "middle"}
        />
      </Card>

      <Modal
        title="Add Model"
        open={modalOpen}
        onCancel={resetModalState}
        footer={null}
        width={isMobile ? "100%" : 600}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        <Form form={form} layout="vertical" onFinish={handleCreate}>
          {/* Model Source Selection */}
          <Form.Item label="Select Model From">
            <Space size="middle">
              <Button
                size="large"
                onClick={() => setHfPickerOpen(true)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  height: 48,
                  paddingLeft: 16,
                  paddingRight: 16,
                }}
              >
                <HuggingFaceLogo height={20} />
                HuggingFace
              </Button>
              <Button
                size="large"
                onClick={() => setOllamaPickerOpen(true)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  height: 48,
                  paddingLeft: 16,
                  paddingRight: 16,
                }}
              >
                <OllamaLogo height={20} isDark={isDark} />
                Ollama
              </Button>
            </Space>
          </Form.Item>

          {/* Model ID */}
          <Form.Item
            label="Model ID"
            required
            extra={
              currentSource === "ollama"
                ? "Ollama model name (name:tag format)"
                : "HuggingFace model ID (org/model-name format)"
            }
          >
            <Form.Item
              name="model_id"
              noStyle
              rules={[{ required: true, message: "Please select a model" }]}
            >
              <Input
                placeholder={
                  currentSource === "ollama"
                    ? "e.g., qwen2.5:7b"
                    : "e.g., Qwen/Qwen2.5-7B-Instruct"
                }
                value={currentModelId}
                onChange={(e) => {
                  const value = e.target.value;
                  setCurrentModelId(value);
                  setCurrentSource(detectSource(value));
                  form.setFieldValue("model_id", value);
                }}
              />
            </Form.Item>
          </Form.Item>

          {/* Show source indicator */}
          {currentModelId && (
            <div style={{ marginBottom: 16 }}>
              {renderSourceTag(currentSource)}
            </div>
          )}

          {/* Model Preview - Show for HuggingFace models */}
          {currentModelId && currentSource === "huggingface" && (
            <ModelCompatibilityCheck
              modelId={currentModelId}
              backend="vllm"
              precision="fp16"
            />
          )}

          <Form.Item
            name="name"
            label="Display Name"
            rules={[{ required: true, message: "Please enter display name" }]}
          >
            <Input placeholder="e.g., Qwen2.5-7B" />
          </Form.Item>

          <Form.Item name="description" label="Description">
            <Input.TextArea placeholder="Optional description" rows={2} />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                Create
              </Button>
              <Button onClick={resetModalState}>Cancel</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Edit Modal */}
      <Modal
        title="Edit Model"
        open={editModalOpen}
        onCancel={() => {
          setEditModalOpen(false);
          setEditingModel(null);
          editForm.resetFields();
        }}
        footer={null}
        width={isMobile ? "100%" : 500}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        <Form form={editForm} layout="vertical" onFinish={handleUpdate}>
          {editingModel && (
            <div
              style={{
                marginBottom: 16,
                padding: 12,
                background: isDark ? "#1f1f1f" : "#f5f5f5",
                borderRadius: 8,
              }}
            >
              <div style={{ fontSize: 12, color: "#888", marginBottom: 4 }}>
                Model ID
              </div>
              <code style={{ fontSize: 13 }}>{editingModel.model_id}</code>
              <div style={{ marginTop: 8 }}>
                {renderSourceTag(editingModel.source)}
              </div>
            </div>
          )}

          <Form.Item
            name="name"
            label="Display Name"
            rules={[{ required: true, message: "Please enter display name" }]}
          >
            <Input placeholder="e.g., Qwen2.5-7B" />
          </Form.Item>

          <Form.Item name="description" label="Description">
            <Input.TextArea placeholder="Optional description" rows={2} />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                Save
              </Button>
              <Button
                onClick={() => {
                  setEditModalOpen(false);
                  setEditingModel(null);
                  editForm.resetFields();
                }}
              >
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* HuggingFace Model Picker */}
      <HuggingFaceModelPicker
        open={hfPickerOpen}
        onClose={() => setHfPickerOpen(false)}
        onSelect={handleHfModelSelect}
        backend="vllm"
      />

      {/* Ollama Model Picker */}
      <OllamaModelPicker
        open={ollamaPickerOpen}
        onClose={() => setOllamaPickerOpen(false)}
        onSelect={handleOllamaModelSelect}
      />
    </div>
  );
}
