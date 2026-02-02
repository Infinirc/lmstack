/**
 * Advanced Deployment Parameters Form
 *
 * Provides backend-specific configuration options for deployments.
 * Supports vLLM, SGLang, and Ollama backends with appropriate parameters.
 */
import {
  Collapse,
  Form,
  Input,
  InputNumber,
  Select,
  Switch,
  Tabs,
  Typography,
  Space,
} from "antd";
import { SettingOutlined } from "@ant-design/icons";
import type { FormInstance } from "antd";

const { Text } = Typography;

interface DeploymentAdvancedFormProps {
  backend: "vllm" | "sglang" | "ollama" | "mlx" | "llama_cpp";
  form: FormInstance;
}

// vLLM Parameter definitions
const VllmPerformanceParams = () => (
  <>
    <Form.Item
      name={["extra_params", "gpu-memory-utilization"]}
      label="GPU Memory Utilization"
      extra="Fraction of GPU memory to use (0.0-1.0)"
    >
      <InputNumber
        min={0.1}
        max={1.0}
        step={0.05}
        placeholder="0.9"
        style={{ width: "100%" }}
      />
    </Form.Item>
    <Form.Item
      name={["extra_params", "max-model-len"]}
      label="Max Model Length"
      extra="Maximum context length (leave empty for auto)"
    >
      <InputNumber
        min={128}
        max={131072}
        step={256}
        placeholder="Auto"
        style={{ width: "100%" }}
      />
    </Form.Item>
  </>
);

const VllmSchedulingParams = () => (
  <>
    <Form.Item
      name={["extra_params", "tensor-parallel-size"]}
      label="Tensor Parallel Size"
      extra="Number of GPUs for tensor parallelism"
    >
      <InputNumber min={1} max={8} placeholder="1" style={{ width: "100%" }} />
    </Form.Item>
    <Form.Item
      name={["extra_params", "pipeline-parallel-size"]}
      label="Pipeline Parallel Size"
      extra="Number of GPUs for pipeline parallelism"
    >
      <InputNumber min={1} max={8} placeholder="1" style={{ width: "100%" }} />
    </Form.Item>
  </>
);

const VllmAdvancedParams = () => (
  <>
    <Form.Item
      name={["extra_params", "enforce-eager"]}
      label="Enforce Eager Mode"
      valuePropName="checked"
      extra="Disable CUDA graphs for debugging"
    >
      <Switch />
    </Form.Item>
    <Form.Item
      name={["extra_params", "enable-prefix-caching"]}
      label="Enable Prefix Caching"
      valuePropName="checked"
      extra="Cache prompt prefixes for faster inference"
    >
      <Switch />
    </Form.Item>
    <Form.Item
      name={["extra_params", "enable-auto-tool-choice"]}
      label="Enable Tool Calling"
      valuePropName="checked"
      extra="Enable function/tool calling support for Agent mode"
    >
      <Switch />
    </Form.Item>
    <Form.Item
      name={["extra_params", "dtype"]}
      label="Model Dtype"
      extra="Data type for model weights"
    >
      <Select
        placeholder="auto"
        allowClear
        options={[
          { label: "Auto", value: "auto" },
          { label: "float16", value: "float16" },
          { label: "bfloat16", value: "bfloat16" },
        ]}
      />
    </Form.Item>
  </>
);

// SGLang Parameter definitions
const SglangPerformanceParams = () => (
  <>
    <Form.Item
      name={["extra_params", "mem-fraction-static"]}
      label="Memory Fraction Static"
      extra="Static memory allocation ratio (0.0-1.0)"
    >
      <InputNumber
        min={0.1}
        max={1.0}
        step={0.05}
        placeholder="Auto"
        style={{ width: "100%" }}
      />
    </Form.Item>
    <Form.Item
      name={["extra_params", "context-length"]}
      label="Context Length"
      extra="Maximum context length (leave empty for auto)"
    >
      <InputNumber
        min={128}
        max={131072}
        step={256}
        placeholder="Auto"
        style={{ width: "100%" }}
      />
    </Form.Item>
    <Form.Item
      name={["extra_params", "chunked-prefill-size"]}
      label="Chunked Prefill Size"
      extra="Prefill chunk size for better memory efficiency"
    >
      <InputNumber
        min={256}
        max={32768}
        step={256}
        placeholder="Auto"
        style={{ width: "100%" }}
      />
    </Form.Item>
  </>
);

const SglangSchedulingParams = () => (
  <>
    <Form.Item
      name={["extra_params", "tp-size"]}
      label="Tensor Parallel Size"
      extra="Number of GPUs for tensor parallelism"
    >
      <InputNumber min={1} max={8} placeholder="1" style={{ width: "100%" }} />
    </Form.Item>
    <Form.Item
      name={["extra_params", "dp-size"]}
      label="Data Parallel Size"
      extra="Number of replicas for data parallelism"
    >
      <InputNumber min={1} max={8} placeholder="1" style={{ width: "100%" }} />
    </Form.Item>
  </>
);

const SglangAdvancedParams = () => (
  <>
    <Form.Item
      name={["extra_params", "enable-mixed-chunk"]}
      label="Enable Mixed Chunk"
      valuePropName="checked"
      extra="Mixed chunk prefill for better throughput"
    >
      <Switch />
    </Form.Item>
    <Form.Item
      name={["extra_params", "attention-backend"]}
      label="Attention Backend"
      extra="Backend for attention computation"
    >
      <Select
        placeholder="auto"
        allowClear
        options={[
          { label: "Auto", value: "auto" },
          { label: "FlashInfer", value: "flashinfer" },
          { label: "Triton", value: "triton" },
        ]}
      />
    </Form.Item>
  </>
);

// Custom CLI Arguments component
const CustomArgsInput = ({ backend }: { backend: string }) => (
  <Form.Item
    name={["extra_params", "custom_args"]}
    label="Custom CLI Arguments"
    extra={
      backend === "ollama"
        ? "Additional environment variables (e.g., OLLAMA_DEBUG=1)"
        : `Additional CLI arguments (e.g., --max-model-len 4096 --trust-remote-code)`
    }
  >
    <Input.TextArea
      placeholder={
        backend === "ollama"
          ? "OLLAMA_DEBUG=1\nOLLAMA_FLASH_ATTENTION=1"
          : "--max-model-len 4096\n--trust-remote-code"
      }
      rows={3}
      style={{ fontFamily: "monospace" }}
    />
  </Form.Item>
);

// Ollama Parameter definitions
const OllamaParams = () => (
  <>
    <Form.Item
      name={["extra_params", "OLLAMA_KEEP_ALIVE"]}
      label="Keep Alive"
      extra="Duration to keep model loaded (e.g., 5m, 1h, 24h)"
    >
      <Select
        placeholder="5m"
        allowClear
        options={[
          { label: "5 minutes", value: "5m" },
          { label: "30 minutes", value: "30m" },
          { label: "1 hour", value: "1h" },
          { label: "4 hours", value: "4h" },
          { label: "24 hours", value: "24h" },
          { label: "Forever", value: "-1" },
        ]}
      />
    </Form.Item>
    <Form.Item
      name={["extra_params", "num_parallel"]}
      label="Parallel Requests"
      extra="Number of concurrent requests to handle"
    >
      <InputNumber min={1} max={16} placeholder="4" style={{ width: "100%" }} />
    </Form.Item>
    <Form.Item
      name={["extra_params", "max_loaded_models"]}
      label="Max Loaded Models"
      extra="Maximum models to keep in memory"
    >
      <InputNumber min={1} max={8} placeholder="1" style={{ width: "100%" }} />
    </Form.Item>
    <Form.Item
      name={["extra_params", "OLLAMA_MAX_QUEUE"]}
      label="Max Queue Size"
      extra="Maximum queued requests before returning 503"
    >
      <InputNumber
        min={1}
        max={1024}
        placeholder="512"
        style={{ width: "100%" }}
      />
    </Form.Item>
    <Form.Item
      name={["extra_params", "OLLAMA_KV_CACHE_TYPE"]}
      label="KV Cache Type"
      extra="Quantization for KV cache (requires Flash Attention)"
    >
      <Select
        placeholder="f16"
        allowClear
        options={[
          { label: "float16 (default)", value: "f16" },
          { label: "int8", value: "q8_0" },
          { label: "int4", value: "q4_0" },
        ]}
      />
    </Form.Item>
  </>
);

export default function DeploymentAdvancedForm({
  backend,
}: DeploymentAdvancedFormProps) {
  const renderBackendParams = () => {
    // Native Mac backends
    if (backend === "ollama") {
      return (
        <div style={{ padding: "16px 0" }}>
          <OllamaParams />
          <CustomArgsInput backend={backend} />
        </div>
      );
    }

    // MLX and llama.cpp - no advanced settings for now
    if (backend === "mlx" || backend === "llama_cpp") {
      return null;
    }

    // vLLM and SGLang (Docker-based)
    const isVllm = backend === "vllm";
    const tabItems = [
      {
        key: "performance",
        label: "Performance",
        children: isVllm ? (
          <VllmPerformanceParams />
        ) : (
          <SglangPerformanceParams />
        ),
      },
      {
        key: "scheduling",
        label: "Scheduling",
        children: isVllm ? (
          <VllmSchedulingParams />
        ) : (
          <SglangSchedulingParams />
        ),
      },
      {
        key: "advanced",
        label: "Advanced",
        children: (
          <>
            {isVllm ? <VllmAdvancedParams /> : <SglangAdvancedParams />}
            <CustomArgsInput backend={backend} />
          </>
        ),
      },
    ];

    return <Tabs items={tabItems} size="small" />;
  };

  return (
    <Collapse
      ghost
      items={[
        {
          key: "advanced",
          label: (
            <Space>
              <SettingOutlined />
              <Text>Advanced Settings</Text>
              <Text type="secondary" style={{ fontSize: 12 }}>
                ({backend.toUpperCase()} parameters)
              </Text>
            </Space>
          ),
          children: renderBackendParams(),
        },
      ]}
    />
  );
}
