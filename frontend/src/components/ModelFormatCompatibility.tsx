/**
 * ModelFormatCompatibility Component
 *
 * Displays model format compatibility information and conversion warnings
 * for MLX and GGUF formats used by Mac native backends.
 */
import { useState, useEffect } from "react";
import { Tag, Space, Alert, Tooltip, Typography } from "antd";
import {
  CheckCircleOutlined,
  WarningOutlined,
  SyncOutlined,
  InfoCircleOutlined,
} from "@ant-design/icons";
import { huggingfaceApi, type ModelFormatInfo } from "../services/api";

const { Text } = Typography;

interface ModelFormatCompatibilityProps {
  modelId: string;
  backend?: "mlx" | "llama_cpp" | "vllm" | "sglang" | "ollama";
  showDetails?: boolean;
  compact?: boolean;
}

export default function ModelFormatCompatibility({
  modelId,
  backend,
  showDetails = true,
  compact = false,
}: ModelFormatCompatibilityProps) {
  const [formatInfo, setFormatInfo] = useState<ModelFormatInfo | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchFormatInfo = async () => {
      if (!modelId) return;

      setLoading(true);
      try {
        const info = await huggingfaceApi.getFormatInfo(modelId);
        setFormatInfo(info);
      } catch (error) {
        console.error("Failed to fetch format info:", error);
        setFormatInfo(null);
      } finally {
        setLoading(false);
      }
    };

    fetchFormatInfo();
  }, [modelId]);

  if (loading) {
    return (
      <Tag icon={<SyncOutlined spin />} color="processing">
        Checking format...
      </Tag>
    );
  }

  if (!formatInfo) {
    return null;
  }

  // Determine if format is compatible with selected backend
  const isCompatible = (() => {
    if (!backend) return true;
    if (backend === "mlx") return formatInfo.is_mlx_ready;
    if (backend === "llama_cpp") return formatInfo.is_gguf_ready;
    return true; // vllm, sglang, ollama don't need format conversion
  })();

  const needsConversion = backend === "mlx" || backend === "llama_cpp";

  if (compact) {
    return (
      <Space size={4}>
        {formatInfo.is_mlx_ready && (
          <Tooltip title="This model is MLX-ready (from mlx-community)">
            <Tag color="green" style={{ margin: 0 }}>
              MLX
            </Tag>
          </Tooltip>
        )}
        {formatInfo.is_gguf_ready && (
          <Tooltip title="This model has GGUF files available">
            <Tag color="blue" style={{ margin: 0 }}>
              GGUF
            </Tag>
          </Tooltip>
        )}
        {needsConversion && !isCompatible && (
          <Tooltip title="Model will be converted automatically">
            <Tag color="orange" style={{ margin: 0 }}>
              <SyncOutlined /> Convert
            </Tag>
          </Tooltip>
        )}
      </Space>
    );
  }

  // Show warning if conversion is needed
  if (needsConversion && !isCompatible) {
    const conversionTarget = backend === "mlx" ? "MLX" : "GGUF";

    return (
      <Alert
        type="warning"
        showIcon
        icon={<WarningOutlined />}
        message={`Model will be converted to ${conversionTarget} format`}
        description={
          showDetails ? (
            <div style={{ marginTop: 8 }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                This HuggingFace model is not in {conversionTarget} format. It
                will be automatically converted on the worker before deployment.
                This may take several minutes depending on model size.
              </Text>

              {backend === "mlx" && formatInfo.mlx_variants.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    Tip: Consider using an existing MLX model instead:
                  </Text>
                  <div style={{ marginTop: 4 }}>
                    {formatInfo.mlx_variants.slice(0, 3).map((variant) => (
                      <Tag
                        key={variant}
                        style={{ marginRight: 4, marginBottom: 4 }}
                      >
                        {variant}
                      </Tag>
                    ))}
                  </div>
                </div>
              )}

              {backend === "llama_cpp" &&
                formatInfo.gguf_files.length === 0 && (
                  <div style={{ marginTop: 8 }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      Tip: Search for "{modelId.split("/").pop()}-GGUF" to find
                      pre-converted models.
                    </Text>
                  </div>
                )}
            </div>
          ) : null
        }
        style={{ marginBottom: 16 }}
      />
    );
  }

  // Show success if format is ready
  if (needsConversion && isCompatible) {
    return (
      <Alert
        type="success"
        showIcon
        icon={<CheckCircleOutlined />}
        message={`Model is ${backend === "mlx" ? "MLX" : "GGUF"}-ready`}
        description={
          showDetails ? (
            <Text type="secondary" style={{ fontSize: 12 }}>
              {backend === "mlx"
                ? "This model is from mlx-community and optimized for Apple Silicon."
                : `This model has ${formatInfo.gguf_files.length} GGUF file(s) available.`}
            </Text>
          ) : null
        }
        style={{ marginBottom: 16 }}
      />
    );
  }

  // General format info display
  if (showDetails) {
    return (
      <Alert
        type="info"
        showIcon
        icon={<InfoCircleOutlined />}
        message="Model Format Information"
        description={
          <Space direction="vertical" size={4} style={{ marginTop: 8 }}>
            <div>
              <Tag color={formatInfo.is_mlx_ready ? "green" : "default"}>
                {formatInfo.is_mlx_ready ? <CheckCircleOutlined /> : null} MLX{" "}
                {formatInfo.is_mlx_ready ? "Ready" : "Needs Conversion"}
              </Tag>
              <Tag color={formatInfo.is_gguf_ready ? "green" : "default"}>
                {formatInfo.is_gguf_ready ? <CheckCircleOutlined /> : null} GGUF{" "}
                {formatInfo.is_gguf_ready ? "Ready" : "Needs Conversion"}
              </Tag>
            </div>
            {formatInfo.gguf_files.length > 0 && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                {formatInfo.gguf_files.length} GGUF file(s) available
              </Text>
            )}
            {formatInfo.mlx_variants.length > 0 && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                {formatInfo.mlx_variants.length} MLX variant(s) found
              </Text>
            )}
          </Space>
        }
        style={{ marginBottom: 16 }}
      />
    );
  }

  return null;
}
