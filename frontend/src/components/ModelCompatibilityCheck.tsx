/**
 * Model Compatibility Check Component
 *
 * Displays HuggingFace model info and VRAM estimation
 * for deployment compatibility checking.
 */
import { useState, useEffect, useCallback } from 'react'
import {
  Card,
  Space,
  Tag,
  Typography,
  Progress,
  Tooltip,
  Descriptions,
  Button,
} from 'antd'
import Loading from './Loading'
import {
  CheckCircleOutlined,
  WarningOutlined,
  CloseCircleOutlined,
  DownloadOutlined,
  HeartOutlined,
  InfoCircleOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import { huggingfaceApi, type HFModelInfo, type VRAMEstimate } from '../services/api'
import { useAppTheme } from '../hooks/useTheme'

interface ModelCompatibilityCheckProps {
  modelId: string // HuggingFace model ID (e.g., "Qwen/Qwen2.5-7B-Instruct")
  precision?: string // fp32, fp16, bf16, int8, int4
  gpuMemoryGb?: number // Available GPU memory for compatibility check
  contextLength?: number
  backend?: 'vllm' | 'sglang' | 'ollama'
}

const { Text, Title } = Typography

export default function ModelCompatibilityCheck({
  modelId,
  precision = 'fp16',
  gpuMemoryGb,
  contextLength = 4096,
  backend = 'vllm',
}: ModelCompatibilityCheckProps) {
  const [loading, setLoading] = useState(false)
  const [modelInfo, setModelInfo] = useState<HFModelInfo | null>(null)
  const [vramEstimate, setVramEstimate] = useState<VRAMEstimate | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { isDark, colors } = useAppTheme()

  const fetchModelData = useCallback(async () => {
    if (!modelId || backend === 'ollama') {
      // Ollama doesn't use HuggingFace models
      setModelInfo(null)
      setVramEstimate(null)
      setError(null)
      return
    }

    setLoading(true)
    setError(null)

    try {
      // Fetch model info and VRAM estimation in parallel
      const [info, estimate] = await Promise.all([
        huggingfaceApi.getModelInfo(modelId).catch(() => null),
        huggingfaceApi.estimateVRAM(modelId, {
          precision,
          context_length: contextLength,
          gpu_memory_gb: gpuMemoryGb,
        }).catch(() => null),
      ])

      setModelInfo(info)
      setVramEstimate(estimate)

      if (!info && !estimate) {
        setError('Could not fetch model information')
      }
    } catch (err) {
      setError('Failed to evaluate model compatibility')
      console.error('Model compatibility check error:', err)
    } finally {
      setLoading(false)
    }
  }, [modelId, precision, gpuMemoryGb, contextLength, backend])

  useEffect(() => {
    const debounceTimer = setTimeout(() => {
      fetchModelData()
    }, 500) // Debounce to avoid too many API calls

    return () => clearTimeout(debounceTimer)
  }, [fetchModelData])

  // Don't show anything for Ollama or if no model ID
  if (backend === 'ollama' || !modelId) {
    return null
  }

  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  const formatBytes = (bytes: number): string => {
    if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
    if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    return `${(bytes / 1024).toFixed(1)} KB`
  }

  const getCompatibilityStatus = () => {
    if (!vramEstimate) return null

    if (vramEstimate.compatible) {
      const hasWarning = vramEstimate.messages.some(m => m.includes('90%'))
      return {
        icon: hasWarning ? <WarningOutlined /> : <CheckCircleOutlined />,
        color: hasWarning ? '#faad14' : '#52c41a',
        status: hasWarning ? 'warning' : 'success',
        text: hasWarning ? 'Compatible (High Usage)' : 'Compatible',
      }
    } else {
      return {
        icon: <CloseCircleOutlined />,
        color: '#ff4d4f',
        status: 'error',
        text: 'Insufficient VRAM',
      }
    }
  }

  const compatStatus = getCompatibilityStatus()

  // Calculate VRAM usage percentage for progress bar
  const vramPercentage = gpuMemoryGb && vramEstimate
    ? Math.min(100, (vramEstimate.estimated_vram_gb / gpuMemoryGb) * 100)
    : null

  return (
    <Card
      size="small"
      style={{
        marginTop: 12,
        marginBottom: 12,
        background: isDark ? '#1a1a1a' : '#fafafa',
        border: `1px solid ${isDark ? '#303030' : '#e8e8e8'}`,
      }}
    >
      {loading ? (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px 0', gap: 8 }}>
          <Loading size="small" />
          <Text type="secondary">
            Evaluating model compatibility...
          </Text>
        </div>
      ) : error ? (
        <div style={{ padding: '12px 16px' }}>
          <Space>
            <InfoCircleOutlined style={{ color: '#888' }} />
            <div>
              <Text type="secondary" style={{ fontSize: 13 }}>
                {error}
              </Text>
              <br />
              <Text type="secondary" style={{ fontSize: 12 }}>
                Model ID: <Text code style={{ fontSize: 11 }}>{modelId}</Text>
              </Text>
              <br />
              <Button
                type="link"
                size="small"
                style={{ padding: 0, height: 'auto', fontSize: 12 }}
                href={`https://huggingface.co/${modelId}`}
                target="_blank"
              >
                View on HuggingFace →
              </Button>
            </div>
          </Space>
        </div>
      ) : (
        <div>
          {/* Model Info Header */}
          {modelInfo && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <Title level={5} style={{ margin: 0, fontSize: 14 }}>
                  {modelInfo.model_id}
                </Title>
                {modelInfo.private && <Tag color="orange">Private</Tag>}
                {modelInfo.gated && <Tag color="purple">Gated</Tag>}
              </div>

              <Space size={12} wrap style={{ marginBottom: 8 }}>
                {modelInfo.downloads > 0 && (
                  <Tooltip title="Downloads">
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      <DownloadOutlined style={{ marginRight: 4 }} />
                      {formatNumber(modelInfo.downloads)}
                    </Text>
                  </Tooltip>
                )}
                {modelInfo.likes > 0 && (
                  <Tooltip title="Likes">
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      <HeartOutlined style={{ marginRight: 4 }} />
                      {formatNumber(modelInfo.likes)}
                    </Text>
                  </Tooltip>
                )}
                {modelInfo.parameter_count && (
                  <Tag color="blue" style={{ fontSize: 11 }}>
                    {modelInfo.parameter_count} params
                  </Tag>
                )}
                {modelInfo.size_bytes && (
                  <Tag style={{ fontSize: 11 }}>
                    {formatBytes(modelInfo.size_bytes)}
                  </Tag>
                )}
              </Space>

              {/* Tags */}
              {modelInfo.tags.length > 0 && (
                <div style={{ marginTop: 4 }}>
                  {modelInfo.tags.slice(0, 5).map(tag => (
                    <Tag key={tag} style={{ fontSize: 10, marginBottom: 4 }}>
                      {tag}
                    </Tag>
                  ))}
                  {modelInfo.tags.length > 5 && (
                    <Tooltip title={modelInfo.tags.slice(5).join(', ')}>
                      <Tag style={{ fontSize: 10, marginBottom: 4 }}>
                        +{modelInfo.tags.length - 5} more
                      </Tag>
                    </Tooltip>
                  )}
                </div>
              )}
            </div>
          )}

          {/* VRAM Estimation */}
          {vramEstimate && (
            <div
              style={{
                padding: 12,
                background: isDark ? '#141414' : '#fff',
                borderRadius: 6,
                border: `1px solid ${compatStatus?.color || colors.border}`,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                <Space>
                  <ThunderboltOutlined style={{ color: '#1890ff' }} />
                  <Text strong style={{ fontSize: 13 }}>VRAM Estimation</Text>
                </Space>
                {compatStatus && (
                  <Tag
                    icon={compatStatus.icon}
                    color={compatStatus.status as 'success' | 'warning' | 'error'}
                  >
                    {compatStatus.text}
                  </Tag>
                )}
              </div>

              {/* VRAM Progress Bar */}
              {vramPercentage !== null && (
                <div style={{ marginBottom: 8 }}>
                  <Progress
                    percent={Math.round(vramPercentage)}
                    size="small"
                    status={
                      vramPercentage > 100 ? 'exception' :
                      vramPercentage > 90 ? 'active' :
                      'success'
                    }
                    format={() => `${vramEstimate.estimated_vram_gb.toFixed(1)} / ${gpuMemoryGb} GB`}
                  />
                </div>
              )}

              {/* VRAM Details */}
              <Descriptions size="small" column={2} style={{ marginTop: 8 }}>
                <Descriptions.Item label="Total VRAM">
                  <Text strong>{vramEstimate.estimated_vram_gb.toFixed(2)} GB</Text>
                </Descriptions.Item>
                <Descriptions.Item label="Precision">
                  <Tag color="cyan" style={{ fontSize: 11 }}>{vramEstimate.precision.toUpperCase()}</Tag>
                </Descriptions.Item>
                <Descriptions.Item label="Model Weights">
                  {vramEstimate.breakdown.model_weights.toFixed(2)} GB
                </Descriptions.Item>
                <Descriptions.Item label="KV Cache">
                  {vramEstimate.breakdown.kv_cache.toFixed(2)} GB
                </Descriptions.Item>
                <Descriptions.Item label="Activations">
                  {vramEstimate.breakdown.activations.toFixed(2)} GB
                </Descriptions.Item>
                <Descriptions.Item label="Overhead">
                  {vramEstimate.breakdown.overhead.toFixed(2)} GB
                </Descriptions.Item>
              </Descriptions>

              {/* Compatibility Messages */}
              {vramEstimate.messages.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  {vramEstimate.messages.map((msg, idx) => (
                    <Text
                      key={idx}
                      type={vramEstimate.compatible ? 'secondary' : 'danger'}
                      style={{ display: 'block', fontSize: 12 }}
                    >
                      {msg}
                    </Text>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </Card>
  )
}
