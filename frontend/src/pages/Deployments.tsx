import { useEffect, useState, useCallback, useRef } from 'react'
import {
  Button,
  Card,
  Form,
  Input,
  Modal,
  Select,
  Space,
  Table,
  Tag,
  message,
  Popconfirm,
  Typography,
  Tooltip,
} from 'antd'
import {
  PlusOutlined,
  DeleteOutlined,
  FileTextOutlined,
  ReloadOutlined,
  FullscreenOutlined,
  FullscreenExitOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  VerticalAlignBottomOutlined,
  ExclamationCircleOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import { useAppTheme } from '../hooks/useTheme'
import {
  VllmLogo,
  OllamaLogo,
  SGLangLogo,
  HuggingFaceLogo,
  DockerIcon,
  getBackendConfig,
} from '../components/logos'
import { getDeploymentStatusColor } from '../utils'
import { deploymentsApi, workersApi, modelsApi } from '../services/api'
import type { Deployment, DeploymentCreate, Worker, LLMModel } from '../types'
import { useResponsive } from '../hooks'
import DeploymentAdvancedForm from '../components/DeploymentAdvancedForm'
import ModelCompatibilityCheck from '../components/ModelCompatibilityCheck'
import backendVersionsData from '../constants/backendVersions.json'
import dayjs from 'dayjs'

const { Text } = Typography

const REFRESH_INTERVAL = 5000

export default function Deployments() {
  const [deployments, setDeployments] = useState<Deployment[]>([])
  const [workers, setWorkers] = useState<Worker[]>([])
  const [models, setModels] = useState<LLMModel[]>([])
  const [loading, setLoading] = useState(true)
  const [modalOpen, setModalOpen] = useState(false)
  const [logsModal, setLogsModal] = useState<{ id: number; name: string } | null>(null)
  const [logs, setLogs] = useState<string>('')
  const [logsLoading, setLogsLoading] = useState(false)
  const [logsFullscreen, setLogsFullscreen] = useState(true)
  const [autoScroll, setAutoScroll] = useState(true)
  const logsRef = useRef<HTMLPreElement>(null)
  const [form] = Form.useForm()
  const [editForm] = Form.useForm()
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null)
  const [selectedWorkerId, setSelectedWorkerId] = useState<number | null>(null)
  const [selectedGpuIndexes, setSelectedGpuIndexes] = useState<number[]>([])
  const [selectedBackend, setSelectedBackend] = useState<'vllm' | 'sglang' | 'ollama'>('vllm')
  const [editingDeployment, setEditingDeployment] = useState<Deployment | null>(null)
  const { isMobile } = useResponsive()
  const { isDark } = useAppTheme()

  // Get the selected model
  const selectedModel = models.find(m => m.id === selectedModelId)

  // Determine available backends based on model source
  const availableBackends = selectedModel?.source === 'ollama'
    ? ['ollama'] as const
    : ['vllm', 'sglang'] as const

  // Get the selected worker's GPU info
  const selectedWorker = workers.find(w => w.id === selectedWorkerId)
  const workerGpus = selectedWorker?.gpu_info || []

  // Calculate total available GPU memory for selected GPUs
  const selectedGpuMemoryGb = (() => {
    if (!selectedWorkerId || workerGpus.length === 0) return undefined
    const gpuIndexesToCheck = selectedGpuIndexes.length > 0 ? selectedGpuIndexes : [0]
    const totalMemory = gpuIndexesToCheck.reduce((sum, idx) => {
      const gpu = workerGpus.find(g => g.index === idx)
      return sum + (gpu?.memory_free || gpu?.memory_total || 0)
    }, 0)
    return totalMemory > 0 ? totalMemory / (1024 * 1024 * 1024) : undefined
  })()

  // Get the editing deployment's backend
  const editingBackend = (editingDeployment?.backend || 'vllm') as 'vllm' | 'sglang' | 'ollama'

  const BACKEND_CONFIG = getBackendConfig(isDark)

  const fetchDeployments = useCallback(async () => {
    try {
      const response = await deploymentsApi.list()
      setDeployments(response.items)
    } catch (error) {
      console.error('Failed to fetch deployments:', error)
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchWorkersAndModels = async () => {
    try {
      const [workersRes, modelsRes] = await Promise.all([
        workersApi.list({ status: 'online' }),
        modelsApi.list(),
      ])
      setWorkers(workersRes.items)
      setModels(modelsRes.items)
    } catch (error) {
      console.error('Failed to fetch workers/models:', error)
    }
  }

  useEffect(() => {
    fetchDeployments()
    fetchWorkersAndModels()

    const interval = setInterval(fetchDeployments, REFRESH_INTERVAL)
    return () => clearInterval(interval)
  }, [fetchDeployments])

  // Auto-refresh logs when modal is open
  useEffect(() => {
    if (!logsModal) return

    const interval = setInterval(async () => {
      try {
        const response = await deploymentsApi.getLogs(logsModal.id, 500)
        setLogs(response.logs)
      } catch (error) {
        // Silently fail on auto-refresh
      }
    }, 2000) // Refresh every 2 seconds

    return () => clearInterval(interval)
  }, [logsModal])

  // Auto-scroll logs to bottom
  useEffect(() => {
    if (autoScroll && logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const handleCreate = async (values: DeploymentCreate) => {
    try {
      await deploymentsApi.create(values)
      message.success('Deployment created successfully')
      setModalOpen(false)
      form.resetFields()
      fetchDeployments()
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || 'Failed to create deployment')
    }
  }

  const handleDelete = async (id: number) => {
    try {
      await deploymentsApi.delete(id)
      message.success('Deployment deleted successfully')
      fetchDeployments()
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || 'Failed to delete deployment')
    }
  }

  const handleStop = async (id: number) => {
    try {
      await deploymentsApi.stop(id)
      message.success('Deployment stopped successfully')
      fetchDeployments()
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || 'Failed to stop deployment')
    }
  }

  const handleStart = async (id: number) => {
    try {
      await deploymentsApi.start(id)
      message.success('Deployment starting...')
      fetchDeployments()
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || 'Failed to start deployment')
    }
  }

  const handleViewLogs = async (deployment: Deployment) => {
    setLogsModal({ id: deployment.id, name: deployment.name })
    setLogsLoading(true)
    try {
      const response = await deploymentsApi.getLogs(deployment.id, 500)
      setLogs(response.logs)
    } catch (error) {
      setLogs('Failed to fetch logs')
    } finally {
      setLogsLoading(false)
    }
  }

  const refreshLogs = async () => {
    if (!logsModal) return
    setLogsLoading(true)
    try {
      const response = await deploymentsApi.getLogs(logsModal.id, 500)
      setLogs(response.logs)
    } catch (error) {
      setLogs('Failed to fetch logs')
    } finally {
      setLogsLoading(false)
    }
  }

  const handleEditDeployment = (deployment: Deployment) => {
    setEditingDeployment(deployment)
    // Filter out docker_image from extra_params for the form
    const { docker_image, ...otherParams } = deployment.extra_params || {}
    editForm.setFieldsValue({
      extra_params: {
        docker_image,
        ...otherParams,
      },
    })
  }

  const handleUpdateDeployment = async (values: { extra_params?: Record<string, unknown> }) => {
    if (!editingDeployment) return
    try {
      await deploymentsApi.update(editingDeployment.id, {
        extra_params: values.extra_params,
      })
      message.success('Settings saved. Restart deployment to apply changes.')
      setEditingDeployment(null)
      editForm.resetFields()
      fetchDeployments()
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || 'Failed to update deployment')
    }
  }

  // Use shared status color utility
  const getStatusColor = getDeploymentStatusColor

  // Common tag style helper
  const getTagStyle = (size: 'small' | 'normal' = 'normal') => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    padding: '0 6px',
    fontSize: size === 'small' ? 10 : 11,
    background: isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.06)',
    border: `1px solid ${isDark ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.15)'}`,
    color: isDark ? '#ffffff' : '#000000',
  })

  // Helper to render model source tag with icon
  const renderSourceTag = (source: string | undefined, size: 'small' | 'normal' = 'normal') => {
    const iconHeight = size === 'small' ? 10 : 12
    const isOllama = source === 'ollama'
    return (
      <Tag style={getTagStyle(size)}>
        {isOllama ? <OllamaLogo height={iconHeight} isDark={isDark} /> : <HuggingFaceLogo height={iconHeight} />}
        {isOllama ? 'Ollama' : 'HuggingFace'}
      </Tag>
    )
  }

  // Mobile columns
  const mobileColumns = [
    {
      title: 'Deployment',
      key: 'deployment',
      render: (_: unknown, record: Deployment) => {
        const backend = record.backend || 'vllm'
        const config = BACKEND_CONFIG[backend] || BACKEND_CONFIG.vllm
        return (
          <div>
            <div style={{ fontWeight: 500 }}>{record.name}</div>
            <div style={{ fontSize: 12, color: '#888', display: 'flex', alignItems: 'center', gap: 4, marginTop: 2, flexWrap: 'wrap' }}>
              {record.model?.source !== 'ollama' && renderSourceTag(record.model?.source, 'small')}
              <Tag style={getTagStyle('small')}>
                {backend === 'vllm' && <VllmLogo height={10} isDark={isDark} />}
                {backend === 'sglang' && <SGLangLogo height={10} />}
                {backend === 'ollama' && <OllamaLogo height={10} isDark={isDark} />}
                <span>{config.label}</span>
              </Tag>
              {record.model?.name} @ {record.worker?.name}
            </div>
            <Space size={4} style={{ marginTop: 4 }}>
              <Tag color={getStatusColor(record.status)}>
                {record.status.toUpperCase()}
              </Tag>
              {record.container_id && (
                <Tag
                  style={{
                    fontSize: 10,
                    padding: '0 4px',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 2,
                    background: 'rgba(13, 148, 227, 0.1)',
                    border: '1px solid rgba(13, 148, 227, 0.3)',
                    color: '#0d94e3',
                  }}
                >
                  <DockerIcon size={10} />
                </Tag>
              )}
            </Space>
            {record.status_message && (
              record.status === 'error' ? (
                <Tooltip
                  title={record.status_message}
                  placement="topLeft"
                  overlayStyle={{ maxWidth: 300 }}
                >
                  <div
                    style={{
                      fontSize: 11,
                      marginTop: 2,
                      color: '#ff4d4f',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: 4,
                    }}
                  >
                    <ExclamationCircleOutlined style={{ marginTop: 1, flexShrink: 0 }} />
                    <span
                      style={{
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        wordBreak: 'break-word',
                      }}
                    >
                      {record.status_message}
                    </span>
                  </div>
                </Tooltip>
              ) : (
                <Text type="secondary" style={{ fontSize: 11, display: 'block', marginTop: 2 }}>
                  {record.status_message}
                </Text>
              )
            )}
            {record.status === 'running' && record.port && record.worker && (
              <div style={{ fontSize: 12, marginTop: 4 }}>
                Port: <a
                  href={`http://${record.worker.address.split(':')[0]}:${record.port}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {record.port}
                </a>
              </div>
            )}
          </div>
        )
      },
    },
    {
      title: '',
      key: 'actions',
      width: 60,
      render: (_: unknown, record: Deployment) => (
        <Space direction="vertical" size={4}>
          <Button
            type="text"
            size="small"
            icon={<FileTextOutlined />}
            onClick={() => handleViewLogs(record)}
          />
          <Button
            type="text"
            size="small"
            icon={<SettingOutlined />}
            onClick={() => handleEditDeployment(record)}
          />
          {(record.status === 'running' || record.status === 'starting' || record.status === 'downloading') && (
            <Popconfirm
              title="Stop?"
              onConfirm={() => handleStop(record.id)}
              okText="Stop"
              okButtonProps={{ danger: true }}
            >
              <Button type="text" size="small" icon={<PauseCircleOutlined />} />
            </Popconfirm>
          )}
          {(record.status === 'stopped' || record.status === 'error') && (
            <Button
              type="text"
              size="small"
              icon={<PlayCircleOutlined style={{ color: '#52c41a' }} />}
              onClick={() => handleStart(record.id)}
            />
          )}
          <Popconfirm
            title="Delete?"
            onConfirm={() => handleDelete(record.id)}
            okText="Delete"
            okButtonProps={{ danger: true }}
          >
            <Button type="text" size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  // Desktop columns
  const desktopColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: Deployment) => (
        <div>
          <span style={{ fontWeight: 500 }}>{name}</span>
          {record.container_id && (
            <div style={{ marginTop: 2 }}>
              <Tag
                style={{
                  fontSize: 10,
                  padding: '0 4px',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 3,
                  background: 'rgba(13, 148, 227, 0.1)',
                  border: '1px solid rgba(13, 148, 227, 0.3)',
                  color: '#0d94e3',
                }}
              >
                <DockerIcon size={10} />
                <span>Docker</span>
              </Tag>
            </div>
          )}
        </div>
      ),
    },
    {
      title: 'Model',
      key: 'model',
      render: (_: unknown, record: Deployment) => {
        const backend = record.backend || 'vllm'
        const config = BACKEND_CONFIG[backend] || BACKEND_CONFIG.vllm
        return (
          <div>
            <div style={{ fontWeight: 500 }}>{record.model?.name || '-'}</div>
            <Space size={4} style={{ marginTop: 4 }}>
              {record.model?.source !== 'ollama' && renderSourceTag(record.model?.source, 'small')}
              <Tag style={getTagStyle('small')}>
                {backend === 'vllm' && <VllmLogo height={10} isDark={isDark} />}
                {backend === 'sglang' && <SGLangLogo height={10} />}
                {backend === 'ollama' && <OllamaLogo height={10} isDark={isDark} />}
                <span>{config.label}</span>
              </Tag>
            </Space>
          </div>
        )
      },
    },
    {
      title: 'Worker',
      dataIndex: ['worker', 'name'],
      key: 'worker',
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string, record: Deployment) => (
        <div>
          <Tag color={getStatusColor(status)}>{status.toUpperCase()}</Tag>
          {record.status_message && (
            status === 'error' ? (
              <Tooltip
                title={record.status_message}
                placement="topLeft"
                overlayStyle={{ maxWidth: 500 }}
              >
                <div
                  style={{
                    fontSize: 12,
                    marginTop: 4,
                    color: '#ff4d4f',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 4,
                  }}
                >
                  <ExclamationCircleOutlined style={{ marginTop: 2, flexShrink: 0 }} />
                  <span
                    style={{
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      display: '-webkit-box',
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: 'vertical',
                      wordBreak: 'break-word',
                    }}
                  >
                    {record.status_message}
                  </span>
                </div>
              </Tooltip>
            ) : (
              <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 4 }}>
                {record.status_message}
              </Text>
            )
          )}
        </div>
      ),
    },
    {
      title: 'Port',
      dataIndex: 'port',
      key: 'port',
      render: (port: number | null, record: Deployment) =>
        record.status === 'running' && port && record.worker ? (
          <a
            href={`http://${record.worker.address.split(':')[0]}:${port}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            {port}
          </a>
        ) : (
          '-'
        ),
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => dayjs(time).format('YYYY-MM-DD HH:mm'),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: unknown, record: Deployment) => (
        <Space>
          <Button
            type="text"
            icon={<FileTextOutlined />}
            onClick={() => handleViewLogs(record)}
            title="View Logs"
          />
          <Button
            type="text"
            icon={<SettingOutlined />}
            onClick={() => handleEditDeployment(record)}
            title="Settings"
          />
          {(record.status === 'running' || record.status === 'starting' || record.status === 'downloading') && (
            <Popconfirm
              title="Stop this deployment?"
              description="The container will be stopped but deployment record kept."
              onConfirm={() => handleStop(record.id)}
              okText="Stop"
              okButtonProps={{ danger: true }}
            >
              <Button type="text" icon={<PauseCircleOutlined />} title="Stop" />
            </Popconfirm>
          )}
          {(record.status === 'stopped' || record.status === 'error') && (
            <Button
              type="text"
              icon={<PlayCircleOutlined style={{ color: '#52c41a' }} />}
              onClick={() => handleStart(record.id)}
              title="Start"
            />
          )}
          <Popconfirm
            title="Delete this deployment?"
            description="This will stop the container and delete the deployment."
            onConfirm={() => handleDelete(record.id)}
            okText="Delete"
            okButtonProps={{ danger: true }}
          >
            <Button type="text" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div>
      <Card
        style={{ borderRadius: 12 }}
        title={
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8 }}>
            <span>Deployments</span>
            <Tag color="processing" style={{ borderRadius: 6 }}>{deployments.length}</Tag>
            <Tag color="success" style={{ borderRadius: 6 }}>{deployments.filter(d => d.status === 'running').length} running</Tag>
          </div>
        }
        extra={
          <Space wrap>
            <Button
              icon={<ReloadOutlined />}
              onClick={fetchDeployments}
              size={isMobile ? 'small' : 'middle'}
            >
              {!isMobile && 'Refresh'}
            </Button>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setModalOpen(true)}
              size={isMobile ? 'small' : 'middle'}
            >
              {isMobile ? 'New' : 'New Deployment'}
            </Button>
          </Space>
        }
      >
        <Table
          dataSource={deployments}
          columns={isMobile ? mobileColumns : desktopColumns}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 10 }}
          scroll={isMobile ? undefined : { x: 900 }}
          size={isMobile ? 'small' : 'middle'}
        />
      </Card>

      <Modal
        title="New Deployment"
        open={modalOpen}
        onCancel={() => {
          setModalOpen(false)
          setSelectedModelId(null)
          setSelectedWorkerId(null)
          setSelectedGpuIndexes([])
          setSelectedBackend('vllm')
          form.resetFields()
        }}
        footer={null}
        width={isMobile ? '100%' : 600}
        style={isMobile ? { top: 20, maxWidth: '100%', margin: '0 8px' } : undefined}
      >
        <Form form={form} layout="vertical" onFinish={handleCreate}>
          <Form.Item
            name="name"
            label="Deployment Name"
            rules={[{ required: true, message: 'Please enter deployment name' }]}
          >
            <Input placeholder="e.g., qwen3-0.6b-prod" />
          </Form.Item>

          <Form.Item
            name="model_id"
            label="Model"
            rules={[{ required: true, message: 'Please select a model' }]}
          >
            <Select
              placeholder="Select a model"
              optionLabelProp="label"
              onChange={(value) => {
                setSelectedModelId(value)
                // Auto-select backend based on model source
                const model = models.find(m => m.id === value)
                if (model?.source === 'ollama') {
                  setSelectedBackend('ollama')
                  form.setFieldValue('backend', 'ollama')
                } else {
                  setSelectedBackend('vllm')
                  form.setFieldValue('backend', 'vllm')
                }
              }}
              options={models.map(m => {
                const sourceIcon = m.source === 'ollama' ? <OllamaLogo height={10} isDark={isDark} /> : <HuggingFaceLogo height={10} />
                const sourceLabel = m.source === 'ollama' ? 'Ollama' : 'HuggingFace'
                return {
                  label: (
                    <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Tag style={{ ...getTagStyle('small'), margin: 0 }}>
                        {sourceIcon}
                        {sourceLabel}
                      </Tag>
                      {m.name}
                    </span>
                  ),
                  value: m.id,
                }
              })}
            />
          </Form.Item>

          <Form.Item
            name="backend"
            label="Inference Backend"
            rules={[{ required: true, message: 'Please select a backend' }]}
            extra={selectedModel?.source === 'ollama'
              ? 'Ollama models can only use Ollama backend'
              : 'HuggingFace models can use vLLM or SGLang'
            }
          >
            <Select
              placeholder="Select a backend"
              disabled={!selectedModelId}
              value={selectedBackend}
              onChange={(value) => setSelectedBackend(value)}
              options={availableBackends.map(b => {
                const config = BACKEND_CONFIG[b]
                return {
                  label: (
                    <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      {config.icon}
                      {config.label}
                    </span>
                  ),
                  value: b,
                }
              })}
            />
          </Form.Item>

          <Form.Item
            name="worker_id"
            label="Worker"
            rules={[{ required: true, message: 'Please select a worker' }]}
          >
            <Select
              placeholder="Select a worker"
              onChange={(value) => {
                setSelectedWorkerId(value)
                // Reset GPU selection when worker changes
                setSelectedGpuIndexes([])
                form.setFieldValue('gpu_indexes', undefined)
              }}
              options={workers.map(w => ({
                label: (
                  <span style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>{w.name} ({w.address})</span>
                    {w.gpu_info && w.gpu_info.length > 0 && (
                      <Tag color="blue" style={{ marginLeft: 8 }}>{w.gpu_info.length} GPU{w.gpu_info.length > 1 ? 's' : ''}</Tag>
                    )}
                  </span>
                ),
                value: w.id,
              }))}
            />
          </Form.Item>

          <Form.Item
            name="gpu_indexes"
            label="GPU Indexes"
            extra={selectedWorkerId ? (workerGpus.length > 0 ? 'Leave empty to use GPU 0' : 'No GPUs detected on this worker') : 'Select a worker first'}
          >
            <Select
              mode="multiple"
              placeholder={selectedWorkerId ? 'Select GPUs (default: 0)' : 'Select a worker first'}
              disabled={!selectedWorkerId}
              onChange={(values: number[]) => setSelectedGpuIndexes(values)}
              options={workerGpus.length > 0
                ? workerGpus.map(gpu => ({
                    label: (
                      <span style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span>GPU {gpu.index}: {gpu.name}</span>
                        <Tag color={gpu.memory_free / gpu.memory_total > 0.5 ? 'green' : 'orange'} style={{ marginLeft: 8, fontSize: 11 }}>
                          {Math.round(gpu.memory_free / 1024 / 1024 / 1024)}GB free
                        </Tag>
                      </span>
                    ),
                    value: gpu.index,
                  }))
                : [{ label: (<span>GPU 0</span>), value: 0 }]
              }
            />
          </Form.Item>

          {/* Model Compatibility Check - Show when model is selected for vLLM/SGLang */}
          {selectedModel && selectedModel.source !== 'ollama' && (
            <ModelCompatibilityCheck
              modelId={selectedModel.model_id}
              backend={selectedBackend}
              gpuMemoryGb={selectedGpuMemoryGb}
              precision="fp16"
            />
          )}

          {/* Version Override - Show when model is selected */}
          {selectedModelId && (
            <Form.Item
              name={['extra_params', 'docker_image']}
              label={`${BACKEND_CONFIG[selectedBackend]?.label || 'Backend'} Version`}
              extra="Override the model's default backend version for this deployment"
            >
              <Select
                placeholder="Use model default"
                allowClear
                showSearch
                options={((backendVersionsData as Record<string, { versions: Array<{ version: string; image: string; recommended?: boolean }> }>)[selectedBackend]?.versions || []).map(v => ({
                  label: (
                    <span>
                      {v.version}
                      {v.recommended && <Tag color="green" style={{ marginLeft: 8, fontSize: 10 }}>Recommended</Tag>}
                    </span>
                  ),
                  value: v.image,
                }))}
              />
            </Form.Item>
          )}

          {/* Advanced Parameters - Show when model is selected */}
          {selectedModelId && (
            <DeploymentAdvancedForm backend={selectedBackend} form={form} />
          )}

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                Deploy
              </Button>
              <Button onClick={() => setModalOpen(false)}>Cancel</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title={
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8 }}>
            <span>Logs: {logsModal?.name}</span>
            <Tag color="green">Auto-refresh</Tag>
          </div>
        }
        open={!!logsModal}
        onCancel={() => {
          setLogsModal(null)
          setLogsFullscreen(false)
          setAutoScroll(true)
        }}
        footer={
          <Space wrap>
            <Button
              icon={<ReloadOutlined spin={logsLoading} />}
              onClick={refreshLogs}
              size="small"
            >
              Refresh
            </Button>
            <Button
              type={autoScroll ? 'primary' : 'default'}
              icon={<VerticalAlignBottomOutlined />}
              onClick={() => setAutoScroll(!autoScroll)}
              size="small"
            >
              Auto-scroll
            </Button>
            <Button
              icon={logsFullscreen ? <FullscreenExitOutlined /> : <FullscreenOutlined />}
              onClick={() => setLogsFullscreen(!logsFullscreen)}
              size="small"
            >
              {logsFullscreen ? 'Exit' : 'Fullscreen'}
            </Button>
          </Space>
        }
        width={logsFullscreen ? '95vw' : (isMobile ? '100%' : 1000)}
        style={logsFullscreen ? { top: 20 } : (isMobile ? { top: 20, maxWidth: '100%', margin: '0 8px' } : undefined)}
      >
        <pre
          ref={logsRef}
          style={{
            background: '#1e1e1e',
            color: '#d4d4d4',
            padding: isMobile ? 12 : 16,
            borderRadius: 4,
            height: logsFullscreen ? 'calc(100vh - 200px)' : (isMobile ? 300 : 500),
            overflow: 'auto',
            fontSize: isMobile ? 11 : 13,
            fontFamily: "'Fira Code', 'Consolas', monospace",
            lineHeight: 1.5,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all',
          }}
        >
          {logsLoading && !logs ? 'Loading...' : logs || 'No logs available'}
        </pre>
      </Modal>

      {/* Edit Deployment Settings Modal */}
      <Modal
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <SettingOutlined />
            <span>Deployment Settings: {editingDeployment?.name}</span>
          </div>
        }
        open={!!editingDeployment}
        onCancel={() => {
          setEditingDeployment(null)
          editForm.resetFields()
        }}
        footer={null}
        width={isMobile ? '100%' : 600}
        style={isMobile ? { top: 20, maxWidth: '100%', margin: '0 8px' } : undefined}
      >
        {editingDeployment && (
          <Form form={editForm} layout="vertical" onFinish={handleUpdateDeployment}>
            {/* Deployment Info */}
            <div style={{ marginBottom: 16, padding: 12, background: isDark ? '#1f1f1f' : '#f5f5f5', borderRadius: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <div>
                  <div style={{ fontSize: 12, color: '#888' }}>Model</div>
                  <div style={{ fontWeight: 500 }}>{editingDeployment.model?.name}</div>
                </div>
                <Tag color={getStatusColor(editingDeployment.status)}>{editingDeployment.status.toUpperCase()}</Tag>
              </div>
              <div style={{ fontSize: 12, color: '#888' }}>Worker: {editingDeployment.worker?.name}</div>
            </div>

            {/* Version Override */}
            <Form.Item
              name={['extra_params', 'docker_image']}
              label={`${BACKEND_CONFIG[editingBackend]?.label || 'Backend'} Version`}
              extra="Change will take effect after restart"
            >
              <Select
                placeholder="Use model default"
                allowClear
                showSearch
                options={((backendVersionsData as Record<string, { versions: Array<{ version: string; image: string; recommended?: boolean }> }>)[editingBackend]?.versions || []).map(v => ({
                  label: (
                    <span>
                      {v.version}
                      {v.recommended && <Tag color="green" style={{ marginLeft: 8, fontSize: 10 }}>Recommended</Tag>}
                    </span>
                  ),
                  value: v.image,
                }))}
              />
            </Form.Item>

            {/* Advanced Parameters */}
            <DeploymentAdvancedForm backend={editingBackend} form={editForm} />

            <Form.Item style={{ marginTop: 16 }}>
              <Space>
                <Button type="primary" htmlType="submit">
                  Save
                </Button>
                <Button onClick={() => {
                  setEditingDeployment(null)
                  editForm.resetFields()
                }}>
                  Cancel
                </Button>
              </Space>
            </Form.Item>
          </Form>
        )}
      </Modal>
    </div>
  )
}
