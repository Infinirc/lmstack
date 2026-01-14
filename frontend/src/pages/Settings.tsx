/**
 * Settings Page
 *
 * System management page for administrators.
 * Includes database backup/restore and statistics management.
 */
import { useState, useEffect, useCallback } from 'react'
import {
  Button,
  Card,
  Space,
  Table,
  message,
  Popconfirm,
  Typography,
  Upload,
  Divider,
  Alert,
} from 'antd'
import {
  CloudUploadOutlined,
  CloudDownloadOutlined,
  DeleteOutlined,
  ReloadOutlined,
  WarningOutlined,
  DatabaseOutlined,
  BarChartOutlined,
  UploadOutlined,
} from '@ant-design/icons'
import { systemApi, type BackupInfo } from '../services/api'
import { useAppTheme } from '../hooks/useTheme'

const { Text } = Typography

export default function Settings() {
  const [backups, setBackups] = useState<BackupInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const { colors } = useAppTheme()

  const fetchBackups = useCallback(async () => {
    try {
      setLoading(true)
      const response = await systemApi.listBackups()
      setBackups(response.items)
    } catch (error) {
      console.error('Failed to fetch backups:', error)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchBackups()
  }, [fetchBackups])

  const handleClearStats = async () => {
    try {
      setActionLoading('clear-stats')
      const result = await systemApi.clearStats()
      message.success(result.message)
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || 'Failed to clear statistics')
    } finally {
      setActionLoading(null)
    }
  }

  const handleCreateBackup = async () => {
    try {
      setActionLoading('create-backup')
      const result = await systemApi.createBackup()
      message.success(result.message)
      fetchBackups()
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || 'Failed to create backup')
    } finally {
      setActionLoading(null)
    }
  }

  const handleRestoreBackup = async (filename: string) => {
    try {
      setActionLoading(`restore-${filename}`)
      const result = await systemApi.restoreBackup(filename)
      message.success(result.message)
      message.info('Please refresh the page or restart the server')
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || 'Failed to restore backup')
    } finally {
      setActionLoading(null)
    }
  }

  const handleDeleteBackup = async (filename: string) => {
    try {
      setActionLoading(`delete-${filename}`)
      await systemApi.deleteBackup(filename)
      message.success('Backup deleted')
      fetchBackups()
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || 'Failed to delete backup')
    } finally {
      setActionLoading(null)
    }
  }

  const handleDownloadBackup = (filename: string) => {
    const url = systemApi.downloadBackup(filename)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const handleUploadRestore = async (file: File) => {
    try {
      setActionLoading('upload-restore')
      const result = await systemApi.restoreFromUpload(file)
      message.success(result.message)
      message.info('Please refresh the page or restart the server')
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || 'Failed to restore from upload')
    } finally {
      setActionLoading(null)
    }
    return false // Prevent default upload behavior
  }

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleString()
  }

  const backupColumns = [
    {
      title: 'Filename',
      dataIndex: 'filename',
      key: 'filename',
      render: (filename: string) => (
        <Text code style={{ fontSize: 13 }}>{filename}</Text>
      ),
    },
    {
      title: 'Size',
      dataIndex: 'size',
      key: 'size',
      width: 100,
      render: (size: number) => formatBytes(size),
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      render: (date: string) => formatDate(date),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 200,
      render: (_: unknown, record: BackupInfo) => (
        <Space size={4}>
          <Button
            type="text"
            size="small"
            icon={<CloudDownloadOutlined />}
            onClick={() => handleDownloadBackup(record.filename)}
          >
            Download
          </Button>
          <Popconfirm
            title="Restore this backup?"
            description={
              <div style={{ maxWidth: 300 }}>
                <p>This will replace the current database.</p>
                <p style={{ color: '#faad14' }}>
                  <WarningOutlined /> The server should be restarted after restore.
                </p>
              </div>
            }
            onConfirm={() => handleRestoreBackup(record.filename)}
            okText="Restore"
            okButtonProps={{ danger: true }}
          >
            <Button
              type="text"
              size="small"
              icon={<ReloadOutlined />}
              loading={actionLoading === `restore-${record.filename}`}
            >
              Restore
            </Button>
          </Popconfirm>
          <Popconfirm
            title="Delete this backup?"
            onConfirm={() => handleDeleteBackup(record.filename)}
            okText="Delete"
            okButtonProps={{ danger: true }}
          >
            <Button
              type="text"
              size="small"
              danger
              icon={<DeleteOutlined />}
              loading={actionLoading === `delete-${record.filename}`}
            />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div>
      {/* Statistics Management */}
      <Card
        title={
          <Space>
            <BarChartOutlined />
            <span>Statistics</span>
          </Space>
        }
        style={{ marginBottom: 16 }}
      >
        <Alert
          message="Clear Statistics"
          description="This will permanently delete all API usage statistics data. This action cannot be undone."
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
        <Popconfirm
          title="Clear all statistics?"
          description="This will delete all usage records. This action cannot be undone."
          onConfirm={handleClearStats}
          okText="Clear"
          okButtonProps={{ danger: true }}
        >
          <Button
            danger
            icon={<DeleteOutlined />}
            loading={actionLoading === 'clear-stats'}
          >
            Clear Statistics
          </Button>
        </Popconfirm>
      </Card>

      {/* Database Backup & Restore */}
      <Card
        title={
          <Space>
            <DatabaseOutlined />
            <span>Database Backup & Restore</span>
          </Space>
        }
        extra={
          <Space>
            <Upload
              accept=".db"
              showUploadList={false}
              beforeUpload={handleUploadRestore}
            >
              <Button
                icon={<UploadOutlined />}
                loading={actionLoading === 'upload-restore'}
              >
                Upload & Restore
              </Button>
            </Upload>
            <Button
              type="primary"
              icon={<CloudUploadOutlined />}
              onClick={handleCreateBackup}
              loading={actionLoading === 'create-backup'}
            >
              Create Backup
            </Button>
          </Space>
        }
      >
        <Alert
          message="Database Backup"
          description="Create backups of your database before making major changes. You can restore from any backup or upload a backup file."
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />

        <Divider orientation="left">Available Backups</Divider>

        <Table
          dataSource={backups}
          columns={backupColumns}
          rowKey="filename"
          loading={loading}
          pagination={false}
          size="small"
          locale={{
            emptyText: (
              <div style={{ padding: 24, color: colors.textMuted }}>
                No backups yet. Create your first backup above.
              </div>
            ),
          }}
        />
      </Card>
    </div>
  )
}
