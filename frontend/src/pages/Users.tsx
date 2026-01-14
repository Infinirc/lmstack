import { useEffect, useState } from 'react'
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
  Switch,
} from 'antd'
import {
  PlusOutlined,
  DeleteOutlined,
  EditOutlined,
  UserOutlined,
  ReloadOutlined,
  CrownOutlined,
} from '@ant-design/icons'
import { usersApi } from '../services/api'
import { useAuth } from '../contexts/AuthContext'
import type { User, UserCreate, UserUpdate } from '../types'
import { useResponsive } from '../hooks'
import dayjs from 'dayjs'

export default function Users() {
  const [users, setUsers] = useState<User[]>([])
  const [loading, setLoading] = useState(true)
  const [modalOpen, setModalOpen] = useState(false)
  const [editingUser, setEditingUser] = useState<User | null>(null)
  const [form] = Form.useForm()
  const { user: currentUser } = useAuth()
  const { isMobile } = useResponsive()

  const fetchUsers = async () => {
    try {
      const response = await usersApi.list()
      setUsers(response.items)
    } catch (error) {
      console.error('Failed to fetch users:', error)
      message.error('Failed to load users')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchUsers()
  }, [])

  const handleCreate = async (values: UserCreate) => {
    try {
      await usersApi.create(values)
      message.success('User created successfully')
      setModalOpen(false)
      form.resetFields()
      fetchUsers()
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || 'Failed to create user')
    }
  }

  const handleUpdate = async (values: UserUpdate) => {
    if (!editingUser) return

    try {
      await usersApi.update(editingUser.id, values)
      message.success('User updated successfully')
      setEditingUser(null)
      form.resetFields()
      fetchUsers()
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || 'Failed to update user')
    }
  }

  const handleDelete = async (id: number) => {
    try {
      await usersApi.delete(id)
      message.success('User deleted successfully')
      fetchUsers()
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || 'Failed to delete user')
    }
  }

  const handleToggleActive = async (user: User) => {
    try {
      await usersApi.update(user.id, { is_active: !user.is_active })
      message.success(`User ${user.is_active ? 'disabled' : 'enabled'} successfully`)
      fetchUsers()
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || 'Failed to update user')
    }
  }

  const openEditModal = (user: User) => {
    setEditingUser(user)
    form.setFieldsValue({
      email: user.email,
      display_name: user.display_name,
      role: user.role,
    })
  }

  const roleColors: Record<string, string> = {
    admin: 'red',
    operator: 'blue',
    viewer: 'default',
  }

  // Mobile columns
  const mobileColumns = [
    {
      title: 'User',
      key: 'user',
      render: (_: unknown, record: User) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            {record.role === 'admin' && <CrownOutlined style={{ color: '#faad14' }} />}
            <span style={{ fontWeight: 500 }}>{record.username}</span>
            {record.id === currentUser?.id && <Tag color="green" style={{ margin: 0 }}>You</Tag>}
          </div>
          <div style={{ marginBottom: 4 }}>
            <Tag color={roleColors[record.role]} style={{ margin: 0 }}>
              {record.role.toUpperCase()}
            </Tag>
            <Switch
              checked={record.is_active}
              onChange={() => handleToggleActive(record)}
              disabled={record.id === currentUser?.id}
              size="small"
              style={{ marginLeft: 8 }}
            />
          </div>
          {record.display_name && (
            <div style={{ fontSize: 12, color: '#888' }}>{record.display_name}</div>
          )}
        </div>
      ),
    },
    {
      title: '',
      key: 'actions',
      width: 80,
      render: (_: unknown, record: User) => (
        <Space direction="vertical" size={4}>
          <Button
            type="text"
            size="small"
            icon={<EditOutlined />}
            onClick={() => openEditModal(record)}
          />
          {record.id !== currentUser?.id && (
            <Popconfirm
              title="Delete this user?"
              description="This action cannot be undone."
              onConfirm={() => handleDelete(record.id)}
              okText="Delete"
              okButtonProps={{ danger: true }}
            >
              <Button type="text" size="small" danger icon={<DeleteOutlined />} />
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ]

  // Desktop columns
  const desktopColumns = [
    {
      title: 'Username',
      dataIndex: 'username',
      key: 'username',
      render: (username: string, record: User) => (
        <Space>
          {record.role === 'admin' && <CrownOutlined style={{ color: '#faad14' }} />}
          <UserOutlined />
          <span style={{ fontWeight: 500 }}>{username}</span>
          {record.id === currentUser?.id && <Tag color="green">You</Tag>}
        </Space>
      ),
    },
    {
      title: 'Display Name',
      dataIndex: 'display_name',
      key: 'display_name',
    },
    {
      title: 'Email',
      dataIndex: 'email',
      key: 'email',
      render: (email: string) => email || '-',
    },
    {
      title: 'Role',
      dataIndex: 'role',
      key: 'role',
      render: (role: string) => (
        <Tag color={roleColors[role]}>{role.toUpperCase()}</Tag>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'is_active',
      key: 'is_active',
      render: (isActive: boolean, record: User) => (
        <Switch
          checked={isActive}
          onChange={() => handleToggleActive(record)}
          disabled={record.id === currentUser?.id}
          checkedChildren="Active"
          unCheckedChildren="Disabled"
        />
      ),
    },
    {
      title: 'Last Login',
      dataIndex: 'last_login_at',
      key: 'last_login_at',
      render: (date: string) => date ? dayjs(date).format('YYYY-MM-DD HH:mm') : 'Never',
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: unknown, record: User) => (
        <Space>
          <Button
            type="text"
            icon={<EditOutlined />}
            onClick={() => openEditModal(record)}
          />
          {record.id !== currentUser?.id && (
            <Popconfirm
              title="Delete this user?"
              description="This action cannot be undone."
              onConfirm={() => handleDelete(record.id)}
              okText="Delete"
              okButtonProps={{ danger: true }}
            >
              <Button type="text" danger icon={<DeleteOutlined />} />
            </Popconfirm>
          )}
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
            <UserOutlined />
            <span>Users</span>
            <Tag color="processing" style={{ borderRadius: 6 }}>{users.length}</Tag>
          </div>
        }
        extra={
          <Space wrap>
            <Button
              icon={<ReloadOutlined />}
              onClick={fetchUsers}
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
              {isMobile ? 'Add' : 'Add User'}
            </Button>
          </Space>
        }
      >
        <Table
          dataSource={users}
          columns={isMobile ? mobileColumns : desktopColumns}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 10 }}
          size={isMobile ? 'small' : 'middle'}
        />
      </Card>

      {/* Create User Modal */}
      <Modal
        title="Create User"
        open={modalOpen}
        onCancel={() => {
          setModalOpen(false)
          form.resetFields()
        }}
        footer={null}
        width={isMobile ? '100%' : 500}
        style={isMobile ? { top: 20, maxWidth: '100%', margin: '0 8px' } : undefined}
      >
        <Form form={form} layout="vertical" onFinish={handleCreate}>
          <Form.Item
            name="username"
            label="Username"
            rules={[
              { required: true, message: 'Please enter a username' },
              { min: 3, message: 'Username must be at least 3 characters' },
            ]}
          >
            <Input placeholder="username" />
          </Form.Item>

          <Form.Item
            name="password"
            label="Password"
            rules={[
              { required: true, message: 'Please enter a password' },
              { min: 6, message: 'Password must be at least 6 characters' },
            ]}
          >
            <Input.Password placeholder="password" />
          </Form.Item>

          <Form.Item name="email" label="Email">
            <Input placeholder="user@example.com" />
          </Form.Item>

          <Form.Item name="display_name" label="Display Name">
            <Input placeholder="Display Name" />
          </Form.Item>

          <Form.Item
            name="role"
            label="Role"
            initialValue="viewer"
            rules={[{ required: true }]}
          >
            <Select>
              <Select.Option value="admin">Admin - Full access</Select.Option>
              <Select.Option value="operator">Operator - Manage deployments</Select.Option>
              <Select.Option value="viewer">Viewer - Read only</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                Create
              </Button>
              <Button onClick={() => setModalOpen(false)}>Cancel</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Edit User Modal */}
      <Modal
        title="Edit User"
        open={!!editingUser}
        onCancel={() => {
          setEditingUser(null)
          form.resetFields()
        }}
        footer={null}
        width={isMobile ? '100%' : 500}
        style={isMobile ? { top: 20, maxWidth: '100%', margin: '0 8px' } : undefined}
      >
        <Form form={form} layout="vertical" onFinish={handleUpdate}>
          <Form.Item name="email" label="Email">
            <Input placeholder="user@example.com" />
          </Form.Item>

          <Form.Item name="display_name" label="Display Name">
            <Input placeholder="Display Name" />
          </Form.Item>

          <Form.Item
            name="role"
            label="Role"
            rules={[{ required: true }]}
          >
            <Select disabled={editingUser?.id === currentUser?.id}>
              <Select.Option value="admin">Admin - Full access</Select.Option>
              <Select.Option value="operator">Operator - Manage deployments</Select.Option>
              <Select.Option value="viewer">Viewer - Read only</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                Save
              </Button>
              <Button onClick={() => setEditingUser(null)}>Cancel</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
