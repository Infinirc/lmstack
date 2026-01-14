/**
 * Headscale Management Page
 *
 * Manage Headscale VPN server for connecting remote workers.
 * Includes status display, server control, preauth key generation, and node management.
 */
import { useState, useEffect, useCallback } from "react";
import {
  Button,
  Card,
  Space,
  Table,
  Tag,
  message,
  Popconfirm,
  Typography,
  Alert,
  Row,
  Col,
  Statistic,
  Modal,
  Form,
  InputNumber,
  Input,
  Switch,
  Tooltip,
  Divider,
} from "antd";
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  DeleteOutlined,
  KeyOutlined,
  CloudServerOutlined,
  CopyOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  GlobalOutlined,
  NodeIndexOutlined,
} from "@ant-design/icons";
import {
  headscaleApi,
  type HeadscaleStatus,
  type HeadscaleNode,
  type PreauthKeyResponse,
} from "../services/api";
import { useAppTheme } from "../hooks/useTheme";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import utc from "dayjs/plugin/utc";

dayjs.extend(relativeTime);
dayjs.extend(utc);

const { Text, Paragraph } = Typography;

const REFRESH_INTERVAL = 10000; // 10 seconds

export default function Headscale() {
  const [status, setStatus] = useState<HeadscaleStatus | null>(null);
  const [nodes, setNodes] = useState<HeadscaleNode[]>([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [startModalOpen, setStartModalOpen] = useState(false);
  const [preauthKeyModal, setPreauthKeyModal] = useState(false);
  const [generatedKey, setGeneratedKey] = useState<PreauthKeyResponse | null>(
    null,
  );
  const [startForm] = Form.useForm();
  const [preauthForm] = Form.useForm();
  const { colors } = useAppTheme();

  const fetchStatus = useCallback(async () => {
    try {
      const data = await headscaleApi.getStatus();
      setStatus(data);
    } catch (error) {
      console.error("Failed to fetch Headscale status:", error);
    }
  }, []);

  const fetchNodes = useCallback(async () => {
    try {
      const data = await headscaleApi.listNodes();
      setNodes(data.items);
    } catch (error) {
      console.error("Failed to fetch nodes:", error);
    }
  }, []);

  const fetchAll = useCallback(async () => {
    await Promise.all([fetchStatus(), fetchNodes()]);
    setLoading(false);
  }, [fetchStatus, fetchNodes]);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchAll]);

  const handleStart = async (values: {
    server_url?: string;
    http_port?: number;
    grpc_port?: number;
  }) => {
    try {
      setActionLoading("start");
      await headscaleApi.start(values);
      message.success("Headscale started successfully");
      setStartModalOpen(false);
      startForm.resetFields();
      fetchAll();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to start Headscale");
    } finally {
      setActionLoading(null);
    }
  };

  const handleStop = async () => {
    try {
      setActionLoading("stop");
      await headscaleApi.stop();
      message.success("Headscale stopped");
      fetchAll();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to stop Headscale");
    } finally {
      setActionLoading(null);
    }
  };

  const handleCreatePreauthKey = async (values: {
    reusable: boolean;
    ephemeral: boolean;
    expiration: string;
  }) => {
    try {
      setActionLoading("create-key");
      const result = await headscaleApi.createPreauthKey(values);
      setGeneratedKey(result);
      message.success("Pre-auth key created");
      preauthForm.resetFields();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(
        err.response?.data?.detail || "Failed to create pre-auth key",
      );
    } finally {
      setActionLoading(null);
    }
  };

  const handleDeleteNode = async (nodeId: number) => {
    try {
      setActionLoading(`delete-${nodeId}`);
      await headscaleApi.deleteNode(nodeId);
      message.success("Node deleted");
      fetchNodes();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to delete node");
    } finally {
      setActionLoading(null);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    message.success("Copied to clipboard");
  };

  const nodeColumns = [
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
      render: (name: string, record: HeadscaleNode) => (
        <div>
          <Text strong>{record.given_name || name}</Text>
          {record.given_name && record.given_name !== name && (
            <div>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {name}
              </Text>
            </div>
          )}
        </div>
      ),
    },
    {
      title: "IP Address",
      key: "ip",
      render: (_: unknown, record: HeadscaleNode) => (
        <Space direction="vertical" size={0}>
          {record.ipv4 && <Text code>{record.ipv4}</Text>}
          {record.ip_addresses
            ?.filter((ip) => ip !== record.ipv4)
            .map((ip, idx) => (
              <Text key={idx} type="secondary" style={{ fontSize: 12 }}>
                {ip}
              </Text>
            ))}
        </Space>
      ),
    },
    {
      title: "Status",
      dataIndex: "online",
      key: "online",
      width: 100,
      render: (online: boolean) => (
        <Tag
          icon={online ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
          color={online ? "success" : "default"}
        >
          {online ? "Online" : "Offline"}
        </Tag>
      ),
    },
    {
      title: "Last Seen",
      dataIndex: "last_seen",
      key: "last_seen",
      width: 150,
      render: (time: string | null) => (time ? dayjs.utc(time).fromNow() : "-"),
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      width: 150,
      render: (time: string | null) =>
        time ? dayjs.utc(time).local().format("YYYY-MM-DD HH:mm") : "-",
    },
    {
      title: "Actions",
      key: "actions",
      width: 80,
      render: (_: unknown, record: HeadscaleNode) => (
        <Popconfirm
          title="Delete this node?"
          description="This will remove the node from the VPN network."
          onConfirm={() => handleDeleteNode(record.id)}
          okText="Delete"
          okButtonProps={{ danger: true }}
        >
          <Button
            type="text"
            danger
            size="small"
            icon={<DeleteOutlined />}
            loading={actionLoading === `delete-${record.id}`}
          />
        </Popconfirm>
      ),
    },
  ];

  const onlineCount = nodes.filter((n) => n.online).length;

  return (
    <div>
      {/* Status Card */}
      <Card
        title={
          <Space>
            <GlobalOutlined />
            <span>Headscale VPN Server</span>
          </Space>
        }
        extra={
          <Space>
            <Button
              icon={<ReloadOutlined />}
              onClick={fetchAll}
              loading={loading}
            >
              Refresh
            </Button>
            {status?.running ? (
              <Popconfirm
                title="Stop Headscale?"
                description="This will disconnect all VPN nodes."
                onConfirm={handleStop}
                okText="Stop"
                okButtonProps={{ danger: true }}
              >
                <Button
                  danger
                  icon={<PauseCircleOutlined />}
                  loading={actionLoading === "stop"}
                >
                  Stop Server
                </Button>
              </Popconfirm>
            ) : (
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={() => setStartModalOpen(true)}
              >
                Start Server
              </Button>
            )}
          </Space>
        }
        style={{ marginBottom: 16 }}
      >
        <Alert
          message="VPN Network for Remote Workers"
          description="Headscale provides a secure VPN mesh network that allows workers behind NAT or firewall to connect to LMStack. Workers join using Tailscale client with a pre-authentication key."
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />

        <Row gutter={[16, 16]}>
          <Col xs={12} sm={6}>
            <Statistic
              title="Status"
              value={status?.running ? "Running" : "Stopped"}
              valueStyle={{
                color: status?.running ? "#52c41a" : colors.textMuted,
              }}
              prefix={
                status?.running ? (
                  <CheckCircleOutlined />
                ) : (
                  <CloseCircleOutlined />
                )
              }
            />
          </Col>
          <Col xs={12} sm={6}>
            <Statistic
              title="Total Nodes"
              value={status?.nodes_count ?? 0}
              prefix={<NodeIndexOutlined />}
            />
          </Col>
          <Col xs={12} sm={6}>
            <Statistic
              title="Online Nodes"
              value={status?.online_nodes ?? 0}
              valueStyle={{
                color:
                  (status?.online_nodes ?? 0) > 0
                    ? "#52c41a"
                    : colors.textMuted,
              }}
              prefix={<CloudServerOutlined />}
            />
          </Col>
          <Col xs={12} sm={6}>
            <Statistic
              title="Server URL"
              value={status?.server_url || "-"}
              valueStyle={{ fontSize: 14 }}
            />
          </Col>
        </Row>
      </Card>

      {/* Pre-auth Key Card */}
      <Card
        title={
          <Space>
            <KeyOutlined />
            <span>Pre-authentication Key</span>
          </Space>
        }
        extra={
          <Button
            type="primary"
            icon={<KeyOutlined />}
            onClick={() => {
              setGeneratedKey(null);
              setPreauthKeyModal(true);
            }}
            disabled={!status?.running}
          >
            Generate Key
          </Button>
        }
        style={{ marginBottom: 16 }}
      >
        <Alert
          message="Worker Registration"
          description={
            <div>
              <p>
                Generate a pre-authentication key for workers to join the VPN
                network. Workers can then join using the Tailscale client:
              </p>
              <ol style={{ paddingLeft: 20, margin: "8px 0" }}>
                <li>Install Tailscale on the worker machine</li>
                <li>Run the join command with the pre-auth key</li>
                <li>The worker will appear in the nodes list below</li>
              </ol>
            </div>
          }
          type="info"
          showIcon
        />
      </Card>

      {/* Nodes Card */}
      <Card
        title={
          <Space>
            <CloudServerOutlined />
            <span>VPN Nodes</span>
            <Tag color="processing">{nodes.length}</Tag>
            <Tag color="success">{onlineCount} online</Tag>
          </Space>
        }
      >
        <Table
          dataSource={nodes}
          columns={nodeColumns}
          rowKey="id"
          loading={loading}
          pagination={false}
          size="small"
          locale={{
            emptyText: (
              <div style={{ padding: 24, color: colors.textMuted }}>
                {status?.running
                  ? "No nodes have joined yet. Generate a pre-auth key to register workers."
                  : "Start the Headscale server to manage VPN nodes."}
              </div>
            ),
          }}
        />
      </Card>

      {/* Start Server Modal */}
      <Modal
        title="Start Headscale Server"
        open={startModalOpen}
        onCancel={() => setStartModalOpen(false)}
        footer={null}
        width={500}
      >
        <Form
          form={startForm}
          layout="vertical"
          onFinish={handleStart}
          initialValues={{
            http_port: 8090,
            grpc_port: 50443,
          }}
        >
          <Alert
            message="Server URL"
            description="The server URL should be reachable by workers. If left empty, it will be auto-detected from your current connection."
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
          />
          <Form.Item
            name="server_url"
            label="Server URL"
            extra="Optional. Leave empty for auto-detection. Example: http://192.168.1.100"
          >
            <Input placeholder="Auto-detect" />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="http_port"
                label="HTTP Port"
                rules={[{ required: true, message: "Required" }]}
              >
                <InputNumber style={{ width: "100%" }} min={1} max={65535} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="grpc_port"
                label="gRPC Port"
                rules={[{ required: true, message: "Required" }]}
              >
                <InputNumber style={{ width: "100%" }} min={1} max={65535} />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item style={{ marginBottom: 0 }}>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                loading={actionLoading === "start"}
              >
                Start Server
              </Button>
              <Button onClick={() => setStartModalOpen(false)}>Cancel</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Pre-auth Key Modal */}
      <Modal
        title="Generate Pre-authentication Key"
        open={preauthKeyModal}
        onCancel={() => {
          setPreauthKeyModal(false);
          setGeneratedKey(null);
          preauthForm.resetFields();
        }}
        footer={null}
        width={600}
      >
        {!generatedKey ? (
          <Form
            form={preauthForm}
            layout="vertical"
            onFinish={handleCreatePreauthKey}
            initialValues={{
              reusable: true,
              ephemeral: false,
              expiration: "720h",
            }}
          >
            <Form.Item
              name="reusable"
              label="Reusable"
              valuePropName="checked"
              extra="Allow multiple workers to use the same key"
            >
              <Switch />
            </Form.Item>
            <Form.Item
              name="ephemeral"
              label="Ephemeral"
              valuePropName="checked"
              extra="Nodes using this key will be removed when they disconnect"
            >
              <Switch />
            </Form.Item>
            <Form.Item
              name="expiration"
              label="Expiration"
              rules={[{ required: true, message: "Required" }]}
              extra="Key validity period (e.g., 24h, 720h, 8760h)"
            >
              <Input placeholder="720h" />
            </Form.Item>
            <Form.Item style={{ marginBottom: 0 }}>
              <Space>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={actionLoading === "create-key"}
                >
                  Generate Key
                </Button>
                <Button onClick={() => setPreauthKeyModal(false)}>
                  Cancel
                </Button>
              </Space>
            </Form.Item>
          </Form>
        ) : (
          <div>
            <Alert
              message="Key Generated Successfully"
              description="Save this key now. It will not be shown again."
              type="success"
              showIcon
              style={{ marginBottom: 16 }}
            />

            <Divider orientation="left">Pre-auth Key</Divider>
            <div
              style={{
                background: colors.cardBg,
                padding: 12,
                borderRadius: 8,
                border: `1px solid ${colors.border}`,
                marginBottom: 16,
              }}
            >
              <Space style={{ width: "100%", justifyContent: "space-between" }}>
                <Text code style={{ wordBreak: "break-all" }}>
                  {generatedKey.key}
                </Text>
                <Tooltip title="Copy key">
                  <Button
                    type="text"
                    icon={<CopyOutlined />}
                    onClick={() => copyToClipboard(generatedKey.key)}
                  />
                </Tooltip>
              </Space>
            </div>

            <Divider orientation="left">Join Command</Divider>
            <Alert
              message="Run this command on the worker machine"
              description={
                <div style={{ marginTop: 8 }}>
                  <Paragraph
                    code
                    copyable={{ text: generatedKey.join_command }}
                    style={{
                      background: colors.cardBg,
                      padding: 12,
                      borderRadius: 4,
                      marginBottom: 0,
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-all",
                    }}
                  >
                    {generatedKey.join_command}
                  </Paragraph>
                </div>
              }
              type="info"
            />

            <div style={{ marginTop: 16, textAlign: "right" }}>
              <Button
                type="primary"
                onClick={() => {
                  setPreauthKeyModal(false);
                  setGeneratedKey(null);
                  preauthForm.resetFields();
                }}
              >
                Done
              </Button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
