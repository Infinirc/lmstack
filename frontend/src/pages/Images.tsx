import { useEffect, useState, useCallback } from "react";
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
  Select,
  Tooltip,
} from "antd";
import {
  DeleteOutlined,
  EyeOutlined,
  ReloadOutlined,
  CloudDownloadOutlined,
  BuildOutlined,
  SearchOutlined,
  DesktopOutlined,
} from "@ant-design/icons";
import { imagesApi, workersApi } from "../services/api";
import type { ContainerImage, ContainerImageDetail, Worker } from "../types";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import utc from "dayjs/plugin/utc";

dayjs.extend(relativeTime);
dayjs.extend(utc);

const REFRESH_INTERVAL = 10000;

function useResponsive() {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return { isMobile };
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

export default function Images() {
  const [images, setImages] = useState<ContainerImage[]>([]);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [loading, setLoading] = useState(true);
  const [pullModalOpen, setPullModalOpen] = useState(false);
  const [buildModalOpen, setBuildModalOpen] = useState(false);
  const [detailModal, setDetailModal] = useState<ContainerImageDetail | null>(
    null,
  );
  const [detailLoading, setDetailLoading] = useState(false);
  const [workerFilter, setWorkerFilter] = useState<number | undefined>();
  const [repoFilter, setRepoFilter] = useState<string>("");
  const [pullForm] = Form.useForm();
  const [buildForm] = Form.useForm();
  const { isMobile } = useResponsive();

  const fetchData = useCallback(async () => {
    try {
      const [imagesRes, workersRes] = await Promise.all([
        imagesApi.list({
          worker_id: workerFilter,
          repository: repoFilter || undefined,
        }),
        workersApi.list(),
      ]);
      setImages(imagesRes.items);
      setWorkers(workersRes.items);
    } catch (error) {
      console.error("Failed to fetch images:", error);
    } finally {
      setLoading(false);
    }
  }, [workerFilter, repoFilter]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleViewDetail = async (image: ContainerImage) => {
    setDetailLoading(true);
    try {
      const detail = await imagesApi.get(image.id, image.worker_id);
      setDetailModal(detail);
    } catch (error) {
      message.error("Failed to fetch image details");
    } finally {
      setDetailLoading(false);
    }
  };

  const handleDelete = async (image: ContainerImage) => {
    try {
      await imagesApi.delete(image.id, image.worker_id);
      message.success("Image deleted successfully");
      fetchData();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to delete image");
    }
  };

  const handlePull = async (values: { worker_id: number; image: string }) => {
    try {
      await imagesApi.pull(values);
      message.success("Image pull started");
      setPullModalOpen(false);
      pullForm.resetFields();
      fetchData();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to pull image");
    }
  };

  const handleBuild = async (values: {
    worker_id: number;
    dockerfile: string;
    tag: string;
  }) => {
    try {
      await imagesApi.build(values);
      message.success("Image build started");
      setBuildModalOpen(false);
      buildForm.resetFields();
      fetchData();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to build image");
    }
  };

  const mobileColumns = [
    {
      title: "Image",
      key: "image",
      render: (_: unknown, record: ContainerImage) => (
        <div>
          <div style={{ fontWeight: 500 }}>{record.repository}</div>
          <Tag style={{ marginTop: 4 }}>{record.tag}</Tag>
          <div style={{ fontSize: 11, color: "#888", marginTop: 4 }}>
            <DesktopOutlined style={{ marginRight: 4 }} />
            {record.worker_name}
          </div>
          <div style={{ fontSize: 11, color: "#888" }}>
            {formatBytes(record.size)} ·{" "}
            {dayjs.utc(record.created_at).fromNow()}
          </div>
        </div>
      ),
    },
    {
      title: "",
      key: "actions",
      width: 80,
      render: (_: unknown, record: ContainerImage) => (
        <Space direction="vertical" size={4}>
          <Button
            type="text"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => handleViewDetail(record)}
            loading={detailLoading}
          />
          <Popconfirm
            title="Delete this image?"
            description="This action cannot be undone."
            onConfirm={() => handleDelete(record)}
            okText="Delete"
            okButtonProps={{ danger: true }}
          >
            <Button type="text" size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const desktopColumns = [
    {
      title: "Image",
      key: "image",
      render: (_: unknown, record: ContainerImage) => (
        <div>
          <div style={{ fontWeight: 500 }}>{record.repository}</div>
          <Tag>{record.tag}</Tag>
        </div>
      ),
    },
    {
      title: "Worker",
      key: "worker",
      render: (_: unknown, record: ContainerImage) => (
        <Space>
          <DesktopOutlined />
          {record.worker_name}
        </Space>
      ),
    },
    {
      title: "Size",
      key: "size",
      render: (_: unknown, record: ContainerImage) => formatBytes(record.size),
    },
    {
      title: "Created",
      key: "created",
      render: (_: unknown, record: ContainerImage) => (
        <Tooltip
          title={dayjs
            .utc(record.created_at)
            .local()
            .format("YYYY-MM-DD HH:mm:ss")}
        >
          {dayjs.utc(record.created_at).fromNow()}
        </Tooltip>
      ),
    },
    {
      title: "Actions",
      key: "actions",
      width: 100,
      render: (_: unknown, record: ContainerImage) => (
        <Space>
          <Tooltip title="View Details">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => handleViewDetail(record)}
              loading={detailLoading}
            />
          </Tooltip>
          <Popconfirm
            title="Delete this image?"
            description="This action cannot be undone."
            onConfirm={() => handleDelete(record)}
            okText="Delete"
            okButtonProps={{ danger: true }}
          >
            <Tooltip title="Delete">
              <Button type="text" danger icon={<DeleteOutlined />} />
            </Tooltip>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Card
        style={{ borderRadius: 12 }}
        title={
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              alignItems: "center",
              gap: 8,
            }}
          >
            <span>Images</span>
            <Tag color="processing" style={{ borderRadius: 6 }}>
              {images.length}
            </Tag>
          </div>
        }
        extra={
          <Space wrap>
            <Button
              icon={<ReloadOutlined />}
              onClick={fetchData}
              size={isMobile ? "small" : "middle"}
            >
              {!isMobile && "Refresh"}
            </Button>
            <Button
              icon={<CloudDownloadOutlined />}
              onClick={() => setPullModalOpen(true)}
              size={isMobile ? "small" : "middle"}
            >
              {isMobile ? "Pull" : "Pull Image"}
            </Button>
            <Button
              type="primary"
              icon={<BuildOutlined />}
              onClick={() => setBuildModalOpen(true)}
              size={isMobile ? "small" : "middle"}
            >
              {isMobile ? "Build" : "Build Image"}
            </Button>
          </Space>
        }
      >
        <Space wrap style={{ marginBottom: 16 }}>
          <Select
            placeholder="Filter by Worker"
            allowClear
            style={{ width: isMobile ? 130 : 180 }}
            size={isMobile ? "small" : "middle"}
            onChange={(value) => setWorkerFilter(value)}
            options={workers.map((w) => ({ label: w.name, value: w.id }))}
          />
          <Input.Search
            placeholder="Search repository"
            allowClear
            style={{ width: isMobile ? 150 : 200 }}
            size={isMobile ? "small" : "middle"}
            onSearch={(value) => setRepoFilter(value)}
            prefix={<SearchOutlined />}
          />
        </Space>

        <Table
          dataSource={images}
          columns={isMobile ? mobileColumns : desktopColumns}
          rowKey={(record) => `${record.id}-${record.worker_id}`}
          loading={loading}
          pagination={{ pageSize: 10 }}
          size={isMobile ? "small" : "middle"}
        />
      </Card>

      {/* Pull Image Modal */}
      <Modal
        title="Pull Image"
        open={pullModalOpen}
        onCancel={() => {
          setPullModalOpen(false);
          pullForm.resetFields();
        }}
        footer={null}
        width={isMobile ? "100%" : 520}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        <Form form={pullForm} layout="vertical" onFinish={handlePull}>
          <Form.Item
            name="worker_id"
            label="Worker"
            rules={[{ required: true, message: "Please select a worker" }]}
          >
            <Select
              placeholder="Select worker"
              options={workers
                .filter((w) => w.status === "online")
                .map((w) => ({ label: w.name, value: w.id }))}
            />
          </Form.Item>
          <Form.Item
            name="image"
            label="Image"
            rules={[{ required: true, message: "Please enter image name" }]}
            extra="e.g., vllm/vllm-openai:latest or nvidia/cuda:12.1-base-ubuntu22.04"
          >
            <Input placeholder="repository:tag" />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                icon={<CloudDownloadOutlined />}
              >
                Pull
              </Button>
              <Button onClick={() => setPullModalOpen(false)}>Cancel</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Build Image Modal */}
      <Modal
        title="Build Image"
        open={buildModalOpen}
        onCancel={() => {
          setBuildModalOpen(false);
          buildForm.resetFields();
        }}
        footer={null}
        width={isMobile ? "100%" : 700}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        <Form form={buildForm} layout="vertical" onFinish={handleBuild}>
          <Form.Item
            name="worker_id"
            label="Worker"
            rules={[{ required: true, message: "Please select a worker" }]}
          >
            <Select
              placeholder="Select worker"
              options={workers
                .filter((w) => w.status === "online")
                .map((w) => ({ label: w.name, value: w.id }))}
            />
          </Form.Item>
          <Form.Item
            name="tag"
            label="Image Tag"
            rules={[{ required: true, message: "Please enter image tag" }]}
            extra="e.g., my-image:v1.0"
          >
            <Input placeholder="my-image:tag" />
          </Form.Item>
          <Form.Item
            name="dockerfile"
            label="Dockerfile"
            rules={[
              { required: true, message: "Please enter Dockerfile content" },
            ]}
          >
            <Input.TextArea
              rows={12}
              placeholder={`FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD ["python", "main.py"]`}
              style={{ fontFamily: "monospace" }}
            />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" icon={<BuildOutlined />}>
                Build
              </Button>
              <Button onClick={() => setBuildModalOpen(false)}>Cancel</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Image Detail Modal */}
      <Modal
        title={
          detailModal
            ? `Image: ${detailModal.repository}:${detailModal.tag}`
            : "Image Details"
        }
        open={!!detailModal}
        onCancel={() => setDetailModal(null)}
        footer={null}
        width={isMobile ? "100%" : 700}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        {detailModal && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 16,
                  marginBottom: 12,
                }}
              >
                <div>
                  <div style={{ fontSize: 12, color: "#888" }}>ID</div>
                  <code>{detailModal.id}</code>
                </div>
                <div>
                  <div style={{ fontSize: 12, color: "#888" }}>Size</div>
                  <span>{formatBytes(detailModal.size)}</span>
                </div>
                <div>
                  <div style={{ fontSize: 12, color: "#888" }}>Worker</div>
                  <span>{detailModal.worker_name}</span>
                </div>
                <div>
                  <div style={{ fontSize: 12, color: "#888" }}>Created</div>
                  <span>
                    {dayjs(detailModal.created_at)
                      .local()
                      .format("YYYY-MM-DD HH:mm:ss")}
                  </span>
                </div>
              </div>
            </div>

            {detailModal.config && (
              <div style={{ marginBottom: 16 }}>
                <h4>Configuration</h4>
                <div
                  style={{
                    background: "#f5f5f5",
                    padding: 12,
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                >
                  {detailModal.config.cmd && (
                    <div>
                      <strong>CMD:</strong> {detailModal.config.cmd.join(" ")}
                    </div>
                  )}
                  {detailModal.config.entrypoint && (
                    <div>
                      <strong>Entrypoint:</strong>{" "}
                      {detailModal.config.entrypoint.join(" ")}
                    </div>
                  )}
                  {detailModal.config.working_dir && (
                    <div>
                      <strong>Working Dir:</strong>{" "}
                      {detailModal.config.working_dir}
                    </div>
                  )}
                  {detailModal.config.exposed_ports &&
                    detailModal.config.exposed_ports.length > 0 && (
                      <div>
                        <strong>Exposed Ports:</strong>{" "}
                        {detailModal.config.exposed_ports.join(", ")}
                      </div>
                    )}
                </div>
              </div>
            )}

            {detailModal.layers && detailModal.layers.length > 0 && (
              <div>
                <h4>Layers ({detailModal.layers.length})</h4>
                <div style={{ maxHeight: 200, overflow: "auto" }}>
                  {detailModal.layers.map((layer, idx) => (
                    <div
                      key={idx}
                      style={{
                        padding: "8px 12px",
                        borderBottom: "1px solid #f0f0f0",
                        fontSize: 12,
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        <code style={{ fontSize: 10 }}>
                          {layer.digest.substring(0, 20)}...
                        </code>
                        <span>{formatBytes(layer.size)}</span>
                      </div>
                      {layer.instruction && (
                        <div
                          style={{
                            color: "#666",
                            marginTop: 4,
                            wordBreak: "break-all",
                          }}
                        >
                          {layer.instruction}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}
