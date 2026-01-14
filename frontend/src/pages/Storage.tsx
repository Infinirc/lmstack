import { useEffect, useState, useCallback } from "react";
import {
  Button,
  Card,
  Space,
  Table,
  Tag,
  message,
  Popconfirm,
  Select,
  Tooltip,
  Tabs,
  Modal,
  Checkbox,
} from "antd";
import {
  DeleteOutlined,
  ReloadOutlined,
  ClearOutlined,
  HddOutlined,
  DatabaseOutlined,
  AppstoreOutlined,
  CodeSandboxOutlined,
  BuildOutlined,
  DesktopOutlined,
  ExclamationCircleOutlined,
} from "@ant-design/icons";
import {
  storageApi,
  workersApi,
  type DiskUsage,
  type Volume,
} from "../services/api";
import type { Worker } from "../types";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import utc from "dayjs/plugin/utc";

dayjs.extend(relativeTime);
dayjs.extend(utc);

const REFRESH_INTERVAL = 10000;

import { useAuth } from "../contexts/AuthContext";

function useResponsive() {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return { isMobile };
}

const formatBytes = (bytes: number) => {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
};

export default function Storage() {
  const [diskUsage, setDiskUsage] = useState<DiskUsage[]>([]);
  const [volumes, setVolumes] = useState<Volume[]>([]);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [loading, setLoading] = useState(true);
  const [volumesLoading, setVolumesLoading] = useState(true);
  const [workerFilter, setWorkerFilter] = useState<number | undefined>();
  const [pruneModalOpen, setPruneModalOpen] = useState(false);
  const [pruneWorker, setPruneWorker] = useState<DiskUsage | null>(null);
  const [pruneOptions, setPruneOptions] = useState({
    images: true,
    containers: true,
    volumes: false,
    build_cache: true,
  });
  const [pruning, setPruning] = useState(false);
  const { isMobile } = useResponsive();
  const { canEdit } = useAuth();

  const fetchData = useCallback(async () => {
    try {
      const [diskRes, workersRes] = await Promise.all([
        storageApi.getDiskUsage(workerFilter),
        workersApi.list(),
      ]);
      setDiskUsage(diskRes);
      setWorkers(workersRes.items);
    } catch (error) {
      console.error("Failed to fetch disk usage:", error);
    } finally {
      setLoading(false);
    }
  }, [workerFilter]);

  const fetchVolumes = useCallback(async () => {
    setVolumesLoading(true);
    try {
      const res = await storageApi.listVolumes(workerFilter);
      setVolumes(res);
    } catch (error) {
      console.error("Failed to fetch volumes:", error);
    } finally {
      setVolumesLoading(false);
    }
  }, [workerFilter]);

  useEffect(() => {
    fetchData();
    fetchVolumes();
    const interval = setInterval(() => {
      fetchData();
      fetchVolumes();
    }, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchData, fetchVolumes]);

  const handleDeleteVolume = async (volume: Volume) => {
    try {
      await storageApi.deleteVolume(volume.name, volume.worker_id, true);
      message.success("Volume deleted");
      fetchVolumes();
      fetchData();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to delete volume");
    }
  };

  const handlePrune = async () => {
    if (!pruneWorker) return;
    setPruning(true);
    try {
      const results = await storageApi.prune(
        pruneOptions,
        pruneWorker.worker_id,
      );
      const result = results[0];
      if (result) {
        message.success(
          `Cleanup complete: ${formatBytes(result.space_reclaimed)} reclaimed`,
        );
      }
      setPruneModalOpen(false);
      fetchData();
      fetchVolumes();
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      message.error(err.response?.data?.detail || "Failed to prune storage");
    } finally {
      setPruning(false);
    }
  };

  const diskUsageColumns = [
    {
      title: "Worker",
      key: "worker",
      render: (_: unknown, record: DiskUsage) => (
        <Space>
          <DesktopOutlined />
          <span style={{ fontWeight: 500 }}>{record.worker_name}</span>
        </Space>
      ),
    },
    {
      title: "Images",
      key: "images",
      render: (_: unknown, record: DiskUsage) => (
        <div>
          <div>
            <AppstoreOutlined style={{ marginRight: 4 }} />
            {record.images.count} images
          </div>
          <div style={{ fontSize: 12, color: "#888" }}>
            {formatBytes(record.images.size)}
          </div>
        </div>
      ),
    },
    {
      title: "Containers",
      key: "containers",
      render: (_: unknown, record: DiskUsage) => (
        <div>
          <div>
            <CodeSandboxOutlined style={{ marginRight: 4 }} />
            {record.containers.count} containers
          </div>
          <div style={{ fontSize: 12, color: "#888" }}>
            {formatBytes(record.containers.size)}
          </div>
        </div>
      ),
    },
    {
      title: "Volumes",
      key: "volumes",
      render: (_: unknown, record: DiskUsage) => (
        <div>
          <div>
            <DatabaseOutlined style={{ marginRight: 4 }} />
            {record.volumes.count} volumes
          </div>
          <div style={{ fontSize: 12, color: "#888" }}>
            {formatBytes(record.volumes.size)}
          </div>
        </div>
      ),
    },
    {
      title: "Build Cache",
      key: "build_cache",
      render: (_: unknown, record: DiskUsage) => (
        <div>
          <div>
            <BuildOutlined style={{ marginRight: 4 }} />
            {record.build_cache.count} items
          </div>
          <div style={{ fontSize: 12, color: "#888" }}>
            {formatBytes(record.build_cache.size)}
          </div>
        </div>
      ),
    },
    {
      title: "Total",
      key: "total",
      render: (_: unknown, record: DiskUsage) => (
        <div>
          <div style={{ fontWeight: 500 }}>
            {formatBytes(record.total_size)}
          </div>
          <Tag
            color={
              record.total_reclaimable > 1024 * 1024 * 1024
                ? "warning"
                : "default"
            }
          >
            {formatBytes(record.total_reclaimable)} reclaimable
          </Tag>
        </div>
      ),
    },
    {
      title: "Actions",
      key: "actions",
      width: 100,
      render: (_: unknown, record: DiskUsage) =>
        canEdit && (
          <Tooltip title="Cleanup">
            <Button
              type="text"
              danger
              icon={<ClearOutlined />}
              onClick={() => {
                setPruneWorker(record);
                setPruneModalOpen(true);
              }}
            >
              Prune
            </Button>
          </Tooltip>
        ),
    },
  ];

  const mobileDiskUsageColumns = [
    {
      title: "Worker",
      key: "worker",
      render: (_: unknown, record: DiskUsage) => (
        <div>
          <div style={{ fontWeight: 500 }}>
            <DesktopOutlined style={{ marginRight: 4 }} />
            {record.worker_name}
          </div>
          <div style={{ fontSize: 12, color: "#888", marginTop: 4 }}>
            {record.images.count} images · {record.containers.count} containers
            · {record.volumes.count} volumes
          </div>
          <div style={{ marginTop: 4 }}>
            <Tag>{formatBytes(record.total_size)}</Tag>
            <Tag color="warning">
              {formatBytes(record.total_reclaimable)} reclaimable
            </Tag>
          </div>
        </div>
      ),
    },
    {
      title: "",
      key: "actions",
      width: 80,
      render: (_: unknown, record: DiskUsage) =>
        canEdit && (
          <Button
            type="text"
            danger
            icon={<ClearOutlined />}
            size="small"
            onClick={() => {
              setPruneWorker(record);
              setPruneModalOpen(true);
            }}
          />
        ),
    },
  ];

  const volumeColumns = [
    {
      title: "Volume",
      key: "volume",
      render: (_: unknown, record: Volume) => (
        <div>
          <div style={{ fontWeight: 500 }}>{record.name}</div>
          <code style={{ fontSize: 11, color: "#888" }}>
            {record.mountpoint}
          </code>
        </div>
      ),
    },
    {
      title: "Worker",
      key: "worker",
      render: (_: unknown, record: Volume) => (
        <Space>
          <DesktopOutlined />
          {record.worker_name}
        </Space>
      ),
    },
    {
      title: "Driver",
      key: "driver",
      dataIndex: "driver",
    },
    {
      title: "Scope",
      key: "scope",
      render: (_: unknown, record: Volume) => <Tag>{record.scope}</Tag>,
    },
    {
      title: "Created",
      key: "created",
      render: (_: unknown, record: Volume) =>
        record.created_at ? dayjs.utc(record.created_at).fromNow() : "-",
    },
    {
      title: "Actions",
      key: "actions",
      width: 100,
      render: (_: unknown, record: Volume) =>
        canEdit && (
          <Popconfirm
            title="Delete volume?"
            description="This action cannot be undone."
            onConfirm={() => handleDeleteVolume(record)}
            okText="Delete"
            okButtonProps={{ danger: true }}
          >
            <Tooltip title="Delete">
              <Button type="text" danger icon={<DeleteOutlined />} />
            </Tooltip>
          </Popconfirm>
        ),
    },
  ];

  const mobileVolumeColumns = [
    {
      title: "Volume",
      key: "volume",
      render: (_: unknown, record: Volume) => (
        <div>
          <div style={{ fontWeight: 500 }}>{record.name}</div>
          <div style={{ fontSize: 11, color: "#888" }}>
            <DesktopOutlined style={{ marginRight: 4 }} />
            {record.worker_name}
          </div>
          <div style={{ marginTop: 4 }}>
            <Tag>{record.driver}</Tag>
            <Tag>{record.scope}</Tag>
          </div>
        </div>
      ),
    },
    {
      title: "",
      key: "actions",
      width: 60,
      render: (_: unknown, record: Volume) =>
        canEdit && (
          <Popconfirm
            title="Delete?"
            onConfirm={() => handleDeleteVolume(record)}
            okText="Yes"
            okButtonProps={{ danger: true }}
          >
            <Button type="text" danger icon={<DeleteOutlined />} size="small" />
          </Popconfirm>
        ),
    },
  ];

  const totalSize = diskUsage.reduce((sum, d) => sum + d.total_size, 0);
  const totalReclaimable = diskUsage.reduce(
    (sum, d) => sum + d.total_reclaimable,
    0,
  );

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
            <HddOutlined />
            <span>Docker Storage</span>
            <Tag color="processing" style={{ borderRadius: 6 }}>
              {diskUsage.length} workers
            </Tag>
            <Tag color="default" style={{ borderRadius: 6 }}>
              {formatBytes(totalSize)} total
            </Tag>
            {totalReclaimable > 0 && (
              <Tag color="warning" style={{ borderRadius: 6 }}>
                {formatBytes(totalReclaimable)} reclaimable
              </Tag>
            )}
          </div>
        }
        extra={
          <Space wrap>
            <Button
              icon={<ReloadOutlined />}
              onClick={() => {
                fetchData();
                fetchVolumes();
              }}
              size={isMobile ? "small" : "middle"}
            >
              {!isMobile && "Refresh"}
            </Button>
          </Space>
        }
      >
        <Space wrap style={{ marginBottom: 16 }}>
          <Select
            placeholder="Worker"
            allowClear
            style={{ width: isMobile ? 120 : 150 }}
            size={isMobile ? "small" : "middle"}
            onChange={(value) => setWorkerFilter(value)}
            options={workers
              .filter((w) => w.status === "online")
              .map((w) => ({ label: w.name, value: w.id }))}
          />
        </Space>

        <Tabs
          defaultActiveKey="usage"
          items={[
            {
              key: "usage",
              label: (
                <span>
                  <HddOutlined />
                  Disk Usage
                </span>
              ),
              children: (
                <Table
                  dataSource={diskUsage}
                  columns={isMobile ? mobileDiskUsageColumns : diskUsageColumns}
                  rowKey="worker_id"
                  loading={loading}
                  pagination={false}
                  size={isMobile ? "small" : "middle"}
                />
              ),
            },
            {
              key: "volumes",
              label: (
                <span>
                  <DatabaseOutlined />
                  Volumes ({volumes.length})
                </span>
              ),
              children: (
                <Table
                  dataSource={volumes}
                  columns={isMobile ? mobileVolumeColumns : volumeColumns}
                  rowKey={(r) => `${r.worker_id}-${r.name}`}
                  loading={volumesLoading}
                  pagination={{ pageSize: 10 }}
                  size={isMobile ? "small" : "middle"}
                />
              ),
            },
          ]}
        />
      </Card>

      {/* Prune Modal */}
      <Modal
        title={
          <Space>
            <ExclamationCircleOutlined style={{ color: "#faad14" }} />
            <span>Cleanup Storage: {pruneWorker?.worker_name}</span>
          </Space>
        }
        open={pruneModalOpen}
        onCancel={() => setPruneModalOpen(false)}
        onOk={handlePrune}
        okText="Cleanup"
        okButtonProps={{ danger: true, loading: pruning }}
        width={isMobile ? "100%" : 500}
        style={
          isMobile ? { top: 20, maxWidth: "100%", margin: "0 8px" } : undefined
        }
      >
        <div style={{ marginBottom: 16 }}>
          <p>Select resources to clean up:</p>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <Checkbox
              checked={pruneOptions.containers}
              onChange={(e) =>
                setPruneOptions({
                  ...pruneOptions,
                  containers: e.target.checked,
                })
              }
            >
              <Space>
                <CodeSandboxOutlined />
                Stopped containers
                {pruneWorker && (
                  <Tag>{formatBytes(pruneWorker.containers.reclaimable)}</Tag>
                )}
              </Space>
            </Checkbox>
            <Checkbox
              checked={pruneOptions.images}
              onChange={(e) =>
                setPruneOptions({ ...pruneOptions, images: e.target.checked })
              }
            >
              <Space>
                <AppstoreOutlined />
                Unused images
                {pruneWorker && (
                  <Tag>{formatBytes(pruneWorker.images.reclaimable)}</Tag>
                )}
              </Space>
            </Checkbox>
            <Checkbox
              checked={pruneOptions.build_cache}
              onChange={(e) =>
                setPruneOptions({
                  ...pruneOptions,
                  build_cache: e.target.checked,
                })
              }
            >
              <Space>
                <BuildOutlined />
                Build cache
                {pruneWorker && (
                  <Tag>{formatBytes(pruneWorker.build_cache.reclaimable)}</Tag>
                )}
              </Space>
            </Checkbox>
            <Checkbox
              checked={pruneOptions.volumes}
              onChange={(e) =>
                setPruneOptions({ ...pruneOptions, volumes: e.target.checked })
              }
            >
              <Space>
                <DatabaseOutlined />
                Unused volumes
                {pruneWorker && (
                  <Tag color="warning">
                    {formatBytes(pruneWorker.volumes.reclaimable)}
                  </Tag>
                )}
                <Tag color="error">Dangerous</Tag>
              </Space>
            </Checkbox>
          </div>
        </div>
        {pruneWorker && (
          <div style={{ background: "#f5f5f5", padding: 12, borderRadius: 8 }}>
            <div style={{ fontWeight: 500, marginBottom: 8 }}>
              Estimated space to reclaim:
            </div>
            <div style={{ fontSize: 24, fontWeight: 600, color: "#52c41a" }}>
              {formatBytes(
                (pruneOptions.containers
                  ? pruneWorker.containers.reclaimable
                  : 0) +
                  (pruneOptions.images ? pruneWorker.images.reclaimable : 0) +
                  (pruneOptions.build_cache
                    ? pruneWorker.build_cache.reclaimable
                    : 0) +
                  (pruneOptions.volumes ? pruneWorker.volumes.reclaimable : 0),
              )}
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
