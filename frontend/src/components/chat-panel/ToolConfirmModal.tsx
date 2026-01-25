/**
 * Tool Confirmation Modal
 *
 * A modal dialog that asks for user confirmation before executing
 * AI-requested tool actions.
 */
import { Modal, Button, Tag, Descriptions } from "antd";
import {
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  DeleteOutlined,
  RocketOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  PlusOutlined,
  StopOutlined,
  ThunderboltOutlined,
  KeyOutlined,
  CloudDownloadOutlined,
  DatabaseOutlined,
  PieChartOutlined,
  ClearOutlined,
} from "@ant-design/icons";
import type { PendingToolExecution } from "./tools";

interface ToolConfirmModalProps {
  visible: boolean;
  pendingTools: PendingToolExecution[];
  onConfirm: () => void;
  onCancel: () => void;
  isDark: boolean;
}

/**
 * Get icon for tool
 */
function getToolIcon(iconName: string) {
  const iconMap: Record<string, React.ReactNode> = {
    delete: <DeleteOutlined />,
    rocket: <RocketOutlined />,
    "pause-circle": <PauseCircleOutlined />,
    "play-circle": <PlayCircleOutlined />,
    plus: <PlusOutlined />,
    stop: <StopOutlined />,
    thunderbolt: <ThunderboltOutlined />,
    key: <KeyOutlined />,
    download: <CloudDownloadOutlined />,
    database: <DatabaseOutlined />,
    "pie-chart": <PieChartOutlined />,
    clear: <ClearOutlined />,
  };
  return iconMap[iconName] || <CheckCircleOutlined />;
}

/**
 * Format argument value for display
 */
function formatArgValue(value: any): string {
  if (value === null || value === undefined) {
    return "-";
  }
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

/**
 * Tool Confirmation Modal Component
 */
export function ToolConfirmModal({
  visible,
  pendingTools,
  onConfirm,
  onCancel,
  isDark,
}: ToolConfirmModalProps) {
  if (pendingTools.length === 0) return null;

  const hasDangerous = pendingTools.some((t) => t.meta.dangerous);

  return (
    <Modal
      open={visible}
      title={
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {hasDangerous ? (
            <ExclamationCircleOutlined
              style={{ color: "#faad14", fontSize: 20 }}
            />
          ) : (
            <CheckCircleOutlined style={{ color: "#52c41a", fontSize: 20 }} />
          )}
          <span>Confirm Action</span>
        </div>
      }
      onCancel={onCancel}
      footer={[
        <Button key="cancel" onClick={onCancel}>
          Cancel
        </Button>,
        <Button
          key="confirm"
          type="primary"
          danger={hasDangerous}
          onClick={onConfirm}
        >
          Confirm
        </Button>,
      ]}
      width={500}
      centered
      styles={{
        body: {
          background: isDark ? "#1f1f1f" : "#ffffff",
        },
        header: {
          background: isDark ? "#1f1f1f" : "#ffffff",
        },
        content: {
          background: isDark ? "#1f1f1f" : "#ffffff",
        },
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div style={{ color: isDark ? "#a1a1aa" : "#52525b", fontSize: 14 }}>
          AI assistant wants to execute the following actions:
        </div>

        {pendingTools.map((tool, index) => (
          <div
            key={index}
            style={{
              padding: 16,
              borderRadius: 8,
              background: isDark ? "#262626" : "#f5f5f5",
              border: tool.meta.dangerous
                ? "1px solid #ff4d4f"
                : "1px solid transparent",
            }}
          >
            {/* Tool header */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 12,
              }}
            >
              <span
                style={{
                  color: tool.meta.dangerous ? "#ff4d4f" : "#1890ff",
                  fontSize: 18,
                }}
              >
                {getToolIcon(tool.meta.icon)}
              </span>
              <span
                style={{
                  fontWeight: 600,
                  fontSize: 15,
                  color: isDark ? "#fafafa" : "#09090b",
                }}
              >
                {tool.meta.displayName}
              </span>
              {tool.meta.dangerous && (
                <Tag color="error" style={{ marginLeft: "auto" }}>
                  Dangerous
                </Tag>
              )}
            </div>

            {/* Tool arguments */}
            <Descriptions
              column={1}
              size="small"
              labelStyle={{
                color: isDark ? "#71717a" : "#a1a1aa",
                width: 120,
              }}
              contentStyle={{
                color: isDark ? "#fafafa" : "#09090b",
              }}
            >
              {Object.entries(tool.parsedArgs).map(([key, value]) => (
                <Descriptions.Item key={key} label={key.replace(/_/g, " ")}>
                  {formatArgValue(value)}
                </Descriptions.Item>
              ))}
            </Descriptions>
          </div>
        ))}

        {hasDangerous && (
          <div
            style={{
              padding: "8px 12px",
              borderRadius: 6,
              background: isDark
                ? "rgba(255, 77, 79, 0.1)"
                : "rgba(255, 77, 79, 0.08)",
              color: "#ff4d4f",
              fontSize: 13,
            }}
          >
            <ExclamationCircleOutlined style={{ marginRight: 8 }} />
            This action may be irreversible. Please confirm before proceeding.
          </div>
        )}
      </div>
    </Modal>
  );
}
