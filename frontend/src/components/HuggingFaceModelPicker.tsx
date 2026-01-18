/**
 * HuggingFace Model Picker Component
 *
 * Search and select models from HuggingFace Hub.
 * Shows model info, VRAM estimation, and compatibility check.
 */
import { useState, useEffect, useCallback, useRef } from "react";
import {
  Drawer,
  Input,
  List,
  Tag,
  Typography,
  Space,
  Button,
  Descriptions,
  Progress,
  Empty,
  Pagination,
  Divider,
} from "antd";
import Loading from "./Loading";
import {
  SearchOutlined,
  HeartOutlined,
  DownloadOutlined,
  CheckCircleOutlined,
  CheckOutlined,
  WarningOutlined,
  CloseCircleOutlined,
  ThunderboltOutlined,
  LinkOutlined,
  FileTextOutlined,
} from "@ant-design/icons";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";
import {
  huggingfaceApi,
  type HFModelInfo,
  type VRAMEstimate,
  type HFSearchResult,
} from "../services/api";
import { useAppTheme } from "../hooks/useTheme";

interface HuggingFaceModelPickerProps {
  open: boolean;
  onClose: () => void;
  onSelect: (modelId: string, modelInfo?: HFModelInfo) => void;
  gpuMemoryGb?: number; // For compatibility check
  backend?: "vllm" | "sglang" | "ollama"; // Reserved for future use
}

const { Text, Title } = Typography;

// Strip YAML frontmatter from markdown content
const stripFrontmatter = (content: string): string => {
  if (content.startsWith("---")) {
    const endIndex = content.indexOf("---", 3);
    if (endIndex !== -1) {
      return content.slice(endIndex + 3).trim();
    }
  }
  return content;
};

// Hook for responsive detection
function useResponsive() {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return { isMobile };
}

export default function HuggingFaceModelPicker({
  open,
  onClose,
  onSelect,
  gpuMemoryGb,
  backend: _backend = "vllm",
}: HuggingFaceModelPickerProps) {
  void _backend; // Reserved for future use
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<HFSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [modelInfo, setModelInfo] = useState<HFModelInfo | null>(null);
  const [vramEstimate, setVramEstimate] = useState<VRAMEstimate | null>(null);
  const [readme, setReadme] = useState<string | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalResults, setTotalResults] = useState(0);
  const [showDetails, setShowDetails] = useState(false); // Mobile: toggle between list and details
  const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { isDark, colors } = useAppTheme();
  const { isMobile } = useResponsive();

  const pageSize = 20;

  // Load popular models
  const loadPopularModels = useCallback(async () => {
    setSearching(true);
    try {
      const results = await huggingfaceApi.getPopular(pageSize);
      setSearchResults(results);
      setTotalResults(results.length);
    } catch (error) {
      console.error("Failed to load popular models:", error);
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  }, []);

  // Search models
  const searchModels = useCallback(
    async (query: string) => {
      if (!query.trim()) {
        // Load popular models when search is cleared
        loadPopularModels();
        return;
      }

      setSearching(true);
      try {
        const results = await huggingfaceApi.search(query, {
          limit: pageSize,
          filter_task: "text-generation",
        });
        setSearchResults(results);
        // Estimate total (HF API doesn't return total count)
        setTotalResults(
          results.length >= pageSize ? pageSize * 50 : results.length,
        );
      } catch (error) {
        console.error("Search failed:", error);
        setSearchResults([]);
      } finally {
        setSearching(false);
      }
    },
    [loadPopularModels],
  );

  // Debounced search
  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    setCurrentPage(1);

    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    searchTimeoutRef.current = setTimeout(() => {
      searchModels(value);
    }, 500);
  };

  // Load model details when selected
  const handleSelectModel = useCallback(
    async (modelId: string) => {
      setSelectedModelId(modelId);
      setLoadingDetail(true);
      setModelInfo(null);
      setVramEstimate(null);
      setReadme(null);
      if (isMobile) setShowDetails(true);

      try {
        const [info, estimate, readmeResult] = await Promise.all([
          huggingfaceApi.getModelInfo(modelId).catch((err) => {
            console.error("Failed to get model info:", err);
            return null;
          }),
          huggingfaceApi
            .estimateVRAM(modelId, {
              precision: "fp16",
              gpu_memory_gb: gpuMemoryGb,
            })
            .catch((err) => {
              console.error("Failed to estimate VRAM:", err);
              return null;
            }),
          huggingfaceApi.getReadme(modelId).catch((err) => {
            console.error("Failed to get README:", err);
            return { content: null };
          }),
        ]);

        setModelInfo(info);
        setVramEstimate(estimate);

        // Process README content
        if (readmeResult?.content) {
          const processedReadme = stripFrontmatter(readmeResult.content);
          setReadme(processedReadme);
        } else {
          setReadme(null);
        }
      } catch (error) {
        console.error("Failed to load model details:", error);
      } finally {
        setLoadingDetail(false);
      }
    },
    [gpuMemoryGb, isMobile],
  );

  // Load popular models on open
  useEffect(() => {
    if (open && searchResults.length === 0 && !searchQuery) {
      loadPopularModels();
    }
  }, [open, searchResults.length, searchQuery, loadPopularModels]);

  // Reset on close
  useEffect(() => {
    if (!open) {
      setSelectedModelId(null);
      setModelInfo(null);
      setVramEstimate(null);
      setReadme(null);
      setShowDetails(false);
    }
  }, [open]);

  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const getCompatibilityStatus = () => {
    if (!vramEstimate) return null;

    if (vramEstimate.compatible) {
      const hasWarning = vramEstimate.messages.some((m) => m.includes("90%"));
      return {
        icon: hasWarning ? <WarningOutlined /> : <CheckCircleOutlined />,
        color: hasWarning ? "#faad14" : "#52c41a",
        status: hasWarning ? "warning" : "success",
        text: hasWarning ? "High VRAM Usage" : "Compatible",
      };
    } else {
      return {
        icon: <CloseCircleOutlined />,
        color: "#ff4d4f",
        status: "error",
        text: "Insufficient VRAM",
      };
    }
  };

  const compatStatus = getCompatibilityStatus();
  const vramPercentage =
    gpuMemoryGb && vramEstimate
      ? Math.min(100, (vramEstimate.estimated_vram_gb / gpuMemoryGb) * 100)
      : null;

  const handleConfirmSelect = () => {
    if (selectedModelId) {
      onSelect(selectedModelId, modelInfo || undefined);
      onClose();
    }
  };

  return (
    <Drawer
      title={
        isMobile && showDetails ? (
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Button
              type="text"
              size="small"
              onClick={() => setShowDetails(false)}
            >
              ← Back
            </Button>
            <span>Model Details</span>
          </div>
        ) : (
          "Select Model from HuggingFace"
        )
      }
      placement="right"
      width={isMobile ? "100%" : 1200}
      open={open}
      onClose={onClose}
      styles={{
        body: { padding: 0, display: "flex", height: "100%" },
      }}
      footer={
        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            alignItems: "center",
            gap: 12,
            padding: "12px 0",
            borderTop: `1px solid ${colors.border}`,
          }}
        >
          <Button size={isMobile ? "middle" : "large"} onClick={onClose}>
            Cancel
          </Button>
          <Button
            type="primary"
            size={isMobile ? "middle" : "large"}
            disabled={
              !selectedModelId || !!(vramEstimate && !vramEstimate.compatible)
            }
            onClick={handleConfirmSelect}
            icon={<CheckOutlined />}
            style={{
              minWidth: isMobile ? 120 : 160,
              fontWeight: 500,
            }}
          >
            Select
          </Button>
        </div>
      }
    >
      <div
        style={{
          display: "flex",
          width: "100%",
          height: "100%",
          flexDirection: isMobile ? "column" : "row",
        }}
      >
        {/* Left Panel - Search Results */}
        <div
          style={{
            width: isMobile ? "100%" : 400,
            borderRight: isMobile ? "none" : `1px solid ${colors.border}`,
            display: isMobile && showDetails ? "none" : "flex",
            flexDirection: "column",
            height: isMobile ? "100%" : "auto",
          }}
        >
          {/* Search Input */}
          <div style={{ padding: 16 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "8px 12px",
                borderRadius: 8,
                border: `1px solid ${colors.border}`,
                background: isDark ? "#141414" : "#fff",
              }}
            >
              <SearchOutlined
                style={{ color: colors.textMuted, fontSize: 16 }}
              />
              <Input
                placeholder="Search HuggingFace models..."
                value={searchQuery}
                onChange={(e) => handleSearchChange(e.target.value)}
                allowClear
                variant="borderless"
                style={{ padding: 0 }}
              />
            </div>
          </div>

          {/* Results List */}
          <div style={{ flex: 1, overflow: "auto" }}>
            {searching ? (
              <div style={{ textAlign: "center", padding: 40 }}>
                <Loading />
                <div style={{ marginTop: 8, color: colors.textMuted }}>
                  Searching...
                </div>
              </div>
            ) : searchResults.length === 0 ? (
              <Empty description="Search for models" style={{ padding: 40 }} />
            ) : (
              <List
                dataSource={searchResults}
                renderItem={(item) => (
                  <List.Item
                    onClick={() => handleSelectModel(item.id)}
                    style={{
                      padding: "12px 16px",
                      cursor: "pointer",
                      background:
                        selectedModelId === item.id
                          ? isDark
                            ? "rgba(255,255,255,0.08)"
                            : "rgba(0,0,0,0.04)"
                          : "transparent",
                      borderLeft:
                        selectedModelId === item.id
                          ? "3px solid #1890ff"
                          : "3px solid transparent",
                    }}
                  >
                    <div style={{ width: "100%" }}>
                      <div style={{ fontWeight: 500, marginBottom: 4 }}>
                        {item.id}
                      </div>
                      <Space size={12}>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          <HeartOutlined style={{ marginRight: 4 }} />
                          {formatNumber(item.likes)}
                        </Text>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          <DownloadOutlined style={{ marginRight: 4 }} />
                          {formatNumber(item.downloads)}
                        </Text>
                      </Space>
                      {item.tags && item.tags.length > 0 && (
                        <div style={{ marginTop: 4 }}>
                          {item.tags.slice(0, 3).map((tag) => (
                            <Tag
                              key={tag}
                              style={{ fontSize: 10, marginRight: 4 }}
                            >
                              {tag}
                            </Tag>
                          ))}
                        </div>
                      )}
                    </div>
                  </List.Item>
                )}
              />
            )}
          </div>

          {/* Pagination */}
          {totalResults > pageSize && (
            <div
              style={{ padding: 12, borderTop: `1px solid ${colors.border}` }}
            >
              <Pagination
                simple
                current={currentPage}
                total={totalResults}
                pageSize={pageSize}
                onChange={(page) => {
                  setCurrentPage(page);
                  searchModels(searchQuery);
                }}
              />
            </div>
          )}
        </div>

        {/* Right Panel - Model Details */}
        <div
          style={{
            flex: 1,
            overflow: "auto",
            display: isMobile && !showDetails ? "none" : "flex",
            flexDirection: "column",
            height: isMobile ? "100%" : "auto",
          }}
        >
          {!selectedModelId ? (
            <div
              style={{
                textAlign: "center",
                padding: isMobile ? 40 : 60,
                color: colors.textMuted,
              }}
            >
              Select a model from the list to see details
            </div>
          ) : loadingDetail ? (
            <div style={{ textAlign: "center", padding: 60 }}>
              <Loading size="large" />
              <div style={{ marginTop: 16, color: colors.textMuted }}>
                Loading model information...
              </div>
            </div>
          ) : (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                height: "100%",
              }}
            >
              {/* Model Header */}
              <div style={{ padding: 16, paddingBottom: 0 }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    marginBottom: 8,
                  }}
                >
                  <Title level={4} style={{ margin: 0 }}>
                    {modelInfo?.model_id || selectedModelId}
                  </Title>
                  <Button
                    type="link"
                    size="small"
                    icon={<LinkOutlined />}
                    href={`https://huggingface.co/${selectedModelId}`}
                    target="_blank"
                  />
                </div>

                {modelInfo && (
                  <Space size={12} wrap>
                    <Text type="secondary">
                      <DownloadOutlined style={{ marginRight: 4 }} />
                      {formatNumber(modelInfo.downloads)} downloads
                    </Text>
                    <Text type="secondary">
                      <HeartOutlined style={{ marginRight: 4 }} />
                      {formatNumber(modelInfo.likes)} likes
                    </Text>
                    {modelInfo.parameter_count && (
                      <Tag color="blue">{modelInfo.parameter_count} params</Tag>
                    )}
                    {modelInfo.pipeline_tag && (
                      <Tag color="geekblue">{modelInfo.pipeline_tag}</Tag>
                    )}
                  </Space>
                )}

                {modelInfo?.tags && modelInfo.tags.length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    {modelInfo.tags.slice(0, 8).map((tag) => (
                      <Tag key={tag} style={{ marginBottom: 4 }}>
                        {tag}
                      </Tag>
                    ))}
                    {modelInfo.tags.length > 8 && (
                      <Tag>+{modelInfo.tags.length - 8} more</Tag>
                    )}
                  </div>
                )}
              </div>

              {/* Scrollable Content */}
              <div
                style={{
                  flex: 1,
                  overflow: "auto",
                  padding: 16,
                  paddingTop: 8,
                  paddingBottom: 32,
                }}
              >
                {/* Model ID Display */}
                <div
                  style={{
                    padding: 16,
                    background: isDark ? "#1a1a1a" : "#f5f5f5",
                    borderRadius: 8,
                    marginBottom: 16,
                    border: `2px solid ${isDark ? "#3f3f46" : "#d4d4d8"}`,
                  }}
                >
                  <Text
                    type="secondary"
                    style={{ fontSize: 12, marginBottom: 4, display: "block" }}
                  >
                    Model ID:
                  </Text>
                  <Text code style={{ fontSize: 16, fontWeight: 600 }}>
                    {selectedModelId}
                  </Text>
                </div>

                {/* VRAM Estimation & Compatibility */}
                {vramEstimate && (
                  <div
                    style={{
                      padding: 16,
                      background: isDark ? "#1a1a1a" : "#fafafa",
                      borderRadius: 8,
                      border: `1px solid ${compatStatus?.color || colors.border}`,
                      marginBottom: 16,
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        marginBottom: 12,
                      }}
                    >
                      <Space>
                        <ThunderboltOutlined style={{ color: "#1890ff" }} />
                        <Text strong>VRAM Estimation</Text>
                      </Space>
                      {compatStatus && (
                        <Tag
                          icon={compatStatus.icon}
                          color={
                            compatStatus.status as
                              | "success"
                              | "warning"
                              | "error"
                          }
                        >
                          {compatStatus.text}
                        </Tag>
                      )}
                    </div>

                    {vramPercentage !== null && (
                      <div style={{ marginBottom: 12 }}>
                        <Progress
                          percent={Math.round(vramPercentage)}
                          size="small"
                          status={
                            vramPercentage > 100
                              ? "exception"
                              : vramPercentage > 90
                                ? "active"
                                : "success"
                          }
                          format={() =>
                            `${vramEstimate.estimated_vram_gb.toFixed(1)} / ${gpuMemoryGb?.toFixed(1) ?? "N/A"} GB`
                          }
                        />
                      </div>
                    )}

                    <Descriptions size="small" column={2}>
                      <Descriptions.Item label="Total VRAM">
                        <Text strong>
                          {vramEstimate.estimated_vram_gb.toFixed(2)} GB
                        </Text>
                      </Descriptions.Item>
                      <Descriptions.Item label="Precision">
                        <Tag color="cyan">
                          {vramEstimate.precision.toUpperCase()}
                        </Tag>
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

                    {vramEstimate.messages.length > 0 && (
                      <div style={{ marginTop: 8 }}>
                        {vramEstimate.messages.map((msg, idx) => (
                          <Text
                            key={idx}
                            type={
                              vramEstimate.compatible ? "secondary" : "danger"
                            }
                            style={{ display: "block", fontSize: 12 }}
                          >
                            {msg}
                          </Text>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* README Section */}
                <Divider orientation="left">
                  <Space>
                    <FileTextOutlined />
                    README
                  </Space>
                </Divider>
                {readme ? (
                  <div
                    className="markdown-body"
                    style={{
                      fontSize: 14,
                      lineHeight: 1.7,
                      color: colors.text,
                      paddingRight: 8,
                      marginBottom: 24,
                    }}
                  >
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeRaw]}
                      components={{
                        h1: ({ children }) => (
                          <h1
                            style={{
                              fontSize: "1.5em",
                              fontWeight: 600,
                              marginTop: 24,
                              marginBottom: 16,
                              borderBottom: `1px solid ${colors.border}`,
                              paddingBottom: 8,
                            }}
                          >
                            {children}
                          </h1>
                        ),
                        h2: ({ children }) => (
                          <h2
                            style={{
                              fontSize: "1.3em",
                              fontWeight: 600,
                              marginTop: 24,
                              marginBottom: 16,
                              borderBottom: `1px solid ${colors.border}`,
                              paddingBottom: 8,
                            }}
                          >
                            {children}
                          </h2>
                        ),
                        h3: ({ children }) => (
                          <h3
                            style={{
                              fontSize: "1.1em",
                              fontWeight: 600,
                              marginTop: 24,
                              marginBottom: 16,
                            }}
                          >
                            {children}
                          </h3>
                        ),
                        p: ({ children }) => (
                          <p style={{ marginTop: 0, marginBottom: 16 }}>
                            {children}
                          </p>
                        ),
                        ul: ({ children }) => (
                          <ul
                            style={{
                              paddingLeft: 24,
                              marginTop: 0,
                              marginBottom: 16,
                            }}
                          >
                            {children}
                          </ul>
                        ),
                        ol: ({ children }) => (
                          <ol
                            style={{
                              paddingLeft: 24,
                              marginTop: 0,
                              marginBottom: 16,
                            }}
                          >
                            {children}
                          </ol>
                        ),
                        li: ({ children }) => (
                          <li style={{ marginBottom: 4 }}>{children}</li>
                        ),
                        code: ({ className, children }) => {
                          const isBlock = className?.includes("language-");
                          return isBlock ? (
                            <pre
                              style={{
                                background: isDark ? "#1e1e1e" : "#f6f8fa",
                                padding: 16,
                                borderRadius: 6,
                                overflow: "auto",
                                marginBottom: 16,
                              }}
                            >
                              <code style={{ fontSize: 13 }}>{children}</code>
                            </pre>
                          ) : (
                            <code
                              style={{
                                background: isDark ? "#1e1e1e" : "#f6f8fa",
                                padding: "2px 6px",
                                borderRadius: 4,
                                fontSize: 13,
                              }}
                            >
                              {children}
                            </code>
                          );
                        },
                        pre: ({ children }) => <>{children}</>,
                        a: ({ href, children }) => (
                          <a
                            href={href}
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{ color: "#1890ff" }}
                          >
                            {children}
                          </a>
                        ),
                        img: ({ src, alt }) => (
                          <img
                            src={src}
                            alt={alt || ""}
                            style={{
                              maxWidth: "100%",
                              height: "auto",
                              borderRadius: 4,
                              marginBottom: 16,
                            }}
                          />
                        ),
                        blockquote: ({ children }) => (
                          <blockquote
                            style={{
                              margin: "0 0 16px",
                              padding: "0 16px",
                              borderLeft: `4px solid ${colors.border}`,
                              color: colors.textMuted,
                            }}
                          >
                            {children}
                          </blockquote>
                        ),
                        table: ({ children }) => (
                          <div style={{ overflowX: "auto", marginBottom: 16 }}>
                            <table
                              style={{
                                borderCollapse: "collapse",
                                width: "100%",
                                minWidth: 400,
                              }}
                            >
                              {children}
                            </table>
                          </div>
                        ),
                        thead: ({ children }) => <thead>{children}</thead>,
                        tbody: ({ children }) => <tbody>{children}</tbody>,
                        tr: ({ children }) => <tr>{children}</tr>,
                        th: ({ children }) => (
                          <th
                            style={{
                              border: `1px solid ${colors.border}`,
                              padding: "8px 12px",
                              background: isDark ? "#1e1e1e" : "#f6f8fa",
                              fontWeight: 600,
                              textAlign: "left",
                              whiteSpace: "nowrap",
                            }}
                          >
                            {children}
                          </th>
                        ),
                        td: ({ children }) => (
                          <td
                            style={{
                              border: `1px solid ${colors.border}`,
                              padding: "8px 12px",
                            }}
                          >
                            {children}
                          </td>
                        ),
                        hr: () => (
                          <hr
                            style={{
                              border: "none",
                              borderTop: `1px solid ${colors.border}`,
                              margin: "24px 0",
                            }}
                          />
                        ),
                      }}
                    >
                      {readme}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <div
                    style={{
                      padding: "16px 0",
                      textAlign: "center",
                      marginBottom: 24,
                    }}
                  >
                    <Text type="secondary">
                      README not available (This may be a gated model requiring
                      HuggingFace authentication)
                    </Text>
                    <br />
                    <Button
                      type="link"
                      icon={<LinkOutlined />}
                      href={`https://huggingface.co/${selectedModelId}`}
                      target="_blank"
                      style={{ marginTop: 8 }}
                    >
                      View on HuggingFace
                    </Button>
                  </div>
                )}

                {/* Error State */}
                {!modelInfo && !vramEstimate && !readme && (
                  <div style={{ padding: 20, textAlign: "center" }}>
                    <Text
                      type="secondary"
                      style={{ display: "block", marginBottom: 12 }}
                    >
                      Could not fetch detailed model information.
                    </Text>
                    <Text
                      type="secondary"
                      style={{ display: "block", marginBottom: 16 }}
                    >
                      You can still select this model if the ID is correct.
                    </Text>
                    <Button
                      type="link"
                      icon={<LinkOutlined />}
                      href={`https://huggingface.co/${selectedModelId}`}
                      target="_blank"
                    >
                      View on HuggingFace
                    </Button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </Drawer>
  );
}
