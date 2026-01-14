/**
 * Ollama Model Picker Component
 *
 * Browse and select models from Ollama library.
 * Shows model info, available tags/sizes, and capabilities.
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
  Empty,
  Select,
  Divider,
} from "antd";
import Loading from "./Loading";
import {
  SearchOutlined,
  DownloadOutlined,
  LinkOutlined,
  EyeOutlined,
  CodeOutlined,
  RobotOutlined,
  BulbOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  CheckOutlined,
} from "@ant-design/icons";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import {
  ollamaApi,
  type OllamaModel,
  type OllamaTagInfo,
} from "../services/api";
import { useAppTheme } from "../hooks/useTheme";

interface OllamaModelPickerProps {
  open: boolean;
  onClose: () => void;
  onSelect: (modelName: string, tag?: string) => void;
}

const { Text, Title } = Typography;

// Capability icons
const CAPABILITY_ICONS: Record<string, React.ReactNode> = {
  vision: <EyeOutlined />,
  code: <CodeOutlined />,
  tools: <RobotOutlined />,
  thinking: <BulbOutlined />,
  embedding: <DatabaseOutlined />,
};

// Capability colors
const CAPABILITY_COLORS: Record<string, string> = {
  vision: "purple",
  code: "blue",
  tools: "green",
  thinking: "orange",
  embedding: "cyan",
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

export default function OllamaModelPicker({
  open,
  onClose,
  onSelect,
}: OllamaModelPickerProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [searching, setSearching] = useState(false);
  const [selectedModel, setSelectedModel] = useState<OllamaModel | null>(null);
  const [tags, setTags] = useState<OllamaTagInfo[]>([]);
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const [loadingTags, setLoadingTags] = useState(false);
  const [loadingReadme, setLoadingReadme] = useState(false);
  const [capabilityFilter, setCapabilityFilter] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false); // Mobile: toggle between list and details
  const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { isDark, colors } = useAppTheme();
  const { isMobile } = useResponsive();

  // Load popular models
  const loadPopularModels = useCallback(async () => {
    setSearching(true);
    try {
      const results = await ollamaApi.getPopular(30);
      setModels(results);
    } catch (error) {
      console.error("Failed to load popular models:", error);
      setModels([]);
    } finally {
      setSearching(false);
    }
  }, []);

  // Search models
  const searchModels = useCallback(
    async (query: string, capability?: string | null) => {
      if (!query.trim() && !capability) {
        loadPopularModels();
        return;
      }

      setSearching(true);
      try {
        const results = await ollamaApi.listModels({
          search: query || undefined,
          capability: capability || undefined,
          limit: 50,
        });
        setModels(results);
      } catch (error) {
        console.error("Search failed:", error);
        setModels([]);
      } finally {
        setSearching(false);
      }
    },
    [loadPopularModels],
  );

  // Debounced search
  const handleSearchChange = (value: string) => {
    setSearchQuery(value);

    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    searchTimeoutRef.current = setTimeout(() => {
      searchModels(value, capabilityFilter);
    }, 300);
  };

  // Handle capability filter change
  const handleCapabilityChange = (value: string | null) => {
    setCapabilityFilter(value);
    searchModels(searchQuery, value);
  };

  // Load model tags and README when selected
  const handleSelectModel = useCallback(
    async (model: OllamaModel) => {
      setSelectedModel(model);
      setSelectedTag(null);
      setLoadingTags(true);
      setLoadingReadme(true);
      setTags([]);
      if (isMobile) setShowDetails(true);

      // Fetch tags and model info (with README) in parallel
      const tagsPromise = ollamaApi.getModelTags(model.name);
      const modelInfoPromise = ollamaApi.getModelInfo(model.name);

      try {
        const result = await tagsPromise;
        setTags(result.tags);
        // Auto-select default tag if available
        if (result.tags.length > 0) {
          const latestTag =
            result.tags.find((t) => t.name === "latest") || result.tags[0];
          setSelectedTag(latestTag.name);
        }
      } catch (error) {
        console.error("Failed to load tags:", error);
        // Use sizes from model info as fallback
        const fallbackTags: OllamaTagInfo[] = model.sizes.map((size) => ({
          name: size,
          full_name: `${model.name}:${size}`,
          size,
        }));
        if (fallbackTags.length === 0) {
          fallbackTags.push({
            name: "latest",
            full_name: `${model.name}:latest`,
          });
        }
        setTags(fallbackTags);
        setSelectedTag(fallbackTags[0]?.name || "latest");
      } finally {
        setLoadingTags(false);
      }

      try {
        const modelInfo = await modelInfoPromise;
        // Update selectedModel with README from fetched info
        setSelectedModel((prev) =>
          prev ? { ...prev, readme: modelInfo.readme } : prev,
        );
      } catch (error) {
        console.error("Failed to load model info:", error);
      } finally {
        setLoadingReadme(false);
      }
    },
    [isMobile],
  );

  // Load popular models on open
  useEffect(() => {
    if (open && models.length === 0) {
      loadPopularModels();
    }
  }, [open, models.length, loadPopularModels]);

  // Reset on close
  useEffect(() => {
    if (!open) {
      setSelectedModel(null);
      setTags([]);
      setSelectedTag(null);
      setShowDetails(false);
    }
  }, [open]);

  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const handleConfirmSelect = () => {
    if (selectedModel) {
      const fullName = selectedTag
        ? `${selectedModel.name}:${selectedTag}`
        : selectedModel.name;
      onSelect(fullName, selectedTag || undefined);
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
          "Select Model from Ollama Library"
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
            disabled={!selectedModel}
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
                marginBottom: 12,
              }}
            >
              <SearchOutlined
                style={{ color: colors.textMuted, fontSize: 16 }}
              />
              <Input
                placeholder="Search Ollama models..."
                value={searchQuery}
                onChange={(e) => handleSearchChange(e.target.value)}
                allowClear
                variant="borderless"
                style={{ padding: 0 }}
              />
            </div>
            <Select
              placeholder="Filter by capability"
              allowClear
              style={{ width: "100%" }}
              value={capabilityFilter}
              onChange={handleCapabilityChange}
              options={[
                { label: "Vision", value: "vision" },
                { label: "Tools", value: "tools" },
                { label: "Code", value: "code" },
                { label: "Thinking", value: "thinking" },
                { label: "Embedding", value: "embedding" },
              ]}
            />
          </div>

          {/* Results List */}
          <div style={{ flex: 1, overflow: "auto" }}>
            {searching ? (
              <div style={{ textAlign: "center", padding: 40 }}>
                <Loading />
                <div style={{ marginTop: 8, color: colors.textMuted }}>
                  Loading models...
                </div>
              </div>
            ) : models.length === 0 ? (
              <Empty description="No models found" style={{ padding: 40 }} />
            ) : (
              <List
                dataSource={models}
                renderItem={(item) => (
                  <List.Item
                    onClick={() => handleSelectModel(item)}
                    style={{
                      padding: "12px 16px",
                      cursor: "pointer",
                      background:
                        selectedModel?.name === item.name
                          ? isDark
                            ? "rgba(255,255,255,0.08)"
                            : "rgba(0,0,0,0.04)"
                          : "transparent",
                      borderLeft:
                        selectedModel?.name === item.name
                          ? "3px solid #1890ff"
                          : "3px solid transparent",
                    }}
                  >
                    <div style={{ width: "100%" }}>
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 8,
                          marginBottom: 4,
                        }}
                      >
                        <span style={{ fontWeight: 500 }}>{item.name}</span>
                        {item.capabilities.map((cap) => (
                          <Tag
                            key={cap}
                            color={CAPABILITY_COLORS[cap] || "default"}
                            style={{
                              fontSize: 10,
                              padding: "0 4px",
                              lineHeight: "16px",
                            }}
                          >
                            {CAPABILITY_ICONS[cap]} {cap}
                          </Tag>
                        ))}
                      </div>
                      {item.description && (
                        <Text
                          type="secondary"
                          style={{
                            fontSize: 12,
                            display: "block",
                            marginBottom: 4,
                          }}
                          ellipsis={{ tooltip: item.description }}
                        >
                          {item.description}
                        </Text>
                      )}
                      <Space size={12}>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          <DownloadOutlined style={{ marginRight: 4 }} />
                          {formatNumber(item.pulls)} pulls
                        </Text>
                        {item.sizes.length > 0 && (
                          <Text type="secondary" style={{ fontSize: 11 }}>
                            {item.sizes.slice(0, 4).join(", ")}
                            {item.sizes.length > 4 && "..."}
                          </Text>
                        )}
                      </Space>
                    </div>
                  </List.Item>
                )}
              />
            )}
          </div>
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
          {!selectedModel ? (
            <div
              style={{
                textAlign: "center",
                padding: 60,
                color: colors.textMuted,
              }}
            >
              Select a model from the list to see details
            </div>
          ) : (
            <div style={{ padding: 20 }}>
              {/* Model Header */}
              <div style={{ marginBottom: 20 }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    marginBottom: 8,
                  }}
                >
                  <Title level={4} style={{ margin: 0 }}>
                    {selectedModel.name}
                  </Title>
                  <Button
                    type="link"
                    size="small"
                    icon={<LinkOutlined />}
                    href={`https://ollama.com/library/${selectedModel.name}`}
                    target="_blank"
                  />
                </div>

                {selectedModel.description && (
                  <Text
                    type="secondary"
                    style={{ display: "block", marginBottom: 12 }}
                  >
                    {selectedModel.description}
                  </Text>
                )}

                {/* Capabilities */}
                {selectedModel.capabilities.length > 0 && (
                  <div style={{ marginBottom: 12 }}>
                    {selectedModel.capabilities.map((cap) => (
                      <Tag
                        key={cap}
                        color={CAPABILITY_COLORS[cap] || "default"}
                        icon={CAPABILITY_ICONS[cap]}
                        style={{ marginRight: 8 }}
                      >
                        {cap.charAt(0).toUpperCase() + cap.slice(1)}
                      </Tag>
                    ))}
                  </div>
                )}

                <Text type="secondary">
                  <DownloadOutlined style={{ marginRight: 4 }} />
                  {formatNumber(selectedModel.pulls)} pulls
                </Text>
              </div>

              <Divider />

              {/* Tag Selection */}
              <div>
                <Title level={5}>Select Version</Title>
                {loadingTags ? (
                  <div style={{ textAlign: "center", padding: 20 }}>
                    <Loading size="small" />
                    <Text type="secondary" style={{ marginLeft: 8 }}>
                      Loading available tags...
                    </Text>
                  </div>
                ) : (
                  <div>
                    <Select
                      style={{ width: "100%", marginBottom: 16 }}
                      placeholder="Select a tag/version"
                      value={selectedTag}
                      onChange={setSelectedTag}
                      options={tags.map((tag) => ({
                        label: (
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              alignItems: "center",
                            }}
                          >
                            <span>{tag.name}</span>
                            <Space size={4}>
                              {tag.size && (
                                <Tag color="blue" style={{ margin: 0 }}>
                                  {tag.size}
                                </Tag>
                              )}
                              {tag.quantization && (
                                <Tag style={{ margin: 0 }}>
                                  {tag.quantization}
                                </Tag>
                              )}
                            </Space>
                          </div>
                        ),
                        value: tag.name,
                      }))}
                    />

                    {/* Preview full model name */}
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
                        style={{
                          fontSize: 12,
                          marginBottom: 4,
                          display: "block",
                        }}
                      >
                        Model ID:
                      </Text>
                      <Text code style={{ fontSize: 16, fontWeight: 600 }}>
                        {selectedModel.name}
                        {selectedTag ? `:${selectedTag}` : ""}
                      </Text>
                    </div>

                    {/* Available sizes info */}
                    {selectedModel.sizes.length > 0 && (
                      <div>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          Available sizes: {selectedModel.sizes.join(", ")}
                        </Text>
                      </div>
                    )}
                  </div>
                )}
              </div>

              <Divider />

              {/* Usage info */}
              <div>
                <Title level={5}>Usage</Title>
                <div
                  style={{
                    padding: 12,
                    background: isDark ? "#1a1a1a" : "#f5f5f5",
                    borderRadius: 8,
                    fontFamily: "monospace",
                    fontSize: 13,
                  }}
                >
                  ollama run {selectedModel.name}
                  {selectedTag && selectedTag !== "latest"
                    ? `:${selectedTag}`
                    : ""}
                </div>
              </div>

              {/* README Section */}
              <Divider>
                <Space>
                  <FileTextOutlined />
                  <span>README</span>
                </Space>
              </Divider>
              {loadingReadme ? (
                <div style={{ textAlign: "center", padding: 20 }}>
                  <Loading size="small" />
                  <Text type="secondary" style={{ marginLeft: 8 }}>
                    Loading README...
                  </Text>
                </div>
              ) : selectedModel.readme ? (
                <div
                  className="markdown-body"
                  style={{
                    fontSize: 14,
                    lineHeight: 1.7,
                    color: colors.text,
                    paddingRight: 8,
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
                    {selectedModel.readme}
                  </ReactMarkdown>
                </div>
              ) : (
                <div style={{ padding: "16px 0", textAlign: "center" }}>
                  <Text type="secondary">README not available</Text>
                  <br />
                  <Button
                    type="link"
                    icon={<LinkOutlined />}
                    href={`https://ollama.com/library/${selectedModel.name}`}
                    target="_blank"
                  >
                    View on Ollama
                  </Button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </Drawer>
  );
}
