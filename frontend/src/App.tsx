/**
 * Application Root Component
 *
 * Main application entry point handling routing, authentication,
 * theme management, and layout structure with responsive design.
 */
import { useEffect, useState, useCallback } from "react";
import {
  Routes,
  Route,
  Navigate,
  useLocation,
  useNavigate,
} from "react-router-dom";
import { Layout, ConfigProvider, theme, Button, Tooltip } from "antd";
import {
  DashboardOutlined,
  CloudServerOutlined,
  ExperimentOutlined,
  DeploymentUnitOutlined,
  MessageOutlined,
  ApiOutlined,
  LogoutOutlined,
  TeamOutlined,
  AppstoreOutlined,
  CodeSandboxOutlined,
  SettingOutlined,
  RocketOutlined,
  GlobalOutlined,
  HddOutlined,
  CommentOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";

import { AuthProvider, useAuth } from "./contexts/AuthContext";
import { useAppTheme, useResponsive } from "./hooks";
import { Header, Sidebar, MobileSidebar } from "./components/layout";
import {
  ChatPanel,
  CHAT_PANEL_STORAGE_KEY,
  DEFAULT_PANEL_WIDTH,
  TUNING_JOB_EVENT_KEY,
} from "./components/chat-panel";
import Loading from "./components/Loading";

// Page Components
import Dashboard from "./pages/Dashboard";
import Workers from "./pages/Workers";
import Images from "./pages/Images";
import Containers from "./pages/Containers";
import Storage from "./pages/Storage";
import Models from "./pages/Models";
import Deployments from "./pages/Deployments";
import DeployApps from "./pages/DeployApps";
import Chat from "./pages/Chat";
import ApiKeys from "./pages/ApiKeys";
import Users from "./pages/Users";
import Settings from "./pages/Settings";
import Headscale from "./pages/Headscale";
import Login from "./pages/Login";
import Setup from "./pages/Setup";
import AutoTuning from "./pages/AutoTuning";

const { Content } = Layout;

// ============================================================================
// Route Protection
// ============================================================================

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, isLoading, isInitialized } = useAuth();
  const location = useLocation();

  if (isLoading || isInitialized === null) {
    return <LoadingScreen />;
  }

  if (!isInitialized) {
    return <Navigate to="/setup" state={{ from: location }} replace />;
  }

  if (!user) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
}

function RequireAdmin({ children }: { children: React.ReactNode }) {
  const { isAdmin } = useAuth();

  if (!isAdmin) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
}

function LoadingScreen() {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
        background: "#09090b",
      }}
    >
      <Loading size="large" color="#fafafa" />
    </div>
  );
}

// ============================================================================
// Navigation Configuration
// ============================================================================

// Docker icon SVG component
const DockerIcon = ({ size = 14 }: { size?: number }) => (
  <span role="img" className="anticon" style={{ marginRight: 1 }}>
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M13.983 11.078h2.119a.186.186 0 00.186-.185V9.006a.186.186 0 00-.186-.186h-2.119a.185.185 0 00-.185.185v1.888c0 .102.083.185.185.185zm-2.954-5.43h2.118a.186.186 0 00.186-.186V3.574a.186.186 0 00-.186-.185h-2.118a.185.185 0 00-.185.185v1.888c0 .102.082.185.185.185zm0 2.716h2.118a.187.187 0 00.186-.186V6.29a.186.186 0 00-.186-.185h-2.118a.185.185 0 00-.185.185v1.887c0 .102.082.186.185.186zm-2.93 0h2.12a.186.186 0 00.184-.186V6.29a.185.185 0 00-.185-.185H8.1a.185.185 0 00-.185.185v1.887c0 .102.083.186.185.186zm-2.964 0h2.119a.186.186 0 00.185-.186V6.29a.185.185 0 00-.185-.185H5.136a.186.186 0 00-.186.185v1.887c0 .102.084.186.186.186zm5.893 2.715h2.118a.186.186 0 00.186-.185V9.006a.186.186 0 00-.186-.186h-2.118a.185.185 0 00-.185.185v1.888c0 .102.082.185.185.185zm-2.93 0h2.12a.185.185 0 00.184-.185V9.006a.185.185 0 00-.184-.186h-2.12a.185.185 0 00-.184.185v1.888c0 .102.083.185.185.185zm-2.964 0h2.119a.185.185 0 00.185-.185V9.006a.185.185 0 00-.185-.186h-2.12a.186.186 0 00-.185.186v1.887c0 .102.084.185.186.185zm-2.92 0h2.12a.185.185 0 00.184-.185V9.006a.185.185 0 00-.184-.186h-2.12a.185.185 0 00-.184.185v1.888c0 .102.082.185.185.185zM23.763 9.89c-.065-.051-.672-.51-1.954-.51-.338.001-.676.03-1.01.087-.248-1.7-1.653-2.53-1.716-2.566l-.344-.199-.226.327c-.284.438-.49.922-.612 1.43-.23.97-.09 1.882.403 2.661-.595.332-1.55.413-1.744.42H.751a.751.751 0 00-.75.748 11.376 11.376 0 00.692 4.062c.545 1.428 1.355 2.48 2.41 3.124 1.18.723 3.1 1.137 5.275 1.137.983.003 1.963-.086 2.93-.266a12.248 12.248 0 003.823-1.389c.98-.567 1.86-1.288 2.61-2.136 1.252-1.418 1.998-2.997 2.553-4.4h.221c1.372 0 2.215-.549 2.68-1.009.309-.293.55-.65.707-1.046l.098-.288z" />
    </svg>
  </span>
);

function getMenuItems(isAdmin: boolean) {
  const workersChildren: any[] = [
    { key: "/workers", icon: <CloudServerOutlined />, label: "Worker Nodes" },
    {
      key: "docker-group",
      icon: <DockerIcon />,
      label: "Docker",
      children: [
        { key: "/images", icon: <AppstoreOutlined />, label: "Images" },
        {
          key: "/containers",
          icon: <CodeSandboxOutlined />,
          label: "Containers",
        },
        { key: "/storage", icon: <HddOutlined />, label: "Storage" },
      ],
    },
  ];

  if (isAdmin) {
    workersChildren.push({
      key: "/headscale",
      icon: <GlobalOutlined />,
      label: "Headscale VPN",
    });
  }

  const items: any[] = [
    { key: "/dashboard", icon: <DashboardOutlined />, label: "Dashboard" },
    { key: "/chat", icon: <MessageOutlined />, label: "Chat" },
    {
      key: "workers-group",
      icon: <CloudServerOutlined />,
      label: "Workers",
      style: { marginTop: 16 },
      children: workersChildren,
    },
    {
      key: "/models",
      icon: <ExperimentOutlined />,
      label: "Models",
      style: { marginTop: 16 },
    },
    {
      key: "/deployments",
      icon: <DeploymentUnitOutlined />,
      label: "Deploy Model",
    },
    { key: "/deploy-apps", icon: <RocketOutlined />, label: "Deploy Apps" },
    {
      key: "/auto-tuning",
      icon: <ThunderboltOutlined />,
      label: "Auto-Tuning",
    },
    {
      key: "/api-keys",
      icon: <ApiOutlined />,
      label: "API Gateway",
      style: { marginTop: 16 },
    },
  ];

  if (isAdmin) {
    items.push({ key: "/users", icon: <TeamOutlined />, label: "Users" });
    items.push({
      key: "/settings",
      icon: <SettingOutlined />,
      label: "Settings",
    });
  }

  return items;
}

function getOpenKeys(pathname: string, collapsed: boolean) {
  if (collapsed) return [];
  const dockerPaths = ["/images", "/containers", "/storage"];
  const workerPaths = ["/workers", "/headscale", ...dockerPaths];
  if (workerPaths.includes(pathname)) {
    if (dockerPaths.includes(pathname)) {
      return ["workers-group", "docker-group"];
    }
    return ["workers-group"];
  }
  return [];
}

function getCurrentPageTitle(menuItems: any[], pathname: string) {
  for (const item of menuItems) {
    if (item.key === pathname) {
      return item.label;
    }
    if (item.children) {
      for (const child of item.children) {
        if (child.key === pathname) {
          return child.label;
        }
        if (child.children) {
          const subChild = child.children.find(
            (c: { key: string }) => c.key === pathname,
          );
          if (subChild) return subChild.label;
        }
      }
    }
  }
  return "LMStack";
}

// ============================================================================
// Main Layout
// ============================================================================

/**
 * Load chat panel state from localStorage
 */
function loadChatPanelState(): { isOpen: boolean; width: number } {
  try {
    const saved = localStorage.getItem(CHAT_PANEL_STORAGE_KEY);
    if (saved) {
      const state = JSON.parse(saved);
      return {
        isOpen: state.isOpen ?? false,
        width: state.width ?? DEFAULT_PANEL_WIDTH,
      };
    }
  } catch {
    // Ignore
  }
  return { isOpen: false, width: DEFAULT_PANEL_WIDTH };
}

function AppLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuth();
  const { isDark, colors, toggleTheme } = useAppTheme();
  const { isMobile, isTablet, isDesktop } = useResponsive();

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);
  const [chatPanelOpen, setChatPanelOpen] = useState(
    () => loadChatPanelState().isOpen,
  );
  const [chatPanelWidth, setChatPanelWidth] = useState(
    () => loadChatPanelState().width,
  );

  // Persist chat panel state
  useEffect(() => {
    try {
      const current = localStorage.getItem(CHAT_PANEL_STORAGE_KEY);
      const state = current ? JSON.parse(current) : {};
      localStorage.setItem(
        CHAT_PANEL_STORAGE_KEY,
        JSON.stringify({ ...state, isOpen: chatPanelOpen }),
      );
    } catch {
      // Ignore
    }
  }, [chatPanelOpen]);

  // Listen for tuning job events to auto-open chat panel
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === TUNING_JOB_EVENT_KEY && e.newValue) {
        try {
          const data = JSON.parse(e.newValue);
          if (data.jobId) {
            setChatPanelOpen(true);
          }
        } catch {
          // Ignore
        }
      }
    };

    // Check on mount if there's a pending tuning job
    const checkInitial = () => {
      const stored = localStorage.getItem(TUNING_JOB_EVENT_KEY);
      if (stored) {
        try {
          const data = JSON.parse(stored);
          if (
            data.jobId &&
            data.timestamp &&
            Date.now() - data.timestamp < 5000
          ) {
            setChatPanelOpen(true);
          }
        } catch {
          // Ignore
        }
      }
    };

    checkInitial();
    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, []);

  useEffect(() => {
    document.body.setAttribute("data-theme", isDark ? "dark" : "light");
  }, [isDark]);

  useEffect(() => {
    setMobileDrawerOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    if (isTablet && !sidebarCollapsed) {
      setSidebarCollapsed(true);
    } else if (isDesktop && sidebarCollapsed) {
      setSidebarCollapsed(false);
    }
  }, [isTablet, isDesktop]);

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  const handleNavigate = useCallback(
    (path: string) => {
      navigate(path);
      if (isMobile) {
        setMobileDrawerOpen(false);
      }
    },
    [navigate, isMobile],
  );

  const menuItems = getMenuItems(user?.role === "admin");
  const openKeys = getOpenKeys(location.pathname, sidebarCollapsed);
  const currentPageTitle = getCurrentPageTitle(menuItems, location.pathname);
  const sidebarWidth = sidebarCollapsed ? 80 : 240;

  const userMenuItems = [
    {
      key: "user-info",
      label: (
        <div style={{ padding: "8px 0" }}>
          <div style={{ fontWeight: 500, color: colors.text }}>
            {user?.display_name || user?.username}
          </div>
          <span style={{ fontSize: 12, color: colors.textMuted }}>
            {user?.role.toUpperCase()}
          </span>
        </div>
      ),
      disabled: true,
    },
    { key: "divider", type: "divider" as const },
    {
      key: "logout",
      icon: <LogoutOutlined />,
      label: "Logout",
      onClick: handleLogout,
    },
  ];

  return (
    <ConfigProvider
      theme={{
        algorithm: isDark ? theme.darkAlgorithm : theme.defaultAlgorithm,
        token: {
          colorPrimary: colors.accent,
          colorBgContainer: colors.cardBg,
          colorBorder: colors.border,
          colorText: colors.text,
          colorTextSecondary: colors.textSecondary,
          borderRadius: 8,
        },
        components: {
          Menu: {
            itemBg: "transparent",
            itemSelectedBg: colors.menuItemSelected,
            itemHoverBg: colors.menuItemHover,
            itemSelectedColor: colors.text,
            itemColor: colors.textSecondary,
            iconSize: 16,
            itemHeight: 40,
            itemMarginInline: 8,
            itemBorderRadius: 8,
          },
          Card: {
            colorBgContainer: colors.cardBg,
            colorBorder: colors.border,
          },
          Table: {
            colorBgContainer: colors.cardBg,
            headerBg: colors.cardBg,
            rowHoverBg: colors.menuItemHover,
          },
          Button: {
            colorPrimary: colors.accent,
            colorPrimaryHover: colors.textSecondary,
            primaryColor: isDark ? "#09090b" : "#fafafa",
          },
          Input: {
            colorBgContainer: isDark ? "#18181b" : "#ffffff",
            colorBorder: colors.border,
          },
          Select: {
            colorBgContainer: isDark ? "#18181b" : "#ffffff",
            colorBorder: colors.border,
          },
          Modal: {
            contentBg: isDark ? "#1c1c1e" : colors.cardBg,
            headerBg: isDark ? "#1c1c1e" : colors.cardBg,
          },
          Dropdown: {
            colorBgElevated: colors.cardBg,
          },
          Statistic: {
            colorTextDescription: colors.textSecondary,
          },
          Tag: {
            defaultBg: colors.menuItemHover,
            defaultColor: colors.textSecondary,
          },
          Progress: {
            remainingColor: colors.border,
          },
          Drawer: {
            colorBgElevated: colors.siderBg,
          },
          Popconfirm: {
            colorBgElevated: isDark ? "#18181b" : "#ffffff",
          },
          Popover: {
            colorBgElevated: isDark ? "#18181b" : "#ffffff",
          },
          Switch: {
            colorPrimary: isDark ? "#3b82f6" : "#0f172a",
            colorPrimaryHover: isDark ? "#60a5fa" : "#1e293b",
          },
        },
      }}
    >
      <Layout style={{ minHeight: "100vh", background: colors.bg }}>
        {isMobile && (
          <MobileSidebar
            open={mobileDrawerOpen}
            onClose={() => setMobileDrawerOpen(false)}
            menuItems={menuItems}
            selectedKey={location.pathname}
            openKeys={openKeys}
            onNavigate={handleNavigate}
            isDark={isDark}
            colors={colors}
          />
        )}

        {!isMobile && (
          <Sidebar
            collapsed={sidebarCollapsed}
            onCollapse={setSidebarCollapsed}
            menuItems={menuItems}
            selectedKey={location.pathname}
            openKeys={openKeys}
            onNavigate={handleNavigate}
            isDark={isDark}
            colors={colors}
          />
        )}

        <Layout
          style={{
            marginLeft: isMobile ? 0 : sidebarWidth,
            marginRight: chatPanelOpen && !isMobile ? chatPanelWidth : 0,
            background: colors.bg,
            transition: "margin-left 0.2s ease, margin-right 0.2s ease",
          }}
        >
          <Header
            title={currentPageTitle}
            isDark={isDark}
            colors={colors}
            user={user}
            userMenuItems={userMenuItems}
            onToggleTheme={toggleTheme}
            isMobile={isMobile}
            onMenuClick={() => setMobileDrawerOpen(true)}
          />
          <Content
            style={{
              padding: isMobile ? "16px" : "24px 32px 32px",
              background: colors.bg,
              minHeight: "calc(100vh - 56px)",
            }}
          >
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/chat" element={<Chat />} />
              <Route path="/workers" element={<Workers />} />
              <Route path="/images" element={<Images />} />
              <Route path="/containers" element={<Containers />} />
              <Route path="/storage" element={<Storage />} />
              <Route path="/models" element={<Models />} />
              <Route path="/deployments" element={<Deployments />} />
              <Route path="/deploy-apps" element={<DeployApps />} />
              <Route path="/auto-tuning" element={<AutoTuning />} />
              <Route path="/api-keys" element={<ApiKeys />} />
              <Route
                path="/users"
                element={
                  <RequireAdmin>
                    <Users />
                  </RequireAdmin>
                }
              />
              <Route
                path="/headscale"
                element={
                  <RequireAdmin>
                    <Headscale />
                  </RequireAdmin>
                }
              />
              <Route
                path="/settings"
                element={
                  <RequireAdmin>
                    <Settings />
                  </RequireAdmin>
                }
              />
            </Routes>
          </Content>
        </Layout>

        {/* Floating chat button */}
        {!chatPanelOpen && (
          <Tooltip title="Open AI Chat" placement="left">
            <Button
              type="primary"
              shape="circle"
              size="large"
              icon={<CommentOutlined style={{ fontSize: 20 }} />}
              onClick={() => setChatPanelOpen(true)}
              style={{
                position: "fixed",
                bottom: 24,
                right: 24,
                width: 56,
                height: 56,
                zIndex: 998,
                boxShadow: isDark
                  ? "0 4px 16px rgba(0, 0, 0, 0.4)"
                  : "0 4px 16px rgba(0, 0, 0, 0.15)",
              }}
            />
          </Tooltip>
        )}

        {/* Chat panel */}
        <ChatPanel
          isOpen={chatPanelOpen}
          onClose={() => setChatPanelOpen(false)}
          onWidthChange={setChatPanelWidth}
          isDark={isDark}
          colors={colors}
        />
      </Layout>
    </ConfigProvider>
  );
}

// ============================================================================
// Route Management
// ============================================================================

function AppRoutes() {
  const { isLoading, isInitialized, user } = useAuth();
  const location = useLocation();

  if (isLoading || isInitialized === null) {
    return <LoadingScreen />;
  }

  if (!isInitialized && location.pathname !== "/setup") {
    return <Navigate to="/setup" replace />;
  }

  if (isInitialized && location.pathname === "/setup") {
    return <Navigate to="/login" replace />;
  }

  if (
    isInitialized &&
    !user &&
    !["/login", "/setup"].includes(location.pathname)
  ) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (user && location.pathname === "/login") {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/setup" element={<Setup />} />
      <Route
        path="/*"
        element={
          <ProtectedRoute>
            <AppLayout />
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

// ============================================================================
// App Entry Point
// ============================================================================

export default function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  );
}
