/**
 * Desktop Sidebar
 */
import { useState } from "react";
import { Layout, Menu, Button } from "antd";
import { MenuFoldOutlined, MenuUnfoldOutlined } from "@ant-design/icons";
import { LogoIcon } from "./Logo";
import type { useAppTheme } from "../../hooks/useTheme";

const { Sider } = Layout;

interface SidebarProps {
  collapsed: boolean;
  onCollapse: (collapsed: boolean) => void;
  menuItems: any[];
  selectedKey: string;
  openKeys: string[];
  onNavigate: (path: string) => void;
  isDark: boolean;
  colors: ReturnType<typeof useAppTheme>["colors"];
}

export function Sidebar({
  collapsed,
  onCollapse,
  menuItems,
  selectedKey,
  openKeys,
  onNavigate,
  isDark,
  colors,
}: SidebarProps) {
  const [logoHovered, setLogoHovered] = useState(false);
  const sidebarWidth = collapsed ? 80 : 240;

  return (
    <Sider
      width={sidebarWidth}
      collapsed={collapsed}
      collapsedWidth={80}
      style={{
        background: colors.siderBg,
        borderRight: `1px solid ${colors.border}`,
        position: "fixed",
        height: "100vh",
        left: 0,
        top: 0,
        zIndex: 100,
        transition: "width 0.2s ease",
      }}
    >
      {/* Logo and Collapse Button */}
      <div
        style={{
          padding: collapsed ? "24px 16px" : "20px 10px 16px 1px",
          display: "flex",
          alignItems: "center",
          justifyContent: collapsed ? "center" : "space-between",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            cursor: collapsed ? "pointer" : "default",
          }}
          onClick={() => collapsed && onCollapse(false)}
          onMouseEnter={() => collapsed && setLogoHovered(true)}
          onMouseLeave={() => setLogoHovered(false)}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
              transition: "all 0.15s ease",
            }}
          >
            {collapsed && logoHovered ? (
              <MenuUnfoldOutlined
                style={{ fontSize: 14, color: colors.text }}
              />
            ) : (
              <LogoIcon
                width={collapsed ? 56 : 200}
                height={collapsed ? 38 : 48}
                isDark={isDark}
              />
            )}
          </div>
        </div>
        {!collapsed && (
          <Button
            type="text"
            size="small"
            icon={<MenuFoldOutlined style={{ fontSize: 14 }} />}
            onClick={() => onCollapse(true)}
            style={{
              color: colors.textMuted,
              width: 28,
              height: 28,
              padding: 0,
            }}
          />
        )}
      </div>

      {/* Navigation Menu */}
      <Menu
        mode="inline"
        selectedKeys={[selectedKey]}
        defaultOpenKeys={openKeys}
        items={menuItems}
        onClick={({ key }) => {
          if (
            !key.startsWith("workers-group") &&
            !key.startsWith("docker-group")
          ) {
            onNavigate(key);
          }
        }}
        inlineCollapsed={collapsed}
        style={{
          marginTop: 4,
          border: "none",
          background: "transparent",
          padding: collapsed ? "0 8px" : "0 12px",
        }}
      />
    </Sider>
  );
}
