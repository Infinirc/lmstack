/**
 * Mobile Sidebar (Drawer)
 */
import { Drawer, Menu, Button } from "antd";
import { MenuFoldOutlined } from "@ant-design/icons";
import { LogoIcon } from "./Logo";
import type { useAppTheme } from "../../hooks/useTheme";

interface MobileSidebarProps {
  open: boolean;
  onClose: () => void;
  menuItems: any[];
  selectedKey: string;
  openKeys: string[];
  onNavigate: (path: string) => void;
  isDark: boolean;
  colors: ReturnType<typeof useAppTheme>["colors"];
}

export function MobileSidebar({
  open,
  onClose,
  menuItems,
  selectedKey,
  openKeys,
  onNavigate,
  isDark,
  colors,
}: MobileSidebarProps) {
  return (
    <Drawer
      placement="left"
      open={open}
      onClose={onClose}
      width={280}
      styles={{
        body: { padding: 0, background: colors.siderBg },
        header: { display: "none" },
      }}
    >
      <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
        {/* Logo and Close Button */}
        <div
          style={{
            padding: "16px 12px 16px 20px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            borderBottom: `1px solid ${colors.border}`,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div
              style={{
                width: 32,
                height: 32,
                background: isDark
                  ? "rgba(255, 255, 255, 0.1)"
                  : "rgba(0, 0, 0, 0.08)",
                borderRadius: 8,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <LogoIcon color={colors.text} />
            </div>
            <span
              style={{
                fontSize: 18,
                fontWeight: 600,
                color: colors.text,
                letterSpacing: "-0.02em",
              }}
            >
              LMStack
            </span>
          </div>
          <Button
            type="text"
            size="small"
            icon={<MenuFoldOutlined style={{ fontSize: 16 }} />}
            onClick={onClose}
            style={{
              color: colors.textMuted,
              width: 32,
              height: 32,
              padding: 0,
            }}
          />
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
          style={{
            flex: 1,
            marginTop: 8,
            border: "none",
            background: "transparent",
            padding: "0 12px",
          }}
        />
      </div>
    </Drawer>
  );
}
