/**
 * Desktop Sidebar
 */
import { useState } from 'react'
import { Layout, Menu, Button } from 'antd'
import { MenuFoldOutlined, MenuUnfoldOutlined } from '@ant-design/icons'
import { LogoIcon } from './Logo'
import type { useAppTheme } from '../../hooks/useTheme'

const { Sider } = Layout

interface SidebarProps {
  collapsed: boolean
  onCollapse: (collapsed: boolean) => void
  menuItems: any[]
  selectedKey: string
  openKeys: string[]
  onNavigate: (path: string) => void
  isDark: boolean
  colors: ReturnType<typeof useAppTheme>['colors']
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
  const [logoHovered, setLogoHovered] = useState(false)
  const sidebarWidth = collapsed ? 80 : 240

  return (
    <Sider
      width={sidebarWidth}
      collapsed={collapsed}
      collapsedWidth={80}
      style={{
        background: colors.siderBg,
        borderRight: `1px solid ${colors.border}`,
        position: 'fixed',
        height: '100vh',
        left: 0,
        top: 0,
        zIndex: 100,
        transition: 'width 0.2s ease',
      }}
    >
      {/* Logo and Collapse Button */}
      <div
        style={{
          padding: collapsed ? '16px' : '16px 12px 16px 20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: collapsed ? 'center' : 'space-between',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            cursor: collapsed ? 'pointer' : 'default',
          }}
          onClick={() => collapsed && onCollapse(false)}
          onMouseEnter={() => collapsed && setLogoHovered(true)}
          onMouseLeave={() => setLogoHovered(false)}
        >
          <div
            style={{
              width: 28,
              height: 28,
              background: isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.08)',
              borderRadius: 6,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              transition: 'background 0.15s ease',
            }}
          >
            {collapsed && logoHovered ? (
              <MenuUnfoldOutlined style={{ fontSize: 14, color: colors.text }} />
            ) : (
              <LogoIcon color={colors.text} />
            )}
          </div>
          {!collapsed && (
            <span
              style={{
                fontSize: 16,
                fontWeight: 600,
                color: colors.text,
                letterSpacing: '-0.02em',
                whiteSpace: 'nowrap',
              }}
            >
              LMStack
            </span>
          )}
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
          if (!key.startsWith('workers-group') && !key.startsWith('docker-group')) {
            onNavigate(key)
          }
        }}
        inlineCollapsed={collapsed}
        style={{
          marginTop: 4,
          border: 'none',
          background: 'transparent',
          padding: collapsed ? '0 8px' : '0 12px',
        }}
      />
    </Sider>
  )
}
