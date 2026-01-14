/**
 * Application Header
 */
import { Switch, Space, Dropdown, Avatar, Button } from 'antd'
import { SunOutlined, MoonOutlined, MenuOutlined } from '@ant-design/icons'
import type { useAppTheme } from '../../hooks/useTheme'
import type { useAuth } from '../../contexts/AuthContext'

interface HeaderProps {
  title: string
  isDark: boolean
  colors: ReturnType<typeof useAppTheme>['colors']
  user: ReturnType<typeof useAuth>['user']
  userMenuItems: any[]
  onToggleTheme: (dark: boolean) => void
  isMobile: boolean
  onMenuClick: () => void
}

export function Header({
  title,
  isDark,
  colors,
  user,
  userMenuItems,
  onToggleTheme,
  isMobile,
  onMenuClick,
}: HeaderProps) {
  return (
    <header
      style={{
        background: colors.headerBg,
        padding: isMobile ? '0 16px' : '0 32px',
        height: 56,
        borderBottom: `1px solid ${colors.border}`,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        position: 'sticky',
        top: 0,
        zIndex: 99,
        backdropFilter: 'blur(12px)',
      }}
    >
      {/* Left Section */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        {isMobile && (
          <Button
            type="text"
            icon={<MenuOutlined />}
            onClick={onMenuClick}
            style={{ color: colors.text }}
          />
        )}
        <span
          style={{
            fontSize: isMobile ? 14 : 13,
            fontWeight: 500,
            color: colors.textSecondary,
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
          }}
        >
          {title}
        </span>
      </div>

      {/* Header Actions */}
      <Space size={isMobile ? 8 : 12}>
        {/* Theme Toggle */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '6px 12px',
            background: isDark ? 'rgba(255, 255, 255, 0.08)' : '#f1f5f9',
            borderRadius: 20,
            border: `1px solid ${isDark ? 'rgba(255, 255, 255, 0.15)' : '#e2e8f0'}`,
          }}
        >
          <SunOutlined style={{ fontSize: 14, color: isDark ? 'rgba(255,255,255,0.4)' : '#f59e0b' }} />
          <Switch
            checked={isDark}
            onChange={onToggleTheme}
            size="small"
            style={{
              background: isDark ? '#3b82f6' : '#94a3b8',
              minWidth: 36,
            }}
          />
          <MoonOutlined style={{ fontSize: 14, color: isDark ? '#60a5fa' : 'rgba(0,0,0,0.3)' }} />
        </div>

        {/* User Menu */}
        <Dropdown menu={{ items: userMenuItems }} placement="bottomRight" trigger={['click']}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: isMobile ? 0 : 8,
              cursor: 'pointer',
              padding: isMobile ? '4px' : '4px 12px 4px 4px',
              borderRadius: 20,
              border: `1px solid ${colors.border}`,
              transition: 'all 0.2s ease',
            }}
          >
            <Avatar
              size={24}
              style={{
                backgroundColor: colors.accent,
                color: isDark ? '#000000' : '#ffffff',
                fontSize: 11,
                fontWeight: 600,
              }}
            >
              {(user?.display_name || user?.username || 'U').charAt(0).toUpperCase()}
            </Avatar>
            {!isMobile && (
              <span
                style={{
                  fontSize: 13,
                  color: colors.text,
                  fontWeight: 500,
                  fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                }}
              >
                {user?.username}
              </span>
            )}
          </div>
        </Dropdown>
      </Space>
    </header>
  )
}
