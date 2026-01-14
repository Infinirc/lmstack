import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Form, Input, Button, message, ConfigProvider, theme } from 'antd'
import { authApi } from '../services/api'
import { useAuth } from '../contexts/AuthContext'
import type { LoginRequest } from '../types'

export default function Login() {
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const { login } = useAuth()

  // Theme detection
  const getTheme = () => {
    const saved = localStorage.getItem('lmstack-theme')
    if (saved) return saved === 'dark'
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  }

  const [isDark, setIsDark] = useState(getTheme)

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleChange = () => {
      if (!localStorage.getItem('lmstack-theme')) {
        setIsDark(mediaQuery.matches)
      }
    }
    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  const onFinish = async (values: LoginRequest) => {
    setLoading(true)
    try {
      const response = await authApi.login(values)
      login(response.access_token, response.user)
      message.success('Login successful')
      navigate('/dashboard')
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } }
      message.error(err.response?.data?.detail || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  // Theme colors
  const colors = isDark ? {
    bg: '#0a0a0a',
    text: '#fafafa',
    textSecondary: '#a1a1aa',
    textMuted: '#71717a',
    inputBg: '#18181b',
    inputBorder: '#27272a',
    inputBorderHover: '#3f3f46',
    buttonBg: '#fafafa',
    buttonText: '#09090b',
    logoBg: '#fafafa',
    logoColor: '#09090b',
  } : {
    bg: '#fafafa',
    text: '#09090b',
    textSecondary: '#52525b',
    textMuted: '#a1a1aa',
    inputBg: '#ffffff',
    inputBorder: '#e4e4e7',
    inputBorderHover: '#a1a1aa',
    buttonBg: '#09090b',
    buttonText: '#fafafa',
    logoBg: '#09090b',
    logoColor: '#fafafa',
  }

  return (
    <ConfigProvider
      theme={{
        algorithm: isDark ? theme.darkAlgorithm : theme.defaultAlgorithm,
        token: {
          colorBgContainer: colors.inputBg,
          colorBorder: colors.inputBorder,
          colorText: colors.text,
          colorTextPlaceholder: colors.textMuted,
        },
      }}
    >
      <div
        style={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          background: colors.bg,
          padding: 24,
        }}
      >
        <div style={{ width: '100%', maxWidth: 360 }}>
          {/* Logo */}
          <div style={{ textAlign: 'center', marginBottom: 48 }}>
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 48,
                height: 48,
                background: colors.logoBg,
                borderRadius: 10,
                marginBottom: 24,
              }}
            >
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke={colors.logoColor} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M2 17L12 22L22 17" stroke={colors.logoColor} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M2 12L12 17L22 12" stroke={colors.logoColor} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <h1 style={{ color: colors.text, fontSize: 24, fontWeight: 600, margin: 0, letterSpacing: '-0.025em' }}>
              Welcome back
            </h1>
            <p style={{ color: colors.textSecondary, fontSize: 14, marginTop: 8 }}>
              Log in to LMStack
            </p>
          </div>

          {/* Form */}
          <Form name="login" onFinish={onFinish} layout="vertical" requiredMark={false}>
            <Form.Item name="username" rules={[{ required: true, message: '' }]} style={{ marginBottom: 16 }}>
              <Input
                placeholder="Username"
                autoComplete="username"
                size="large"
                style={{
                  background: colors.inputBg,
                  borderColor: colors.inputBorder,
                  borderRadius: 8,
                  height: 44,
                  fontSize: 14,
                }}
              />
            </Form.Item>

            <Form.Item name="password" rules={[{ required: true, message: '' }]} style={{ marginBottom: 24 }}>
              <Input.Password
                placeholder="Password"
                autoComplete="current-password"
                size="large"
                style={{
                  background: colors.inputBg,
                  borderColor: colors.inputBorder,
                  borderRadius: 8,
                  height: 44,
                  fontSize: 14,
                }}
              />
            </Form.Item>

            <Form.Item style={{ marginBottom: 0 }}>
              <Button
                type="primary"
                htmlType="submit"
                loading={loading}
                block
                style={{
                  height: 44,
                  borderRadius: 8,
                  fontSize: 14,
                  fontWeight: 500,
                  background: colors.buttonBg,
                  color: colors.buttonText,
                  border: 'none',
                }}
              >
                Login
              </Button>
            </Form.Item>
          </Form>

          <p style={{ color: colors.textMuted, fontSize: 12, textAlign: 'center', marginTop: 32 }}>
            © Infinirc LLC
          </p>
        </div>
      </div>
    </ConfigProvider>
  )
}
