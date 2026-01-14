import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useResponsive } from './useResponsive'

describe('useResponsive', () => {
  const originalInnerWidth = window.innerWidth

  beforeEach(() => {
    // Reset window width
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1200,
    })
  })

  afterEach(() => {
    // Restore original width
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: originalInnerWidth,
    })
  })

  it('returns desktop state for large screens', () => {
    Object.defineProperty(window, 'innerWidth', { value: 1200 })

    const { result } = renderHook(() => useResponsive())

    expect(result.current.isDesktop).toBe(true)
    expect(result.current.isTablet).toBe(false)
    expect(result.current.isMobile).toBe(false)
    expect(result.current.windowWidth).toBe(1200)
  })

  it('returns tablet state for medium screens', () => {
    Object.defineProperty(window, 'innerWidth', { value: 900 })

    const { result } = renderHook(() => useResponsive())

    expect(result.current.isDesktop).toBe(false)
    expect(result.current.isTablet).toBe(true)
    expect(result.current.isMobile).toBe(false)
  })

  it('returns mobile state for small screens', () => {
    Object.defineProperty(window, 'innerWidth', { value: 500 })

    const { result } = renderHook(() => useResponsive())

    expect(result.current.isDesktop).toBe(false)
    expect(result.current.isTablet).toBe(false)
    expect(result.current.isMobile).toBe(true)
  })

  it('responds to window resize', () => {
    Object.defineProperty(window, 'innerWidth', { value: 1200 })

    const { result } = renderHook(() => useResponsive())

    expect(result.current.isDesktop).toBe(true)

    // Simulate resize to mobile
    act(() => {
      Object.defineProperty(window, 'innerWidth', { value: 500 })
      window.dispatchEvent(new Event('resize'))
    })

    expect(result.current.isMobile).toBe(true)
  })

  it('handles boundary values correctly', () => {
    // At mobile breakpoint (768)
    Object.defineProperty(window, 'innerWidth', { value: 768 })
    const { result: result768 } = renderHook(() => useResponsive())
    expect(result768.current.isMobile).toBe(false)
    expect(result768.current.isTablet).toBe(true)

    // Just below mobile breakpoint
    Object.defineProperty(window, 'innerWidth', { value: 767 })
    const { result: result767 } = renderHook(() => useResponsive())
    expect(result767.current.isMobile).toBe(true)

    // At tablet breakpoint (1024)
    Object.defineProperty(window, 'innerWidth', { value: 1024 })
    const { result: result1024 } = renderHook(() => useResponsive())
    expect(result1024.current.isDesktop).toBe(true)
    expect(result1024.current.isTablet).toBe(false)
  })

  it('cleans up event listener on unmount', () => {
    const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener')

    const { unmount } = renderHook(() => useResponsive())
    unmount()

    expect(removeEventListenerSpy).toHaveBeenCalledWith(
      'resize',
      expect.any(Function)
    )
  })
})
