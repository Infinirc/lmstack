/**
 * Responsive Hook
 *
 * Detects window size and device type for responsive layouts.
 */
import { useState, useEffect } from 'react'

const MOBILE_BREAKPOINT = 768
const TABLET_BREAKPOINT = 1024

export function useResponsive() {
  const [windowWidth, setWindowWidth] = useState(
    typeof window !== 'undefined' ? window.innerWidth : 1200
  )

  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth)
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  return {
    isMobile: windowWidth < MOBILE_BREAKPOINT,
    isTablet: windowWidth >= MOBILE_BREAKPOINT && windowWidth < TABLET_BREAKPOINT,
    isDesktop: windowWidth >= TABLET_BREAKPOINT,
    windowWidth,
  }
}
