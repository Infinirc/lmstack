/**
 * Theme Hook
 *
 * Provides theme detection and synchronization with system/localStorage preferences.
 */
import { useState, useEffect, useCallback } from "react";
import { getThemeColors, type ThemeColors } from "./types";

const THEME_STORAGE_KEY = "lmstack-theme";
const THEME_CHECK_INTERVAL = 500;

/**
 * Hook for managing theme state
 * Syncs with localStorage and system preferences
 */
export function useTheme() {
  const getInitialTheme = useCallback(() => {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved) return saved === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }, []);

  const [isDark, setIsDark] = useState(getInitialTheme);

  useEffect(() => {
    const checkTheme = () => {
      const saved = localStorage.getItem(THEME_STORAGE_KEY);
      if (saved) {
        setIsDark(saved === "dark");
      } else {
        setIsDark(window.matchMedia("(prefers-color-scheme: dark)").matches);
      }
    };

    // Check periodically to sync with App.tsx theme changes
    const interval = setInterval(checkTheme, THEME_CHECK_INTERVAL);

    // Also listen to system theme changes
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = () => {
      if (!localStorage.getItem(THEME_STORAGE_KEY)) {
        setIsDark(mediaQuery.matches);
      }
    };
    mediaQuery.addEventListener("change", handleChange);

    return () => {
      clearInterval(interval);
      mediaQuery.removeEventListener("change", handleChange);
    };
  }, []);

  const colors: ThemeColors = getThemeColors(isDark);

  return { isDark, colors };
}
