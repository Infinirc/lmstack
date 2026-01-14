/**
 * Theme Hook
 *
 * Provides theme detection and management functionality.
 * Supports system preference detection and manual override.
 *
 * @module hooks/useTheme
 */
import { useState, useEffect, useCallback } from "react";
import { STORAGE_KEYS } from "../constants";

/**
 * Application color palette
 */
export interface AppColors {
  // Backgrounds
  bg: string;
  siderBg: string;
  headerBg: string;
  contentBg: string;
  cardBg: string;

  // Text
  text: string;
  textSecondary: string;
  textMuted: string;

  // Borders
  border: string;
  borderLight: string;

  // Interactive states
  menuItemHover: string;
  menuItemSelected: string;
  accent: string;
}

/**
 * Auth page specific colors
 */
export interface AuthColors {
  bg: string;
  text: string;
  textSecondary: string;
  textMuted: string;
  inputBg: string;
  inputBorder: string;
  inputBorderHover: string;
  buttonBg: string;
  buttonText: string;
  logoBg: string;
  logoColor: string;
}

/**
 * Dark theme colors - OpenAI/SpaceX style
 */
const DARK_COLORS: AppColors = {
  bg: "#000000",
  siderBg: "#000000",
  headerBg: "#000000",
  contentBg: "#000000",
  cardBg: "rgba(255, 255, 255, 0.03)",
  text: "#ffffff",
  textSecondary: "rgba(255, 255, 255, 0.6)",
  textMuted: "rgba(255, 255, 255, 0.4)",
  border: "rgba(255, 255, 255, 0.08)",
  borderLight: "rgba(255, 255, 255, 0.04)",
  menuItemHover: "rgba(255, 255, 255, 0.06)",
  menuItemSelected: "rgba(255, 255, 255, 0.1)",
  accent: "#ffffff",
};

/**
 * Light theme colors - Clean Professional style
 */
const LIGHT_COLORS: AppColors = {
  bg: "#f8fafc",
  siderBg: "#ffffff",
  headerBg: "#ffffff",
  contentBg: "#f8fafc",
  cardBg: "#ffffff",
  text: "#0f172a",
  textSecondary: "#475569",
  textMuted: "#64748b",
  border: "#e2e8f0",
  borderLight: "#f1f5f9",
  menuItemHover: "#f1f5f9",
  menuItemSelected: "#e2e8f0",
  accent: "#0f172a",
};

/**
 * Dark theme auth colors - OpenAI/SpaceX style
 */
const DARK_AUTH_COLORS: AuthColors = {
  bg: "#000000",
  text: "#ffffff",
  textSecondary: "rgba(255, 255, 255, 0.6)",
  textMuted: "rgba(255, 255, 255, 0.4)",
  inputBg: "rgba(255, 255, 255, 0.05)",
  inputBorder: "rgba(255, 255, 255, 0.1)",
  inputBorderHover: "rgba(255, 255, 255, 0.2)",
  buttonBg: "#ffffff",
  buttonText: "#000000",
  logoBg: "#ffffff",
  logoColor: "#000000",
};

/**
 * Light theme auth colors - Clean Professional style
 */
const LIGHT_AUTH_COLORS: AuthColors = {
  bg: "#f8fafc",
  text: "#0f172a",
  textSecondary: "#475569",
  textMuted: "#64748b",
  inputBg: "#ffffff",
  inputBorder: "#e2e8f0",
  inputBorderHover: "#cbd5e1",
  buttonBg: "#0f172a",
  buttonText: "#ffffff",
  logoBg: "#f1f5f9",
  logoColor: "#0f172a",
};

/**
 * Get initial theme preference from storage or system
 */
function getInitialTheme(): boolean {
  if (typeof window === "undefined") return false;

  const saved = localStorage.getItem(STORAGE_KEYS.THEME);
  if (saved) return saved === "dark";

  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

/**
 * Theme hook return type
 */
export interface UseThemeReturn {
  isDark: boolean;
  colors: AppColors;
  authColors: AuthColors;
  toggleTheme: (dark: boolean) => void;
}

/**
 * Hook for theme management
 *
 * @returns Theme state and colors
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { isDark, colors, toggleTheme } = useAppTheme()
 *
 *   return (
 *     <div style={{ background: colors.bg }}>
 *       <Switch checked={isDark} onChange={toggleTheme} />
 *     </div>
 *   )
 * }
 * ```
 */
export function useAppTheme(): UseThemeReturn {
  const [isDark, setIsDark] = useState(getInitialTheme);

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

    const handleChange = (e: MediaQueryListEvent) => {
      // Only update if user hasn't set a preference
      if (!localStorage.getItem(STORAGE_KEYS.THEME)) {
        setIsDark(e.matches);
      }
    };

    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, []);

  // Listen for storage changes (sync theme across components)
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === STORAGE_KEYS.THEME && e.newValue) {
        setIsDark(e.newValue === "dark");
      }
    };

    // Also listen for custom theme change events (same tab)
    const handleThemeChange = (e: CustomEvent<{ isDark: boolean }>) => {
      setIsDark(e.detail.isDark);
    };

    window.addEventListener("storage", handleStorageChange);
    window.addEventListener("themeChange", handleThemeChange as EventListener);

    return () => {
      window.removeEventListener("storage", handleStorageChange);
      window.removeEventListener(
        "themeChange",
        handleThemeChange as EventListener,
      );
    };
  }, []);

  const toggleTheme = useCallback((dark: boolean) => {
    setIsDark(dark);
    localStorage.setItem(STORAGE_KEYS.THEME, dark ? "dark" : "light");
    // Dispatch custom event for same-tab sync
    window.dispatchEvent(
      new CustomEvent("themeChange", { detail: { isDark: dark } }),
    );
  }, []);

  return {
    isDark,
    colors: isDark ? DARK_COLORS : LIGHT_COLORS,
    authColors: isDark ? DARK_AUTH_COLORS : LIGHT_AUTH_COLORS,
    toggleTheme,
  };
}

/**
 * Get theme colors without hook (for non-component usage)
 */
export function getAppColors(isDark: boolean): AppColors {
  return isDark ? DARK_COLORS : LIGHT_COLORS;
}

/**
 * Get auth colors without hook
 */
export function getAuthColors(isDark: boolean): AuthColors {
  return isDark ? DARK_AUTH_COLORS : LIGHT_AUTH_COLORS;
}
