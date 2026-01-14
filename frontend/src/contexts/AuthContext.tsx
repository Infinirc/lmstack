/**
 * Authentication Context
 *
 * Provides authentication state and methods throughout the application.
 * Handles token storage, user session management, and initialization checks.
 *
 * @module contexts/AuthContext
 */
import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react";
import { STORAGE_KEYS } from "../constants";
import { authApi } from "../services/api";
import type { User, UserRole } from "../types";

/**
 * Role hierarchy for permission checks
 * Higher number = more permissions
 */
const ROLE_HIERARCHY: Record<UserRole, number> = {
  viewer: 0,
  operator: 1,
  admin: 2,
};

/**
 * Authentication context value interface
 */
interface AuthContextValue {
  /** Current authenticated user, null if not logged in */
  user: User | null;
  /** Authentication token */
  token: string | null;
  /** Whether auth state is being loaded */
  isLoading: boolean;
  /** Whether the system has been initialized (admin created) */
  isInitialized: boolean | null;
  /** Log in a user with token and user data */
  login: (token: string, user: User) => void;
  /** Log out the current user */
  logout: () => void;
  /** Re-check authentication status */
  checkAuth: () => Promise<void>;
  /** Check if system has been initialized */
  checkSetupStatus: () => Promise<boolean>;
  /** Check if current user has at least the required role */
  hasPermission: (requiredRole: UserRole) => boolean;
  /** Whether current user is an admin */
  isAdmin: boolean;
  /** Whether current user is at least an operator (operator+) */
  isOperator: boolean;
  /** Whether current user can edit resources (operator+) */
  canEdit: boolean;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

/**
 * Authentication provider props
 */
interface AuthProviderProps {
  children: ReactNode;
}

/**
 * Authentication Provider Component
 *
 * Wraps the application to provide authentication context.
 *
 * @example
 * ```tsx
 * function App() {
 *   return (
 *     <AuthProvider>
 *       <MyApp />
 *     </AuthProvider>
 *   )
 * }
 * ```
 */
export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isInitialized, setIsInitialized] = useState<boolean | null>(null);

  /**
   * Store credentials and update state after successful login
   */
  const login = useCallback((newToken: string, newUser: User) => {
    localStorage.setItem(STORAGE_KEYS.TOKEN, newToken);
    localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(newUser));
    setToken(newToken);
    setUser(newUser);
    setIsInitialized(true);
  }, []);

  /**
   * Clear credentials and reset state on logout
   */
  const logout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEYS.TOKEN);
    localStorage.removeItem(STORAGE_KEYS.USER);
    setToken(null);
    setUser(null);
  }, []);

  /**
   * Verify stored token is still valid
   */
  const checkAuth = useCallback(async () => {
    const storedToken = localStorage.getItem(STORAGE_KEYS.TOKEN);

    if (!storedToken) {
      setIsLoading(false);
      return;
    }

    try {
      const currentUser = await authApi.getCurrentUser();
      setToken(storedToken);
      setUser(currentUser);
    } catch {
      // Token is invalid or expired
      localStorage.removeItem(STORAGE_KEYS.TOKEN);
      localStorage.removeItem(STORAGE_KEYS.USER);
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Check if the system has been initialized with an admin user
   */
  const checkSetupStatus = useCallback(async (): Promise<boolean> => {
    try {
      const status = await authApi.getSetupStatus();
      setIsInitialized(status.initialized);
      return status.initialized;
    } catch {
      return false;
    }
  }, []);

  /**
   * Check if current user has at least the required role
   */
  const hasPermission = useCallback(
    (requiredRole: UserRole): boolean => {
      if (!user) return false;
      return ROLE_HIERARCHY[user.role] >= ROLE_HIERARCHY[requiredRole];
    },
    [user],
  );

  // Computed permission flags
  const isAdmin = user?.role === "admin";
  const isOperator = hasPermission("operator");
  const canEdit = isOperator; // operator+ can edit resources

  // Initialize auth state on mount
  useEffect(() => {
    const init = async () => {
      await checkSetupStatus();
      await checkAuth();
    };
    init();
  }, [checkSetupStatus, checkAuth]);

  const value: AuthContextValue = {
    user,
    token,
    isLoading,
    isInitialized,
    login,
    logout,
    checkAuth,
    checkSetupStatus,
    hasPermission,
    isAdmin,
    isOperator,
    canEdit,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/**
 * Hook to access authentication context
 *
 * @returns Authentication context value
 * @throws Error if used outside of AuthProvider
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { user, logout } = useAuth()
 *
 *   return (
 *     <div>
 *       <span>Hello, {user?.username}</span>
 *       <button onClick={logout}>Logout</button>
 *     </div>
 *   )
 * }
 * ```
 */
export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);

  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }

  return context;
}
