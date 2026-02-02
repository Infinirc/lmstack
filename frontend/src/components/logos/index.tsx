/**
 * Logo Components
 *
 * Shared logo components for inference backends and model sources.
 *
 * @module components/logos
 */

// Backend logos
import vllmLogoDark from "../../assets/vllm-logo-text-dark.webp";
import vllmLogoLight from "../../assets/vllm-logo-text-light.webp";
import ollamaLogoDark from "../../assets/ollama-dark.png";
import ollamaLogoLight from "../../assets/ollama-light.png";
import sglangLogo from "../../assets/sglang.png";
import huggingfaceLogo from "../../assets/huggingface-2.svg";
import mlxLogo from "../../assets/mlx-logo.png";
import mlxLogoDark from "../../assets/mlx-logo-dark.png";

// =============================================================================
// Props Types
// =============================================================================

interface LogoProps {
  height?: number;
  isDark?: boolean;
  style?: React.CSSProperties;
}

// =============================================================================
// Backend Logos
// =============================================================================

export function VllmLogo({ height = 16, isDark = false, style }: LogoProps) {
  return (
    <img
      src={isDark ? vllmLogoDark : vllmLogoLight}
      alt="vLLM"
      style={{ height, width: "auto", objectFit: "contain", ...style }}
    />
  );
}

export function OllamaLogo({ height = 16, isDark = false, style }: LogoProps) {
  return (
    <img
      src={isDark ? ollamaLogoDark : ollamaLogoLight}
      alt="Ollama"
      style={{ height, width: "auto", objectFit: "contain", ...style }}
    />
  );
}

export function SGLangLogo({ height = 16, style }: Omit<LogoProps, "isDark">) {
  return (
    <img
      src={sglangLogo}
      alt="SGLang"
      style={{ height, width: "auto", objectFit: "contain", ...style }}
    />
  );
}

export function HuggingFaceLogo({
  height = 16,
  style,
}: Omit<LogoProps, "isDark">) {
  return (
    <img
      src={huggingfaceLogo}
      alt="HuggingFace"
      style={{ height, width: "auto", objectFit: "contain", ...style }}
    />
  );
}

/**
 * MLX Logo - Apple's ML framework for Apple Silicon
 */
export function MLXLogo({ height = 16, isDark = false, style }: LogoProps) {
  return (
    <img
      src={isDark ? mlxLogoDark : mlxLogo}
      alt="MLX"
      style={{ height, width: "auto", objectFit: "contain", ...style }}
    />
  );
}

/**
 * Llama.cpp Logo - High-performance LLM inference
 * Uses official branding colors: white text with orange C++
 */
export function LlamaCppLogo({
  height = 16,
  isDark = false,
  style,
}: LogoProps) {
  const textColor = isDark ? "#ffffff" : "#1b1f20";
  return (
    <svg width={height * 4.5} height={height} viewBox="0 0 90 20" style={style}>
      <text
        x="0"
        y="15"
        fontSize="14"
        fontWeight="700"
        fontFamily="system-ui, -apple-system, sans-serif"
        fill={textColor}
      >
        llama
      </text>
      <text
        x="44"
        y="15"
        fontSize="14"
        fontWeight="700"
        fontFamily="system-ui, -apple-system, sans-serif"
        fill="#ff8236"
      >
        .cpp
      </text>
    </svg>
  );
}

// =============================================================================
// Icons
// =============================================================================

interface IconProps {
  size?: number;
  style?: React.CSSProperties;
  className?: string;
}

export function DockerIcon({ size = 14, style, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      style={style}
      className={className}
    >
      <path d="M13.983 11.078h2.119a.186.186 0 00.186-.185V9.006a.186.186 0 00-.186-.186h-2.119a.185.185 0 00-.185.185v1.888c0 .102.083.185.185.185zm-2.954-5.43h2.118a.186.186 0 00.186-.186V3.574a.186.186 0 00-.186-.185h-2.118a.185.185 0 00-.185.185v1.888c0 .102.082.185.185.185zm0 2.716h2.118a.187.187 0 00.186-.186V6.29a.186.186 0 00-.186-.185h-2.118a.185.185 0 00-.185.185v1.887c0 .102.082.186.185.186zm-2.93 0h2.12a.186.186 0 00.184-.186V6.29a.185.185 0 00-.185-.185H8.1a.185.185 0 00-.185.185v1.887c0 .102.083.186.185.186zm-2.964 0h2.119a.186.186 0 00.185-.186V6.29a.185.185 0 00-.185-.185H5.136a.186.186 0 00-.186.185v1.887c0 .102.084.186.186.186zm5.893 2.715h2.118a.186.186 0 00.186-.185V9.006a.186.186 0 00-.186-.186h-2.118a.185.185 0 00-.185.185v1.888c0 .102.082.185.185.185zm-2.93 0h2.12a.185.185 0 00.184-.185V9.006a.185.185 0 00-.184-.186h-2.12a.185.185 0 00-.184.185v1.888c0 .102.083.185.185.185zm-2.964 0h2.119a.185.185 0 00.185-.185V9.006a.185.185 0 00-.185-.186h-2.12a.186.186 0 00-.185.186v1.887c0 .102.084.185.186.185zm-2.92 0h2.12a.185.185 0 00.184-.185V9.006a.185.185 0 00-.184-.186h-2.12a.185.185 0 00-.184.185v1.888c0 .102.082.185.185.185zM23.763 9.89c-.065-.051-.672-.51-1.954-.51-.338.001-.676.03-1.01.087-.248-1.7-1.653-2.53-1.716-2.566l-.344-.199-.226.327c-.284.438-.49.922-.612 1.43-.23.97-.09 1.882.403 2.661-.595.332-1.55.413-1.744.42H.751a.751.751 0 00-.75.748 11.376 11.376 0 00.692 4.062c.545 1.428 1.355 2.48 2.41 3.124 1.18.723 3.1 1.137 5.275 1.137.983.003 1.963-.086 2.93-.266a12.248 12.248 0 003.823-1.389c.98-.567 1.86-1.288 2.61-2.136 1.252-1.418 1.998-2.997 2.553-4.4h.221c1.372 0 2.215-.549 2.68-1.009.309-.293.55-.65.707-1.046l.098-.288z" />
    </svg>
  );
}

// =============================================================================
// Backend Config Helper
// =============================================================================

export interface BackendConfig {
  label: string;
  color: string;
  icon: React.ReactNode;
}

/**
 * Get backend configuration for UI display
 */
export function getBackendConfig(
  isDark: boolean,
): Record<string, BackendConfig> {
  const tagColor = isDark ? "#ffffff" : "#000000";
  return {
    vllm: {
      label: "vLLM",
      color: tagColor,
      icon: <VllmLogo height={16} isDark={isDark} />,
    },
    sglang: {
      label: "SGLang",
      color: tagColor,
      icon: <SGLangLogo height={16} />,
    },
    ollama: {
      label: "Ollama",
      color: tagColor,
      icon: <OllamaLogo height={16} isDark={isDark} />,
    },
    mlx: {
      label: "MLX",
      color: tagColor,
      icon: <MLXLogo height={16} isDark={isDark} />,
    },
    llama_cpp: {
      label: "llama.cpp",
      color: tagColor,
      icon: <LlamaCppLogo height={16} isDark={isDark} />,
    },
  };
}
