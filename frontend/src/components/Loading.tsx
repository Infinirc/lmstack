/**
 * Loading Component
 *
 * Unified loading indicator using ldrs Bouncy animation.
 */
import { Bouncy } from "ldrs/react";
import "ldrs/react/Bouncy.css";

interface LoadingProps {
  size?: "small" | "default" | "large";
  color?: string;
}

const SIZE_MAP = {
  small: 20,
  default: 35,
  large: 45,
};

export default function Loading({
  size = "default",
  color = "#a1a1aa",
}: LoadingProps) {
  return <Bouncy size={SIZE_MAP[size]} speed={1.75} color={color} />;
}

export function LoadingCenter({ size = "default", color }: LoadingProps) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: 20,
      }}
    >
      <Loading size={size} color={color} />
    </div>
  );
}
