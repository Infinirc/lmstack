/**
 * Logo Icon Component
 */
import logoLight from "../../assets/logo/LMStack-light.png";
import logoDark from "../../assets/logo/LMStack-dark.png";

interface LogoIconProps {
  color?: string;
  width?: number;
  height?: number;
  isDark?: boolean;
}

export function LogoIcon({
  width = 160,
  height = 36,
  isDark = false,
}: LogoIconProps) {
  return (
    <img
      src={isDark ? logoDark : logoLight}
      alt="LMStack"
      style={{
        width: width,
        height: height,
        objectFit: "cover",
        objectPosition: "center 46%",
      }}
    />
  );
}
