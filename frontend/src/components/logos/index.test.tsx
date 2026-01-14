import { describe, it, expect } from "vitest";
import { render, screen } from "../../test/test-utils";
import {
  VllmLogo,
  OllamaLogo,
  SGLangLogo,
  HuggingFaceLogo,
  DockerIcon,
  getBackendConfig,
} from "./index";

describe("Logo Components", () => {
  describe("VllmLogo", () => {
    it("renders with default height", () => {
      render(<VllmLogo />);
      const img = screen.getByRole("img");
      expect(img).toBeInTheDocument();
      expect(img).toHaveStyle({ height: "16px" });
    });

    it("renders with custom height", () => {
      render(<VllmLogo height={32} />);
      const img = screen.getByRole("img");
      expect(img).toHaveStyle({ height: "32px" });
    });

    it("applies custom style", () => {
      render(<VllmLogo style={{ marginLeft: "10px" }} />);
      const img = screen.getByRole("img");
      expect(img).toHaveStyle({ marginLeft: "10px" });
    });
  });

  describe("OllamaLogo", () => {
    it("renders with default props", () => {
      render(<OllamaLogo />);
      const img = screen.getByRole("img");
      expect(img).toBeInTheDocument();
    });

    it("renders with custom height", () => {
      render(<OllamaLogo height={24} />);
      const img = screen.getByRole("img");
      expect(img).toHaveStyle({ height: "24px" });
    });
  });

  describe("SGLangLogo", () => {
    it("renders with default props", () => {
      render(<SGLangLogo />);
      const img = screen.getByRole("img");
      expect(img).toBeInTheDocument();
    });
  });

  describe("HuggingFaceLogo", () => {
    it("renders with default props", () => {
      render(<HuggingFaceLogo />);
      const img = screen.getByRole("img");
      expect(img).toBeInTheDocument();
    });
  });

  describe("DockerIcon", () => {
    it("renders SVG icon", () => {
      const { container } = render(<DockerIcon />);
      const svg = container.querySelector("svg");
      expect(svg).toBeInTheDocument();
    });

    it("renders with custom size", () => {
      const { container } = render(<DockerIcon size={24} />);
      const svg = container.querySelector("svg");
      expect(svg).toHaveAttribute("width", "24");
      expect(svg).toHaveAttribute("height", "24");
    });

    it("applies custom className", () => {
      const { container } = render(<DockerIcon className="custom-class" />);
      const svg = container.querySelector("svg");
      expect(svg).toHaveClass("custom-class");
    });
  });
});

describe("getBackendConfig", () => {
  it("returns config for light mode", () => {
    const config = getBackendConfig(false);

    expect(config).toHaveProperty("vllm");
    expect(config).toHaveProperty("ollama");
    expect(config).toHaveProperty("sglang");

    expect(config.vllm.name).toBe("vLLM");
    expect(config.ollama.name).toBe("Ollama");
    expect(config.sglang.name).toBe("SGLang");
  });

  it("returns config for dark mode", () => {
    const config = getBackendConfig(true);

    expect(config).toHaveProperty("vllm");
    expect(config).toHaveProperty("ollama");
    expect(config).toHaveProperty("sglang");
  });

  it("each backend has required properties", () => {
    const config = getBackendConfig(false);

    for (const key of Object.keys(config)) {
      const backend = config[key];
      expect(backend).toHaveProperty("name");
      expect(backend).toHaveProperty("logo");
      expect(typeof backend.name).toBe("string");
    }
  });
});
