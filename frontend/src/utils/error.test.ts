import { describe, it, expect, vi } from "vitest";
import { getErrorMessage } from "./error";

// Mock antd message
vi.mock("antd", () => ({
  message: {
    error: vi.fn(),
    success: vi.fn(),
    info: vi.fn(),
  },
}));

describe("getErrorMessage", () => {
  it("returns fallback for null/undefined", () => {
    expect(getErrorMessage(null)).toBe("An error occurred");
    expect(getErrorMessage(undefined)).toBe("An error occurred");
  });

  it("returns custom fallback when provided", () => {
    expect(getErrorMessage(null, "Custom error")).toBe("Custom error");
  });

  it("extracts detail from API error response", () => {
    const error = {
      response: {
        data: {
          detail: "API error detail",
        },
      },
    };
    expect(getErrorMessage(error)).toBe("API error detail");
  });

  it("extracts message from API error response", () => {
    const error = {
      response: {
        data: {
          message: "API error message",
        },
      },
    };
    expect(getErrorMessage(error)).toBe("API error message");
  });

  it("extracts message from Error object", () => {
    const error = new Error("Test error message");
    expect(getErrorMessage(error)).toBe("Test error message");
  });

  it("returns string error directly", () => {
    expect(getErrorMessage("String error")).toBe("String error");
  });

  it("returns fallback for unknown error types", () => {
    expect(getErrorMessage({ unknown: "object" })).toBe("An error occurred");
    expect(getErrorMessage(123)).toBe("An error occurred");
  });

  it("prefers detail over message in API error", () => {
    const error = {
      response: {
        data: {
          detail: "Detail message",
          message: "Message message",
        },
      },
    };
    expect(getErrorMessage(error)).toBe("Detail message");
  });
});
