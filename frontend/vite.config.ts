import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 3000,
    proxy: {
      // Use regex to match /api/ paths only (not /api-keys)
      "^/api/": {
        target: "http://localhost:52000",
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on("proxyReq", (proxyReq, req) => {
            // Forward original host for URL generation
            const host = req.headers.host;
            if (host) {
              proxyReq.setHeader("X-Forwarded-Host", host);
              proxyReq.setHeader("X-Forwarded-Proto", "http");
            }
          });
          // Disable buffering for SSE/streaming responses
          proxy.on("proxyRes", (proxyRes, req, res) => {
            // Check if this is a streaming response
            const contentType = proxyRes.headers["content-type"] || "";
            if (
              contentType.includes("text/event-stream") ||
              contentType.includes("application/stream") ||
              req.url?.includes("/chat")
            ) {
              // Disable compression and buffering for streaming
              res.setHeader("X-Accel-Buffering", "no");
              res.setHeader("Cache-Control", "no-cache, no-transform");
              // Flush headers immediately
              res.flushHeaders?.();
            }
          });
        },
      },
      "/v1": {
        target: "http://localhost:52000",
        changeOrigin: true,
      },
    },
  },
  preview: {
    host: "0.0.0.0",
    port: 3000,
  },
});
