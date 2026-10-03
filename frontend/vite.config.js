import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Vite configuration.
// In development (`npm run dev`) every request to /api is forwarded to the
// FastAPI backend on port 8000, so the browser only talks to one origin.
// In Docker, nginx does the same job (see nginx.conf).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
