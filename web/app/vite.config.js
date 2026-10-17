// Builds the React app into one self-contained ../index.html, which scripts/serve_demo.py serves as before.
// `npm run dev` serves it with hot reload on :5173 and forwards /api to the Python server on :8000.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: { outDir: "..", emptyOutDir: false },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
