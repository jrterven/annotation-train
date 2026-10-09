import { defineConfig, mergeConfig } from "vitest/config";
import vite from "./vite.config";
export default mergeConfig(
  vite,
  defineConfig({ test: { setupFiles: ["./src/__tests__/setup.ts"] } }),
);
