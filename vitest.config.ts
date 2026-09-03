import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // 纯 node 环境：数学库无 DOM / WebGPU 运行时依赖
    environment: "node",
    include: ["tests/**/*.spec.ts"],
  },
});
