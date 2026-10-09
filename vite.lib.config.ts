import { defineConfig } from 'vite';
import { wgslLoaderPlugin } from './vite.plugins';

/**
 * vite.lib.config.ts – P1-G 库构建：
 * entry = src/index.ts，ESM + UMD（global `CesiumGPU`），产物 dist-lib/。
 * 与应用构建（vite.config.ts → dist/）互不影响；WGSL 内联走共享插件
 * （vite.plugins.ts）。pngjs 在 src 中无引用，不会进 bundle。
 *
 * ```bash
 * npm run build:lib        # 产物 + 类型声明（dist-lib/）
 * ```
 */
export default defineConfig({
  plugins: [wgslLoaderPlugin()],
  build: {
    outDir: 'dist-lib',
    sourcemap: true,
    target: 'esnext',
    lib: {
      entry: 'src/index.ts',
      name: 'CesiumGPU',
      formats: ['es', 'umd'],
      fileName: (format) => `cesiumgpu.${format === 'es' ? 'es' : 'umd'}.js`,
    },
  },
});
