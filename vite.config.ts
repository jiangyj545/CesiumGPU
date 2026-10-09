import { defineConfig } from 'vite';
import { wgslLoaderPlugin, examplesDirectoryIndexPlugin } from './vite.plugins';

export default defineConfig({
  root: '.',
  publicDir: 'public',
  build: {
    outDir: 'dist',
    sourcemap: true,
    target: 'esnext',
  },
  server: {
    port: 3000,
    open: true,
    fs: {
      // 场景引导页(public/scenes)经 /@fs 读平台仓(仿真)场景 JSON 透传 &spec=
      // —— 90 页城市锚点只在带 spec 时对准 SSE 航线(缺省会偏 ~270m 出视锥)
      allow: ['.', 'D:/work/仿真'],
    },
  },
  // Inline WGSL shader files as raw strings（插件实现在 vite.plugins.ts，
  // 与 vite.lib.config.ts 共享）
  assetsInclude: ['**/*.wgsl'],
  plugins: [examplesDirectoryIndexPlugin(), wgslLoaderPlugin()],
});
