import { defineConfig } from 'vite';

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
  // Inline WGSL shader files as raw strings
  assetsInclude: ['**/*.wgsl'],
  plugins: [
    {
      // 目录 URL /examples/ 会被 SPA fallback 回退到引擎主页；
      // 精确 rewrite 到真实门户文件，不影响根路径 SPA 行为与 HMR
      name: 'examples-directory-index',
      configureServer(server) {
        server.middlewares.use((req, _res, next) => {
          if (req.url === '/examples/' || req.url === '/examples') {
            req.url = '/examples/index.html';
          }
          next();
        });
      },
    },
    {
      name: 'wgsl-loader',
      transform(src, id) {
        if (id.endsWith('.wgsl')) {
          return {
            code: `export default ${JSON.stringify(src)};`,
            map: null,
          };
        }
      },
    },
  ],
});
