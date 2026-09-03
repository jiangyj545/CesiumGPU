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
