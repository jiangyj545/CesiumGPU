import type { Plugin } from 'vite';

/**
 * vite.plugins.ts – vite.config.ts / vite.lib.config.ts 共用的插件工厂
 * （P1-G：lib 构建复用同一 WGSL loader，不复制两份实现）。
 */

/** Inline WGSL shader files as default-exported raw strings. */
export function wgslLoaderPlugin(): Plugin {
  return {
    name: 'wgsl-loader',
    transform(src, id) {
      if (id.endsWith('.wgsl')) {
        return {
          code: `export default ${JSON.stringify(src)};`,
          map: null,
        };
      }
      return undefined;
    },
  };
}

/**
 * 目录 URL /examples/ /scenes/ 的 SPA fallback 精确 rewrite（仅 dev
 * server 有意义，lib 构建不用）。
 */
export function examplesDirectoryIndexPlugin(): Plugin {
  return {
    name: 'examples-directory-index',
    configureServer(server) {
      server.middlewares.use((req, _res, next) => {
        if (req.url === '/examples/' || req.url === '/examples') {
          req.url = '/examples/index.html';
        }
        if (req.url === '/scenes/' || req.url === '/scenes') {
          req.url = '/scenes/index.html';
        }
        next();
      });
    },
  };
}
