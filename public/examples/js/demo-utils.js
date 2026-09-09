/**
 * CesiumGPU 示例公共工具（demo-utils.js）
 * ============================================================
 * 所有示例 HTML 的公共依赖（对齐 EV-Globe demo 公共脚本模式）。
 * 本文件为普通脚本（非 module），挂载 window.DemoUtils，
 * 自身【不 import 引擎】——由各示例 HTML 以 ES module 引入引擎后传入，
 * 因此本文件与引擎内部实现完全解耦，可放心作为稳定依赖。
 *
 * ── 示例 HTML 标准骨架 ────────────────────────────────────────
 * <!DOCTYPE html>
 * <html lang="zh-CN">
 * <head>
 *   <meta charset="UTF-8" />
 *   <meta name="viewport" content="width=device-width, initial-scale=1.0" />
 *   <title>示例：xxx</title>
 * </head>
 * <body>
 *   <div id="cesiumContainer"></div>
 *   <script src="./js/demo-utils.js"></script>
 *   <script type="module">
 *     import * as engine from '/src/index.ts';
 *
 *     async function demo(viewer, engine) {
 *       // ── 示例核心代码写在这里 ──
 *     }
 *
 *     const { viewer } = await DemoUtils.startup({
 *       engine,                                   // 必填: import * as engine
 *       view: [116.39, 39.9, 1_500_000],          // 可选: 初始视角 [经度, 纬度, 高度m]
 *       imagery: 'gaode-sat',                     // 可选: 'gaode-sat'|'gaode-vec'|'osm'|'none'
 *       title: 'XYZ 瓦片图层',                     // 可选: 右下角示例名徽标
 *     });
 *     await demo(viewer, engine);
 *     DemoUtils.showCode(demo, '示例源码');         // 可选: 展示本函数源码（高亮）
 *   </script>
 * </body>
 * </html>
 *
 * ── API 一览 ─────────────────────────────────────────────────
 * DemoUtils.startup(options) -> Promise<{ viewer, engine }>
 * DemoUtils.showCode(fnOrString, title?)      代码面板（toString + 语法高亮）
 * DemoUtils.showError(message, err?)          错误面板（红框，可关闭）
 * DemoUtils.showBadge(text)                   右下角徽标（startup 的 title 用它实现）
 * DemoUtils.toast(text)                       顶部轻提示
 * DemoUtils.createFPS() -> { stop() }         FPS HUD（requestAnimationFrame 计数）
 * DemoUtils.highlight(code) -> html           JS 语法高亮（门户与代码面板共用）
 * DemoUtils.IMAGERY_PRESETS                   底图预设参数表
 * DemoUtils.DEFAULT_VIEW                      默认初始视角 [105, 36, 12_000_000]
 */
(function (global) {
  'use strict';

  /** 默认初始视角：中国全境概览 [经度, 纬度, 高度(米)] */
  var DEFAULT_VIEW = [105, 36, 12000000];

  /** 影像图层预设（与 src/spikes/*.ts 中已验证的 provider 参数一致） */
  var IMAGERY_PRESETS = {
    // 高德卫星影像
    'gaode-sat': {
      url: 'https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
      subdomains: ['1', '2', '3', '4'],
      maximumLevel: 18,
      tileSize: 256,
      // 80.7°N 以北是 (7,6,40) 黑填充（L2 顶行 43~104 行，L3 部分整张黑）——
      // 顶行从 78°N 行开始采样（北冰洋浮冰），黑带不再进入画面。
      polarSampleClampNorth: 78,
    },
    // 高德矢量地图
    'gaode-vec': {
      url: 'https://webst0{s}.is.autonavi.com/appmaptile?style=7&x={x}&y={y}&z={z}',
      subdomains: ['1', '2', '3', '4'],
      maximumLevel: 18,
      tileSize: 256,
      polarSampleClampNorth: 78,
    },
    // OpenStreetMap
    osm: {
      url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      subdomains: ['a', 'b', 'c'],
      maximumLevel: 19,
      tileSize: 256,
    },
  };

  // ── 内部：样式注入（幂等） ─────────────────────────────────────
  var CSS_TEXT = [
    // 页面骨架：示例可独立打开（不经门户 iframe），必须自持全屏布局，
    // 否则 #cesiumContainer 高度塌陷 → canvas 半屏、SSE 视口因子减半、LOD 细分不足。
    'html,body{width:100%;height:100%;margin:0;overflow:hidden;background:#05080d;}',
    '#cesiumContainer{width:100%;height:100%;position:relative;}',

    '.cg-fps{position:absolute;top:12px;right:12px;z-index:9000;',
    'color:#0f0;font:12px/1.4 Consolas,Monaco,monospace;background:rgba(0,0,0,.6);',
    'padding:6px 10px;border-radius:6px;pointer-events:none;}',

    '.cg-badge{position:absolute;bottom:12px;right:12px;z-index:9000;',
    'color:#8ecafc;font:12px/1.6 Consolas,Monaco,monospace;background:rgba(0,0,0,.7);',
    'border:1px solid rgba(255,255,255,.15);border-radius:6px;padding:6px 12px;',
    'pointer-events:none;text-align:right;white-space:pre;}',

    '.cg-error{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);',
    'z-index:9100;max-width:70%;max-height:70%;overflow:auto;background:rgba(30,4,8,.92);',
    'border:1px solid #e5484d;border-radius:8px;padding:16px 20px;',
    'color:#ffb4b6;font:13px/1.7 Consolas,Monaco,monospace;white-space:pre-wrap;}',
    '.cg-error h3{margin:0 0 8px;color:#ff6369;font-size:15px;}',
    '.cg-error button{margin-top:10px;background:rgba(255,255,255,.1);color:#fff;',
    'border:1px solid rgba(255,255,255,.3);border-radius:4px;padding:4px 14px;cursor:pointer;}',

    '.cg-code{position:fixed;top:12px;right:60px;z-index:9050;width:520px;max-width:calc(100% - 80px);',
    'max-height:calc(100% - 24px);display:flex;flex-direction:column;',
    'background:rgba(8,12,18,.94);border:1px solid rgba(92,207,255,.35);border-radius:8px;',
    'box-shadow:0 8px 32px rgba(0,0,0,.6);overflow:hidden;}',
    '.cg-code.hidden{display:none;}',
    '.cg-code-head{display:flex;align-items:center;gap:8px;padding:8px 12px;',
    'background:rgba(92,207,255,.12);border-bottom:1px solid rgba(92,207,255,.25);',
    'color:#8ee6ff;font:13px/1.4 system-ui,sans-serif;flex-shrink:0;}',
    '.cg-code-head .cg-code-title{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}',
    '.cg-code-head button{background:rgba(255,255,255,.08);color:#cde;',
    'border:1px solid rgba(255,255,255,.2);border-radius:4px;padding:3px 10px;',
    'font:12px/1.4 system-ui,sans-serif;cursor:pointer;flex-shrink:0;}',
    '.cg-code-head button:hover{background:rgba(255,255,255,.18);}',
    '.cg-code pre{margin:0;overflow:auto;padding:12px 14px;flex:1;}',
    '.cg-code code{font:12px/1.65 Consolas,Monaco,monospace;color:#d8e0e8;}',
    '.tok-kw{color:#5cd8ff;}.tok-str{color:#ffb86c;}',
    '.tok-num{color:#c792ea;}.tok-com{color:#5f7a5f;font-style:italic;}',

    '.cg-toast{position:fixed;top:16px;left:50%;transform:translateX(-50%);z-index:9200;',
    'background:rgba(8,12,18,.95);border:1px solid rgba(92,207,255,.4);color:#8ee6ff;',
    'font:13px/1.4 system-ui,sans-serif;padding:8px 18px;border-radius:6px;',
    'box-shadow:0 4px 16px rgba(0,0,0,.5);}',
  ].join('');

  var styleInjected = false;
  function injectStyleOnce() {
    if (styleInjected) return;
    styleInjected = true;
    var style = document.createElement('style');
    style.id = 'cg-demo-utils-style';
    style.textContent = CSS_TEXT;
    document.head.appendChild(style);
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  // ── JS 语法高亮（单遍 tokenizer，门户页也使用同一套配色） ───────
  var KEYWORDS =
    'const|let|var|function|return|if|else|for|while|of|in|new|class|extends|super|' +
    'async|await|import|from|export|default|try|catch|finally|throw|typeof|instanceof|' +
    'this|null|undefined|true|false|break|continue|switch|case|do|delete|void|yield|static';
  var TOKEN_RE = new RegExp(
    '(\\/\\/[^\\n]*|\\/\\*[\\s\\S]*?\\*\\/)' +                                        // 1 注释
      "|('(?:\\\\.|[^'\\\\\\n])*'|\"(?:\\\\.|[^\"\\\\\\n])*\"|`(?:\\\\.|[^`\\\\])*`)" + // 2 字符串
      '|(\\b\\d+(?:\\.\\d+)?(?:_\\d+)*(?:e[+-]?\\d+)?\\b)' +                          // 3 数字
      '|\\b(' + KEYWORDS + ')\\b',                                                    // 4 关键字
    'g'
  );

  /**
   * 高亮一段 JS 源码，返回 HTML 片段（输入先做 HTML 转义）。
   * @param {string} code
   * @returns {string} html
   */
  function highlight(code) {
    var out = '';
    var last = 0;
    var m;
    TOKEN_RE.lastIndex = 0;
    while ((m = TOKEN_RE.exec(code)) !== null) {
      out += escapeHtml(code.slice(last, m.index));
      if (m[1]) out += '<span class="tok-com">' + escapeHtml(m[1]) + '</span>';
      else if (m[2]) out += '<span class="tok-str">' + escapeHtml(m[2]) + '</span>';
      else if (m[3]) out += '<span class="tok-num">' + escapeHtml(m[3]) + '</span>';
      else out += '<span class="tok-kw">' + escapeHtml(m[4]) + '</span>';
      last = m.index + m[0].length;
    }
    out += escapeHtml(code.slice(last));
    return out;
  }

  // ── FPS HUD ────────────────────────────────────────────────────
  /**
   * 创建 FPS 统计悬浮块（top-right，requestAnimationFrame 独立计数，与渲染循环解耦）。
   * @returns {{ el: HTMLElement, stop: function(): void }}
   */
  function createFPS() {
    injectStyleOnce();
    var el = document.createElement('div');
    el.className = 'cg-fps';
    el.textContent = '-- FPS';
    document.body.appendChild(el);

    var frames = 0;
    var last = performance.now();
    var rafId = 0;
    function loop(now) {
      frames++;
      if (now - last >= 1000) {
        el.textContent = Math.round((frames * 1000) / (now - last)) + ' FPS';
        frames = 0;
        last = now;
      }
      rafId = global.requestAnimationFrame(loop);
    }
    rafId = global.requestAnimationFrame(loop);
    return {
      el: el,
      stop: function () {
        global.cancelAnimationFrame(rafId);
        el.remove();
      },
    };
  }

  // ── 错误面板 ───────────────────────────────────────────────────
  /**
   * 显示错误面板（居中，红框，可关闭）。任何初始化/运行期错误都应走这里；
   * startup 同时注册了 window error / unhandledrejection 兜底。
   * @param {string} message
   * @param {unknown} [err]
   */
  function showError(message, err) {
    injectStyleOnce();
    var detail = err
      ? (err instanceof Error && err.stack ? String(err.stack) : String(err))
      : '';
    var box = document.createElement('div');
    box.className = 'cg-error';
    var h = document.createElement('h3');
    h.textContent = '示例运行出错';
    var pre = document.createElement('div');
    pre.textContent = String(message) + (detail ? '\n\n' + detail : '');
    var btn = document.createElement('button');
    btn.textContent = '关闭';
    btn.addEventListener('click', function () { box.remove(); });
    box.appendChild(h);
    box.appendChild(pre);
    box.appendChild(btn);
    document.body.appendChild(box);
    console.error('[DemoUtils] ' + message, err || '');
  }

  // ── 右下角示例名徽标 ───────────────────────────────────────────
  /**
   * 右下角显示示例名/说明徽标（同一时刻只保留一个）。
   * @param {string} text 支持多行（用 \n 分隔）
   * @returns {HTMLElement}
   */
  function showBadge(text) {
    injectStyleOnce();
    var old = document.querySelector('.cg-badge');
    if (old) old.remove();
    var el = document.createElement('div');
    el.className = 'cg-badge';
    el.textContent = String(text);
    document.body.appendChild(el);
    return el;
  }

  // ── 轻量 toast ────────────────────────────────────────────────
  var toastTimer = 0;
  /**
   * 顶部居中轻提示，2.4s 自动消失。
   * @param {string} text
   */
  function toast(text) {
    injectStyleOnce();
    var old = document.querySelector('.cg-toast');
    if (old) old.remove();
    global.clearTimeout(toastTimer);
    var el = document.createElement('div');
    el.className = 'cg-toast';
    el.textContent = String(text);
    document.body.appendChild(el);
    toastTimer = global.setTimeout(function () { el.remove(); }, 2400);
  }

  // ── 代码面板 ───────────────────────────────────────────────────
  /**
   * 展示示例源码面板（语法高亮 + 复制按钮）。
   * 同一时刻只显示一个面板，再次调用会更新内容。
   * @param {Function|string} fnOrString 示例函数（toString 后展示）或源码字符串
   * @param {string} [title] 面板标题
   */
  function showCode(fnOrString, title) {
    injectStyleOnce();
    var source = typeof fnOrString === 'function'
      ? fnOrString.toString()
      : String(fnOrString);

    var panel = document.querySelector('.cg-code');
    if (!panel) {
      panel = document.createElement('div');
      panel.className = 'cg-code hidden';
      panel.innerHTML =
        '<div class="cg-code-head">' +
        '  <span class="cg-code-title"></span>' +
        '  <button data-act="copy">复制</button>' +
        '  <button data-act="close">&#10005;</button>' +
        '</div>' +
        '<pre><code></code></pre>';
      document.body.appendChild(panel);
      panel.querySelector('[data-act="close"]').addEventListener('click', function () {
        panel.classList.add('hidden');
      });
      panel.querySelector('[data-act="copy"]').addEventListener('click', function () {
        var code = panel.querySelector('code').textContent;
        navigator.clipboard.writeText(code).then(
          function () { toast('源码已复制到剪贴板'); },
          function (err) {
            toast('复制失败，请手动选择复制');
            console.error('[DemoUtils] copy failed', err);
          }
        );
      });
    }

    panel.querySelector('.cg-code-title').textContent = title || '示例源码';
    panel.querySelector('code').innerHTML = highlight(source);
    panel.classList.remove('hidden');
    panel.querySelector('pre').scrollTop = 0;
  }

  // ── startup：示例标准初始化模板 ────────────────────────────────
  /**
   * 示例引擎初始化模板：创建 Viewer → 加载底图 → 设置初始视角 → FPS HUD →
   * 全局错误兜底。示例只需关心自己的核心逻辑。
   *
   * @param {object} [options]
   * @param {object} options.engine        必填：`import * as engine from '/src/index.ts'` 的命名空间
   * @param {string|HTMLElement} [options.container='#cesiumContainer'] viewer 容器
   * @param {'gaode-sat'|'gaode-vec'|'osm'|'none'} [options.imagery='gaode-sat'] 底图预设
   * @param {number[]|null} [options.view=DEFAULT_VIEW] 初始视角 [经度,纬度,高度m]，传 null 跳过
   * @param {boolean} [options.fps=true] 是否注入 FPS HUD
   * @param {string} [options.title] 右下角示例名徽标
   * @returns {Promise<{viewer: object, engine: object}>}
   * @throws 初始化失败时显示错误面板并向上抛出原始错误
   */
  async function startup(options) {
    var opts = options || {};
    var engine = opts.engine;
    if (!engine || typeof engine.Viewer !== 'function') {
      throw new Error(
        'DemoUtils.startup: 缺少 engine 参数。请在示例 HTML 中使用 ' +
        "`import * as engine from '/src/index.ts'` 并传入 { engine }。"
      );
    }
    injectStyleOnce();
    if (!startup._errorHooksInstalled) {
      startup._errorHooksInstalled = true;
      // 已知良性浏览器噪声：ResizeObserver 回调改布局后 Chrome 在下一帧投递的
      // 提示性报错（引擎 resize canvas 时必然发生），不是示例故障，不弹错误面板。
      var BENIGN_ERROR_RE = /ResizeObserver loop|viewport size.*mismatch/i;
      global.addEventListener('error', function (ev) {
        if (BENIGN_ERROR_RE.test(ev.message || '')) return;
        showError(ev.message || '未知错误', ev.error);
      });
      global.addEventListener('unhandledrejection', function (ev) {
        var r = ev.reason;
        var msg = (r && (r.message || r.toString && r.toString())) || '';
        if (BENIGN_ERROR_RE.test(msg)) return;
        showError('未处理的 Promise 拒绝', r instanceof Error ? r : new Error(String(r)));
      });
    }

    var container = resolveContainer(opts.container);
    var viewer;
    try {
      viewer = new engine.Viewer(container);
      await viewer.initialize();
    } catch (err) {
      showError('引擎初始化失败：需要支持 WebGPU 的浏览器（Chrome/Edge 113+）', err);
      throw err;
    }
    global.viewer = viewer; // console 调试钩子

    // 底图
    var imagery = opts.imagery === undefined ? 'gaode-sat' : opts.imagery;
    if (imagery !== 'none') {
      var preset = IMAGERY_PRESETS[imagery] || IMAGERY_PRESETS['gaode-sat'];
      if (typeof engine.XyzImageryProvider !== 'function') {
        console.error('[DemoUtils] engine.XyzImageryProvider 缺失，跳过底图加载');
      } else if (!viewer.globe || !viewer.globe.layers) {
        console.error('[DemoUtils] viewer.globe.layers 缺失，跳过底图加载');
      } else {
        viewer.globe.layers.addImageryProvider(new engine.XyzImageryProvider(preset));
      }
    }

    // 初始视角
    var view = opts.view === undefined ? DEFAULT_VIEW : opts.view;
    if (view && engine.Cartesian3 && typeof engine.Cartesian3.fromDegrees === 'function') {
      viewer.camera.setView({
        destination: engine.Cartesian3.fromDegrees(view[0], view[1], view[2]),
      });
    }

    if (opts.fps !== false) createFPS();
    if (opts.title) showBadge(opts.title);

    return { viewer: viewer, engine: engine };
  }

  /**
   * 解析容器参数：元素 / id 字符串 / 选择器。
   * @param {string|HTMLElement} [sel]
   * @returns {HTMLElement}
   */
  function resolveContainer(sel) {
    var el = sel || '#cesiumContainer';
    if (typeof el === 'string') {
      el = document.getElementById(el) || document.querySelector(el);
    }
    if (!el || !el.appendChild) {
      throw new Error('DemoUtils.startup: 未找到容器元素 ' +
        (typeof sel === 'string' ? sel : '#cesiumContainer'));
    }
    return el;
  }

  // ── 导出 ───────────────────────────────────────────────────────
  global.DemoUtils = {
    startup: startup,
    showCode: showCode,
    showError: showError,
    showBadge: showBadge,
    toast: toast,
    createFPS: createFPS,
    highlight: highlight,
    IMAGERY_PRESETS: IMAGERY_PRESETS,
    DEFAULT_VIEW: DEFAULT_VIEW,
  };
})(window);
