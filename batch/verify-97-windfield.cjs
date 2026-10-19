#!/usr/bin/node
/**
 * verify-97-windfield.cjs —— B2-3 大屏风场层集成探针（自验断言源）。
 *
 * 复用已起服务（vite 3006 / sim-kernel 8895，均后台自起），headless Chrome
 * 打开 97 大屏 ?windfield=1，采集：__DEMO_OK / __DASH 摘要 / __WF 探针 /
 * JS 异常数 / rAF 帧率 / 截图，断言 primCount>0、arrowCount>0、零异常。
 * 风场层不依赖 SSE（城市白模+箭头先渲染，机标随 SSE 帧就位）；无 sim 时
 * sseState=error 不算失败。用法：node batch/verify-97-windfield.cjs
 */
const fs = require('fs');
const path = require('path');
// 复用平台仓已验证的 CDP 封装（只读依赖；cdp.launchChrome 的 stdio:'ignore'
// 会吞 Chrome 154 早退原因，故照 diag-90 内联 spawn）。
const cdp = require('D:/work/仿真/batch/cdp.js');

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const OUT_DIR = path.join(__dirname, '..', '.omc', 'tmp', 'wf97');
const VITE_PORT = process.env.WF97_VITE_PORT || '3006';
const SIM_URL = process.env.WF97_SIM_URL || '';   // 空=不起 SSE（风场层独立可验）
// WF97_NUDE=1 = 裸页回归（不带 windfield 参数，断言风场层默认不加载——零回归红线）
const NUDE = process.env.WF97_NUDE === '1';
const URL_97 =
  `http://localhost:${VITE_PORT}/examples/97_dashboard.html` +
  (NUDE ? '' : '?windfield=1') +
  (SIM_URL ? `&simUrl=${encodeURIComponent(SIM_URL)}` : '');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const port = 9900 + (process.pid % 200);
  // Chrome 154：user-data-dir 含中文（仿真）静默起不来——profile 一律 ASCII 绝对路径
  const profile = `E:\\temp\\wf97-profile-${process.pid}`;
  const { spawn } = require('child_process');
  const proc = spawn(CHROME, [
    '--headless=new', '--enable-unsafe-webgpu', '--no-sandbox', '--disable-gpu-sandbox',
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`,
    '--no-first-run', '--window-size=1600,900', 'about:blank',
  ], { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  let __stderr = '';
  proc.stderr.on('data', (d) => { __stderr += d; });
  proc.on('exit', (c, s) => console.error(`[wf97] chrome EXIT code=${c} signal=${s} stderr-tail=${__stderr.slice(-200)}`));
  await cdp.waitForDevTools(port, 30000);
  const t0 = Date.now();
  const { cdp: page } = await cdp.connect(port, { url: URL_97 });
  await page.send('Runtime.enable').catch(() => {});
  const exceptions = [];
  page.on('Runtime.exceptionThrown', (p) => {
    const d = p.exceptionDetails || {};
    exceptions.push(((d.exception && d.exception.description) || d.text || '').slice(0, 200));
  });
  // 等 __DEMO_OK（大屏就绪）&&（nude 模式跳过）__WF.loaded（风场层入场）
  let ready = false;
  for (let i = 0; i < 80; i++) {
    await sleep(500);
    const r = await page.send('Runtime.evaluate', {
      expression: NUDE
        ? 'window.__DEMO_OK === true'
        : 'window.__DEMO_OK === true && !!(window.__WF && window.__WF.loaded)',
      returnByValue: true,
    }).catch(() => null);
    if (r && r.result && r.result.value === true) { ready = true; break; }
  }
  const readyMs = Date.now() - t0;
  // rAF 帧率采样 5s
  await page.send('Runtime.evaluate', {
    expression: 'window.__fpsN=0;window.__fpsT0=performance.now();(function loop(){window.__fpsN++;requestAnimationFrame(loop);})();',
  }).catch(() => {});
  await sleep(5000);
  const fpsRes = await page.send('Runtime.evaluate', {
    expression: '({n:window.__fpsN,dt:(performance.now()-window.__fpsT0)/1000})',
    returnByValue: true,
  }).catch(() => null);
  const fps = fpsRes && fpsRes.result.value ? (fpsRes.result.value.n / fpsRes.result.value.dt).toFixed(1) : 'n/a';
  // 诊断汇总
  const diag = await page.send('Runtime.evaluate', {
    expression: `JSON.stringify({
      demoOk: window.__DEMO_OK,
      wf: window.__WF || null,
      dash: window.__DASH ? {
        sseState: window.__DASH.sseState, sseMsgCount: window.__DASH.sseMsgCount,
        badFrames: window.__DASH.badFrames, nVehicles: window.__DASH.nVehicles,
        specOk: window.__DASH.specOk,
      } : null,
    })`,
    returnByValue: true,
  }).catch(() => null);
  const shot = await page.send('Page.captureScreenshot', { format: 'png' }).catch(() => null);
  if (shot) fs.writeFileSync(path.join(OUT_DIR, 'wf97.png'), Buffer.from(shot.data, 'base64'));
  const d = diag ? JSON.parse(diag.result.value) : null;
  const wf = (d && d.wf) || {};
  // nude：风场层必须未加载（__WF 保持初始 primCount:0/loaded:false）——默认零回归
  const pass = ready && exceptions.length === 0 &&
    (NUDE
      ? wf.loaded !== true && wf.primCount === 0
      : wf.primCount > 0 && wf.arrowCount > 0);
  console.log(JSON.stringify({
    pass, url: URL_97, readyOk: ready, readyMs, fps5s: fps,
    wf, dash: d ? d.dash : null,
    exceptionCount: exceptions.length, exceptions: exceptions.slice(0, 5),
    shot: shot ? path.join(OUT_DIR, 'wf97.png') : null,
  }, null, 1));
  await cdp.closeBrowser(port).catch(() => {});
  cdp.killProcessTree(proc.pid);
  process.exit(pass ? 0 : 1);
})().catch((e) => { console.error('[wf97] fatal:', e && (e.stack || e.message || e)); process.exit(2); });
