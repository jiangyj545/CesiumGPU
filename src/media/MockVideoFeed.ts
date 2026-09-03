/**
 * MockVideoFeed - Procedural gimbal-camera feed drawn with pure Canvas 2D
 * API (no external resources). Used with {@link CanvasAnimationSource} to
 * exercise the video overlay pipeline without a real video stream, including
 * the degraded-signal path via periodic noise bursts.
 *
 * Deterministic noise: the PRNG is seeded from the burst index
 * `floor(tSec / noiseEverySec)`, so every frame within one burst draws the
 * same noise field — repeatable for screenshots and visual regression.
 */

/** Visual knobs for {@link drawMockFeed}; all optional. */
export interface MockFeedStyle {
  /** Parallax grid lines. Default: 'rgba(110,180,255,0.38)' */
  gridColor?: string;
  /** Reticle / frame-bracket color. Default: '#38e07b' */
  accentColor?: string;
  /** Top-left overlay text. Default: 'MOCK CAM 01' */
  label?: string;
  /** Noise-burst period in seconds. Default: 3 */
  noiseEverySec?: number;
}

/**
 * Draw one simulated camera frame at time `tSec` into `ctx` at `w` x `h`.
 * Pure function of (ctx, w, h, tSec, style) — safe to call at any rate.
 */
export function drawMockFeed(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  tSec: number,
  style: MockFeedStyle = {},
): void {
  if (w <= 0 || h <= 0) return;

  const gridColor   = style.gridColor   ?? 'rgba(110,180,255,0.38)';
  const accentColor = style.accentColor ?? '#38e07b';
  const label       = style.label       ?? 'MOCK CAM 01';
  const noisePeriod = Math.max(0.1, style.noiseEverySec ?? 3);
  const unit     = Math.min(w, h);
  const fontSize = Math.max(10, Math.round(unit * 0.045));

  // ── Base: dark-gray vertical gradient ──────────────────────────────────────
  const base = ctx.createLinearGradient(0, 0, 0, h);
  base.addColorStop(0, '#16181d');
  base.addColorStop(1, '#2b2f36');
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, w, h);

  // ── Scrolling perspective grid (ground plane below the horizon) ───────────
  const horizonY = h * 0.42;
  ctx.save();
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = Math.max(1, unit * 0.004);

  // Ground "verticals" fan out from a vanishing point; `frac` scrolls them
  // horizontally with time (motion parallax).
  const spacing = w / 12;
  const frac    = (tSec * 0.35) % 1; // one lane per spacing unit
  const vx = w / 2;
  for (let i = -14; i <= 14; i++) {
    const bottomX = vx + (i + frac) * spacing * 2.2;
    ctx.beginPath();
    ctx.moveTo(vx, horizonY);
    ctx.lineTo(bottomX, h);
    ctx.stroke();
  }
  // Ground "horizontals" sweep toward the viewer (quadratic spacing).
  const rows  = 7;
  const phase = (tSec * 0.45) % (1 / rows);
  for (let i = 0; i <= rows; i++) {
    const u = Math.min(1, i / rows + phase);
    const y = horizonY + (h - horizonY) * u * u;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
  // Horizon line.
  ctx.beginPath();
  ctx.moveTo(0, horizonY);
  ctx.lineTo(w, horizonY);
  ctx.stroke();
  ctx.restore();

  // ── Center reticle + corner brackets ──────────────────────────────────────
  const cx = w / 2;
  const cy = h / 2;
  const arm = unit * 0.07;
  const gap = unit * 0.02;
  ctx.save();
  ctx.strokeStyle = accentColor;
  ctx.lineWidth = Math.max(1, unit * 0.005);
  ctx.beginPath();
  ctx.moveTo(cx - arm - gap, cy); ctx.lineTo(cx - gap, cy);
  ctx.moveTo(cx + gap, cy);       ctx.lineTo(cx + arm + gap, cy);
  ctx.moveTo(cx, cy - arm - gap); ctx.lineTo(cx, cy - gap);
  ctx.moveTo(cx, cy + gap);       ctx.lineTo(cx, cy + arm + gap);
  ctx.stroke();

  // Corner viewfinder brackets.
  const inset = unit * 0.08;
  const bl    = unit * 0.09; // bracket length
  ctx.beginPath();
  ctx.moveTo(inset, inset + bl);         ctx.lineTo(inset, inset);         ctx.lineTo(inset + bl, inset);
  ctx.moveTo(w - inset - bl, inset);     ctx.lineTo(w - inset, inset);     ctx.lineTo(w - inset, inset + bl);
  ctx.moveTo(inset, h - inset - bl);     ctx.lineTo(inset, h - inset);     ctx.lineTo(inset + bl, h - inset);
  ctx.moveTo(w - inset - bl, h - inset); ctx.lineTo(w - inset, h - inset); ctx.lineTo(w - inset, h - inset - bl);
  ctx.stroke();
  ctx.restore();

  // ── HUD text ──────────────────────────────────────────────────────────────
  ctx.save();
  ctx.font = `${fontSize}px monospace`;
  ctx.fillStyle = 'rgba(255,255,255,0.88)';
  ctx.textBaseline = 'top';

  ctx.fillText(label, inset, inset + bl + fontSize * 0.4);

  // Fake clock HH:MM:SS.mmm synthesised from tSec.
  const totalMs = Math.floor(tSec * 1000);
  const hh = Math.floor(totalMs / 3600000) % 24;
  const mm = Math.floor(totalMs / 60000) % 60;
  const ss = Math.floor(totalMs / 1000) % 60;
  const ms = totalMs % 1000;
  const pad2 = (n: number): string => String(n).padStart(2, '0');
  const stamp = `${pad2(hh)}:${pad2(mm)}:${pad2(ss)}.${String(ms).padStart(3, '0')}`;
  ctx.textAlign = 'right';
  ctx.fillText(stamp, w - inset, inset + bl + fontSize * 0.4);

  // Bottom-center FOV + blinking REC dot (on for the first half of each second).
  ctx.textAlign = 'center';
  const recText = 'FOV 45°  REC';
  const recY = h - inset - fontSize * 1.4;
  ctx.fillText(recText, cx, recY);
  if (tSec % 1 < 0.5) {
    ctx.fillStyle = '#ff3b30';
    const dotR  = fontSize * 0.32;
    const textW = ctx.measureText(recText).width;
    ctx.beginPath();
    ctx.arc(cx + textW / 2 + dotR * 2.2, recY + fontSize * 0.55, dotR, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();

  // ── Degraded-signal noise burst ────────────────────────────────────────────
  const burstPhase = tSec % noisePeriod;
  if (burstPhase < 0.15) {
    const seed = Math.floor(tSec / noisePeriod);
    const rand = mulberry32(seed);
    ctx.save();
    ctx.fillStyle = '#ffffff';
    const count = Math.floor(unit * 1.2);
    for (let i = 0; i < count; i++) {
      const nx = rand() * w;
      const ny = rand() * h;
      const s  = 1 + rand() * 2;
      ctx.globalAlpha = 0.25 + rand() * 0.75;
      ctx.fillRect(nx, ny, s, s);
    }
    ctx.restore();
  }
}

/**
 * Deterministic 32-bit PRNG (mulberry32). Same seed → same sequence, so a
 * noise burst renders identically on every frame of that burst.
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return (): number => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
