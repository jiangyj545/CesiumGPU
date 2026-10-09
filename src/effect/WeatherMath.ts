/**
 * WeatherMath – 雾 / 闪电纯逻辑（P2-7，对齐 weather-fog.js / weather-lightning.js）。
 */

/**
 * 指数距离雾因子（Cesium scene.fog 语义的归一化形式）：
 * `factor = 1 − exp(−density·dist·exp(−heightFalloff·height))`
 *
 * 高度衰减：像素海拔（米）越高雾越薄（逆温层观感）；density ∈ [0,1] 线性
 * 放大到引擎尺度（×3e-5 / 米 ≈ density=1 时 33 km 全遮蔽）。
 * 返回 [0,1)：0 = 完全清晰，→1 = 完全雾化。
 */
export function computeFogFactor(
  distanceMetres: number,
  pixelHeightMetres: number,
  density: number,
  heightFalloff: number
): number {
  const d = clamp01(density) * 3e-5;
  const f = 1 - Math.exp(-d * Math.max(0, distanceMetres) * Math.exp(-Math.max(0, heightFalloff) * pixelHeightMetres));
  return clamp01(f);
}

/**
 * 闪电包络（strike 后 t 秒）：急升（attack 线性）→ 主衰减（τ=decay）→
 * 余辉（afterglow 长尾）。峰值 = 1（调用方乘 intensity）。
 * - t < 0 → 0（未触发）
 * - t ∈ [0, attack) → t/attack
 * - t ≥ attack → 0.75·exp(−(t−attack)/decay) + 0.25·exp(−(t−attack)/afterglow)
 */
export function lightningEnvelope(
  tSinceStrike: number,
  attack = 0.02,
  decay = 0.08,
  afterglow = 0.5
): number {
  if (tSinceStrike < 0) return 0;
  if (attack <= 0) return 0;
  if (tSinceStrike < attack) return tSinceStrike / attack;
  const dt = tSinceStrike - attack;
  return 0.75 * Math.exp(-dt / decay) + 0.25 * Math.exp(-dt / afterglow);
}

/** 雾参数（FogEffect 用；构造时钳制到合法域）。 */
export interface FogParams {
  /** 0..1（0=关）。 */
  density: number;
  /** 高度衰减（1/km；0 = 不衰减）。 */
  heightFalloff: number;
  /** 雾染色（rgba，hexapod fogColor 语义）。 */
  tintColor: [number, number, number, number];
  /** 天空像素（无场景深度）的雾占比（0..1）。 */
  skyTint: number;
}

/** 闪电参数。 */
export interface LightningParams {
  /** 0..1 峰值强度。 */
  intensity: number;
  /** 闪色（hexapod flashColor 蓝白语义）。 */
  color: [number, number, number];
  /** 随机自动序列：平均间隔秒（0 = 仅手动 strike()）。 */
  meanIntervalSec: number;
  /** 双闪概率（0..1）。 */
  doubleStrikeChance: number;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** 钳制雾参数（负密度/衰减钳 0；skyTint 钳 [0,1]）。 */
export function sanitizeFogParams(p: Partial<FogParams>): FogParams {
  return {
    density: clamp01(p.density ?? 0.5),
    heightFalloff: Math.max(0, p.heightFalloff ?? 0.12),
    tintColor: p.tintColor ?? [0.78, 0.82, 0.88, 1],
    skyTint: clamp01(p.skyTint ?? 0.35),
  };
}

/** 钳制闪电参数（强度/概率钳 [0,1]；间隔钳非负）。 */
export function sanitizeLightningParams(p: Partial<LightningParams>): LightningParams {
  return {
    intensity: clamp01(p.intensity ?? 0.8),
    color: p.color ?? [0.85, 0.9, 1.0],
    meanIntervalSec: Math.max(0, p.meanIntervalSec ?? 0),
    doubleStrikeChance: clamp01(p.doubleStrikeChance ?? 0.3),
  };
}
