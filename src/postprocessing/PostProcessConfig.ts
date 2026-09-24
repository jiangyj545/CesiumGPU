/**
 * PostProcessConfig - 后处理配置类型 + 纯函数校验/夹取（P2）。
 *
 * 纯逻辑无 GPU 依赖 —— bloomConfig.spec.ts 直接驱动本文件。
 * 约定：非法值（NaN / 负数 / 越界）不抛异常打崩渲染循环，而是
 * 夹取到合法域并返回告警列表（validatePostProcessConfig），调用方
 * （示例 UI / 调试面板）可据此显示"参数已重置"。
 */

export type ToneMappingMode = 'none' | 'reinhard' | 'aces';

export interface BloomParams {
  /** bloom 总开关（独立于后处理 enabled 总开关）。 */
  enabled: boolean;
  /** 亮度阈值（线性域）。高于它的像素才进入 bloom。 */
  threshold: number;
  /** soft-knee 宽度（0 = 硬阈值，1 = 最宽过渡）。 */
  knee: number;
  /** bloom 叠加强度。 */
  intensity: number;
  /** 金字塔级数，夹取到 [4, 6]。 */
  levels: number;
}

export interface LensFlareParams {
  enabled: boolean;
  intensity: number;
  /** ghost 光斑数量，夹取到 [4, 8]。 */
  ghosts: number;
  /** 镜头污渍纹理；null = 使用程序化默认 dirt。 */
  dirtTexture: GPUTexture | null;
}

export interface PostProcessConfig {
  /** 总开关。false 时 Scene 走旧 swapChain 直渲路径（零回归约束）。 */
  enabled: boolean;
  bloom: BloomParams;
  lensFlare: LensFlareParams;
  toneMapping: ToneMappingMode;
}

/** levels 合法域。 */
export const BLOOM_LEVELS_MIN = 4;
export const BLOOM_LEVELS_MAX = 6;

/** 默认配置（enabled=false —— 关闭时全走旧路径）。 */
export function defaultPostProcessConfig(): PostProcessConfig {
  return {
    enabled: false,
    bloom: { enabled: true, threshold: 0.8, knee: 0.5, intensity: 1.0, levels: 5 },
    lensFlare: { enabled: true, intensity: 0.8, ghosts: 5, dirtTexture: null },
    toneMapping: 'aces',
  };
}

function clampNum(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/**
 * 夹取 bloom 参数到合法域（就地修改并返回）。
 * NaN 归到域下界 —— NaN 会穿透 min/max 比较，必须显式处理。
 */
export function sanitizeBloomParams(bloom: BloomParams): BloomParams {
  bloom.threshold = Number.isFinite(bloom.threshold)
    ? clampNum(bloom.threshold, 0, 4)
    : 0;
  bloom.knee = Number.isFinite(bloom.knee) ? clampNum(bloom.knee, 0, 1) : 0;
  bloom.intensity = Number.isFinite(bloom.intensity)
    ? clampNum(bloom.intensity, 0, 10)
    : 0;
  bloom.levels = Number.isFinite(bloom.levels)
    ? Math.round(clampNum(bloom.levels, BLOOM_LEVELS_MIN, BLOOM_LEVELS_MAX))
    : BLOOM_LEVELS_MIN;
  return bloom;
}

/** ghost 数量夹取到 [4, 8]。 */
export function sanitizeGhostCount(ghosts: number): number {
  return Number.isFinite(ghosts)
    ? Math.round(clampNum(ghosts, 4, 8))
    : 4;
}

/**
 * 校验配置，返回告警信息列表（空数组 = 合法）。
 * 拒绝项：NaN / 负值 / levels 越界 / 未知 toneMapping。
 */
export function validatePostProcessConfig(cfg: PostProcessConfig): string[] {
  const problems: string[] = [];
  if (!Number.isFinite(cfg.bloom.threshold) || cfg.bloom.threshold < 0) {
    problems.push(`bloom.threshold 非法: ${cfg.bloom.threshold}`);
  }
  if (!Number.isFinite(cfg.bloom.knee) || cfg.bloom.knee < 0 || cfg.bloom.knee > 1) {
    problems.push(`bloom.knee 非法: ${cfg.bloom.knee}`);
  }
  if (!Number.isFinite(cfg.bloom.intensity) || cfg.bloom.intensity < 0) {
    problems.push(`bloom.intensity 非法: ${cfg.bloom.intensity}`);
  }
  if (!Number.isFinite(cfg.bloom.levels) ||
      cfg.bloom.levels < BLOOM_LEVELS_MIN || cfg.bloom.levels > BLOOM_LEVELS_MAX) {
    problems.push(
      `bloom.levels 越界: ${cfg.bloom.levels}（合法域 [${BLOOM_LEVELS_MIN}, ${BLOOM_LEVELS_MAX}]）`,
    );
  }
  if (!Number.isFinite(cfg.lensFlare.intensity) || cfg.lensFlare.intensity < 0) {
    problems.push(`lensFlare.intensity 非法: ${cfg.lensFlare.intensity}`);
  }
  if (!Number.isFinite(cfg.lensFlare.ghosts) ||
      cfg.lensFlare.ghosts < 4 || cfg.lensFlare.ghosts > 8) {
    problems.push(`lensFlare.ghosts 越界: ${cfg.lensFlare.ghosts}（合法域 [4, 8]）`);
  }
  if (cfg.toneMapping !== 'none' && cfg.toneMapping !== 'reinhard' && cfg.toneMapping !== 'aces') {
    problems.push(`toneMapping 未知: ${String(cfg.toneMapping)}`);
  }
  return problems;
}
