import type { IPass } from '../scene/passes/IPass';

/**
 * Effect - 特效的完整形状：一个 IPass 加上可见性开关。
 */
export type Effect = IPass & { show: boolean };

/**
 * EffectManager - 特效统一挂载点（T3.1）。
 *
 * 在 `Scene.render()` 中于 Transparent composite 之后逐个 `execute`：
 * 特效多为叠加发光（emissive / additive），画在最后不被后续 pass 覆盖。
 * 时间统一由 Scene 传入（秒），特效自身不做时钟管理。
 */
export class EffectManager {
  private _effects: Effect[] = [];

  /** 挂载一个特效。不检查重复；同一实例重复 add 会执行两次。 */
  add(effect: Effect): Effect {
    this._effects.push(effect);
    return effect;
  }

  /** 移除一个特效（不 destroy，生命周期归调用方）。 */
  remove(effect: Effect): boolean {
    const idx = this._effects.indexOf(effect);
    if (idx === -1) return false;
    this._effects.splice(idx, 1);
    return true;
  }

  /** 当前挂载的特效列表（只读视图）。 */
  get list(): readonly Effect[] {
    return this._effects;
  }

  /** 在 Transparent composite 之后执行所有可见特效。 */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void {
    for (const effect of this._effects) {
      if (!effect.show) continue;
      effect.execute(encoder, target, time);
    }
  }

  /** 转发画布 resize 给所有支持 resize 的特效。 */
  resize(width: number, height: number): void {
    for (const effect of this._effects) {
      effect.resize?.(width, height);
    }
  }

  /** 销毁所有特效并清空列表。 */
  destroy(): void {
    for (const effect of this._effects) effect.destroy();
    this._effects = [];
  }
}
