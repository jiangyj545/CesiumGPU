import type { IPass } from '../scene/passes/IPass';

/**
 * Effect - 特效的完整形状：一个 IPass 加上可见性开关。
 */
export type Effect = IPass & { show: boolean };

/**
 * T5 真 HDR 可选协议：实现 `setTargetFormat` 的特效在渲染目标格式变更时
 * 重建渲染管线（Scene.render 逐帧下发当前目标格式——postProcess.enabled
 * 时为 rgba16float 离屏 RT，否则为交换链 preferredFormat）。未实现的特效
 * （OceanFFTWater / AtmosphereSky 暂未接线）保持 preferredFormat 旧行为
 * 零回归；仿 `IPass.updateCamera?` 的可选方法协议，不进 IPass 接口。
 */
export interface TargetFormatAware {
  setTargetFormat(format: GPUTextureFormat): void;
}

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

  /**
   * 在 Transparent composite 之后执行所有可见特效。
   *
   * @param targetFormat 当前渲染目标的颜色格式（T5 真 HDR，可选）。下发时
   *   逐个调用实现了 {@link TargetFormatAware} 协议的特效（变更即重建管线、
   *   同格式幂等早退由特效自守）；不传（向后兼容）不下发。
   */
  execute(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    time: number,
    targetFormat?: GPUTextureFormat,
  ): void {
    for (const effect of this._effects) {
      if (!effect.show) continue;
      if (targetFormat !== undefined) {
        (effect as Effect & Partial<TargetFormatAware>).setTargetFormat?.(targetFormat);
      }
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
