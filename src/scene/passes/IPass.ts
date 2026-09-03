import type { Camera } from '../Camera';

/**
 * IPass - 统一的渲染 Pass 抽象（特效层地基，T3.1）。
 *
 * 场景管线与特效 pass 均实现本接口；`Scene.render()` 按固定顺序遍历
 * `IPass[]` 编排整帧（gbuffer → lighting → overlay → transparent，特效由
 * EffectManager 追加在最后）。当前实现：
 *
 * | Pass            | 状态                                                                       |
 * |-----------------|----------------------------------------------------------------------------|
 * | GBufferPass     | 已实现 — 资源持有者 + 不透明几何光栅化（draw inputs 由 Scene 构造时注入）     |
 * | LightingPass    | 已实现 — 延迟光照，输出到 swap-chain                                        |
 * | TransparentPass | 已实现 — execute 内封装 WBOIT accumulation + composite 双子 pass             |
 * | OverlayPass     | 已实现 — 实体点/标注 billboard（画在 transparent 之前，WBOIT 叠加在其上）    |
 * | Weather / DynamicWall / CircleScan / ParticleSystem | 已实现（EffectManager 挂载的特效）      |
 */
export interface IPass {
  /** 调试标签（也用于 RenderPass descriptor 的 label）。 */
  readonly label: string;

  /** 幂等初始化（管线 / bindGroup）。可安全重复调用。 */
  prepare(): void;

  /**
   * 每帧相机相关准备（可选）：上传相机 UBO / CPU 侧投影。
   * 由 Scene 在 execute 之前统一调用；不需要相机状态的 pass 可不实现。
   */
  updateCamera?(camera: Camera): void;

  /**
   * 执行单个 pass。
   * @param encoder - 当前帧共享的 command encoder
   * @param target  - 本 pass 的颜色输出（通常为 swap-chain view；自持目标的
   *                  pass 如 GBufferPass 忽略此参数）
   * @param time    - 统一效果时间（秒，performance.now() / 1000），供动画特效使用
   */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void;

  /** 画布尺寸变化时调用（可选）。 */
  resize?(width: number, height: number): void;

  /** 释放 GPU 资源。调用后本 pass 不可再用。 */
  destroy(): void;
}
