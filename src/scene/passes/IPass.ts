/**
 * IPass - 统一的渲染 Pass 抽象（特效层地基，T3.1）。
 *
 * 所有新增特效 pass（动态墙 / 扫描 / 粒子等）必须实现本接口。
 * 既有 pass 渐进迁移，当前状态：
 *
 * | Pass            | 状态                                   |
 * |-----------------|----------------------------------------|
 * | OverlayPass     | 已实现 IPass                            |
 * | GBufferPass     | future — 资源持有者，`beginRenderPass` 由 `Scene.render()` 驱动，无独立 execute；待渲染编排重构后迁移 |
 * | LightingPass    | future — `execute(encoder, outputView)` 已接近签名，但缺 label/prepare/resize，且深度耦合 Scene.render() |
 * | TransparentPass | future — 双子 pass（accumulation + composite）拆分执行，与 Scene.render() 的条件分支耦合 |
 */
export interface IPass {
  /** 调试标签（也用于 RenderPass descriptor 的 label）。 */
  readonly label: string;

  /** 幂等初始化（管线 / bindGroup）。可安全重复调用。 */
  prepare(): void;

  /**
   * 执行单个 pass。
   * @param encoder - 当前帧共享的 command encoder
   * @param target  - 本 pass 的颜色输出（通常为 swap-chain view）
   * @param time    - 统一效果时间（秒，performance.now() / 1000），供动画特效使用
   */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void;

  /** 画布尺寸变化时调用（可选）。 */
  resize?(width: number, height: number): void;

  /** 释放 GPU 资源。调用后本 pass 不可再用。 */
  destroy(): void;
}
