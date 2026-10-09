/**
 * RenderScheduler – P2-5 requestRenderMode 的按需渲染调度器（纯逻辑，
 * node 可测）。Viewer 每帧 rAF 询问 {@link shouldRender}；脏则渲染并
 * 清脏，净则跳过 Scene.render（省 CPU，对齐 hexapod mapController
 * requestRenderMode + maximumRenderTimeChange=Infinity 的用法）。
 *
 * 脏源由宿主（Viewer）喂数据标记：
 * - 相机 / 场景内容指纹变化 → markDirty（宿主逐帧比对）
 * - 显式 requestRender()（等价 Cesium scene.requestRender）
 * - maximumRenderTimeChange：连续静止超过该秒数后自动补渲一帧
 *  （默认 Infinity = 永不，对齐 hexapod）。
 */
export class RenderScheduler {
  private _dirty = true;
  private _lastActivitySec: number;
  private readonly _maximumRenderTimeChange: number;

  constructor(maximumRenderTimeChange = Number.POSITIVE_INFINITY, nowSec = 0) {
    this._maximumRenderTimeChange = maximumRenderTimeChange;
    this._lastActivitySec = nowSec;
  }

  /** 显式请求渲染下一帧（Cesium scene.requestRender 等价）。 */
  requestRender(): void {
    this._dirty = true;
  }

  /** 任一脏源触发（指纹变化、实体/图元/瓦片增删、Property 更新等）。 */
  markDirty(): void {
    this._dirty = true;
  }

  /** 当前是否已标记脏（诊断 / 测试）。 */
  get isDirty(): boolean {
    return this._dirty;
  }

  /**
   * 帧决策：脏 → 清脏并返回 true（渲染本帧）；净且静止超过
   * maximumRenderTimeChange → 补渲一帧；否则 false（跳过渲染）。
   */
  shouldRender(nowSec: number): boolean {
    if (this._dirty) {
      this._dirty = false;
      this._lastActivitySec = nowSec;
      return true;
    }
    if (
      Number.isFinite(this._maximumRenderTimeChange) &&
      nowSec - this._lastActivitySec >= this._maximumRenderTimeChange
    ) {
      this._lastActivitySec = nowSec;
      return true;
    }
    return false;
  }
}

export default RenderScheduler;
