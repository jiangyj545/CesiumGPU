/**
 * 进程级 23-bit 拾取实例 id 分配器（P1-E 打回修复）。
 *
 * PickShader 把实例拾取像素编码为 `0x800000 | instanceId`，但 id 通道只
 * 有低 23 位可用——两个实例生产方（F12 `CityBlockBuilder` 建筑序号、
 * P1-E `TilesetPickRegistry` b3dm 批次）此前各自从 1 起分配，同一 id 会
 * 同时属于「3 号楼」和「某瓦片 batch 2」，读回后互相串扰。
 *
 * 本模块提供唯一的进程级单调分配器：两系统（及未来任何实例拾取生产方）
 * 经 {@link allocatePickInstanceIds} 领取互不相交的区间，解码端
 * （PickPass / SensorCam）无需改动即可正确路由。
 */
const MAX_INSTANCE_ID = 0x7fffff; // 低 23 位（bit23 = 实例标记位）

let nextInstanceId = 1;

/**
 * 领取 `count` 个连续实例 id，返回区间基址（区间 = base .. base+count-1）。
 * 23-bit 空间耗尽时抛错（调用方场景不可能：≈838 万实例）。
 */
export function allocatePickInstanceIds(count: number): number {
  if (!Number.isInteger(count) || count < 0) {
    throw new Error(`allocatePickInstanceIds: 非法数量 ${count}`);
  }
  if (count === 0) return nextInstanceId;
  const base = nextInstanceId;
  if (base + count - 1 > MAX_INSTANCE_ID) {
    throw new Error(
      `allocatePickInstanceIds: 23-bit 实例 id 空间耗尽（base ${base}, count ${count}）`
    );
  }
  nextInstanceId = base + count;
  return base;
}

/** 下一个未分配 id（诊断 / 测试）。 */
export function peekNextPickInstanceId(): number {
  return nextInstanceId;
}
