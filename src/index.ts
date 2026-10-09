/**
 * CesiumGPU - Public API Entry Point
 *
 * Mirrors Cesium's top-level namespace exports.
 */
export { Viewer } from './core/Viewer';
export { Clock } from './core/Clock';
export { CameraFlightPath } from './scene/CameraFlightPath';
export { ScreenSpaceEventHandler } from './core/ScreenSpaceEventHandler';
export type {
  ScreenSpaceEventType,
  ScreenSpacePosition,
  WheelPosition,
  PinchPosition,
} from './core/ScreenSpaceEventHandler';

// Math
export { CesiumMath } from './math/CesiumMath';
export { Cartesian2 } from './math/Cartesian2';
export { Cartesian3 } from './math/Cartesian3';
export { Cartesian4 } from './math/Cartesian4';
export { Matrix4 } from './math/Matrix4';
export { Transforms } from './math/Transforms';
export { Quaternion } from './math/Quaternion';
export { Ellipsoid } from './math/Ellipsoid';
export { Color } from './math/Color';
export { JulianDate } from './math/JulianDate';
export { Cartographic } from './math/Cartographic';
export { EasingFunction } from './math/EasingFunction';
export { CallbackProperty, resolveProperty, isProperty } from './dataSource/CallbackProperty';
export { HeadingPitchRoll } from './math/HeadingPitchRoll';
export { composeEntityFrameMatrix } from './scene/EntityModelMatrix';
export { CylinderGeometry } from './geometry/CylinderGeometry';
export {
  LabelStyle,
  HorizontalOrigin,
  VerticalOrigin,
  HeightReference,
  billboardAnchorOffset,
  billboardScreenOffset,
  depthTestDisabled,
  applyHeightReference,
} from './dataSource/BillboardStyle';
export type { TerrainHeightSampler } from './dataSource/BillboardStyle';
export { RenderScheduler } from './core/RenderScheduler';
export type { PropertyLike, PropertyOrValue } from './dataSource/CallbackProperty';
export { HermiteSpline, CatmullRomSpline } from './math/Spline';

// Analysis
export { Measure, MeasureCollection } from './analysis/Measure';
export type { MeasureType, MeasureItem } from './analysis/Measure';
// 水深/淹没分析（P1，纯计算不渲染；渲染组合 FloodWater / Entity）
export { FloodAnalysis } from './analysis/FloodAnalysis';
export type { FloodResult, RectangleDeg, FloodAnalysisOptions } from './analysis/FloodAnalysis';

// Draw interaction (T2.5)
export { DrawHandler } from './handler/DrawHandler';
export type { DrawType, DrawEndEvent } from './handler/DrawHandler';

// Scene
export { Scene } from './scene/Scene';
export { Camera } from './scene/Camera';
export { Globe } from './scene/Globe';
export { DirectionalLight } from './scene/DirectionalLight';
export { Primitive, PrimitiveCollection } from './scene/Primitive';
export { EllipsoidGeometry } from './scene/EllipsoidGeometry';
export { HeadingPitchRange } from './scene/HeadingPitchRange';
export { BoundingSphere } from './scene/BoundingSphere';
export { SceneTransforms } from './scene/SceneTransforms';
export type { SceneTransformsScene } from './scene/SceneTransforms';
export { EntityTracker, headingPitchRangeFromCamera } from './scene/EntityTracker';
export { PathHistory, sampleLeadPositions } from './scene/PathHistory';
export type { PathSample } from './scene/PathHistory';
export { composeEntityModelMatrix, resolveMinimumPixelScale } from './scene/EntityModelMatrix';
export { SunPosition } from './scene/SunPosition';

// Geometry (F2b 城市建筑白模批量渲染)
export { ExtrudedGeometry } from './geometry/ExtrudedGeometry';
export type {
  ExtrudedGeometryOptions,
  ExtrudedGeometryBuffers,
  LonLatPoint,
} from './geometry/ExtrudedGeometry';
export { CityBlockBuilder } from './geometry/CityBlockBuilder';
export type {
  CityBlockBuilderOptions,
  CityBlockChunk,
} from './geometry/CityBlockBuilder';

// Model loading
export { Model } from './scene/Model';
export { GltfLoader } from './loader/GltfLoader';
export type { ModelFromGltfOptions } from './scene/Model';
export type { ParsedGltfModel, ParsedMeshPrimitive, ParsedMaterial } from './loader/GltfLoader';

// Imagery (tile streaming)
export { TileKey } from './imagery/TileKey';
export type { TileRectangle } from './imagery/TileKey';
export type { ImageryProvider } from './imagery/ImageryProvider';
export { rotateSubdomain } from './imagery/ImageryProvider';
export { XyzImageryProvider } from './imagery/XyzImageryProvider';
export type { XyzImageryProviderOptions } from './imagery/XyzImageryProvider';
export { ImageryLayer } from './imagery/ImageryLayer';
export type {
  ImageryLayerOptions,
  ImageryLayerStats,
  ResolvedTileTexture,
} from './imagery/ImageryLayer';
export { ImageryLayerCollection } from './imagery/ImageryLayerCollection';

// GPU picking (T2.3)
export { PickPass } from './picking/PickPass';
export type { PickResult } from './picking/PickPass';

// 合成传感器通道（F3 SensorCam）
export {
  SensorCam,
  SensorRecorder,
  PALETTE_MAP_VERSION,
  DEFAULT_PALETTE,
  SEMANTIC,
  linearizeDepthMeters,
  quatFromFLU,
  decodeSegPixel,
  f32ToBase64,
  defaultEncodeUris,
  buildFrameRecord,
} from './sensors/SensorCam';
export type {
  SensorCamParams,
  SensorCamOptions,
  SensorPose,
  SensorFrame,
  SensorFrameMeta,
  PickIdToSemantic,
  FrameRecord,
  EpisodeResult,
  SensorRecorderOptions,
} from './sensors/SensorCam';

// Data sources (Entity / GeoJSON)
export { Event } from './dataSource/Event';
export { Entity } from './dataSource/Entity';
export type {
  EntityOptions,
  PointGraphics,
  PolylineGraphics,
  PolygonGraphics,
  PolygonHierarchy,
  LabelGraphics,
  PathGraphics,
  CylinderGraphics,
} from './dataSource/Entity';
export { EntityCollection } from './dataSource/EntityCollection';
export { GeoJsonDataSource } from './dataSource/GeoJsonDataSource';
export type { GeoJsonLoadOptions } from './dataSource/GeoJsonDataSource';

// Rendering passes (only the extension point is public — concrete passes
// are internal scene wiring reachable via deep paths if ever needed)
export type { IPass } from './scene/passes/IPass';
// 相机 UBO 布局写入器：自定义 IPass 特效（256B 相机 uniform）的公共依赖
export { CAMERA_UBO_SIZE, writeCameraUniforms } from './renderer/CameraUniforms';

// Effects (T3.1)
export { EffectManager } from './effect/EffectManager';
export type { Effect } from './effect/EffectManager';

// Circle scan effect (T3.3)
export { CircleScan } from './effect/CircleScan';
// P2-6 FlowArrow 流动箭头线（polyline.material = { type: 'flow', ... }）
export { FlowArrowPolyline } from './effect/FlowArrowPolyline';
export {
  flowArrowPhase,
  flowArrowMask,
  buildFlowArrowGeometry,
  FLOW_ARROW_SHAPE,
  FLOW_ARROW_DEFAULTS,
} from './effect/FlowArrowMath';
export type { FlowArrowPolylineMaterial } from './effect/FlowArrowMath';
// P2-7 雾 / 闪电
export { FogEffect, LightningEffect } from './effect/WeatherFx';
export {
  computeFogFactor,
  lightningEnvelope,
  sanitizeFogParams,
  sanitizeLightningParams,
} from './effect/WeatherMath';
export type { FogParams, LightningParams } from './effect/WeatherMath';
export type { CircleScanOptions } from './effect/CircleScan';

// Dynamic wall effect (T3.2, 对齐 EV_DynamicWall)
export { DynamicWall } from './effect/DynamicWall';
export type { DynamicWallOptions } from './effect/DynamicWall';

// Dynamic water surface effect (对齐 EV_DynamicWater / EV_Ocean)
export { DynamicWater } from './effect/DynamicWater';
export type { DynamicWaterOptions } from './effect/DynamicWater';

// 水深/淹没水面特效（P1：水位走 uniform 零重建，片元水深色带）
export { FloodWater } from './effect/FloodWater';
export type { FloodWaterOptions, HeightGridData } from './effect/FloodWater';

// FFT 谱海洋特效（移植 tidewater：JONSWAP 谱 + 多级联 IFFT + Jacobian 泡沫）
export { OceanFFTWater, OCEAN_FFT_SIZE } from './effect/OceanFFTWater';
export type { OceanFFTWaterOptions } from './effect/OceanFFTWater';

// Track C 体积云层（全屏云场 + 逐帧云影烘焙：地形 LightingPass 组 3 /
// 海面 OceanFFTWater.bindCloudShadow 双通道消费）
export { CloudLayer } from './effect/CloudLayer';
export type { CloudLayerOptions, CloudShadowInfo } from './effect/CloudLayer';

// B1 L2 五件套集成层（焦散 / 物理大气 / 泡沫图案 / 船迹波场）
export { OceanCaustics } from './effect/OceanCaustics';
export type { OceanCausticsOptions, FFTDataSource } from './effect/OceanCaustics';
export { CausticsFloor } from './effect/CausticsFloor';
export type { CausticsFloorOptions } from './effect/CausticsFloor';
export { AtmosphereSky } from './effect/AtmosphereSky';
export type { AtmosphereSkyOptions } from './effect/AtmosphereSky';
// Track W 岸线冲刷浅水模拟场（IPass；CausticsFloor.bindSwash 消费其 swashView）
export { SwashField } from './effect/SwashField';
export type { SwashFieldOptions } from './effect/SwashField';
export { FoamPatternTexture } from './effect/FoamPatternTexture';
export type { FoamPatternTextureOptions } from './effect/FoamPatternTexture';
export { WakeField } from './effect/WakeField';
export type { WakeFieldOptions, WakeSource } from './effect/WakeField';
// L4 海洋地基：程序化沙洲高度场（岸线泡沫/湿沙线消费）
export { SeabedHeightField } from './effect/SeabedHeightField';
export type { SeabedHeightOptions } from './effect/SeabedHeightField';

// L4 船首浪喷雾（tidewater BoatSpray 直译裁剪版）
export { BowSpray, computeEmitDemand, sprayGate } from './effect/BowSpray';
export type { BowSprayBoatState, BowSprayOptions } from './effect/BowSpray';

// L5/T4 岸线破碎波喷雾（架构照抄 BowSpray：发射源为破碎带 setBand + 海床吸附）
export { BreakSpray, computeEmitDemand as breakSprayEmitDemand } from './effect/BreakSpray';
export type { BreakSprayBand, BreakSprayOptions } from './effect/BreakSpray';

// L3 水下观感全屏叠加（蓝绿深度雾 + 浮游颗粒 + 焦散缓动）
export { UnderwaterFog } from './effect/UnderwaterFog';
export type { UnderwaterFogOptions } from './effect/UnderwaterFog';
// 归一化 ECEF 单一来源：示例层（96 船体定位）与引擎共用同一换算
export { ecefFromLonLatHeight } from './terrain/TerrainProvider';

// Ground heatmap overlay effect (对齐 EV_Heatmap)
export { HeatmapLayer } from './effect/HeatmapLayer';
export type { HeatmapLayerOptions, HeatPoint, HeatGradientStop } from './effect/HeatmapLayer';

// Particle system (T3.4, 对齐 EV_ParticleSystem)
export { ParticleSystem } from './effect/ParticleSystem';
export type {
  ParticleOptions,
  ParticleMode,
  ParticleOrigin,
  ParticlePreset,
  ParticleColorMode,
} from './effect/ParticleSystem';

// Weather effects (T3.5, 对齐 EV_Rain / EV_Snow)
export { Weather } from './effect/Weather';
export type { WeatherType, WeatherOptions, WeatherArea } from './effect/Weather';

// Gimbal video overlay（云台视频流贴地投影，device_osd 遥测驱动）
export { VideoOverlay } from './media/VideoOverlay';
export type { VideoOverlayOptions, VideoOverlayState } from './media/VideoOverlay';
export {
  WGS84_POLAR_RATIO,
  EARTH_SCALE as MEDIA_EARTH_SCALE,
  NDC_TO_UV_Y_SIGN,
  intersectRayEllipsoid,
  computeFootprint,
  buildTextureMatrix,
} from './media/GimbalMath';
export type { GimbalPose, Footprint } from './media/GimbalMath';
export { DeviceOsdParser, lerpGimbal, TelemetryBuffer } from './media/Telemetry';
export type { GimbalState } from './media/Telemetry';
export { MockTelemetrySource, WebSocketTelemetrySource } from './media/TelemetrySource';
export type { TelemetrySource, MockTelemetryOptions } from './media/TelemetrySource';
export {
  HtmlVideoSource,
  CanvasAnimationSource,
  VideoFrameSource,
} from './media/VideoSource';
export type { VideoFrameSurface } from './media/VideoSource';
export { VideoTexture, createVideoSampler } from './media/VideoTexture';
export { drawMockFeed } from './media/MockVideoFeed';
export type { MockFeedStyle } from './media/MockVideoFeed';

// 4K 视频输出（MediaRecorder 画布录制）
export { VideoRecorder } from './media/VideoRecorder';
export type {
  RecordingOptions,
  VideoRecorderHost,
  VideoRecorderState,
} from './media/VideoRecorder';

// Terrain (DEM streaming, Wave1)
export type {
  TerrainProvider,
  TerrainMeshData,
  TerrainTilingScheme,
} from './terrain/TerrainProvider';
export { HttpTerrainProviderBase } from './terrain/TerrainProvider';
export { LocalQuantizedMeshProvider } from './terrain/LocalQuantizedMeshProvider';

// P1-H 地形采样（Cesium sampleTerrainMostDetailed 语义）
export { sampleTerrainMostDetailed, sampleTerrain } from './terrain/sampleTerrain';
// P2-8 LERC (ArcGIS Terrain3D) 地形
export { LercTerrainProvider, DEFAULT_LERC_URL } from './terrain/LercTerrainProvider';
export type { LercTerrainProviderOptions } from './terrain/LercTerrainProvider';
export { decodeLerc1 } from './terrain/LercDecoder';
export type { Lerc1Tile } from './terrain/LercDecoder';
export type {
  LocalQuantizedMeshProviderOptions,
} from './terrain/LocalQuantizedMeshProvider';
export { TerrariumProvider } from './terrain/TerrariumProvider';
export type { TerrariumProviderOptions } from './terrain/TerrariumProvider';

// Terrain streaming + height sampling (Wave2-A)
export { TerrainLayer } from './terrain/TerrainLayer';
export type {
  TerrainLayerOptions,
  TerrainLayerStats,
  TerrainRenderTile,
  TerrainTileBounds,
} from './terrain/TerrainLayer';

// 独立后处理管线（P2：Bloom / LensFlare / HDR ToneMapping）
export { PostProcessManager } from './postprocessing/PostProcessManager';
export type {
  PostProcessConfig,
  ToneMappingMode,
} from './postprocessing/PostProcessConfig';
export { RenderTexture } from './renderer/RenderTexture';
export { acesFilmic, reinhard, toneMapFunction } from './postprocessing/ToneMapping';

// 3DTiles 调度与集成层（P3：树遍历 / SSE / LRU / b3dm / Google 数据源）
export { Tileset3D, computeTileSse, shouldRefine } from './tileset/Tileset3D';
export type { Tileset3DOptions, Tileset3DStats } from './tileset/Tileset3D';
export { GoogleTiles3D } from './tileset/GoogleTilesProvider';
export type { GoogleTiles3DOptions } from './tileset/GoogleTilesProvider';
export {
  fetchTilesetJson,

  parseTileset,
  TILESET_EARTH_SCALE,
} from './tileset/TilesetContentLoader';

// P1-E 3DTiles feature 挑拣（b3dm _BATCHID + batchTable）
export {
  Cesium3DTileFeature,
  TilesetPickRegistry,
  tilesetPickRegistry,
  decodePickInstanceId,
  buildTileInstanceIds,
  PICK_INSTANCE_FLAG,
  PICK_INSTANCE_MASK,
} from './tileset/Cesium3DTileFeature';
export type {
  TileNode,
  TileContent,
  TileRefine,
  TileContentKind,
  TileUrlDecorator,
  LoadTilesetOptions,
} from './tileset/TilesetContentLoader';
export { TilesetTileCache } from './tileset/TilesetTileCache';
export type {
  TileCacheEntry,
  TilesetTileCacheOptions,
  TilesetTileCacheStats,
} from './tileset/TilesetTileCache';
export { loadTileContent, buildTilePrimitives } from './tileset/TilesetPrimitive';
export type { BuiltTileContent, TileContentLoader } from './tileset/TilesetPrimitive';
export {
  extractGlb,
  computeGlbOffset,
  parseFeatureTableJson,
  parseRtcCenter,
} from './tileset/B3dmDecoder';
export type { RtcCenter } from './tileset/B3dmDecoder';

// A1 CSSM 级联阴影（太阳方向平行光 3 级联正交阴影）
export { ShadowMap } from './shadow/ShadowMap';
export type { ShadowReceiverBinding } from './shadow/ShadowMap';
export {
  SHADOW_RECEIVER_UBO_SIZE,
  SHADOW_CASCADE_COUNT,
  computeCascadeSplitEnds,
  computeLightBasis,
  computeCascadeCorners,
  fitLightOrtho,
} from './shadow/ShadowMap';
export { SHADOW_DEPTH_SHADER, SHADOW_RECEIVER_MODULE } from './shaders/ShadowShader';
export {
  LIGHTING_SHADER_SHADOW,
  LIGHTING_SHADER_SHADOW_HDR,
  injectShadowCore,
} from './shaders/LightingShader';
