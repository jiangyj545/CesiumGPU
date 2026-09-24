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
