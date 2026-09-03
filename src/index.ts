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

// Effects (T3.1)
export { EffectManager } from './effect/EffectManager';
export type { Effect } from './effect/EffectManager';

// Circle scan effect (T3.3)
export { CircleScan } from './effect/CircleScan';
export type { CircleScanOptions } from './effect/CircleScan';

// Dynamic wall effect (T3.2, 对齐 EV_DynamicWall)
export { DynamicWall } from './effect/DynamicWall';
export type { DynamicWallOptions } from './effect/DynamicWall';

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
