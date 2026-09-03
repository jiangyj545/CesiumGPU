/**
 * CesiumGPU 示例演示数据（data/demo-data.js）
 * ============================================================
 * 常用演示数据集合（对齐 EV-Globe EV_DemoData 模式）。
 * 普通脚本，挂载 window.DemoData；所有示例 HTML 可直接引用，
 * 贯彻"demo 数据与代码分离"，避免把坐标硬编码进各示例。
 *
 * ── 结构 ─────────────────────────────────────────────────────
 * DemoData.DEFAULT_VIEW    默认中国全境视角 [lon, lat, height(m)]
 * DemoData.CITIES          { beijing: {name, lon, lat, height}, ... } 键名小写拼音
 * DemoData.CITY_LIST       CITIES 的数组视图 [{key, name, lon, lat, height}, ...]
 * DemoData.CHINA_BOUNDARY  中国边界（简化）GeoJSON FeatureCollection
 *                          （MultiPolygon: 主边界 ~64 点 + 海南 + 台湾，仅供演示）
 * DemoData.FLIGHT_ROUTES   { beijingToSanya, coastal, bohaiLoop } 飞行路径点数组
 *                          每点 { name, lon, lat, height }（height 单位: 米）
 *
 * 注意：坐标为公开的常用城市经纬度；边界为强简化示意轮廓，
 * 不可用于严肃的地理/测绘用途。
 */
(function (global) {
  'use strict';

  /** 默认中国全境概览视角 [经度, 纬度, 高度(米)] */
  var DEFAULT_VIEW = [105, 36, 12000000];

  /** 常用演示城市（WGS84 经纬度，height 单位: 米） */
  var CITIES = {
    beijing:   { name: '北京',     lon: 116.397, lat: 39.909, height: 0 },
    shanghai:  { name: '上海',     lon: 121.474, lat: 31.230, height: 0 },
    guangzhou: { name: '广州',     lon: 113.264, lat: 23.129, height: 0 },
    shenzhen:  { name: '深圳',     lon: 114.058, lat: 22.543, height: 0 },
    chengdu:   { name: '成都',     lon: 104.066, lat: 30.572, height: 0 },
    chongqing: { name: '重庆',     lon: 106.551, lat: 29.563, height: 0 },
    wuhan:     { name: '武汉',     lon: 114.306, lat: 30.593, height: 0 },
    xian:      { name: '西安',     lon: 108.940, lat: 34.341, height: 0 },
    zhengzhou: { name: '郑州',     lon: 113.625, lat: 34.747, height: 0 },
    taiyuan:   { name: '太原',     lon: 112.549, lat: 37.870, height: 0 },
    kunming:   { name: '昆明',     lon: 102.833, lat: 24.880, height: 0 },
    haerbin:   { name: '哈尔滨',   lon: 126.535, lat: 45.803, height: 0 },
    urumqi:    { name: '乌鲁木齐', lon: 87.617,  lat: 43.793, height: 0 },
    lhasa:     { name: '拉萨',     lon: 91.114,  lat: 29.646, height: 0 },
    sanya:     { name: '三亚',     lon: 109.508, lat: 18.247, height: 0 },
  };

  /** CITIES 的数组视图，便于遍历打点/列表展示 */
  var CITY_LIST = Object.keys(CITIES).map(function (key) {
    var c = CITIES[key];
    return { key: key, name: c.name, lon: c.lon, lat: c.lat, height: c.height };
  });

  // ── 中国边界（强简化示意，MultiPolygon: 主轮廓 / 海南 / 台湾） ────
  // 主轮廓 ~64 点，自东北角顺时针；仅供示例渲染演示。
  var CHINA_MAIN_RING = [
    [134.30, 48.40], [131.00, 45.00], [130.60, 42.90], [128.00, 41.50],
    [124.40, 39.90], [121.60, 39.00], [117.70, 39.00], [119.00, 37.20],
    [122.70, 37.40], [119.20, 34.70], [121.90, 31.00], [120.20, 27.00],
    [118.10, 24.50], [113.60, 22.20], [110.40, 21.00], [108.60, 21.60],
    [106.70, 20.90], [105.30, 23.30], [103.90, 22.50], [101.60, 22.40],
    [99.90, 22.00],  [97.80, 23.90],  [97.50, 25.00],  [98.70, 27.50],
    [98.00, 28.20],  [96.40, 28.40],  [95.40, 29.00],  [92.10, 26.90],
    [91.60, 27.60],  [89.10, 27.30],  [88.10, 27.90],  [85.80, 28.20],
    [84.20, 28.90],  [82.10, 30.10],  [81.00, 30.20],  [79.00, 31.40],
    [78.70, 32.60],  [79.20, 33.20],  [78.00, 34.60],  [76.20, 35.50],
    [74.90, 36.90],  [75.00, 38.50],  [73.60, 39.50],  [75.20, 40.50],
    [76.50, 41.00],  [80.20, 42.20],  [82.60, 43.20],  [83.00, 44.90],
    [83.00, 47.20],  [87.30, 49.10],  [90.90, 47.90],  [96.40, 43.00],
    [99.50, 42.60],  [105.00, 41.80], [109.50, 42.40], [113.60, 44.80],
    [116.70, 46.40], [119.90, 46.70], [117.40, 49.60], [119.10, 50.30],
    [120.70, 52.50], [122.40, 53.50], [124.90, 53.30], [127.50, 50.30],
    [130.70, 48.90], [134.30, 48.40],
  ];

  var HAINAN_RING = [
    [110.10, 20.00], [110.80, 20.10], [111.00, 19.60], [110.60, 18.60],
    [110.00, 18.40], [109.00, 18.30], [108.60, 19.30], [109.20, 19.90],
    [110.10, 20.00],
  ];

  var TAIWAN_RING = [
    [121.00, 25.30], [121.90, 25.00], [121.60, 23.50], [120.90, 22.60],
    [120.20, 23.10], [120.10, 23.80], [121.00, 25.30],
  ];

  /**
   * 中国边界（简化）GeoJSON FeatureCollection。
   * coordinates: [ [ [lon, lat], ... ] ]（GeoJSON 标准为 [lon, lat] 顺序）
   */
  var CHINA_BOUNDARY = {
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        properties: { name: '中国边界（简化示意）' },
        geometry: {
          type: 'MultiPolygon',
          coordinates: [CHINA_MAIN_RING, HAINAN_RING, TAIWAN_RING].map(
            function (ring) { return [ring]; }
          ),
        },
      },
    ],
  };

  // ── 飞行路径点数组（height 单位: 米，用于路径飞行/漫游示例） ─────
  /** 北京 → 西安 → 成都 → 昆明 → 三亚 高空航线（中段抬升形成弧线） */
  var ROUTE_BEIJING_TO_SANYA = [
    { name: '北京', lon: CITIES.beijing.lon, lat: CITIES.beijing.lat, height: 5000 },
    { name: '太原', lon: CITIES.taiyuan.lon, lat: CITIES.taiyuan.lat, height: 40000 },
    { name: '西安', lon: CITIES.xian.lon,    lat: CITIES.xian.lat,    height: 80000 },
    { name: '成都', lon: CITIES.chengdu.lon, lat: CITIES.chengdu.lat, height: 90000 },
    { name: '昆明', lon: CITIES.kunming.lon, lat: CITIES.kunming.lat, height: 70000 },
    { name: '三亚', lon: CITIES.sanya.lon,   lat: CITIES.sanya.lat,   height: 5000 },
  ];

  /** 京津冀—长三角—珠三角 沿海经济带航线 */
  var ROUTE_COASTAL = [
    { name: '北京', lon: CITIES.beijing.lon,  lat: CITIES.beijing.lat,  height: 10000 },
    { name: '济南', lon: 117.120,             lat: 36.651,              height: 45000 },
    { name: '南京', lon: 118.797,             lat: 32.060,              height: 60000 },
    { name: '上海', lon: CITIES.shanghai.lon, lat: CITIES.shanghai.lat, height: 40000 },
    { name: '杭州', lon: 120.155,             lat: 30.274,              height: 55000 },
    { name: '福州', lon: 119.297,             lat: 26.074,              height: 60000 },
    { name: '深圳', lon: CITIES.shenzhen.lon, lat: CITIES.shenzhen.lat, height: 10000 },
  ];

  /** 环渤海封闭航线（首尾相同，适合循环漫游） */
  var ROUTE_BOHAI_LOOP = [
    { name: '天津',   lon: 117.200, lat: 39.084, height: 30000 },
    { name: '东营',   lon: 118.675, lat: 37.434, height: 30000 },
    { name: '烟台',   lon: 121.391, lat: 37.539, height: 30000 },
    { name: '大连',   lon: 121.615, lat: 38.914, height: 30000 },
    { name: '秦皇岛', lon: 119.600, lat: 39.935, height: 30000 },
    { name: '天津',   lon: 117.200, lat: 39.084, height: 30000 },
  ];

  var FLIGHT_ROUTES = {
    beijingToSanya: ROUTE_BEIJING_TO_SANYA,
    coastal: ROUTE_COASTAL,
    bohaiLoop: ROUTE_BOHAI_LOOP,
  };

  // ── 导出 ───────────────────────────────────────────────────────
  global.DemoData = {
    DEFAULT_VIEW: DEFAULT_VIEW,
    CITIES: CITIES,
    CITY_LIST: CITY_LIST,
    CHINA_BOUNDARY: CHINA_BOUNDARY,
    FLIGHT_ROUTES: FLIGHT_ROUTES,
  };
})(window);
