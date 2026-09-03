/**
 * CesiumGPU 示例门户分类配置
 *
 * 结构对齐 EV-Globe 示例门户：分类树 {name, thumbnail?, fileName} 三元组驱动
 * 左侧菜单（public/examples/index.html）。
 *
 * - fileName 相对于本目录（/examples/），由 T4.2 逐个补齐对应示例 HTML；
 *   尚未创建的示例在门户中点击会显示"示例编写中"占位。
 * - thumbnail 可选：存在则用作卡片缩略图，缺省时门户自动渲染"编号占位色块"。
 * - 新增示例时按分类追加 items 即可，门户无需改代码。
 */
window.EXAMPLES_CONFIG = {
  categories: [
    {
      name: '影像图层',
      items: [
        { name: 'XYZ 瓦片图层', fileName: '01_imagery.html' },
      ],
    },
    {
      name: '相机与视角',
      items: [
        { name: 'setView 与 flyTo', fileName: '02_camera.html' },
        { name: '路径飞行漫游', fileName: '07_flyRoute.html' },
      ],
    },
    {
      name: '矢量数据',
      items: [
        { name: 'GeoJSON 加载', fileName: '03_geojson.html' },
        { name: '几何体', fileName: '04_geometry.html' },
      ],
    },
    {
      name: '交互',
      items: [
        { name: '绘制 DrawHandler', fileName: '05_draw.html' },
        { name: 'GPU 拾取', fileName: '06_pick.html' },
      ],
    },
    {
      name: '空间分析',
      items: [
        { name: '距离/面积/高度量测', fileName: '10_measure.html' },
      ],
    },
    {
      name: '特效',
      items: [
        { name: '动态墙', fileName: '08_dynamicWall.html' },
        { name: '圆形扫描', fileName: '08_circleScan.html' },
        { name: '粒子系统', fileName: '09_particle.html' },
        { name: '雨雪天气', fileName: '11_weather.html' },
      ],
    },
  ],
};
