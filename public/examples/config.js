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
        { name: '多源影像切换', fileName: '15_multiImagery.html' },
        { name: '影像图层组管理', fileName: '16_layerGroup.html' },
        { name: '本地单图贴地', fileName: '17_localImage.html' },
        { name: '卷帘对比', fileName: '60_contrastRoller.html' },
        { name: '双球对比', fileName: '61_contrastGlobe.html' },
        { name: '鹰眼图', fileName: '62_eagleEyeMap.html' },
      ],
    },
    {
      name: '相机与视角',
      items: [
        { name: 'setView 与 flyTo', fileName: '02_camera.html' },
        { name: '路径飞行漫游', fileName: '07_flyRoute.html' },
        { name: '绕点环绕飞行', fileName: '41_aroundPointFly.html' },
        { name: '路径漫游（进度控制）', fileName: '42_flyRoam.html' },
        { name: '街景式地面漫游', fileName: '43_googleRoam.html' },
        { name: '定点旋转', fileName: '44_spotRotation.html' },
        { name: '地下视角', fileName: '45_undergroundCamera.html' },
        { name: '视角历史与书签', fileName: '46_cameraHistory.html' },
      ],
    },
    {
      name: '矢量数据',
      items: [
        { name: 'GeoJSON 加载', fileName: '03_geojson.html' },
        { name: '几何体', fileName: '04_geometry.html' },
        { name: 'KML 数据加载', fileName: '20_loadKml.html' },
        { name: 'CZML 时间轴', fileName: '21_loadCzml.html' },
        { name: 'SHP 矢量加载', fileName: '22_loadShp.html' },
        { name: '点聚合 PointCluster', fileName: '23_pointCluster.html' },
        { name: '大数据聚合', fileName: '24_clusterLargeData.html' },
      ],
    },
    {
      name: '几何与标绘',
      items: [
        { name: '广告牌标注', fileName: '25_billboard.html' },
        { name: '椭圆与圆面', fileName: '26_ellipse.html' },
        { name: '圆柱/方体/锥体', fileName: '27_cylinder.html' },
        { name: '动态墙与走廊', fileName: '28_wall.html' },
        { name: '光晕流动线', fileName: '29_haloLine.html' },
        { name: '虚线与样式线', fileName: '30_dashedLine.html' },
        { name: '球体与管体', fileName: '31_sphere.html' },
        { name: '自定义几何顶点', fileName: '32_customGeometry.html' },
        { name: '雷达探测', fileName: '33_probing.html' },
        { name: '几何顶点编辑', fileName: '34_geometryEdit.html' },
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
        { name: '距离/面积/高度量测', fileName: '11_measure.html' },
        { name: '地形高程剖面', fileName: '19_demProfile.html' },
        { name: '缓冲区分析', fileName: '76_bufferZone.html' },
        { name: '天际线分析', fileName: '77_skyLineAnalysis.html' },
        { name: '可视域分析', fileName: '78_visibleRange.html' },
        { name: '淹没分析', fileName: '79_submergeAnalysis.html' },
        { name: '地形剖面（多点路径）', fileName: '80_terrainProfile.html' },
        { name: '路径分析', fileName: '86_routeAnalysis.html' },
        { name: '水深/淹没模拟', fileName: '92_floodSim.html' },
      ],
    },
    {
      name: '工具',
      items: [
        { name: '鼠标坐标拾取', fileName: '71_currentPosition.html' },
        { name: '导航罗盘/比例尺', fileName: '72_navigation.html' },
        { name: '场景截图', fileName: '73_screenCapture.html' },
        { name: '快捷键', fileName: '74_shortcutKey.html' },
        { name: '场景时间与昼夜', fileName: '75_sceneTime.html' },
      ],
    },
    {
      name: '物理仿真',
      items: [
        { name: '自由落体物理', fileName: '81_ballFall.html' },
        { name: '弹簧弹跳物理', fileName: '82_ballWiggle.html' },
      ],
    },
    {
      name: '标绘',
      items: [
        { name: '3D 雷达锥', fileName: '83_radar3D.html' },
        { name: '雷达融合', fileName: '84_radarFusion.html' },
        { name: '简版军标标绘', fileName: '85_milStdPlot.html' },
      ],
    },
    {
      name: '数据可视化',
      items: [
        { name: 'mapV 迁徙图近似', fileName: '87_mapVMigrate.html' },
        { name: 'echarts 飞线近似', fileName: '88_echartsFlow.html' },
        { name: '独立后处理管线', fileName: '93_postProcess.html' },
      ],
    },
    {
      name: '地形',
      items: [
        { name: '地形与贴地', fileName: '14_terrain.html' },
        { name: '离线 DEM 地形加载', fileName: '18_dem.html' },
        { name: '高程分层设色', fileName: '63_elevation.html' },
        { name: '彩色地形', fileName: '64_colorTerrain.html' },
      ],
    },
    {
      name: '特效',
      items: [
        { name: '动态墙', fileName: '08_dynamicWall.html' },
        { name: '圆形扫描', fileName: '09_circleScan.html' },
        { name: '粒子系统', fileName: '10_particle.html' },
        { name: '雨雪天气', fileName: '12_weather.html' },
        { name: '云台视频叠加', fileName: '13_videoOverlay.html' },
        { name: '动态水面', fileName: '47_dynamicWater.html' },
        { name: '海洋效果', fileName: '48_ocean.html' },
        { name: '流动线', fileName: '49_flowPolyline.html' },
        { name: '重复流线', fileName: '50_repeatFlowLine.html' },
        { name: '航迹线回放', fileName: '51_trackPolyline.html' },
        { name: '航班弧线', fileName: '52_airLines.html' },
        { name: '热力图', fileName: '53_heatmap.html' },
        { name: '3D 热力柱', fileName: '54_heatmap3D.html' },
        { name: '昼夜交替', fileName: '55_dayAndNight.html' },
        { name: '星空', fileName: '56_starry.html' },
        { name: '聚光探照灯', fileName: '57_spotlight.html' },
        { name: '城市光柱', fileName: '58_weiboLight.html' },
        { name: '泛光后处理', fileName: '59_bloom.html' },
        { name: '天气云层', fileName: '67_cloud.html' },
        { name: '云集合', fileName: '68_cloudCollection.html' },
        { name: '体积圆柱', fileName: '69_volumeCylinder.html' },
        { name: '建筑特效', fileName: '70_bulidingEffect.html' },
        { name: 'FFT 海洋仿真', fileName: '96_oceanFFT.html' },
      ],
    },
    {
      name: '球体视图',
      items: [
        { name: '半透明球', fileName: '65_globeTranslucence.html' },
        { name: '球体模式切换', fileName: '66_switchGlobeMode.html' },
      ],
    },
    {
      name: '三维模型',
      items: [
        { name: 'glTF 模型加载', fileName: '35_loadGltf.html' },
        { name: '模型动画', fileName: '36_modelAnimation.html' },
        { name: '模型小场景', fileName: '37_smallScene.html' },
        { name: '点云渲染（近似）', fileName: '38_loadPointCloud.html' },
        { name: '模型剖切（近似）', fileName: '39_planeClip.html' },
        { name: '倾斜摄影加载（占位）', fileName: '40_e3mProxy.html' },
        { name: '3DTiles 瓦片加载', fileName: '94_tileset3d.html' },
        { name: 'Google 摄影实景 3DTiles', fileName: '95_googleTiles.html' },
      ],
    },
  ],
};
