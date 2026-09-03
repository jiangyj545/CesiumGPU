module.exports = {
  root: true,
  env: {
    browser: true,
    es2022: true,
    node: true,
  },
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module',
  },
  plugins: ['@typescript-eslint'],
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
  ],
  ignorePatterns: ['src/spikes/', 'spikes/', 'node_modules/', 'dist/', 'public/'],
  rules: {
    // TS 编译器已负责未定义变量检查（含 @webgpu/types），no-undef 在 TS 下误报
    'no-undef': 'off',
    // 噪音规则：项目大量使用 any，暂降为 warn
    '@typescript-eslint/no-explicit-any': 'warn',
    // 未用变量降为 warn，豁免下划线开头参数
    '@typescript-eslint/no-unused-vars': [
      'warn',
      {
        argsIgnorePattern: '^_',
        varsIgnorePattern: '^_',
        caughtErrorsIgnorePattern: '^_',
      },
    ],
    // CesiumMath / EasingFunction 使用 export namespace 分组工具函数（模拟 Cesium API），属合法 TS 模式
    '@typescript-eslint/no-namespace': 'off',
    // recommended 中过于严格的项降级
    '@typescript-eslint/no-empty-function': 'off',
    '@typescript-eslint/ban-ts-comment': 'off',
  },
};
