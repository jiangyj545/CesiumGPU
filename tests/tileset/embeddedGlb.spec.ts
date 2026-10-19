/**
 * embeddedGlb.spec.ts – 3DTiles 1.1 内嵌 content（data: URI base64 glb）。
 *
 * 覆盖三层：
 * ① parseTileset：content.uri 为 data: URI → kind 判定、resolvedUrl 原样、
 *    urlDecorator 不注入（自包含数据无网络语义）；
 * ② decodeDataUri：base64 / 明文 / 非法格式；
 * ③ loadTileContent：data: URI 全链（解码 → parseGlbBuffer → 组装），
 *    fetch 断言零调用（不发网络请求）。
 * 全部合成数据（tests/loader/helpers.buildMinimalGlb）。
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { parseTileset } from '../../src/tileset/TilesetContentLoader';
import { decodeDataUri, loadTileContent } from '../../src/tileset/TilesetPrimitive';
import { GltfLoader } from '../../src/loader/GltfLoader';
import { buildMinimalGlb } from '../loader/helpers';
import { Matrix4 } from '../../src/math/Matrix4';
import type { Context } from '../../src/renderer/Context';

// Primitive 构造需要 GPU context：测试缝替换为记录型假件（先例：
// tests/tileset/tileset3dIntegration.spec.ts 的 FakePrimitive）
vi.mock('../../src/scene/Primitive', () => ({
  Primitive: class FakePrimitive {
    modelMatrix: unknown = null;
    setTexture(): void { /* 无纹理路径不触达 */ }
  },
}));

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('parseTileset – data: URI 内嵌 content', () => {
  it('model/gltf-binary → kind=glb；resolvedUrl 原样不相对解析', () => {
    const node = parseTileset({
      root: {
        geometricError: 100,
        boundingVolume: { sphere: [0, 0, 0, 1] },
        content: { uri: 'data:model/gltf-binary;base64,AAAA' },
      },
    }, 'http://test/dir/tileset.json');

    expect(node.content?.kind).toBe('glb');
    expect(node.content?.resolvedUrl).toBe('data:model/gltf-binary;base64,AAAA');
  });

  it('urlDecorator 不作用于 data: URI（与网络 content 对比）', () => {
    const decorator = (u: string): string => `${u}?key=K`;
    const nodes = [
      'data:model/gltf-binary;base64,AAAA',
      'tile.glb',
    ].map((uri) => parseTileset({
      root: {
        geometricError: 100,
        boundingVolume: { sphere: [0, 0, 0, 1] },
        content: { uri },
      },
    }, 'http://test/dir/tileset.json', { urlDecorator: decorator }).content);

    expect(nodes[0]?.resolvedUrl).toBe('data:model/gltf-binary;base64,AAAA'); // 内嵌：不注入
    expect(nodes[1]?.resolvedUrl).toBe('http://test/dir/tile.glb?key=K');     // 网络：注入
  });

  it('非 glTF mime → kind=unknown（仍按 GLB 试解析语义，不静默吞）', () => {
    const node = parseTileset({
      root: {
        geometricError: 100,
        boundingVolume: { sphere: [0, 0, 0, 1] },
        content: { uri: 'data:application/octet-stream;base64,AAAA' },
      },
    }, 'http://test/tileset.json');
    expect(node.content?.kind).toBe('unknown');
  });
});

describe('decodeDataUri', () => {
  it('base64 载荷 → 原字节还原', () => {
    const raw = new Uint8Array([0x67, 0x6c, 0x54, 0x46, 0x00, 0xff, 0x10]);
    const b64 = Buffer.from(raw).toString('base64');
    const out = new Uint8Array(decodeDataUri(`data:model/gltf-binary;base64,${b64}`));
    expect(Array.from(out)).toEqual(Array.from(raw));
  });

  it('明文载荷（percent-encoded）→ UTF-8 字节', () => {
    const out = new Uint8Array(decodeDataUri('data:text/plain,hello%20world'));
    expect(new TextDecoder().decode(out)).toBe('hello world');
  });

  it('缺 "," → 抛错（不静默）', () => {
    expect(() => decodeDataUri('data:model/gltf-binary;base64AAAA')).toThrow(/data: URI/);
  });
});

describe('loadTileContent – data: URI 全链（零网络）', () => {
  it('内嵌 glb 解码 → parseGlbBuffer → 组装 Primitive；fetch 零调用', async () => {
    const glb = buildMinimalGlb();
    const b64 = Buffer.from(glb).toString('base64');
    const uri = `data:model/gltf-binary;base64,${b64}`;

    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);

    const built = await loadTileContent(
      {} as Context,
      { uri, resolvedUrl: uri, kind: 'glb' },
      Matrix4.identity()
    );

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(built.primitives.length).toBeGreaterThan(0);
    expect(built.textures.length).toBe(0);
  });

  it('内嵌 b3dm data: URI → 剥壳路径同样成立（kind=b3dm 显式声明）', async () => {
    // b3dm = 28B 头 + 空 featureTable + 最小 GLB（复用 b3dmDecoder.spec 布局）
    const glb = new Uint8Array(buildMinimalGlb());
    const header = new ArrayBuffer(28);
    const dv = new DataView(header);
    dv.setUint32(0, 0x6d643362, true);   // 'b3dm'
    dv.setUint32(4, 1, true);            // version
    dv.setUint32(8, 2, true);            // featureTableJsonLength
    dv.setUint32(12, 0, true);
    dv.setUint32(16, 0, true);
    dv.setUint32(20, 0, true);
    dv.setUint32(24, 0, true);           // gltfFormat = 0
    const ftJson = new TextEncoder().encode('{}');
    const bytes = new Uint8Array(28 + ftJson.length + glb.length);
    bytes.set(new Uint8Array(header), 0);
    bytes.set(ftJson, 28);
    bytes.set(glb, 28 + ftJson.length);

    const b64 = Buffer.from(bytes).toString('base64');
    const uri = `data:model/gltf-binary;base64,${b64}`;
    vi.stubGlobal('fetch', vi.fn());

    const built = await loadTileContent(
      {} as Context,
      { uri, resolvedUrl: uri, kind: 'b3dm' },
      Matrix4.identity()
    );
    expect(built.primitives.length).toBeGreaterThan(0);
  });

  it('真实 GLB sanity：parseGlbBuffer 可独立解析 buildMinimalGlb（解析层基线）', async () => {
    const parsed = await GltfLoader.parseGlbBuffer(buildMinimalGlb());
    expect(parsed.primitives.length).toBeGreaterThan(0);
  });
});
