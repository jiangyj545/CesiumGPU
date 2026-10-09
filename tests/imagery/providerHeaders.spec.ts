/**
 * providerHeaders.spec.ts – P1-F 认证 headers 透传：
 * XyzImageryProvider / LocalQuantizedMeshProvider / fetchTilesetJson(+Tileset3D
 * content) 的每次 fetch 都携带构造时传入的 headers；不传时行为不变。
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { XyzImageryProvider } from '../../src/imagery/XyzImageryProvider';
import { LocalQuantizedMeshProvider } from '../../src/terrain/LocalQuantizedMeshProvider';
import { TerrariumProvider } from '../../src/terrain/TerrariumProvider';
import { fetchTilesetJson } from '../../src/tileset/TilesetContentLoader';
import { TileKey } from '../../src/imagery/TileKey';

const AUTH = { Authorization: 'Bearer test-token' };

type FetchCall = { url: string; init: RequestInit | undefined };

function stubFetch(body: string | ArrayBuffer = '{}'): FetchCall[] {
  const calls: FetchCall[] = [];
  const mock = vi.fn(async (url: string | URL, init?: RequestInit) => {
    calls.push({ url: String(url), init });
    return new Response(body, { status: 200 });
  });
  vi.stubGlobal('fetch', mock);
  return calls;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('XyzImageryProvider headers', () => {
  it('每个瓦片请求都带 headers', async () => {
    const calls = stubFetch(new Uint8Array([0x89, 0x50]).buffer);
    const provider = new XyzImageryProvider({
      url: 'https://example.com/{z}/{x}/{y}.png',
      headers: AUTH,
    });
    await provider.requestTile(new TileKey(1, 0, 0)); // createImageBitmap 在 node 不可用 → null，fetch 已发出
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe('https://example.com/1/0/0.png');
    expect(calls[0].init?.headers).toEqual(AUTH);
  });

  it('不传 headers 时 init.headers 为 undefined（行为不变）', async () => {
    const calls = stubFetch(new Uint8Array([0x89, 0x50]).buffer);
    const provider = new XyzImageryProvider({ url: 'https://example.com/{z}/{x}/{y}.png' });
    await provider.requestTile(new TileKey(1, 0, 0));
    expect(calls[0].init?.headers).toBeUndefined();
  });
});

describe('HTTP terrain provider headers', () => {
  it('LocalQuantizedMeshProvider 瓦片请求带 headers', async () => {
    const calls = stubFetch(new Uint8Array(8).buffer);
    const provider = new LocalQuantizedMeshProvider({ headers: AUTH });
    await provider.requestTileGeometry(1, 0, 0); // 解码失败 → null，fetch 已发出
    expect(calls).toHaveLength(1);
    expect(calls[0].init?.headers).toEqual(AUTH);
    provider.destroy();
  });

  it('TerrariumProvider options.headers 透传到基类', () => {
    const provider = new TerrariumProvider({ headers: AUTH });
    expect(provider.headers).toEqual(AUTH);
    provider.destroy();
  });

  it('不传 headers 时 init.headers 为 undefined', async () => {
    const calls = stubFetch(new Uint8Array(8).buffer);
    const provider = new LocalQuantizedMeshProvider();
    await provider.requestTileGeometry(1, 0, 0);
    expect(calls[0].init?.headers).toBeUndefined();
    provider.destroy();
  });
});

describe('tileset.json headers（fetchTilesetJson）', () => {
  it('tileset.json 请求携带 headers', async () => {
    const calls = stubFetch(JSON.stringify({ asset: { version: '1.1' }, root: {} }));
    const { json } = await fetchTilesetJson('https://example.com/tileset.json', { headers: AUTH });
    expect(json.root).toBeDefined();
    expect(calls).toHaveLength(1);
    expect(calls[0].init?.headers).toEqual(AUTH);
  });

  it('不传 headers 时不注入头', async () => {
    const calls = stubFetch(JSON.stringify({ asset: { version: '1.1' }, root: {} }));
    await fetchTilesetJson('https://example.com/tileset.json');
    expect(calls[0].init?.headers).toBeUndefined();
  });
});
