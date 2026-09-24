/**
 * fetchWithGzip – 3DTiles 加载层共用的网络取数工具。
 *
 * 说明：若服务器通过 `Content-Encoding: gzip` 响应头传输 gzip 内容，
 * 浏览器 fetch 会自动解压（arrayBuffer() 拿到的已是明文），无需处理。
 * 本模块只负责「文件本体以 .gz 存放、但未设置 Content-Encoding」的显式场景，
 * 以及调用方明确要求解压的 forceGzip 选项。
 */

export interface FetchArrayBufferOptions {
  /** 即使 URL 不以 .gz 结尾也强制走 gzip 解压（服务器把 .gz 内容裸发时用）。 */
  forceGzip?: boolean;
  /** 透传给 fetch 的 RequestInit（headers 等）。 */
  init?: RequestInit;
}

/**
 * Fetch an URL as ArrayBuffer, transparently decompressing gzip payloads
 * when the URL ends with `.gz` (or `opts.forceGzip` is set).
 *
 * 供 tileset 加载层复用（tileset.json / b3dm / .gz 压缩资产等）。
 */
export async function fetchArrayBuffer(
  url: string,
  opts?: FetchArrayBufferOptions
): Promise<ArrayBuffer> {
  const res = await fetch(url, opts?.init);
  if (!res.ok) {
    throw new Error(`fetchArrayBuffer: failed to fetch "${url}" (${res.status} ${res.statusText})`);
  }

  const buffer = await res.arrayBuffer();

  const looksGzipped = /\.gz(?:[?#]|$)/i.test(url);
  if (!opts?.forceGzip && !looksGzipped) return buffer;

  if (typeof DecompressionStream === 'undefined') {
    throw new Error(`fetchArrayBuffer: DecompressionStream is not available in this environment (url "${url}")`);
  }

  // Blob → stream → gzip 解压 → 回读为 ArrayBuffer
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}
