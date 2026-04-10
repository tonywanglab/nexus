/**
 * Embedding provider interface and implementations for dense vector embeddings.
 */

export interface EmbeddingProvider {
  embed(text: string): Promise<Float32Array>;
  embedBatch(texts: string[]): Promise<Float32Array[]>;
  readonly dims: number;
  dispose(): Promise<void>;
}

// ── TransformersIframeProvider ─────────────────────────────────────────

const CDN_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers";
const DEFAULT_MODEL = "Snowflake/snowflake-arctic-embed-xs";
const DEFAULT_DIMS = 384;

/**
 * Connector script injected into the hidden iframe via srcdoc.
 * Runs @huggingface/transformers in an isolated browsing context
 * so ONNX/WASM never blocks the Obsidian main thread.
 */
function buildIframeSrcdoc(model: string): string {
  return `<!DOCTYPE html>
<html><head><meta charset="utf-8"></head><body><script type="module">
import { pipeline, env } from "${CDN_URL}";

env.allowLocalModels = false;
env.useBrowserCache = true;

let extractor = null;

async function getExtractor() {
  if (!extractor) {
    extractor = await pipeline("feature-extraction", "${model}", {
      quantized: true,
    });
  }
  return extractor;
}

window.addEventListener("message", async (event) => {
  const { id, method, params } = event.data;
  if (!id || !method) return;

  try {
    if (method === "embed_batch") {
      const ext = await getExtractor();
      const results = [];
      for (const text of params.texts) {
        const output = await ext(text, { pooling: "mean", normalize: true });
        results.push(Array.from(output.data));
      }
      parent.postMessage({ id, result: results }, "*");
    } else if (method === "ping") {
      parent.postMessage({ id, result: "pong" }, "*");
    }
  } catch (err) {
    parent.postMessage({ id, error: err.message || String(err) }, "*");
  }
});

// Signal ready
parent.postMessage({ id: "__ready__", result: true }, "*");
</${"script"}></body></html>`;
}

interface PendingRequest {
  resolve: (value: any) => void;
  reject: (reason: any) => void;
}

/**
 * Runs @huggingface/transformers inside a hidden iframe for WASM isolation.
 * Communication via postMessage with promise-based request/response matching.
 */
export class TransformersIframeProvider implements EmbeddingProvider {
  readonly dims: number;
  private iframe: HTMLIFrameElement | null = null;
  private pending = new Map<string, PendingRequest>();
  private messageHandler: ((event: MessageEvent) => void) | null = null;
  private readyPromise: Promise<void> | null = null;
  private nextId = 0;
  private model: string;

  constructor(model: string = DEFAULT_MODEL, dims: number = DEFAULT_DIMS) {
    this.model = model;
    this.dims = dims;
  }

  private ensureIframe(): Promise<void> {
    if (this.readyPromise) return this.readyPromise;

    this.readyPromise = new Promise<void>((resolve, reject) => {
      const iframe = document.createElement("iframe");
      iframe.style.display = "none";
      iframe.sandbox.add("allow-scripts");
      iframe.srcdoc = buildIframeSrcdoc(this.model);
      this.iframe = iframe;

      const timer = setTimeout(() => reject(new Error("Iframe initialization timed out")), 30_000);

      this.messageHandler = (event: MessageEvent) => {
        const { id, result, error } = event.data ?? {};
        if (!id) return;

        if (id === "__ready__") {
          clearTimeout(timer);
          resolve();
          return;
        }

        const req = this.pending.get(id);
        if (!req) return;
        this.pending.delete(id);

        if (error) {
          req.reject(new Error(error));
        } else {
          req.resolve(result);
        }
      };

      window.addEventListener("message", this.messageHandler);
      document.body.appendChild(iframe);

    });

    return this.readyPromise;
  }

  private async postMessage<T>(method: string, params: Record<string, unknown>): Promise<T> {
    await this.ensureIframe();
    const id = `emb_${this.nextId++}`;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.iframe!.contentWindow!.postMessage({ id, method, params }, "*");
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`Embedding request ${id} timed out`));
        }
      }, 60_000);
    });
  }

  async embed(text: string): Promise<Float32Array> {
    const results = await this.embedBatch([text]);
    return results[0];
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    const raw = await this.postMessage<number[][]>("embed_batch", { texts });
    return raw.map(arr => new Float32Array(arr));
  }

  async dispose(): Promise<void> {
    if (this.messageHandler) {
      window.removeEventListener("message", this.messageHandler);
      this.messageHandler = null;
    }
    if (this.iframe) {
      this.iframe.remove();
      this.iframe = null;
    }
    this.pending.clear();
    this.readyPromise = null;
  }
}
