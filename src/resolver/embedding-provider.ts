/**
 * Embedding provider interface and implementations for dense vector embeddings.
 */

import { isEmbeddingGemmaModelId } from "./gemma-embedding";

export type EmbeddingPriority = "high" | "normal";

export interface EmbeddingRequestOptions {
  priority?: EmbeddingPriority;
}

export interface EmbeddingProvider {
  embed(text: string, options?: EmbeddingRequestOptions): Promise<Float32Array>;
  embedBatch(texts: string[], options?: EmbeddingRequestOptions): Promise<Float32Array[]>;
  readonly dims: number;
  dispose(): Promise<void>;
}

export type EmbeddingProgressEvent =
  | { type: "download"; file: string; pct: number }
  | { type: "ready" };

export type EmbeddingProgressListener = (ev: EmbeddingProgressEvent) => void;

// ── TransformersIframeProvider ─────────────────────────────────────────

const CDN_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers";
const DEFAULT_MODEL = "onnx-community/embeddinggemma-300m-ONNX";
const DEFAULT_DIMS = 768;

function buildPipelineIframeSrcdoc(model: string): string {
  return `<!DOCTYPE html>
<html><head><meta charset="utf-8"></head><body><script type="module">
import { pipeline, env } from "${CDN_URL}";

env.allowLocalModels = false;
env.useBrowserCache = false;

let extractor = null;
let extractorPromise = null;

function logToParent(msg) {
  parent.postMessage({ id: "__log__", result: msg }, "*");
}

async function getExtractor() {
  if (extractor) return extractor;
  if (!extractorPromise) {
    logToParent("loading model (first request, may download ~30MB)…");
    const t0 = performance.now();
    extractorPromise = pipeline("feature-extraction", "${model}", {
      quantized: true,
      progress_callback: (p) => {
        if (p && p.status === "progress" && p.file) {
          logToParent(\`downloading \${p.file}: \${Math.round(p.progress || 0)}%\`);
        } else if (p && p.status === "ready") {
          logToParent("model ready");
        }
      },
    }).then((ext) => {
      logToParent(\`model loaded in \${Math.round(performance.now() - t0)}ms\`);
      extractor = ext;
      return ext;
    });
  }
  return extractorPromise;
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

parent.postMessage({ id: "__ready__", result: true }, "*");
</${"script"}></body></html>`;
}

/**
 * EmbeddingGemma: AutoModel + tokenizer + mean pool + L2 (aligned with Node GemmaNodeProvider).
 * Iframe source kept in sync with src/resolver/gemma-embedding.ts pooling math.
 */
function buildGemmaIframeSrcdoc(model: string): string {
  const modelJs = JSON.stringify(model);
  return `<!DOCTYPE html>
<html><head><meta charset="utf-8"></head><body><script type="module">
import { AutoModel, AutoTokenizer, env } from "${CDN_URL}";

env.allowLocalModels = false;
env.useBrowserCache = false;

const MODEL_ID = ${modelJs};

let tokenizer = null;
let mdl = null;
let loadPromise = null;

function logToParent(msg) {
  parent.postMessage({ id: "__log__", result: msg }, "*");
}

function progressCallback(p) {
  if (p && p.status === "progress" && p.file) {
    logToParent(\`downloading \${p.file}: \${Math.round(p.progress || 0)}%\`);
  } else if (p && p.status === "ready") {
    logToParent("model ready");
  }
}

function maskVal(maskData, i) {
  // Gemma tokenizer may return BigInt64Array; coerce to number before comparing.
  const v = maskData[i];
  return typeof v === "bigint" ? Number(v) : v;
}

function poolMeanNormBatched(embeddings, attentionMask) {
  const dims = embeddings.dims;       // [batch, seqLen, hiddenDim]
  const batch = dims[0];
  const seqLen = dims[1];
  const hiddenDim = dims[2];
  const data = embeddings.data;       // Float32Array
  const maskData = attentionMask.data;
  const results = [];
  for (let b = 0; b < batch; b++) {
    const pooled = new Float32Array(hiddenDim);
    let count = 0;
    for (let s = 0; s < seqLen; s++) {
      if (maskVal(maskData, b * seqLen + s) === 0) continue;
      count++;
      const offset = (b * seqLen + s) * hiddenDim;
      for (let d = 0; d < hiddenDim; d++) pooled[d] += data[offset + d];
    }
    for (let d = 0; d < hiddenDim; d++) pooled[d] /= count || 1;
    let norm = 0;
    for (let d = 0; d < hiddenDim; d++) norm += pooled[d] * pooled[d];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let d = 0; d < hiddenDim; d++) pooled[d] /= norm;
    results.push(Array.from(pooled));
  }
  return results;
}

async function ensureModel() {
  if (mdl && tokenizer) return;
  if (!loadPromise) {
    logToParent("loading EmbeddingGemma (first request may download model)…");
    const t0 = performance.now();
    loadPromise = (async () => {
      tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
        progress_callback: progressCallback,
      });
      mdl = await AutoModel.from_pretrained(MODEL_ID, { dtype: "q8", progress_callback: progressCallback });
      logToParent(\`model loaded dtype=q8 in \${Math.round(performance.now() - t0)}ms\`);
    })();
  }
  await loadPromise;
}

window.addEventListener("message", async (event) => {
  const { id, method, params } = event.data;
  if (!id || !method) return;

  try {
    if (method === "embed_batch") {
      await ensureModel();
      // Tokenize the whole batch at once and run a single forward pass.
      const inputs = await tokenizer(params.texts, { padding: true, truncation: true });
      const output = await mdl(inputs);
      const results = poolMeanNormBatched(output.last_hidden_state, inputs.attention_mask);
      parent.postMessage({ id, result: results }, "*");
    } else if (method === "ping") {
      parent.postMessage({ id, result: "pong" }, "*");
    }
  } catch (err) {
    parent.postMessage({ id, error: err.message || String(err) }, "*");
  }
});

parent.postMessage({ id: "__ready__", result: true }, "*");
</${"script"}></body></html>`;
}

function buildIframeSrcdoc(model: string): string {
  return isEmbeddingGemmaModelId(model) ? buildGemmaIframeSrcdoc(model) : buildPipelineIframeSrcdoc(model);
}

interface PendingRequest {
  resolve: (value: any) => void;
  reject: (reason: any) => void;
}

/**
 * Two-priority FIFO queue: `high` entries drain before `normal` ones, while
 * preserving FIFO order within a priority. Used to serialize embedding work
 * against the single-threaded iframe pipeline while letting user-facing
 * requests cut ahead of bulk indexing.
 */
class PriorityQueue<T> {
  private buckets: Record<EmbeddingPriority, T[]> = { high: [], normal: [] };

  push(priority: EmbeddingPriority, item: T): void {
    this.buckets[priority].push(item);
  }

  shift(): T | undefined {
    return this.buckets.high.shift() ?? this.buckets.normal.shift();
  }

  get size(): number {
    return this.buckets.high.length + this.buckets.normal.length;
  }
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
  private modelLoaded = false;
  private progressListener: EmbeddingProgressListener | null = null;
  // Serialize requests: the ONNX pipeline in the iframe is single-threaded, so
  // concurrent posts just interleave and make per-request timers fire on work
  // that hasn't started yet. `high` drains before `normal` so user-facing work
  // (active file, approve-triggered reprocess) at worst waits for the single
  // batch currently in flight.
  private queue = new PriorityQueue<() => Promise<void>>();
  private draining = false;
  /** Time to wait for iframe `__ready__` (large Gemma downloads need minutes). */
  private readonly iframeReadyTimeoutMs: number;

  constructor(model: string = DEFAULT_MODEL, dims: number = DEFAULT_DIMS) {
    this.model = model;
    this.dims = dims;
    this.iframeReadyTimeoutMs = isEmbeddingGemmaModelId(model) ? 180_000 : 30_000;
  }

  onProgress(listener: EmbeddingProgressListener | null): void {
    this.progressListener = listener;
  }

  private ensureIframe(): Promise<void> {
    if (this.readyPromise) return this.readyPromise;

    console.log(`Nexus: initializing embedding iframe (model=${this.model})`);
    const t0 = performance.now();

    this.readyPromise = new Promise<void>((resolve, reject) => {
      const iframe = document.createElement("iframe");
      iframe.style.display = "none";
      iframe.sandbox.add("allow-scripts");
      iframe.srcdoc = buildIframeSrcdoc(this.model);
      this.iframe = iframe;

      const timer = setTimeout(
        () =>
          reject(
            new Error(`Iframe initialization timed out (${Math.round(this.iframeReadyTimeoutMs / 1000)}s)`),
          ),
        this.iframeReadyTimeoutMs,
      );

      this.messageHandler = (event: MessageEvent) => {
        const { id, result, error } = event.data ?? {};
        if (!id) return;

        if (id === "__ready__") {
          clearTimeout(timer);
          console.log(
            `Nexus: embedding iframe ready (${Math.round(performance.now() - t0)}ms)`,
          );
          resolve();
          return;
        }

        if (id === "__log__") {
          console.log(`Nexus: [iframe] ${result}`);
          this.emitProgressFromLog(String(result));
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
      console.log("Nexus: embedding iframe appended, waiting for model download…");
    });

    return this.readyPromise;
  }

  private async postMessage<T>(
    method: string,
    params: Record<string, unknown>,
    priority: EmbeddingPriority = "normal",
  ): Promise<T> {
    await this.ensureIframe();
    return new Promise<T>((resolve, reject) => {
      const task = () => {
        const id = `emb_${this.nextId++}`;
        // First request carries the model download; give it much more headroom.
        const timeoutMs = this.modelLoaded ? 60_000 : 300_000;
        return new Promise<void>((done) => {
          this.pending.set(id, {
            resolve: (v) => {
              this.modelLoaded = true;
              resolve(v as T);
              done();
            },
            reject: (e) => {
              reject(e);
              done();
            },
          });
          this.iframe!.contentWindow!.postMessage({ id, method, params }, "*");
          setTimeout(() => {
            if (this.pending.has(id)) {
              this.pending.delete(id);
              reject(new Error(`Embedding request ${id} timed out after ${timeoutMs}ms`));
              done();
            }
          }, timeoutMs);
        });
      };
      this.queue.push(priority, task);
      void this.drain();
    });
  }

  private async drain(): Promise<void> {
    if (this.draining) return;
    this.draining = true;
    try {
      while (this.queue.size > 0) {
        const next = this.queue.shift();
        if (!next) break;
        await next();
      }
    } finally {
      this.draining = false;
    }
  }

  async embed(text: string, options?: EmbeddingRequestOptions): Promise<Float32Array> {
    const results = await this.embedBatch([text], options);
    return results[0];
  }

  async embedBatch(texts: string[], options?: EmbeddingRequestOptions): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    // q8 Gemma-300M in WASM can't process arbitrary batch sizes within the 60s
    // postMessage timeout — a reindex that passes 200+ note titles as one batch
    // will hang the iframe. Chunk so each postMessage gets its own budget.
    const CHUNK = 16;
    const results: Float32Array[] = [];
    for (let i = 0; i < texts.length; i += CHUNK) {
      const chunk = texts.slice(i, i + CHUNK);
      const raw = await this.postMessage<number[][]>("embed_batch", { texts: chunk }, options?.priority);
      for (const arr of raw) results.push(new Float32Array(arr));
    }
    return results;
  }

  private emitProgressFromLog(msg: string): void {
    if (!this.progressListener) return;
    const m = msg.match(/^downloading (\S+): (\d+)%$/);
    if (m) {
      this.progressListener({ type: "download", file: m[1], pct: parseInt(m[2], 10) });
      return;
    }
    if (msg === "model ready") {
      this.progressListener({ type: "ready" });
    }
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
