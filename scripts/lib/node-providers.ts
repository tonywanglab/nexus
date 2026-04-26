/**
 * Node.js embedding providers used by offline scripts (eval, training corpus).
 * Not imported by the runtime plugin.
 */

import { EmbeddingProvider } from "../../src/resolver/embedding-provider";
import {
  isEmbeddingGemmaModelId,
  meanPoolNormalizeL2BatchedFromLastHidden,
  meanPoolNormalizeL2FromLastHidden,
} from "../../src/resolver/gemma-embedding";

/**
 * Pipeline-based provider for models that work with the `feature-extraction`
 * pipeline (e.g. `Snowflake/snowflake-arctic-embed-xs`).
 */
export class PipelineEmbeddingProvider implements EmbeddingProvider {
  readonly dims: number;
  private modelId: string;
  private extractor: any = null;

  constructor(modelId: string, dims: number) {
    this.modelId = modelId;
    this.dims = dims;
  }

  async embed(text: string): Promise<Float32Array> {
    const [result] = await this.embedBatch([text]);
    return result;
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    if (!this.extractor) {
      const { pipeline, env } = require("@huggingface/transformers");
      env.allowLocalModels = true;
      env.useBrowserCache = false;
      this.extractor = await pipeline("feature-extraction", this.modelId, {
        quantized: true,
      });
    }
    const results: Float32Array[] = [];
    for (const text of texts) {
      const output = await this.extractor(text, { pooling: "mean", normalize: true });
      results.push(new Float32Array(output.data));
    }
    return results;
  }

  async dispose(): Promise<void> {
    this.extractor = null;
  }
}

/**
 * AutoModel + tokenizer path for EmbeddingGemma (matches runtime iframe Gemma path).
 */
export class GemmaNodeProvider implements EmbeddingProvider {
  readonly dims: number;
  private modelId: string;
  private dtype: string;
  /** Transformers.js device (e.g. coreml on macOS Node, cpu, webgpu). */
  private device: string;
  private model: any = null;
  private tokenizer: any = null;

  constructor(
    modelId: string,
    dims: number,
    opts: { dtype?: string; device?: string } = {},
  ) {
    this.modelId = modelId;
    this.dims = dims;
    this.dtype = opts.dtype ?? "q8";
    this.device = opts.device ?? process.env.NEXUS_EMBED_DEVICE ?? "cpu";
  }

  private callCount = 0;

  private async ensureModel(): Promise<void> {
    if (this.model) return;
    const { AutoModel, AutoTokenizer, env } = require("@huggingface/transformers");
    env.allowLocalModels = true;
    env.useBrowserCache = false;
    process.stderr.write(
      `[${new Date().toISOString()}] GemmaNodeProvider: ONNX device=${this.device} dtype=${this.dtype}\n`,
    );
    const t0 = Date.now();
    this.tokenizer = await AutoTokenizer.from_pretrained(this.modelId);
    process.stderr.write(
      `[${new Date().toISOString()}] GemmaNodeProvider: tokenizer loaded (${Date.now() - t0}ms)\n`,
    );
    const t1 = Date.now();
    this.model = await AutoModel.from_pretrained(this.modelId, {
      dtype: this.dtype,
      device: this.device,
    });
    process.stderr.write(
      `[${new Date().toISOString()}] GemmaNodeProvider: model loaded (${Date.now() - t1}ms, total=${Date.now() - t0}ms)\n`,
    );
  }

  async embed(text: string): Promise<Float32Array> {
    const [result] = await this.embedBatch([text]);
    return result;
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    await this.ensureModel();
    const callId = ++this.callCount;

    if (texts.length === 1) {
      const tTok = Date.now();
      const inputs = await this.tokenizer(texts[0], { padding: true, truncation: true });
      const tokMs = Date.now() - tTok;
      const inputSeqLen = (inputs.input_ids?.dims as number[] | undefined)?.[1] ?? -1;
      const tFwd = Date.now();
      const output = await this.model(inputs);
      const fwdMs = Date.now() - tFwd;
      const embeddings = output.last_hidden_state;
      const tensorDims = embeddings.dims as number[];
      const seqLen = tensorDims[1];
      const hiddenDim = tensorDims[2];
      const data = embeddings.data as Float32Array;
      process.stderr.write(
        `[${new Date().toISOString()}] GemmaNodeProvider[#${callId}] batch=1 seqLen=${inputSeqLen} tok=${tokMs}ms fwd=${fwdMs}ms\n`,
      );
      return [meanPoolNormalizeL2FromLastHidden(data, seqLen, hiddenDim)];
    }

    const tTok = Date.now();
    const inputs = await this.tokenizer(texts, { padding: true, truncation: true });
    const tokMs = Date.now() - tTok;
    const inputDims = inputs.input_ids?.dims as number[] | undefined;
    const inputBatch = inputDims?.[0] ?? texts.length;
    const inputSeqLen = inputDims?.[1] ?? -1;
    const tFwd = Date.now();
    const output = await this.model(inputs);
    const fwdMs = Date.now() - tFwd;
    const embeddings = output.last_hidden_state;
    const tensorDims = embeddings.dims as number[];
    const batch = tensorDims[0];
    const seqLen = tensorDims[1];
    const hiddenDim = tensorDims[2];
    const data = embeddings.data as Float32Array;
    const maskTensor = inputs.attention_mask;
    if (!maskTensor?.data) {
      throw new Error("GemmaNodeProvider: tokenizer did not return attention_mask (required for batched pooling).");
    }
    process.stderr.write(
      `[${new Date().toISOString()}] GemmaNodeProvider[#${callId}] batch=${inputBatch} seqLen=${inputSeqLen} tok=${tokMs}ms fwd=${fwdMs}ms\n`,
    );
    return meanPoolNormalizeL2BatchedFromLastHidden(
      data,
      batch,
      seqLen,
      hiddenDim,
      maskTensor.data as ArrayBufferView,
    );
  }

  async dispose(): Promise<void> {
    this.model = null;
    this.tokenizer = null;
  }
}

/** Pick pipeline or Gemma provider from `--model` id (matches embed-corpus / embed-vocab). */
export function createNodeEmbeddingProvider(
  modelId: string,
  dims: number,
  opts: { device?: string } = {},
): EmbeddingProvider {
  if (isEmbeddingGemmaModelId(modelId)) {
    return new GemmaNodeProvider(modelId, dims, { ...opts, dtype: "q8" });
  }
  return new PipelineEmbeddingProvider(modelId, dims);
}

export { isEmbeddingGemmaModelId };
