/**
 * Node.js embedding providers used by offline scripts (eval, training corpus).
 * Not imported by the runtime plugin.
 */

import { EmbeddingProvider } from "../../src/resolver/embedding-provider";
import {
  isEmbeddingGemmaModelId,
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
  private model: any = null;
  private tokenizer: any = null;

  constructor(modelId: string, dims: number, dtype: string = "q4") {
    this.modelId = modelId;
    this.dims = dims;
    this.dtype = dtype;
  }

  private async ensureModel(): Promise<void> {
    if (this.model) return;
    const { AutoModel, AutoTokenizer, env } = require("@huggingface/transformers");
    env.allowLocalModels = true;
    env.useBrowserCache = false;
    this.tokenizer = await AutoTokenizer.from_pretrained(this.modelId);
    this.model = await AutoModel.from_pretrained(this.modelId, {
      dtype: this.dtype,
    });
  }

  async embed(text: string): Promise<Float32Array> {
    const [result] = await this.embedBatch([text]);
    return result;
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    await this.ensureModel();
    const results: Float32Array[] = [];
    for (const text of texts) {
      const inputs = await this.tokenizer(text, { padding: true, truncation: true });
      const output = await this.model(inputs);
      const embeddings = output.last_hidden_state;
      const tensorDims = embeddings.dims as number[];
      const seqLen = tensorDims[1];
      const hiddenDim = tensorDims[2];
      const data = embeddings.data as Float32Array;
      results.push(meanPoolNormalizeL2FromLastHidden(data, seqLen, hiddenDim));
    }
    return results;
  }

  async dispose(): Promise<void> {
    this.model = null;
    this.tokenizer = null;
  }
}

/** Pick pipeline or Gemma provider from `--model` id (matches embed-corpus / embed-vocab). */
export function createNodeEmbeddingProvider(modelId: string, dims: number): EmbeddingProvider {
  if (isEmbeddingGemmaModelId(modelId)) {
    return new GemmaNodeProvider(modelId, dims, "q4");
  }
  return new PipelineEmbeddingProvider(modelId, dims);
}

export { isEmbeddingGemmaModelId };
