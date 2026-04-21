/**
 * Node.js embedding providers used by offline scripts (eval, training corpus).
 * Not imported by the runtime plugin.
 */

import { EmbeddingProvider } from "../../src/resolver/embedding-provider";

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
