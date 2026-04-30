#!/usr/bin/env python3
"""
Recall@K evaluation: measures how well sparse SAE cosine similarity
recovers the dense embedding neighborhood.

For each probe term, find top-20 neighbors by dense cosine sim (oracle),
then find top-20 by sparse cosine sim for each SAE config.
Recall = |sparse_top20 ∩ dense_top20| / 20.
"""
import json, struct, argparse
import numpy as np
from pathlib import Path


def load_vocab_embeddings(path: Path, n_vocab: int, dims: int) -> np.ndarray:
    raw = path.read_bytes()
    arr = np.frombuffer(raw, dtype=np.float32).reshape(n_vocab, dims)
    return arr


def parse_sae(data: bytes):
    assert data[:4] == b"SAE1"
    hlen = struct.unpack_from("<I", data, 4)[0]
    header = json.loads(data[8: 8 + hlen].decode("utf-8").strip())
    d_model  = header["dModel"]
    d_hidden = header["dHidden"]
    k        = header["k"]
    off = 8 + hlen
    def take(shape):
        nonlocal off
        n = int(np.prod(shape))
        arr = np.frombuffer(data, dtype=np.float32, count=n, offset=off).reshape(shape)
        off += n * 4
        return arr
    w_enc = take((d_hidden, d_model))
    b_enc = take((d_hidden,))
    b_pre = take((d_model,))
    w_dec = take((d_hidden, d_model))  # stored transposed
    b_dec = take((d_model,))
    return dict(d_model=d_model, d_hidden=d_hidden, k=k,
                w_enc=w_enc, b_enc=b_enc, b_pre=b_pre, w_dec=w_dec, b_dec=b_dec)


def sae_encode(sae, x: np.ndarray) -> np.ndarray:
    """Encode batch x (N, d_model) → sparse activations (N, d_hidden)."""
    centered = x - sae["b_pre"]
    pre = centered @ sae["w_enc"].T + sae["b_enc"]
    feats = np.maximum(pre, 0)
    k = sae["k"]
    idx = np.argpartition(-feats, k, axis=1)[:, :k]
    sparse = np.zeros_like(feats)
    np.put_along_axis(sparse, idx, np.take_along_axis(feats, idx, axis=1), axis=1)
    return sparse


def sparse_cosine_matrix(probes: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """(P, d_hidden) × (C, d_hidden) → (P, C) cosine similarities."""
    dots = probes @ corpus.T
    pn = np.linalg.norm(probes, axis=1, keepdims=True).clip(min=1e-8)
    cn = np.linalg.norm(corpus, axis=1, keepdims=True).clip(min=1e-8)
    return dots / pn / cn.T


def dense_cosine_matrix(probes: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    return probes @ corpus.T  # already L2-normalized


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab",            default="../nexus/data/vocab.txt")
    p.add_argument("--vocab-embeddings", default="../nexus/data/vocab-embeddings.f32.bin")
    p.add_argument("--configs", nargs="+", required=True,
                   help="label:path pairs e.g. k3-21x:modal-out-k3-21x/sae-weights.bin")
    p.add_argument("--n-probes",  type=int, default=500)
    p.add_argument("--top-n",     type=int, default=20)
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    vocab = [l.strip() for l in Path(args.vocab).read_text().splitlines() if l.strip()]
    n_vocab = len(vocab)
    dims = 768
    print(f"Vocab: {n_vocab:,} terms")

    emb = load_vocab_embeddings(Path(args.vocab_embeddings), n_vocab, dims)
    print(f"Embeddings loaded: {emb.shape}")

    rng = np.random.default_rng(args.seed)
    probe_idx = rng.choice(n_vocab, size=args.n_probes, replace=False)
    probe_emb = emb[probe_idx]

    # Dense oracle: top-N neighbors for each probe (excluding self)
    dense_sim = dense_cosine_matrix(probe_emb, emb)
    dense_sim[np.arange(args.n_probes), probe_idx] = -2.0  # exclude self
    dense_top = np.argsort(-dense_sim, axis=1)[:, :args.top_n]
    dense_sets = [set(row) for row in dense_top]

    print(f"\nProbes: {args.n_probes}  Top-N: {args.top_n}\n")
    print(f"  {'config':<14}  {'k':>4}  {'recall@'+str(args.top_n):>10}  {'median':>8}  {'p25':>6}  {'p75':>6}")
    print(f"  {'-'*14}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*6}")

    for spec in args.configs:
        label, wpath = spec.split(":", 1)
        sae = parse_sae(Path(wpath).read_bytes())
        k = sae["k"]

        # Encode probes (small — n_probes × d_hidden)
        sparse_probes = sae_encode(sae, probe_emb)
        probe_norms = np.linalg.norm(sparse_probes, axis=1, keepdims=True).clip(min=1e-8)

        # Chunked cosine similarity — never materialise the full (n_vocab, d_hidden) matrix
        CHUNK = 1024
        top_scores = np.full((args.n_probes, args.top_n), -2.0, dtype=np.float32)
        top_indices = np.zeros((args.n_probes, args.top_n), dtype=np.int32)

        for start in range(0, n_vocab, CHUNK):
            end = min(start + CHUNK, n_vocab)
            chunk_sparse = sae_encode(sae, emb[start:end])          # (chunk, d_hidden)
            chunk_norms  = np.linalg.norm(chunk_sparse, axis=1, keepdims=True).clip(min=1e-8)
            sim = (sparse_probes @ chunk_sparse.T) / probe_norms / chunk_norms.T  # (n_probes, chunk)

            # Exclude self
            for pi, gi in enumerate(probe_idx):
                if start <= gi < end:
                    sim[pi, gi - start] = -2.0

            # Merge into running top-N
            combined_scores  = np.concatenate([top_scores, sim], axis=1)
            combined_indices = np.concatenate([
                top_indices,
                np.arange(start, end, dtype=np.int32)[np.newaxis, :].repeat(args.n_probes, axis=0)
            ], axis=1)
            top_k_pos = np.argpartition(-combined_scores, args.top_n, axis=1)[:, :args.top_n]
            top_scores  = np.take_along_axis(combined_scores,  top_k_pos, axis=1)
            top_indices = np.take_along_axis(combined_indices, top_k_pos, axis=1)

        recalls = np.array([
            len(set(top_indices[i]) & dense_sets[i]) / args.top_n
            for i in range(args.n_probes)
        ])
        print(f"  {label:<14}  {k:>4}  {recalls.mean():>9.1%}   {np.median(recalls):>7.1%}  {np.percentile(recalls,25):>5.1%}  {np.percentile(recalls,75):>5.1%}")


if __name__ == "__main__":
    main()
