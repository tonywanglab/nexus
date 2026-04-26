"""
Modal pipeline: train SAE v2 on a broader short-phrase corpus.

Same distribution as v1 (short keyphrases ≤ 5 tokens — matches SpanExtractor runtime),
but with two additional sources beyond Numberbatch + Wikipedia titles:
  - ConceptNet English nodes  (~2M relational concept phrases)
  - WordNet synset lemma names (~150k precise concept terms)

Architecture is identical to v1 (dHidden=16128, k=6, dModel=768) so the v2 weights
are drop-in compatible with the existing sae.ts deserializer and all runtime code.

Usage:
  modal run scripts/modal_train_sae_v2.py \\
    --vocab data/vocab.txt \\
    --out-dir ./modal-out-v2

Copy outputs:
  cp modal-out-v2/sae-weights.bin assets/sae-weights-v2.bin
  cp modal-out-v2/sae-training-report.json assets/sae-training-report-v2.json
"""
from __future__ import annotations

import gzip
import json
import math
import re
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

MODEL_ID = "google/embeddinggemma-300m"
DIMS = 768
GPU = "L4"

DECODER_RENORM_EVERY = 100
WIKI_TITLES_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz"
CONCEPTNET_URL = "https://conceptnet.s3.amazonaws.com/downloads/2019/conceptnet-assertions-5.7.0.csv.gz"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.45",
        "numpy",
        "tqdm",
        "sentencepiece",
        "accelerate",
        "requests",
        "nltk",
    )
)

app = modal.App("nexus-train-sae-v2")


def iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def pack_sae_weights(d_model, d_hidden, k, w_enc, b_enc, b_pre, w_dec, b_dec) -> bytes:
    import numpy as np
    header = json.dumps({"dModel": d_model, "dHidden": d_hidden, "k": k}, separators=(",", ":"))
    raw = header.encode("utf-8")
    padding = (4 - ((8 + len(raw)) % 4)) % 4
    padded = raw + b" " * padding
    buf = bytearray()
    buf += b"SAE1"
    buf += struct.pack("<I", len(padded))
    buf += padded
    for arr in (w_enc, b_enc, b_pre, w_dec, b_dec):
        buf += np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    return bytes(buf)


# ── Phrase filters ────────────────────────────────────────────────────────────

_SKIP_WIKI_PREFIXES = (
    "wikipedia:", "file:", "template:", "category:", "portal:", "help:",
    "mediawiki:", "list of", "index of", "outline of", "glossary of",
    "history of", "timeline of",
)


def normalize_short_phrase(raw: str, *, max_tokens: int = 5, max_chars: int = 60) -> str | None:
    phrase = raw.strip().replace("_", " ").lower()
    if not phrase or len(phrase) < 2 or len(phrase) > max_chars:
        return None
    if any(c in phrase for c in "([{<"):
        return None
    tokens = phrase.split()
    if not (1 <= len(tokens) <= max_tokens):
        return None
    return phrase


def normalize_wiki_title(raw: str) -> str | None:
    title = raw.strip().replace("_", " ").lower()
    if not title:
        return None
    for pfx in _SKIP_WIKI_PREFIXES:
        if title.startswith(pfx):
            return None
    if title.endswith("(disambiguation)"):
        return None
    return normalize_short_phrase(title)


# ── Source fetchers ────────────────────────────────────────────────────────────

def fetch_wiki_titles(quota: int, seen: set[str]) -> list[str]:
    import requests
    print(f"[wiki-titles] downloading from Wikimedia dump (quota={quota:,})...")
    resp = requests.get(WIKI_TITLES_URL, stream=True, timeout=120,
                        headers={"User-Agent": "nexus-sae-v2/1.0"})
    resp.raise_for_status()
    lines = gzip.decompress(b"".join(resp.iter_content(1 << 20))).decode("utf-8", errors="replace").splitlines()
    print(f"[wiki-titles] {len(lines):,} raw lines")
    result: list[str] = []
    for line in lines:
        if len(result) >= quota:
            break
        norm = normalize_wiki_title(line)
        if norm and norm not in seen:
            seen.add(norm)
            result.append(norm)
    print(f"[wiki-titles] kept {len(result):,}")
    return result


def fetch_conceptnet(quota: int, seen: set[str]) -> list[str]:
    """Extract unique English concept nodes from ConceptNet 5 assertions."""
    import requests
    print(f"[conceptnet] downloading assertions (~300 MB, quota={quota:,})...")
    resp = requests.get(CONCEPTNET_URL, stream=True, timeout=300,
                        headers={"User-Agent": "nexus-sae-v2/1.0"})
    resp.raise_for_status()
    raw_gz = b"".join(resp.iter_content(1 << 20))
    print(f"[conceptnet] {len(raw_gz):,} bytes compressed, decompressing...")
    lines = gzip.decompress(raw_gz).decode("utf-8", errors="replace").splitlines()
    print(f"[conceptnet] {len(lines):,} assertion lines")

    # Each line: /a/[/r/...,/c/en/CONCEPT,...] \t /r/Rel \t /c/en/CONCEPT \t /c/en/CONCEPT \t {...}
    # Extract all /c/en/ node names from columns 2 and 3.
    _en_re = re.compile(r"/c/en/([^/\t\s]+)")
    result: list[str] = []
    for line in lines:
        if len(result) >= quota:
            break
        for m in _en_re.finditer(line):
            if len(result) >= quota:
                break
            raw_concept = m.group(1)
            norm = normalize_short_phrase(raw_concept, max_tokens=4, max_chars=50)
            if norm and norm not in seen:
                seen.add(norm)
                result.append(norm)

    print(f"[conceptnet] kept {len(result):,} unique English concepts")
    return result


def fetch_wordnet(quota: int, seen: set[str]) -> list[str]:
    """Collect synset lemma names and single-phrase definitions from WordNet."""
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn

    print(f"[wordnet] collecting lemma names (quota={quota:,})...")
    result: list[str] = []
    for ss in wn.all_synsets():
        if len(result) >= quota:
            break
        for lemma in ss.lemmas():
            if len(result) >= quota:
                break
            norm = normalize_short_phrase(lemma.name(), max_tokens=4, max_chars=40)
            if norm and norm not in seen:
                seen.add(norm)
                result.append(norm)
    print(f"[wordnet] kept {len(result):,}")
    return result


# ── Remote training function ───────────────────────────────────────────────────

@app.function(
    image=image,
    gpu=GPU,
    timeout=4 * 3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_sae_v2_remote(
    vocab_bytes: bytes,
    d_hidden: int,
    k: int,
    epochs: int,
    batch_size_embed: int,
    batch_size_train: int,
    lr: float,
    val_fraction: float,
    seed: int,
    wiki_quota: int,
    conceptnet_quota: int,
    wordnet_quota: int,
) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # ── 1. Assemble phrase corpus ──────────────────────────────────────────────
    vocab = [l.strip() for l in vocab_bytes.decode("utf-8").splitlines() if l.strip()]
    n_vocab = len(vocab)
    print(f"base vocab: {n_vocab:,} terms")

    seen: set[str] = set(vocab)
    combined: list[str] = list(vocab)

    for name, fn, quota in [
        ("wiki-titles",  lambda q: fetch_wiki_titles(q, seen),   wiki_quota),
        ("conceptnet",   lambda q: fetch_conceptnet(q, seen),    conceptnet_quota),
        ("wordnet",      lambda q: fetch_wordnet(q, seen),       wordnet_quota),
    ]:
        try:
            batch = fn(quota)
        except Exception as e:
            print(f"[{name}] WARNING: failed ({e}), skipping")
            batch = []
        combined.extend(batch)
        print(f"  [{name}] +{len(batch):,} phrases → total {len(combined):,}")

    n_total = len(combined)
    print(f"Total corpus: {n_total:,} short phrases")

    # ── 2. Load embedding model ────────────────────────────────────────────────
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    mdl.eval()
    print(f"model loaded in {time.time() - t_load:.1f}s")

    # ── 3. Embed all phrases ───────────────────────────────────────────────────
    t_embed = time.time()
    emb = torch.empty((n_total, DIMS), dtype=torch.float32, device=device)
    for i in tqdm(range(0, n_total, batch_size_embed), desc="embed"):
        batch = combined[i: i + batch_size_embed]
        enc = tok(batch, padding=True, truncation=True, return_tensors="pt")
        enc = {key: v.to(device) for key, v in enc.items()}
        with torch.no_grad():
            last = mdl(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        emb[i: i + len(batch)] = F.normalize(pooled, p=2, dim=1)
    embed_wall = time.time() - t_embed
    print(f"embedded {n_total:,} phrases in {embed_wall:.1f}s")

    del mdl, tok
    torch.cuda.empty_cache()

    # ── 4. Train/val split ─────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    perm = torch.from_numpy(rng.permutation(n_total)).to(device)
    n_val = min(4096, max(1, int(n_total * val_fraction)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    n_train = tr_idx.numel()
    x_val = emb.index_select(0, val_idx)
    print(f"train: {n_train:,}  val: {n_val:,}")

    # ── 5. Initialize SAE ─────────────────────────────────────────────────────
    d_model = DIMS
    pre_bias_init = emb.index_select(0, tr_idx[:min(4096, n_train)]).mean(dim=0)
    g = torch.Generator(device=device).manual_seed(seed)
    w_enc = nn.Parameter(torch.randn(d_hidden, d_model, generator=g, device=device) / math.sqrt(d_model))
    b_enc = nn.Parameter(torch.zeros(d_hidden, device=device))
    b_pre = nn.Parameter(pre_bias_init.clone())
    w_dec_init = torch.randn(d_model, d_hidden, generator=g, device=device) / math.sqrt(d_hidden)
    with torch.no_grad():
        w_dec_init.div_(w_dec_init.norm(dim=0, keepdim=True).clamp(min=1e-8))
    w_dec = nn.Parameter(w_dec_init)
    b_dec = nn.Parameter(torch.zeros(d_model, device=device))

    opt = torch.optim.Adam([w_enc, b_enc, b_pre, w_dec, b_dec], lr=lr)

    def forward(x):
        centered = x - b_pre
        pre = torch.addmm(b_enc, centered, w_enc.t())
        feats = torch.relu(pre)
        topv, topi = feats.topk(k, dim=1)
        mask = torch.zeros_like(feats).scatter_(1, topi, 1.0)
        sparse = feats * mask
        recon = torch.addmm(b_dec, sparse, w_dec.t()) + b_pre
        return sparse, recon

    # ── 6. Training loop ───────────────────────────────────────────────────────
    t_train = time.time()
    epoch_log = []
    fire_counts = torch.zeros(d_hidden, device=device)
    print(f"training: dHidden={d_hidden}, k={k}, epochs={epochs}, batch={batch_size_train}, lr={lr}")

    for ep in range(epochs):
        t_ep = time.time()
        shuf = torch.randperm(n_train, device=device)
        losses = []
        for bi, i in enumerate(range(0, n_train, batch_size_train)):
            xb = emb.index_select(0, tr_idx[shuf[i: i + batch_size_train]])
            sparse, recon = forward(xb)
            loss = F.mse_loss(recon, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if bi % DECODER_RENORM_EVERY == 0:
                with torch.no_grad():
                    w_dec.div_(w_dec.norm(dim=0, keepdim=True).clamp(min=1e-8))
            with torch.no_grad():
                fire_counts += (sparse > 0).sum(dim=0).float()

        with torch.no_grad():
            _, recon_val = forward(x_val)
            val_mse = ((recon_val - x_val) ** 2).mean().item()
            dec_norm = w_dec.norm(dim=0).mean().item()
        dead = int((fire_counts == 0).sum().item())
        row = {
            "epoch": ep + 1,
            "trainLoss": float(np.mean(losses)),
            "valLoss": val_mse,
            "avgL0": int(k),
            "meanDecColNorm": float(dec_norm),
            "deadFeatures": dead,
            "totalFeatures": int(d_hidden),
            "elapsedSec": round(time.time() - t_ep, 3),
        }
        epoch_log.append(row)
        print(f"  ep {ep+1}/{epochs}  train={row['trainLoss']:.5f}  val={val_mse:.5f}  dead={dead}/{d_hidden}  {row['elapsedSec']}s")

    with torch.no_grad():
        w_dec.div_(w_dec.norm(dim=0, keepdim=True).clamp(min=1e-8))
        _, recon_val = forward(x_val)
        val_mse_final = ((recon_val - x_val) ** 2).mean().item()

    train_wall = time.time() - t_train
    print(f"training done in {train_wall:.1f}s, final val MSE={val_mse_final:.5f}")

    # ── 7. Pack weights ────────────────────────────────────────────────────────
    w_enc_np = w_enc.detach().cpu().float().numpy()
    b_enc_np = b_enc.detach().cpu().float().numpy()
    b_pre_np = b_pre.detach().cpu().float().numpy()
    w_dec_np = w_dec.detach().cpu().float().numpy().T  # store as [dHidden, dModel]
    b_dec_np = b_dec.detach().cpu().float().numpy()

    weights_bytes = pack_sae_weights(d_model, d_hidden, k, w_enc_np, b_enc_np, b_pre_np, w_dec_np, b_dec_np)

    report = {
        "createdAt": iso_utc(),
        "dModel": d_model,
        "dHidden": d_hidden,
        "k": k,
        "epochs": epochs,
        "batchSizeEmbed": batch_size_embed,
        "batchSizeTrain": batch_size_train,
        "lr": lr,
        "corpusSize": n_total,
        "corpusSources": {
            "vocab": n_vocab,
            "wikiTitles": wiki_quota,
            "conceptnet": conceptnet_quota,
            "wordnet": wordnet_quota,
        },
        "finalValMSE": val_mse_final,
        "trainWallSec": round(train_wall, 1),
        "epochLog": epoch_log,
    }

    return {"weights": weights_bytes, "report": report}


@app.local_entrypoint()
def main(
    vocab: str = "data/vocab.txt",
    out_dir: str = "./modal-out-v2",
    d_hidden: int = 16128,
    k: int = 6,
    epochs: int = 25,
    batch_size_embed: int = 256,
    batch_size_train: int = 256,
    lr: float = 3e-4,
    val_fraction: float = 0.002,
    seed: int = 42,
    wiki_quota: int = 1_900_000,
    conceptnet_quota: int = 1_500_000,
    wordnet_quota: int = 150_000,
) -> None:
    vocab_path = Path(vocab)
    if not vocab_path.exists():
        raise SystemExit(f"Vocab not found: {vocab}")

    vocab_bytes = vocab_path.read_bytes()
    print(f"Vocab: {vocab_path} ({len(vocab_bytes):,} bytes)")
    print(f"d_hidden={d_hidden}, k={k}, epochs={epochs}, lr={lr}")
    print(f"Corpus quotas: wiki={wiki_quota:,}, conceptnet={conceptnet_quota:,}, wordnet={wordnet_quota:,}")
    print()

    t0 = time.time()
    result = train_sae_v2_remote.remote(
        vocab_bytes=vocab_bytes,
        d_hidden=d_hidden,
        k=k,
        epochs=epochs,
        batch_size_embed=batch_size_embed,
        batch_size_train=batch_size_train,
        lr=lr,
        val_fraction=val_fraction,
        seed=seed,
        wiki_quota=wiki_quota,
        conceptnet_quota=conceptnet_quota,
        wordnet_quota=wordnet_quota,
    )
    wall = time.time() - t0

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sae-weights.bin").write_bytes(result["weights"])
    (out / "sae-training-report.json").write_text(json.dumps(result["report"], indent=2))

    print()
    print(f"=== done in {wall / 60:.1f} min ===")
    print(f"Final val MSE: {result['report']['finalValMSE']:.5f}")
    print(f"Corpus: {result['report']['corpusSize']:,} short phrases")
    print(f"Outputs: {out}/sae-weights.bin, {out}/sae-training-report.json")
    print()
    print("Next steps:")
    print(f"  cp {out}/sae-weights.bin assets/sae-weights-v2.bin")
    print(f"  cp {out}/sae-training-report.json assets/sae-training-report-v2.json")
    print("  modal run scripts/modal_autointerp_vocab.py --weights assets/sae-weights-v2.bin --out assets/sae-feature-labels-v2.json")
