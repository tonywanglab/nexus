"""
Modal pipeline: LLM autointerp labeling via SAE decoder-direction × vocab-embedding similarity.

For each SAE decoder atom:
  1. Compute cosine similarity between the (L2-normalized) decoder direction and all vocab
     embeddings — done locally in numpy (no GPU needed).
  2. Take the top-20 vocab terms per atom as interpretability evidence.
  3. Prompt Qwen 2.5 7B Instruct (on Modal) to synthesize a 3-10 word concept label.

With --broad-vocab, augments the labeling vocab with ConceptNet + WordNet terms
(fetched + embedded on Modal, cached locally for re-runs). WikiTitles are excluded
because they are too entity-specific to produce good atom labels.

Usage:
  modal run scripts/modal_autointerp_vocab.py \\
    --weights assets/sae-weights-v2.bin \\
    --out assets/sae-feature-labels-v2.json

  # With broader vocab (first run fetches + embeds ~1.5M extra terms):
  modal run scripts/modal_autointerp_vocab.py \\
    --weights assets/sae-weights-v2.bin \\
    --out assets/sae-feature-labels-v2.json \\
    --broad-vocab
"""
from __future__ import annotations

import gzip
import json
import random
import re
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import modal
import numpy as np

LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL_ID = "google/embeddinggemma-300m"
EMBED_DIMS = 768
CONCEPTNET_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"

# Default paths for the cached broad-vocab files
DEFAULT_EXTRA_VOCAB_PATH = "data/broad-vocab-extra.txt"
DEFAULT_EXTRA_VOCAB_EMB_PATH = "data/broad-vocab-extra-embeddings.f32.bin"

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

app = modal.App("nexus-autointerp-vocab")


# ── Helpers ────────────────────────────────────────────────────────────────────

def iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_sae_header(data: bytes) -> dict:
    magic = data[:4].decode("ascii")
    if magic != "SAE1":
        raise ValueError(f"Bad SAE magic: {magic!r}")
    header_len = struct.unpack_from("<I", data, 4)[0]
    return json.loads(data[8: 8 + header_len])


def _extract_wdec_normalized(data: bytes) -> tuple[int, int, int, "np.ndarray"]:
    """Return (d_model, d_hidden, k, w_dec_normalized) from SAE1 binary."""
    h = _parse_sae_header(data)
    d_model, d_hidden, k = h["dModel"], h["dHidden"], h["k"]
    off = 8 + struct.unpack_from("<I", data, 4)[0]
    off += d_hidden * d_model * 4  # skip wEnc
    off += d_hidden * 4            # skip bEnc
    off += d_model * 4             # skip bPre
    w_dec = np.frombuffer(data, dtype=np.float32, count=d_hidden * d_model, offset=off).reshape(d_hidden, d_model).copy()
    norms = np.linalg.norm(w_dec, axis=1, keepdims=True)
    w_dec /= np.maximum(norms, 1e-8)
    return d_model, d_hidden, k, w_dec


def _clean_label(raw: str) -> str:
    label = raw.strip().split("\n")[0].strip()
    for prefix in (
        "Concept:", "concept:", "Label:", "label:", "Answer:", "answer:",
        "Common concept:", "Common theme:", "Common thread:", "Theme:",
    ):
        if label.startswith(prefix):
            label = label[len(prefix):].strip()
            break
    if len(label) >= 2 and label[0] in ('"', "'") and label[-1] == label[0]:
        label = label[1:-1].strip()
    return label


# ── Broad-vocab helpers (mirrored from modal_train_sae_v2.py) ─────────────────

_STRIP_RE = re.compile(r"[_\-]+")
_PAREN_RE = re.compile(r"\([^)]*\)")
_MULTI_RE = re.compile(r"\s{2,}")

def normalize_short_phrase(raw: str, *, max_tokens: int = 5, max_chars: int = 60) -> str | None:
    text = _PAREN_RE.sub("", raw.replace("_", " ").replace("-", " "))
    text = _MULTI_RE.sub(" ", text).strip().lower()
    if not text or len(text) > max_chars:
        return None
    if any(c in text for c in "([{<"):
        return None
    tokens = text.split()
    if len(tokens) > max_tokens or len(tokens) == 0:
        return None
    return text


def _fetch_conceptnet(quota: int, seen: set[str]) -> list[str]:
    import requests
    print(f"[conceptnet] downloading assertions (~300 MB, quota={quota:,})...")
    resp = requests.get(CONCEPTNET_URL, stream=True, timeout=300,
                        headers={"User-Agent": "nexus-autointerp/1.0"})
    resp.raise_for_status()
    raw_gz = b"".join(resp.iter_content(1 << 20))
    lines = gzip.decompress(raw_gz).decode("utf-8", errors="replace").splitlines()
    print(f"[conceptnet] {len(lines):,} assertion lines")

    _en_re = re.compile(r"/c/en/([^/\t\s]+)")
    result: list[str] = []
    for line in lines:
        if len(result) >= quota:
            break
        for m in _en_re.finditer(line):
            if len(result) >= quota:
                break
            norm = normalize_short_phrase(m.group(1), max_tokens=4, max_chars=50)
            if norm and norm not in seen:
                seen.add(norm)
                result.append(norm)

    print(f"[conceptnet] kept {len(result):,} unique English concepts")
    return result


def _fetch_wordnet(quota: int, seen: set[str]) -> list[str]:
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn

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


# ── Modal: fetch extra vocab (ConceptNet + WordNet, no WikiTitles) ─────────────

@app.function(image=image, timeout=3600)
def fetch_extra_vocab_remote(existing_vocab: list[str]) -> list[str]:
    """
    Fetch ConceptNet English nodes + WordNet lemma names, deduplicated against
    the existing Numberbatch vocab. Returns only the *new* terms.
    WikiTitles are intentionally excluded — too entity-specific for labeling.
    """
    seen = set(existing_vocab)
    cn = _fetch_conceptnet(1_500_000, seen)
    wn = _fetch_wordnet(150_000, seen)
    extra = cn + wn
    print(f"Total extra terms: {len(extra):,}")
    return extra


# ── Modal: embed a batch of terms with Gemma-300m ─────────────────────────────

@app.function(
    image=image,
    gpu="L4",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def embed_batch_remote(terms: list[str]) -> bytes:
    """
    Embed terms with google/embeddinggemma-300m (mean-pool + L2-normalize),
    matching the normalization used during SAE training. Returns raw float32 bytes.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    mdl = AutoModel.from_pretrained(EMBED_MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    mdl.eval()

    BATCH = 256
    all_embs = []
    for i in tqdm(range(0, len(terms), BATCH), desc="embed"):
        batch = terms[i: i + BATCH]
        enc = tok(batch, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            last = mdl(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        normed = F.normalize(pooled, p=2, dim=1).cpu().numpy().astype(np.float32)
        all_embs.append(normed)

    emb = np.concatenate(all_embs, axis=0)
    return emb.tobytes()


# ── Broad-vocab preparation (local orchestration) ─────────────────────────────

def prepare_broad_vocab(
    existing_vocab: list[str],
    extra_vocab_path: Path,
    extra_vocab_emb_path: Path,
    embed_chunk: int = 50_000,
) -> None:
    """
    Fetch + embed ConceptNet/WordNet extra vocab if not already cached.
    Idempotent: skips if both output files already exist and are consistent.
    """
    if extra_vocab_path.exists() and extra_vocab_emb_path.exists():
        n_terms = sum(1 for _ in extra_vocab_path.open() if _.strip())
        n_emb = extra_vocab_emb_path.stat().st_size // (EMBED_DIMS * 4)
        if n_terms == n_emb and n_terms > 0:
            print(f"Using cached extra vocab: {n_terms:,} terms + embeddings")
            return
        print(f"Cache inconsistent (terms={n_terms}, emb rows={n_emb}), rebuilding...")

    print("=== Fetching ConceptNet + WordNet extra vocab on Modal ===")
    extra_terms = fetch_extra_vocab_remote.remote(existing_vocab)
    print(f"Received {len(extra_terms):,} new terms")

    extra_vocab_path.parent.mkdir(parents=True, exist_ok=True)
    extra_vocab_path.write_text("\n".join(extra_terms))
    print(f"Saved extra vocab to {extra_vocab_path}")

    print(f"=== Embedding {len(extra_terms):,} extra terms on Modal (chunks of {embed_chunk:,}) ===")
    n_chunks = (len(extra_terms) + embed_chunk - 1) // embed_chunk
    with open(extra_vocab_emb_path, "wb") as f:
        for ci, chunk_start in enumerate(range(0, len(extra_terms), embed_chunk)):
            chunk = extra_terms[chunk_start: chunk_start + embed_chunk]
            print(f"  chunk {ci + 1}/{n_chunks} ({len(chunk):,} terms)...")
            emb_bytes = embed_batch_remote.remote(chunk)
            f.write(emb_bytes)

    n_saved = extra_vocab_emb_path.stat().st_size // (EMBED_DIMS * 4)
    print(f"Saved {n_saved:,} extra embeddings to {extra_vocab_emb_path}")


# ── Cosine sim: top-K vocab terms per atom ────────────────────────────────────

def compute_topk_vocab_local(
    sae_bytes: bytes,
    vocab: list[str],
    vocab_emb_path: Path,
    top_k: int,
    min_sim: float,
    extra_vocab: list[str] | None = None,
    extra_vocab_emb_path: Path | None = None,
) -> tuple[int, list[list[str]]]:
    """
    For each SAE atom, find the top-K cosine-similar vocab terms by comparing
    the (normalized) decoder direction against all vocab embeddings.

    Streams both the primary and (optionally) extra vocab embeddings in chunks
    to keep peak memory low. Returns (d_hidden, atom_vocab_terms).
    """
    from tqdm import tqdm

    d_model, d_hidden, k, w_dec = _extract_wdec_normalized(sae_bytes)

    print(f"SAE: dHidden={d_hidden}, dModel={d_model}, k={k}")
    n_primary = len(vocab)
    n_extra = len(extra_vocab) if extra_vocab else 0
    n_total = n_primary + n_extra
    print(f"Vocab: {n_primary:,} primary + {n_extra:,} extra = {n_total:,} total  |  top_k={top_k}, min_sim={min_sim}")

    best_scores = np.full((d_hidden, top_k), -2.0, dtype=np.float32)
    best_indices = np.zeros((d_hidden, top_k), dtype=np.int32)

    CHUNK = 4096

    def _stream_pass(emb_path: Path, offset: int, n_terms: int, desc: str) -> None:
        t0 = time.time()
        with open(emb_path, "rb") as f:
            for chunk_start in tqdm(range(0, n_terms, CHUNK), desc=desc):
                chunk_size = min(CHUNK, n_terms - chunk_start)
                raw = f.read(chunk_size * d_model * 4)
                actual = len(raw) // (d_model * 4)
                if actual == 0:
                    break
                vocab_chunk = np.frombuffer(raw[: actual * d_model * 4], dtype=np.float32).reshape(actual, d_model)
                sim = w_dec @ vocab_chunk.T
                k_c = min(top_k, actual)
                part = np.argpartition(-sim, k_c - 1, axis=1)[:, :k_c] if k_c < actual else np.tile(np.arange(actual), (d_hidden, 1))
                chunk_scores = np.take_along_axis(sim, part, axis=1)
                chunk_indices = (offset + chunk_start + part).astype(np.int32)
                merged_s = np.concatenate([best_scores, chunk_scores], axis=1)
                merged_i = np.concatenate([best_indices, chunk_indices], axis=1)
                total = merged_s.shape[1]
                part2 = np.argpartition(-merged_s, top_k - 1, axis=1)[:, :top_k] if total > top_k else np.tile(np.arange(total), (d_hidden, 1))
                best_scores[:] = np.take_along_axis(merged_s, part2, axis=1)
                best_indices[:] = np.take_along_axis(merged_i, part2, axis=1)
        print(f"{desc} done in {time.time() - t0:.1f}s")

    _stream_pass(vocab_emb_path, 0, n_primary, "cosine sim (primary)")
    if extra_vocab and extra_vocab_emb_path:
        _stream_pass(extra_vocab_emb_path, n_primary, n_extra, "cosine sim (extra)")

    order = np.argsort(-best_scores, axis=1)
    best_scores = np.take_along_axis(best_scores, order, axis=1)
    best_indices = np.take_along_axis(best_indices, order, axis=1)

    all_vocab = vocab + (extra_vocab or [])
    half_min = min_sim / 2.0
    result: list[list[str]] = []
    qualified = 0
    for atom in range(d_hidden):
        if float(best_scores[atom, 0]) < min_sim:
            result.append([])
        else:
            terms = [
                all_vocab[int(best_indices[atom, j])]
                for j in range(top_k)
                if float(best_scores[atom, j]) >= half_min
            ]
            result.append(terms)
            qualified += 1

    print(f"Atoms with top-sim >= {min_sim}: {qualified:,}/{d_hidden:,}")
    return d_hidden, result


# ── Modal: LLM autointerp ─────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="L4",
    timeout=4 * 3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def autointerp_vocab_remote(
    atom_vocab_terms: list[list[str]],
    d_hidden: int,
    llm_batch_size: int,
    seed: int,
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    rng = random.Random(seed)

    SYSTEM = (
        "You are a machine learning interpretability assistant. "
        "Identify the single common concept that unites the given words or phrases. "
        "Reply with a phrase of 3-10 words only — no explanation, no quotes, no punctuation."
    )

    def build_prompt(terms: list[str]) -> str:
        shuffled = list(terms)
        rng.shuffle(shuffled)
        lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(shuffled))
        return (
            "These words and phrases all strongly activate the same neural network feature.\n"
            f"What single concept do they share?\n\n{lines}\n\nConcept:"
        )

    to_label: list[tuple[int, str]] = [
        (atom_idx, build_prompt(terms))
        for atom_idx, terms in enumerate(atom_vocab_terms)
        if terms
    ]
    print(f"Labeling {len(to_label):,} atoms with {LLM_MODEL_ID}...")

    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    mdl.eval()
    print(f"LLM loaded in {time.time() - t_load:.1f}s")

    atom_labels: dict[int, str] = {}
    t_llm = time.time()

    for batch_start in tqdm(range(0, len(to_label), llm_batch_size), desc="labeling"):
        batch = to_label[batch_start: batch_start + llm_batch_size]
        atom_indices = [x[0] for x in batch]
        prompts = [x[1] for x in batch]
        chat_texts = [
            tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        inputs = tok(chat_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = mdl.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, atom_idx in enumerate(atom_indices):
            raw = tok.decode(outputs[i, input_len:], skip_special_tokens=True)
            label = _clean_label(raw)
            if 3 <= len(label) <= 80:
                atom_labels[atom_idx] = label

    llm_wall = time.time() - t_llm
    print(f"LLM done in {llm_wall:.1f}s — {len(atom_labels):,} labels generated")

    labels = []
    for j in range(d_hidden):
        label = atom_labels.get(j)
        terms = atom_vocab_terms[j][:10] if j < len(atom_vocab_terms) else []
        if label:
            labels.append({"candidates": [label], "scores": [1.0], "topTerms": terms})
        else:
            labels.append({"candidates": [], "scores": [], "topTerms": terms})

    live_count = sum(1 for l in labels if l["candidates"])
    live_indices = [i for i, l in enumerate(labels) if l["candidates"]]
    sample = sorted(random.sample(live_indices, min(40, len(live_indices))))

    preview_lines = [
        "# SAE Feature Labels Preview (Autointerp / Vocab Decoder)",
        "",
        f"LLM: {LLM_MODEL_ID}  ",
        f"Method: decoder-direction × vocab-embeddings cosine similarity  ",
        f"Live features: {live_count} / {d_hidden}  ",
        "",
        "| Feature | Label | Top Terms |",
        "|---------|-------|-----------|",
    ] + [
        f"| {i} | {labels[i]['candidates'][0]} | {' · '.join(labels[i]['topTerms'][:5])} |"
        for i in sample
    ]

    labels_json = {
        "createdAt": iso_utc(),
        "dHidden": d_hidden,
        "vocabSize": 0,
        "vocabSource": f"autointerp-vocab-{LLM_MODEL_ID.split('/')[-1].lower()}",
        "minScore": 0,
        "labelsPerFeature": 1,
        "labels": labels,
    }

    return {
        "labels": labels_json,
        "preview": "\n".join(preview_lines) + "\n",
        "liveCount": live_count,
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    weights: str = "assets/sae-weights-v2.bin",
    vocab: str = "data/vocab.txt",
    vocab_embeddings: str = "data/vocab-embeddings.f32.bin",
    out: str = "assets/sae-feature-labels-v2.json",
    broad_vocab: bool = False,
    extra_vocab: str = DEFAULT_EXTRA_VOCAB_PATH,
    extra_vocab_embeddings: str = DEFAULT_EXTRA_VOCAB_EMB_PATH,
    top_k: int = 20,
    min_sim: float = 0.10,
    llm_batch_size: int = 16,
    seed: int = 42,
) -> None:
    weights_path = Path(weights)
    vocab_path = Path(vocab)
    vocab_emb_path = Path(vocab_embeddings)

    for p in [weights_path, vocab_path, vocab_emb_path]:
        if not p.exists():
            raise SystemExit(f"File not found: {p}")

    sae_bytes = weights_path.read_bytes()
    vocab_terms = [t for t in vocab_path.read_text().splitlines() if t.strip()]
    print(f"SAE weights: {len(sae_bytes):,} bytes")
    print(f"Primary vocab: {len(vocab_terms):,} terms")
    print(f"broad_vocab={broad_vocab}, top_k={top_k}, min_sim={min_sim}, llm_batch_size={llm_batch_size}")
    print()

    # Optionally build/load extra ConceptNet + WordNet vocab
    extra_vocab_terms: list[str] | None = None
    extra_vocab_emb_path: Path | None = None
    if broad_vocab:
        ev_path = Path(extra_vocab)
        ev_emb_path = Path(extra_vocab_embeddings)
        prepare_broad_vocab(vocab_terms, ev_path, ev_emb_path)
        extra_vocab_terms = [t for t in ev_path.read_text().splitlines() if t.strip()]
        extra_vocab_emb_path = ev_emb_path
        print()

    # Step 1: local cosine similarity
    print("=== Step 1: top-K vocab terms per atom (local numpy) ===")
    d_hidden, atom_vocab_terms = compute_topk_vocab_local(
        sae_bytes=sae_bytes,
        vocab=vocab_terms,
        vocab_emb_path=vocab_emb_path,
        top_k=top_k,
        min_sim=min_sim,
        extra_vocab=extra_vocab_terms,
        extra_vocab_emb_path=extra_vocab_emb_path,
    )
    qualified = sum(1 for t in atom_vocab_terms if t)
    print(f"Qualified atoms: {qualified:,}/{d_hidden:,}")
    print()

    # Step 2: Modal LLM inference
    print("=== Step 2: LLM autointerp on Modal ===")
    t0 = time.time()
    result = autointerp_vocab_remote.remote(
        atom_vocab_terms=atom_vocab_terms,
        d_hidden=d_hidden,
        llm_batch_size=llm_batch_size,
        seed=seed,
    )
    wall = time.time() - t0

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result["labels"]))

    preview_path = out_path.with_name(out_path.stem + "-preview.md")
    preview_path.write_text(result["preview"])

    print()
    print(f"=== done in {wall / 60:.1f} min ===")
    print(f"Live features: {result['liveCount']:,} / {d_hidden:,}")
    print(f"Labels:  {out}")
    print(f"Preview: {preview_path}")
    print()
    print("Spot-check:")
    labels = result["labels"]["labels"]
    live = [(i, l["candidates"][0]) for i, l in enumerate(labels) if l["candidates"]]
    step = max(1, len(live) // 5)
    for i, (atom_idx, label) in enumerate(live[::step][:5]):
        print(f"  feature {atom_idx}: {label!r}")
