# Runtime Benchmark — Nexus Pipeline

Generated: 2026-04-26T04:06:29.270Z  
Host: darwin/arm64, Node v24.9.0, Apple M4 Pro  
Model: `onnx-community/embeddinggemma-300m-ONNX` device=`coreml` dtype=`q8`  
Vault: `/Users/tonywang/Development/nexus/senior-thesis-vault` (20 notes)  
Startup: vault load 16ms, model load 3430ms

## Headline (sorted by warm p50, ascending)

| Configuration | Cold p50 | Warm p50 | Warm p95 | Title embed (cold sum) | Phrase embed (warm sum) |
|---|---:|---:|---:|---:|---:|
| Span+LCS+Dense+Sparse | maxN=1 | MAX_EMBED=60 | STRONG_LCS=0.95 | thr=0.82 | 768.92ms | 1.48ms | 3.81ms | 796ms | 0ms |
| Span+LCS+Dense+Sparse | maxN=3 | MAX_EMBED=60 | STRONG_LCS=0.95 | thr=0.82 | 805.51ms | 1.83ms | 4.98ms | 807ms | 0ms |
| Span+LCS+Dense+Sparse | maxN=5 | MAX_EMBED=60 | STRONG_LCS=0.95 | thr=0.82 | 823.54ms | 2.18ms | 7.13ms | 834ms | 0ms |
| Span+LCS | maxN=10 | MAX_EMBED=60 | STRONG_LCS=0.95 | thr=0.82 | 2.42ms | 2.6ms | 15.11ms | 0ms | 0ms |
| Span+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=10 | STRONG_LCS=0.95 | thr=0.82 | 461.25ms | 2.6ms | 15.54ms | 814ms | 0ms |
| Span+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=30 | STRONG_LCS=0.95 | thr=0.82 | 626.37ms | 3.39ms | 15.59ms | 884ms | 0ms |
| Span+LCS+Dense | maxN=10 | MAX_EMBED=60 | STRONG_LCS=0.95 | thr=0.82 | 492.89ms | 3.46ms | 16.26ms | 851ms | 0ms |
| Span+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=60 | STRONG_LCS=0.95 | thr=0.82 | 814.25ms | 3.54ms | 16.27ms | 859ms | 0ms |
| Span+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=60 | STRONG_LCS=1 | thr=0.82 | 828.99ms | 3.61ms | 16.05ms | 918ms | 0ms |
| Span+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=60 | STRONG_LCS=0.85 | thr=0.82 | 824.06ms | 3.9ms | 17ms | 1028ms | 0ms |
| YAKE+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=60 | STRONG_LCS=0.95 | thr=0.82 | 901.13ms | 4.84ms | 65.67ms | 915ms | 0ms |
| Span+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=200 | STRONG_LCS=0.95 | thr=0.82 | 2034.2ms | 6.17ms | 19.65ms | 903ms | 0ms |
| Span+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=500 | STRONG_LCS=0.95 | thr=0.82 | 5297.34ms | 9.91ms | 24.4ms | 978ms | 0ms |

## Per-phase breakdown (warm pass, p50 ms)

| Configuration | extract | lcs | titleEmbed | phraseEmbed | denseMatch | sparseEncode | sparseMatch | merge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Span+LCS+Dense+Sparse | maxN=1 | MAX_EMBED=60 | STRONG_LCS=0.95 | thr=0.82 | 0.35 | 0.05 | 0 | 0 | 0.87 | 0.03 | 0.19 | 0 |
| Span+LCS+Dense+Sparse | maxN=3 | MAX_EMBED=60 | STRONG_LCS=0.95 | thr=0.82 | 0.38 | 0.34 | 0 | 0 | 0.91 | 0.03 | 0.2 | 0 |
| Span+LCS+Dense+Sparse | maxN=5 | MAX_EMBED=60 | STRONG_LCS=0.95 | thr=0.82 | 0.4 | 0.79 | 0 | 0 | 0.9 | 0.03 | 0.19 | 0 |
| Span+LCS | maxN=10 | MAX_EMBED=60 | STRONG_LCS=0.95 | thr=0.82 | 0.44 | 2.13 | 0 | 0 | 0 | 0 | 0 | 0 |
| Span+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=10 | STRONG_LCS=0.95 | thr=0.82 | 0.45 | 1.93 | 0 | 0 | 0.15 | 0.02 | 0.04 | 0 |
| Span+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=30 | STRONG_LCS=0.95 | thr=0.82 | 0.49 | 2.26 | 0 | 0 | 0.47 | 0.03 | 0.12 | 0 |
| Span+LCS+Dense | maxN=10 | MAX_EMBED=60 | STRONG_LCS=0.95 | thr=0.82 | 0.45 | 2.03 | 0 | 0 | 0.93 | 0 | 0 | 0 |
| Span+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=60 | STRONG_LCS=0.95 | thr=0.82 | 0.53 | 1.86 | 0 | 0 | 0.89 | 0.04 | 0.22 | 0.01 |
| Span+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=60 | STRONG_LCS=1 | thr=0.82 | 0.48 | 1.99 | 0 | 0 | 0.92 | 0.03 | 0.19 | 0 |
| Span+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=60 | STRONG_LCS=0.85 | thr=0.82 | 0.45 | 2.04 | 0 | 0 | 0.92 | 0.03 | 0.2 | 0.01 |
| YAKE+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=60 | STRONG_LCS=0.95 | thr=0.82 | 2.89 | 1.07 | 0 | 0 | 0.86 | 0.04 | 0.2 | 0.01 |
| Span+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=200 | STRONG_LCS=0.95 | thr=0.82 | 0.46 | 2 | 0 | 0 | 3.03 | 0.05 | 0.57 | 0.01 |
| Span+LCS+Dense+Sparse | maxN=10 | MAX_EMBED=500 | STRONG_LCS=0.95 | thr=0.82 | 0.45 | 2.06 | 0 | 0 | 6.39 | 0.08 | 1.19 | 0.01 |
