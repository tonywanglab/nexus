"""Renders the result figures embedded in README.md.

Numbers are transcribed from the thesis evaluation tables (Tables 4.1-4.5), which
were produced by `npm run eval` and `npm run bench:runtime:full`.

Usage: python3 scripts/plot_readme_figures.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "readme")
BLUE = "#4a90d9"
ORANGE = "#f5a623"
RED = "#d95f5f"
GREEN = "#5aa469"


def extractor_tradeoff() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    names = ["SpanExtractor", "YAKE-lite"]

    candidates = [25564, 6840]
    bars = ax1.bar(names, candidates, color=[BLUE, ORANGE], width=0.55)
    ax1.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=10)
    ax1.set_ylabel("Candidate phrases generated")
    ax1.set_title("Resolver workload", fontweight="bold")
    ax1.set_ylim(0, 30000)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    missed = [0, 7]
    bars = ax2.bar(names, missed, color=[GREEN, RED], width=0.55)
    ax2.bar_label(bars, fmt="{:.0f}", padding=3, fontsize=10)
    ax2.set_ylabel("Ground-truth links missed")
    ax2.set_title("Recall cost (129 ground-truth links)", fontweight="bold")
    ax2.set_ylim(0, 9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    fig.suptitle("Candidate generation: 73% less work costs 7 links", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "extractor-tradeoff.png"), dpi=140)
    plt.close(fig)


def resolver_quality() -> None:
    configs = ["Span\n+ LCS", "Span\n+ Gemma", "YAKE\n+ LCS", "YAKE\n+ Gemma"]
    precision = [0.7910, 0.6750, 0.8030, 0.7290]
    recall = [1.0000, 1.0000, 0.9460, 0.9610]
    f1 = [0.8840, 0.8060, 0.8680, 0.8290]

    x = np.arange(len(configs))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for offset, values, color, label in [
        (-width, precision, ORANGE, "Precision"),
        (0.0, recall, BLUE, "Recall"),
        (width, f1, GREEN, "F1"),
    ]:
        bars = ax.bar(x + offset, values, width, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    ax.set_xticks(x, configs)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(
        "Extractor x resolver on the 78-note vault (129 ground-truth links)",
        fontweight="bold",
    )
    ax.legend(ncols=3, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "resolver-quality.png"), dpi=140)
    plt.close(fig)


def latency() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    stages = ["LCS only", "+ Dense", "+ Dense\n+ Sparse"]
    cold = [2.42, 493, 814]
    warm = [2.6, 3.46, 3.54]
    x = np.arange(len(stages))
    width = 0.34
    b1 = ax1.bar(x - width / 2, cold, width, color=ORANGE, label="Cold p50")
    b2 = ax1.bar(x + width / 2, warm, width, color=BLUE, label="Warm p50")
    ax1.bar_label(b1, fmt="%.4g ms", padding=3, fontsize=8)
    ax1.bar_label(b2, fmt="%.3g ms", padding=3, fontsize=8)
    ax1.set_yscale("log")
    ax1.set_ylim(1, 4000)
    ax1.set_xticks(x, stages)
    ax1.set_ylabel("Latency per note (ms, log scale)")
    ax1.set_title("Caching absorbs the embedding cost", fontweight="bold")
    ax1.legend(framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    pool = [10, 30, 60, 200, 500]
    cold_pool = [461, 626, 814, 2034, 5297]
    warm_pool = [2.6, 3.39, 3.54, 6.17, 9.91]
    ax2.plot(pool, cold_pool, "o-", color=ORANGE, label="Cold p50")
    ax2.plot(pool, warm_pool, "o-", color=BLUE, label="Warm p50")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xticks(pool, [str(p) for p in pool])
    ax2.axvline(60, color=GREEN, linestyle="--", alpha=0.8)
    ax2.annotate(
        "default = 60",
        xy=(60, 30),
        xytext=(66, 30),
        color=GREEN,
        fontsize=9,
        fontweight="bold",
    )
    ax2.set_xlabel("MAX_EMBED_PHRASES (titles embedded per note)")
    ax2.set_ylabel("Latency per note (ms, log scale)")
    ax2.set_title("Embedding pool size dominates cold cost", fontweight="bold")
    ax2.legend(framealpha=0.9)
    ax2.grid(alpha=0.3)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "latency.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    extractor_tradeoff()
    resolver_quality()
    latency()
    print(f"wrote figures to {OUT_DIR}")
