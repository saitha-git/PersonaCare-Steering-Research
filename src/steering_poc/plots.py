"""Dose-response plot: paired word-count change vs alpha with bootstrap 95% CIs.

Reads the aggregate JSON written by steering_poc.evaluate (censoring-aware:
only uncensored pairs contribute; censored counts are annotated in the data).

Usage:
    python -m steering_poc.plots [--data artifacts/eval_doseresponse.json]
        [--out artifacts/dose_response.png]
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (fixed slot order; CVD-checked).
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def plot_dose_response(data: dict, out_path: str, title: str | None = None):
    cells = data["cells"]
    layers = sorted({c["layer"] for c in cells})
    spearman = data.get("spearman_by_layer", {})

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    series, end_ys = [], []
    for i, L in enumerate(layers):
        pts = sorted(
            (c["alpha"], c["mean_paired_delta_words"],
             c["delta_ci95_low"], c["delta_ci95_high"])
            for c in cells if c["layer"] == L
        )
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        lo = [p[1] - p[2] for p in pts]
        hi = [p[3] - p[1] for p in pts]
        color = SERIES[i % len(SERIES)]
        rho = spearman.get(str(L), spearman.get(L, {})).get("rho")
        label = f"layer {L}" + (f"  (ρ={rho:+.2f})" if rho is not None else "")
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, linewidth=2, marker="o",
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=1,
                    capsize=3, elinewidth=1.25, label=label, zorder=3)
        series.append((L, xs[-1], ys[-1]))
        end_ys.append(ys[-1])

    # Direct end labels, pushed apart when line ends nearly coincide.
    span = (max(end_ys) - min(end_ys)) or 1.0
    min_gap = max(4.0, span * 0.06)
    placed: list[float] = []
    for L, x_end, y_end in sorted(series, key=lambda s: s[2]):
        y_lab = y_end
        while any(abs(y_lab - p) < min_gap for p in placed):
            y_lab += min_gap
        placed.append(y_lab)
        ax.annotate(f"layer {L}", (x_end, y_lab), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK_2)

    ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    ax.axvline(0, color=BASELINE, linewidth=1, zorder=1)
    ax.set_xlabel("steering strength alpha  (- concise <-> verbose +)",
                  color=INK_2, fontsize=10)
    ax.set_ylabel("paired Δ words vs alpha=0  (95% bootstrap CI)",
                  color=INK_2, fontsize=10)
    n = data.get("n_prompts", "?")
    ax.set_title(title or f"Dose-response: verbosity steering "
                          f"(Qwen3-0.6B, greedy, {n} prompts, uncensored pairs)",
                 color=INK, fontsize=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.75, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    ax.margins(x=0.14)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="artifacts/eval_doseresponse.json")
    parser.add_argument("--out", default="artifacts/dose_response.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = plot_dose_response(data, args.out, args.title)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
