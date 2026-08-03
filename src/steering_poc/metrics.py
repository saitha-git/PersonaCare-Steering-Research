"""Quantitative metrics for steering evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def logit_metrics(
    baseline_logits: torch.Tensor, steered_logits: torch.Tensor, top_k: int = 10
) -> dict:
    """Compare next-token logits ([V] or [B, V]) between baseline and steered runs."""
    a = baseline_logits.float().reshape(-1, baseline_logits.shape[-1])
    b = steered_logits.float().reshape(-1, steered_logits.shape[-1])

    diff = (a - b).abs()
    cos = F.cosine_similarity(a, b, dim=-1).mean().item()

    top_a = a.topk(top_k, dim=-1).indices
    top_b = b.topk(top_k, dim=-1).indices
    agree = []
    for row_a, row_b in zip(top_a, top_b):
        agree.append(len(set(row_a.tolist()) & set(row_b.tolist())) / top_k)

    log_p = F.log_softmax(a, dim=-1)
    log_q = F.log_softmax(b, dim=-1)
    kl = F.kl_div(log_q, log_p, log_target=True, reduction="batchmean").item()

    return {
        "max_abs_logit_diff": diff.max().item(),
        "mean_abs_logit_diff": diff.mean().item(),
        "logit_cosine": cos,
        f"top{top_k}_agreement": sum(agree) / len(agree),
        "kl_baseline_vs_steered": kl,
    }


def verbosity_score(text: str) -> float:
    """Behavioral score for the verbosity concept: word count of the response."""
    return float(len(text.split()))


def bootstrap_mean_ci(values, n_boot: int = 10000, ci: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI for the mean of `values` (resampling over prompts)."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(arr.mean()), float(lo), float(hi)


def spearman(x, y):
    """Spearman rank correlation (average ranks for ties; no scipy needed)."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return float("nan")

    def _rank(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, a.size + 1)
        # average ranks over ties
        for v in np.unique(a):
            m = a == v
            ranks[m] = ranks[m].mean()
        return ranks

    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


@torch.no_grad()
def sequence_nll(model, tokenizer, prompt_ids: torch.Tensor, text: str) -> float:
    """Fluency proxy: mean per-token NLL of `text` under the UNSTEERED model,
    conditioned on the prompt. Lower = more fluent according to the model."""
    device = prompt_ids.device
    cont = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    cont_ids = cont.input_ids.to(device)
    if cont_ids.shape[1] == 0:
        return float("nan")
    full = torch.cat([prompt_ids, cont_ids], dim=1)
    logits = model(full).logits
    # Predicting continuation tokens only.
    start = prompt_ids.shape[1]
    pred = logits[:, start - 1 : full.shape[1] - 1, :]
    loss = F.cross_entropy(
        pred.reshape(-1, pred.shape[-1]).float(), cont_ids.reshape(-1)
    )
    return loss.item()
