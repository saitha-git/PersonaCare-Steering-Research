"""Shared helpers: config loading, dataset IO, hashing, artifact paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ARTIFACTS_DIR = Path("artifacts")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def vector_path(layer: int, tag: str = "") -> Path:
    suffix = f"_{tag}" if tag else ""
    return ARTIFACTS_DIR / f"vector_layer_{layer}{suffix}.pt"


def split_pairs(n: int, holdout_fraction: float, seed: int):
    """Deterministic train/holdout split of pair indices.

    Guarantees extraction pairs and evaluation pairs are disjoint.
    """
    import random

    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_holdout = max(1, int(round(n * holdout_fraction)))
    return sorted(idx[n_holdout:]), sorted(idx[:n_holdout])
