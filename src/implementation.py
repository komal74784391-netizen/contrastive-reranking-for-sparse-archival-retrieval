"""Sparse archival retrieval with contrastive reranking."""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class Candidate:
    doc_id: str
    dense_score: float
    sparse_score: float


def normalize(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    span = values.max() - values.min()
    if span == 0:
        return np.zeros_like(values)
    return (values - values.min()) / span


def fuse_scores(candidates: Iterable[Candidate], sparse_weight: float = 0.25) -> list[Candidate]:
    """Blend dense and sparse retrieval scores into a single ranking."""
    items = list(candidates)
    dense = normalize([c.dense_score for c in items])
    sparse = normalize([c.sparse_score for c in items])
    ranked = sorted(
        items,
        key=lambda c: -float(
            (1.0 - sparse_weight) * dense[items.index(c)] + sparse_weight * sparse[items.index(c)]
        ),
    )
    return ranked


def recall_at_k(ranked: Sequence[Candidate], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for c in ranked[:k] if c.doc_id in relevant)
    return hits / len(relevant)
