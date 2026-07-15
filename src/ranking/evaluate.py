
from __future__ import annotations

import math

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


class ClassificationMetrics:
    def __init__(self, accuracy: float, precision: float, recall: float, f1: float, auc: float) -> None:
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.auc = auc

    def __str__(self) -> str:
        return (
            f"acc={self.accuracy:.4f} precision={self.precision:.4f} "
            f"recall={self.recall:.4f} f1={self.f1:.4f} auc={self.auc:.4f}"
        )


def _to_device(features: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in features.items()}


def _collect_scores_and_labels(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[float], list[float]]:
    """Runs one forward pass over the loader, returns raw sigmoid scores +
    true labels. Batches come as {"target": {...}, "candidate": {...},
    "label": Tensor} -- see dataset.py's ranking_collate_fn."""
    model.eval()
    scores: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for batch in loader:
            target = _to_device(batch["target"], device)
            candidate = _to_device(batch["candidate"], device)
            preds = model(target, candidate)
            scores.extend(preds.cpu().tolist())
            labels.extend(batch["label"].cpu().tolist())
    return scores, labels


def evaluate_classification_metrics(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5
) -> ClassificationMetrics:
    """Computes accuracy/precision/recall/F1 (at `threshold`) plus AUC via
    scikit-learn. Note: 0.5 is a naive default -- see find_best_threshold()
    for calibrating properly on a validation set."""
    scores, labels = _collect_scores_and_labels(model, loader, device)
    preds = [1 if s >= threshold else 0 for s in scores]

    return ClassificationMetrics(
        accuracy=accuracy_score(labels, preds),
        precision=precision_score(labels, preds, zero_division=0),
        recall=recall_score(labels, preds, zero_division=0),
        f1=f1_score(labels, preds, zero_division=0),
        auc=roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan"),
    )


def find_best_threshold(scores: list[float], labels: list[float]) -> tuple[float, float]:
    """Scans thresholds 0.01-0.99, returns the one maximizing F1. Must be
    run on the VALIDATION set, never on test (that would be leakage)."""
    best_threshold, best_f1 = 0.5, -1.0
    for i in range(1, 100):
        t = i / 100
        preds = [1 if s >= t else 0 for s in scores]
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = t, f1
    return best_threshold, best_f1


def precision_at_k(relevant: list[bool], k: int) -> float:
    """Fraction of the top-k results that are relevant."""
    top_k = relevant[:k]
    if not top_k:
        return 0.0
    return sum(top_k) / len(top_k)


def ndcg_at_k(relevant: list[bool], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ k for binary relevance."""
    top_k = relevant[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, rel in enumerate(top_k) if rel)
    ideal_hits = min(sum(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg