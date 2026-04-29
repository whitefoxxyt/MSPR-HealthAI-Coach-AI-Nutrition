from __future__ import annotations

from collections import Counter


def top_k_accuracy(
    predictions: list[list[str]],
    truths: list[str],
    k: int,
) -> float:
    """Renvoie le ratio de samples ou la verite est dans le top-k de predictions.

    Convention : sur une liste vide, renvoie 0.0 plutot que de lever, par
    coherence avec json_validity_rate / fallback_rate / latency_percentiles.
    """
    if not truths:
        return 0.0
    hits = sum(1 for preds, truth in zip(predictions, truths) if truth in preds[:k])
    return hits / len(truths)


def per_class_metrics(
    top1: list[str],
    truths: list[str],
) -> dict[str, dict[str, float | int]]:
    """Calcule precision, rappel, F1 et support par classe (one-vs-rest).

    Convention : si une classe n'a aucun TP+FP elle a precision=0 ;
    si elle n'a aucun TP+FN elle a recall=0. Ces zeros propagent un F1=0.
    """
    classes = sorted(set(truths) | set(top1))
    truth_count = Counter(truths)
    pred_count = Counter(top1)
    tp_count: Counter[str] = Counter()
    for pred, truth in zip(top1, truths):
        if pred == truth:
            tp_count[pred] += 1

    metrics: dict[str, dict[str, float | int]] = {}
    for cls in classes:
        tp = tp_count[cls]
        fp = pred_count[cls] - tp
        fn = truth_count[cls] - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": truth_count[cls],
        }
    return metrics


def confusion_matrix(
    top1: list[str],
    truths: list[str],
    labels: list[str],
) -> list[list[int]]:
    """Renvoie une matrice [truth_idx][pred_idx] de comptes.

    Les samples dont la verite OU la prediction n'est pas dans `labels`
    sont ignores ; pratique pour zoomer sur un sous-ensemble de classes.
    """
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    for pred, truth in zip(top1, truths):
        if truth not in label_to_idx or pred not in label_to_idx:
            continue
        matrix[label_to_idx[truth]][label_to_idx[pred]] += 1
    return matrix
