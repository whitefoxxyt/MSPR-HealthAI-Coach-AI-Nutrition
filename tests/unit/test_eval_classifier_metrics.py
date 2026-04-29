from __future__ import annotations

from scripts.eval.classifier_metrics import (
    confusion_matrix,
    per_class_metrics,
    top_k_accuracy,
)


def test_top_k_accuracy_all_truth_in_topk() -> None:
    predictions = [
        ["pizza", "lasagna", "tiramisu"],
        ["sushi", "tacos", "ramen"],
    ]
    truths = ["pizza", "sushi"]

    assert top_k_accuracy(predictions, truths, k=3) == 1.0


def test_top_k_accuracy_top1_vs_top5() -> None:
    # Top-1 ne capture que la premiere prediction, top-5 voit la verite plus loin.
    predictions = [
        ["pizza", "lasagna", "tiramisu", "tacos", "ramen"],
        ["lasagna", "pizza", "tacos", "tiramisu", "ramen"],
    ]
    truths = ["pizza", "pizza"]

    assert top_k_accuracy(predictions, truths, k=1) == 0.5
    assert top_k_accuracy(predictions, truths, k=5) == 1.0


def test_top_k_accuracy_no_match() -> None:
    predictions = [["lasagna", "tacos"]]
    truths = ["pizza"]

    assert top_k_accuracy(predictions, truths, k=2) == 0.0


def test_per_class_metrics_perfect_predictions() -> None:
    truths = ["pizza", "sushi", "pizza", "sushi"]
    top1 = ["pizza", "sushi", "pizza", "sushi"]

    metrics = per_class_metrics(top1, truths)

    assert metrics["pizza"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 2,
    }
    assert metrics["sushi"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 2,
    }


def test_per_class_metrics_one_misclassification() -> None:
    # 2 pizzas reelles : 1 reconnue comme pizza, 1 confondue avec sushi.
    # 2 sushis reels : 2 reconnus comme sushi.
    # Pour pizza : TP=1 FN=1 FP=0 -> P=1.0 R=0.5 F1=2/3
    # Pour sushi : TP=2 FN=0 FP=1 -> P=2/3 R=1.0 F1=4/5
    truths = ["pizza", "pizza", "sushi", "sushi"]
    top1 = ["pizza", "sushi", "sushi", "sushi"]

    metrics = per_class_metrics(top1, truths)

    assert metrics["pizza"]["precision"] == 1.0
    assert metrics["pizza"]["recall"] == 0.5
    assert round(metrics["pizza"]["f1"], 4) == round(2 / 3, 4)
    assert metrics["pizza"]["support"] == 2

    assert round(metrics["sushi"]["precision"], 4) == round(2 / 3, 4)
    assert metrics["sushi"]["recall"] == 1.0
    assert round(metrics["sushi"]["f1"], 4) == 0.8
    assert metrics["sushi"]["support"] == 2


def test_per_class_metrics_class_with_zero_predictions_has_zero_precision() -> None:
    # ramen apparait dans la verite mais n'est jamais predit -> precision indefinie
    # convention : precision = 0.0 dans ce cas (pas de FP/TP -> 0/0 -> 0).
    truths = ["pizza", "ramen"]
    top1 = ["pizza", "pizza"]

    metrics = per_class_metrics(top1, truths)

    assert metrics["ramen"]["precision"] == 0.0
    assert metrics["ramen"]["recall"] == 0.0
    assert metrics["ramen"]["f1"] == 0.0
    assert metrics["ramen"]["support"] == 1


def test_confusion_matrix_counts_per_truth_pred_pair() -> None:
    # Format : matrix[truth_idx][pred_idx]
    truths = ["pizza", "pizza", "sushi", "sushi"]
    top1 = ["pizza", "sushi", "sushi", "sushi"]
    labels = ["pizza", "sushi"]

    matrix = confusion_matrix(top1, truths, labels)

    # pizza ligne 0 : 1 predit pizza, 1 predit sushi
    # sushi ligne 1 : 0 predit pizza, 2 predit sushi
    assert matrix == [[1, 1], [0, 2]]


def test_confusion_matrix_ignores_predictions_outside_labels() -> None:
    # Si une prediction n'est pas dans le label set fourni, elle est ignoree
    # (case d'usage : limiter la matrice aux N classes les plus frequentes).
    truths = ["pizza", "pizza"]
    top1 = ["pizza", "ramen"]
    labels = ["pizza"]

    matrix = confusion_matrix(top1, truths, labels)

    assert matrix == [[1]]
