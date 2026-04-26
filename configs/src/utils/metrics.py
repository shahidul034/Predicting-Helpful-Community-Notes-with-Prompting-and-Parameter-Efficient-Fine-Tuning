from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)


def compute_all_metrics(
    y_true: List[int], y_pred: List[int], y_prob: List[float]
) -> Dict[str, float]:
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc_roc"] = float("nan")

    try:
        metrics["auc_pr"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["auc_pr"] = float("nan")

    return metrics


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def print_classification_report(
    y_true: List[int], y_pred: List[int], target_names: List[str] = None
) -> str:
    if target_names is None:
        target_names = ["Not Helpful", "Helpful"]
    return classification_report(y_true, y_pred, target_names=target_names)
