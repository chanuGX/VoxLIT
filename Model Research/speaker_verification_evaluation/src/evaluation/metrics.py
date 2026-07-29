from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _coerce_scores_frame(scores: pd.DataFrame, required_columns: set[str]) -> pd.DataFrame:
    missing = required_columns.difference(scores.columns)
    if missing:
        raise ValueError(f"Scores CSV is missing columns: {sorted(missing)}")

    frame = scores.copy()
    if "status" in frame.columns:
        frame = frame.loc[frame["status"] == "success"].copy()

    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame["similarity"] = pd.to_numeric(frame["similarity"], errors="coerce")
    frame = frame.dropna(subset=["label", "similarity"])

    if frame.empty:
        raise ValueError("No valid scores found.")

    frame["label"] = frame["label"].astype(int)

    invalid_labels = sorted(set(frame["label"].unique()).difference({0, 1}))
    if invalid_labels:
        raise ValueError(f"Invalid labels found: {invalid_labels}")

    if frame["label"].nunique() != 2:
        raise ValueError("Scores must contain both same-speaker and different-speaker pairs.")

    similarities = frame["similarity"].to_numpy(dtype=np.float64)
    if not np.isfinite(similarities).all():
        raise ValueError("Scores contain non-finite similarity values.")

    return frame


def load_valid_scores(scores_path: str | Path, required_columns: set[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(Path(scores_path))
    return _coerce_scores_frame(frame, required_columns or {"label", "similarity"})


def _validate_label_and_score_arrays(labels: np.ndarray, similarities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    similarities = np.asarray(similarities, dtype=np.float64).reshape(-1)

    if labels.size == 0:
        raise ValueError("No labels were provided.")

    if labels.size != similarities.size:
        raise ValueError("Labels and similarities must have the same length.")

    if not np.isfinite(similarities).all():
        raise ValueError("Similarities contain non-finite values.")

    invalid_labels = sorted(set(np.unique(labels)).difference({0, 1}))
    if invalid_labels:
        raise ValueError(f"Invalid labels found: {invalid_labels}")

    if np.unique(labels).size != 2:
        raise ValueError("Scores must contain both same-speaker and different-speaker pairs.")

    return labels, similarities


def calculate_roc_curve_data(labels: np.ndarray, similarities: np.ndarray) -> dict[str, list[float]]:
    labels, similarities = _validate_label_and_score_arrays(labels, similarities)
    fpr, tpr, thresholds = roc_curve(labels, similarities, pos_label=1, drop_intermediate=False)

    finite_mask = np.isfinite(thresholds)
    if not np.any(finite_mask):
        raise ValueError("No finite ROC thresholds were produced.")

    fpr = fpr[finite_mask]
    tpr = tpr[finite_mask]
    thresholds = thresholds[finite_mask]

    return {
        "fpr": [float(value) for value in fpr],
        "tpr": [float(value) for value in tpr],
        "thresholds": [float(value) for value in thresholds],
        "fnr": [float(value) for value in (1.0 - tpr)],
    }


def _roc_operating_points(labels: np.ndarray, similarities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    roc_data = calculate_roc_curve_data(labels, similarities)
    fpr = np.asarray(roc_data["fpr"], dtype=np.float64)
    tpr = np.asarray(roc_data["tpr"], dtype=np.float64)
    thresholds = np.asarray(roc_data["thresholds"], dtype=np.float64)
    fnr = 1.0 - tpr
    return fpr, tpr, thresholds, fnr


def calculate_eer_threshold(labels: np.ndarray, similarities: np.ndarray) -> float:
    fpr, _, thresholds, fnr = _roc_operating_points(labels, similarities)
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float(thresholds[index])


def calculate_eer(labels: np.ndarray, similarities: np.ndarray) -> dict[str, float]:
    fpr, _, thresholds, fnr = _roc_operating_points(labels, similarities)
    index = int(np.argmin(np.abs(fpr - fnr)))
    return {
        "eer": float((fpr[index] + fnr[index]) / 2.0),
        "eer_threshold": float(thresholds[index]),
        "eer_far": float(fpr[index]),
        "eer_frr": float(fnr[index]),
    }


def calculate_threshold_candidates(labels: np.ndarray, similarities: np.ndarray) -> dict[str, float]:
    fpr, tpr, thresholds, fnr = _roc_operating_points(labels, similarities)
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    balanced_accuracies = (tpr + (1.0 - fpr)) / 2.0
    balanced_index = int(np.argmax(balanced_accuracies))
    youden_values = tpr - fpr
    youden_index = int(np.argmax(youden_values))

    return {
        "eer": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
        "eer_threshold": float(thresholds[eer_index]),
        "eer_far": float(fpr[eer_index]),
        "eer_frr": float(fnr[eer_index]),
        "balanced_accuracy_threshold": float(thresholds[balanced_index]),
        "best_balanced_accuracy": float(balanced_accuracies[balanced_index]),
        "youden_threshold": float(thresholds[youden_index]),
        "youden_value": float(youden_values[youden_index]),
    }


def calculate_confusion_matrix_values(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)

    if labels.size == 0:
        raise ValueError("No labels were provided.")

    if labels.size != predictions.size:
        raise ValueError("Labels and predictions must have the same length.")

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def calculate_far(false_positives: int, true_negatives: int) -> float:
    denominator = false_positives + true_negatives
    return float(false_positives / denominator) if denominator > 0 else 0.0


def calculate_frr(false_negatives: int, true_positives: int) -> float:
    denominator = false_negatives + true_positives
    return float(false_negatives / denominator) if denominator > 0 else 0.0


def calculate_specificity(true_negatives: int, false_positives: int) -> float:
    denominator = true_negatives + false_positives
    return float(true_negatives / denominator) if denominator > 0 else 0.0


def calculate_metrics_at_threshold(
    labels: np.ndarray,
    similarities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels, similarities = _validate_label_and_score_arrays(labels, similarities)
    predictions = (similarities >= float(threshold)).astype(np.int64)
    confusion = calculate_confusion_matrix_values(labels, predictions)

    tn = confusion["true_negatives"]
    fp = confusion["false_positives"]
    fn = confusion["false_negatives"]
    tp = confusion["true_positives"]

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": calculate_specificity(tn, fp),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "far": calculate_far(fp, tn),
        "frr": calculate_frr(fn, tp),
        **confusion,
    }


def calculate_score_statistics(labels: np.ndarray, similarities: np.ndarray) -> dict[str, dict[str, float]]:
    labels, similarities = _validate_label_and_score_arrays(labels, similarities)
    same_scores = similarities[labels == 1]
    different_scores = similarities[labels == 0]

    if same_scores.size == 0 or different_scores.size == 0:
        raise ValueError("Score statistics require both same-speaker and different-speaker scores.")

    def _summarise(values: np.ndarray) -> dict[str, float]:
        standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        return {
            "mean": float(np.mean(values)),
            "standard_deviation": standard_deviation,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    return {
        "same_speaker": _summarise(same_scores),
        "different_speaker": _summarise(different_scores),
    }


def calculate_roc_auc(labels: np.ndarray, similarities: np.ndarray) -> float:
    labels, similarities = _validate_label_and_score_arrays(labels, similarities)
    return float(roc_auc_score(labels, similarities))
