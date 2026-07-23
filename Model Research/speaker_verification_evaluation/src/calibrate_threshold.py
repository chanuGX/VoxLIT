from __future__ import annotations

import argparse
import json
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


def load_valid_scores(
    scores_path: Path,
) -> pd.DataFrame:
    scores = pd.read_csv(scores_path)

    required_columns = {
        "label",
        "similarity",
    }

    missing = required_columns.difference(
        scores.columns
    )

    if missing:
        raise ValueError(
            f"Scores CSV is missing columns: "
            f"{sorted(missing)}"
        )

    if "status" in scores.columns:
        scores = scores[
            scores["status"] == "success"
        ].copy()

    scores["label"] = pd.to_numeric(
        scores["label"],
        errors="coerce",
    )

    scores["similarity"] = pd.to_numeric(
        scores["similarity"],
        errors="coerce",
    )

    scores = scores.dropna(
        subset=["label", "similarity"]
    )

    scores["label"] = (
        scores["label"].astype(int)
    )

    invalid_labels = set(
        scores["label"].unique()
    ).difference({0, 1})

    if invalid_labels:
        raise ValueError(
            f"Invalid labels found: "
            f"{sorted(invalid_labels)}"
        )

    if scores["label"].nunique() != 2:
        raise ValueError(
            "Calibration data must contain both "
            "same-speaker and different-speaker pairs."
        )

    if scores.empty:
        raise ValueError(
            "No valid calibration scores found."
        )

    return scores


def calculate_thresholds(
    labels: np.ndarray,
    similarities: np.ndarray,
) -> dict:
    fpr, tpr, thresholds = roc_curve(
        labels,
        similarities,
        pos_label=1,
        drop_intermediate=False,
    )

    finite_mask = np.isfinite(thresholds)

    fpr = fpr[finite_mask]
    tpr = tpr[finite_mask]
    thresholds = thresholds[finite_mask]

    fnr = 1.0 - tpr

    # EER operating point: FAR and FRR are closest.
    eer_index = int(
        np.argmin(np.abs(fpr - fnr))
    )

    eer_threshold = float(
        thresholds[eer_index]
    )

    eer = float(
        (
            fpr[eer_index]
            + fnr[eer_index]
        )
        / 2.0
    )

    balanced_accuracies = (
        tpr + (1.0 - fpr)
    ) / 2.0

    balanced_index = int(
        np.argmax(balanced_accuracies)
    )

    balanced_threshold = float(
        thresholds[balanced_index]
    )

    best_balanced_accuracy = float(
        balanced_accuracies[balanced_index]
    )

    youden_values = tpr - fpr

    youden_index = int(
        np.argmax(youden_values)
    )

    youden_threshold = float(
        thresholds[youden_index]
    )

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "eer_far": float(fpr[eer_index]),
        "eer_frr": float(fnr[eer_index]),
        "balanced_accuracy_threshold": (
            balanced_threshold
        ),
        "best_balanced_accuracy": (
            best_balanced_accuracy
        ),
        "youden_threshold": (
            youden_threshold
        ),
        "youden_value": float(
            youden_values[youden_index]
        ),
    }


def calculate_metrics_at_threshold(
    labels: np.ndarray,
    similarities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (
        similarities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    far = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    frr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "f1_score": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "far": float(far),
        "frr": float(frr),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the x-vector decision threshold "
            "using calibration pair scores."
        )
    )

    parser.add_argument(
        "--scores",
        type=Path,
        required=True,
        help=(
            "Path to "
            "xvector_calibration_scores.csv."
        ),
    )

    parser.add_argument(
        "--criterion",
        choices=[
            "eer",
            "balanced_accuracy",
            "youden",
        ],
        default="eer",
        help=(
            "Method used to select the final "
            "decision threshold."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/metrics/"
            "xvector_threshold.json"
        ),
        help="Path used to save threshold details.",
    )

    args = parser.parse_args()

    if not args.scores.exists():
        raise FileNotFoundError(
            f"Scores file not found: {args.scores}"
        )

    scores = load_valid_scores(
        args.scores
    )

    labels = scores["label"].to_numpy(
        dtype=np.int64
    )

    similarities = scores[
        "similarity"
    ].to_numpy(
        dtype=np.float64
    )

    threshold_results = calculate_thresholds(
        labels,
        similarities,
    )

    threshold_lookup = {
        "eer": threshold_results[
            "eer_threshold"
        ],
        "balanced_accuracy": threshold_results[
            "balanced_accuracy_threshold"
        ],
        "youden": threshold_results[
            "youden_threshold"
        ],
    }

    selected_threshold = float(
        threshold_lookup[args.criterion]
    )

    selected_metrics = (
        calculate_metrics_at_threshold(
            labels,
            similarities,
            selected_threshold,
        )
    )

    same_scores = similarities[
        labels == 1
    ]

    different_scores = similarities[
        labels == 0
    ]

    output_data = {
        "model": (
            "speechbrain/"
            "spkrec-xvect-voxceleb"
        ),
        "scoring_method": (
            "cosine_similarity"
        ),
        "embedding_normalization": (
            "normalize_false"
        ),
        "calibration_scores_file": str(
            args.scores
        ),
        "selection_criterion": (
            args.criterion
        ),
        "selected_threshold": (
            selected_threshold
        ),
        "number_of_pairs": int(
            len(scores)
        ),
        "same_speaker_pairs": int(
            np.sum(labels == 1)
        ),
        "different_speaker_pairs": int(
            np.sum(labels == 0)
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                similarities,
            )
        ),
        "score_statistics": {
            "same_speaker": {
                "mean": float(
                    same_scores.mean()
                ),
                "standard_deviation": float(
                    same_scores.std()
                ),
                "minimum": float(
                    same_scores.min()
                ),
                "maximum": float(
                    same_scores.max()
                ),
            },
            "different_speaker": {
                "mean": float(
                    different_scores.mean()
                ),
                "standard_deviation": float(
                    different_scores.std()
                ),
                "minimum": float(
                    different_scores.min()
                ),
                "maximum": float(
                    different_scores.max()
                ),
            },
        },
        "threshold_candidates": (
            threshold_results
        ),
        "selected_threshold_metrics": (
            selected_metrics
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_data,
            file,
            indent=2,
        )

    print("\nThreshold calibration completed")
    print("--------------------------------")
    print(
        f"Calibration pairs:    "
        f"{len(scores)}"
    )
    print(
        f"ROC-AUC:              "
        f"{output_data['roc_auc']:.4f}"
    )
    print(
        f"EER:                  "
        f"{threshold_results['eer']:.4f}"
    )
    print(
        f"EER threshold:        "
        f"{threshold_results['eer_threshold']:.4f}"
    )
    print(
        f"Balanced threshold:   "
        f"{threshold_results['balanced_accuracy_threshold']:.4f}"
    )
    print(
        f"Selected criterion:   "
        f"{args.criterion}"
    )
    print(
        f"Selected threshold:   "
        f"{selected_threshold:.4f}"
    )
    print(
        f"Calibration accuracy: "
        f"{selected_metrics['accuracy']:.4f}"
    )
    print(
        f"Threshold saved to:   "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()