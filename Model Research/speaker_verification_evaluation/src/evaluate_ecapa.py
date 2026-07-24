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


def load_test_scores(
    scores_path: Path,
) -> pd.DataFrame:
    scores = pd.read_csv(scores_path)

    required_columns = {
        "pair_id",
        "label",
        "similarity",
    }

    missing = required_columns.difference(
        scores.columns
    )

    if missing:
        raise ValueError(
            f"Test scores CSV is missing columns: "
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
            "Test data must contain both "
            "same-speaker and different-speaker pairs."
        )

    if scores.empty:
        raise ValueError(
            "No valid test scores found."
        )

    return scores


def load_threshold(
    threshold_path: Path,
) -> tuple[float, dict]:
    with threshold_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        threshold_data = json.load(file)

    if "selected_threshold" not in threshold_data:
        raise ValueError(
            "Threshold JSON does not contain "
            "'selected_threshold'."
        )

    threshold = float(
        threshold_data["selected_threshold"]
    )

    return threshold, threshold_data


def calculate_eer(
    labels: np.ndarray,
    similarities: np.ndarray,
) -> tuple[float, float]:
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

    index = int(
        np.argmin(np.abs(fpr - fnr))
    )

    eer = float(
        (
            fpr[index]
            + fnr[index]
        )
        / 2.0
    )

    eer_threshold = float(
        thresholds[index]
    )

    return eer, eer_threshold


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the x-vector baseline using "
            "the locked calibration threshold."
        )
    )

    parser.add_argument(
        "--scores",
        type=Path,
        required=True,
        help=(
            "Path to xvector_test_scores.csv."
        ),
    )

    parser.add_argument(
        "--threshold-json",
        type=Path,
        required=True,
        help=(
            "Path to xvector_threshold.json "
            "created from calibration data."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/metrics/"
            "xvector_test_metrics.json"
        ),
        help="Final metrics JSON output.",
    )

    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path(
            "outputs/predictions/"
            "xvector_test_predictions.csv"
        ),
        help="CSV containing pair predictions.",
    )

    args = parser.parse_args()

    if not args.scores.exists():
        raise FileNotFoundError(
            f"Test scores not found: "
            f"{args.scores}"
        )

    if not args.threshold_json.exists():
        raise FileNotFoundError(
            f"Threshold file not found: "
            f"{args.threshold_json}"
        )

    scores = load_test_scores(
        args.scores
    )

    threshold, threshold_data = (
        load_threshold(
            args.threshold_json
        )
    )

    labels = scores["label"].to_numpy(
        dtype=np.int64
    )

    similarities = scores[
        "similarity"
    ].to_numpy(
        dtype=np.float64
    )

    predictions = (
        similarities >= threshold
    ).astype(np.int64)

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

    eer, test_eer_threshold = calculate_eer(
        labels,
        similarities,
    )

    same_scores = similarities[
        labels == 1
    ]

    different_scores = similarities[
        labels == 0
    ]

    metrics = {
        "model": (
            "speechbrain/"
            "spkrec-ecapa-voxceleb"
        ),
        "model_role": "candidate",
        "embedding_dimension": 192,
        "scoring_method": (
            "cosine_similarity"
        ),
        "embedding_normalization": (
            "normalize_false"
        ),
        "test_scores_file": str(
            args.scores
        ),
        "threshold_file": str(
            args.threshold_json
        ),
        "threshold_selection_criterion": (
            threshold_data.get(
                "selection_criterion",
                "unknown",
            )
        ),
        "locked_threshold": float(
            threshold
        ),
        "number_of_test_pairs": int(
            len(scores)
        ),
        "same_speaker_pairs": int(
            np.sum(labels == 1)
        ),
        "different_speaker_pairs": int(
            np.sum(labels == 0)
        ),
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
        "specificity": float(
            specificity
        ),
        "f1_score": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                similarities,
            )
        ),
        "eer": float(eer),
        "test_eer_threshold": float(
            test_eer_threshold
        ),
        "far": float(far),
        "frr": float(frr),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
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
    }

    predictions_df = scores.copy()

    predictions_df[
        "threshold"
    ] = threshold

    predictions_df[
        "predicted_label"
    ] = predictions

    predictions_df[
        "actual_class"
    ] = np.where(
        labels == 1,
        "same_speaker",
        "different_speaker",
    )

    predictions_df[
        "predicted_class"
    ] = np.where(
        predictions == 1,
        "same_speaker",
        "different_speaker",
    )

    predictions_df[
        "correct"
    ] = labels == predictions

    predictions_df[
        "threshold_margin"
    ] = similarities - threshold

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.predictions_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    predictions_df.to_csv(
        args.predictions_output,
        index=False,
    )

    print("\nBaseline test evaluation")
    print("------------------------")
    print(
        f"Model:                "
        f"{metrics['model']}"
    )
    print(
        f"Locked threshold:     "
        f"{threshold:.4f}"
    )
    print(
        f"Test pairs:           "
        f"{len(scores)}"
    )
    print(
        f"Accuracy:             "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced accuracy:    "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"F1 score:             "
        f"{metrics['f1_score']:.4f}"
    )
    print(
        f"ROC-AUC:              "
        f"{metrics['roc_auc']:.4f}"
    )
    print(
        f"EER:                  "
        f"{metrics['eer']:.4f}"
    )
    print(
        f"FAR:                  "
        f"{metrics['far']:.4f}"
    )
    print(
        f"FRR:                  "
        f"{metrics['frr']:.4f}"
    )
    print(
        f"True positives:       {tp}"
    )
    print(
        f"True negatives:       {tn}"
    )
    print(
        f"False positives:      {fp}"
    )
    print(
        f"False negatives:      {fn}"
    )
    print(
        f"Metrics saved to:     "
        f"{args.output.resolve()}"
    )
    print(
        f"Predictions saved to: "
        f"{args.predictions_output.resolve()}"
    )


if __name__ == "__main__":
    main()