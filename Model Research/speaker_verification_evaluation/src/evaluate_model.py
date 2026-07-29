from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.metrics import (
    calculate_eer,
    calculate_metrics_at_threshold,
    calculate_roc_auc,
    calculate_score_statistics,
    load_valid_scores,
)
from models.registry import get_model_config, list_model_keys


def load_threshold(threshold_path: Path) -> dict:
    with threshold_path.open("r", encoding="utf-8") as file:
        threshold_data = json.load(file)

    if "selected_threshold" not in threshold_data:
        raise ValueError("Threshold JSON does not contain 'selected_threshold'.")

    return threshold_data


def validate_threshold_owner(config, threshold_data: dict) -> float:
    threshold_model_key = threshold_data.get("model_key")
    threshold_model_id = threshold_data.get("model")

    if threshold_model_key is not None and threshold_model_key != config.key:
        raise ValueError(f"Threshold JSON belongs to model_key '{threshold_model_key}', not '{config.key}'.")

    if threshold_model_key is None and threshold_model_id is not None and threshold_model_id != config.model_id:
        raise ValueError(f"Threshold JSON belongs to model '{threshold_model_id}', not '{config.model_id}'.")

    return float(threshold_data["selected_threshold"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a speaker-verification model using a locked calibration threshold.")
    parser.add_argument("--model", required=True, choices=list_model_keys(), help="Model key from the registry.")
    parser.add_argument("--scores", type=Path, required=True, help="Path to test pair scores CSV.")
    parser.add_argument("--threshold-json", type=Path, required=True, help="Path to the locked calibration threshold JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Final metrics JSON output.")
    parser.add_argument("--predictions-output", type=Path, default=None, help="CSV containing pair predictions.")
    args = parser.parse_args()

    config = get_model_config(args.model)
    output_path = args.output or Path(f"outputs/metrics/{config.key}_test_metrics.json")
    predictions_path = args.predictions_output or Path(f"outputs/predictions/{config.key}_test_predictions.csv")

    if not args.scores.exists():
        raise FileNotFoundError(f"Test scores not found: {args.scores}")
    if not args.threshold_json.exists():
        raise FileNotFoundError(f"Threshold file not found: {args.threshold_json}")

    scores = load_valid_scores(args.scores, required_columns={"pair_id", "label", "similarity"})
    threshold_data = load_threshold(args.threshold_json)
    threshold = validate_threshold_owner(config, threshold_data)

    labels = scores["label"].to_numpy(dtype=np.int64)
    similarities = scores["similarity"].to_numpy(dtype=np.float64)
    predictions = (similarities >= threshold).astype(np.int64)
    metrics_at_threshold = calculate_metrics_at_threshold(labels, similarities, threshold)
    eer_results = calculate_eer(labels, similarities)

    metrics = {
        "model_key": config.key,
        "model": config.model_id,
        "model_role": config.role,
        "embedding_dimension": config.embedding_dimension,
        "scoring_method": config.scoring_method,
        "embedding_normalization": config.normalisation,
        "test_scores_file": str(args.scores),
        "threshold_file": str(args.threshold_json),
        "threshold_selection_criterion": threshold_data.get("selection_criterion", "unknown"),
        "locked_threshold": float(threshold),
        "number_of_test_pairs": int(len(scores)),
        "same_speaker_pairs": int(np.sum(labels == 1)),
        "different_speaker_pairs": int(np.sum(labels == 0)),
        "accuracy": metrics_at_threshold["accuracy"],
        "balanced_accuracy": metrics_at_threshold["balanced_accuracy"],
        "precision": metrics_at_threshold["precision"],
        "recall": metrics_at_threshold["recall"],
        "specificity": metrics_at_threshold["specificity"],
        "f1_score": metrics_at_threshold["f1_score"],
        "roc_auc": calculate_roc_auc(labels, similarities),
        "test_eer": eer_results["eer"],
        "test_eer_threshold": eer_results["eer_threshold"],
        "far_at_locked_threshold": metrics_at_threshold["far"],
        "frr_at_locked_threshold": metrics_at_threshold["frr"],
        "confusion_matrix": {
            "true_negatives": metrics_at_threshold["true_negatives"],
            "false_positives": metrics_at_threshold["false_positives"],
            "false_negatives": metrics_at_threshold["false_negatives"],
            "true_positives": metrics_at_threshold["true_positives"],
        },
        "score_statistics": calculate_score_statistics(labels, similarities),
    }

    predictions_df = scores.copy()
    predictions_df["threshold"] = threshold
    predictions_df["predicted_label"] = predictions
    predictions_df["actual_class"] = np.where(labels == 1, "same_speaker", "different_speaker")
    predictions_df["predicted_class"] = np.where(predictions == 1, "same_speaker", "different_speaker")
    predictions_df["correct"] = labels == predictions
    predictions_df["threshold_margin"] = similarities - threshold
    predictions_df["model_key"] = config.key
    predictions_df["model_id"] = config.model_id

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    predictions_df.to_csv(predictions_path, index=False)

    print("\nTest evaluation completed")
    print("-------------------------")
    print(f"Model:                {config.key}")
    print(f"Locked threshold:     {threshold:.4f}")
    print(f"Test pairs:           {len(scores)}")
    print(f"Accuracy:             {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy:    {metrics['balanced_accuracy']:.4f}")
    print(f"F1 score:             {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:              {metrics['roc_auc']:.4f}")
    print(f"Test EER:             {metrics['test_eer']:.4f}")
    print(f"FAR:                  {metrics['far_at_locked_threshold']:.4f}")
    print(f"FRR:                  {metrics['frr_at_locked_threshold']:.4f}")
    print(f"True positives:       {metrics['confusion_matrix']['true_positives']}")
    print(f"True negatives:       {metrics['confusion_matrix']['true_negatives']}")


if __name__ == "__main__":
    main()
