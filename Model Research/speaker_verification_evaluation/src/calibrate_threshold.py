from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.metrics import (
    calculate_metrics_at_threshold,
    calculate_roc_auc,
    calculate_score_statistics,
    calculate_threshold_candidates,
    load_valid_scores,
)
from models.registry import get_model_config, list_model_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a model decision threshold using calibration pair scores.")
    parser.add_argument("--model", required=True, choices=list_model_keys(), help="Model key from the registry.")
    parser.add_argument("--scores", type=Path, required=True, help="Path to calibration pair scores CSV.")
    parser.add_argument("--criterion", choices=["eer", "balanced_accuracy", "youden"], default="eer", help="Threshold selection method.")
    parser.add_argument("--output", type=Path, default=None, help="Path used to save threshold details.")
    args = parser.parse_args()

    config = get_model_config(args.model)
    output_path = args.output or Path(f"outputs/metrics/{config.key}_threshold.json")

    if not args.scores.exists():
        raise FileNotFoundError(f"Scores file not found: {args.scores}")

    scores = load_valid_scores(args.scores, required_columns={"label", "similarity"})
    labels = scores["label"].to_numpy(dtype=np.int64)
    similarities = scores["similarity"].to_numpy(dtype=np.float64)

    threshold_candidates = calculate_threshold_candidates(labels, similarities)
    threshold_lookup = {
        "eer": threshold_candidates["eer_threshold"],
        "balanced_accuracy": threshold_candidates["balanced_accuracy_threshold"],
        "youden": threshold_candidates["youden_threshold"],
    }
    selected_threshold = float(threshold_lookup[args.criterion])
    selected_metrics = calculate_metrics_at_threshold(labels, similarities, selected_threshold)

    output_data = {
        "model_key": config.key,
        "model": config.model_id,
        "model_role": config.role,
        "embedding_dimension": config.embedding_dimension,
        "scoring_method": config.scoring_method,
        "embedding_normalization": config.normalisation,
        "calibration_scores_file": str(args.scores),
        "selection_criterion": args.criterion,
        "selected_threshold": selected_threshold,
        "number_of_pairs": int(len(scores)),
        "same_speaker_pairs": int(np.sum(labels == 1)),
        "different_speaker_pairs": int(np.sum(labels == 0)),
        "roc_auc": calculate_roc_auc(labels, similarities),
        "score_statistics": calculate_score_statistics(labels, similarities),
        "threshold_candidates": threshold_candidates,
        "selected_threshold_metrics": selected_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2)

    print("\nThreshold calibration completed")
    print("--------------------------------")
    print(f"Model:               {config.key}")
    print(f"Calibration pairs:   {len(scores)}")
    print(f"ROC-AUC:             {output_data['roc_auc']:.4f}")
    print(f"Selected criterion:   {args.criterion}")
    print(f"Selected threshold:   {selected_threshold:.4f}")
    print(f"Calibration accuracy: {selected_metrics['accuracy']:.4f}")
    print(f"Threshold saved to:   {output_path.resolve()}")


if __name__ == "__main__":
    main()
