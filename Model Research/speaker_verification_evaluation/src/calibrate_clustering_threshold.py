from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_speaker_clustering_with_avg import (
    calculate_similarity_matrix,
    load_embeddings,
    load_threshold,
)
from evaluation.clustering_calibration import (
    evaluate_threshold_once,
    reconcile_embedding_manifest,
    select_calibrated_threshold,
)
from models.registry import get_model_config

_MODEL_CHOICES = ["ecapa", "wespeaker"]


def _distance_matrix_from_embeddings(embeddings: np.ndarray) -> np.ndarray:
    similarity_matrix = calculate_similarity_matrix(embeddings)

    distance_matrix = 1.0 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0.0, 2.0)
    np.fill_diagonal(distance_matrix, 0.0)

    return distance_matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate an average-linkage clustering distance threshold "
            "independently of the pair-verification EER threshold, using "
            "the existing speaker-disjoint calibration/test split."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=_MODEL_CHOICES,
        help="Speaker-verification model key (production candidates only).",
    )
    parser.add_argument(
        "--utterances",
        type=Path,
        default=Path("data/prepared/voxceleb1_indian_verification/utterances.csv"),
        help="Path to utterances.csv.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Directory containing saved embeddings. The registry default is used when omitted.",
    )
    parser.add_argument(
        "--pair-threshold-json",
        type=Path,
        default=None,
        help=(
            "Path to the model's EER pair-verification threshold JSON, used "
            "only to record a reference comparison. Defaults to "
            "outputs/metrics/<model>_threshold.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/clustering"),
        help="Directory used to save calibration artifacts.",
    )

    args = parser.parse_args()

    model_config = get_model_config(args.model)

    embeddings_directory = (
        args.embeddings_dir
        if args.embeddings_dir is not None
        else Path(model_config.default_embeddings_dir)
    )

    pair_threshold_json = (
        args.pair_threshold_json
        if args.pair_threshold_json is not None
        else Path("outputs/metrics") / f"{args.model}_threshold.json"
    )

    if not args.utterances.exists():
        raise FileNotFoundError(f"Utterances file not found: {args.utterances}")

    if not embeddings_directory.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_directory}")

    utterances = pd.read_csv(args.utterances)

    required_columns = {"utterance_id", "speaker_id", "split"}
    missing_columns = required_columns.difference(utterances.columns)

    if missing_columns:
        raise ValueError(f"Utterances CSV is missing columns: {sorted(missing_columns)}")

    embedding_stems = [
        path.stem for path in embeddings_directory.glob("*.npy")
    ]

    reconcile_embedding_manifest(
        utterances["utterance_id"].tolist(),
        embedding_stems,
    )

    calibration_samples = utterances[utterances["split"] == "calibration"].reset_index(drop=True)
    holdout_samples = utterances[utterances["split"] == "test"].reset_index(drop=True)

    if calibration_samples.empty:
        raise ValueError("No rows found for the 'calibration' split.")

    if holdout_samples.empty:
        raise ValueError("No rows found for the 'test' split.")

    calibration_embeddings = load_embeddings(
        samples=calibration_samples,
        embeddings_directory=embeddings_directory,
        expected_dimension=model_config.embedding_dimension,
    )
    holdout_embeddings = load_embeddings(
        samples=holdout_samples,
        embeddings_directory=embeddings_directory,
        expected_dimension=model_config.embedding_dimension,
    )

    calibration_distance = _distance_matrix_from_embeddings(calibration_embeddings)
    holdout_distance = _distance_matrix_from_embeddings(holdout_embeddings)

    calibration_true_labels = calibration_samples["speaker_id"].astype(str).to_numpy()
    holdout_true_labels = holdout_samples["speaker_id"].astype(str).to_numpy()

    selection_result = select_calibrated_threshold(
        calibration_distance,
        calibration_true_labels,
    )
    selected_threshold = selection_result["selected_distance_threshold"]

    holdout_metrics = evaluate_threshold_once(
        holdout_distance,
        holdout_true_labels,
        selected_threshold,
    )

    pair_eer_comparison = None

    if pair_threshold_json.exists():
        eer_threshold = load_threshold(
            pair_threshold_json,
            args.model,
            model_config.model_id,
        )
        pair_eer_comparison = {
            "eer_threshold": eer_threshold,
            "one_minus_eer_distance": float(1.0 - eer_threshold),
            "note": "Reference only. Not used to select the clustering threshold.",
        }

    artifact = {
        "model_key": args.model,
        "model_id": model_config.model_id,
        "embedding_dimension": model_config.embedding_dimension,
        "embeddings_source_dir": str(embeddings_directory),
        "linkage": "average",
        "distance_metric": "cosine_precomputed",
        "split_strategy": (
            "Existing speaker-disjoint calibration/test split from utterances.csv. "
            "Threshold selected on 'calibration' only; evaluated once on 'test' as holdout."
        ),
        "calibration_speaker_count": int(calibration_samples["speaker_id"].nunique()),
        "calibration_recording_count": int(len(calibration_samples)),
        "holdout_speaker_count": int(holdout_samples["speaker_id"].nunique()),
        "holdout_recording_count": int(len(holdout_samples)),
        "search_method": (
            "Merge-distance derived candidates: one per midpoint between "
            "consecutive unique average-linkage merge heights, plus the two "
            "boundary candidates (all-singleton below the smallest merge, "
            "single-cluster above the largest). Full observed range, not an "
            "assumed grid."
        ),
        "candidate_count": selection_result["candidate_count"],
        "selection_metric": (
            "ARI (primary), NMI (tie-break), widest stable merge-height "
            "interval (tie-break), smaller threshold (final tie-break)"
        ),
        "selected_distance_threshold": selected_threshold,
        "calibration_metrics": selection_result["calibration_metrics"],
        "holdout_metrics": holdout_metrics,
        "pair_eer_threshold_comparison": pair_eer_comparison,
    }

    output_directory = args.output_dir
    output_directory.mkdir(parents=True, exist_ok=True)

    artifact_path = output_directory / f"{args.model}_calibrated_clustering_threshold.json"
    search_csv_path = output_directory / f"{args.model}_clustering_threshold_search.csv"

    with artifact_path.open("w", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2)

    pd.DataFrame(selection_result["search_rows"]).to_csv(search_csv_path, index=False)

    print("\nClustering threshold calibration")
    print("---------------------------------")
    print(f"Model:                       {model_config.model_id}")
    print(f"Calibration recordings:      {len(calibration_samples)} ({artifact['calibration_speaker_count']} speakers)")
    print(f"Holdout recordings:          {len(holdout_samples)} ({artifact['holdout_speaker_count']} speakers)")
    print(f"Candidate thresholds tried:  {selection_result['candidate_count']}")
    print(f"Selected distance threshold: {selected_threshold:.6f}")
    print(f"Calibration ARI/NMI/purity:  {selection_result['calibration_metrics']['ari']:.4f} / "
          f"{selection_result['calibration_metrics']['nmi']:.4f} / "
          f"{selection_result['calibration_metrics']['purity']:.4f}")
    print(f"Holdout ARI/NMI/purity:      {holdout_metrics['ari']:.4f} / "
          f"{holdout_metrics['nmi']:.4f} / {holdout_metrics['purity']:.4f}")
    if pair_eer_comparison is not None:
        print(f"1 - EER distance (reference only): {pair_eer_comparison['one_minus_eer_distance']:.6f}")
    print(f"Artifact:                    {artifact_path.resolve()}")
    print(f"Search table:                {search_csv_path.resolve()}")


if __name__ == "__main__":
    main()
