from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.registry import get_model_config, list_model_keys


def normalize_file_path(value: str) -> str:
    normalized = str(value).strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def load_embedding(embedding_path: Path, expected_dimension: int) -> np.ndarray:
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding not found: {embedding_path}")

    embedding = np.load(embedding_path, allow_pickle=False)
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

    if embedding.size == 0:
        raise ValueError(f"Embedding is empty: {embedding_path}")

    if embedding.size != expected_dimension:
        raise ValueError(
            f"Embedding dimension mismatch for {embedding_path}: expected {expected_dimension}, got {embedding.size}"
        )

    if not np.isfinite(embedding).all():
        raise ValueError(f"Embedding contains invalid values: {embedding_path}")

    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError(f"Embedding has zero magnitude: {embedding_path}")

    return embedding


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    if embedding_a.shape != embedding_b.shape:
        raise ValueError(f"Embedding dimensions do not match: {embedding_a.shape} and {embedding_b.shape}")

    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Embedding has zero magnitude.")

    denominator = norm_a * norm_b

    return float(np.dot(embedding_a, embedding_b) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score speaker-verification pairs using saved speaker embeddings.")
    parser.add_argument("--model", required=True, choices=list_model_keys(), help="Model key from the registry.")
    parser.add_argument("--pairs", type=Path, required=True, help="Path to calibration_pairs.csv or test_pairs.csv.")
    parser.add_argument("--utterances", type=Path, required=True, help="Path to utterances.csv.")
    parser.add_argument("--embeddings-dir", type=Path, default=None, help="Directory containing saved .npy embeddings.")
    parser.add_argument("--output", type=Path, required=True, help="CSV file used to save pair scores.")
    args = parser.parse_args()

    config = get_model_config(args.model)
    embeddings_dir = args.embeddings_dir or config.default_embeddings_dir

    if not args.pairs.exists():
        raise FileNotFoundError(f"Pair file not found: {args.pairs}")
    if not args.utterances.exists():
        raise FileNotFoundError(f"Utterances file not found: {args.utterances}")
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embedding directory not found: {embeddings_dir}")

    pairs = pd.read_csv(args.pairs)
    utterances = pd.read_csv(args.utterances)

    required_pair_columns = {"pair_id", "file_a", "file_b", "label"}
    missing_pair_columns = required_pair_columns.difference(pairs.columns)
    if missing_pair_columns:
        raise ValueError(f"Pair CSV is missing columns: {sorted(missing_pair_columns)}")

    required_utterance_columns = {"utterance_id", "file_path"}
    missing_utterance_columns = required_utterance_columns.difference(utterances.columns)
    if missing_utterance_columns:
        raise ValueError(f"Utterances CSV is missing columns: {sorted(missing_utterance_columns)}")

    utterances = utterances.copy()
    utterances["normalized_file_path"] = utterances["file_path"].astype(str).apply(normalize_file_path)
    duplicated_paths = utterances[utterances["normalized_file_path"].duplicated(keep=False)]
    if not duplicated_paths.empty:
        duplicate_values = duplicated_paths["normalized_file_path"].unique().tolist()
        raise ValueError(f"Duplicate file paths found in utterances.csv: {duplicate_values[:10]}")

    path_to_utterance = dict(zip(utterances["normalized_file_path"], utterances["utterance_id"].astype(str)))
    embedding_cache: dict[str, np.ndarray] = {}

    def get_embedding(utterance_id: str) -> np.ndarray:
        if utterance_id not in embedding_cache:
            embedding_path = embeddings_dir / f"{utterance_id}.npy"
            embedding_cache[utterance_id] = load_embedding(embedding_path, config.embedding_dimension)
        return embedding_cache[utterance_id]

    results = []
    successful = 0
    failed = 0

    for row in tqdm(pairs.itertuples(index=False), total=len(pairs), desc="Scoring pairs"):
        pair_id = str(row.pair_id)
        file_a = normalize_file_path(row.file_a)
        file_b = normalize_file_path(row.file_b)

        try:
            label = int(row.label)
            if label not in {0, 1}:
                raise ValueError(f"Invalid label {label}. Expected 0 or 1.")

            if file_a not in path_to_utterance:
                raise KeyError(f"File A was not found in utterances.csv: {file_a}")
            if file_b not in path_to_utterance:
                raise KeyError(f"File B was not found in utterances.csv: {file_b}")

            utterance_id_a = path_to_utterance[file_a]
            utterance_id_b = path_to_utterance[file_b]
            embedding_a = get_embedding(utterance_id_a)
            embedding_b = get_embedding(utterance_id_b)
            similarity = cosine_similarity(embedding_a, embedding_b)

            results.append(
                {
                    "pair_id": pair_id,
                    "file_a": file_a,
                    "file_b": file_b,
                    "utterance_id_a": utterance_id_a,
                    "utterance_id_b": utterance_id_b,
                    "label": label,
                    "similarity": similarity,
                    "status": "success",
                    "error": "",
                    "model_key": config.key,
                    "model_id": config.model_id,
                    "embedding_dimension": config.embedding_dimension,
                }
            )
            successful += 1
        except Exception as error:
            results.append(
                {
                    "pair_id": pair_id,
                    "file_a": file_a,
                    "file_b": file_b,
                    "utterance_id_a": "",
                    "utterance_id_b": "",
                    "label": getattr(row, "label", ""),
                    "similarity": np.nan,
                    "status": "failed",
                    "error": str(error),
                    "model_key": config.key,
                    "model_id": config.model_id,
                    "embedding_dimension": config.embedding_dimension,
                }
            )
            failed += 1

    results_df = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)

    successful_results = results_df[results_df["status"] == "success"]
    print("\nPair scoring completed")
    print("----------------------")
    print(f"Model:             {config.key}")
    print(f"Input pairs:       {len(pairs)}")
    print(f"Successful pairs:  {successful}")
    print(f"Failed pairs:      {failed}")
    print(f"Unique embeddings: {len(embedding_cache)}")

    if not successful_results.empty:
        same_scores = successful_results.loc[successful_results["label"] == 1, "similarity"]
        different_scores = successful_results.loc[successful_results["label"] == 0, "similarity"]
        if not same_scores.empty:
            print(f"Same-speaker mean: {same_scores.mean():.4f}")
        if not different_scores.empty:
            print(f"Different-speaker mean: {different_scores.mean():.4f}")

    print(f"Scores saved to:   {args.output.resolve()}")


if __name__ == "__main__":
    main()
