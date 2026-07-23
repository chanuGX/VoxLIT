from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def normalize_file_path(value: str) -> str:
    """Create a consistent relative-path format."""

    normalized = str(value).strip().replace("\\", "/")

    while normalized.startswith("./"):
        normalized = normalized[2:]

    return normalized


def load_embedding(
    embedding_path: Path,
) -> np.ndarray:
    """Load and validate one saved speaker embedding."""

    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embedding not found: {embedding_path}"
        )

    embedding = np.load(
        embedding_path,
        allow_pickle=False,
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(-1)

    if embedding.size == 0:
        raise ValueError(
            f"Embedding is empty: {embedding_path}"
        )

    if not np.isfinite(embedding).all():
        raise ValueError(
            f"Embedding contains invalid values: "
            f"{embedding_path}"
        )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError(
            f"Embedding has zero magnitude: "
            f"{embedding_path}"
        )

    return embedding


def cosine_similarity(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
) -> float:
    """Calculate cosine similarity between two embeddings."""

    if embedding_a.shape != embedding_b.shape:
        raise ValueError(
            "Embedding dimensions do not match: "
            f"{embedding_a.shape} and "
            f"{embedding_b.shape}"
        )

    denominator = (
        np.linalg.norm(embedding_a)
        * np.linalg.norm(embedding_b)
    )

    if denominator == 0:
        raise ValueError(
            "Cannot calculate cosine similarity "
            "for zero-length embeddings."
        )

    score = np.dot(
        embedding_a,
        embedding_b,
    ) / denominator

    return float(score)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score speaker-verification pairs using "
            "saved x-vector embeddings."
        )
    )

    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help=(
            "Path to calibration_pairs.csv "
            "or test_pairs.csv."
        ),
    )

    parser.add_argument(
        "--utterances",
        type=Path,
        required=True,
        help="Path to utterances.csv.",
    )

    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path(
            "outputs/embeddings/xvector"
        ),
        help=(
            "Directory containing files such as "
            "utt_000001.npy."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV file used to save pair scores.",
    )

    args = parser.parse_args()

    if not args.pairs.exists():
        raise FileNotFoundError(
            f"Pair file not found: {args.pairs}"
        )

    if not args.utterances.exists():
        raise FileNotFoundError(
            f"Utterances file not found: "
            f"{args.utterances}"
        )

    if not args.embeddings_dir.exists():
        raise FileNotFoundError(
            f"Embedding directory not found: "
            f"{args.embeddings_dir}"
        )

    pairs = pd.read_csv(args.pairs)
    utterances = pd.read_csv(args.utterances)

    required_pair_columns = {
        "pair_id",
        "file_a",
        "file_b",
        "label",
    }

    missing_pair_columns = (
        required_pair_columns.difference(
            pairs.columns
        )
    )

    if missing_pair_columns:
        raise ValueError(
            "Pair CSV is missing columns: "
            f"{sorted(missing_pair_columns)}"
        )

    required_utterance_columns = {
        "utterance_id",
        "file_path",
    }

    missing_utterance_columns = (
        required_utterance_columns.difference(
            utterances.columns
        )
    )

    if missing_utterance_columns:
        raise ValueError(
            "Utterances CSV is missing columns: "
            f"{sorted(missing_utterance_columns)}"
        )

    utterances = utterances.copy()

    utterances["normalized_file_path"] = (
        utterances["file_path"]
        .astype(str)
        .apply(normalize_file_path)
    )

    duplicated_paths = utterances[
        utterances["normalized_file_path"]
        .duplicated(keep=False)
    ]

    if not duplicated_paths.empty:
        duplicate_values = (
            duplicated_paths[
                "normalized_file_path"
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate file paths found in "
            "utterances.csv: "
            f"{duplicate_values[:10]}"
        )

    path_to_utterance = dict(
        zip(
            utterances["normalized_file_path"],
            utterances["utterance_id"].astype(str),
        )
    )

    embedding_cache: dict[str, np.ndarray] = {}

    def get_embedding(
        utterance_id: str,
    ) -> np.ndarray:
        if utterance_id not in embedding_cache:
            embedding_path = (
                args.embeddings_dir
                / f"{utterance_id}.npy"
            )

            embedding_cache[utterance_id] = (
                load_embedding(embedding_path)
            )

        return embedding_cache[utterance_id]

    results = []
    successful = 0
    failed = 0

    for row in tqdm(
        pairs.itertuples(index=False),
        total=len(pairs),
        desc="Scoring pairs",
    ):
        pair_id = str(row.pair_id)
        file_a = normalize_file_path(row.file_a)
        file_b = normalize_file_path(row.file_b)

        try:
            label = int(row.label)

            if label not in {0, 1}:
                raise ValueError(
                    f"Invalid label {label}. "
                    "Expected 0 or 1."
                )

            if file_a not in path_to_utterance:
                raise KeyError(
                    "File A was not found in "
                    f"utterances.csv: {file_a}"
                )

            if file_b not in path_to_utterance:
                raise KeyError(
                    "File B was not found in "
                    f"utterances.csv: {file_b}"
                )

            utterance_id_a = (
                path_to_utterance[file_a]
            )

            utterance_id_b = (
                path_to_utterance[file_b]
            )

            embedding_a = get_embedding(
                utterance_id_a
            )

            embedding_b = get_embedding(
                utterance_id_b
            )

            similarity = cosine_similarity(
                embedding_a,
                embedding_b,
            )

            results.append(
                {
                    "pair_id": pair_id,
                    "file_a": file_a,
                    "file_b": file_b,
                    "utterance_id_a": (
                        utterance_id_a
                    ),
                    "utterance_id_b": (
                        utterance_id_b
                    ),
                    "label": label,
                    "similarity": similarity,
                    "status": "success",
                    "error": "",
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
                    "label": getattr(
                        row,
                        "label",
                        "",
                    ),
                    "similarity": np.nan,
                    "status": "failed",
                    "error": str(error),
                }
            )

            failed += 1

    results_df = pd.DataFrame(results)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        args.output,
        index=False,
    )

    successful_results = results_df[
        results_df["status"] == "success"
    ]

    print("\nPair scoring completed")
    print("----------------------")
    print(f"Input pairs:       {len(pairs)}")
    print(f"Successful pairs:  {successful}")
    print(f"Failed pairs:      {failed}")
    print(
        f"Unique embeddings: "
        f"{len(embedding_cache)}"
    )

    if not successful_results.empty:
        same_scores = successful_results.loc[
            successful_results["label"] == 1,
            "similarity",
        ]

        different_scores = successful_results.loc[
            successful_results["label"] == 0,
            "similarity",
        ]

        if not same_scores.empty:
            print(
                f"Same-speaker mean: "
                f"{same_scores.mean():.4f}"
            )

        if not different_scores.empty:
            print(
                f"Different-speaker mean: "
                f"{different_scores.mean():.4f}"
            )

    print(f"Scores saved to:   {args.output.resolve()}")


if __name__ == "__main__":
    main()