from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from models.registry import get_model_config, list_model_keys


def load_threshold(
    threshold_path: Path,
    model_key: str,
    model_id: str,
) -> float:
    """Load and validate the calibrated model threshold."""

    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {threshold_path}"
        )

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

    saved_model_key = threshold_data.get("model_key")
    saved_model_id = threshold_data.get("model")

    if (
        saved_model_key is not None
        and saved_model_key != model_key
    ):
        raise ValueError(
            f"Threshold belongs to model "
            f"'{saved_model_key}', not '{model_key}'."
        )

    if (
        saved_model_key is None
        and saved_model_id is not None
        and saved_model_id != model_id
    ):
        raise ValueError(
            f"Threshold belongs to model "
            f"'{saved_model_id}', not '{model_id}'."
        )

    return float(
        threshold_data["selected_threshold"]
    )


def select_balanced_samples(
    utterances: pd.DataFrame,
    number_of_speakers: int,
    clips_per_speaker: int,
    minimum_events: int,
    seed: int,
) -> pd.DataFrame:
    """
    Select the same number of clips from each speaker.

    Clips are selected from multiple event folders whenever
    possible.
    """

    rng = random.Random(seed)

    eligible_speakers = []

    for speaker_id, group in utterances.groupby(
        "speaker_id"
    ):
        clip_count = len(group)
        event_count = group["event_id"].nunique()

        if (
            clip_count >= clips_per_speaker
            and event_count >= minimum_events
        ):
            eligible_speakers.append(
                str(speaker_id)
            )

    eligible_speakers = sorted(
        eligible_speakers
    )

    if len(eligible_speakers) < number_of_speakers:
        raise ValueError(
            f"Only {len(eligible_speakers)} speakers "
            f"have at least {clips_per_speaker} clips "
            f"from {minimum_events} events. "
            f"{number_of_speakers} speakers are required."
        )

    selected_speakers = rng.sample(
        eligible_speakers,
        number_of_speakers,
    )

    selected_rows = []

    for speaker_id in selected_speakers:
        speaker_rows = utterances[
            utterances["speaker_id"].astype(str)
            == speaker_id
        ].copy()

        event_groups = {
            str(event_id): group.sort_values(
                "file_path"
            )
            for event_id, group
            in speaker_rows.groupby("event_id")
        }

        event_ids = sorted(
            event_groups.keys()
        )

        rng.shuffle(event_ids)

        selected_indices = []

        # First select one clip from different events.
        for event_id in event_ids:
            event_indices = list(
                event_groups[event_id].index
            )

            selected_index = rng.choice(
                event_indices
            )

            selected_indices.append(
                selected_index
            )

            if (
                len(selected_indices)
                >= min(
                    clips_per_speaker,
                    len(event_ids),
                )
            ):
                break

        # Fill the remaining places from unused clips.
        remaining_indices = [
            index
            for index in speaker_rows.index
            if index not in selected_indices
        ]

        rng.shuffle(remaining_indices)

        required = (
            clips_per_speaker
            - len(selected_indices)
        )

        selected_indices.extend(
            remaining_indices[:required]
        )

        speaker_selection = utterances.loc[
            selected_indices
        ].copy()

        if (
            speaker_selection["event_id"].nunique()
            < minimum_events
        ):
            raise RuntimeError(
                f"Could not select clips from at least "
                f"{minimum_events} events for "
                f"speaker {speaker_id}."
            )

        selected_rows.append(
            speaker_selection
        )

    selected = pd.concat(
        selected_rows,
        ignore_index=True,
    )

    # Randomise the final audio order.
    selected = selected.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    selected.insert(
        0,
        "sample_index",
        range(1, len(selected) + 1),
    )

    return selected


def load_embeddings(
    samples: pd.DataFrame,
    embeddings_directory: Path,
    expected_dimension: int,
) -> np.ndarray:
    """Load and validate embeddings for selected samples."""

    embeddings = []

    missing_files = []

    for row in samples.itertuples(index=False):
        embedding_path = (
            embeddings_directory
            / f"{row.utterance_id}.npy"
        )

        if not embedding_path.exists():
            missing_files.append(
                str(embedding_path)
            )
            continue

        embedding = np.load(
            embedding_path,
            allow_pickle=False,
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if embedding.size != expected_dimension:
            raise ValueError(
                f"Unexpected embedding dimension for "
                f"{embedding_path}. "
                f"Expected {expected_dimension}, "
                f"found {embedding.size}."
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

        # Ensure consistent cosine similarity.
        embedding = embedding / norm

        embeddings.append(embedding)

    if missing_files:
        examples = "\n".join(
            missing_files[:10]
        )

        raise FileNotFoundError(
            f"{len(missing_files)} selected embeddings "
            f"were not found.\n{examples}"
        )

    return np.stack(
        embeddings,
        axis=0,
    )


def calculate_similarity_matrix(
    embeddings: np.ndarray,
) -> np.ndarray:
    """Calculate the full cosine-similarity matrix."""

    similarity_matrix = (
        embeddings @ embeddings.T
    )

    similarity_matrix = np.clip(
        similarity_matrix,
        -1.0,
        1.0,
    )

    np.fill_diagonal(
        similarity_matrix,
        1.0,
    )

    return similarity_matrix


def perform_clustering(
    distance_matrix: np.ndarray,
    distance_threshold: float,
    linkage: str,
) -> np.ndarray:
    """Perform threshold-based agglomerative clustering."""

    try:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage=linkage,
            distance_threshold=distance_threshold,
        )
    except TypeError:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage=linkage,
            distance_threshold=distance_threshold,
        )

    predicted_labels = clusterer.fit_predict(
        distance_matrix
    )

    return remap_cluster_labels(
        predicted_labels
    )

def remap_cluster_labels(
    labels: np.ndarray,
) -> np.ndarray:
    """Convert arbitrary cluster IDs into readable 1-based IDs."""

    mapping = {}
    next_cluster_id = 1
    remapped = []

    for label in labels:
        label = int(label)

        if label not in mapping:
            mapping[label] = next_cluster_id
            next_cluster_id += 1

        remapped.append(
            mapping[label]
        )

    return np.asarray(
        remapped,
        dtype=np.int64,
    )


def calculate_cluster_purity(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> float:
    """Calculate weighted cluster purity."""

    correct = 0

    for cluster_id in np.unique(
        predicted_labels
    ):
        cluster_true_labels = true_labels[
            predicted_labels == cluster_id
        ]

        _, counts = np.unique(
            cluster_true_labels,
            return_counts=True,
        )

        correct += int(
            counts.max()
        )

    return float(
        correct / len(true_labels)
    )


def calculate_pairwise_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict:
    """
    Evaluate whether every possible audio pair is correctly
    grouped as same-speaker or different-speaker.
    """

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    sample_count = len(true_labels)

    for index_a in range(sample_count):
        for index_b in range(
            index_a + 1,
            sample_count,
        ):
            actual_same = (
                true_labels[index_a]
                == true_labels[index_b]
            )

            predicted_same = (
                predicted_labels[index_a]
                == predicted_labels[index_b]
            )

            if actual_same and predicted_same:
                true_positive += 1
            elif not actual_same and not predicted_same:
                true_negative += 1
            elif not actual_same and predicted_same:
                false_positive += 1
            else:
                false_negative += 1

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    f1_score = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    total_pairs = (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    )

    accuracy = (
        (true_positive + true_negative)
        / total_pairs
        if total_pairs > 0
        else 0.0
    )

    return {
        "total_unique_pairs": int(
            total_pairs
        ),
        "true_positive_pairs": int(
            true_positive
        ),
        "true_negative_pairs": int(
            true_negative
        ),
        "false_positive_pairs": int(
            false_positive
        ),
        "false_negative_pairs": int(
            false_negative
        ),
        "pairwise_accuracy": float(
            accuracy
        ),
        "pairwise_precision": float(
            precision
        ),
        "pairwise_recall": float(
            recall
        ),
        "pairwise_f1_score": float(
            f1_score
        ),
    }


def create_similarity_dataframe(
    similarity_matrix: np.ndarray,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """Create a labelled similarity-matrix dataframe."""

    labels = [
        f"{row.utterance_id}_{row.speaker_id}"
        for row in samples.itertuples(
            index=False
        )
    ]

    return pd.DataFrame(
        similarity_matrix,
        index=labels,
        columns=labels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate threshold-based speaker clustering "
            "using 10 speakers and 50 audio clips."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=list_model_keys(),
        help="Speaker-verification model key.",
    )

    parser.add_argument(
        "--utterances",
        type=Path,
        required=True,
        help="Path to utterances.csv.",
    )

    parser.add_argument(
        "--threshold-json",
        type=Path,
        required=True,
        help=(
            "Path to the selected model's calibrated "
            "threshold JSON file."
        ),
    )
    parser.add_argument(
        "--linkage",
        choices=["complete", "average"],
        default="average",
        help="Agglomerative clustering linkage method.",
    )

    parser.add_argument(
        "--use-all",
        action="store_true",
        help="Use every audio file in the selected split.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing saved embeddings. "
            "The registry default is used when omitted."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used to save clustering results.",
    )

    parser.add_argument(
        "--split",
        choices=[
            "test",
            "calibration",
            "all",
        ],
        default="test",
        help=(
            "Use test speakers by default so clustering "
            "is evaluated independently of calibration."
        ),
    )

    parser.add_argument(
        "--speakers",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--clips-per-speaker",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--minimum-events",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    model_config = get_model_config(
        args.model
    )

    embeddings_directory = (
        args.embeddings_dir
        if args.embeddings_dir is not None
        else Path(
            model_config.default_embeddings_dir
        )
    )

    output_directory = (
        args.output_dir
        if args.output_dir is not None
        else Path(
            "outputs/clustering"
        )
        / (
            f"{args.model}_"
            f"{args.speakers}speakers_"
            f"{args.speakers * args.clips_per_speaker}clips"
        )
    )

    if not args.utterances.exists():
        raise FileNotFoundError(
            f"Utterances file not found: "
            f"{args.utterances}"
        )

    if not embeddings_directory.exists():
        raise FileNotFoundError(
            f"Embeddings directory not found: "
            f"{embeddings_directory}"
        )

    utterances = pd.read_csv(
        args.utterances
    )

    required_columns = {
        "utterance_id",
        "speaker_id",
        "event_id",
        "file_path",
    }

    missing_columns = required_columns.difference(
        utterances.columns
    )

    if missing_columns:
        raise ValueError(
            f"Utterances CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if args.split != "all":
        if "split" not in utterances.columns:
            raise ValueError(
                "The requested split cannot be used because "
                "utterances.csv has no 'split' column."
            )

        utterances = utterances[
            utterances["split"].astype(str)
            == args.split
        ].copy()

    if utterances.empty:
        raise ValueError(
            f"No utterances were found for split: "
            f"{args.split}"
        )

    threshold = load_threshold(
        args.threshold_json,
        args.model,
        model_config.model_id,
    )

    if args.use_all:
        samples = utterances.copy().reset_index(
            drop=True
        )

        samples.insert(
            0,
            "sample_index",
            range(1, len(samples) + 1),
        )
    else:
        samples = select_balanced_samples(
            utterances=utterances,
            number_of_speakers=args.speakers,
            clips_per_speaker=args.clips_per_speaker,
            minimum_events=args.minimum_events,
            seed=args.seed,
        )

    embeddings = load_embeddings(
        samples=samples,
        embeddings_directory=embeddings_directory,
        expected_dimension=(
            model_config.embedding_dimension
        ),
    )

    similarity_matrix = (
        calculate_similarity_matrix(
            embeddings
        )
    )

    distance_matrix = (
        1.0 - similarity_matrix
    )

    distance_matrix = np.clip(
        distance_matrix,
        0.0,
        2.0,
    )

    np.fill_diagonal(
        distance_matrix,
        0.0,
    )

    distance_threshold = (
        1.0 - threshold
    )

    predicted_clusters = perform_clustering(
        distance_matrix=distance_matrix,
        distance_threshold=distance_threshold,
        linkage=args.linkage,
    )

    true_speakers = samples[
        "speaker_id"
    ].astype(str).to_numpy()

    samples["predicted_cluster"] = (
        predicted_clusters
    )

    samples["model_key"] = args.model
    samples["model_id"] = (
        model_config.model_id
    )

    adjusted_rand_index = (
        adjusted_rand_score(
            true_speakers,
            predicted_clusters,
        )
    )

    normalized_mutual_information = (
        normalized_mutual_info_score(
            true_speakers,
            predicted_clusters,
        )
    )

    purity = calculate_cluster_purity(
        true_speakers,
        predicted_clusters,
    )

    pairwise_metrics = (
        calculate_pairwise_metrics(
            true_speakers,
            predicted_clusters,
        )
    )

    number_of_predicted_clusters = int(
        len(np.unique(predicted_clusters))
    )

    silhouette = None

    if (
        number_of_predicted_clusters > 1
        and number_of_predicted_clusters
        < len(samples)
    ):
        silhouette = float(
            silhouette_score(
                distance_matrix,
                predicted_clusters,
                metric="precomputed",
            )
        )

    metrics = {
        "model_key": args.model,
        "model": model_config.model_id,
        "embedding_dimension": (
            model_config.embedding_dimension
        ),
        "split": args.split,
        "random_seed": args.seed,
        "number_of_audio_files": int(
            len(samples)
        ),
        "true_number_of_speakers": int(
            samples["speaker_id"].nunique()
        ),
        "predicted_number_of_clusters": (
            number_of_predicted_clusters
        ),
        "clips_per_speaker": (
            args.clips_per_speaker
        ),
        "minimum_events_per_speaker": (
            args.minimum_events
        ),
        "similarity_threshold": float(
            threshold
        ),
        "distance_threshold": float(
            distance_threshold
        ),
        "use_all_audio": args.use_all,
        "clustering_method": (
            f"agglomerative_{args.linkage}_linkage"
        ),
        "adjusted_rand_index": float(
            adjusted_rand_index
        ),
        "normalized_mutual_information": float(
            normalized_mutual_information
        ),
        "cluster_purity": float(
            purity
        ),
        "silhouette_score": silhouette,
        **pairwise_metrics,

        
    }

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples_output = (
        output_directory
        / "clustering_results.csv"
    )

    similarity_output = (
        output_directory
        / "similarity_matrix.csv"
    )

    metrics_output = (
        output_directory
        / "clustering_metrics.json"
    )

    cross_table_output = (
        output_directory
        / "speaker_cluster_crosstab.csv"
    )

    samples.to_csv(
        samples_output,
        index=False,
    )

    similarity_dataframe = (
        create_similarity_dataframe(
            similarity_matrix,
            samples,
        )
    )

    similarity_dataframe.to_csv(
        similarity_output
    )

    speaker_cluster_crosstab = pd.crosstab(
        samples["speaker_id"],
        samples["predicted_cluster"],
        rownames=["true_speaker"],
        colnames=["predicted_cluster"],
    )

    speaker_cluster_crosstab.to_csv(
        cross_table_output
    )

    with metrics_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print("\nSpeaker clustering evaluation")
    print("-----------------------------")
    print(
        f"Model:                      "
        f"{model_config.model_id}"
    )
    print(
        f"Audio files:                "
        f"{len(samples)}"
    )
    print(
        f"True speakers:              "
        f"{metrics['true_number_of_speakers']}"
    )
    print(
        f"Predicted clusters:         "
        f"{number_of_predicted_clusters}"
    )
    print(
        f"Similarity threshold:       "
        f"{threshold:.4f}"
    )
    print(
        f"Distance threshold:         "
        f"{distance_threshold:.4f}"
    )
    print(
        f"Unique audio pairs:         "
        f"{pairwise_metrics['total_unique_pairs']}"
    )
    print(
        f"Adjusted Rand Index:        "
        f"{adjusted_rand_index:.4f}"
    )
    print(
        f"Normalised Mutual Info:     "
        f"{normalized_mutual_information:.4f}"
    )
    print(
        f"Cluster purity:             "
        f"{purity:.4f}"
    )
    print(
        f"Pairwise accuracy:          "
        f"{pairwise_metrics['pairwise_accuracy']:.4f}"
    )
    print(
        f"Pairwise precision:         "
        f"{pairwise_metrics['pairwise_precision']:.4f}"
    )
    print(
        f"Pairwise recall:            "
        f"{pairwise_metrics['pairwise_recall']:.4f}"
    )
    print(
        f"Pairwise F1 score:          "
        f"{pairwise_metrics['pairwise_f1_score']:.4f}"
    )

    if silhouette is not None:
        print(
            f"Silhouette score:           "
            f"{silhouette:.4f}"
        )

    print(
        f"Results:                    "
        f"{samples_output.resolve()}"
    )
    print(
        f"Similarity matrix:          "
        f"{similarity_output.resolve()}"
    )
    print(
        f"Metrics:                    "
        f"{metrics_output.resolve()}"
    )
    print(
        f"Speaker-cluster table:      "
        f"{cross_table_output.resolve()}"
    )


if __name__ == "__main__":
    main()