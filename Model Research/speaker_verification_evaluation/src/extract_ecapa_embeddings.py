from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.ecapa_adapter import ECAPAAdapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract x-vector embeddings for all audio files."
    )

    parser.add_argument(
        "--utterances",
        type=Path,
        required=True,
        help="Path to utterances.csv",
    )

    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help=(
            "Root folder containing speaker folders such as "
            "id10002 and id10003."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/embeddings/ecapa"),
        help="Folder used to save embedding files.",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "pretrained_models/spkrec-ecapa-voxceleb"
        ),
        help="Folder containing the downloaded model.",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use. Default selects CUDA when available.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of audio files for a test run.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate embeddings that already exist.",
    )

    args = parser.parse_args()

    if not args.utterances.exists():
        raise FileNotFoundError(
            f"utterances.csv not found: {args.utterances}"
        )

    if not args.audio_root.exists():
        raise FileNotFoundError(
            f"Audio root not found: {args.audio_root}"
        )

    utterances = pd.read_csv(args.utterances)

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
            f"Missing CSV columns: {sorted(missing_columns)}"
        )

    if args.limit is not None:
        utterances = utterances.head(args.limit)

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading x-vector model...")

    adapter = ECAPAAdapter(
        cache_dir=args.cache_dir,
        device=args.device,
    )

    manifest_rows = []
    successful = 0
    skipped = 0
    failed = 0

    for row in tqdm(
        utterances.itertuples(index=False),
        total=len(utterances),
        desc="Extracting embeddings",
    ):
        utterance_id = str(row.utterance_id)
        audio_path = args.audio_root / Path(
            str(row.file_path)
        )

        embedding_path = (
            args.output / f"{utterance_id}.npy"
        )

        start_time = time.perf_counter()

        try:
            if (
                embedding_path.exists()
                and not args.overwrite
            ):
                existing_embedding = np.load(
                    embedding_path
                )

                embedding_dimension = int(
                    existing_embedding.size
                )

                status = "skipped"
                skipped += 1

            else:
                if not audio_path.exists():
                    raise FileNotFoundError(
                        f"Audio file not found: {audio_path}"
                    )

                embedding = adapter.extract_embedding(
                    audio_path
                )

                embedding_array = (
                    embedding
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    .reshape(-1)
                )

                np.save(
                    embedding_path,
                    embedding_array,
                )

                embedding_dimension = int(
                    embedding_array.size
                )

                status = "success"
                successful += 1

            elapsed_seconds = (
                time.perf_counter() - start_time
            )

            manifest_rows.append(
                {
                    "utterance_id": utterance_id,
                    "speaker_id": row.speaker_id,
                    "event_id": row.event_id,
                    "audio_path": str(audio_path),
                    "embedding_path": str(
                        embedding_path
                    ),
                    "embedding_dimension": (
                        embedding_dimension
                    ),
                    "extraction_time_seconds": round(
                        elapsed_seconds,
                        6,
                    ),
                    "status": status,
                    "error": "",
                }
            )

        except Exception as error:
            failed += 1

            manifest_rows.append(
                {
                    "utterance_id": utterance_id,
                    "speaker_id": row.speaker_id,
                    "event_id": row.event_id,
                    "audio_path": str(audio_path),
                    "embedding_path": "",
                    "embedding_dimension": "",
                    "extraction_time_seconds": round(
                        time.perf_counter()
                        - start_time,
                        6,
                    ),
                    "status": "failed",
                    "error": str(error),
                }
            )

    manifest = pd.DataFrame(manifest_rows)

    manifest_path = (
        args.output / "embedding_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    print("\nEmbedding extraction completed")
    print("--------------------------------")
    print(f"Total audio files: {len(utterances)}")
    print(f"Successfully created: {successful}")
    print(f"Already available: {skipped}")
    print(f"Failed: {failed}")
    print(f"Embedding folder: {args.output.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()