from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.registry import create_adapter, get_model_config, list_model_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract speaker embeddings for selected utterances.")
    parser.add_argument("--model", required=True, choices=list_model_keys(), help="Model key from the registry.")
    parser.add_argument("--utterances", type=Path, required=True, help="Path to utterances.csv.")
    parser.add_argument("--audio-root", type=Path, required=True, help="Root directory containing the audio files.")
    parser.add_argument("--output", type=Path, default=None, help="Embedding output directory.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Model cache directory where applicable.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Processing device.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a smaller run.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings.")
    args = parser.parse_args()

    config = get_model_config(args.model)
    output_dir = args.output or config.default_embeddings_dir

    if not args.utterances.exists():
        raise FileNotFoundError(f"utterances.csv not found: {args.utterances}")

    if not args.audio_root.exists():
        raise FileNotFoundError(f"Audio root not found: {args.audio_root}")

    utterances = pd.read_csv(args.utterances)
    required_columns = {"utterance_id", "speaker_id", "event_id", "file_path"}
    missing_columns = required_columns.difference(utterances.columns)
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")

    if args.limit is not None:
        utterances = utterances.head(args.limit)

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = create_adapter(args.model, device=args.device, cache_dir=args.cache_dir)

    print(f"Selected model: {config.key} ({config.model_id})")
    print(f"Selected device: {adapter.device}")
    print(f"Output directory: {output_dir.resolve()}")

    manifest_rows: list[dict[str, object]] = []
    successful = 0
    skipped = 0
    failed = 0

    for row in tqdm(utterances.itertuples(index=False), total=len(utterances), desc="Extracting embeddings"):
        utterance_id = str(row.utterance_id)
        audio_path = args.audio_root / Path(str(row.file_path))
        embedding_path = output_dir / f"{utterance_id}.npy"
        start_time = time.perf_counter()

        try:
            if embedding_path.exists() and not args.overwrite:
                existing_embedding = np.load(embedding_path, allow_pickle=False)
                embedding_array = np.asarray(existing_embedding, dtype=np.float32).reshape(-1)
                if embedding_array.size != config.embedding_dimension:
                    raise ValueError(
                        f"Existing embedding has dimension {embedding_array.size}, expected {config.embedding_dimension}."
                    )
                if not np.isfinite(embedding_array).all():
                    raise ValueError(f"Existing embedding contains invalid values: {embedding_path}")
                status = "skipped"
                skipped += 1
            else:
                embedding = adapter.extract_embedding(audio_path)
                embedding_array = np.asarray(embedding.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
                if embedding_array.size != config.embedding_dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {config.embedding_dimension}, got {embedding_array.size}"
                    )
                if not np.isfinite(embedding_array).all():
                    raise ValueError(f"Embedding contains invalid values: {audio_path}")
                np.save(embedding_path, embedding_array.astype(np.float32))
                status = "success"
                successful += 1

            elapsed_seconds = time.perf_counter() - start_time
            manifest_rows.append(
                {
                    "utterance_id": utterance_id,
                    "speaker_id": row.speaker_id,
                    "event_id": row.event_id,
                    "audio_path": str(audio_path),
                    "embedding_path": str(embedding_path),
                    "embedding_dimension": int(config.embedding_dimension),
                    "extraction_time_seconds": round(elapsed_seconds, 6),
                    "status": status,
                    "error": "",
                    "model_key": config.key,
                    "model_id": config.model_id,
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
                    "extraction_time_seconds": round(time.perf_counter() - start_time, 6),
                    "status": "failed",
                    "error": str(error),
                    "model_key": config.key,
                    "model_id": config.model_id,
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "embedding_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("\nEmbedding extraction completed")
    print("--------------------------------")
    print(f"Selected model:    {config.key}")
    print(f"Selected device:   {adapter.device}")
    print(f"Total files:       {len(utterances)}")
    print(f"Successful:        {successful}")
    print(f"Skipped:           {skipped}")
    print(f"Failed:            {failed}")
    print(f"Output directory:  {output_dir.resolve()}")
    print(f"Manifest path:     {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
