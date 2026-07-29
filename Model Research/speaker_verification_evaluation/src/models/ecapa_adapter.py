from __future__ import annotations

import argparse
from pathlib import Path

from models.speechbrain_adapter import SpeechBrainAdapter


class ECAPAAdapter(SpeechBrainAdapter):
    """SpeechBrain ECAPA-TDNN speaker-verification adapter."""

    MODEL_KEY = "ecapa"
    MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
    MODEL_ROLE = "candidate"
    EMBEDDING_DIMENSION = 192
    DEFAULT_CACHE_DIR = Path("pretrained_models/spkrec-ecapa-voxceleb")
    NORMALISATION_DESCRIPTION = "checkpoint_normalize_false_then_l2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual verification test for the ECAPA speaker-verification model.")
    parser.add_argument("--audio-a", type=Path, required=True, help="Path to the first audio clip.")
    parser.add_argument("--audio-b", type=Path, required=True, help="Path to the second audio clip.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Temporary manual threshold only. Use a calibrated threshold for real evaluation.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Processing device.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ECAPAAdapter.DEFAULT_CACHE_DIR,
        help="Model download/cache directory.",
    )
    args = parser.parse_args()

    adapter = ECAPAAdapter(cache_dir=args.cache_dir, device=args.device)
    result = adapter.verify(args.audio_a, args.audio_b, args.threshold)

    print("\nECAPA speaker verification result")
    print("---------------------------------")
    print(f"Model:               {result['model']}")
    print(f"Embedding dimension: {result['embedding_dimension']}")
    print(f"Similarity:          {result['similarity']:.4f}")
    print(f"Threshold:           {result['threshold']:.4f}")
    print("Prediction:          " + ("Same speaker" if result["same_speaker"] else "Different speakers"))


if __name__ == "__main__":
    main()
