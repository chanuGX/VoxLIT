from __future__ import annotations

import argparse
from pathlib import Path

from models.speechbrain_adapter import SpeechBrainAdapter


class XVectorAdapter(SpeechBrainAdapter):
    """SpeechBrain x-vector speaker-verification adapter."""

    MODEL_KEY = "xvector"
    MODEL_ID = "speechbrain/spkrec-xvect-voxceleb"
    MODEL_ROLE = "baseline"
    EMBEDDING_DIMENSION = 512
    DEFAULT_CACHE_DIR = Path("pretrained_models/spkrec-xvect-voxceleb")
    NORMALISATION_DESCRIPTION = "checkpoint_normalize_false_then_l2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual verification test for the x-vector speaker-verification model.")
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
        default=XVectorAdapter.DEFAULT_CACHE_DIR,
        help="Model download/cache directory.",
    )
    args = parser.parse_args()

    adapter = XVectorAdapter(cache_dir=args.cache_dir, device=args.device)
    result = adapter.verify(args.audio_a, args.audio_b, args.threshold)

    print("\nSpeaker verification result")
    print("---------------------------")
    print(f"Model:               {result['model']}")
    print(f"Embedding dimension: {result['embedding_dimension']}")
    print(f"Similarity:          {result['similarity']:.4f}")
    print(f"Threshold:           {result['threshold']:.4f}")
    print("Prediction:          " + ("Same speaker" if result["same_speaker"] else "Different speakers"))


if __name__ == "__main__":
    main()
