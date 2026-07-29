from __future__ import annotations

import argparse
from pathlib import Path
import os

import numpy as np
import torch
from pyannote.audio import Inference, Model

from models.base_adapter import BaseSpeakerAdapter


class WeSpeakerAdapter(BaseSpeakerAdapter):
    """pyannote.audio adapter for the WeSpeaker ResNet34-LM model."""

    MODEL_KEY = "wespeaker"
    MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
    MODEL_ROLE = "candidate"
    EMBEDDING_DIMENSION = 256
    NORMALISATION_DESCRIPTION = "l2_normalized"

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device=device)

        previous_setting = os.environ.get(
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        )

        os.environ[
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        ] = "1"

        try:
            model = Model.from_pretrained(
                self.MODEL_ID
            )
        finally:
            if previous_setting is None:
                os.environ.pop(
                    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
                    None,
                )
            else:
                os.environ[
                    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
                ] = previous_setting
        if model is None:
            raise RuntimeError(f"Could not load model: {self.MODEL_ID}")

        if hasattr(model, "eval"):
            model.eval()

        self.inference = Inference(model, window="whole")
        self.inference.to(self.device)

    def extract_embedding(self, audio_path: str | Path) -> torch.Tensor:
        waveform = self.load_audio(audio_path)
        audio = {"waveform": waveform, "sample_rate": self.TARGET_SAMPLE_RATE}

        with torch.inference_mode():
            embedding = self.inference(audio)

        embedding_array = np.asarray(embedding, dtype=np.float32).reshape(-1)
        embedding_tensor = torch.from_numpy(embedding_array)
        embedding_tensor = self.validate_embedding(embedding_tensor, expected_dimension=self.EMBEDDING_DIMENSION)
        embedding_tensor = self.l2_normalize(embedding_tensor)
        embedding_tensor = self.validate_embedding(embedding_tensor, expected_dimension=self.EMBEDDING_DIMENSION)
        return embedding_tensor.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual verification test for the WeSpeaker speaker-verification model.")
    parser.add_argument("--audio-a", type=Path, required=True, help="Path to the first audio clip.")
    parser.add_argument("--audio-b", type=Path, required=True, help="Path to the second audio clip.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Temporary manual threshold only. Use a calibrated threshold for real evaluation.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Processing device.")
    args = parser.parse_args()

    adapter = WeSpeakerAdapter(device=args.device)
    result = adapter.verify(args.audio_a, args.audio_b, args.threshold)

    print("\nWeSpeaker verification result")
    print("-----------------------------")
    print(f"Model:               {result['model']}")
    print(f"Embedding dimension: {result['embedding_dimension']}")
    print(f"Similarity:          {result['similarity']:.4f}")
    print(f"Threshold:           {result['threshold']:.4f}")
    print("Prediction:          " + ("Same speaker" if result["same_speaker"] else "Different speakers"))


if __name__ == "__main__":
    main()
