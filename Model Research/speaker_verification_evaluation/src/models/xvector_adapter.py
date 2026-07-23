from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

class XVectorAdapter:
    """Baseline adapter for SpeechBrain's pretrained x-vector model."""

    MODEL_ID = "speechbrain/spkrec-xvect-voxceleb"
    TARGET_SAMPLE_RATE = 16_000

    def __init__(
        self,
        cache_dir: str | Path = "pretrained_models/spkrec-xvect-voxceleb",
        device: str | None = None,
    ) -> None:
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = EncoderClassifier.from_hparams(
            source=self.MODEL_ID,
            savedir=str(cache_dir),
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,
        )

    def load_audio(self, audio_path: str | Path) -> torch.Tensor:
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Convert stereo or multichannel audio to mono.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Convert audio to 16 kHz.
        if sample_rate != self.TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.TARGET_SAMPLE_RATE,
            )

        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        return waveform.to(self.device)

    def extract_embedding(
        self,
        audio_path: str | Path,
    ) -> torch.Tensor:
        waveform = self.load_audio(audio_path)

        relative_length = torch.tensor(
            [1.0],
            device=self.device,
        )

        with torch.inference_mode():
            embedding = self.model.encode_batch(
                waveform,
                wav_lens=relative_length,
                normalize=False,
            )

        # Expected model output is approximately [1, 1, embedding_dimension].
        embedding = embedding.squeeze()

        # Normalize so cosine comparison is consistent.
        embedding = F.normalize(
            embedding,
            p=2,
            dim=0,
        )

        return embedding.cpu()

    @staticmethod
    def calculate_similarity(
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
    ) -> float:
        similarity = F.cosine_similarity(
            embedding_a.unsqueeze(0),
            embedding_b.unsqueeze(0),
            dim=1,
        )

        return float(similarity.item())

    def verify(
        self,
        audio_a: str | Path,
        audio_b: str | Path,
        threshold: float,
    ) -> dict:
        embedding_a = self.extract_embedding(audio_a)
        embedding_b = self.extract_embedding(audio_b)

        similarity = self.calculate_similarity(
            embedding_a,
            embedding_b,
        )

        return {
            "model": self.MODEL_ID,
            "audio_a": str(audio_a),
            "audio_b": str(audio_b),
            "embedding_dimension": embedding_a.numel(),
            "similarity": similarity,
            "threshold": threshold,
            "same_speaker": similarity >= threshold,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the baseline x-vector speaker-verification model."
    )

    parser.add_argument(
        "--audio-a",
        required=True,
        help="Path to the first audio clip.",
    )

    parser.add_argument(
        "--audio-b",
        required=True,
        help="Path to the second audio clip.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help=(
            "Temporary threshold for testing only. "
            "The final threshold must be selected using calibration data."
        ),
    )

    args = parser.parse_args()

    adapter = XVectorAdapter()

    result = adapter.verify(
        audio_a=args.audio_a,
        audio_b=args.audio_b,
        threshold=args.threshold,
    )

    print("\nSpeaker verification result")
    print("---------------------------")
    print(f"Model:               {result['model']}")
    print(f"Embedding dimension: {result['embedding_dimension']}")
    print(f"Similarity:          {result['similarity']:.4f}")
    print(f"Threshold:           {result['threshold']:.4f}")
    print(
        "Prediction:          "
        + (
            "Same speaker"
            if result["same_speaker"]
            else "Different speakers"
        )
    )


if __name__ == "__main__":
    main()