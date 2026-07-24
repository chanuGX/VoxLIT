from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


class ECAPAAdapter:
    """Adapter for SpeechBrain's pretrained ECAPA-TDNN model."""

    MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
    TARGET_SAMPLE_RATE = 16_000

    def __init__(
        self,
        cache_dir: str | Path = (
            "pretrained_models/spkrec-ecapa-voxceleb"
        ),
        device: str | None = None,
    ) -> None:
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")
        print(f"Loading model: {self.MODEL_ID}")

        self.model = EncoderClassifier.from_hparams(
            source=self.MODEL_ID,
            savedir=str(cache_dir),
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,
        )

    def load_audio(
        self,
        audio_path: str | Path,
    ) -> torch.Tensor:
        """Load audio and convert it to mono, 16 kHz."""

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        waveform, sample_rate = torchaudio.load(
            str(audio_path)
        )

        if waveform.numel() == 0:
            raise ValueError(
                f"Audio file is empty: {audio_path}"
            )

        # Convert stereo or multichannel audio to mono.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

        # Resample to the model's required 16 kHz.
        if sample_rate != self.TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.TARGET_SAMPLE_RATE,
            )

        waveform = waveform.to(
            device=self.device,
            dtype=torch.float32,
        )

        if not torch.isfinite(waveform).all():
            raise ValueError(
                f"Audio contains invalid values: {audio_path}"
            )

        return waveform

    def extract_embedding(
        self,
        audio_path: str | Path,
    ) -> torch.Tensor:
        """Extract one ECAPA speaker embedding."""

        waveform = self.load_audio(audio_path)

        relative_length = torch.tensor(
            [1.0],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.inference_mode():
            embedding = self.model.encode_batch(
                waveform,
                wav_lens=relative_length,
                normalize=False,
            )

        # Usually changes [1, 1, embedding_dimension]
        # into [embedding_dimension].
        embedding = embedding.squeeze().reshape(-1)

        # L2 normalization for consistent cosine scoring.
        embedding = F.normalize(
            embedding,
            p=2,
            dim=0,
        )

        if not torch.isfinite(embedding).all():
            raise ValueError(
                f"Invalid embedding produced for: {audio_path}"
            )

        return embedding.detach().cpu()

    @staticmethod
    def calculate_similarity(
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
    ) -> float:
        """Calculate cosine similarity."""

        embedding_a = embedding_a.reshape(-1)
        embedding_b = embedding_b.reshape(-1)

        if embedding_a.shape != embedding_b.shape:
            raise ValueError(
                "Embedding dimensions do not match: "
                f"{embedding_a.shape} and {embedding_b.shape}"
            )

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
        """Compare two audio clips."""

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
            "embedding_dimension": int(
                embedding_a.numel()
            ),
            "similarity": similarity,
            "threshold": threshold,
            "same_speaker": similarity >= threshold,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test the SpeechBrain ECAPA-TDNN "
            "speaker-verification model."
        )
    )

    parser.add_argument(
        "--audio-a",
        type=Path,
        required=True,
        help="Path to the first WAV file.",
    )

    parser.add_argument(
        "--audio-b",
        type=Path,
        required=True,
        help="Path to the second WAV file.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help=(
            "Temporary test threshold. The final threshold "
            "must be determined from ECAPA calibration scores."
        ),
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Processing device.",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "pretrained_models/spkrec-ecapa-voxceleb"
        ),
        help="Model download and cache directory.",
    )

    args = parser.parse_args()

    adapter = ECAPAAdapter(
        cache_dir=args.cache_dir,
        device=args.device,
    )

    result = adapter.verify(
        audio_a=args.audio_a,
        audio_b=args.audio_b,
        threshold=args.threshold,
    )

    print("\nECAPA speaker verification result")
    print("---------------------------------")
    print(f"Model:               {result['model']}")
    print(
        f"Embedding dimension: "
        f"{result['embedding_dimension']}"
    )
    print(
        f"Similarity:          "
        f"{result['similarity']:.4f}"
    )
    print(
        f"Threshold:           "
        f"{result['threshold']:.4f}"
    )
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