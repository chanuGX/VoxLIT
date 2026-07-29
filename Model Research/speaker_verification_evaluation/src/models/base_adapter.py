from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio


class BaseSpeakerAdapter(ABC):
    """Common functionality for speaker-verification model adapters."""

    TARGET_SAMPLE_RATE = 16_000
    MODEL_ID: str
    MODEL_KEY: str
    EMBEDDING_DIMENSION: int
    MODEL_ROLE: str
    NORMALISATION_DESCRIPTION: str

    def __init__(self, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def load_audio(self, audio_path: str | Path) -> torch.Tensor:
        """Load a WAV file, convert to mono, resample to 16 kHz, and validate it."""

        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        waveform, sample_rate = torchaudio.load(str(path))

        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty: {path}")

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.TARGET_SAMPLE_RATE,
            )

        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty after resampling: {path}")

        waveform = waveform.to(device=self.device, dtype=torch.float32)

        if not torch.isfinite(waveform).all():
            raise ValueError(f"Audio contains invalid values: {path}")

        return waveform

    @staticmethod
    def validate_embedding(
        embedding: torch.Tensor,
        *,
        expected_dimension: int | None = None,
    ) -> torch.Tensor:
        """Validate that an embedding is one-dimensional, finite, and non-empty."""

        tensor = torch.as_tensor(embedding, dtype=torch.float32).reshape(-1)

        if tensor.numel() == 0:
            raise ValueError("Embedding is empty.")

        if not torch.isfinite(tensor).all():
            raise ValueError("Embedding contains non-finite values.")

        if expected_dimension is not None and tensor.numel() != expected_dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {expected_dimension}, got {tensor.numel()}"
            )

        if float(torch.linalg.vector_norm(tensor).item()) == 0.0:
            raise ValueError("Embedding has zero magnitude.")

        return tensor

    @staticmethod
    def l2_normalize(embedding: torch.Tensor) -> torch.Tensor:
        """Return an L2-normalised embedding tensor."""

        tensor = BaseSpeakerAdapter.validate_embedding(embedding)
        return F.normalize(tensor, p=2, dim=0)

    def calculate_similarity(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
    ) -> float:
        """Calculate cosine similarity for two embeddings."""

        embedding_a = self.validate_embedding(embedding_a)
        embedding_b = self.validate_embedding(embedding_b)

        if embedding_a.shape != embedding_b.shape:
            raise ValueError(
                "Embedding dimensions do not match: "
                f"{tuple(embedding_a.shape)} and {tuple(embedding_b.shape)}"
            )

        similarity = F.cosine_similarity(
            embedding_a.unsqueeze(0),
            embedding_b.unsqueeze(0),
            dim=1,
        )

        return float(similarity.item())

    @abstractmethod
    def extract_embedding(self, audio_path: str | Path) -> torch.Tensor:
        """Extract a speaker embedding from an audio file."""

    def verify(
        self,
        audio_a: str | Path,
        audio_b: str | Path,
        threshold: float,
    ) -> dict[str, object]:
        """Compare two audio files using a cosine-similarity threshold."""

        embedding_a = self.extract_embedding(audio_a)
        embedding_b = self.extract_embedding(audio_b)
        similarity = self.calculate_similarity(embedding_a, embedding_b)

        return {
            "model": self.MODEL_ID,
            "audio_a": str(audio_a),
            "audio_b": str(audio_b),
            "embedding_dimension": int(self.EMBEDDING_DIMENSION),
            "similarity": similarity,
            "threshold": float(threshold),
            "same_speaker": similarity >= threshold,
        }
