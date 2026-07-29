from __future__ import annotations

from pathlib import Path

import torch
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from models.base_adapter import BaseSpeakerAdapter


class SpeechBrainAdapter(BaseSpeakerAdapter):
    """Shared logic for SpeechBrain-based speaker-verification models."""

    DEFAULT_CACHE_DIR: Path | None = None

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(device=device)

        effective_cache_dir = Path(cache_dir) if cache_dir is not None else self.DEFAULT_CACHE_DIR
        if effective_cache_dir is None:
            raise ValueError(f"No cache directory configured for model {self.MODEL_ID}.")

        self.cache_dir = effective_cache_dir

        self.model = EncoderClassifier.from_hparams(
            source=self.MODEL_ID,
            savedir=str(self.cache_dir),
            run_opts={"device": str(self.device)},
            local_strategy=LocalStrategy.COPY,
        )

        if hasattr(self.model, "to"):
            self.model.to(self.device)

        if hasattr(self.model, "eval"):
            self.model.eval()

    def extract_embedding(self, audio_path: str | Path) -> torch.Tensor:
        waveform = self.load_audio(audio_path)
        relative_length = torch.tensor([1.0], dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            embedding = self.model.encode_batch(
                waveform,
                wav_lens=relative_length,
                normalize=False,
            )

        embedding = torch.as_tensor(embedding, dtype=torch.float32, device=self.device).reshape(-1)
        embedding = self.validate_embedding(embedding, expected_dimension=self.EMBEDDING_DIMENSION)
        embedding = self.l2_normalize(embedding)
        embedding = self.validate_embedding(embedding, expected_dimension=self.EMBEDDING_DIMENSION)
        return embedding.detach().cpu()
