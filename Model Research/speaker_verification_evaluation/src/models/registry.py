from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from models.base_adapter import BaseSpeakerAdapter


@dataclass(frozen=True, slots=True)
class ModelConfig:
    key: str
    model_id: str
    role: str
    embedding_dimension: int
    normalisation: str
    scoring_method: str
    default_embeddings_dir: Path
    default_cache_dir: Path | None
    adapter_factory: Callable[[str | None, Path | None], BaseSpeakerAdapter]


def _create_xvector_adapter(device: str | None, cache_dir: Path | None) -> BaseSpeakerAdapter:
    from models.xvector_adapter import XVectorAdapter

    return XVectorAdapter(cache_dir=cache_dir, device=device)


def _create_ecapa_adapter(device: str | None, cache_dir: Path | None) -> BaseSpeakerAdapter:
    from models.ecapa_adapter import ECAPAAdapter

    return ECAPAAdapter(cache_dir=cache_dir, device=device)


def _create_wespeaker_adapter(device: str | None, cache_dir: Path | None) -> BaseSpeakerAdapter:
    from models.wespeaker_adapter import WeSpeakerAdapter

    return WeSpeakerAdapter(device=device)


_MODEL_CONFIGS: dict[str, ModelConfig] = {
    "xvector": ModelConfig(
        key="xvector",
        model_id="speechbrain/spkrec-xvect-voxceleb",
        role="baseline",
        embedding_dimension=512,
        normalisation="checkpoint_normalize_false_then_l2",
        scoring_method="cosine_similarity",
        default_embeddings_dir=Path("outputs/embeddings/xvector"),
        default_cache_dir=Path("pretrained_models/spkrec-xvect-voxceleb"),
        adapter_factory=_create_xvector_adapter,
    ),
    "ecapa": ModelConfig(
        key="ecapa",
        model_id="speechbrain/spkrec-ecapa-voxceleb",
        role="candidate",
        embedding_dimension=192,
        normalisation="checkpoint_normalize_false_then_l2",
        scoring_method="cosine_similarity",
        default_embeddings_dir=Path("outputs/embeddings/ecapa"),
        default_cache_dir=Path("pretrained_models/spkrec-ecapa-voxceleb"),
        adapter_factory=_create_ecapa_adapter,
    ),
    "wespeaker": ModelConfig(
        key="wespeaker",
        model_id="pyannote/wespeaker-voxceleb-resnet34-LM",
        role="candidate",
        embedding_dimension=256,
        normalisation="l2_normalized",
        scoring_method="cosine_similarity",
        default_embeddings_dir=Path("outputs/embeddings/wespeaker"),
        default_cache_dir=None,
        adapter_factory=_create_wespeaker_adapter,
    ),
}


def get_model_config(model_key: str) -> ModelConfig:
    try:
        return _MODEL_CONFIGS[model_key]
    except KeyError as error:
        valid_keys = ", ".join(list_model_keys())
        raise ValueError(f"Unsupported model key '{model_key}'. Valid keys: {valid_keys}") from error


def list_model_keys() -> list[str]:
    return sorted(_MODEL_CONFIGS)


def create_adapter(
    model_key: str,
    device: str | None = None,
    cache_dir: Path | None = None,
) -> BaseSpeakerAdapter:
    config = get_model_config(model_key)
    effective_cache_dir = cache_dir if cache_dir is not None else config.default_cache_dir
    return config.adapter_factory(device, effective_cache_dir)
