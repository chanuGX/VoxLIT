"""Focused tests for centralized speaker-verification model storage (Commit 1).

These tests never download or load a real model: SpeechBrain's
EncoderClassifier.from_hparams and pyannote.audio's Model.from_pretrained /
Inference are monkeypatched at their real, lazily-imported module locations
so we can assert on the exact arguments the adapters pass to them.
"""

import importlib
from unittest.mock import MagicMock

import pytest

from app.core.settings import settings
from app.tasks.verification import service


def test_model_root_and_subdirectories_are_absolute_and_scoped():
    root = settings.SPEAKER_VERIFICATION_MODEL_ROOT
    assert root.is_absolute()

    ecapa_dir = settings.speaker_verification_ecapa_dir
    hf_cache_dir = settings.speaker_verification_hf_cache_dir

    assert ecapa_dir.is_absolute()
    assert hf_cache_dir.is_absolute()
    assert ecapa_dir.is_relative_to(root)
    assert hf_cache_dir.is_relative_to(root)


def test_ecapa_adapter_uses_controlled_savedir_and_hf_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_MODEL_ROOT", tmp_path)

    captured = {}

    class _FakeEncoderClassifier:
        @classmethod
        def from_hparams(cls, **kwargs):
            captured.update(kwargs)
            return object()

    import speechbrain.inference.classifiers as sb_classifiers

    monkeypatch.setattr(sb_classifiers, "EncoderClassifier", _FakeEncoderClassifier)

    service._ECAPAAdapter(service.MODEL_SPECS["ecapa-tdnn"])

    expected_savedir = tmp_path / "ecapa-tdnn"
    expected_hf_cache_dir = tmp_path / "huggingface"

    assert captured["savedir"] == str(expected_savedir)
    assert expected_savedir.is_dir()

    # No fallback to the user-level Hugging Face cache: the explicit
    # fetch_config override must be present and point at the controlled dir.
    assert captured["fetch_config"].huggingface_cache_dir == str(expected_hf_cache_dir)
    assert expected_hf_cache_dir.is_dir()


def test_resnet_adapter_uses_controlled_hf_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_MODEL_ROOT", tmp_path)

    captured = {}

    class _FakeModel:
        @staticmethod
        def from_pretrained(model_id, cache_dir=None, **kwargs):
            captured["model_id"] = model_id
            captured["cache_dir"] = cache_dir
            return MagicMock()

    class _FakeInference:
        def __init__(self, model, window=None):
            self.model = model

        def to(self, device):
            return self

    import pyannote.audio as pyannote_audio

    monkeypatch.setattr(pyannote_audio, "Model", _FakeModel)
    monkeypatch.setattr(pyannote_audio, "Inference", _FakeInference)

    service._WeSpeakerAdapter(service.MODEL_SPECS["resnet34-lm"])

    expected_hf_cache_dir = tmp_path / "huggingface"

    # No fallback to the user-level Hugging Face cache: cache_dir must be
    # the controlled directory, never None/default.
    assert captured["cache_dir"] == str(expected_hf_cache_dir)
    assert expected_hf_cache_dir.is_dir()


def test_reloading_service_never_calls_real_model_loaders(monkeypatch):
    """Importing/reloading the module must never load or download a model,
    independent of any other test's mutation of the in-process model cache.
    """
    ecapa_loader = MagicMock(name="EncoderClassifier.from_hparams")
    resnet_loader = MagicMock(name="Model.from_pretrained")

    import speechbrain.inference.classifiers as sb_classifiers
    import pyannote.audio as pyannote_audio

    monkeypatch.setattr(sb_classifiers.EncoderClassifier, "from_hparams", ecapa_loader)
    monkeypatch.setattr(pyannote_audio.Model, "from_pretrained", resnet_loader)

    importlib.reload(service)

    ecapa_loader.assert_not_called()
    resnet_loader.assert_not_called()
