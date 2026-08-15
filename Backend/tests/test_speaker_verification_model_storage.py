"""Focused tests for centralized speaker-verification model storage (Commit 1).

These tests never download or load a real model: SpeechBrain's
EncoderClassifier.from_hparams and pyannote.audio's Model.from_pretrained /
Inference are monkeypatched at their real, lazily-imported module locations
so we can assert on the exact arguments the adapters pass to them.
"""

import dataclasses
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

    spec = service.MODEL_SPECS["ecapa-tdnn"]
    service._ECAPAAdapter(spec)

    expected_savedir = tmp_path / "ecapa-tdnn" / spec.revision
    expected_hf_cache_dir = tmp_path / "huggingface"

    assert captured["savedir"] == str(expected_savedir)
    assert expected_savedir.is_dir()

    # No fallback to the user-level Hugging Face cache: the explicit
    # fetch_config override must be present and point at the controlled dir.
    assert captured["fetch_config"].huggingface_cache_dir == str(expected_hf_cache_dir)
    assert expected_hf_cache_dir.is_dir()

    # The pinned revision must reach the fetch config, not just the savedir.
    assert captured["fetch_config"].revision == spec.revision


def test_ecapa_adapter_savedir_is_namespaced_by_revision(monkeypatch, tmp_path):
    """A different pinned revision must resolve to a different savedir.

    SpeechBrain's fetch() reuses whatever file already exists at a fixed
    savedir path regardless of the requested revision, so revision-namespacing
    the savedir is what actually guarantees the pinned SHA is what gets
    loaded, rather than a stale copy from an older revision.
    """

    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_MODEL_ROOT", tmp_path)

    captured: list[dict] = []

    class _FakeEncoderClassifier:
        @classmethod
        def from_hparams(cls, **kwargs):
            captured.append(kwargs)
            return object()

    import speechbrain.inference.classifiers as sb_classifiers

    monkeypatch.setattr(sb_classifiers, "EncoderClassifier", _FakeEncoderClassifier)

    base_spec = service.MODEL_SPECS["ecapa-tdnn"]
    spec_a = dataclasses.replace(base_spec, revision="revision-a")
    spec_b = dataclasses.replace(base_spec, revision="revision-b")

    service._ECAPAAdapter(spec_a)
    service._ECAPAAdapter(spec_b)

    savedir_a, savedir_b = captured[0]["savedir"], captured[1]["savedir"]
    assert savedir_a != savedir_b
    assert savedir_a == str(settings.speaker_verification_ecapa_dir_for_revision("revision-a"))
    assert savedir_b == str(settings.speaker_verification_ecapa_dir_for_revision("revision-b"))
    assert captured[0]["fetch_config"].revision == "revision-a"
    assert captured[1]["fetch_config"].revision == "revision-b"


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

    spec = service.MODEL_SPECS["resnet34-lm"]
    service._WeSpeakerAdapter(spec)

    expected_hf_cache_dir = tmp_path / "huggingface"

    # No fallback to the user-level Hugging Face cache: cache_dir must be
    # the controlled directory, never None/default.
    assert captured["cache_dir"] == str(expected_hf_cache_dir)
    assert expected_hf_cache_dir.is_dir()

    # The pinned revision must reach the loader via pyannote's "id@revision"
    # checkpoint syntax, not just the bare repo id.
    assert captured["model_id"] == f"{spec.model_id}@{spec.revision}"


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

    # Reload replaces every module-level class/function (e.g.
    # UnsupportedSpeakerModel, get_model_spec) with new objects, which would
    # desync other modules that already imported names from `service` (e.g.
    # router.py's `except UnsupportedSpeakerModel` clause) for the rest of
    # the test session. Snapshot and restore so this test stays isolated.
    original_state = dict(vars(service))
    try:
        importlib.reload(service)
        ecapa_loader.assert_not_called()
        resnet_loader.assert_not_called()
    finally:
        vars(service).clear()
        vars(service).update(original_state)
