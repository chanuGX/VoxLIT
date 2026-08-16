"""
Tests for the cross-platform pitch-shift timeout fix in
app.services.pertubation_service.apply_pitch_shift.

The old implementation used signal.SIGALRM, which does not exist on Windows
and raises ValueError when signal.signal() is called outside the main
thread of the main interpreter (exactly how Starlette runs sync `def` route
handlers). Both failures were silently swallowed, so pitch shift silently
became a no-op instead of crashing. These tests guard the new
multiprocessing-based replacement.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
import torch

from app.services import pertubation_service
from app.services.pertubation_service import add_gaussian_noise, apply_pitch_shift, apply_time_stretch


def _make_waveform(sample_audio_data: np.ndarray, seconds: float = 1.0, sample_rate: int = 16000) -> torch.Tensor:
    n_samples = int(seconds * sample_rate)
    audio = sample_audio_data[:n_samples].astype(np.float32)
    return torch.from_numpy(audio).unsqueeze(0)


class TestApplyPitchShiftSuccess:
    """Real spawned subprocess, explicit spawn context."""

    def test_pitch_shift_actually_shifts_audio(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data)
        result = apply_pitch_shift(waveform, 16000, 2.0)

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 2
        assert result.shape[0] == 1
        assert result.shape == waveform.shape
        assert not torch.allclose(result.double(), waveform.double())


class TestApplyPitchShiftFromThreadPoolWorker:
    """
    Reproduces the exact execution context that broke signal.SIGALRM:
    Starlette runs sync route handlers in a threadpool worker thread, and
    signal.signal() raises ValueError there. This test calls apply_pitch_shift
    from a real worker thread and must not raise.
    """

    def test_pitch_shift_from_worker_thread(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(apply_pitch_shift, waveform, 16000, 2.0)
            result = future.result(timeout=60)

        assert isinstance(result, torch.Tensor)
        assert result.shape == waveform.shape
        assert not torch.allclose(result.double(), waveform.double())


class _FakeProcessBase:
    """Base fake standing in for _MP_CONTEXT.Process in mocked tests."""

    def __init__(self, target=None, args=(), daemon=None):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.terminate_called = False
        self.kill_called = False

    def close(self):
        pass


class _HangingThenKilledProcess(_FakeProcessBase):
    """Simulates a process that is still alive after the initial join()
    (timeout), then reports dead once terminate()/kill() have been called."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alive = True

    def start(self):
        pass

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminate_called = True
        self._alive = False

    def kill(self):
        self.kill_called = True
        self._alive = False


class _ErrorReportingProcess(_FakeProcessBase):
    """Simulates a child that exits quickly after writing error_path,
    mirroring the real worker's failure-reporting contract."""

    def start(self):
        _, _, _, _, error_path = self.args
        with open(error_path, "w") as f:
            f.write("boom")

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False

    def terminate(self):
        pass

    def kill(self):
        pass


class _NeverProducesOutputProcess(_FakeProcessBase):
    """Exits immediately without writing output or error - used to inspect
    the args a real Process construction would have received."""

    def start(self):
        pass

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False

    def terminate(self):
        pass

    def kill(self):
        pass


class TestApplyPitchShiftClamping:
    def test_semitones_clamped_before_reaching_worker(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data)

        with patch.object(pertubation_service._MP_CONTEXT, "Process", side_effect=_NeverProducesOutputProcess) as mock_process:
            result = apply_pitch_shift(waveform, 16000, 20.0)

        assert torch.equal(result, waveform)  # falls back: no output produced
        assert mock_process.called
        _, _, n_steps, _, _ = mock_process.call_args.kwargs["args"]
        assert n_steps == 6

    def test_negative_semitones_clamped(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data)

        with patch.object(pertubation_service._MP_CONTEXT, "Process", side_effect=_NeverProducesOutputProcess) as mock_process:
            apply_pitch_shift(waveform, 16000, -20.0)

        _, _, n_steps, _, _ = mock_process.call_args.kwargs["args"]
        assert n_steps == -6


class TestApplyPitchShiftTimeout:
    def test_timeout_falls_back_to_original_waveform(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data)

        with patch.object(pertubation_service._MP_CONTEXT, "Process", side_effect=_HangingThenKilledProcess):
            result = apply_pitch_shift(waveform, 16000, 2.0, timeout_seconds=0.1)

        assert torch.equal(result, waveform)

    def test_timeout_calls_terminate(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data)
        created = []

        def _factory(*args, **kwargs):
            proc = _HangingThenKilledProcess(*args, **kwargs)
            created.append(proc)
            return proc

        with patch.object(pertubation_service._MP_CONTEXT, "Process", side_effect=_factory):
            apply_pitch_shift(waveform, 16000, 2.0, timeout_seconds=0.1)

        assert len(created) == 1
        assert created[0].terminate_called


class TestApplyPitchShiftChildException:
    def test_child_error_falls_back_to_original_waveform(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data)

        with patch.object(pertubation_service._MP_CONTEXT, "Process", side_effect=_ErrorReportingProcess):
            result = apply_pitch_shift(waveform, 16000, 2.0)

        assert torch.equal(result, waveform)


class TestTempDirectoryCleanup:
    def test_temp_dir_removed_after_timeout(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data)
        observed_paths = []

        def _factory(*args, **kwargs):
            proc = _HangingThenKilledProcess(*args, **kwargs)
            observed_paths.append(kwargs["args"][3])  # output_path
            return proc

        with patch.object(pertubation_service._MP_CONTEXT, "Process", side_effect=_factory):
            apply_pitch_shift(waveform, 16000, 2.0, timeout_seconds=0.1)

        assert observed_paths
        tmp_dir = os.path.dirname(observed_paths[0])
        assert not os.path.exists(tmp_dir)

    def test_temp_dir_removed_after_child_error(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data)
        observed_paths = []

        def _factory(*args, **kwargs):
            proc = _ErrorReportingProcess(*args, **kwargs)
            observed_paths.append(kwargs["args"][3])  # output_path
            return proc

        with patch.object(pertubation_service._MP_CONTEXT, "Process", side_effect=_factory):
            apply_pitch_shift(waveform, 16000, 2.0)

        assert observed_paths
        tmp_dir = os.path.dirname(observed_paths[0])
        assert not os.path.exists(tmp_dir)


# ---------------------------------------------------------------------------
# Time stretch: same spawned-subprocess isolation pattern as pitch shift.
#
# apply_time_stretch used to call librosa.effects.time_stretch directly
# in-process. When a SpeechBrain speaker-verification model has already been
# loaded in that same process, SpeechBrain's lazy-module machinery
# interferes with librosa's own import resolution and time_stretch raises
# ("Lazy import of LazyModule(...) failed"), which the old code silently
# swallowed and returned the original waveform. These tests guard the
# subprocess-isolated replacement.
# ---------------------------------------------------------------------------


class TestApplyTimeStretchSuccess:
    """Real spawned subprocess, explicit spawn context."""

    def test_time_stretch_lengthens_audio_for_rate_below_one(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data, seconds=3.0)
        result = apply_time_stretch(waveform, 0.7)

        assert isinstance(result, torch.Tensor)
        assert result.shape[-1] > waveform.shape[-1]

    def test_time_stretch_shortens_audio_for_rate_above_one(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data, seconds=3.0)
        result = apply_time_stretch(waveform, 1.6)

        assert isinstance(result, torch.Tensor)
        assert result.shape[-1] < waveform.shape[-1]

    def test_time_stretch_output_differs_from_original(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data, seconds=3.0)
        result = apply_time_stretch(waveform, 1.6)

        assert result.shape != waveform.shape

    def test_time_stretch_output_is_finite_and_mono(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data, seconds=3.0)
        result = apply_time_stretch(waveform, 0.7)

        assert torch.isfinite(result).all()
        assert result.dim() == 2
        assert result.shape[0] == 1


class TestApplyTimeStretchFromThreadPoolWorker:
    """Reproduces Starlette's sync-route threadpool-worker execution context."""

    def test_time_stretch_from_worker_thread(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data, seconds=3.0)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(apply_time_stretch, waveform, 1.5)
            result = future.result(timeout=60)

        assert isinstance(result, torch.Tensor)
        assert result.shape[-1] < waveform.shape[-1]


class TestApplyTimeStretchAlongsideSpeechBrain:
    """
    Reproduces the exact reported failure: SpeechBrain's lazy-import
    machinery active in the parent process while a librosa transform runs.
    Only imports the package (no .from_hparams(...) call, no weights, no
    network access) -- this must never download or load a real model, per
    project test policy.
    """

    def test_time_stretch_succeeds_after_speechbrain_import(self, sample_audio_data):
        import speechbrain  # noqa: F401
        from speechbrain.inference.classifiers import EncoderClassifier  # noqa: F401

        waveform = _make_waveform(sample_audio_data, seconds=3.0)
        result = apply_time_stretch(waveform, 1.5)

        assert isinstance(result, torch.Tensor)
        assert result.shape[-1] < waveform.shape[-1]


class _ErrorReportingTimeStretchProcess(_FakeProcessBase):
    """Simulates a child that exits quickly after writing error_path.
    Time stretch's worker args are a 4-tuple (audio_np, rate, output_path,
    error_path) -- one shorter than pitch shift's, since no sample_rate is
    needed."""

    def start(self):
        _, _, _, error_path = self.args
        with open(error_path, "w") as f:
            f.write("boom")

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False

    def terminate(self):
        pass

    def kill(self):
        pass


class TestApplyTimeStretchTimeout:
    def test_timeout_falls_back_to_original_waveform(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data)

        with patch.object(pertubation_service._MP_CONTEXT, "Process", side_effect=_HangingThenKilledProcess):
            result = apply_time_stretch(waveform, 1.5, timeout_seconds=0.1)

        assert torch.equal(result, waveform)

    def test_timeout_calls_terminate(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data)
        created = []

        def _factory(*args, **kwargs):
            proc = _HangingThenKilledProcess(*args, **kwargs)
            created.append(proc)
            return proc

        with patch.object(pertubation_service._MP_CONTEXT, "Process", side_effect=_factory):
            apply_time_stretch(waveform, 1.5, timeout_seconds=0.1)

        assert len(created) == 1
        assert created[0].terminate_called


class TestApplyTimeStretchChildException:
    def test_child_error_falls_back_to_original_waveform(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data)

        with patch.object(pertubation_service._MP_CONTEXT, "Process", side_effect=_ErrorReportingTimeStretchProcess):
            result = apply_time_stretch(waveform, 1.5)

        assert torch.equal(result, waveform)


class TestTimeStretchTempDirectoryCleanup:
    def test_temp_dir_removed_after_timeout(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data)
        observed_paths = []

        def _factory(*args, **kwargs):
            proc = _HangingThenKilledProcess(*args, **kwargs)
            observed_paths.append(kwargs["args"][2])  # output_path
            return proc

        with patch.object(pertubation_service._MP_CONTEXT, "Process", side_effect=_factory):
            apply_time_stretch(waveform, 1.5, timeout_seconds=0.1)

        assert observed_paths
        tmp_dir = os.path.dirname(observed_paths[0])
        assert not os.path.exists(tmp_dir)

    def test_temp_dir_removed_after_child_error(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data)
        observed_paths = []

        def _factory(*args, **kwargs):
            proc = _ErrorReportingTimeStretchProcess(*args, **kwargs)
            observed_paths.append(kwargs["args"][2])  # output_path
            return proc

        with patch.object(pertubation_service._MP_CONTEXT, "Process", side_effect=_factory):
            apply_time_stretch(waveform, 1.5)

        assert observed_paths
        tmp_dir = os.path.dirname(observed_paths[0])
        assert not os.path.exists(tmp_dir)


class TestAddGaussianNoiseScaling:
    """
    Regression coverage for a manual-testing report that 1% and 50% noise
    "sound almost identical". Measurement with the real function (same seed,
    same waveform) showed the scaling was already correct end to end
    (noise_rms ~= noise_level, ~34 dB SNR gap between 1% and 50%) -- these
    tests lock that behavior in so clamping/renormalization can't silently
    regress it later.
    """

    def _noise_rms(self, waveform: torch.Tensor, noise_level: float, seed: int = 0) -> float:
        out = add_gaussian_noise(waveform, noise_level=noise_level, seed=seed)
        added_noise = out - waveform
        return added_noise.pow(2).mean().sqrt().item()

    def test_noise_rms_at_fifty_percent_is_far_greater_than_at_one_percent(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data, seconds=3.0)

        rms_low = self._noise_rms(waveform, noise_level=0.01)
        rms_high = self._noise_rms(waveform, noise_level=0.5)

        assert rms_high > rms_low * 10

    def test_noise_outputs_at_different_levels_differ_and_are_finite(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data, seconds=3.0)

        out_low = add_gaussian_noise(waveform, noise_level=0.01, seed=0)
        out_high = add_gaussian_noise(waveform, noise_level=0.5, seed=0)

        assert torch.isfinite(out_low).all()
        assert torch.isfinite(out_high).all()
        assert not torch.equal(out_low, out_high)

    def test_noise_rms_increases_monotonically_with_level(self, sample_audio_data):
        waveform = _make_waveform(sample_audio_data, seconds=3.0)
        levels = [0.01, 0.05, 0.2, 0.5]

        rms_values = [self._noise_rms(waveform, noise_level=level) for level in levels]

        assert rms_values == sorted(rms_values)
        assert rms_values[0] < rms_values[-1]
