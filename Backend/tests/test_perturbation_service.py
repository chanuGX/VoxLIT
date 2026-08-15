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
from app.services.pertubation_service import apply_pitch_shift


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
