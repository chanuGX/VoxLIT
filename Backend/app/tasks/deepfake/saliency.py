"""Feature 3 — waveform-aligned saliency.

Implements SRS DF-14 (a temporal attribution for the detector's decision,
rendered aligned with the waveform) and DF-15 (the method is named in the
interface, and the shared saliency duration caps apply).

WHAT IT ANSWERS
---------------
Which MOMENTS of the audio pushed the model toward "spoof". If a clip is
called synthetic but the bright regions sit on silence or a microphone click,
the score was never evidence about the voice — which is the same lesson
Feature 2 delivers by ablation, arrived at from the other direction.

METHOD
------
The gradient of the spoof logit with respect to the model's input, taken from
the loaded model itself: one forward and one backward pass, no dataset and no
labels. Magnitude is used, because a researcher asks "did this moment matter",
not "which way did it push" — and the sign of a raw-waveform gradient flips
with the carrier phase, so it carries no interpretable direction here.

Integrated Gradients would be better founded but costs ~32 forward AND
backward passes; on a CPU-only machine Model C alone takes ~9 s per pass, so
it is out of reach. The method actually used is reported in the payload and
shown in the interface (DF-15) rather than being implied.

WHERE IT LIVES
--------------
DF-14 says "reusing the shared saliency service and visualisation". The
shared service (app/services/saliency_service.py) is marked frozen in
STRUCTURE.md, and the Software Architecture Document records that the
deepfake task carries its own preprocessing, loading and inference with no
common adapter -- so it cannot dispatch this task's models. What IS reused:
the shared service's duration cap (same env var, same default) and its exact
response contract, so the payload is interchangeable with the shared
visualisation's data shape.
"""

from __future__ import annotations

import os
from pathlib import Path

from .service import get_model, get_model_spec

# The SAME cap the shared saliency service applies, read from the same
# environment variable with the same default (SRS DF-15). Deliberately read
# rather than imported: importing app.services.saliency_service eagerly loads
# the emotion model as a side effect.
MAX_SALIENCY_SECONDS = int(os.getenv("MAX_SALIENCY_SECONDS", "12"))

# Enough resolution to see structure inside a word, few enough to stay legible
# on a waveform a few hundred pixels wide.
DEFAULT_SEGMENTS = 60

METHOD = "input-gradient"
METHOD_LABEL = "Input gradient (|d spoof logit / d input|)"


class SaliencyUnavailable(RuntimeError):
    """Raised when attribution cannot be produced for this model."""


def _smooth(values, window: int):
    """Moving average — raw sample gradients are far too spiky to read."""
    import numpy as np

    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def _attribution_over_time(adapter, audio_path: str | Path, max_seconds: float):
    """|gradient| of the spoof logit w.r.t. the input, plus its time span.

    Returns (attribution per input position, seconds of audio analysed).
    Handles both tiers: a raw-waveform model gives one value per sample, a
    spectrogram model one per frame. Either way the result is a sequence in
    time order, which is all the caller needs.
    """
    import numpy as np
    import torch

    from .service import _load_waveform

    waveform, sample_rate = _load_waveform(audio_path)

    # Whichever is tighter: the shared saliency cap, or what the model itself
    # will look at (AST truncates to 10.24 s regardless).
    window = min(max_seconds, getattr(adapter, "analysis_window_seconds", max_seconds))
    max_samples = int(window * sample_rate)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    analysed_seconds = waveform.shape[1] / sample_rate

    feature_extractor = getattr(adapter, "feature_extractor", None)

    if feature_extractor is not None:
        # Tier A — the checkpoint's own preprocessing builds the input.
        inputs = feature_extractor(
            waveform.squeeze(0).numpy(), sampling_rate=sample_rate, return_tensors="pt"
        )
        inputs = {name: value.to(adapter.device) for name, value in inputs.items()}
        target = inputs["input_values"].requires_grad_(True)
        logits = adapter.model(**inputs).logits
    else:
        # Tier B — our own architecture, fed the raw waveform.
        prepared = adapter._xlsr_mamba.pad_or_tile(waveform.squeeze(0).numpy())
        target = torch.from_numpy(prepared).float().unsqueeze(0).requires_grad_(True)
        logits = adapter.model(target)

    adapter.model.zero_grad(set_to_none=True)
    logits[0, adapter.spoof_index].backward()

    if target.grad is None:
        raise SaliencyUnavailable(
            "No gradient reached the model input, so no attribution can be produced."
        )

    gradient = target.grad.detach().abs().squeeze(0).cpu().numpy()
    if gradient.ndim > 1:
        # Spectrogram input (frames x mel bins): collapse the frequency axis,
        # leaving one value per time frame.
        gradient = gradient.sum(axis=-1)

    if not np.isfinite(gradient).all():
        raise SaliencyUnavailable("Attribution contained non-finite values.")

    return gradient.astype(np.float32), analysed_seconds


def _to_segments(attribution, analysed_seconds: float, segment_count: int):
    """Average the attribution into equal time slices.

    Shape matches the shared saliency service exactly (start_time, end_time,
    saliency, intensity) so the payload is interchangeable with it.
    """
    import numpy as np

    positions = attribution.size
    if positions == 0 or analysed_seconds <= 0:
        return [], []

    # Smooth over roughly one segment's worth of input before slicing.
    attribution = _smooth(attribution, max(1, positions // max(segment_count, 1)))

    edges = np.linspace(0, positions, segment_count + 1, dtype=int)
    means = np.array(
        [
            float(attribution[start:end].mean()) if end > start else 0.0
            for start, end in zip(edges[:-1], edges[1:])
        ],
        dtype=np.float32,
    )

    # Normalise to 0..1 so the colour scale is readable. This is a RANKING
    # across this clip only -- attribution magnitudes are not comparable
    # between clips or between models.
    peak = float(means.max())
    normalised = means / peak if peak > 0 else means

    segment_seconds = analysed_seconds / segment_count
    segments = [
        {
            "start_time": round(index * segment_seconds, 4),
            "end_time": round((index + 1) * segment_seconds, 4),
            "saliency": round(float(normalised[index]), 6),
            "intensity": round(float(normalised[index]), 6),
        }
        for index in range(segment_count)
    ]
    return segments, [round(float(value), 6) for value in normalised]


def _speech_overlay(audio_path: str | Path, segments: list[dict]) -> dict:
    """Where the speech is, and how much attribution lands inside it.

    Reuses Feature 2's segmentation so both views draw the same boundaries.
    This is the number that answers the question the feature exists for: if
    the model says "spoof" but the attribution sits outside the speech, the
    score was never evidence about the voice.
    """
    from .service import _load_waveform
    from .silence_probe import SILENCE_TOP_DB, segment_speech

    waveform, sample_rate = _load_waveform(audio_path)
    segmentation = segment_speech(waveform.squeeze(0).numpy(), sample_rate)

    intervals = [
        (start / sample_rate, end / sample_rate)
        for start, end in segmentation.speech_intervals
    ]

    def inside_speech(segment: dict) -> bool:
        midpoint = (segment["start_time"] + segment["end_time"]) / 2
        return any(start <= midpoint < end for start, end in intervals)

    total = sum(segment["saliency"] for segment in segments)
    in_speech = sum(segment["saliency"] for segment in segments if inside_speech(segment))

    return {
        "speech_intervals": [[round(start, 3), round(end, 3)] for start, end in intervals],
        "silence_top_db": SILENCE_TOP_DB,
        "saliency_in_speech_fraction": round(in_speech / total, 4) if total > 0 else None,
    }


def generate_saliency(
    model_key: str,
    audio_path: str | Path,
    segment_count: int = DEFAULT_SEGMENTS,
) -> dict:
    """Temporal attribution for one clip, in the shared saliency contract."""
    spec = get_model_spec(model_key)
    adapter = get_model(model_key)

    attribution, analysed_seconds = _attribution_over_time(
        adapter, audio_path, MAX_SALIENCY_SECONDS
    )
    segments, series = _to_segments(attribution, analysed_seconds, segment_count)

    if not segments:
        raise SaliencyUnavailable("The clip was too short to attribute.")

    return {
        # --- shared saliency service contract ---
        "model": model_key,
        "method": METHOD,
        "segments": segments,
        "total_duration": round(analysed_seconds, 4),
        "series": series,
        # --- deepfake specifics ---
        "model_label": spec.label,
        # DF-15: the method is named, not implied.
        "method_label": METHOD_LABEL,
        "target": "spoof logit",
        "max_saliency_seconds": MAX_SALIENCY_SECONDS,
        "analysis_window_seconds": getattr(adapter, "analysis_window_seconds", None),
        "truncated": analysed_seconds < _clip_seconds(audio_path),
        "normalised": True,
        # Lets the view show whether the heat sits on speech or on silence.
        **_speech_overlay(audio_path, segments),
    }


def _clip_seconds(audio_path: str | Path) -> float:
    import soundfile as sf

    info = sf.info(str(audio_path))
    return info.frames / float(info.samplerate)
