"""Offline validation: does low segment confidence predict diarization errors?

For each AMI recording (via the running task-b API, so results are cached and
demo-ready), predicted segments are compared against the ground-truth RTTM
using the standard optimal speaker mapping. Output: error rate per confidence
bucket (high / medium / uncertain) as a bar chart + CSV.

Ground truth NEVER touches the API — it is read from disk here, offline only.

Usage: backend must be running, then
    python scripts/validate_confidence.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import httpx

API = "http://localhost:8000/tasks/task-b"
MODEL = "pyannote-3.1"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "speaker_diarization" / "ami_subset"
RTTM_DIR = DATA_DIR / "rttm"
OUTPUT_DIR = Path(__file__).resolve().parent / "validation_output"

BUCKET_ORDER = ["high", "medium", "uncertain"]


def load_reference(rttm_path: Path):
    """Parse an RTTM file into a pyannote Annotation."""
    from pyannote.core import Annotation, Segment

    reference = Annotation()
    with open(rttm_path) as handle:
        for line in handle:
            parts = line.split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start, duration, speaker = float(parts[3]), float(parts[4]), parts[7]
            reference[Segment(start, start + duration), len(reference)] = speaker
    return reference


def hypothesis_from(segments: list[dict]):
    from pyannote.core import Annotation, Segment

    hypothesis = Annotation()
    for index, segment in enumerate(segments):
        hypothesis[Segment(segment["start"], segment["end"]), index] = segment["speaker"]
    return hypothesis


def evaluate_recording(recording: dict, client: httpx.Client) -> list[dict]:
    """Return one row per confidence-scored segment: bucket + error fraction."""
    from pyannote.core import Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    meeting = recording["display_filename"].split(".")[0]
    rttm_path = RTTM_DIR / f"{meeting}.rttm"
    if not rttm_path.exists():
        print(f"  no RTTM for {recording['display_filename']} — skipped")
        return []

    print(f"  diarizing via API (instant if cached)…")
    response = client.post(
        f"{API}/run",
        json={"model": MODEL, "recording_id": recording["recording_id"]},
        timeout=3600.0,
    )
    response.raise_for_status()
    result = response.json()
    print(f"  {len(result['segments'])} segments (cached={result['cached']})")

    reference = load_reference(rttm_path)
    hypothesis = hypothesis_from(result["segments"])
    mapping = DiarizationErrorRate().optimal_mapping(reference, hypothesis)

    rows = []
    for segment in result["segments"]:
        if segment["confidence"] is None:
            continue
        duration = segment["end"] - segment["start"]
        if duration <= 0:
            continue
        mapped = mapping.get(segment["speaker"])
        if mapped is None:
            correct_seconds = 0.0  # predicted speaker matches no reference speaker
        else:
            window = Segment(segment["start"], segment["end"])
            correct_seconds = (
                reference.label_timeline(mapped).crop(window).duration()
            )
        error_fraction = 1.0 - min(1.0, correct_seconds / duration)
        rows.append(
            {
                "meeting": meeting,
                "segment_id": segment["id"],
                "bucket": segment["confidence_bucket"],
                "confidence": segment["confidence"],
                "duration": duration,
                "error_fraction": error_fraction,
                "mostly_wrong": error_fraction > 0.5,
            }
        )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    with httpx.Client() as client:
        recordings = client.get(f"{API}/dataset/recordings").json()["recordings"]
        all_rows: list[dict] = []
        for recording in recordings:
            print(f"== {recording['display_filename']} ==")
            all_rows.extend(evaluate_recording(recording, client))

    if not all_rows:
        raise SystemExit("No scored segments — is the dataset in place?")

    # Aggregate per bucket: duration-weighted error rate + binary rate.
    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in all_rows:
        bucket = stats[row["bucket"]]
        bucket["segments"] += 1
        bucket["total_seconds"] += row["duration"]
        bucket["error_seconds"] += row["duration"] * row["error_fraction"]
        bucket["mostly_wrong"] += 1 if row["mostly_wrong"] else 0

    print("\nbucket      segs   dur(s)   error-rate   mostly-wrong")
    table = []
    for bucket_name in BUCKET_ORDER:
        bucket = stats.get(bucket_name)
        if not bucket:
            continue
        error_rate = bucket["error_seconds"] / bucket["total_seconds"]
        wrong_rate = bucket["mostly_wrong"] / bucket["segments"]
        table.append((bucket_name, bucket, error_rate, wrong_rate))
        print(
            f"{bucket_name:<10} {int(bucket['segments']):>5} "
            f"{bucket['total_seconds']:>8.1f} {error_rate:>10.1%} {wrong_rate:>12.1%}"
        )

    with open(OUTPUT_DIR / "per_segment.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [t[0] for t in table]
    rates = [t[2] * 100 for t in table]
    counts = [int(t[1]["segments"]) for t in table]
    colors = {"high": "#16a34a", "medium": "#d97706", "uncertain": "#dc2626"}

    figure, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(names, rates, color=[colors[n] for n in names])
    for bar, count in zip(bars, counts):
        axis.annotate(
            f"n={count}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=9,
        )
    axis.set_ylabel("Duration-weighted error rate (%)")
    axis.set_xlabel("Confidence bucket (G1 shading)")
    axis.set_title(
        "Low-confidence segments are wrong more often\n"
        "(3 AMI meetings vs. ground truth, pyannote 3.1)"
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "error_rate_by_confidence.png", dpi=200)
    print(f"\nsaved: {OUTPUT_DIR}/error_rate_by_confidence.png + per_segment.csv")


if __name__ == "__main__":
    main()