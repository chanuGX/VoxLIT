"""Feature 1 endpoint tests — POST /tasks/deepfake/evaluation.

`run_detection` is stubbed throughout: these test the aggregation, the
error mapping, the cache sharing with /run, and above all that the view
leaks no per-recording ground truth. They never load a model.
"""

import json
from importlib import import_module

import pytest

from app.core.settings import settings

deepfake_evaluation = import_module("app.tasks.deepfake.evaluation")

# Two clearly separable populations so the EER is predictable.
PROTOCOL = {
    "LA_E_3000001": ("-", "bonafide"),
    "LA_E_3000002": ("-", "bonafide"),
    "LA_E_3000003": ("A10", "spoof"),
    "LA_E_3000004": ("A19", "spoof"),
}

SCORES = {
    "LA_E_3000001": 0.02,
    "LA_E_3000002": 0.08,
    "LA_E_3000003": 0.91,
    "LA_E_3000004": 0.88,
}


@pytest.fixture
def labelled_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path)
    audio_dir = tmp_path / "asvspoof2019_la" / "flac"
    audio_dir.mkdir(parents=True)
    for index, file_id in enumerate(PROTOCOL):
        # Distinct bytes so each file gets its own content hash / cache slot.
        (audio_dir / f"{file_id}.flac").write_bytes(b"fLaC" + bytes([index]))
    (tmp_path / "asvspoof2019_la" / "protocol.txt").write_text(
        "\n".join(
            f"LA_0069 {file_id} - {system_id} {key}"
            for file_id, (system_id, key) in PROTOCOL.items()
        )
        + "\n"
    )
    return audio_dir


@pytest.fixture
def stub_scoring(monkeypatch):
    """Score by filename, so each clip gets its scripted value."""
    calls = []

    def _fake_run_detection(model_key, audio_path):
        from pathlib import Path

        stem = Path(audio_path).stem
        calls.append(stem)
        return {"spoof_probability": SCORES[stem], "bonafide_probability": 1 - SCORES[stem]}

    monkeypatch.setattr(deepfake_evaluation, "run_detection", _fake_run_detection)
    return calls


async def test_evaluation_reports_eer_and_counts(client, labelled_dataset, stub_scoring):
    response = await client.post(
        "/tasks/deepfake/evaluation", json={"model": "xlsr-deepfake"}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scored"] == 4
    assert payload["bonafide_count"] == 2
    assert payload["spoof_count"] == 2
    # Cleanly separated populations -> a perfect detector.
    assert payload["eer_percent"] == 0.0
    assert 0.08 < payload["eer_threshold"] <= 0.91


async def test_evaluation_returns_two_distributions_on_a_common_axis(
    client, labelled_dataset, stub_scoring
):
    """SRS DF-6."""
    payload = (
        await client.post("/tasks/deepfake/evaluation", json={"model": "xlsr-deepfake"})
    ).json()

    bonafide = payload["distributions"]["bonafide"]
    spoof = payload["distributions"]["spoof"]

    assert len(bonafide) == len(spoof)
    assert bonafide[0]["bin_start"] == spoof[0]["bin_start"] == 0.0
    assert bonafide[-1]["bin_end"] == spoof[-1]["bin_end"] == 1.0
    assert sum(b["count"] for b in bonafide) == 2
    assert sum(b["count"] for b in spoof) == 2


async def test_evaluation_returns_a_det_curve_spanning_all_thresholds(
    client, labelled_dataset, stub_scoring
):
    """SRS DF-7."""
    payload = (
        await client.post("/tasks/deepfake/evaluation", json={"model": "xlsr-deepfake"})
    ).json()

    curve = payload["det_curve"]

    assert len(curve) >= len(SCORES)
    assert any(point["false_acceptance_rate"] == 0.0 for point in curve)
    assert any(point["false_acceptance_rate"] == 1.0 for point in curve)


async def test_threshold_is_reported_with_its_dataset(client, labelled_dataset, stub_scoring):
    """SRS DF-9 — thresholds do not transfer between datasets."""
    payload = (
        await client.post("/tasks/deepfake/evaluation", json={"model": "xlsr-deepfake"})
    ).json()

    assert payload["dataset_id"] == "asvspoof2019-la"
    assert "asvspoof2019-la" in payload["threshold_provenance"]
    assert "do not transfer" in payload["threshold_provenance"]


async def test_operating_point_shows_what_the_shipped_threshold_costs(
    client, labelled_dataset, stub_scoring
):
    payload = (
        await client.post("/tasks/deepfake/evaluation", json={"model": "xlsr-deepfake"})
    ).json()

    operating = payload["operating_point"]

    assert operating["threshold"] == 0.5
    assert operating["calibrated"] is False
    assert operating["false_acceptance_rate"] == 0.0
    assert operating["false_rejection_rate"] == 0.0


async def test_evaluation_never_returns_a_per_recording_label(
    client, labelled_dataset, stub_scoring
):
    """The whole point of the workbench is that the user judges a clip first.

    Aggregates are fine; a file-id-to-answer mapping would give the game away.
    """
    payload = (
        await client.post("/tasks/deepfake/evaluation", json={"model": "xlsr-deepfake"})
    ).json()

    body = json.dumps(payload)
    for file_id in PROTOCOL:
        assert file_id not in body
    assert "recording_id" not in body


async def test_evaluation_rejects_an_unknown_model(client, labelled_dataset):
    response = await client.post("/tasks/deepfake/evaluation", json={"model": "nope"})

    assert response.status_code == 400


async def test_evaluation_needs_both_classes(client, monkeypatch, tmp_path, stub_scoring):
    """An EER over one class is meaningless, so refuse rather than invent one."""
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path)
    audio_dir = tmp_path / "asvspoof2019_la" / "flac"
    audio_dir.mkdir(parents=True)
    (audio_dir / "LA_E_3000003.flac").write_bytes(b"fLaC-only-spoof")
    (tmp_path / "asvspoof2019_la" / "protocol.txt").write_text(
        "LA_0069 LA_E_3000003 - A10 spoof\n"
    )

    response = await client.post(
        "/tasks/deepfake/evaluation", json={"model": "xlsr-deepfake"}
    )

    assert response.status_code == 422


async def test_evaluation_reports_mean_score_per_attack(
    client, labelled_dataset, stub_scoring
):
    payload = (
        await client.post("/tasks/deepfake/evaluation", json={"model": "xlsr-deepfake"})
    ).json()

    by_attack = {row["attack"]: row for row in payload["per_attack"]}

    assert by_attack["bonafide"]["is_spoof"] is False
    assert by_attack["bonafide"]["count"] == 2
    assert by_attack["A10"]["mean_score"] == pytest.approx(0.91)
    assert by_attack["A19"]["is_spoof"] is True


async def test_evaluation_reuses_cached_scores_on_a_second_run(
    client, labelled_dataset, stub_scoring
):
    """The second pass must not re-run inference for clips already scored."""
    await client.post("/tasks/deepfake/evaluation", json={"model": "xlsr-deepfake"})
    first_pass = len(stub_scoring)
    stub_scoring.clear()

    await client.post("/tasks/deepfake/evaluation", json={"model": "xlsr-deepfake"})

    assert first_pass == len(PROTOCOL)
    assert stub_scoring == []
