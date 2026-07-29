from __future__ import annotations

import argparse
import random
from collections import defaultdict
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.registry import create_adapter


def build_speaker_events(utterances: pd.DataFrame, audio_root: Path) -> dict[str, dict[str, list[Path]]]:
    speaker_events = defaultdict(lambda: defaultdict(list))

    for row in utterances.itertuples(index=False):
        audio_path = audio_root / Path(str(row.file_path))
        if audio_path.exists():
            speaker_events[str(row.speaker_id)][str(row.event_id)].append(audio_path)

    return speaker_events


def create_same_speaker_pairs(speaker_events: dict, target_count: int, rng: random.Random) -> list[dict]:
    eligible_speakers = [speaker_id for speaker_id, events in speaker_events.items() if len(events) >= 2]
    pairs = []
    used_pairs = set()

    shuffled_speakers = eligible_speakers.copy()
    rng.shuffle(shuffled_speakers)

    for speaker_id in shuffled_speakers:
        events = list(speaker_events[speaker_id].keys())
        event_a, event_b = rng.sample(events, 2)
        audio_a = rng.choice(speaker_events[speaker_id][event_a])
        audio_b = rng.choice(speaker_events[speaker_id][event_b])
        pair_key = tuple(sorted((str(audio_a), str(audio_b))))

        if pair_key not in used_pairs:
            used_pairs.add(pair_key)
            pairs.append({"speaker_a": speaker_id, "speaker_b": speaker_id, "event_a": event_a, "event_b": event_b, "audio_a": audio_a, "audio_b": audio_b, "label": 1})

        if len(pairs) >= target_count:
            return pairs

    attempts = 0
    maximum_attempts = target_count * 100
    while len(pairs) < target_count and attempts < maximum_attempts:
        attempts += 1
        speaker_id = rng.choice(eligible_speakers)
        events = list(speaker_events[speaker_id].keys())
        event_a, event_b = rng.sample(events, 2)
        audio_a = rng.choice(speaker_events[speaker_id][event_a])
        audio_b = rng.choice(speaker_events[speaker_id][event_b])
        pair_key = tuple(sorted((str(audio_a), str(audio_b))))
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)
        pairs.append({"speaker_a": speaker_id, "speaker_b": speaker_id, "event_a": event_a, "event_b": event_b, "audio_a": audio_a, "audio_b": audio_b, "label": 1})

    return pairs


def create_different_speaker_pairs(speaker_events: dict, target_count: int, rng: random.Random) -> list[dict]:
    speakers = list(speaker_events.keys())
    pairs = []
    used_pairs = set()

    attempts = 0
    maximum_attempts = target_count * 100
    while len(pairs) < target_count and attempts < maximum_attempts:
        attempts += 1
        speaker_a, speaker_b = rng.sample(speakers, 2)
        event_a = rng.choice(list(speaker_events[speaker_a].keys()))
        event_b = rng.choice(list(speaker_events[speaker_b].keys()))
        audio_a = rng.choice(speaker_events[speaker_a][event_a])
        audio_b = rng.choice(speaker_events[speaker_b][event_b])
        pair_key = tuple(sorted((str(audio_a), str(audio_b))))
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)
        pairs.append({"speaker_a": speaker_a, "speaker_b": speaker_b, "event_a": event_a, "event_b": event_b, "audio_a": audio_a, "audio_b": audio_b, "label": 0})

    return pairs


def find_best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 2000)
    best_threshold = 0.0
    best_accuracy = 0.0

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        accuracy = float(np.mean(predictions == labels))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, best_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Check x-vector cosine scores using multiple speakers.")
    parser.add_argument("--utterances", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--same-pairs", type=int, default=50)
    parser.add_argument("--different-pairs", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/xvector_multi_person_scores.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    utterances = pd.read_csv(args.utterances)

    required_columns = {"speaker_id", "event_id", "file_path"}
    missing = required_columns.difference(utterances.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    speaker_events = build_speaker_events(utterances, args.audio_root)
    print(f"Speakers with available audio: {len(speaker_events)}")

    same_pairs = create_same_speaker_pairs(speaker_events, args.same_pairs, rng)
    different_pairs = create_different_speaker_pairs(speaker_events, args.different_pairs, rng)
    all_pairs = same_pairs + different_pairs
    rng.shuffle(all_pairs)

    print(f"Same-speaker pairs: {len(same_pairs)}")
    print(f"Different-speaker pairs: {len(different_pairs)}")

    adapter = create_adapter("xvector", device=args.device)
    embedding_cache: dict[str, torch.Tensor] = {}
    results = []

    def get_embedding(audio_path: Path):
        path_key = str(audio_path.resolve())
        if path_key not in embedding_cache:
            embedding_cache[path_key] = adapter.extract_embedding(audio_path)
        return embedding_cache[path_key]

    for pair_id, pair in enumerate(tqdm(all_pairs, desc="Scoring pairs"), start=1):
        embedding_a = get_embedding(pair["audio_a"])
        embedding_b = get_embedding(pair["audio_b"])
        similarity = adapter.calculate_similarity(embedding_a, embedding_b)
        results.append({"pair_id": f"pair_{pair_id:04d}", "speaker_a": pair["speaker_a"], "speaker_b": pair["speaker_b"], "event_a": pair["event_a"], "event_b": pair["event_b"], "audio_a": str(pair["audio_a"]), "audio_b": str(pair["audio_b"]), "label": pair["label"], "similarity": similarity})

    results_df = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)

    same_scores = results_df.loc[results_df["label"] == 1, "similarity"].to_numpy()
    different_scores = results_df.loc[results_df["label"] == 0, "similarity"].to_numpy()
    labels = results_df["label"].to_numpy()
    scores = results_df["similarity"].to_numpy()
    roc_auc = roc_auc_score(labels, scores)
    best_threshold, best_accuracy = find_best_threshold(labels, scores)

    print("\nX-vector score diagnostic")
    print("-------------------------")
    print("\nSame-speaker scores")
    print(f"Mean:   {same_scores.mean():.4f}")
    print(f"Median: {np.median(same_scores):.4f}")
    print(f"Minimum:{same_scores.min():.4f}")
    print(f"Maximum:{same_scores.max():.4f}")
    print("\nDifferent-speaker scores")
    print(f"Mean:   {different_scores.mean():.4f}")
    print(f"Median: {np.median(different_scores):.4f}")
    print(f"Minimum:{different_scores.min():.4f}")
    print(f"Maximum:{different_scores.max():.4f}")
    print("\nOverall diagnostic")
    print(f"ROC-AUC:             {roc_auc:.4f}")
    print(f"Temporary threshold: {best_threshold:.4f}")
    print(f"Temporary accuracy:  {best_accuracy:.4f}")
    print(f"Unique audio files:  {len(embedding_cache)}")
    print(f"Scores saved to:     {args.output.resolve()}")
    print("\nInterpretation")
    if roc_auc >= 0.90:
        print("The model separates the speakers well on these sampled pairs.")
    elif roc_auc >= 0.70:
        print("The model has some separation, but the score distributions overlap.")
    else:
        print("The raw cosine scores provide weak speaker separation on this dataset.")


if __name__ == "__main__":
    main()
