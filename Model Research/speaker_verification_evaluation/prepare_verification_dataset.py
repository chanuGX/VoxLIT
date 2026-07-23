from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


RANDOM_SEED = 42

# Approximately 35% of speakers for threshold calibration.
CALIBRATION_RATIO = 0.35

MIN_CLIPS_PER_SPEAKER = 5
MIN_EVENTS_PER_SPEAKER = 2

CALIBRATION_PAIRS_PER_CLASS = 1_000
TEST_PAIRS_PER_CLASS = 2_500


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    The VoxCeleb metadata file is tab-separated,
    even though it may have a .csv extension.
    """
    metadata = pd.read_csv(metadata_path, sep="\t")

    required_columns = {
        "VoxCeleb1 ID",
        "VGGFace1 ID",
        "Gender",
        "Nationality",
        "Set",
    }

    missing = required_columns.difference(metadata.columns)

    if missing:
        raise ValueError(
            f"Metadata file is missing columns: {sorted(missing)}"
        )

    metadata = metadata[
        metadata["Nationality"]
        .astype(str)
        .str.casefold()
        .eq("india")
    ].copy()

    return metadata


def scan_audio_files(
    raw_root: Path,
    metadata: pd.DataFrame,
) -> pd.DataFrame:

    metadata_lookup = (
        metadata
        .set_index("VoxCeleb1 ID")
        .to_dict("index")
    )

    records = []

    for speaker_directory in sorted(raw_root.iterdir()):

        if not speaker_directory.is_dir():
            continue

        speaker_id = speaker_directory.name

        if speaker_id not in metadata_lookup:
            continue

        speaker_details = metadata_lookup[speaker_id]

        for audio_path in sorted(
            speaker_directory.rglob("*.wav")
        ):
            event_id = audio_path.parent.name

            records.append(
                {
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_details[
                        "VGGFace1 ID"
                    ],
                    "gender": speaker_details["Gender"],
                    "nationality": speaker_details[
                        "Nationality"
                    ],
                    "original_set": speaker_details["Set"],
                    "event_id": event_id,
                    "file_path": audio_path
                    .relative_to(raw_root)
                    .as_posix(),
                }
            )

    if not records:
        raise RuntimeError(
            f"No WAV files were found under: {raw_root}"
        )

    return pd.DataFrame(records)


def remove_ineligible_speakers(
    utterances: pd.DataFrame,
) -> pd.DataFrame:

    eligible_speakers = []

    for speaker_id, group in utterances.groupby(
        "speaker_id"
    ):
        clip_count = len(group)
        event_count = group["event_id"].nunique()

        if (
            clip_count >= MIN_CLIPS_PER_SPEAKER
            and event_count >= MIN_EVENTS_PER_SPEAKER
        ):
            eligible_speakers.append(speaker_id)
        else:
            print(
                f"Excluded {speaker_id}: "
                f"{clip_count} clips, "
                f"{event_count} events"
            )

    filtered = utterances[
        utterances["speaker_id"].isin(
            eligible_speakers
        )
    ].copy()

    if filtered["speaker_id"].nunique() < 4:
        raise RuntimeError(
            "Not enough eligible speakers to create "
            "calibration and test sets."
        )

    return filtered


def split_speakers(
    utterances: pd.DataFrame,
    random_generator: random.Random,
) -> tuple[set[str], set[str]]:
    """
    Creates speaker-disjoint calibration and test sets
    while approximately preserving gender distribution.
    """

    speaker_information = (
        utterances[
            ["speaker_id", "gender"]
        ]
        .drop_duplicates()
    )

    calibration_speakers = set()
    test_speakers = set()

    for _, gender_group in speaker_information.groupby(
        "gender"
    ):
        speakers = gender_group[
            "speaker_id"
        ].tolist()

        random_generator.shuffle(speakers)

        calibration_count = round(
            len(speakers) * CALIBRATION_RATIO
        )

        calibration_count = max(
            1,
            calibration_count,
        )

        if len(speakers) > 1:
            calibration_count = min(
                calibration_count,
                len(speakers) - 1,
            )

        calibration_speakers.update(
            speakers[:calibration_count]
        )

        test_speakers.update(
            speakers[calibration_count:]
        )

    return calibration_speakers, test_speakers


def generate_same_speaker_pairs(
    utterances: pd.DataFrame,
    target_count: int,
    random_generator: random.Random,
) -> list[tuple[str, str, int]]:
    """
    Same-speaker recordings are selected from
    different event folders.
    """

    speaker_events = defaultdict(
        lambda: defaultdict(list)
    )

    for row in utterances.itertuples():
        speaker_events[row.speaker_id][
            row.event_id
        ].append(row.file_path)

    eligible_speakers = [
        speaker_id
        for speaker_id, events
        in speaker_events.items()
        if len(events) >= 2
    ]

    pairs = set()
    maximum_attempts = target_count * 200

    for _ in range(maximum_attempts):

        if len(pairs) >= target_count:
            break

        speaker_id = random_generator.choice(
            eligible_speakers
        )

        events = list(
            speaker_events[speaker_id].keys()
        )

        event_a, event_b = random_generator.sample(
            events,
            2,
        )

        file_a = random_generator.choice(
            speaker_events[speaker_id][event_a]
        )

        file_b = random_generator.choice(
            speaker_events[speaker_id][event_b]
        )

        pair = tuple(sorted((file_a, file_b)))
        pairs.add(pair)

    return [
        (file_a, file_b, 1)
        for file_a, file_b in pairs
    ]


def generate_different_speaker_pairs(
    utterances: pd.DataFrame,
    target_count: int,
    random_generator: random.Random,
) -> list[tuple[str, str, int]]:

    speaker_files = defaultdict(list)

    for row in utterances.itertuples():
        speaker_files[row.speaker_id].append(
            row.file_path
        )

    speakers = list(speaker_files.keys())
    pairs = set()

    maximum_attempts = target_count * 100

    for _ in range(maximum_attempts):

        if len(pairs) >= target_count:
            break

        speaker_a, speaker_b = (
            random_generator.sample(
                speakers,
                2,
            )
        )

        file_a = random_generator.choice(
            speaker_files[speaker_a]
        )

        file_b = random_generator.choice(
            speaker_files[speaker_b]
        )

        pair = tuple(sorted((file_a, file_b)))
        pairs.add(pair)

    return [
        (file_a, file_b, 0)
        for file_a, file_b in pairs
    ]


def create_pair_file(
    utterances: pd.DataFrame,
    split_name: str,
    pairs_per_class: int,
    output_path: Path,
    random_generator: random.Random,
) -> dict:

    same_pairs = generate_same_speaker_pairs(
        utterances,
        pairs_per_class,
        random_generator,
    )

    different_pairs = (
        generate_different_speaker_pairs(
            utterances,
            pairs_per_class,
            random_generator,
        )
    )

    # Keep the final dataset balanced.
    balanced_count = min(
        len(same_pairs),
        len(different_pairs),
    )

    same_pairs = random_generator.sample(
        same_pairs,
        balanced_count,
    )

    different_pairs = random_generator.sample(
        different_pairs,
        balanced_count,
    )

    pairs = same_pairs + different_pairs
    random_generator.shuffle(pairs)

    rows = []

    for index, (
        file_a,
        file_b,
        label,
    ) in enumerate(pairs, start=1):

        rows.append(
            {
                "pair_id": (
                    f"{split_name}_{index:06d}"
                ),
                "file_a": file_a,
                "file_b": file_b,
                "label": label,
            }
        )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )

    return {
        "same_speaker_pairs": balanced_count,
        "different_speaker_pairs": balanced_count,
        "total_pairs": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help=(
            "Folder containing speaker ID folders, "
            "such as id10002 and id10003."
        ),
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to the VoxCeleb metadata CSV file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/prepared/"
            "voxceleb1_indian_verification"
        ),
    )

    args = parser.parse_args()

    random_generator = random.Random(
        RANDOM_SEED
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = load_metadata(args.metadata)

    utterances = scan_audio_files(
        args.raw_root,
        metadata,
    )

    utterances = remove_ineligible_speakers(
        utterances
    )

    calibration_speakers, test_speakers = (
        split_speakers(
            utterances,
            random_generator,
        )
    )

    utterances["split"] = utterances[
        "speaker_id"
    ].apply(
        lambda speaker_id: (
            "calibration"
            if speaker_id
            in calibration_speakers
            else "test"
        )
    )

    utterances.insert(
        0,
        "utterance_id",
        [
            f"utt_{index:06d}"
            for index in range(
                1,
                len(utterances) + 1,
            )
        ],
    )

    utterances.to_csv(
        args.output / "utterances.csv",
        index=False,
    )

    calibration_utterances = utterances[
        utterances["split"] == "calibration"
    ]

    test_utterances = utterances[
        utterances["split"] == "test"
    ]

    calibration_summary = create_pair_file(
        calibration_utterances,
        "calibration",
        CALIBRATION_PAIRS_PER_CLASS,
        args.output / "calibration_pairs.csv",
        random_generator,
    )

    test_summary = create_pair_file(
        test_utterances,
        "test",
        TEST_PAIRS_PER_CLASS,
        args.output / "test_pairs.csv",
        random_generator,
    )

    summary = {
        "source": (
            "VoxCeleb1 Indian celebrity "
            "Kaggle subset"
        ),
        "random_seed": RANDOM_SEED,
        "speaker_disjoint_split": True,
        "same_speaker_pair_rule": (
            "Audio clips must come from "
            "different event folders."
        ),
        "total_speakers": (
            utterances["speaker_id"].nunique()
        ),
        "calibration_speakers": len(
            calibration_speakers
        ),
        "test_speakers": len(
            test_speakers
        ),
        "total_audio_files": len(utterances),
        "calibration_pairs": (
            calibration_summary
        ),
        "test_pairs": test_summary,
    }

    with (
        args.output / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nDataset preparation completed.")
    print(f"Output folder: {args.output.resolve()}")
    print(
        f"Total speakers: "
        f"{summary['total_speakers']}"
    )
    print(
        f"Calibration speakers: "
        f"{summary['calibration_speakers']}"
    )
    print(
        f"Test speakers: "
        f"{summary['test_speakers']}"
    )
    print(
        f"Calibration pairs: "
        f"{calibration_summary['total_pairs']}"
    )
    print(
        f"Test pairs: "
        f"{test_summary['total_pairs']}"
    )


if __name__ == "__main__":
    main()