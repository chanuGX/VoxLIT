#!/usr/bin/env python
"""Build the balanced ASVspoof 2019 LA demo subset used by the deepfake task.

The full LA corpus is several GB and `Backend/.gitignore` ignores `/data`, so
nothing here is committed -- this script recreates the subset from a copy you
download yourself.

Prerequisite (manual, one time):
    Download LA.zip from https://datashare.ed.ac.uk/handle/10283/3336
    (Open Data Commons Attribution License -- it requires attribution in any
    report that uses it) and extract it anywhere.

Then, from Backend/:
    python scripts/prepare_asvspoof_la_subset.py --source /path/to/extracted/LA

Produces:
    data/deepfake/asvspoof2019_la/flac/LA_E_*.flac
    data/deepfake/asvspoof2019_la/protocol.txt

The selection is deterministic (fixed seed) and balanced: N bona fide clips,
plus N spoof clips spread evenly across every attack in the partition, so no
single vocoder dominates the demo.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DEST = BACKEND_DIR / "data" / "deepfake" / "asvspoof2019_la"
DEFAULT_SEED = 20260818

PARTITIONS = {
    # partition: (protocol filename, audio subdirectory)
    "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
    "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
    "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
}


def resolve_la_root(source: Path) -> Path:
    """Accept either the extracted `LA/` directory or its parent."""
    if (source / "ASVspoof2019_LA_cm_protocols").is_dir():
        return source
    if (source / "LA" / "ASVspoof2019_LA_cm_protocols").is_dir():
        return source / "LA"
    raise SystemExit(
        f"Could not find ASVspoof2019_LA_cm_protocols under {source}.\n"
        "Point --source at the extracted LA directory (or its parent)."
    )


def parse_protocol(protocol_path: Path) -> list[tuple[str, str, str, str]]:
    """-> [(speaker_id, file_id, system_id, key)] from the CM protocol.

    Line format: SPEAKER_ID  FILE_ID  -  SYSTEM_ID  KEY
    """
    entries: list[tuple[str, str, str, str]] = []
    with open(protocol_path, "r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 5:
                continue
            speaker_id, file_id, _unused, system_id, key = fields[:5]
            entries.append((speaker_id, file_id, system_id, key))
    if not entries:
        raise SystemExit(f"No usable lines in {protocol_path}")
    return entries


def pick_spread_across_attacks(
    spoof_entries: list[tuple[str, str, str, str]], target: int, rng: random.Random
) -> list[tuple[str, str, str, str]]:
    """Round-robin across attack ids so no single system dominates."""
    by_attack: dict[str, list] = defaultdict(list)
    for entry in spoof_entries:
        by_attack[entry[2]].append(entry)

    for attack in by_attack:
        rng.shuffle(by_attack[attack])

    chosen: list[tuple[str, str, str, str]] = []
    attacks = sorted(by_attack)
    while len(chosen) < target and any(by_attack[a] for a in attacks):
        for attack in attacks:
            if len(chosen) >= target:
                break
            if by_attack[attack]:
                chosen.append(by_attack[attack].pop())
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, type=Path, help="Extracted ASVspoof 2019 LA directory"
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--partition", choices=sorted(PARTITIONS), default="eval")
    parser.add_argument("--bonafide", type=int, default=100)
    parser.add_argument("--spoof", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    la_root = resolve_la_root(args.source)
    protocol_name, audio_subdir = PARTITIONS[args.partition]
    protocol_path = la_root / "ASVspoof2019_LA_cm_protocols" / protocol_name
    audio_dir = la_root / audio_subdir / "flac"

    if not protocol_path.is_file():
        raise SystemExit(f"Protocol not found: {protocol_path}")
    if not audio_dir.is_dir():
        raise SystemExit(f"Audio directory not found: {audio_dir}")

    entries = parse_protocol(protocol_path)
    rng = random.Random(args.seed)

    bonafide = [e for e in entries if e[3] == "bonafide"]
    spoof = [e for e in entries if e[3] == "spoof"]
    print(
        f"{args.partition}: {len(bonafide)} bona fide, {len(spoof)} spoof "
        f"across {len({e[2] for e in spoof})} attacks"
    )

    rng.shuffle(bonafide)
    chosen_bonafide = bonafide[: args.bonafide]
    chosen_spoof = pick_spread_across_attacks(spoof, args.spoof, rng)

    if len(chosen_bonafide) < args.bonafide or len(chosen_spoof) < args.spoof:
        print(
            f"WARNING: partition only yielded {len(chosen_bonafide)} bona fide / "
            f"{len(chosen_spoof)} spoof",
            file=sys.stderr,
        )

    # Sort by file id so the on-disk subset (and the workbench dropdown) does
    # not order clips by label.
    chosen = sorted(chosen_bonafide + chosen_spoof, key=lambda e: e[1])

    out_audio = args.dest / "flac"
    out_audio.mkdir(parents=True, exist_ok=True)

    copied = 0
    protocol_lines: list[str] = []
    for speaker_id, file_id, system_id, key in chosen:
        source_file = audio_dir / f"{file_id}.flac"
        if not source_file.is_file():
            print(f"WARNING: missing {source_file}", file=sys.stderr)
            continue
        shutil.copy2(source_file, out_audio / source_file.name)
        protocol_lines.append(f"{speaker_id} {file_id} - {system_id} {key}")
        copied += 1

    (args.dest / "protocol.txt").write_text(
        "\n".join(protocol_lines) + "\n", encoding="utf-8"
    )

    by_attack: dict[str, int] = defaultdict(int)
    for _s, _f, system_id, key in chosen:
        by_attack[system_id if key == "spoof" else "bonafide"] += 1

    total_bytes = sum(p.stat().st_size for p in out_audio.glob("*.flac"))
    print(f"\nWrote {copied} clips to {out_audio} ({total_bytes / 1e6:.1f} MB)")
    print(f"Wrote protocol to {args.dest / 'protocol.txt'}")
    print("Composition:", dict(sorted(by_attack.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
