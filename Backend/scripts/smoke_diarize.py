"""Standalone smoke test: diarize one wav on CPU, print segments.

Usage: python scripts/smoke_diarize.py path/to/audio.wav
Requires HF_TOKEN in the environment.
"""
import os
import sys
import time


def main() -> None:

    


    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/smoke_diarize.py <wav-path>")
    wav_path = sys.argv[1]
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set in environment")

    # Heavy imports deferred, same spirit as the repo's lazy-loading idiom
    from pyannote.audio import Pipeline
    import torch
    from torch.torch_version import TorchVersion
    from pyannote.audio.core.task import Specifications, Problem, Resolution

    torch.serialization.add_safe_globals(
        [TorchVersion, Specifications, Problem, Resolution]
    )

    print("loading pipeline (first run downloads ~100MB of weights)...")
    t0 = time.time()
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    print(f"pipeline loaded in {time.time() - t0:.1f}s")

    print(f"diarizing {wav_path} (CPU — expect minutes for long files)...")
    t0 = time.time()
    diarization = pipeline(wav_path)
    print(f"done in {time.time() - t0:.1f}s\n")

    speakers = set()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
        print(f"{turn.start:7.2f}s -> {turn.end:7.2f}s  {speaker}")
    print(f"\n{len(speakers)} speakers detected")


if __name__ == "__main__":
    main()