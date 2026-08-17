#!/usr/bin/env bash
# Download the 3-meeting AMI headset-mix subset + ground-truth RTTMs.
# Audio: AMI corpus (CC BY 4.0). RTTMs: pyannote/AMI-diarization-setup.
set -euo pipefail

DEST="data/speaker_diarization/ami_subset"
MIRROR="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
RTTM_BASE="https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test"

mkdir -p "$DEST/rttm"

for m in ES2004a IS1009a TS3003a; do
  echo "== $m =="
  curl -L --fail -o "$DEST/${m}.Mix-Headset.wav" "$MIRROR/${m}/audio/${m}.Mix-Headset.wav"
  curl -L --fail -o "$DEST/rttm/${m}.rttm" "$RTTM_BASE/${m}.rttm"
done

echo && ls -lh "$DEST" "$DEST/rttm"