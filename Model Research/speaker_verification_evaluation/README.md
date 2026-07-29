
# Speaker Verification Model Evaluation

This folder contains the research and evaluation pipeline used to select speaker-verification models for integration into VoxLIT.

The evaluation compares three pretrained speaker-verification models using the same dataset, preprocessing, calibration pairs, test pairs, scoring method, and evaluation metrics.

## Evaluated Models

| Model Key     | Model                                       | Role      | Embedding Dimension |
| ------------- | ------------------------------------------- | --------- | ------------------: |
| `xvector`   | `speechbrain/spkrec-xvect-voxceleb`       | Baseline  |                 512 |
| `ecapa`     | `speechbrain/spkrec-ecapa-voxceleb`       | Candidate |                 192 |
| `wespeaker` | `pyannote/wespeaker-voxceleb-resnet34-LM` | Candidate |                 256 |

All models use cosine similarity to compare speaker embeddings.

## Dataset

The evaluation uses a VoxCeleb1 subset containing Indian celebrity speech recordings.

The dataset structure is:

```text
speaker_id/
└── event_id/
    ├── 00001.wav
    ├── 00002.wav
    └── ...
```

Recordings inside the same event folder come from the same recording event.

Same-speaker verification pairs were created using audio clips from different event folders whenever possible. This reduces the possibility of the model matching recording conditions instead of speaker characteristics.

The prepared dataset contains:

- `utterances.csv`
- `calibration_pairs.csv`
- `test_pairs.csv`
- `summary.json`

The calibration and test sets are speaker-disjoint.

### Pair Distribution

| Split       | Same-Speaker Pairs | Different-Speaker Pairs | Total |
| ----------- | -----------------: | ----------------------: | ----: |
| Calibration |              1,000 |                   1,000 | 2,000 |
| Test        |              2,500 |                   2,500 | 5,000 |

## Evaluation Process

The evaluation process is:

```text
Audio files
    ↓
Extract one embedding per audio file
    ↓
Save embeddings as .npy files
    ↓
Score calibration pairs using cosine similarity
    ↓
Select a model-specific threshold
    ↓
Score test pairs
    ↓
Calculate final evaluation metrics
```

Each audio file is processed only once per model. Saved embeddings are reused when scoring multiple verification pairs.

## Threshold Calibration

Each model produces similarity scores in a different range. Therefore, each model uses its own calibrated threshold.

The threshold is selected from the calibration pairs using the Equal Error Rate criterion.

For each possible threshold:

```text
Similarity >= threshold → Same speaker
Similarity < threshold  → Different speakers
```

The selected threshold is the point where the False Acceptance Rate and False Rejection Rate are closest.

The calibration threshold is locked before test evaluation.

| Model                 | Locked Threshold |
| --------------------- | ---------------: |
| X-vector              |           0.9341 |
| ECAPA-TDNN            |           0.3579 |
| WeSpeaker ResNet34-LM |           0.3843 |

## Test Results

| Metric            | X-vector Baseline |       ECAPA-TDNN | WeSpeaker ResNet34-LM |
| ----------------- | ----------------: | ---------------: | --------------------: |
| Accuracy          |            85.58% | **99.02%** |                97.94% |
| Balanced Accuracy |            85.58% | **99.02%** |                97.94% |
| F1 Score          |            86.04% | **99.01%** |                97.92% |
| ROC-AUC           |            0.9392 | **0.9993** |                0.9985 |
| EER               |            14.08% |  **1.08%** |                 2.00% |
| FAR               |            17.72% |  **0.20%** |                 0.96% |
| FRR               |            11.12% |  **1.76%** |                 3.16% |

## Confusion Matrix Results

| Model                 | True Positive | True Negative | False Positive | False Negative |
| --------------------- | ------------: | ------------: | -------------: | -------------: |
| X-vector              |         2,222 |         2,057 |            443 |            278 |
| ECAPA-TDNN            |         2,456 |         2,495 |              5 |             44 |
| WeSpeaker ResNet34-LM |         2,421 |         2,476 |             24 |             79 |

## Model Selection

### ECAPA-TDNN

ECAPA-TDNN achieved the best overall results:

- 99.02% accuracy
- 0.9993 ROC-AUC
- 1.08% EER
- 0.20% FAR
- 1.76% FRR

It is selected as the primary and recommended speaker-verification model for VoxLIT.

### WeSpeaker ResNet34-LM

WeSpeaker also achieved strong results:

- 97.94% accuracy
- 0.9985 ROC-AUC
- 2.00% EER
- 0.96% FAR
- 3.16% FRR

It is selected as a lightweight alternative model and provides a different ResNet-based speaker-embedding architecture.

### X-vector

The x-vector model is retained as the research baseline.

Its performance was significantly lower than ECAPA and WeSpeaker, particularly in False Acceptance Rate and Equal Error Rate.

## Final Decision

The following models were selected for VoxLIT integration:

```text
Speaker Verification
├── ECAPA-TDNN
│   └── Recommended primary model
│
└── WeSpeaker ResNet34-LM
    └── Lightweight alternative model
```

The x-vector model remains available only as the baseline used for model comparison.

The two selected models must remain independent because they produce embeddings with different dimensions and embedding spaces.

- ECAPA-TDNN: 192 dimensions
- WeSpeaker: 256 dimensions

Their raw embeddings must not be directly combined.

## Source Structure

```text
src/
├── models/
│   ├── base_adapter.py
│   ├── speechbrain_adapter.py
│   ├── xvector_adapter.py
│   ├── ecapa_adapter.py
│   ├── wespeaker_adapter.py
│   └── registry.py
├── evaluation/
│   └── metrics.py
├── extract_embeddings.py
├── score_pairs.py
├── calibrate_threshold.py
└── evaluate_model.py
```

## Running the Evaluation Pipeline

Available model keys:

```text
xvector
ecapa
wespeaker
```

### 1. Extract Embeddings

```powershell
python src\extract_embeddings.py `
  --model <model_key> `
  --utterances "data\prepared\voxceleb1_indian_verification\utterances.csv" `
  --audio-root "data\raw\voxceleb1_indian\vox1_indian\content\vox_indian"
```

Example:

```powershell
python src\extract_embeddings.py `
  --model wespeaker `
  --utterances "data\prepared\voxceleb1_indian_verification\utterances.csv" `
  --audio-root "data\raw\voxceleb1_indian\vox1_indian\content\vox_indian"
```

### 2. Score Calibration Pairs

```powershell
python src\score_pairs.py `
  --model <model_key> `
  --pairs "data\prepared\voxceleb1_indian_verification\calibration_pairs.csv" `
  --utterances "data\prepared\voxceleb1_indian_verification\utterances.csv" `
  --output "outputs\pair_scores\<model_key>_calibration_scores.csv"
```

### 3. Calibrate the Threshold

```powershell
python src\calibrate_threshold.py `
  --model <model_key> `
  --scores "outputs\pair_scores\<model_key>_calibration_scores.csv" `
  --criterion eer
```

### 4. Score Test Pairs

```powershell
python src\score_pairs.py `
  --model <model_key> `
  --pairs "data\prepared\voxceleb1_indian_verification\test_pairs.csv" `
  --utterances "data\prepared\voxceleb1_indian_verification\utterances.csv" `
  --output "outputs\pair_scores\<model_key>_test_scores.csv"
```

### 5. Evaluate the Model

```powershell
python src\evaluate_model.py `
  --model <model_key> `
  --scores "outputs\pair_scores\<model_key>_test_scores.csv" `
  --threshold-json "outputs\metrics\<model_key>_threshold.json"
```

## Generated Outputs

```text
outputs/
├── embeddings/
│   ├── xvector/
│   ├── ecapa/
│   └── wespeaker/
├── pair_scores/
├── metrics/
└── predictions/
```

Important output files include:

```text
outputs/metrics/xvector_threshold.json
outputs/metrics/xvector_test_metrics.json

outputs/metrics/ecapa_threshold.json
outputs/metrics/ecapa_test_metrics.json

outputs/metrics/wespeaker_threshold.json
outputs/metrics/wespeaker_test_metrics.json
```

## Limitations

The evaluation dataset is an Indian celebrity subset of VoxCeleb1.

The evaluated pretrained models were trained using VoxCeleb data. Therefore, possible overlap between pretrained-model training speakers and the evaluation subset may produce optimistic results.
