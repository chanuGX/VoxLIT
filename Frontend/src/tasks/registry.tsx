import React from "react";
import { TaskDefinition, TaskId, PredictionResultsProps, WorkbenchCenterProps } from "./types";
import { TranscriptionResults } from "@/features/transcription/TranscriptionResults";
import { ClassificationResults } from "@/features/emotion/ClassificationResults";
import { SpeakerVerificationWorkbench, ClusterAssignmentResults } from "@/features/verification";
import { DiarizationWorkbench } from "@/features/task-b";
import { DeepfakeWorkbench } from "@/features/deepfake";

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * THE central VoxLIT task registry.
 *
 * This is the ONLY file to edit when you:
 *   • rename a task after it is finalized (change `name` / `shortDescription`)
 *   • add a model to a task (add a ModelOption; set `available: true` once the
 *     backend supports its id)
 *   • add a dataset to a task (add a DatasetOption; the id must match the
 *     backend dataset registry)
 *   • activate a placeholder task (flip `status` to "active" and fill in
 *     `capabilities` + `TASK_SLOTS` below)
 *
 * NEVER change a task's `id` or `route` — they are stable identifiers used in
 * URLs and backend route prefixes (Backend/app/tasks/<id>/).
 * ─────────────────────────────────────────────────────────────────────────────
 */
export const TASKS: TaskDefinition[] = [
  {
    id: "transcription",
    route: "/tasks/transcription",
    name: "Speech Transcription",
    shortDescription:
      "Inspect how Whisper transcribes speech: gradient saliency over the waveform, encoder–decoder attention, and WER/CER robustness under perturbation.",
    status: "active",
    models: [
      { id: "whisper-base", label: "Whisper Base", available: true },
      { id: "whisper-large", label: "Whisper Large", available: true },
      // Slot for a future model — flip to available:true once wired in the backend
      { id: "transcription-model-3", label: "Model (to be added)", available: false },
    ],
    defaultModel: "whisper-base",
    datasets: [
      { id: "common-voice", label: "Common Voice", available: true },
      { id: "transcription-dataset-2", label: "Dataset (to be added)", available: false },
    ],
    defaultDataset: "common-voice",
    allowCustomDatasets: true,
    capabilities: {
      saliency: true,
      attention: true,
      perturbation: true,
      resultKind: "transcription",
      batchAnalysis: "transcript-terms",
    },
  },
  {
    id: "emotion",
    route: "/tasks/emotion",
    name: "Emotion Recognition",
    shortDescription:
      "Probe a Wav2Vec2 emotion classifier: per-emotion probabilities, saliency over salient audio regions, and prediction shifts under perturbation.",
    status: "active",
    models: [
      { id: "wav2vec2", label: "Wav2Vec2", available: true },
      { id: "emotion-model-2", label: "Model (to be added)", available: false },
    ],
    defaultModel: "wav2vec2",
    datasets: [
      { id: "ravdess", label: "RAVDESS", available: true },
      { id: "emotion-dataset-2", label: "Dataset (to be added)", available: false },
    ],
    defaultDataset: "ravdess",
    allowCustomDatasets: true,
    capabilities: {
      saliency: true,
      attention: false,
      perturbation: true,
      resultKind: "classification",
      batchAnalysis: "emotion-distribution",
    },
  },
  {
    id: "verification",
    route: "/tasks/verification",
    name: "Speaker Verification",
    shortDescription:
      "Compare an enrolment profile with a probe recording using calibrated speaker embeddings, pair scores, cluster compactness, and perturbation-based explanations.",
    status: "active",
    models: [
      { id: "ecapa-tdnn", label: "ECAPA-TDNN", available: true },
      { id: "resnet34-lm", label: "WeSpeaker ResNet34-LM", available: true },
    ],
    defaultModel: "ecapa-tdnn",
    datasets: [{ id: "voxceleb1-indian-demo", label: "VoxCeleb1 Indian (Demo)", available: true }],
    defaultDataset: "voxceleb1-indian-demo",
    allowCustomDatasets: true,
    capabilities: {
      saliency: false,
      attention: false,
      perturbation: true,
      resultKind: null,
      batchAnalysis: null,
    },
  },
  {
    id: "task-b",
    route: "/tasks/task-b",
    name: "Speaker Diarization",
    shortDescription:
      "Glass-box diarization with pyannote 3.1: per-segment confidence from the clustering's own embedding space, uncertainty shading, and an embedding explorer.",
    status: "active",
    models: [{ id: "pyannote-3.1", label: "pyannote 3.1", available: true }],
    defaultModel: "pyannote-3.1",
    datasets: [{ id: "ami-subset", label: "AMI Meetings (3-file subset)", available: false }],
    defaultDataset: null,
    allowCustomDatasets: false,
    capabilities: {
      saliency: false,
      attention: false,
      perturbation: false,
      resultKind: null,
      batchAnalysis: null,
    },
  },
  {
    id: "deepfake",
    route: "/tasks/deepfake",
    name: "Audio Deepfake Detection",
    shortDescription:
      "Score speech as bona fide or synthetic with three detectors that fail differently — a wav2vec2 XLS-R classifier, a spectrogram transformer, and a dual-column state-space model — against an ASVspoof 2019 LA subset.",
    status: "active",
    models: [
      { id: "xlsr-deepfake", label: "wav2vec2 XLS-R (Model A)", available: true },
      { id: "ast-fakeaudio", label: "Audio Spectrogram Transformer (Model B)", available: true },
      { id: "xlsr-mamba", label: "XLSR-Mamba (Model C)", available: true },
    ],
    defaultModel: "xlsr-deepfake",
    // available:true so the toolbar names the dataset actually in use. Like
    // Speaker Verification's demo set, this dataset has no shared
    // /{dataset}/metadata route — AudioDatasetPanel special-cases it via
    // isDeepfakeDemoDataset and lists it from the task's own endpoint instead.
    datasets: [{ id: "asvspoof2019-la", label: "ASVspoof 2019 LA (subset)", available: true }],
    defaultDataset: "asvspoof2019-la",
    allowCustomDatasets: false,
    capabilities: {
      saliency: true,
      attention: false,
      perturbation: false,
      resultKind: null,
      batchAnalysis: null,
    },
  },
];

export const getTask = (id: string): TaskDefinition | undefined =>
  TASKS.find((t) => t.id === id);

/**
 * All built-in (non-custom) dataset ids that are available in any task.
 * Used by shared panels to decide which dataset ids have backend
 * `/{dataset}/metadata` + `/{dataset}/file/...` routes.
 */
export const BUILTIN_DATASET_IDS: string[] = Array.from(
  new Set(
    TASKS.flatMap((t) => t.datasets.filter((d) => d.available).map((d) => d.id))
  )
);

/**
 * The Speaker Verification demo dataset id. Its `/{dataset}/metadata` +
 * `/inferences/*` generic routes must NEVER be used (ground-truth safety —
 * see Backend/app/tasks/verification/dataset.py's module docstring). Panels
 * that would otherwise treat any BUILTIN_DATASET_IDS member generically must
 * special-case this id and route through `/tasks/verification/dataset/recordings`
 * instead.
 */
export const VERIFICATION_DEMO_DATASET_ID = "voxceleb1-indian-demo";
export const isVerificationDemoDataset = (datasetId?: string | null): boolean =>
  datasetId === VERIFICATION_DEMO_DATASET_ID;

/**
 * Speaker Verification's own custom-dataset selector prefix. Deliberately
 * distinct from Transcription/Emotion's `custom:{session_id}:{name}` (which
 * embeds a raw session id) — Verification's selector never carries a
 * session id anywhere: `verification-custom:<bare dataset name>`.
 */
/**
 * The Audio Deepfake Detection demo dataset id. Same constraint as the
 * Verification demo set above: it has no shared `/{dataset}/metadata` route,
 * and its bona fide/spoof answers must never reach the browser, so panels
 * must list it through `/tasks/deepfake/dataset/recordings` instead.
 */
export const DEEPFAKE_DEMO_DATASET_ID = "asvspoof2019-la";
export const isDeepfakeDemoDataset = (datasetId?: string | null): boolean =>
  datasetId === DEEPFAKE_DEMO_DATASET_ID;

export const VERIFICATION_CUSTOM_DATASET_PREFIX = "verification-custom:";

/** Tab-scoped, session-id-free persistence of the active Verification
 *  custom dataset across a page refresh. Holds only the bare dataset name. */
export const VERIFICATION_CUSTOM_DATASET_STORAGE_KEY = "voxlit:verification:custom-dataset";

/** Mirrors the backend's `validate_dataset_name` character class
 *  (`Backend/app/services/custom_dataset_service.py`) — used to validate a
 *  `sessionStorage`-restored dataset name locally before ever constructing
 *  a `verification-custom:` selector from it. */
export const isValidCustomDatasetName = (name: string): boolean =>
  /^[A-Za-z0-9 _-]{1,100}$/.test(name);

/** Label for a model id within a task (falls back to the raw id). */
export const getModelLabel = (task: TaskDefinition, modelId: string): string =>
  task.models.find((m) => m.id === modelId)?.label ?? modelId;

/**
 * Per-task UI slots. A task with no PredictionResults component renders no
 * results card in the Datapoint Editor (placeholder tasks). Members register
 * their components here when their task becomes active.
 */
export const TASK_SLOTS: Record<
  TaskId,
  {
    PredictionResults?: React.ComponentType<PredictionResultsProps>;
    WorkbenchCenter?: React.ComponentType<WorkbenchCenterProps>;
  }
> = {
  transcription: { PredictionResults: TranscriptionResults },
  emotion: { PredictionResults: ClassificationResults },
  verification: { WorkbenchCenter: SpeakerVerificationWorkbench, PredictionResults: ClusterAssignmentResults },
  "task-b": { WorkbenchCenter: DiarizationWorkbench },
  deepfake: { WorkbenchCenter: DeepfakeWorkbench },
};