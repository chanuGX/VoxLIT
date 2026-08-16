import React from "react";
import { TaskDefinition, TaskId, PredictionResultsProps, WorkbenchCenterProps } from "./types";
import { TranscriptionResults } from "@/features/transcription/TranscriptionResults";
import { ClassificationResults } from "@/features/emotion/ClassificationResults";
import { SpeakerVerificationWorkbench } from "@/features/verification";

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
    defaultDataset: null,
    allowCustomDatasets: false,
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
    name: "Voice Task B",
    shortDescription: "New audio interpretability task — details to be finalized.",
    status: "placeholder",
    models: [{ id: "task-b-model-1", label: "Model (to be added)", available: false }],
    defaultModel: null,
    datasets: [{ id: "task-b-dataset-1", label: "Dataset (to be added)", available: false }],
    defaultDataset: null,
    allowCustomDatasets: true,
    capabilities: {
      saliency: false,
      attention: false,
      perturbation: false,
      resultKind: null,
      batchAnalysis: null,
    },
  },
  {
    id: "task-c",
    route: "/tasks/task-c",
    name: "Voice Task C",
    shortDescription: "New audio interpretability task — details to be finalized.",
    status: "placeholder",
    models: [{ id: "task-c-model-1", label: "Model (to be added)", available: false }],
    defaultModel: null,
    datasets: [{ id: "task-c-dataset-1", label: "Dataset (to be added)", available: false }],
    defaultDataset: null,
    allowCustomDatasets: true,
    capabilities: {
      saliency: false,
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
  verification: { WorkbenchCenter: SpeakerVerificationWorkbench },
  "task-b": {},
  "task-c": {},
};