/**
 * Shared types for the VoxLIT task registry.
 *
 * These types are SHARED across all task pages — treat this file as frozen
 * after setup. To add/rename tasks, models, or datasets edit `registry.tsx`
 * instead.
 */

/**
 * Stable internal task ids. These appear in routes (`/tasks/<id>`) and backend
 * prefixes, so they must NEVER change once created — only display names in
 * `registry.tsx` change when a task is finalized.
 */
export type TaskId = "transcription" | "emotion" | "verification" | "task-b" | "task-c";

export type TaskStatus = "active" | "placeholder";

/**
 * Which prediction pipeline + results card the Datapoint Editor uses.
 * `null` = no prediction results section (placeholder tasks).
 */
export type ResultKind = "transcription" | "classification" | null;

/**
 * Which batch-analysis mode the Embedding panel's lower section uses.
 * `null` = no batch analysis section (placeholder tasks).
 */
export type BatchAnalysisKind = "transcript-terms" | "emotion-distribution" | null;

/** A selectable model. `available: false` renders as a disabled "to be added" entry. */
export interface ModelOption {
  id: string;
  label: string;
  available: boolean;
}

/** A selectable dataset. `available: false` renders as a disabled "to be added" entry. */
export interface DatasetOption {
  id: string;
  label: string;
  available: boolean;
}

export interface TaskCapabilities {
  /** Show the Saliency tab in the explainability panel. */
  saliency: boolean;
  /** Show the Attention tab in the explainability panel. */
  attention: boolean;
  /** Show the Perturbation tab in the explainability panel. */
  perturbation: boolean;
  resultKind: ResultKind;
  batchAnalysis: BatchAnalysisKind;
}

export interface TaskDefinition {
  id: TaskId;
  route: string;
  /** Display name — renaming a task is a one-line edit in registry.tsx. */
  name: string;
  /** Short description shown on the homepage task card. */
  shortDescription: string;
  status: TaskStatus;
  models: ModelOption[];
  defaultModel: string | null;
  datasets: DatasetOption[];
  defaultDataset: string | null;
  /** Whether the task page offers session-scoped custom dataset upload. */
  allowCustomDatasets: boolean;
  capabilities: TaskCapabilities;
}

/** File shape shared by upload/dataset/prediction flows across panels. */
export interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
  prediction?: string;
}

export interface Wav2Vec2Prediction {
  predicted_emotion: string;
  probabilities: Record<string, number>;
  confidence: number;
  ground_truth_emotion?: string;
}

export interface WhisperPrediction {
  predicted_transcript: string;
  ground_truth: string;
  accuracy_percentage: number | null;
  word_error_rate: number | null;
  character_error_rate: number | null;
  levenshtein_distance: number | null;
  exact_match: number | null;
  character_similarity: number | null;
  word_count_predicted: number;
  word_count_truth: number;
}

/** Props every per-task PredictionResults slot component receives. */
export interface PredictionResultsProps {
  selectedFile?: UploadedFile | null;
  selectedEmbeddingFile?: string | null;
  model?: string;
  /** Human-readable label for the current model (looked up from the registry). */
  modelLabel?: string;
  wav2vecPrediction?: Wav2Vec2Prediction | null;
  whisperPrediction?: WhisperPrediction | null;
  perturbedPredictions?: Wav2Vec2Prediction | WhisperPrediction | null;
  isLoading?: boolean;
  isLoadingPerturbed?: boolean;
  error?: string | null;
  showPerturbed?: boolean;
}
