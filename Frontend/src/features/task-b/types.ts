/** Response shapes from /tasks/task-b (see Backend/app/tasks/task_b/router.py). */

export interface RecordingInfo {
  recording_id: string;
  display_filename: string;
  extension: string;
  size_bytes: number;
}

export type ConfidenceBucket = "high" | "medium" | "uncertain" | null;

export interface DiarizationSegment {
  id: string;
  start: number;
  end: number;
  speaker: string;
  confidence: number | null;
  confidence_bucket: ConfidenceBucket;
}

export interface DiarizationResult {
  model: string;
  recording_id: string;
  duration: number;
  num_speakers: number;
  speakers: string[];
  segments: DiarizationSegment[];
  embeddings: Record<string, number[]>;
  cached: boolean;
}

export interface ProjectionPoint {
  id: string;
  x: number;
  y: number;
  speaker: string;
  confidence: number | null;
}

export interface ProjectionResult {
  model: string;
  recording_id: string;
  points: ProjectionPoint[];
}

/** Stable palette: speaker → color by index in result.speakers. */
export const SPEAKER_COLORS = [
  "#2563eb", // blue
  "#dc2626", // red
  "#16a34a", // green
  "#d97706", // amber
  "#9333ea", // purple
  "#0891b2", // cyan
  "#db2777", // pink
  "#65a30d", // lime
];

export const speakerColor = (speakers: string[], speaker: string): string =>
  SPEAKER_COLORS[Math.max(0, speakers.indexOf(speaker)) % SPEAKER_COLORS.length];