export type SaliencyReferenceType = "cluster" | "enrollment";

export interface SaliencySegment {
  segment_index: number;
  start_seconds: number;
  end_seconds: number;
  occluded_similarity: number;
  similarity_change: number;
  influence_strength: number;
}

export interface SaliencyMapResponse {
  model: string;
  model_label: string;
  reference_type: SaliencyReferenceType;
  cluster_id: string | null;
  target_recording_id: string | null;
  reference_count: number;
  baseline_similarity: number;
  threshold: number;
  segment_count: number;
  audio_duration_seconds: number;
  interpretation: string;
  segments: SaliencySegment[];
}
