export type ReductionMethod = "pca" | "tsne" | "umap";

export interface ClusterSummary {
  cluster_id: string;
  member_count: number;
  member_indices: number[];
  member_labels: string[];
  mean_intra_cluster_similarity: number | null;
  min_intra_cluster_similarity: number | null;
  representative_index: number;
  representative_label: string;
  mean_fit_score: number;
}

export interface RecordingClusterStats {
  cluster_id: string;
  mean_similarity_to_cluster: number | null;
  min_similarity_to_cluster: number | null;
  nearest_index: number;
  nearest_label: string;
  nearest_similarity: number;
  nearest_in_same_cluster: boolean;
}

export interface BatchAnalysisResponse {
  model: string;
  model_label: string;
  threshold: number;
  recording_count: number;
  embedding_dimension: number;
  labels: string[];
  embeddings: number[][];
  similarity_matrix: number[][];
  decision_matrix: boolean[][];
  clustering_distance_threshold: number;
  clustering_threshold_version: string;
  cluster_labels: string[];
  cluster_fit_scores: number[];
  cluster_count: number;
  cluster_summaries: ClusterSummary[];
  recording_cluster_stats: RecordingClusterStats[];
}

export interface BatchProjectionRequestBody {
  model: string;
  embeddings: number[][];
  labels: string[];
  reduction_method: ReductionMethod;
  n_components: 2 | 3;
}

export interface BatchProjectionResponse {
  labels: string[];
  model: string;
  reduction_method: ReductionMethod;
  reduction_method_used: ReductionMethod;
  n_components: number;
  effective_components: number;
  coordinates: number[][];
}
