import type { RecordingClusterStats } from "./batchTypes";

/**
 * Cluster-assignment data for the currently selected recording, published by
 * BatchAnalysisPanel (which owns the batch result) and consumed by
 * ClusterAssignmentResults (registered as the verification task's
 * PredictionResults slot). A module-level store, not props, because the
 * frozen PredictionResultsProps contract carries no batch/cluster data --
 * see the plan's investigation notes.
 */
export interface ClusterAssignmentSnapshot {
  fileId: string;
  stats: RecordingClusterStats;
  clusterSize: number;
  modelLabel: string;
  clusteringDistanceThreshold: number;
  clusteringThresholdVersion: string;
}

let snapshot: ClusterAssignmentSnapshot | null = null;
const listeners = new Set<() => void>();

export const clusterAssignmentStore = {
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  getSnapshot(): ClusterAssignmentSnapshot | null {
    return snapshot;
  },
  publish(next: ClusterAssignmentSnapshot | null): void {
    snapshot = next;
    listeners.forEach((listener) => listener());
  },
};
