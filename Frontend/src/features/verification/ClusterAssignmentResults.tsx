import { useSyncExternalStore } from "react";
import { Target } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { PredictionResultsProps } from "@/tasks/types";
import { clusterAssignmentStore } from "./clusterAssignmentStore";

const formatScore = (value: number) => value.toFixed(4);

/**
 * SV-FR-23 "Cluster Assignment Results" per-clip card, registered as the
 * verification task's PredictionResults slot (see registry.tsx). Its data
 * comes from clusterAssignmentStore, not from these props -- the frozen
 * PredictionResultsProps contract carries no batch/cluster fields, so
 * BatchAnalysisPanel (which owns the batch result) publishes into the store
 * instead. `selectedFile` here only confirms the published snapshot still
 * belongs to the current selection.
 */
export const ClusterAssignmentResults = ({ selectedFile }: PredictionResultsProps) => {
  const snapshot = useSyncExternalStore(
    clusterAssignmentStore.subscribe,
    clusterAssignmentStore.getSnapshot
  );

  const cardHeader = (
    <CardHeader>
      <CardTitle className="flex items-center gap-2 text-sm">
        <Target className="h-4 w-4" /> Cluster assignment results
      </CardTitle>
    </CardHeader>
  );

  if (!selectedFile) {
    return (
      <Card>
        {cardHeader}
        <CardContent>
          <p className="text-xs text-muted-foreground">Select a recording to see its cluster assignment.</p>
        </CardContent>
      </Card>
    );
  }

  if (!snapshot || snapshot.fileId !== selectedFile.file_id) {
    return (
      <Card>
        {cardHeader}
        <CardContent>
          <p className="text-xs text-muted-foreground">
            {snapshot
              ? "This recording is not part of the current batch results."
              : "Run a batch analysis to see cluster assignment results."}
          </p>
        </CardContent>
      </Card>
    );
  }

  const { stats, clusterSize, modelLabel, clusteringDistanceThreshold, groundTruthGroup, groundTruthAvailable } =
    snapshot;

  return (
    <Card>
      {cardHeader}
      <CardContent className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs">
        <span className="text-muted-foreground">Predicted cluster</span>
        <Badge variant="secondary" className="w-fit">{stats.cluster_id}</Badge>

        <span className="text-muted-foreground">Ground-truth speaker group</span>
        {groundTruthAvailable && groundTruthGroup !== null ? (
          <Badge variant="secondary" className="w-fit">{groundTruthGroup}</Badge>
        ) : (
          <span className="text-muted-foreground italic">Ground-truth evaluation unavailable for this batch.</span>
        )}

        <span className="text-muted-foreground">Cluster size</span>
        <span>{clusterSize} recording{clusterSize === 1 ? "" : "s"}</span>

        <span className="text-muted-foreground">Avg. similarity to cluster</span>
        <span>
          {stats.mean_similarity_to_cluster === null
            ? "Not applicable (single-recording cluster)"
            : formatScore(stats.mean_similarity_to_cluster)}
        </span>

        <span className="text-muted-foreground">Min intra-cluster similarity</span>
        <span>
          {stats.min_similarity_to_cluster === null
            ? "Not applicable (single-recording cluster)"
            : formatScore(stats.min_similarity_to_cluster)}
        </span>

        <span className="text-muted-foreground">Nearest audio clip</span>
        <span className="truncate">
          {stats.nearest_label}
          {!stats.nearest_in_same_cluster && (
            <Badge variant="outline" className="ml-1.5 text-[9px]">different cluster</Badge>
          )}
        </span>

        <span className="text-muted-foreground">Nearest-neighbour similarity</span>
        <span>{formatScore(stats.nearest_similarity)}</span>

        <span className="text-muted-foreground">Selected model</span>
        <span>{modelLabel}</span>

        <span className="text-muted-foreground">Clustering distance threshold</span>
        <span>{formatScore(clusteringDistanceThreshold)}</span>
      </CardContent>
    </Card>
  );
};
