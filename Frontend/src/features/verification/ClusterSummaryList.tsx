import { Layers } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ClusterSummary } from "./batchTypes";

interface ClusterSummaryListProps {
  clusterSummaries: ClusterSummary[];
  clusterColorMap: Record<string, string>;
}

const formatScore = (value: number) => value.toFixed(4);

export const ClusterSummaryList = ({ clusterSummaries, clusterColorMap }: ClusterSummaryListProps) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <Layers className="h-4 w-4" /> Predicted clusters
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {clusterSummaries.length === 0 && (
          <p className="text-xs text-muted-foreground">No clusters to display.</p>
        )}
        {clusterSummaries.map((cluster) => (
          <div key={cluster.cluster_id} className="space-y-1.5 rounded-md border p-2 text-xs">
            <div className="flex items-center gap-2">
              <span
                className="h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ backgroundColor: clusterColorMap[cluster.cluster_id] }}
              />
              <span className="font-medium">{cluster.cluster_id}</span>
              <span className="text-muted-foreground">
                {cluster.member_count} recording{cluster.member_count === 1 ? "" : "s"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-1 text-muted-foreground">
              <span>
                Mean intra-cluster similarity:{" "}
                {cluster.mean_intra_cluster_similarity === null
                  ? "Not applicable"
                  : formatScore(cluster.mean_intra_cluster_similarity)}
              </span>
              <span>Mean cluster fit score: {formatScore(cluster.mean_fit_score)}</span>
            </div>
            <p className="truncate text-muted-foreground">{cluster.member_labels.join(", ")}</p>
          </div>
        ))}
      </CardContent>
    </Card>
  );
};
