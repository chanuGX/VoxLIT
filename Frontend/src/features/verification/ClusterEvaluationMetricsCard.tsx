import { Target } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { EvaluationMetrics } from "./batchTypes";

interface ClusterEvaluationMetricsCardProps {
  evaluationMetrics: EvaluationMetrics | null;
  groundTruthAvailable: boolean;
  predictedClusterCount: number;
  trueSpeakerCount: number | null;
}

const formatScore = (value: number) => value.toFixed(4);
const formatCount = (value: number) => value.toLocaleString();

export const ClusterEvaluationMetricsCard = ({
  evaluationMetrics,
  groundTruthAvailable,
  predictedClusterCount,
  trueSpeakerCount,
}: ClusterEvaluationMetricsCardProps) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <Target className="h-4 w-4" /> Ground-truth evaluation
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-xs">
        {!groundTruthAvailable || !evaluationMetrics ? (
          <p className="text-muted-foreground">
            Ground-truth speaker groups were not provided for this batch, so clustering-quality
            metrics are not available. Upload recordings with known speaker groupings to see them
            here.
          </p>
        ) : (
          <>
            <div className="space-y-1">
              <p className="font-medium">Partition agreement</p>
              <div className="grid grid-cols-3 gap-1 text-muted-foreground">
                <span>ARI: {formatScore(evaluationMetrics.adjusted_rand_index)}</span>
                <span>NMI: {formatScore(evaluationMetrics.normalized_mutual_information)}</span>
                <span>Purity: {formatScore(evaluationMetrics.cluster_purity)}</span>
              </div>
            </div>

            <div className="space-y-1">
              <p className="font-medium">Pairwise rates</p>
              <div className="grid grid-cols-2 gap-1 text-muted-foreground">
                <span>Precision: {formatScore(evaluationMetrics.pairwise_precision)}</span>
                <span>Recall: {formatScore(evaluationMetrics.pairwise_recall)}</span>
                <span>F1 score: {formatScore(evaluationMetrics.pairwise_f1_score)}</span>
                <span>Accuracy: {formatScore(evaluationMetrics.pairwise_accuracy)}</span>
              </div>
            </div>

            <div className="space-y-1">
              <p className="font-medium">Pair counts</p>
              <div className="grid grid-cols-3 gap-1 text-muted-foreground">
                <span>Total: {formatCount(evaluationMetrics.total_unique_pairs)}</span>
                <span>True positive: {formatCount(evaluationMetrics.true_positive_pairs)}</span>
                <span>True negative: {formatCount(evaluationMetrics.true_negative_pairs)}</span>
                <span>False positive: {formatCount(evaluationMetrics.false_positive_pairs)}</span>
                <span>False negative: {formatCount(evaluationMetrics.false_negative_pairs)}</span>
              </div>
            </div>

            <div className="space-y-1">
              <p className="font-medium">Speaker count</p>
              <div className="grid grid-cols-2 gap-1 text-muted-foreground">
                <span>True speaker groups: {trueSpeakerCount ?? "Not available"}</span>
                <span>Predicted clusters: {predictedClusterCount}</span>
              </div>
            </div>

            <p className="text-muted-foreground">
              These metrics compare the predicted clusters to the ground-truth groups for this
              batch only. They describe how well clustering performed on these recordings, not
              the model&apos;s overall or production-level accuracy.
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
};
