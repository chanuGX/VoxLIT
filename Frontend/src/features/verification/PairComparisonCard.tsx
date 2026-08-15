import { CheckCircle2, GitCompareArrows, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { BatchAnalysisResponse } from "./batchTypes";

interface PairComparisonCardProps {
  selectedLabels: string[];
  batchResult: BatchAnalysisResponse;
  labelToIndex: Map<string, number>;
}

const formatScore = (value: number) => value.toFixed(4);

export const PairComparisonCard = ({ selectedLabels, batchResult, labelToIndex }: PairComparisonCardProps) => {
  const hasPair = selectedLabels.length === 2;
  const indexA = hasPair ? labelToIndex.get(selectedLabels[0]) : undefined;
  const indexB = hasPair ? labelToIndex.get(selectedLabels[1]) : undefined;
  const canCompare = indexA !== undefined && indexB !== undefined;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <GitCompareArrows className="h-4 w-4" /> Pair comparison
        </CardTitle>
      </CardHeader>
      <CardContent>
        {!hasPair && (
          <p className="text-xs text-muted-foreground">Select exactly two points in the plot to compare.</p>
        )}
        {hasPair && !canCompare && (
          <p className="text-xs text-muted-foreground">Selection is no longer valid for the current batch.</p>
        )}
        {canCompare && indexA !== undefined && indexB !== undefined && (
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="truncate">{batchResult.labels[indexA]}</span>
              <span className="text-muted-foreground">↔</span>
              <span className="truncate">{batchResult.labels[indexB]}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Cosine similarity</span>
              <Badge variant="secondary">{formatScore(batchResult.similarity_matrix[indexA][indexB])}</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Pair threshold ({batchResult.model_label})</span>
              <Badge variant="secondary">{formatScore(batchResult.threshold)}</Badge>
            </div>
            <div className="flex items-center gap-2">
              {batchResult.decision_matrix[indexA][indexB] ? (
                <>
                  <CheckCircle2 className="h-4 w-4 text-emerald-700" />
                  <span className="font-medium text-emerald-700">Same speaker</span>
                </>
              ) : (
                <>
                  <XCircle className="h-4 w-4 text-rose-700" />
                  <span className="font-medium text-rose-700">Different speakers</span>
                </>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
