import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, PlayCircle } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useEmbedding } from "@/contexts/EmbeddingContext";
import { VERIFICATION_DEMO_DATASET_ID } from "@/tasks/registry";
import { ClusterSummaryList } from "./ClusterSummaryList";
import { PairComparisonCard } from "./PairComparisonCard";
import { buildClusterColorMap } from "./clusterColors";
import type {
  BatchAnalysisResponse,
  BatchProjectionRequestBody,
  BatchProjectionResponse,
  ReductionMethod,
} from "./batchTypes";

interface BatchAnalysisPanelProps {
  model: string;
  modelLabel: string;
  dataset: string; // effective ("custom" once uploads exist, else the raw toolbar dataset id)
  originalDataset: string;
  uploadedRawFiles: Record<string, File>;
  selectedBatchIds: string[];
  pairSelection: string[];
  onReprojectHandlerChange: (handler: ((method: string, n: number) => void) | null) => void;
  onLabelResolverChange: (resolver: ((label: string) => string | undefined) | null) => void;
}

const MIN_BATCH_SIZE = 2;
const MAX_BATCH_SIZE = 100;

const isAbortError = (caught: unknown) => caught instanceof DOMException && caught.name === "AbortError";

export const BatchAnalysisPanel = ({
  model,
  modelLabel,
  dataset,
  originalDataset,
  uploadedRawFiles,
  selectedBatchIds,
  pairSelection,
  onReprojectHandlerChange,
  onLabelResolverChange,
}: BatchAnalysisPanelProps) => {
  const { setEmbeddingDataDirect } = useEmbedding();

  const [batchResult, setBatchResult] = useState<BatchAnalysisResponse | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [isBatchRunning, setIsBatchRunning] = useState(false);
  const [isProjecting, setIsProjecting] = useState(false);
  const [projectionError, setProjectionError] = useState<string | null>(null);
  // Snapshot of selectedBatchIds at the moment Run was clicked — batchResult.labels
  // are index-aligned to this, not to the (possibly since-changed) live prop.
  const [submittedIds, setSubmittedIds] = useState<string[]>([]);

  const batchAbortRef = useRef<AbortController | null>(null);
  const projectionAbortRef = useRef<AbortController | null>(null);
  // Auto-run: fires runBatch() once per (model, dataset) pair, the first
  // time canRun becomes true for that pair. Covers "run once on initial
  // load" and "rerun when the user changes model," without refiring on
  // selectedBatchIds changes alone (so manual checkbox edits or row/point
  // clicks never auto-rerun it).
  const autoRunFiredForRef = useRef<string | null>(null);

  const inputMode: "dataset" | "upload" | "none" =
    dataset === "custom" ? "upload" : dataset === VERIFICATION_DEMO_DATASET_ID ? "dataset" : "none";

  const canRun =
    inputMode !== "none" &&
    selectedBatchIds.length >= MIN_BATCH_SIZE &&
    selectedBatchIds.length <= MAX_BATCH_SIZE &&
    !isBatchRunning;

  // Any change to model, dataset, or the checked batch selection invalidates
  // the current result — abort in-flight requests and clear everything
  // (including the published graph data) so a stale result never lingers.
  useEffect(() => {
    if (batchAbortRef.current) {
      batchAbortRef.current.abort();
      batchAbortRef.current = null;
    }
    if (projectionAbortRef.current) {
      projectionAbortRef.current.abort();
      projectionAbortRef.current = null;
    }
    setBatchResult(null);
    setBatchError(null);
    setProjectionError(null);
    setSubmittedIds([]);
    setEmbeddingDataDirect(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, dataset, selectedBatchIds]);

  const runBatch = async () => {
    if (!canRun) return;
    if (batchAbortRef.current) {
      batchAbortRef.current.abort();
    }
    if (projectionAbortRef.current) {
      projectionAbortRef.current.abort();
      projectionAbortRef.current = null;
    }
    const controller = new AbortController();
    batchAbortRef.current = controller;

    const idsSnapshot = [...selectedBatchIds];
    setSubmittedIds(idsSnapshot);
    setIsBatchRunning(true);
    setBatchError(null);
    setBatchResult(null);
    setProjectionError(null);
    setEmbeddingDataDirect(null);

    try {
      let response: Response;
      if (inputMode === "upload") {
        const formData = new FormData();
        formData.append("model", model);
        idsSnapshot.forEach((fileId) => {
          const raw = uploadedRawFiles[fileId];
          if (raw) formData.append("files", raw);
        });
        response = await fetch(`${API_BASE}/tasks/verification/batch/upload`, {
          method: "POST",
          credentials: "include",
          body: formData,
          signal: controller.signal,
        });
      } else if (inputMode === "dataset") {
        response = await fetch(`${API_BASE}/tasks/verification/batch/dataset`, {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model, recording_ids: idsSnapshot }),
          signal: controller.signal,
        });
      } else {
        return;
      }
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Batch analysis failed (${response.status}).`);
      }
      setBatchResult(payload as BatchAnalysisResponse);
    } catch (caught) {
      if (isAbortError(caught)) return;
      setBatchError(caught instanceof Error ? caught.message : "Batch analysis failed.");
    } finally {
      setIsBatchRunning(false);
      if (batchAbortRef.current === controller) {
        batchAbortRef.current = null;
      }
    }
  };

  useEffect(() => {
    if (inputMode !== "dataset" || !canRun) return;
    const key = `${model}|${dataset}`;
    if (autoRunFiredForRef.current === key) return;
    autoRunFiredForRef.current = key;
    runBatch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, dataset, canRun, inputMode]);

  // Re-project the already-computed embeddings and publish cluster-colored
  // points into the shared EmbeddingContext (never re-runs inference).
  const reprojectAndPublish = useCallback(
    async (method: string, n: number) => {
      if (!batchResult) return;
      if (projectionAbortRef.current) {
        projectionAbortRef.current.abort();
      }
      const controller = new AbortController();
      projectionAbortRef.current = controller;

      setIsProjecting(true);
      setProjectionError(null);

      const nComponents: 2 | 3 = n === 3 ? 3 : 2;
      const body: BatchProjectionRequestBody = {
        model: batchResult.model,
        embeddings: batchResult.embeddings,
        labels: batchResult.labels,
        reduction_method: method as ReductionMethod,
        n_components: nComponents,
      };

      try {
        const response = await fetch(`${API_BASE}/tasks/verification/batch/project`, {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || `Projection failed (${response.status}).`);
        }
        const projection = payload as BatchProjectionResponse;
        const clusterColorMap = buildClusterColorMap(batchResult.cluster_labels);
        setEmbeddingDataDirect({
          model: batchResult.model,
          dataset: originalDataset,
          reduction_method: method,
          n_components: nComponents,
          embeddings: [],
          total_files: batchResult.labels.length,
          original_dimension: batchResult.embedding_dimension,
          reduction_method_used: projection.reduction_method_used,
          effective_components: projection.effective_components,
          reduced_embeddings: batchResult.labels.map((label, i) => ({
            filename: label,
            coordinates: projection.coordinates[i],
            color: clusterColorMap[batchResult.cluster_labels[i]] ?? "#3b82f6",
            hoverExtra: `${batchResult.cluster_labels[i]} • fit ${batchResult.cluster_fit_scores[i].toFixed(2)}`,
          })),
        });
      } catch (caught) {
        if (isAbortError(caught)) return;
        setProjectionError(caught instanceof Error ? caught.message : "Projection failed.");
      } finally {
        setIsProjecting(false);
        if (projectionAbortRef.current === controller) {
          projectionAbortRef.current = null;
        }
      }
    },
    [batchResult, originalDataset, setEmbeddingDataDirect]
  );

  // Register the reproject handler with TaskWorkbench so EmbeddingPanel's own
  // PCA/UMAP/t-SNE + 2D/3D controls drive it — this also fires the first
  // projection automatically the moment a batch result arrives (EmbeddingPanel's
  // reproject effect re-runs whenever this handler reference changes).
  useEffect(() => {
    onReprojectHandlerChange(batchResult ? reprojectAndPublish : null);
    return () => onReprojectHandlerChange(null);
  }, [batchResult, reprojectAndPublish, onReprojectHandlerChange]);

  // Index-aligned label -> submitted id map (never derived from projected
  // coordinates or array position alone — always this explicit run-order contract).
  const labelToFileId = useCallback(
    (label: string): string | undefined => {
      if (!batchResult) return undefined;
      const index = batchResult.labels.indexOf(label);
      if (index === -1) return undefined;
      return submittedIds[index];
    },
    [batchResult, submittedIds]
  );

  useEffect(() => {
    onLabelResolverChange(batchResult ? labelToFileId : null);
    return () => onLabelResolverChange(null);
  }, [batchResult, labelToFileId, onLabelResolverChange]);

  const labelToIndex = new Map((batchResult?.labels ?? []).map((label, i) => [label, i]));
  const clusterColorMap = batchResult ? buildClusterColorMap(batchResult.cluster_labels) : {};

  const selectionSummary =
    inputMode === "upload"
      ? `${selectedBatchIds.length} uploaded recording${selectedBatchIds.length === 1 ? "" : "s"} selected`
      : inputMode === "dataset"
        ? `${selectedBatchIds.length} demo recording${selectedBatchIds.length === 1 ? "" : "s"} selected`
        : "Select a dataset or upload recordings, then check 2–100 rows in the Audio Dataset panel below.";

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="space-y-2 pt-4 text-xs">
          <p className="text-muted-foreground">{selectionSummary}</p>
          <Button size="sm" onClick={runBatch} disabled={!canRun}>
            <PlayCircle className="h-3.5 w-3.5" />
            {isBatchRunning ? "Extracting embeddings…" : "Run batch analysis"}
          </Button>
          {isProjecting && <p className="text-muted-foreground">Re-projecting embeddings…</p>}
          {projectionError && <p className="text-red-500">{projectionError}</p>}
        </CardContent>
      </Card>

      {batchError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Batch analysis could not be completed</AlertTitle>
          <AlertDescription>{batchError}</AlertDescription>
        </Alert>
      )}

      {isBatchRunning && !batchResult && (
        <div className="flex h-24 items-center justify-center text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            Analysing {submittedIds.length || selectedBatchIds.length} recordings with {modelLabel}…
          </div>
        </div>
      )}

      {batchResult && (
        <>
          <PairComparisonCard selectedLabels={pairSelection} batchResult={batchResult} labelToIndex={labelToIndex} />
          <ClusterSummaryList clusterSummaries={batchResult.cluster_summaries} clusterColorMap={clusterColorMap} />
        </>
      )}
    </div>
  );
};
