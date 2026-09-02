import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, Download, PlayCircle } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useEmbedding } from "@/contexts/EmbeddingContext";
import { VERIFICATION_DEMO_DATASET_ID } from "@/tasks/registry";
import type { UploadedFile } from "@/tasks/types";
import { ClusterSummaryList } from "./ClusterSummaryList";
import { PairComparisonCard } from "./PairComparisonCard";
import { buildClusterColorMap } from "./clusterColors";
import { clusterAssignmentStore } from "./clusterAssignmentStore";
import { SpeakerSaliencyMap } from "./SpeakerSaliencyMap";
import { verificationAudioUrl } from "./audioUrl";
import type { SaliencyMapResponse } from "./saliencyTypes";
import type {
  BatchAnalysisResponse,
  BatchExportRequestBody,
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
  /** Currently selected recording (table row click or graph point click) --
   *  drives cluster-based saliency for whichever recording is selected. */
  selectedFile: UploadedFile | null;
  onReprojectHandlerChange: (handler: ((method: string, n: number) => void) | null) => void;
  onLabelResolverChange: (resolver: ((label: string) => string | undefined) | null) => void;
}

const DEFAULT_SALIENCY_SEGMENT_COUNT = 8;
const isBackendResolvableId = (id: string) => id.startsWith("rec_") || id.startsWith("asset_") || id.startsWith("crec_");

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
  selectedFile,
  onReprojectHandlerChange,
  onLabelResolverChange,
}: BatchAnalysisPanelProps) => {
  const { setEmbeddingDataDirect, focusedClusterId, setFocusedClusterId } = useEmbedding();

  const [batchResult, setBatchResult] = useState<BatchAnalysisResponse | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [isBatchRunning, setIsBatchRunning] = useState(false);
  const [isProjecting, setIsProjecting] = useState(false);
  const [projectionError, setProjectionError] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  // Snapshot of selectedBatchIds at the moment Run was clicked — batchResult.labels
  // are index-aligned to this, not to the (possibly since-changed) live prop.
  const [submittedIds, setSubmittedIds] = useState<string[]>([]);

  // Cluster-based saliency (Mode 1) -- see SpeakerSaliencyMap.
  const [saliencyResult, setSaliencyResult] = useState<SaliencyMapResponse | null>(null);
  const [saliencyError, setSaliencyError] = useState<string | null>(null);
  const [isSaliencyLoading, setIsSaliencyLoading] = useState(false);
  const [saliencySegmentCount, setSaliencySegmentCount] = useState(DEFAULT_SALIENCY_SEGMENT_COUNT);
  const saliencyAbortRef = useRef<AbortController | null>(null);
  const isFirstSegmentCountRender = useRef(true);

  const batchAbortRef = useRef<AbortController | null>(null);
  const projectionAbortRef = useRef<AbortController | null>(null);
  const exportAbortRef = useRef<AbortController | null>(null);
  // Auto-run: fires runBatch() once per (model, dataset) pair, the first
  // time canRun becomes true for that pair. Covers "run once on initial
  // load" and "rerun when the user changes model," without refiring on
  // selectedBatchIds changes alone (so manual checkbox edits or row/point
  // clicks never auto-rerun it).
  const autoRunFiredForRef = useRef<string | null>(null);

  // "dataset" mode covers any real dataset selection -- the built-in demo
  // AND any owned custom dataset -- since /batch/dataset's request body is
  // already fully generic (just a model + a list of recording ids resolved
  // server-side); only the legacy bare "custom" sentinel (raw uploads, no
  // dataset selected) is "upload" mode.
  const inputMode: "dataset" | "upload" | "none" =
    dataset === "custom" ? "upload" : dataset ? "dataset" : "none";

  const canRun =
    inputMode !== "none" &&
    selectedBatchIds.length >= MIN_BATCH_SIZE &&
    selectedBatchIds.length <= MAX_BATCH_SIZE &&
    !isBatchRunning;

  // Any change to model, dataset, or the checked batch selection invalidates
  // the current result — abort in-flight requests and clear everything
  // (including the published graph data) so a stale result never lingers.
  // A replaced/cleared batchResult also invalidates cluster-based saliency,
  // since cluster membership was only ever derived from that result.
  useEffect(() => {
    if (batchAbortRef.current) {
      batchAbortRef.current.abort();
      batchAbortRef.current = null;
    }
    if (projectionAbortRef.current) {
      projectionAbortRef.current.abort();
      projectionAbortRef.current = null;
    }
    if (saliencyAbortRef.current) {
      saliencyAbortRef.current.abort();
      saliencyAbortRef.current = null;
    }
    if (exportAbortRef.current) {
      exportAbortRef.current.abort();
      exportAbortRef.current = null;
    }
    setBatchResult(null);
    setBatchError(null);
    setProjectionError(null);
    setExportError(null);
    setSubmittedIds([]);
    setEmbeddingDataDirect(null);
    setFocusedClusterId(null);
    setSaliencyResult(null);
    setSaliencyError(null);
    clusterAssignmentStore.publish(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, dataset, selectedBatchIds]);

  // Cluster-assignment card (SV-FR-23, rendered in the Datapoint Editor) reads
  // from a module-level store instead of props -- see clusterAssignmentStore.ts.
  // Publishes the selected recording's entry whenever the batch result or
  // selection changes; clears on unmount so a stale card can never outlive
  // this panel.
  useEffect(() => {
    if (!batchResult || !selectedFile) {
      clusterAssignmentStore.publish(null);
      return;
    }
    const index = submittedIds.indexOf(selectedFile.file_id);
    if (index === -1) {
      clusterAssignmentStore.publish(null);
      return;
    }
    const stats = batchResult.recording_cluster_stats[index];
    const clusterSummary = batchResult.cluster_summaries.find(
      (summary) => summary.cluster_id === stats.cluster_id
    );
    clusterAssignmentStore.publish({
      fileId: selectedFile.file_id,
      stats,
      clusterSize: clusterSummary?.member_count ?? 1,
      modelLabel: batchResult.model_label,
      clusteringDistanceThreshold: batchResult.clustering_distance_threshold,
      clusteringThresholdVersion: batchResult.clustering_threshold_version,
    });
  }, [batchResult, selectedFile, submittedIds]);

  useEffect(() => {
    return () => clusterAssignmentStore.publish(null);
  }, []);

  // Selecting a different recording invalidates any prior saliency map (it
  // explained a different target) -- never touches selectedBatchIds, graph
  // selection, or camera state.
  useEffect(() => {
    if (saliencyAbortRef.current) {
      saliencyAbortRef.current.abort();
      saliencyAbortRef.current = null;
    }
    setSaliencyResult(null);
    setSaliencyError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFile?.file_id]);

  // Changing the segment-count control invalidates a displayed map, since it
  // no longer reflects the currently selected segment count.
  useEffect(() => {
    if (isFirstSegmentCountRender.current) {
      isFirstSegmentCountRender.current = false;
      return;
    }
    setSaliencyResult(null);
    setSaliencyError(null);
  }, [saliencySegmentCount]);

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
    setFocusedClusterId(null);

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
    // Auto-run is reserved for the built-in demo dataset only -- a newly
    // selected/loaded custom dataset must never run automatically; the user
    // explicitly clicks "Run batch analysis" for those.
    if (dataset !== VERIFICATION_DEMO_DATASET_ID) return;
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
            clusterId: batchResult.cluster_labels[i],
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

  // Serializes the already-computed batchResult to CSV server-side (SV-FR-36)
  // -- never re-runs the model, never recomputes embeddings/similarity/clustering.
  const exportBatchCsv = async () => {
    if (!batchResult) return;
    if (exportAbortRef.current) {
      exportAbortRef.current.abort();
    }
    const controller = new AbortController();
    exportAbortRef.current = controller;

    setIsExporting(true);
    setExportError(null);

    const body: BatchExportRequestBody = {
      model: batchResult.model,
      labels: batchResult.labels,
      cluster_labels: batchResult.cluster_labels,
      cluster_summaries: batchResult.cluster_summaries,
      recording_cluster_stats: batchResult.recording_cluster_stats,
      cluster_count: batchResult.cluster_count,
    };

    try {
      const response = await fetch(`${API_BASE}/tasks/verification/batch/export`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      // A successful response here is CSV, not JSON, unlike every other call
      // in this file -- response.ok must be checked before deciding whether
      // to parse an error's JSON detail or read a blob.
      if (!response.ok) {
        const errorPayload = await response.json().catch(() => ({}) as { detail?: string });
        throw new Error(errorPayload.detail || `Export failed (${response.status}).`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `voxlit-cluster-export-${batchResult.model}-${stamp}.csv`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (caught) {
      if (isAbortError(caught)) return;
      setExportError(caught instanceof Error ? caught.message : "Export failed.");
    } finally {
      setIsExporting(false);
      if (exportAbortRef.current === controller) {
        exportAbortRef.current = null;
      }
    }
  };

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

  // Cluster membership, resolved entirely client-side from the last batch
  // result -- the backend never stores or looks up cluster assignments.
  // Explicit `!== null && !== undefined` checks throughout: a cluster label
  // is an opaque string ("cluster-1", ...), but the lookup itself must never
  // treat "not found" the same as a falsy-but-valid label via truthiness.
  const targetIndex =
    selectedFile && batchResult ? submittedIds.indexOf(selectedFile.file_id) : -1;
  const targetClusterId: string | null =
    targetIndex >= 0 ? batchResult!.cluster_labels[targetIndex] : null;
  const hasTargetCluster = targetClusterId !== null && targetClusterId !== undefined;
  const clusterMemberIds = hasTargetCluster
    ? submittedIds.filter((_, i) => i !== targetIndex && batchResult!.cluster_labels[i] === targetClusterId)
    : [];
  const isSingletonCluster = hasTargetCluster && clusterMemberIds.length === 0;
  // Only demo (`rec_...`) and registered session-asset (`asset_...`) ids are
  // backend-resolvable -- a raw batch-upload id (client-only, never
  // registered via session-assets/upload) is not, and would 404.
  const isTargetIdResolvable = !!selectedFile && isBackendResolvableId(selectedFile.file_id);

  const saliencyEmptyStateMessage = !hasTargetCluster
    ? null
    : !isTargetIdResolvable
      ? "Cluster saliency needs recordings uploaded via the Speaker Verification upload button, not a raw batch upload."
      : isSingletonCluster
        ? "This recording's predicted cluster has no other members — cluster-based saliency requires at least one other recording in the same cluster."
        : null;

  const runClusterSaliency = async () => {
    if (!selectedFile || !hasTargetCluster || !targetClusterId || isSingletonCluster || !isTargetIdResolvable) return;
    if (saliencyAbortRef.current) {
      saliencyAbortRef.current.abort();
    }
    const controller = new AbortController();
    saliencyAbortRef.current = controller;

    const formData = new FormData();
    formData.append("model", model);
    formData.append("reference_type", "cluster");
    clusterMemberIds.forEach((id) => formData.append("reference_recording_ids", id));
    formData.append("target_recording_id", selectedFile.file_id);
    formData.append("cluster_id", targetClusterId);
    formData.append("segment_count", String(saliencySegmentCount));

    setIsSaliencyLoading(true);
    setSaliencyError(null);
    setSaliencyResult(null);

    try {
      const response = await fetch(`${API_BASE}/tasks/verification/explain/saliency`, {
        method: "POST",
        credentials: "include",
        body: formData,
        signal: controller.signal,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Saliency map failed (${response.status}).`);
      }
      setSaliencyResult(payload as SaliencyMapResponse);
    } catch (caught) {
      if (isAbortError(caught)) return;
      setSaliencyError(caught instanceof Error ? caught.message : "Saliency map failed.");
    } finally {
      setIsSaliencyLoading(false);
      if (saliencyAbortRef.current === controller) {
        saliencyAbortRef.current = null;
      }
    }
  };

  const selectionSummary =
    inputMode === "upload"
      ? `${selectedBatchIds.length} uploaded recording${selectedBatchIds.length === 1 ? "" : "s"} selected`
      : inputMode === "dataset"
        ? `${selectedBatchIds.length} recording${selectedBatchIds.length === 1 ? "" : "s"} selected`
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
          <Button size="sm" variant="outline" onClick={exportBatchCsv} disabled={!batchResult || isExporting}>
            <Download className="h-3.5 w-3.5" />
            {isExporting ? "Exporting…" : "Export CSV"}
          </Button>
          {isProjecting && <p className="text-muted-foreground">Re-projecting embeddings…</p>}
          {projectionError && <p className="text-red-500">{projectionError}</p>}
          {exportError && <p className="text-red-500">{exportError}</p>}
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
          <ClusterSummaryList
            clusterSummaries={batchResult.cluster_summaries}
            clusterColorMap={clusterColorMap}
            focusedClusterId={focusedClusterId}
            onClusterFocusChange={setFocusedClusterId}
          />
        </>
      )}

      {batchResult && selectedFile && hasTargetCluster && (
        <SpeakerSaliencyMap
          title={`Cluster saliency — ${selectedFile.filename}`}
          audioUrl={isTargetIdResolvable ? verificationAudioUrl(selectedFile.file_id) : undefined}
          requireCredentials={true}
          result={saliencyResult}
          isLoading={isSaliencyLoading}
          error={saliencyError}
          staleReason={null}
          emptyStateMessage={saliencyEmptyStateMessage}
          onGenerate={saliencyEmptyStateMessage ? null : runClusterSaliency}
          generateLabel="Generate saliency map"
          segmentCount={saliencySegmentCount}
          onSegmentCountChange={setSaliencySegmentCount}
          clusterBadge={
            targetClusterId ? { label: targetClusterId, color: clusterColorMap[targetClusterId] ?? "#3b82f6" } : null
          }
        />
      )}
    </div>
  );
};
