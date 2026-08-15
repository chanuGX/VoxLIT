import { ChangeEvent, useEffect, useMemo, useState } from "react";
import { AlertCircle, CheckCircle2, FileAudio, ShieldCheck, Users, XCircle } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BatchAnalysisPanel } from "./BatchAnalysisPanel";
import type { WorkbenchCenterProps } from "@/tasks/types";

type SpeakerVerificationWorkbenchProps = WorkbenchCenterProps;

interface ReferenceScore {
  reference_index: number;
  similarity: number;
}

interface VerificationResult {
  model: string;
  model_label: string;
  embedding_dimension: number;
  enrollment_count: number;
  similarity: number;
  threshold: number;
  decision_margin: number;
  same_speaker: boolean;
  enrollment_compactness: number;
  per_reference_scores: ReferenceScore[];
  enrollment_centroid: number[];
  probe_embedding: number[];
  calibration: {
    criterion: string;
    dataset: string;
    threshold_locked_before_test_evaluation: boolean;
  };
}

interface OcclusionSegment {
  segment_index: number;
  start_seconds: number;
  end_seconds: number;
  occluded_similarity: number;
  importance: number;
}

interface OcclusionResult {
  baseline_similarity: number;
  threshold: number;
  segment_count: number;
  interpretation: string;
  segments: OcclusionSegment[];
}

const formatScore = (value: number) => value.toFixed(4);

export const SpeakerVerificationWorkbench = ({
  model,
  modelLabel,
  dataset,
  originalDataset,
  uploadedRawFiles,
  selectedBatchIds,
  pairSelection,
  onReprojectHandlerChange,
  onLabelResolverChange,
}: SpeakerVerificationWorkbenchProps) => {
  const [enrollmentFiles, setEnrollmentFiles] = useState<File[]>([]);
  const [probeFile, setProbeFile] = useState<File | null>(null);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [occlusion, setOcclusion] = useState<OcclusionResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isExplaining, setIsExplaining] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setResult(null);
    setOcclusion(null);
    setError(null);
  }, [model]);

  const canRun = enrollmentFiles.length >= 3 && enrollmentFiles.length <= 5 && !!probeFile && !!model;
  const scoreProgress = useMemo(
    () => (result ? Math.max(0, Math.min(100, ((result.similarity + 1) / 2) * 100)) : 0),
    [result]
  );

  const handleEnrollmentChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    setEnrollmentFiles(files.slice(0, 5));
    setResult(null);
    setOcclusion(null);
    setError(files.length > 5 ? "Only the first five enrolment recordings were selected." : null);
  };

  const handleProbeChange = (event: ChangeEvent<HTMLInputElement>) => {
    setProbeFile(event.target.files?.[0] ?? null);
    setResult(null);
    setOcclusion(null);
    setError(null);
  };

  const runVerification = async () => {
    if (!canRun || !probeFile) return;

    const formData = new FormData();
    formData.append("model", model);
    enrollmentFiles.forEach((file) => formData.append("enrollment_files", file));
    formData.append("probe_file", probeFile);

    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/tasks/verification/verify`, {
        method: "POST",
        credentials: "include",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Verification failed (${response.status})`);
      }
      setResult(payload as VerificationResult);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Speaker verification failed.");
    } finally {
      setIsRunning(false);
    }
  };

  const runTemporalOcclusion = async () => {
    if (!canRun || !probeFile) return;

    const formData = new FormData();
    formData.append("model", model);
    formData.append("segment_count", "8");
    enrollmentFiles.forEach((file) => formData.append("enrollment_files", file));
    formData.append("probe_file", probeFile);

    setIsExplaining(true);
    setOcclusion(null);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/tasks/verification/explain/temporal-occlusion`, {
        method: "POST",
        credentials: "include",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Temporal occlusion failed (${response.status})`);
      }
      setOcclusion(payload as OcclusionResult);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Temporal occlusion failed.");
    } finally {
      setIsExplaining(false);
    }
  };

  return (
    <div className="h-full overflow-y-auto bg-background p-4 scrollbar-thin">
      <div className="mx-auto max-w-4xl space-y-4">
        <div>
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">Explainable Speaker Verification</h2>
            <Badge variant="outline">{modelLabel}</Badge>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            Build a temporary speaker profile from 3–5 recordings, then compare one probe recording with its embedding centroid.
          </p>
        </div>

        <Tabs defaultValue="pair-verification" className="w-full">
          <TabsList>
            <TabsTrigger value="pair-verification">Pair Verification</TabsTrigger>
            <TabsTrigger value="batch-analysis">Batch Analysis</TabsTrigger>
          </TabsList>

          <TabsContent value="pair-verification" className="space-y-4">
        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Users className="h-4 w-4" /> 1. Enrolment recordings
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <input
                type="file"
                accept="audio/wav,audio/mpeg,audio/mp4,audio/flac,.wav,.mp3,.m4a,.flac"
                multiple
                onChange={handleEnrollmentChange}
                className="block w-full text-xs file:mr-3 file:rounded-md file:border-0 file:bg-primary file:px-3 file:py-2 file:text-xs file:text-primary-foreground"
              />
              <p className="text-xs text-muted-foreground">
                {enrollmentFiles.length}/5 selected — at least 3 are required.
              </p>
              <div className="space-y-1">
                {enrollmentFiles.map((file, index) => (
                  <div key={`${file.name}-${file.lastModified}`} className="flex items-center gap-2 text-xs">
                    <FileAudio className="h-3.5 w-3.5 text-muted-foreground" />
                    <span className="truncate">Reference {index + 1}: {file.name}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <FileAudio className="h-4 w-4" /> 2. Probe recording
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <input
                type="file"
                accept="audio/wav,audio/mpeg,audio/mp4,audio/flac,.wav,.mp3,.m4a,.flac"
                onChange={handleProbeChange}
                className="block w-full text-xs file:mr-3 file:rounded-md file:border-0 file:bg-primary file:px-3 file:py-2 file:text-xs file:text-primary-foreground"
              />
              <p className="truncate text-xs text-muted-foreground">
                {probeFile ? probeFile.name : "Select the recording whose identity should be verified."}
              </p>
              <Button className="w-full" disabled={!canRun || isRunning} onClick={runVerification}>
                {isRunning ? "Extracting embeddings…" : "Verify speaker"}
              </Button>
            </CardContent>
          </Card>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Verification could not be completed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {result && (
          <div className="space-y-4">
            <Alert className={result.same_speaker ? "border-emerald-300 bg-emerald-50" : "border-rose-300 bg-rose-50"}>
              {result.same_speaker ? (
                <CheckCircle2 className="h-4 w-4 text-emerald-700" />
              ) : (
                <XCircle className="h-4 w-4 text-rose-700" />
              )}
              <AlertTitle>{result.same_speaker ? "Same speaker" : "Different speakers"}</AlertTitle>
              <AlertDescription>
                Cosine similarity {formatScore(result.similarity)} is {result.same_speaker ? "above" : "below"} the calibrated threshold {formatScore(result.threshold)}.
              </AlertDescription>
            </Alert>

            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <CardHeader><CardTitle className="text-xs">Similarity score</CardTitle></CardHeader>
                <CardContent>
                  <div className="mb-2 text-2xl font-semibold">{formatScore(result.similarity)}</div>
                  <Progress value={scoreProgress} />
                </CardContent>
              </Card>
              <Card>
                <CardHeader><CardTitle className="text-xs">Decision margin</CardTitle></CardHeader>
                <CardContent className="text-2xl font-semibold">
                  {result.decision_margin >= 0 ? "+" : ""}{formatScore(result.decision_margin)}
                </CardContent>
              </Card>
              <Card>
                <CardHeader><CardTitle className="text-xs">Cluster compactness</CardTitle></CardHeader>
                <CardContent className="text-2xl font-semibold">
                  {formatScore(result.enrollment_compactness)}
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Pair comparison</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {result.per_reference_scores.map((item) => (
                  <div key={item.reference_index} className="flex items-center justify-between text-sm">
                    <span>Reference {item.reference_index} ↔ probe</span>
                    <Badge variant="secondary">{formatScore(item.similarity)}</Badge>
                  </div>
                ))}
                <div className="border-t pt-2 text-xs text-muted-foreground">
                  {result.embedding_dimension}-dimensional embeddings · EER-calibrated on the VoxCeleb1 Indian verification subset · threshold locked before test evaluation
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Temporal occlusion saliency</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xs text-muted-foreground">
                  Mute eight probe segments one at a time and measure the change in cluster similarity.
                </p>
                <Button variant="outline" onClick={runTemporalOcclusion} disabled={isExplaining}>
                  {isExplaining ? "Running 8 occlusion passes…" : "Run temporal occlusion"}
                </Button>
                {occlusion && (
                  <div className="space-y-2 border-t pt-3">
                    {occlusion.segments.map((segment) => {
                      const maxImportance = Math.max(
                        ...occlusion.segments.map((item) => Math.abs(item.importance)),
                        0.0001
                      );
                      const width = Math.max(2, (Math.abs(segment.importance) / maxImportance) * 100);
                      return (
                        <div key={segment.segment_index} className="grid grid-cols-[7rem_1fr_4.5rem] items-center gap-2 text-xs">
                          <span>{segment.start_seconds.toFixed(2)}–{segment.end_seconds.toFixed(2)}s</span>
                          <div className="h-3 rounded bg-muted">
                            <div
                              className={`h-3 rounded ${segment.importance >= 0 ? "bg-emerald-500" : "bg-rose-500"}`}
                              style={{ width: `${width}%` }}
                            />
                          </div>
                          <span className="text-right font-mono">
                            {segment.importance >= 0 ? "+" : ""}{segment.importance.toFixed(4)}
                          </span>
                        </div>
                      );
                    })}
                    <p className="pt-1 text-xs text-muted-foreground">{occlusion.interpretation}</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}
          </TabsContent>

          <TabsContent value="batch-analysis" forceMount className="space-y-4 data-[state=inactive]:hidden">
            <BatchAnalysisPanel
              model={model}
              modelLabel={modelLabel}
              dataset={dataset}
              originalDataset={originalDataset}
              uploadedRawFiles={uploadedRawFiles}
              selectedBatchIds={selectedBatchIds}
              pairSelection={pairSelection}
              onReprojectHandlerChange={onReprojectHandlerChange}
              onLabelResolverChange={onLabelResolverChange}
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};