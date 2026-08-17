import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, FileAudio, Users, XCircle } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BatchAnalysisPanel } from "./BatchAnalysisPanel";
import { SpeakerSaliencyMap } from "./SpeakerSaliencyMap";
import { verificationAudioUrl } from "./audioUrl";
import type { SaliencyMapResponse } from "./saliencyTypes";
import { PerturbationTools, type VerificationPerturbationContext } from "@/components/analysis/PerturbationTools";
import type { LocalFilePreview, WorkbenchCenterProps } from "@/tasks/types";

const DEFAULT_SALIENCY_SEGMENT_COUNT = 8;

type VerificationKind = "upload" | "id";

interface RequestKey {
  kind: VerificationKind;
  model: string;
  enrollmentKey: string;
  probeId: string;
}

function keysEqual(a: RequestKey | null, b: RequestKey | null): boolean {
  if (a === null || a === undefined || b === null || b === undefined) return false;
  return a.kind === b.kind && a.model === b.model && a.enrollmentKey === b.enrollmentKey && a.probeId === b.probeId;
}

/** Stable client-only signature for a browser File, used to detect
 *  duplicates/overlaps without touching file content. */
function fileSignature(file: File): string {
  return `${file.name}|${file.size}|${file.type}|${file.lastModified}`;
}

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

const formatScore = (value: number) => value.toFixed(4);

export const SpeakerVerificationWorkbench = ({
  model,
  modelLabel,
  dataset,
  originalDataset,
  uploadedRawFiles,
  selectedFile,
  selectedBatchIds,
  pairSelection,
  datasetRecordings,
  onReprojectHandlerChange,
  onLabelResolverChange,
  onVerificationAssetCreated,
  localPreview,
  onLocalFileSelect,
}: SpeakerVerificationWorkbenchProps) => {
  const [enrollmentRefs, setEnrollmentRefs] = useState<LocalFilePreview[]>([]);
  const [probeRef, setProbeRef] = useState<LocalFilePreview | null>(null);
  const [result, setResult] = useState<VerificationResult | null>(null);
  // Which flow produced `result` -- distinguishes the device-upload flow
  // from the id-mode ("workspace selections") flow, so the two never both
  // render a result/saliency section at once.
  const [resultMode, setResultMode] = useState<VerificationKind | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Saliency -- shared between both modes (only one mode's card is ever
  // interactable/visible at a time, mirroring how `result`/`resultMode`
  // are already shared).
  const [saliencyResult, setSaliencyResult] = useState<SaliencyMapResponse | null>(null);
  const [saliencyError, setSaliencyError] = useState<string | null>(null);
  const [isSaliencyLoading, setIsSaliencyLoading] = useState(false);
  const [saliencySegmentCount, setSaliencySegmentCount] = useState(DEFAULT_SALIENCY_SEGMENT_COUNT);
  const [resultGeneratedFor, setResultGeneratedFor] = useState<RequestKey | null>(null);
  const [saliencyGeneratedFor, setSaliencyGeneratedFor] = useState<(RequestKey & { segmentCount: number }) | null>(
    null
  );

  // Per-request abort controllers + the identity key captured when each
  // request started. Verify (device/id) and saliency (shared) are each
  // independently cancellable so a stale in-flight request from one never
  // clobbers a fresher result from another.
  const deviceVerifyAbortRef = useRef<AbortController | null>(null);
  const deviceVerifyKeyRef = useRef<RequestKey | null>(null);
  const idVerifyAbortRef = useRef<AbortController | null>(null);
  const idVerifyKeyRef = useRef<RequestKey | null>(null);
  const saliencyAbortRef = useRef<AbortController | null>(null);
  const saliencyKeyRef = useRef<(RequestKey & { segmentCount: number }) | null>(null);

  // Resolve an opaque recording_id (rec_... or asset_...) to a display label
  // via the merged demo+session-asset list, falling back to the id itself.
  const resolveRecordingLabel = (id: string): string =>
    datasetRecordings?.find((r) => r.recording_id === id)?.display_filename ?? id;

  // ID-based ("workspace selections") Pair Verification.
  const idModeEnrollmentIds = selectedBatchIds;
  const idModeProbeId = selectedFile?.file_id ?? null;
  const idModeProbeIsEnrollment = !!idModeProbeId && idModeEnrollmentIds.includes(idModeProbeId);
  const idModeValidationMessage: string | null =
    idModeEnrollmentIds.length < 3 || idModeEnrollmentIds.length > 5
      ? "Check 3-5 rows in the Audio Dataset table as enrollment recordings."
      : !idModeProbeId
        ? "Select a probe recording (click a row or a graph point) that is not part of the enrollment set."
        : idModeProbeIsEnrollment
          ? "The probe recording must not also be a checked enrollment recording."
          : null;
  const canRunIdMode = !idModeValidationMessage && !!model;
  const idModeEnrollmentKey = [...idModeEnrollmentIds].sort().join(",");
  const currentIdKey: RequestKey | null =
    idModeProbeId !== null &&
    idModeProbeId !== undefined &&
    idModeEnrollmentIds.length >= 3 &&
    idModeEnrollmentIds.length <= 5
      ? { kind: "id", model, enrollmentKey: idModeEnrollmentKey, probeId: idModeProbeId }
      : null;

  // Device-upload Pair Verification.
  const canRunDevice = enrollmentRefs.length >= 3 && enrollmentRefs.length <= 5 && !!probeRef && !!model;
  const deviceEnrollmentKey = [...enrollmentRefs.map((r) => r.localId)].sort().join(",");
  const currentDeviceKey: RequestKey | null =
    probeRef && enrollmentRefs.length >= 3 && enrollmentRefs.length <= 5
      ? { kind: "upload", model, enrollmentKey: deviceEnrollmentKey, probeId: probeRef.localId }
      : null;

  // "Latest" refs for the two identities -- mutated directly during render
  // (not inside a useEffect) so async response handlers, which otherwise
  // only see the key captured in their own closure at call time, can read
  // the truly current identity at apply-time. Safe because these are only
  // ever read later from callbacks, never read back during this render.
  const currentDeviceKeyRef = useRef<RequestKey | null>(null);
  currentDeviceKeyRef.current = currentDeviceKey;
  const currentIdKeyRef = useRef<RequestKey | null>(null);
  currentIdKeyRef.current = currentIdKey;

  const isResultCurrent =
    result !== null &&
    resultMode !== null &&
    keysEqual(resultGeneratedFor, resultMode === "id" ? currentIdKey : currentDeviceKey);
  const isSaliencyStale =
    saliencyResult !== null &&
    resultMode !== null &&
    (!keysEqual(saliencyGeneratedFor, resultMode === "id" ? currentIdKey : currentDeviceKey) ||
      saliencyGeneratedFor?.segmentCount !== saliencySegmentCount);

  // Actively cancel an in-flight request the moment its captured identity
  // stops matching the live identity -- not just when the next request
  // starts. Segment count is deliberately excluded here: a slider change
  // alone must not cancel an in-flight saliency request (its result should
  // still land and be marked stale by segment count, matching the existing
  // "previous map becomes stale when the slider changes" behavior) -- only
  // a real audio/selection change (model, enrollment set, or probe) should
  // abort it outright.
  useEffect(() => {
    if (deviceVerifyAbortRef.current && !keysEqual(deviceVerifyKeyRef.current, currentDeviceKey)) {
      deviceVerifyAbortRef.current.abort();
    }
    const sk = saliencyKeyRef.current;
    if (sk?.kind === "upload" && saliencyAbortRef.current && !keysEqual(sk, currentDeviceKey)) {
      saliencyAbortRef.current.abort();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentDeviceKey?.model, currentDeviceKey?.enrollmentKey, currentDeviceKey?.probeId]);

  useEffect(() => {
    if (idVerifyAbortRef.current && !keysEqual(idVerifyKeyRef.current, currentIdKey)) {
      idVerifyAbortRef.current.abort();
    }
    const sk = saliencyKeyRef.current;
    if (sk?.kind === "id" && saliencyAbortRef.current && !keysEqual(sk, currentIdKey)) {
      saliencyAbortRef.current.abort();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentIdKey?.model, currentIdKey?.enrollmentKey, currentIdKey?.probeId]);

  const runVerificationById = async () => {
    if (!canRunIdMode || !idModeProbeId || !currentIdKey) return;

    const requestKey = currentIdKey;
    if (idVerifyAbortRef.current) {
      idVerifyAbortRef.current.abort();
    }
    const controller = new AbortController();
    idVerifyAbortRef.current = controller;
    idVerifyKeyRef.current = requestKey;

    const formData = new FormData();
    formData.append("model", model);
    idModeEnrollmentIds.forEach((id) => formData.append("enrollment_recording_ids", id));
    formData.append("probe_recording_id", idModeProbeId);

    setIsRunning(true);
    setError(null);
    setResult(null);
    setResultMode(null);
    setResultGeneratedFor(null);

    try {
      const response = await fetch(`${API_BASE}/tasks/verification/verify`, {
        method: "POST",
        credentials: "include",
        body: formData,
        signal: controller.signal,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Verification failed (${response.status})`);
      }
      if (!keysEqual(requestKey, currentIdKeyRef.current)) return; // stale -- selection moved on
      setResult(payload as VerificationResult);
      setResultMode("id");
      setResultGeneratedFor(requestKey);
      // A saliency map generated for the other mode can never be valid here
      // -- never pair it with this mode's (different) audio.
      if (saliencyGeneratedFor && saliencyGeneratedFor.kind !== "id") {
        setSaliencyResult(null);
        setSaliencyError(null);
        setSaliencyGeneratedFor(null);
        saliencyKeyRef.current = null;
      }
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setError(caught instanceof Error ? caught.message : "Speaker verification failed.");
    } finally {
      setIsRunning(false);
      if (idVerifyAbortRef.current === controller) {
        idVerifyAbortRef.current = null;
      }
    }
  };

  const runEnrollmentSaliency = async () => {
    if (!isResultCurrent || resultMode !== "id" || !idModeProbeId || !currentIdKey) return;
    if (saliencyAbortRef.current) {
      saliencyAbortRef.current.abort();
    }
    const controller = new AbortController();
    saliencyAbortRef.current = controller;
    const requestKey = { ...currentIdKey, segmentCount: saliencySegmentCount };
    saliencyKeyRef.current = requestKey;

    const formData = new FormData();
    formData.append("model", model);
    formData.append("reference_type", "enrollment");
    idModeEnrollmentIds.forEach((id) => formData.append("enrollment_recording_ids", id));
    formData.append("probe_recording_id", idModeProbeId);
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
      // Only the underlying audio/selection identity gates whether this
      // response still applies -- a segment-count-only mismatch against the
      // live slider is expected and left to isSaliencyStale to display as
      // stale, not discarded.
      if (!keysEqual(requestKey, currentIdKeyRef.current)) return;
      setSaliencyResult(payload as SaliencyMapResponse);
      setSaliencyGeneratedFor(requestKey);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setSaliencyError(caught instanceof Error ? caught.message : "Saliency map failed.");
    } finally {
      setIsSaliencyLoading(false);
      if (saliencyAbortRef.current === controller) {
        saliencyAbortRef.current = null;
      }
    }
  };

  // Perturbation tab context — single currently selected recording, driven
  // by the same shared selectedFile the rest of the app uses.
  const verificationPerturbationContext: VerificationPerturbationContext = {
    model,
    selectedRecordingId: selectedFile?.file_id ?? null,
    selectedRecordingLabel: selectedFile?.file_id ? resolveRecordingLabel(selectedFile.file_id) : null,
    onPerturbationApplied: (result) => onVerificationAssetCreated(result.session_asset),
  };

  useEffect(() => {
    setResult(null);
    setResultMode(null);
    setResultGeneratedFor(null);
    setError(null);
    setSaliencyResult(null);
    setSaliencyError(null);
    setSaliencyGeneratedFor(null);
    saliencyKeyRef.current = null;
  }, [model]);

  // Revoke each accepted file's preview URL when it's replaced or the
  // component unmounts. Cleanup runs after React commits the *new* state,
  // and any file still shown in the Datapoint Editor has already had its
  // localPreview cleared by the change handlers below by that point, so a
  // URL is never revoked while still on screen.
  useEffect(() => {
    return () => {
      enrollmentRefs.forEach((ref) => URL.revokeObjectURL(ref.previewUrl));
    };
  }, [enrollmentRefs]);

  useEffect(() => {
    return () => {
      if (probeRef) URL.revokeObjectURL(probeRef.previewUrl);
    };
  }, [probeRef]);

  const scoreProgress = useMemo(
    () => (result ? Math.max(0, Math.min(100, ((result.similarity + 1) / 2) * 100)) : 0),
    [result]
  );

  const handleEnrollmentChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    const messages: string[] = [];

    // Validate/dedupe on the raw FileList *before* creating any object
    // URLs, so a rejected or truncated file never gets a URL created for it
    // -- nothing to revoke after the fact.
    const seen = new Set<string>();
    const deduped: File[] = [];
    let duplicateCount = 0;
    for (const file of files) {
      const signature = fileSignature(file);
      if (seen.has(signature)) {
        duplicateCount += 1;
        continue;
      }
      seen.add(signature);
      deduped.push(file);
    }
    if (duplicateCount > 0) {
      messages.push(`${duplicateCount} duplicate recording${duplicateCount > 1 ? "s" : ""} ignored.`);
    }

    let withoutProbeOverlap = deduped;
    if (probeRef) {
      const probeSignature = fileSignature(probeRef.file);
      const beforeCount = withoutProbeOverlap.length;
      withoutProbeOverlap = withoutProbeOverlap.filter((file) => fileSignature(file) !== probeSignature);
      if (withoutProbeOverlap.length < beforeCount) {
        messages.push("The probe recording cannot also be used as an enrollment reference.");
      }
    }

    let accepted = withoutProbeOverlap;
    if (accepted.length > 5) {
      accepted = accepted.slice(0, 5);
      messages.push("Only the first five enrolment recordings were selected.");
    }

    const newRefs: LocalFilePreview[] = accepted.map((file, index) => ({
      localId: crypto.randomUUID(),
      file,
      previewUrl: URL.createObjectURL(file),
      role: "enrollment" as const,
      label: `Reference ${index + 1}`,
    }));

    // The old enrollment set is always wholesale-replaced by a new pick --
    // if the active Datapoint Editor preview was one of those old files,
    // clear it before the new state (and the revoke-on-replace effect)
    // commits.
    if (localPreview?.role === "enrollment") {
      onLocalFileSelect(null);
    }
    setEnrollmentRefs(newRefs);

    // Clear invalid results/saliency immediately, scoped to device-mode
    // state only -- never touches a currently-displayed workspace-selection
    // result, so switching tabs never loses that flow's freshness.
    if (resultMode === "upload") {
      setResult(null);
      setResultMode(null);
      setResultGeneratedFor(null);
    }
    if (saliencyGeneratedFor?.kind === "upload") {
      setSaliencyResult(null);
      setSaliencyError(null);
      setSaliencyGeneratedFor(null);
      saliencyKeyRef.current = null;
    }

    setError(messages.length > 0 ? messages.join(" ") : null);
  };

  const handleProbeChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;

    if (file) {
      const signature = fileSignature(file);
      const conflict = enrollmentRefs.some((ref) => fileSignature(ref.file) === signature);
      if (conflict) {
        // Reject -- keep the existing probe selection (and everything else)
        // completely untouched, only surface the message.
        setError("This recording is already selected as an enrollment reference.");
        return;
      }
    }

    if (localPreview?.role === "probe") {
      onLocalFileSelect(null);
    }
    setProbeRef(
      file
        ? {
            localId: crypto.randomUUID(),
            file,
            previewUrl: URL.createObjectURL(file),
            role: "probe" as const,
            label: "Probe",
          }
        : null
    );

    if (resultMode === "upload") {
      setResult(null);
      setResultMode(null);
      setResultGeneratedFor(null);
    }
    if (saliencyGeneratedFor?.kind === "upload") {
      setSaliencyResult(null);
      setSaliencyError(null);
      setSaliencyGeneratedFor(null);
      saliencyKeyRef.current = null;
    }
    setError(null);
  };

  const runVerification = async () => {
    if (!canRunDevice || !probeRef || !currentDeviceKey) return;

    const requestKey = currentDeviceKey;
    if (deviceVerifyAbortRef.current) {
      deviceVerifyAbortRef.current.abort();
    }
    const controller = new AbortController();
    deviceVerifyAbortRef.current = controller;
    deviceVerifyKeyRef.current = requestKey;

    const formData = new FormData();
    formData.append("model", model);
    enrollmentRefs.forEach((ref) => formData.append("enrollment_files", ref.file));
    formData.append("probe_file", probeRef.file);

    setIsRunning(true);
    setError(null);
    setResult(null);
    setResultMode(null);
    setResultGeneratedFor(null);

    try {
      const response = await fetch(`${API_BASE}/tasks/verification/verify`, {
        method: "POST",
        credentials: "include",
        body: formData,
        signal: controller.signal,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Verification failed (${response.status})`);
      }
      if (!keysEqual(requestKey, currentDeviceKeyRef.current)) return; // stale -- selection moved on
      setResult(payload as VerificationResult);
      setResultMode("upload");
      setResultGeneratedFor(requestKey);
      if (saliencyGeneratedFor && saliencyGeneratedFor.kind !== "upload") {
        setSaliencyResult(null);
        setSaliencyError(null);
        setSaliencyGeneratedFor(null);
        saliencyKeyRef.current = null;
      }
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setError(caught instanceof Error ? caught.message : "Speaker verification failed.");
    } finally {
      setIsRunning(false);
      if (deviceVerifyAbortRef.current === controller) {
        deviceVerifyAbortRef.current = null;
      }
    }
  };

  const runUploadSaliency = async () => {
    if (!isResultCurrent || resultMode !== "upload" || !probeRef || !currentDeviceKey) return;
    if (saliencyAbortRef.current) {
      saliencyAbortRef.current.abort();
    }
    const controller = new AbortController();
    saliencyAbortRef.current = controller;
    const requestKey = { ...currentDeviceKey, segmentCount: saliencySegmentCount };
    saliencyKeyRef.current = requestKey;

    const formData = new FormData();
    formData.append("model", model);
    formData.append("reference_type", "enrollment");
    enrollmentRefs.forEach((ref) => formData.append("enrollment_files", ref.file));
    formData.append("probe_file", probeRef.file);
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
      if (!keysEqual(requestKey, currentDeviceKeyRef.current)) return;
      setSaliencyResult(payload as SaliencyMapResponse);
      setSaliencyGeneratedFor(requestKey);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setSaliencyError(caught instanceof Error ? caught.message : "Saliency map failed.");
    } finally {
      setIsSaliencyLoading(false);
      if (saliencyAbortRef.current === controller) {
        saliencyAbortRef.current = null;
      }
    }
  };

  return (
    <Tabs defaultValue="pair-verification" className="h-full flex flex-col">
      <div className="flex-shrink-0 bg-panel-header border-b border-border px-3 py-2">
        <TabsList className="h-7 grid w-full grid-cols-3 bg-muted">
          <TabsTrigger value="pair-verification" className="text-xs">Pair Verification</TabsTrigger>
          <TabsTrigger value="batch-analysis" className="text-xs">Batch Analysis</TabsTrigger>
          <TabsTrigger value="perturbation" className="text-xs">Perturbation</TabsTrigger>
        </TabsList>
      </div>

      <div className="flex-1 overflow-y-auto bg-background p-4 scrollbar-thin">
        <div className="mx-auto max-w-4xl space-y-4">
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
                {enrollmentRefs.length}/5 selected — at least 3 are required.
              </p>
              <div className="space-y-1">
                {enrollmentRefs.map((ref) => (
                  <button
                    key={ref.localId}
                    type="button"
                    onClick={() => onLocalFileSelect(ref)}
                    className={`flex w-full items-center gap-2 rounded px-1.5 py-1 text-left text-xs transition-colors ${
                      localPreview?.localId === ref.localId ? "bg-primary/10 text-primary" : "hover:bg-muted"
                    }`}
                  >
                    <FileAudio className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                    <span className="truncate">{ref.label}: {ref.file.name}</span>
                  </button>
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
              {probeRef ? (
                <button
                  type="button"
                  onClick={() => onLocalFileSelect(probeRef)}
                  className={`flex w-full items-center gap-2 rounded px-1.5 py-1 text-left text-xs transition-colors ${
                    localPreview?.localId === probeRef.localId ? "bg-primary/10 text-primary" : "hover:bg-muted"
                  }`}
                >
                  <FileAudio className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                  <span className="truncate">Probe: {probeRef.file.name}</span>
                </button>
              ) : (
                <p className="truncate text-xs text-muted-foreground">
                  Select the recording whose identity should be verified.
                </p>
              )}
              <Button className="w-full" disabled={!canRunDevice || isRunning} onClick={runVerification}>
                {isRunning ? "Extracting embeddings…" : "Verify speaker"}
              </Button>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm">
              <Users className="h-4 w-4" /> Or verify using workspace selections
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="text-xs">
              <span className="font-medium">
                Enrollment ({idModeEnrollmentIds.length}/5 checked, need 3-5):
              </span>
              {idModeEnrollmentIds.length > 0 ? (
                <ul className="list-disc pl-4 mt-1 text-muted-foreground">
                  {idModeEnrollmentIds.map((id) => (
                    <li key={id} className="truncate">{resolveRecordingLabel(id)}</li>
                  ))}
                </ul>
              ) : (
                <span className="text-muted-foreground"> check rows in the Audio Dataset table below</span>
              )}
            </div>
            <div className="text-xs">
              <span className="font-medium">Probe: </span>
              <span className={idModeProbeIsEnrollment ? "text-destructive" : "text-muted-foreground"}>
                {idModeProbeId
                  ? resolveRecordingLabel(idModeProbeId)
                  : "none selected — click a row in the Audio Dataset table or a graph point."}
              </span>
            </div>
            <div className="flex gap-2">
              <Button className="flex-1" disabled={!canRunIdMode || isRunning} onClick={runVerificationById}>
                {isRunning ? "Extracting embeddings…" : "Verify speaker (selections)"}
              </Button>
            </div>
            {idModeValidationMessage && (
              <p className="text-xs text-muted-foreground">{idModeValidationMessage}</p>
            )}
          </CardContent>
        </Card>

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

            {(resultMode === "upload" || resultMode === "id") && (
              <SpeakerSaliencyMap
                title="Temporal occlusion saliency"
                audioUrl={
                  resultMode === "id"
                    ? (idModeProbeId ? verificationAudioUrl(idModeProbeId) : undefined)
                    : probeRef?.previewUrl
                }
                requireCredentials={resultMode === "id"}
                result={saliencyResult}
                isLoading={isSaliencyLoading}
                error={saliencyError}
                staleReason={
                  !isResultCurrent
                    ? "Selections have changed since this verification result was produced — run verification again to enable a saliency map."
                    : isSaliencyStale
                      ? (resultMode === "id"
                          ? "Model, enrollment, or probe selection changed since this map was generated."
                          : "Segment count changed since this map was generated.")
                      : null
                }
                emptyStateMessage={null}
                onGenerate={isResultCurrent ? (resultMode === "id" ? runEnrollmentSaliency : runUploadSaliency) : null}
                generateLabel="Generate saliency map"
                segmentCount={saliencySegmentCount}
                onSegmentCountChange={setSaliencySegmentCount}
              />
            )}
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
              selectedFile={selectedFile}
              onReprojectHandlerChange={onReprojectHandlerChange}
              onLabelResolverChange={onLabelResolverChange}
            />
          </TabsContent>

          <TabsContent value="perturbation" forceMount className="space-y-4 data-[state=inactive]:hidden">
            <PerturbationTools selectedFile={null} verification={verificationPerturbationContext} />
          </TabsContent>
        </div>
      </div>
    </Tabs>
  );
};
