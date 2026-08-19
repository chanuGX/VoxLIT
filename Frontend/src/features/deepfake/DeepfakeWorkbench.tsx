import { useEffect, useState } from "react";
import { AlertCircle, Loader2, Play, ShieldQuestion } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EvaluationPanel } from "./EvaluationPanel";
import { SaliencyPanel } from "./SaliencyPanel";
import { SilenceProbeCard } from "./SilenceProbeCard";
import { ScoreBar } from "./ScoreBar";
import { DeepfakeResult, RecordingInfo } from "./types";

interface DeepfakeWorkbenchProps {
  model: string;
  modelLabel: string;
}

export const DeepfakeWorkbench = ({ model, modelLabel }: DeepfakeWorkbenchProps) => {
  const [recordings, setRecordings] = useState<RecordingInfo[]>([]);
  const [selectedRecordingId, setSelectedRecordingId] = useState<string>("");
  const [result, setResult] = useState<DeepfakeResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadRecordings = async () => {
      try {
        const response = await fetch(`${API_BASE}/tasks/deepfake/dataset/recordings`, {
          credentials: "include",
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Could not list recordings.");
        setRecordings(payload.recordings as RecordingInfo[]);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Could not list recordings.");
      }
    };
    loadRecordings();
  }, []);

  const runDetection = async () => {
    if (!selectedRecordingId || !model) return;
    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/tasks/deepfake/run`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, recording_id: selectedRecordingId }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Detection failed (${response.status})`);
      }
      setResult(payload as DeepfakeResult);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Detection failed.");
    } finally {
      setIsRunning(false);
    }
  };

  const audioUrl = selectedRecordingId
    ? `${API_BASE}/tasks/deepfake/dataset/recordings/${selectedRecordingId}/audio`
    : undefined;

  return (
    <div className="h-full overflow-y-auto bg-background p-4 scrollbar-thin">
      <div className="mx-auto max-w-4xl space-y-4">
        <div>
          <div className="flex items-center gap-2">
            <ShieldQuestion className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">Audio Deepfake Detection</h2>
            <Badge variant="outline">{modelLabel}</Badge>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            Is this speech a real recording or a synthesizer's output? The detector
            returns a score, not a category — the threshold is what turns it into a
            decision.
          </p>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Something went wrong</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">1. Pick a recording</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <select
                className="h-9 rounded-md border bg-background px-2 text-sm"
                value={selectedRecordingId}
                onChange={(event) => {
                  setSelectedRecordingId(event.target.value);
                  setResult(null);
                  setError(null);
                }}
              >
                <option value="">Select a recording…</option>
                {recordings.map((recording) => (
                  <option key={recording.recording_id} value={recording.recording_id}>
                    {recording.display_filename}
                  </option>
                ))}
              </select>
              <Button onClick={runDetection} disabled={!selectedRecordingId || isRunning}>
                {isRunning ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Scoring… (the first
                    run downloads the model, which takes a few minutes)
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" /> Run detection
                  </>
                )}
              </Button>
            </div>
            {audioUrl && <audio controls className="w-full" src={audioUrl} />}
            <p className="text-xs text-muted-foreground">
              The dataset's bona fide/spoof labels are deliberately not shown — listen
              and read the score first, then check the protocol file if you want the
              answer.
            </p>
          </CardContent>
        </Card>

        {result && (
          <Card>
            <CardHeader>
              <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
                2. Score
                <Badge variant={result.decision === "spoof" ? "destructive" : "secondary"}>
                  {result.decision === "spoof" ? "SPOOF" : "BONA FIDE"}
                </Badge>
                {result.cached && (
                  <span className="font-normal text-muted-foreground">(cached)</span>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <ScoreBar
                spoofProbability={result.spoof_probability}
                threshold={result.threshold}
                calibrated={result.threshold_calibrated}
              />

              {!result.threshold_calibrated && (
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>This threshold is not calibrated</AlertTitle>
                  <AlertDescription>
                    {result.threshold.toFixed(2)} is the naive midpoint, not an
                    equal-error-rate operating point. The decision above is only as
                    meaningful as that cut — a calibrated threshold comes from the score
                    distribution and DET curve view.
                  </AlertDescription>
                </Alert>
              )}

              {result.truncated && (
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Clip truncated</AlertTitle>
                  <AlertDescription>
                    Scored the first {result.analysed_seconds}s of a {result.duration}s
                    clip.
                  </AlertDescription>
                </Alert>
              )}

              <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-4">
                <div>
                  <dt className="text-muted-foreground">bona fide score</dt>
                  <dd className="font-mono tabular-nums">
                    {result.bonafide_probability.toFixed(3)}
                  </dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Duration</dt>
                  <dd className="font-mono tabular-nums">{result.duration}s</dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Logits</dt>
                  <dd className="font-mono text-xs tabular-nums">
                    [{result.logits.map((v) => v.toFixed(2)).join(", ")}]
                  </dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Spoof class</dt>
                  <dd className="font-mono text-xs">
                    {result.spoof_index} · {result.id2label[String(result.spoof_index)]}
                  </dd>
                </div>
              </dl>

              <p className="text-xs text-muted-foreground">
                {result.model_id} — the spoof class is read from the checkpoint's own
                label map, not assumed.
              </p>
            </CardContent>
          </Card>
        )}

        {/* Feature 2 — per-clip ablation (SRS DF-10..DF-12). Placed before
            Feature 1 because it is the view that can invalidate it. */}
        {selectedRecordingId && (
          <SilenceProbeCard
            model={model}
            modelLabel={modelLabel}
            recordingId={selectedRecordingId}
          />
        )}

        {/* Feature 3 — waveform-aligned attribution (SRS DF-14, DF-15). */}
        {selectedRecordingId && (
          <SaliencyPanel
            model={model}
            modelLabel={modelLabel}
            recordingId={selectedRecordingId}
          />
        )}

        {/* Feature 1 — the task's batch-analysis view (SRS DF-6..DF-9). */}
        <EvaluationPanel
          model={model}
          modelLabel={modelLabel}
          datasetAvailable={recordings.length > 0}
        />
      </div>
    </div>
  );
};
