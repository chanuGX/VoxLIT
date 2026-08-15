import { useEffect, useRef, useState } from "react";
import { AudioLines, Loader2, Play } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DiarizationTimeline } from "./DiarizationTimeline";
import { EmbeddingScatter } from "./EmbeddingScatter";
import {
  DiarizationResult,
  ProjectionResult,
  RecordingInfo,
} from "./types";

interface DiarizationWorkbenchProps {
  model: string;
  modelLabel: string;
}

export const DiarizationWorkbench = ({ model, modelLabel }: DiarizationWorkbenchProps) => {
  const [recordings, setRecordings] = useState<RecordingInfo[]>([]);
  const [selectedRecordingId, setSelectedRecordingId] = useState<string>("");
  const [result, setResult] = useState<DiarizationResult | null>(null);
  const [projection, setProjection] = useState<ProjectionResult | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    const loadRecordings = async () => {
      try {
        const response = await fetch(`${API_BASE}/tasks/task-b/dataset/recordings`, {
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

  const runDiarization = async () => {
    if (!selectedRecordingId || !model) return;
    setIsRunning(true);
    setError(null);
    setResult(null);
    setProjection(null);
    setSelectedId(null);

    try {
      const runResponse = await fetch(`${API_BASE}/tasks/task-b/run`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, recording_id: selectedRecordingId }),
      });
      const runPayload = await runResponse.json();
      if (!runResponse.ok) {
        throw new Error(runPayload.detail || `Diarization failed (${runResponse.status})`);
      }
      setResult(runPayload as DiarizationResult);

      const projectionResponse = await fetch(
        `${API_BASE}/tasks/task-b/projection?model=${encodeURIComponent(
          model
        )}&recording_id=${encodeURIComponent(selectedRecordingId)}`,
        { credentials: "include" }
      );
      const projectionPayload = await projectionResponse.json();
      if (projectionResponse.ok) {
        setProjection(projectionPayload as ProjectionResult);
      }
      // A projection error (e.g. too few segments) should not hide the timeline.
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Diarization failed.");
    } finally {
      setIsRunning(false);
    }
  };

  const seekToSegment = (segmentId: string) => {
    setSelectedId(segmentId);
    const segment = result?.segments.find((s) => s.id === segmentId);
    const audio = audioRef.current;
    if (segment && audio) {
      audio.currentTime = segment.start;
      audio.play().catch(() => undefined);
    }
  };

  const audioUrl = selectedRecordingId
    ? `${API_BASE}/tasks/task-b/dataset/recordings/${selectedRecordingId}/audio`
    : undefined;

  return (
    <div className="h-full overflow-y-auto bg-background p-4 scrollbar-thin">
      <div className="mx-auto max-w-4xl space-y-4">
        <div>
          <div className="flex items-center gap-2">
            <AudioLines className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">Glass-Box Speaker Diarization</h2>
            <Badge variant="outline">{modelLabel}</Badge>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            Who spoke when — with the pipeline's own internals exposed: per-segment
            confidence from the clustering's embedding space, hatched where uncertain.
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
                  setProjection(null);
                  setSelectedId(null);
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
              <Button onClick={runDiarization} disabled={!selectedRecordingId || isRunning}>
                {isRunning ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Diarizing… (first run
                    on a full meeting takes minutes)
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" /> Run diarization
                  </>
                )}
              </Button>
            </div>
            {audioUrl && <audio ref={audioRef} controls className="w-full" src={audioUrl} />}
          </CardContent>
        </Card>

        {result && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">
                2. Speaker timeline{" "}
                <span className="font-normal text-muted-foreground">
                  — {result.num_speakers} speakers, {result.segments.length} segments
                  {result.cached ? " (cached)" : ""}
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <DiarizationTimeline
                segments={result.segments}
                speakers={result.speakers}
                duration={result.duration}
                hoveredId={hoveredId}
                selectedId={selectedId}
                onHover={setHoveredId}
                onSelect={seekToSegment}
              />
            </CardContent>
          </Card>
        )}

        {result && projection && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">
                3. Segment embeddings (PCA){" "}
                <span className="font-normal text-muted-foreground">
                  — each dot is one segment in the space the clustering used
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <EmbeddingScatter
                points={projection.points}
                speakers={result.speakers}
                hoveredId={hoveredId}
                selectedId={selectedId}
                onHover={setHoveredId}
                onSelect={seekToSegment}
              />
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};