import { useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, Flame, Loader2 } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DeepfakeSaliency } from "./types";

interface SaliencyPanelProps {
  model: string;
  modelLabel: string;
  recordingId: string;
}

const HEIGHT = 120;
const WIDTH = 640;

/**
 * Feature 3 — waveform-aligned saliency (SRS DF-14, DF-15).
 *
 * The gradient of the spoof logit with respect to the input, drawn as heat
 * along the waveform so the moments the model reacted to are visible in
 * time. Speech regions are shaded, because the question this view exists to
 * answer is whether the bright regions sit on the voice or on the silence.
 */
export const SaliencyPanel = ({ model, modelLabel, recordingId }: SaliencyPanelProps) => {
  const [result, setResult] = useState<DeepfakeSaliency | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [peaks, setPeaks] = useState<number[] | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const audioUrl = recordingId
    ? `${API_BASE}/tasks/deepfake/dataset/recordings/${recordingId}/audio`
    : undefined;

  // An attribution belongs to one clip and one model.
  useEffect(() => {
    setResult(null);
    setError(null);
    setPeaks(null);
  }, [recordingId, model]);

  // Decode the clip once so the heat can be drawn over a real waveform
  // rather than over an empty strip.
  useEffect(() => {
    if (!audioUrl) return;
    let cancelled = false;

    const decode = async () => {
      try {
        const response = await fetch(audioUrl, { credentials: "include" });
        const buffer = await response.arrayBuffer();
        const context = new (window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
        const audio = await context.decodeAudioData(buffer);
        const channel = audio.getChannelData(0);
        const bucketSize = Math.floor(channel.length / WIDTH) || 1;
        const envelope: number[] = [];
        for (let index = 0; index < WIDTH; index += 1) {
          let peak = 0;
          const start = index * bucketSize;
          for (let offset = 0; offset < bucketSize; offset += 1) {
            const value = Math.abs(channel[start + offset] ?? 0);
            if (value > peak) peak = value;
          }
          envelope.push(peak);
        }
        const loudest = Math.max(...envelope, 1e-6);
        if (!cancelled) setPeaks(envelope.map((value) => value / loudest));
        context.close();
      } catch {
        if (!cancelled) setPeaks(null); // the heat strip still renders
      }
    };
    decode();
    return () => {
      cancelled = true;
    };
  }, [audioUrl]);

  const runSaliency = async () => {
    setIsRunning(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch(`${API_BASE}/tasks/deepfake/saliency`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, recording_id: recordingId }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `Saliency failed (${response.status})`);
      setResult(payload as DeepfakeSaliency);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Saliency failed.");
    } finally {
      setIsRunning(false);
    }
  };

  const verdict = useMemo(() => (result ? readVerdict(result) : null), [result]);

  const seekTo = (seconds: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = seconds;
    audio.play().catch(() => undefined);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
          <Flame className="h-4 w-4 text-primary" />
          Waveform-aligned saliency
          <Badge variant="outline">{modelLabel}</Badge>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Which moments pushed the model toward &ldquo;spoof&rdquo;, drawn as heat along the
          waveform. If the bright regions sit on silence rather than on the voice, the score
          was never evidence about the speech.
        </p>

        <Button onClick={runSaliency} disabled={isRunning || !recordingId}>
          {isRunning ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Computing attribution…
            </>
          ) : (
            <>
              <Flame className="mr-2 h-4 w-4" /> Show saliency
            </>
          )}
        </Button>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Something went wrong</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {result && (
          <div className="space-y-3">
            <SaliencyStrip
              result={result}
              peaks={peaks}
              onSeek={seekTo}
            />

            {audioUrl && <audio ref={audioRef} controls className="w-full" src={audioUrl} />}

            {verdict && (
              <Alert variant={verdict.alarming ? "destructive" : undefined}>
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>{verdict.title}</AlertTitle>
                <AlertDescription>{verdict.detail}</AlertDescription>
              </Alert>
            )}

            {/* DF-15 — the method is named, and the caps are stated. */}
            <p className="rounded-md border border-dashed bg-muted/40 p-2 text-xs text-muted-foreground">
              {result.method_label}, taken on the {result.target}. Attribution is normalised
              within this clip, so it ranks moments against each other and never compares
              across clips or models. Analysis is capped at {result.max_saliency_seconds}s, the
              same cap the shared saliency service applies
              {result.truncated ? " — this clip was truncated to fit" : ""}. Speech regions
              shaded using an energy threshold {result.silence_top_db} dB below the clip&apos;s
              own peak.
              {result.cached ? " (cached)" : ""}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

/** The heat strip, the waveform, and the speech shading, all on one time axis. */
const SaliencyStrip = ({
  result,
  peaks,
  onSeek,
}: {
  result: DeepfakeSaliency;
  peaks: number[] | null;
  onSeek: (seconds: number) => void;
}) => {
  const duration = result.total_duration || 1;
  const x = (seconds: number) => (seconds / duration) * WIDTH;

  return (
    <div className="space-y-1">
      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="h-auto w-full min-w-[340px] cursor-pointer"
          role="img"
          aria-label="Saliency aligned with the waveform"
          onClick={(event) => {
            const box = event.currentTarget.getBoundingClientRect();
            onSeek(((event.clientX - box.left) / box.width) * duration);
          }}
        >
          {/* Speech regions, so the heat can be read against the voice. */}
          {result.speech_intervals.map(([start, end]) => (
            <rect
              key={`speech-${start}`}
              x={x(start)}
              y={0}
              width={Math.max(1, x(end) - x(start))}
              height={HEIGHT}
              className="fill-primary/10"
            />
          ))}

          {/* The attribution itself: one bar per segment, opacity = strength. */}
          {result.segments.map((segment) => (
            <rect
              key={`heat-${segment.start_time}`}
              x={x(segment.start_time)}
              y={0}
              width={Math.max(1, x(segment.end_time) - x(segment.start_time))}
              height={HEIGHT}
              className="fill-destructive"
              opacity={0.08 + segment.saliency * 0.62}
            />
          ))}

          {/* The waveform on top, so shape and heat are read together. */}
          {peaks && (
            <path
              d={peaks
                .map((peak, index) => {
                  const px = (index / peaks.length) * WIDTH;
                  const half = (peak * HEIGHT) / 2.4;
                  return `M ${px.toFixed(1)} ${(HEIGHT / 2 - half).toFixed(1)} L ${px.toFixed(
                    1
                  )} ${(HEIGHT / 2 + half).toFixed(1)}`;
                })
                .join(" ")}
              className="stroke-foreground/70"
              strokeWidth={1}
            />
          )}

          <line
            x1={0}
            y1={HEIGHT / 2}
            x2={WIDTH}
            y2={HEIGHT / 2}
            className="stroke-border"
            strokeWidth={0.5}
          />
        </svg>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-destructive/70" /> attribution
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-primary/20" /> speech
        </span>
        <span>0.00s</span>
        <span>{duration.toFixed(2)}s</span>
        <span className="italic">click to play from there</span>
      </div>
    </div>
  );
};

/** State the finding the view exists to deliver. */
function readVerdict(result: DeepfakeSaliency) {
  const inSpeech = result.saliency_in_speech_fraction;
  if (inSpeech === null) {
    return null;
  }

  // How much would land in speech if attribution were spread evenly? Compare
  // against that rather than against a flat 50%, so a clip that is mostly
  // speech is not judged by the same bar as one that is mostly silence.
  const speechSeconds = result.speech_intervals.reduce(
    (total, [start, end]) => total + (end - start),
    0
  );
  const speechShare = result.total_duration > 0 ? speechSeconds / result.total_duration : 0;
  const percent = (inSpeech * 100).toFixed(1);
  const expected = (speechShare * 100).toFixed(1);

  if (speechShare > 0 && inSpeech < speechShare * 0.5) {
    return {
      alarming: true,
      title: `Only ${percent}% of the attribution falls on speech`,
      detail:
        `Speech occupies ${expected}% of this clip, so an even spread would put about ` +
        `that much attribution on the voice. The model reacted mostly to the non-speech ` +
        `audio instead — the score is being driven by recording conditions rather than by ` +
        `the speech itself.`,
    };
  }
  return {
    alarming: false,
    title: `${percent}% of the attribution falls on speech`,
    detail:
      `Speech occupies ${expected}% of this clip, so the attribution is broadly where the ` +
      `voice is. On this clip the model is reacting to the speech rather than to the ` +
      `silence around it.`,
  };
}
