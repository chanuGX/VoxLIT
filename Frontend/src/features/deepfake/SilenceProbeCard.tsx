import { useEffect, useState } from "react";
import { AlertCircle, Loader2, Scissors } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ProbeVariant, SilenceProbeResult } from "./types";

interface SilenceProbeCardProps {
  model: string;
  modelLabel: string;
  recordingId: string;
}

/**
 * Feature 2 — the silence and non-speech probe (SRS DF-10..DF-12).
 *
 * Scores the same clip three ways: as submitted, with the outer silence
 * trimmed, and with only the non-speech kept. In ASVspoof 2019 LA the genuine
 * clips carry markedly longer silence than most attacks, because TTS systems
 * trim their silences and real recordings do not — so a detector can score
 * well by reading the corpus rather than the voice.
 *
 * This is the one view that can invalidate everything else on the screen,
 * including the EER: if trimming collapses the score, or silence alone still
 * says "spoof", the number was never evidence about the speech.
 */
export const SilenceProbeCard = ({ model, modelLabel, recordingId }: SilenceProbeCardProps) => {
  const [result, setResult] = useState<SilenceProbeResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // A probe belongs to one clip and one model; showing a stale one would
  // silently attribute another recording's behaviour to this one.
  useEffect(() => {
    setResult(null);
    setError(null);
  }, [recordingId, model]);

  const runProbe = async () => {
    setIsRunning(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch(`${API_BASE}/tasks/deepfake/silence-probe`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, recording_id: recordingId }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Probe failed (${response.status})`);
      }
      setResult(payload as SilenceProbeResult);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Probe failed.");
    } finally {
      setIsRunning(false);
    }
  };

  const verdict = result ? readVerdict(result) : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
          <Scissors className="h-4 w-4 text-primary" />
          Silence and non-speech probe
          <Badge variant="outline">{modelLabel}</Badge>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Scores this clip three times — as it is, with the outer silence trimmed, and with
          only the non-speech kept. If trimming the silence collapses the score, or the
          silence alone still reads as spoof, the detector is going on the recording
          conditions rather than the voice.
        </p>

        <Button onClick={runProbe} disabled={isRunning || !recordingId}>
          {isRunning ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Scoring three ways…
            </>
          ) : (
            <>
              <Scissors className="mr-2 h-4 w-4" /> Run the silence probe
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
          <div className="space-y-4">
            {/* DF-10 — the three scores, side by side. */}
            <div className="grid gap-3 sm:grid-cols-3">
              <VariantTile
                title="As submitted"
                subtitle="the whole clip"
                variant={result.variants.original}
                threshold={result.threshold}
              />
              <VariantTile
                title="Silence trimmed"
                subtitle="outer silence removed"
                variant={result.variants.trimmed}
                threshold={result.threshold}
                comparedTo={result.variants.original}
              />
              <VariantTile
                title="Non-speech only"
                subtitle="the silence by itself"
                variant={result.variants.non_speech}
                threshold={result.threshold}
                comparedTo={result.variants.original}
              />
            </div>

            {verdict && (
              <Alert variant={verdict.alarming ? "destructive" : undefined}>
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>{verdict.title}</AlertTitle>
                <AlertDescription>{verdict.detail}</AlertDescription>
              </Alert>
            )}

            {/* How much of this clip is silence at all. */}
            <div className="space-y-1.5">
              <div className="flex items-baseline justify-between text-xs text-muted-foreground">
                <span>speech vs non-speech</span>
                <span className="font-mono tabular-nums">
                  {result.speech_seconds.toFixed(2)}s / {result.non_speech_seconds.toFixed(2)}s
                </span>
              </div>
              <div
                className="flex h-3 w-full overflow-hidden rounded-md border"
                title={`${(result.non_speech_fraction * 100).toFixed(1)}% non-speech`}
              >
                <div
                  className="bg-primary/60"
                  style={{ width: `${(1 - result.non_speech_fraction) * 100}%` }}
                />
                <div
                  className="bg-muted-foreground/30"
                  style={{ width: `${result.non_speech_fraction * 100}%` }}
                />
              </div>
            </div>

            {/* DF-11 — the threshold that defined "silence" travels with the result. */}
            <p className="rounded-md border border-dashed bg-muted/40 p-2 text-xs text-muted-foreground">
              Non-speech found with an energy threshold {result.silence_top_db} dB below this
              clip&apos;s own peak — relative to the recording, not an absolute noise floor.
              A silence-only score needs at least {result.min_non_speech_seconds.toFixed(2)}s
              of non-speech to be reported.
              {result.cached ? " (cached)" : ""}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

/** One of the three scorings, or a stated reason it could not be run. */
const VariantTile = ({
  title,
  subtitle,
  variant,
  threshold,
  comparedTo,
}: {
  title: string;
  subtitle: string;
  variant: ProbeVariant;
  threshold: number;
  comparedTo?: ProbeVariant;
}) => {
  if (!variant.applicable) {
    // DF-12 — say so, rather than show an unreliable number.
    return (
      <div className="rounded-md border border-dashed p-3">
        <div className="text-xs font-medium">{title}</div>
        <div className="mt-1 text-sm text-muted-foreground">Not applicable</div>
        <div className="mt-1 text-[11px] leading-tight text-muted-foreground">
          {variant.reason}
        </div>
      </div>
    );
  }

  const score = variant.spoof_probability ?? 0;
  const isSpoof = score >= threshold;
  const delta =
    comparedTo && comparedTo.applicable && comparedTo.spoof_probability !== null
      ? score - comparedTo.spoof_probability
      : null;

  return (
    <div className="rounded-md border p-3">
      <div className="text-xs font-medium">{title}</div>
      <div className="text-[11px] text-muted-foreground">{subtitle}</div>
      <div className="mt-2 font-mono text-2xl font-semibold tabular-nums">
        {score.toFixed(3)}
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-1.5">
        <Badge variant={isSpoof ? "destructive" : "secondary"} className="text-[10px]">
          {isSpoof ? "SPOOF" : "BONA FIDE"}
        </Badge>
        {delta !== null && (
          <span className="font-mono text-[11px] text-muted-foreground tabular-nums">
            {delta >= 0 ? "+" : ""}
            {delta.toFixed(3)}
          </span>
        )}
      </div>
      <div className="mt-1 text-[11px] text-muted-foreground">{variant.seconds.toFixed(2)}s</div>
    </div>
  );
};

/** Turn the three scores into the finding the researcher is looking for. */
function readVerdict(result: SilenceProbeResult) {
  const { original, trimmed, non_speech: nonSpeech } = result.variants;
  const base = original.spoof_probability ?? 0;

  const silenceAloneAccuses =
    nonSpeech.applicable &&
    nonSpeech.spoof_probability !== null &&
    nonSpeech.spoof_probability >= result.threshold;

  const trimmingChangedTheAnswer =
    trimmed.applicable &&
    trimmed.spoof_probability !== null &&
    trimmed.decision !== original.decision;

  const trimmingMovedTheScore =
    trimmed.applicable &&
    trimmed.spoof_probability !== null &&
    Math.abs(trimmed.spoof_probability - base) >= 0.25;

  if (silenceAloneAccuses) {
    return {
      alarming: true,
      title: "The silence alone reads as spoof",
      detail:
        "With every speech region removed, the detector still calls this clip synthetic. " +
        "Whatever it is keying on is in the recording conditions, not the voice — treat " +
        "this clip's score, and any EER built from clips like it, with suspicion.",
    };
  }
  if (trimmingChangedTheAnswer) {
    return {
      alarming: true,
      title: "Trimming the silence flipped the decision",
      detail:
        "Removing the leading and trailing silence changed the verdict. The decision was " +
        "resting on the silence rather than on the speech.",
    };
  }
  if (trimmingMovedTheScore) {
    return {
      alarming: false,
      title: "Trimming moved the score noticeably",
      detail:
        "The decision held, but the score shifted by more than 0.25 once the silence was " +
        "removed — some of the evidence was coming from the non-speech regions.",
    };
  }
  // The silence-only leg is the stronger half of the check, so when it could
  // not be run the verdict must not imply that it passed.
  if (!nonSpeech.applicable) {
    return {
      alarming: false,
      title: "Trimming left the score intact, but the silence check could not run",
      detail:
        "Removing the outer silence did not move the decision. There was too little " +
        "non-speech in this clip to score on its own, so the stronger half of the probe " +
        "is untested here — try a clip with more silence before concluding anything.",
    };
  }

  return {
    alarming: false,
    title: "The score survives the ablation",
    detail:
      "Trimming the silence left the decision and the score broadly intact, and the " +
      "silence on its own does not read as spoof. On this clip the detector is going on " +
      "the speech.",
  };
}
