import { useMemo, useRef, useState } from "react";
import type WaveSurfer from "wavesurfer.js";
import { AlertCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { WaveformViewer } from "@/components/audio/WaveformViewer";
import type { SaliencyMapResponse, SaliencySegment } from "./saliencyTypes";

// A deliberately conservative *visualization* cutoff, not a claim about
// measured embedding-extraction noise -- no such calibration exists yet, so
// this does not assert one. Below this per-segment cosine-similarity-change
// magnitude, the map is treated as too small to usefully distinguish "high"
// from "low" influence on a per-result normalized scale.
const MIN_DISPLAY_INFLUENCE_DELTA = 0.01;

const SEGMENT_COUNT_MIN = 4;
const SEGMENT_COUNT_MAX = 20;

function intensityToColor(v: number): string {
  const clamp = (x: number) => Math.max(0, Math.min(1, x));
  const mix = (a: number, b: number, t: number) => a + (b - a) * t;
  // low: light blue -> mid: yellow -> high: orange/red
  let h: number, s: number, l: number;
  if (v < 0.5) {
    const t = clamp(v / 0.5);
    h = mix(205, 45, t);
    s = mix(85, 93, t);
    l = mix(88, 58, t);
  } else {
    const t = clamp((v - 0.5) / 0.5);
    h = mix(45, 9, t);
    s = mix(93, 86, t);
    l = mix(58, 50, t);
  }
  return `hsl(${h} ${s}% ${l}%)`;
}

type SegmentClassification = "supports" | "opposes" | "minimal";

function classifySegment(segment: SaliencySegment): SegmentClassification {
  if (segment.influence_strength < MIN_DISPLAY_INFLUENCE_DELTA) return "minimal";
  return segment.similarity_change > 0 ? "supports" : "opposes";
}

const CLASSIFICATION_LABEL: Record<SegmentClassification, string> = {
  supports: "Supports similarity",
  opposes: "Opposes similarity",
  minimal: "Minimal influence",
};

const CLASSIFICATION_UNDERLINE: Record<SegmentClassification, string> = {
  supports: "#10b981", // emerald-500
  opposes: "#f43f5e", // rose-500
  minimal: "#9ca3af", // gray-400
};

const formatScore = (value: number) => value.toFixed(4);
const formatSigned = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
const formatTime = (seconds: number) => `${seconds.toFixed(2)}s`;

export interface SpeakerSaliencyMapProps {
  title: string;
  audioUrl: string | undefined;
  requireCredentials: boolean;
  result: SaliencyMapResponse | null;
  isLoading: boolean;
  error: string | null;
  /** Non-null: the displayed result no longer matches the current
   *  model/selection/segment-count -- dim it and explain why, rather than
   *  silently presenting it as current or removing it outright. */
  staleReason: string | null;
  /** Non-null: no Generate action is offered at all (e.g. singleton
   *  cluster, no valid batch/verification result) -- this replaces the
   *  Generate button and any prior result entirely. */
  emptyStateMessage: string | null;
  onGenerate: (() => void) | null;
  generateLabel: string;
  segmentCount: number;
  onSegmentCountChange: (segmentCount: number) => void;
  clusterBadge?: { label: string; color: string } | null;
}

export const SpeakerSaliencyMap = ({
  title,
  audioUrl,
  requireCredentials,
  result,
  isLoading,
  error,
  staleReason,
  emptyStateMessage,
  onGenerate,
  generateLabel,
  segmentCount,
  onSegmentCountChange,
  clusterBadge,
}: SpeakerSaliencyMapProps) => {
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const maxInfluence = useMemo(
    () => (result ? Math.max(...result.segments.map((s) => s.influence_strength), 0) : 0),
    [result]
  );
  const isResultBelowDisplayThreshold = maxInfluence < MIN_DISPLAY_INFLUENCE_DELTA;

  const normalizedStrength = (influenceStrength: number): number => {
    if (isResultBelowDisplayThreshold) return 0; // paint everything at the low end
    return Math.min(1, influenceStrength / maxInfluence);
  };

  const seekTo = (segment: SaliencySegment) => {
    if (!result || !wavesurferRef.current) return;
    const fraction = result.audio_duration_seconds > 0 ? segment.start_seconds / result.audio_duration_seconds : 0;
    try {
      wavesurferRef.current.seekTo(Math.max(0, Math.min(1, fraction)));
    } catch {
      // Playback not ready yet -- seeking is a nice-to-have, never fatal.
    }
  };

  const heatmapStrip = result && (
    <div className="space-y-2">
      <div
        className="relative flex h-8 w-full overflow-hidden rounded border border-border"
        onMouseLeave={() => setHoveredIndex(null)}
      >
        {result.segments.map((segment, index) => {
          const classification = classifySegment(segment);
          const widthPct =
            result.audio_duration_seconds > 0
              ? ((segment.end_seconds - segment.start_seconds) / result.audio_duration_seconds) * 100
              : 100 / result.segments.length;
          return (
            <div
              key={segment.segment_index}
              className={`h-full cursor-pointer transition-opacity ${
                hoveredIndex !== null && hoveredIndex !== index ? "opacity-60" : ""
              }`}
              style={{
                width: `${widthPct}%`,
                backgroundColor: intensityToColor(normalizedStrength(segment.influence_strength)),
                borderBottom: `3px solid ${CLASSIFICATION_UNDERLINE[classification]}`,
              }}
              onMouseEnter={() => setHoveredIndex(index)}
              onClick={() => seekTo(segment)}
            />
          );
        })}
        {hoveredIndex !== null && (
          <div className="pointer-events-none absolute -top-2 left-2 z-10 -translate-y-full rounded border border-border bg-popover px-2 py-1 text-xs shadow">
            {(() => {
              const segment = result.segments[hoveredIndex];
              const classification = classifySegment(segment);
              return (
                <div className="space-y-0.5">
                  <div className="font-medium">
                    {formatTime(segment.start_seconds)}–{formatTime(segment.end_seconds)}
                  </div>
                  <div>Baseline similarity: {formatScore(result.baseline_similarity)}</div>
                  <div>Occluded similarity: {formatScore(segment.occluded_similarity)}</div>
                  <div>Change: {formatSigned(segment.similarity_change)}</div>
                  <div>Influence: {formatScore(segment.influence_strength)}</div>
                  <div className="font-medium">{CLASSIFICATION_LABEL[classification]}</div>
                </div>
              );
            })()}
          </div>
        )}
      </div>
      <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: intensityToColor(0) }} />
          Low influence
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: intensityToColor(0.5) }} />
          Medium influence
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: intensityToColor(1) }} />
          High influence
        </span>
      </div>
      {isResultBelowDisplayThreshold && (
        <p className="text-[10px] text-muted-foreground">
          All segment changes are below the minimum display-influence threshold.
        </p>
      )}
    </div>
  );

  const mostInfluential = result
    ? [...result.segments].sort((a, b) => b.influence_strength - a.influence_strength)
    : [];

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-sm">
            {title}
            {clusterBadge && (
              <Badge
                variant="outline"
                style={{ borderColor: clusterBadge.color, color: clusterBadge.color }}
                className="text-[10px]"
              >
                {clusterBadge.label}
              </Badge>
            )}
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {emptyStateMessage ? (
          <p className="text-xs text-muted-foreground">{emptyStateMessage}</p>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex min-w-[10rem] flex-1 items-center gap-2 text-xs text-muted-foreground">
                <span className="whitespace-nowrap">Segments: {segmentCount}</span>
                <Slider
                  className="max-w-[10rem]"
                  min={SEGMENT_COUNT_MIN}
                  max={SEGMENT_COUNT_MAX}
                  step={1}
                  value={[segmentCount]}
                  onValueChange={([value]) => onSegmentCountChange(value)}
                  disabled={isLoading}
                  thumbLabel={`Segments: ${segmentCount}`}
                />
              </div>
              <Button size="sm" variant="outline" onClick={() => onGenerate?.()} disabled={!onGenerate || isLoading}>
                {isLoading ? "Running occlusion passes…" : generateLabel}
              </Button>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Saliency map could not be generated</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {staleReason && (
              <Alert className="border-amber-300 bg-amber-50">
                <AlertCircle className="h-4 w-4 text-amber-700" />
                <AlertDescription>{staleReason}</AlertDescription>
              </Alert>
            )}

            {result && (
              <div className={staleReason ? "space-y-3 opacity-50 pointer-events-none" : "space-y-3"}>
                <WaveformViewer
                  audioUrl={audioUrl}
                  requireCredentials={requireCredentials}
                  onReady={(wavesurfer) => {
                    wavesurferRef.current = wavesurfer;
                  }}
                  timelineBelow={heatmapStrip}
                />

                <div className="space-y-1">
                  <div className="text-xs font-medium">Most influential segments</div>
                  {mostInfluential.map((segment) => {
                    const classification = classifySegment(segment);
                    return (
                      <div
                        key={segment.segment_index}
                        className={`flex items-center justify-between rounded px-2 py-1 text-xs cursor-pointer ${
                          hoveredIndex === segment.segment_index - 1 ? "bg-muted" : ""
                        }`}
                        onMouseEnter={() => setHoveredIndex(segment.segment_index - 1)}
                        onMouseLeave={() => setHoveredIndex(null)}
                        onClick={() => seekTo(segment)}
                      >
                        <span>
                          {formatTime(segment.start_seconds)}–{formatTime(segment.end_seconds)}
                        </span>
                        <span className="font-mono">{formatSigned(segment.similarity_change)}</span>
                        <span className="text-muted-foreground">{CLASSIFICATION_LABEL[classification]}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
};
