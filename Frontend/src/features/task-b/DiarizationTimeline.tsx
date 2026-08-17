import { DiarizationSegment, speakerColor } from "./types";

interface DiarizationTimelineProps {
  segments: DiarizationSegment[];
  speakers: string[];
  duration: number;
  hoveredId: string | null;
  selectedId: string | null;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
}

/** Opacity per confidence bucket; uncertain additionally gets hatching. */
const bucketOpacity = (bucket: DiarizationSegment["confidence_bucket"]): number => {
  if (bucket === "high") return 1;
  if (bucket === "medium") return 0.55;
  if (bucket === "uncertain") return 0.4;
  return 0.75; // null: too short to embed / single speaker — plain bar
};

export const DiarizationTimeline = ({
  segments,
  speakers,
  duration,
  hoveredId,
  selectedId,
  onHover,
  onSelect,
}: DiarizationTimelineProps) => {
  const hovered = segments.find((s) => s.id === hoveredId) ?? null;

  return (
    <div className="space-y-2">
      {/* One lane per speaker so overlapping speech stays readable */}
      <div className="space-y-1">
        {speakers.map((speaker) => (
          <div key={speaker} className="flex items-center gap-2">
            <span className="w-24 shrink-0 truncate text-xs text-muted-foreground">
              {speaker}
            </span>
            <div className="relative h-6 flex-1 rounded bg-muted/40">
              {segments
                .filter((segment) => segment.speaker === speaker)
                .map((segment) => {
                  const color = speakerColor(speakers, speaker);
                  const isActive =
                    segment.id === hoveredId || segment.id === selectedId;
                  const hatched = segment.confidence_bucket === "uncertain";
                  return (
                    <div
                      key={segment.id}
                      role="button"
                      className="absolute top-0 h-full cursor-pointer rounded-sm transition-[outline]"
                      style={{
                        left: `${(segment.start / duration) * 100}%`,
                        width: `${Math.max(
                          ((segment.end - segment.start) / duration) * 100,
                          0.15
                        )}%`,
                        backgroundColor: hatched ? undefined : color,
                        backgroundImage: hatched
                          ? `repeating-linear-gradient(45deg, ${color}, ${color} 3px, transparent 3px, transparent 6px)`
                          : undefined,
                        opacity: bucketOpacity(segment.confidence_bucket),
                        outline: isActive ? "2px solid hsl(var(--foreground))" : "none",
                        outlineOffset: 1,
                      }}
                      onMouseEnter={() => onHover(segment.id)}
                      onMouseLeave={() => onHover(null)}
                      onClick={() => onSelect(segment.id)}
                    />
                  );
                })}
            </div>
          </div>
        ))}
      </div>

      {/* Time axis */}
      <div className="ml-[6.5rem] flex justify-between text-[10px] text-muted-foreground">
        <span>0:00</span>
        <span>{formatTime(duration / 2)}</span>
        <span>{formatTime(duration)}</span>
      </div>

      {/* Hover detail bar (stable height so the layout never jumps) */}
      <div className="h-5 text-xs text-muted-foreground">
        {hovered ? (
          <>
            <span className="font-medium text-foreground">{hovered.speaker}</span>
            {"  "}
            {formatTime(hovered.start)}–{formatTime(hovered.end)}
            {hovered.confidence !== null ? (
              <>
                {"  ·  confidence "}
                <span className="font-medium text-foreground">
                  {hovered.confidence.toFixed(3)}
                </span>{" "}
                ({hovered.confidence_bucket})
              </>
            ) : (
              "  ·  no confidence (segment too short to embed)"
            )}
          </>
        ) : (
          "Hover a bar for details — click to seek the audio."
        )}
      </div>
    </div>
  );
};

const formatTime = (seconds: number): string => {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
};