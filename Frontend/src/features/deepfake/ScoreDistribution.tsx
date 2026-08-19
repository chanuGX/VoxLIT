import { ScoreBin } from "./types";

interface ScoreDistributionProps {
  bonafide: ScoreBin[];
  spoof: ScoreBin[];
  /** Threshold where false acceptance and false rejection are equal. */
  eerThreshold: number;
  /** Where the model's shipped threshold currently sits. */
  operatingThreshold: number;
}

const WIDTH = 640;
const HEIGHT = 200;
const PADDING = { top: 12, right: 12, bottom: 30, left: 40 };

/**
 * SRS DF-6 — the genuine and synthetic score distributions on a COMMON axis.
 *
 * The overlap between the two humps is the whole story: it is what decides
 * whether the detector is usable, and no single-clip score can show it. Both
 * histograms are binned over a fixed 0..1 range so the axis is genuinely
 * shared rather than each being fitted to its own data.
 */
export const ScoreDistribution = ({
  bonafide,
  spoof,
  eerThreshold,
  operatingThreshold,
}: ScoreDistributionProps) => {
  const plotWidth = WIDTH - PADDING.left - PADDING.right;
  const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;

  const peak = Math.max(1, ...bonafide.map((b) => b.count), ...spoof.map((b) => b.count));
  const x = (score: number) => PADDING.left + score * plotWidth;
  const barHeight = (count: number) => (count / peak) * plotHeight;

  const renderBars = (bins: ScoreBin[], className: string) =>
    bins
      .filter((bin) => bin.count > 0)
      .map((bin) => {
        const height = barHeight(bin.count);
        return (
          <rect
            key={`${className}-${bin.bin_start}`}
            x={x(bin.bin_start)}
            y={PADDING.top + plotHeight - height}
            width={Math.max(1, x(bin.bin_end) - x(bin.bin_start) - 1)}
            height={height}
            className={className}
          />
        );
      });

  return (
    <div className="space-y-2">
      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="h-auto w-full min-w-[340px]"
          role="img"
          aria-label="Score distributions for genuine and synthetic clips"
        >
          {/* y axis label */}
          <text x={8} y={PADDING.top + 10} className="fill-muted-foreground text-[10px]">
            clips
          </text>

          {/* Overlaid so the OVERLAP is visible; genuine drawn second and
              semi-transparent so neither hides the other. */}
          {renderBars(spoof, "fill-destructive/60")}
          {renderBars(bonafide, "fill-primary/60")}

          {/* baseline */}
          <line
            x1={PADDING.left}
            y1={PADDING.top + plotHeight}
            x2={PADDING.left + plotWidth}
            y2={PADDING.top + plotHeight}
            className="stroke-border"
            strokeWidth={1}
          />

          {/* The shipped operating point. */}
          <line
            x1={x(operatingThreshold)}
            y1={PADDING.top}
            x2={x(operatingThreshold)}
            y2={PADDING.top + plotHeight}
            className="stroke-muted-foreground"
            strokeWidth={1.5}
            strokeDasharray="4 3"
          />
          {/* The equal-error operating point. */}
          <line
            x1={x(eerThreshold)}
            y1={PADDING.top}
            x2={x(eerThreshold)}
            y2={PADDING.top + plotHeight}
            className="stroke-foreground"
            strokeWidth={1.5}
          />

          {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
            <text
              key={tick}
              x={x(tick)}
              y={HEIGHT - 12}
              textAnchor="middle"
              className="fill-muted-foreground text-[10px]"
            >
              {tick.toFixed(2)}
            </text>
          ))}
          <text
            x={PADDING.left + plotWidth / 2}
            y={HEIGHT - 1}
            textAnchor="middle"
            className="fill-muted-foreground text-[10px]"
          >
            detector score (higher = more spoof-like)
          </text>
        </svg>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-primary/60" /> genuine
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-destructive/60" /> synthetic
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-0.5 bg-foreground" /> equal-error threshold{" "}
          {eerThreshold.toFixed(3)}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-0.5 border-l border-dashed border-muted-foreground" />{" "}
          in force {operatingThreshold.toFixed(2)}
        </span>
      </div>
    </div>
  );
};
