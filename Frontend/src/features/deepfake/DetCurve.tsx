import { DetPoint } from "./types";

interface DetCurveProps {
  points: DetPoint[];
  eerPercent: number;
  /** Rates at the threshold currently in force, to mark the live operating point. */
  operating: { false_acceptance_rate: number; false_rejection_rate: number };
}

const SIZE = 260;
const PADDING = { top: 12, right: 12, bottom: 34, left: 40 };

/**
 * SRS DF-7 — the Detection Error Tradeoff curve.
 *
 * One point on the curve is one threshold; the whole curve is the detector's
 * complete behaviour, independent of where the cut happens to be set. False
 * acceptance is plotted against false rejection, so the diagonal is where the
 * two errors are equal — and where the curve crosses it is the EER.
 */
export const DetCurve = ({ points, eerPercent, operating }: DetCurveProps) => {
  const plot = SIZE - PADDING.left - PADDING.right;
  const height = SIZE - PADDING.top - PADDING.bottom;

  const x = (rate: number) => PADDING.left + rate * plot;
  const y = (rate: number) => PADDING.top + (1 - rate) * height;

  // Ordered along the threshold sweep so the polyline traces the tradeoff.
  const ordered = [...points].sort((a, b) => a.threshold - b.threshold);
  const path = ordered
    .map(
      (point, index) =>
        `${index === 0 ? "M" : "L"} ${x(point.false_rejection_rate).toFixed(2)} ${y(
          point.false_acceptance_rate
        ).toFixed(2)}`
    )
    .join(" ");

  // Where the two errors are closest to equal — the EER point itself.
  const eerPoint = ordered.reduce((best, point) =>
    Math.abs(point.false_acceptance_rate - point.false_rejection_rate) <
    Math.abs(best.false_acceptance_rate - best.false_rejection_rate)
      ? point
      : best
  );

  return (
    <div className="space-y-2">
      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          className="h-auto w-full min-w-[240px] max-w-[320px]"
          role="img"
          aria-label="Detection error tradeoff curve"
        >
          {/* axes */}
          <line
            x1={PADDING.left}
            y1={PADDING.top}
            x2={PADDING.left}
            y2={PADDING.top + height}
            className="stroke-border"
          />
          <line
            x1={PADDING.left}
            y1={PADDING.top + height}
            x2={PADDING.left + plot}
            y2={PADDING.top + height}
            className="stroke-border"
          />

          {/* Equal-error diagonal: the curve crosses it at the EER. */}
          <line
            x1={x(0)}
            y1={y(0)}
            x2={x(1)}
            y2={y(1)}
            className="stroke-muted-foreground/40"
            strokeDasharray="3 3"
          />

          <path d={path} className="fill-none stroke-primary" strokeWidth={1.75} />

          {/* the EER crossing */}
          <circle
            cx={x(eerPoint.false_rejection_rate)}
            cy={y(eerPoint.false_acceptance_rate)}
            r={4}
            className="fill-background stroke-foreground"
            strokeWidth={2}
          />
          {/* where the shipped threshold actually sits */}
          <circle
            cx={x(operating.false_rejection_rate)}
            cy={y(operating.false_acceptance_rate)}
            r={3.5}
            className="fill-destructive/80"
          />

          {[0, 0.5, 1].map((tick) => (
            <g key={tick}>
              <text
                x={x(tick)}
                y={SIZE - 18}
                textAnchor="middle"
                className="fill-muted-foreground text-[9px]"
              >
                {(tick * 100).toFixed(0)}%
              </text>
              <text
                x={PADDING.left - 5}
                y={y(tick) + 3}
                textAnchor="end"
                className="fill-muted-foreground text-[9px]"
              >
                {(tick * 100).toFixed(0)}%
              </text>
            </g>
          ))}
          <text
            x={PADDING.left + plot / 2}
            y={SIZE - 4}
            textAnchor="middle"
            className="fill-muted-foreground text-[9px]"
          >
            false rejection (genuine flagged)
          </text>
        </svg>
      </div>

      <div className="space-y-1 text-xs text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full border-2 border-foreground bg-background" />
          equal error rate — <span className="font-mono">{eerPercent.toFixed(2)}%</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-destructive/80" />
          threshold in force
        </div>
        <div>y axis: false acceptance (spoof let through)</div>
      </div>
    </div>
  );
};
