interface ScoreBarProps {
  /** The spoof score, 0..1 — the number the decision is actually taken on. */
  spoofProbability: number;
  threshold: number;
  calibrated: boolean;
}

/**
 * The score on its 0..1 axis with the operating threshold drawn on it.
 *
 * A detector does not output a category, it outputs a number that a threshold
 * turns into a decision — so the bar shows the score AND where the cut sits,
 * never just the label.
 *
 * Labelled "spoof score", not "P(spoof)": the softmax output is a RANKING
 * score, not a calibrated probability (Software Architecture Document, Use
 * Case 4). Calling it a probability would invite reading 0.90 as "90% likely
 * fake", which it does not mean.
 */
export const ScoreBar = ({ spoofProbability, threshold, calibrated }: ScoreBarProps) => {
  const percent = Math.max(0, Math.min(1, spoofProbability)) * 100;
  const thresholdPercent = Math.max(0, Math.min(1, threshold)) * 100;
  const isSpoof = spoofProbability >= threshold;

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between text-sm">
        <span className="text-muted-foreground">spoof score</span>
        <span className="font-mono text-lg font-semibold tabular-nums">
          {spoofProbability.toFixed(3)}
        </span>
      </div>

      <div className="relative h-6 w-full overflow-hidden rounded-md border bg-muted">
        <div
          className={`h-full transition-all ${isSpoof ? "bg-destructive/70" : "bg-primary/60"}`}
          style={{ width: `${percent}%` }}
        />
        {/* The operating point. */}
        <div
          className="absolute inset-y-0 w-0.5 bg-foreground"
          style={{ left: `${thresholdPercent}%` }}
          aria-hidden
        />
      </div>

      <div className="flex justify-between text-xs text-muted-foreground">
        <span>0.0 — bona fide</span>
        <span>
          threshold {threshold.toFixed(2)}
          {calibrated ? "" : ", uncalibrated"}
        </span>
        <span>spoof — 1.0</span>
      </div>
    </div>
  );
};
