import { useState } from "react";
import { AlertCircle, BarChart3, Loader2, Play } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DetCurve } from "./DetCurve";
import { ScoreDistribution } from "./ScoreDistribution";
import { DeepfakeEvaluation } from "./types";

interface EvaluationPanelProps {
  model: string;
  modelLabel: string;
  /** Disabled until the labelled dataset is actually on disk. */
  datasetAvailable: boolean;
}

/**
 * Feature 1 — score distribution, DET curve and threshold.
 *
 * The task's batch-analysis view (SRS DF-8). Detection is a scoring problem,
 * so a single clip's "spoof, 0.93" hides both of the things that matter: how
 * separable the two populations are, and what the chosen threshold costs.
 * This is also the only way to quote an EER at all, which is what makes the
 * numbers comparable with published work.
 */
export const EvaluationPanel = ({ model, modelLabel, datasetAvailable }: EvaluationPanelProps) => {
  const [result, setResult] = useState<DeepfakeEvaluation | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runEvaluation = async () => {
    setIsRunning(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch(`${API_BASE}/tasks/deepfake/scores`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Evaluation failed (${response.status})`);
      }
      setResult(payload as DeepfakeEvaluation);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Evaluation failed.");
    } finally {
      setIsRunning(false);
    }
  };

  // Stale results would silently belong to a different model.
  const isStale = result !== null && result.model !== model;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
          <BarChart3 className="h-4 w-4 text-primary" />
          Score distribution, DET curve and threshold
          <Badge variant="outline">{modelLabel}</Badge>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Scores every clip in the labelled dataset and lines the scores up against the
          ground truth. How far the two populations separate is what decides whether the
          detector is usable at all — a single clip&apos;s score cannot show it.
        </p>

        {!datasetAvailable && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>No labelled dataset</AlertTitle>
            <AlertDescription>
              This view needs the ASVspoof 2019 LA subset on disk. Build it with{" "}
              <code className="text-xs">scripts/prepare_asvspoof_la_subset.py</code>.
            </AlertDescription>
          </Alert>
        )}

        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={runEvaluation} disabled={isRunning || !datasetAvailable}>
            {isRunning ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Scoring the dataset…
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" /> Evaluate on the labelled dataset
              </>
            )}
          </Button>
          {isRunning && (
            <span className="text-xs text-muted-foreground">
              One forward pass per clip — the first run takes minutes, then scores are cached.
            </span>
          )}
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Something went wrong</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {isStale && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Model changed</AlertTitle>
            <AlertDescription>
              These results are for {result.model_label}. Re-run to evaluate {modelLabel}.
            </AlertDescription>
          </Alert>
        )}

        {result && !isStale && (
          <div className="space-y-5">
            {/* --- the headline numbers (DF-8) --- */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="Equal error rate" value={`${result.eer_percent.toFixed(2)}%`} emphasis />
              <Stat label="at threshold" value={result.eer_threshold.toFixed(4)} />
              <Stat label="Genuine clips" value={String(result.bonafide_count)} />
              <Stat label="Synthetic clips" value={String(result.spoof_count)} />
            </div>

            {/* --- DF-9: a threshold is meaningless without its dataset --- */}
            <p className="rounded-md border border-dashed bg-muted/40 p-2 text-xs text-muted-foreground">
              {result.threshold_provenance}
            </p>

            <section className="space-y-2">
              <h4 className="text-sm font-medium">Score distributions</h4>
              <ScoreDistribution
                bonafide={result.distributions.bonafide}
                spoof={result.distributions.spoof}
                eerThreshold={result.eer_threshold}
                operatingThreshold={result.operating_point.threshold}
              />
            </section>

            <section className="grid gap-4 xl:grid-cols-[auto,1fr]">
              <div className="space-y-2">
                <h4 className="text-sm font-medium">DET curve</h4>
                <DetCurve
                  points={result.det_curve}
                  eerPercent={result.eer_percent}
                  operating={result.operating_point}
                />
              </div>

              <div className="space-y-2">
                <h4 className="text-sm font-medium">
                  At the threshold in force ({result.operating_point.threshold.toFixed(2)}
                  {result.operating_point.calibrated ? "" : ", uncalibrated"})
                </h4>
                <div className="grid grid-cols-2 gap-3">
                  <Stat
                    label="False acceptance"
                    value={`${(result.operating_point.false_acceptance_rate * 100).toFixed(1)}%`}
                    hint="spoofed clips let through as genuine"
                  />
                  <Stat
                    label="False rejection"
                    value={`${(result.operating_point.false_rejection_rate * 100).toFixed(1)}%`}
                    hint="genuine clips wrongly flagged"
                  />
                </div>

                {result.per_attack.length > 1 && (
                  <div className="pt-2">
                    <h4 className="mb-1 text-sm font-medium">Mean score per attack</h4>
                    <div className="max-h-44 overflow-y-auto rounded-md border scrollbar-thin">
                      <table className="w-full text-xs">
                        <tbody>
                          {result.per_attack.map((row) => (
                            <tr key={row.attack} className="border-b last:border-0">
                              <td className="px-2 py-1 font-mono">
                                {row.attack === "bonafide" ? "genuine" : row.attack}
                              </td>
                              <td className="px-2 py-1 text-muted-foreground">n={row.count}</td>
                              <td className="px-2 py-1 text-right font-mono tabular-nums">
                                {row.mean_score.toFixed(3)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </section>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

const Stat = ({
  label,
  value,
  hint,
  emphasis = false,
}: {
  label: string;
  value: string;
  hint?: string;
  emphasis?: boolean;
}) => (
  <div className="rounded-md border p-2">
    <div className="text-xs text-muted-foreground">{label}</div>
    <div className={`font-mono tabular-nums ${emphasis ? "text-xl font-semibold" : "text-base"}`}>
      {value}
    </div>
    {hint && <div className="mt-0.5 text-[10px] leading-tight text-muted-foreground">{hint}</div>}
  </div>
);
