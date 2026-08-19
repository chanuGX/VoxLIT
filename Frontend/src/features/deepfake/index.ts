/**
 * Audio Deepfake Detection — OWNER: Chanupa Gurusinghe.
 * Registered in src/tasks/registry.tsx (TASK_SLOTS.deepfake.WorkbenchCenter).
 */
export { DeepfakeWorkbench } from "./DeepfakeWorkbench";
export { ScoreBar } from "./ScoreBar";
export { EvaluationPanel } from "./EvaluationPanel";
export { ScoreDistribution } from "./ScoreDistribution";
export { DetCurve } from "./DetCurve";
export { SilenceProbeCard } from "./SilenceProbeCard";
export { SaliencyPanel } from "./SaliencyPanel";
export type {
  DeepfakeResult,
  RecordingInfo,
  DeepfakeEvaluation,
  ScoreBin,
  DetPoint,
  AttackSummary,
  SilenceProbeResult,
  ProbeVariant,
  DeepfakeSaliency,
  SaliencySegment,
} from "./types";
