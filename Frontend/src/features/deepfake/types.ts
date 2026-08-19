/** A recording in the ASVspoof 2019 LA demo subset, as returned by
 *  GET /tasks/deepfake/dataset/recordings. Carries no bona fide/spoof label —
 *  the ground truth is deliberately server-side only (see the backend's
 *  app/tasks/deepfake/dataset.py module docstring). */
export interface RecordingInfo {
  recording_id: string;
  display_filename: string;
  extension: string;
  size_bytes: number;
}

/** Response shape of POST /tasks/deepfake/run. */
export interface DeepfakeResult {
  model: string;
  model_label: string;
  model_id: string;
  recording_id: string;

  decision: "spoof" | "bonafide";
  spoof_probability: number;
  bonafide_probability: number;
  logits: number[];

  /** The operating point the decision was taken at. */
  threshold: number;
  /** False until Feature 1 produces an EER-calibrated threshold. */
  threshold_calibrated: boolean;
  threshold_version: string;

  /** Which class index the checkpoint's own config calls "spoof". */
  id2label: Record<string, string>;
  spoof_index: number;

  duration: number;
  analysed_seconds: number;
  /** How much audio this model looks at at all. AST is fixed at 10.24s;
   *  wav2vec2 XLS-R has no architectural limit, so this is our own cap. */
  analysis_window_seconds: number;
  truncated: boolean;
  cached: boolean;
}

/** One bar of a score distribution (SRS DF-6). Bins always span 0..1 so the
 *  genuine and synthetic distributions share a common axis. */
export interface ScoreBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

/** One threshold's worth of the Detection Error Tradeoff curve (SRS DF-7).
 *  "Acceptance" is acceptance as GENUINE, the ASVspoof convention. */
export interface DetPoint {
  threshold: number;
  false_acceptance_rate: number;
  false_rejection_rate: number;
}

/** Mean score for one spoofing system, or for the genuine clips. */
export interface AttackSummary {
  attack: string;
  count: number;
  mean_score: number;
  is_spoof: boolean;
}

/** Response shape of POST /tasks/deepfake/scores — Feature 1.
 *  Aggregates only: no per-recording label is ever returned. */
export interface DeepfakeEvaluation {
  model: string;
  model_label: string;
  dataset_id: string;
  scored: number;
  bonafide_count: number;
  spoof_count: number;

  eer_percent: number;
  eer_threshold: number;
  /** SRS DF-9 — a threshold means nothing without the dataset it came from. */
  threshold_provenance: string;

  distributions: { bonafide: ScoreBin[]; spoof: ScoreBin[] };
  det_curve: DetPoint[];

  /** Where the model's shipped threshold currently sits on that curve. */
  operating_point: {
    threshold: number;
    calibrated: boolean;
    false_acceptance_rate: number;
    false_rejection_rate: number;
  };

  per_attack: AttackSummary[];
}

/** One of the three scorings in the silence probe (SRS DF-10).
 *  `applicable: false` means there was too little audio of that kind to
 *  score meaningfully — DF-12 requires saying so rather than guessing. */
export interface ProbeVariant {
  applicable: boolean;
  seconds: number;
  spoof_probability: number | null;
  decision: "spoof" | "bonafide" | null;
  reason?: string;
}

/** Response shape of POST /tasks/deepfake/silence-probe — Feature 2. */
export interface SilenceProbeResult {
  model: string;
  model_label: string;
  recording_id: string;

  threshold: number;
  threshold_calibrated: boolean;
  /** SRS DF-11 — the energy threshold that defined "silence", reported with
   *  the result. Relative to the clip's own peak, not an absolute floor. */
  silence_top_db: number;
  min_non_speech_seconds: number;

  duration: number;
  speech_seconds: number;
  non_speech_seconds: number;
  non_speech_fraction: number;
  /** [start, end] seconds of each detected speech region. */
  speech_intervals: [number, number][];

  variants: {
    original: ProbeVariant;
    trimmed: ProbeVariant;
    non_speech: ProbeVariant;
  };

  sample_rate: number;
  cached: boolean;
}

/** One time slice of the attribution. Shape matches the shared saliency
 *  service's segments exactly, so the payload is interchangeable with it. */
export interface SaliencySegment {
  start_time: number;
  end_time: number;
  saliency: number;
  intensity: number;
}

/** Response shape of POST /tasks/deepfake/saliency — Feature 3. */
export interface DeepfakeSaliency {
  model: string;
  model_label: string;
  recording_id: string;

  /** Machine name of the attribution method (shared-service contract). */
  method: string;
  /** SRS DF-15 — human-readable method name, shown in the interface. */
  method_label: string;
  /** Which logit the gradient was taken of. */
  target: string;

  segments: SaliencySegment[];
  /** Normalised 0..1 attribution, one value per segment. */
  series: number[];
  total_duration: number;

  /** The shared saliency service's duration cap, applied here too (DF-15). */
  max_saliency_seconds: number;
  analysis_window_seconds: number | null;
  truncated: boolean;
  /** Attribution is ranked within this clip only — never comparable across
   *  clips or models. */
  normalised: boolean;

  /** Speech regions from Feature 2's segmentation, so the heat can be read
   *  against where the voice actually is. */
  speech_intervals: [number, number][];
  silence_top_db: number;
  /** Share of total attribution falling inside speech. Low means the model
   *  reacted to something other than the voice. */
  saliency_in_speech_fraction: number | null;

  cached: boolean;
}
