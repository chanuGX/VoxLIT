/**
 * Feature folder for "Voice Task A" (see src/tasks/registry.tsx).
 *
 * OWNER: member 1 — implement your task-specific UI here:
 *   • an explainability component for the workbench center panel
 *   • a PredictionResults component for the Datapoint Editor (see
 *     features/transcription/TranscriptionResults.tsx as a reference)
 *
 * Then register them in src/tasks/registry.tsx (TASK_SLOTS + capabilities)
 * and flip the task's status to "active".
 */
export { SpeakerVerificationWorkbench } from "./SpeakerVerificationWorkbench.tsx";
export { ClusterAssignmentResults } from "./ClusterAssignmentResults.tsx";