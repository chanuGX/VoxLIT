import { API_BASE } from "@/lib/api";

/**
 * Resolve a Speaker Verification recording id (opaque demo `rec_...` id,
 * session-scoped `asset_...` id, or owned custom-dataset `crec_...` id) to
 * its streamable audio URL. All three routes are session (`sid` cookie)
 * scoped -- callers must pass `requireCredentials` to `WaveformViewer` when
 * using this URL.
 */
export function verificationAudioUrl(recordingId: string): string {
  if (recordingId.startsWith("asset_")) {
    return `${API_BASE}/tasks/verification/session-assets/${encodeURIComponent(recordingId)}/audio`;
  }
  if (recordingId.startsWith("crec_")) {
    return `${API_BASE}/tasks/verification/custom-recordings/${encodeURIComponent(recordingId)}/audio`;
  }
  return `${API_BASE}/tasks/verification/dataset/recordings/${encodeURIComponent(recordingId)}/audio`;
}
