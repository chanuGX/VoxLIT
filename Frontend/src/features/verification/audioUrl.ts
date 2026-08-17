import { API_BASE } from "@/lib/api";

/**
 * Resolve a Speaker Verification recording id (opaque demo `rec_...` id or
 * session-scoped `asset_...` id) to its streamable audio URL. Both routes
 * are session (`sid` cookie) scoped -- callers must pass
 * `requireCredentials` to `WaveformViewer` when using this URL.
 */
export function verificationAudioUrl(recordingId: string): string {
  const isSessionAsset = recordingId.startsWith("asset_");
  return isSessionAsset
    ? `${API_BASE}/tasks/verification/session-assets/${encodeURIComponent(recordingId)}/audio`
    : `${API_BASE}/tasks/verification/dataset/recordings/${encodeURIComponent(recordingId)}/audio`;
}
