import { useState, useRef, useEffect, ReactNode } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { AudioPlayer } from "../audio/AudioPlayer";
import { WaveformViewer } from "../audio/WaveformViewer";
import { Play, Pause, RotateCcw, Trash2, Plus, HelpCircle } from "lucide-react";
import WaveSurfer from "wavesurfer.js";
import { API_BASE } from '@/lib/api';
import { UploadedFile, DatasetRecordingRef, LocalFilePreview } from '@/tasks/types';
import { isVerificationDemoDataset } from '@/tasks/registry';
import { verificationAudioUrl } from '@/features/verification/audioUrl';

interface PerturbationResult {
  perturbed_file: string;
  filename: string;
  duration_ms: number;
  sample_rate: number;
  applied_perturbations: Array<{
    type: string;
    params: Record<string, any>;
    status: string;
    error?: string;
  }>;
  success: boolean;
  error?: string;
}

interface DatapointEditorPanelProps {
  selectedFile?: UploadedFile | null;
  selectedEmbeddingFile?: string | null;
  dataset?: string; // "custom" | dataset key (effective dataset)
  originalDataset?: string; // Original dataset selection from toolbar
  perturbationResult?: PerturbationResult | null;
  predictionMap?: Record<string, string>;
  /** Speaker Verification only: the safe demo-dataset recording list, used to
   *  resolve a selected opaque recording_id to its display_filename/extension/size. */
  datasetRecordings?: DatasetRecordingRef[] | null;
  /** Speaker Verification only: a temporary local device-file preview (Pair
   *  Verification enrollment/probe upload). Takes priority over selectedFile
   *  when set — never a real backend id, never resolved through any backend
   *  audio endpoint. */
  localPreview?: LocalFilePreview | null;
  /**
   * Task-specific results card (from the registry's TASK_SLOTS), rendered
   * between Sample Info and Audio Playback. Receives the current
   * original/perturbed toggle state. Omitted for placeholder tasks.
   */
  renderPredictionResults?: (showPerturbed: boolean) => ReactNode;
}

export const DatapointEditorPanel = ({
  selectedFile,
  selectedEmbeddingFile,
  dataset = "custom",
  originalDataset,
  perturbationResult,
  predictionMap,
  renderPredictionResults,
  datasetRecordings,
  localPreview,
}: DatapointEditorPanelProps) => {
  const [selectedLabel, setSelectedLabel] = useState<string>("neutral");
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [showPerturbed, setShowPerturbed] = useState(false);
  const wavesurferRef = useRef<WaveSurfer | null>(null);

  // Speaker Verification's demo dataset serves audio only through a
  // task-specific, opaque-id-only endpoint (ground-truth safety) — never the
  // generic /{dataset}/file/... route, which would require the real filename.
  const isDemoDatasetPlayback = isVerificationDemoDataset(originalDataset) || isVerificationDemoDataset(dataset);

  // Resolve the selected opaque recording_id against the safe recording list
  // already returned by GET /tasks/verification/dataset/recordings — never
  // fetches anything itself, and never exposes ground truth.
  const demoRecording = isDemoDatasetPlayback && selectedFile
    ? datasetRecordings?.find((r) => r.recording_id === selectedFile.file_id) ?? null
    : null;

  const audioUrl = (() => {
    // Temporary local device file (Speaker Verification Pair Verification
    // upload): play directly from its local blob URL, bypassing every
    // backend URL branch below entirely.
    if (localPreview) {
      return localPreview.previewUrl;
    }

    // Demo-dataset recordings: stream by opaque recording_id only, through
    // the task-specific audio endpoint. Falls through to undefined (safe,
    // same as "no file selected") if the id hasn't resolved against the
    // loaded recording list yet.
    if (isDemoDatasetPlayback) {
      if (!demoRecording) return undefined;
      return verificationAudioUrl(demoRecording.recording_id);
    }

    // If showing perturbed audio and it's available
    if (showPerturbed && perturbationResult?.success) {
      const filename = perturbationResult.filename;
      return `${API_BASE}/upload/file/${filename}`;
    }
    
    // Otherwise show original audio
    if (!selectedFile) return undefined;
    
    // Check if this is an uploaded file - more precise detection
    const isUploadedFile = selectedFile.file_path && (
      selectedFile.file_path.includes('uploads/') || 
      selectedFile.file_path.startsWith('uploads/') ||
      selectedFile.message === "Perturbed file" ||
      selectedFile.message === "File uploaded successfully" ||
      selectedFile.message === "File uploaded and processed successfully"
    ) && selectedFile.message !== "Selected from embeddings" && selectedFile.message !== "Selected from dataset";
    
    if (isUploadedFile) {
      // This is an uploaded file, use the upload endpoint
      return `${API_BASE}/upload/file/${selectedFile.file_id}`;
    }
    
    // For dataset files (including files selected from embeddings)
    // Use original dataset if available and it's a real dataset
    const datasetToUse = originalDataset && originalDataset !== "custom" ? originalDataset : dataset;
    
    if (datasetToUse && datasetToUse !== "custom") {
      // This is a dataset file from built-in or custom datasets
      const filename = encodeURIComponent(selectedFile.filename);
      
      // Handle custom datasets vs built-in datasets
      if (datasetToUse.startsWith('custom:')) {
        // Custom dataset: use the original route /{dataset}/file/{filename}
        // The backend handles the custom dataset format properly
        return `${API_BASE}/${encodeURIComponent(datasetToUse)}/file/${filename}`;
      } else {
        // Built-in dataset: use /{dataset}/file/{filename}
        return `${API_BASE}/${encodeURIComponent(datasetToUse)}/file/${filename}`;
      }
    } else {
      // Fallback to upload endpoint when dataset is "custom" (generic case)
      return `${API_BASE}/upload/file/${selectedFile.file_id}`;
    }
  })();

  // Get current file info (original or perturbed) with better data handling
  const currentFileInfo = (() => {
    // Local device file: filename/extension/size come straight from the
    // browser File object; duration/sample_rate are left undefined here so
    // the render falls through to audioMetadata, populated once WaveSurfer
    // decodes the local blob URL — same fallback chain as demo recordings.
    if (localPreview) {
      const dotIndex = localPreview.file.name.lastIndexOf(".");
      return {
        filename: localPreview.file.name,
        duration: undefined,
        sample_rate: undefined,
        size: localPreview.file.size,
        extension: dotIndex >= 0 ? localPreview.file.name.slice(dotIndex + 1) : undefined,
      };
    }

    if (showPerturbed && perturbationResult?.success) {
      return {
        filename: perturbationResult.filename,
        duration: perturbationResult.duration_ms / 1000,
        sample_rate: perturbationResult.sample_rate,
        size: undefined,
        extension: undefined,
      };
    }

    // Demo-dataset selections: filename/extension/size come from the safe
    // recording list; duration/sample rate are left undefined here so the
    // render falls through to audioMetadata, populated once WaveSurfer
    // decodes the audio loaded from the task-specific audio endpoint.
    if (demoRecording) {
      return {
        filename: demoRecording.display_filename,
        duration: undefined,
        sample_rate: undefined,
        size: demoRecording.size_bytes,
        extension: demoRecording.extension,
      };
    }

    // For original file, try to get the most accurate data
    if (selectedFile) {
      return {
        filename: selectedFile.filename,
        duration: selectedFile.duration || undefined,
        sample_rate: selectedFile.sample_rate || undefined,
        size: selectedFile.size || undefined,
        extension: undefined,
      };
    }

    return null;
  })();

  // Add a state to track audio metadata from wavesurfer
  const [audioMetadata, setAudioMetadata] = useState<{
    duration?: number;
    sampleRate?: number;
  }>({});

  // Debug logging for selectedFile and audioUrl
  useEffect(() => {
  }, [selectedFile, audioUrl, dataset, originalDataset]);

  // Reset playback when file changes or when switching between original/perturbed
  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setAudioMetadata({}); // Reset metadata when file changes
    
    // Reset wavesurfer instance if it exists
    if (wavesurferRef.current) {
      wavesurferRef.current.stop();
    }
  }, [selectedFile?.file_id, dataset, showPerturbed, perturbationResult?.filename, localPreview?.localId]);
  
  return (
    <TooltipProvider>
      <div className="h-full bg-panel-background border-l border-border flex flex-col">
        <div className="bg-panel-header p-3 border-b border-border">
          <h3 className="font-semibold text-sm text-foreground flex items-center gap-1.5">
            Datapoint Editor
            <Tooltip>
              <TooltipTrigger>
                <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
              </TooltipTrigger>
              <TooltipContent>
                Edit and analyze individual audio samples with predictions and perturbations
              </TooltipContent>
            </Tooltip>
          </h3>
        </div>

      <div className="flex-1 min-h-0 p-3 overflow-y-auto space-y-3 scrollbar-thin">
        {/* Sample Info - Top */}
        <Card>
          <CardHeader className="bg-panel-header">
            <div className="flex items-center justify-between">
              <CardTitle className="text-xs flex items-center gap-1.5">
                Sample Info
                <Tooltip>
                  <TooltipTrigger>
                    <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
                  </TooltipTrigger>
                  <TooltipContent>
                    Detailed information about the selected audio sample
                  </TooltipContent>
                </Tooltip>
              </CardTitle>
              {perturbationResult?.success && (
                <div className="flex items-center gap-0.5 p-0.5 bg-muted border border-border rounded-md">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant={!showPerturbed ? "default" : "ghost"}
                        size="sm"
                        onClick={() => setShowPerturbed(false)}
                        className={`text-[10px] h-6 px-2.5 transition-all ${
                          !showPerturbed
                            ? 'bg-primary hover:bg-primary-hover text-primary-foreground shadow-aws-sm'
                            : 'text-muted-foreground hover:bg-background'
                        }`}
                      >
                        Original
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      View the original unmodified audio file
                    </TooltipContent>
                  </Tooltip>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant={showPerturbed ? "default" : "ghost"}
                        size="sm"
                        onClick={() => setShowPerturbed(true)}
                        className={`text-[10px] h-6 px-2.5 transition-all ${
                          showPerturbed
                            ? 'bg-primary hover:bg-primary-hover text-primary-foreground shadow-aws-sm'
                            : 'text-muted-foreground hover:bg-background'
                        }`}
                      >
                        Perturbed
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent className="font-normal">
                      View the modified audio file with applied perturbations
                    </TooltipContent>
                  </Tooltip>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-1.5">
            {localPreview && (
              <div className="text-xs-tight">
                <span className="text-gray-500">Role:</span>
                <span className="ml-2 text-gray-700">
                  {localPreview.role === "enrollment" ? "Enrollment reference" : "Probe"}
                </span>
              </div>
            )}
            <div className="text-xs-tight">
              <span className="text-gray-500">File:</span>
              <span className="ml-2 font-mono text-gray-700">{currentFileInfo?.filename || "No file selected"}</span>
              {showPerturbed && (
                <Badge variant="secondary" className="ml-2 text-[9px] bg-blue-100 text-blue-700 border-blue-200">P</Badge>
              )}
            </div>
            <div className="text-xs-tight">
              <span className="text-gray-500">Duration:</span>
              <span className="ml-2 text-gray-700">
                {currentFileInfo?.duration
                  ? `${currentFileInfo.duration.toFixed(1)}s`
                  : audioMetadata.duration
                  ? `${audioMetadata.duration.toFixed(1)}s`
                  : !localPreview && isDemoDatasetPlayback && !demoRecording ? "Not available" : "Loading..."}
              </span>
            </div>
            <div className="text-xs-tight">
              <span className="text-gray-500">Sample Rate:</span>
              <span className="ml-2 text-gray-700">
                {currentFileInfo?.sample_rate
                  ? `${(currentFileInfo.sample_rate / 1000).toFixed(1)}kHz`
                  : audioMetadata.sampleRate
                  ? `${(audioMetadata.sampleRate / 1000).toFixed(1)}kHz`
                  : !localPreview && isDemoDatasetPlayback && !demoRecording ? "Not available" : "Loading..."}
              </span>
            </div>
            {currentFileInfo?.extension && (
              <div className="text-xs-tight">
                <span className="text-gray-500">Extension:</span>
                <span className="ml-2 text-gray-700">{currentFileInfo.extension}</span>
              </div>
            )}
            {currentFileInfo?.size && (
              <div className="text-xs-tight">
                <span className="text-gray-500">Size:</span>
                <span className="ml-2 text-gray-700">{(currentFileInfo.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>
            )}
            {showPerturbed && perturbationResult?.applied_perturbations && (
              <div className="text-xs-tight">
                <span className="text-gray-500">Applied:</span>
                <div className="ml-2 mt-1 space-y-1">
                  {perturbationResult.applied_perturbations.map((pert, idx) => (
                    <Badge key={idx} variant="outline" className="text-[9px] mr-1 border-blue-300 text-blue-700">
                      {pert.type.replace('_', ' ')}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            {showPerturbed && perturbationResult?.filename && predictionMap && (
              <div className="text-xs-tight mt-2">
                <span className="text-gray-500">Perturbed Prediction:</span>
                <div className="ml-2 mt-1">
                  <Badge variant="secondary" className="text-[10px] bg-blue-100 text-blue-700 border-blue-200">
                    {predictionMap[perturbationResult.filename] || "Loading..."}
                  </Badge>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Predictions Section - Middle (task-specific, from registry slot) */}
        {renderPredictionResults?.(showPerturbed)}

        {/* Audio Player & Waveform - Bottom */}
        <Card>
          <CardHeader className="bg-panel-header">
            <CardTitle className="text-xs flex items-center gap-1.5">
              Audio Playback
              <Tooltip>
                <TooltipTrigger>
                  <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
                </TooltipTrigger>
                <TooltipContent>
                  Interactive audio player with waveform visualization
                </TooltipContent>
              </Tooltip>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2.5">
            <WaveformViewer
              audioUrl={audioUrl}
              isPlaying={isPlaying}
              requireCredentials={!localPreview && isDemoDatasetPlayback}
              onReady={(wavesurfer) => {

                wavesurferRef.current = wavesurfer;
                const duration = wavesurfer.getDuration();
                setDuration(duration);

                // Update metadata state for file info display
                setAudioMetadata({
                  duration: duration,
                  sampleRate: wavesurfer.getDecodedData()?.sampleRate || undefined
                });
              }}
              onProgress={(time, dur) => {
                setCurrentTime(time);
                setDuration(dur);

                // Update duration in metadata if not already set
                if (!audioMetadata.duration && dur > 0) {
                  setAudioMetadata(prev => ({ ...prev, duration: dur }));
                }
              }}
              onFinish={() => {
                // Reset player when the clip ends: play icon + slider/cursor to start
                setIsPlaying(false);
                setCurrentTime(0);
                wavesurferRef.current?.seekTo(0);
              }}
            />
            <AudioPlayer
              isPlaying={isPlaying}
              onPlayPause={() => {
                setIsPlaying(!isPlaying);
                if (wavesurferRef.current) {
                  if (isPlaying) {
                    wavesurferRef.current.pause();
                  } else {
                    wavesurferRef.current.play();
                  }
                }
              }}
              currentTime={currentTime}
              duration={duration}
              onSeek={(time) => {
                if (wavesurferRef.current) {
                  wavesurferRef.current.seekTo(time / duration);
                }
              }}
              onVolumeChange={(volume) => {
                if (wavesurferRef.current) {
                  wavesurferRef.current.setVolume(volume);
                }
              }}
            />
          </CardContent>
        </Card>
      </div>
    </div>
    </TooltipProvider>
  );
};