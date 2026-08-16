"use client"

import React, { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Checkbox } from "@/components/ui/checkbox"
import { RangeSlider } from "@/components/ui/range-slider"
import { Volume2, Scissors, Plus, Play, Zap, Loader2, CheckCircle, XCircle } from "lucide-react"
import { WaveformViewer } from "../audio/WaveformViewer"
import { API_BASE } from '@/lib/api'
import type { SessionAssetMetadata } from '@/tasks/types'

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

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

// Speaker Verification perturbation mode — a distinct explanation surface
// from the generic perturb-and-replay flow above. It runs exactly one
// perturbation against exactly one currently selected recording (a demo
// `rec_...` or session `asset_...` id), and the backend always registers the
// perturbed clip as a new session asset in the same response.
export interface PerturbationApiResponse {
  model: string;
  model_label: string;
  threshold: number;
  perturbation: { type: string; params: Record<string, number> };
  similarity: number;
  same_speaker: boolean;
  embedding_shift: number;
  source_recording_id: string;
  session_asset: SessionAssetMetadata;
}

export interface VerificationPerturbationContext {
  model: string;
  /** The single currently selected recording (table row or embedding-graph
   *  point) — a demo `rec_...` or session `asset_...` id. Null if none selected. */
  selectedRecordingId: string | null;
  /** Display label for the selected recording, for UI purposes only. */
  selectedRecordingLabel: string | null;
  /** Called after a successful perturbation so the caller can append the new
   *  session asset to shared state and make it the active recording. */
  onPerturbationApplied: (result: PerturbationApiResponse) => void;
}

interface PerturbationToolsProps {
  selectedFile: UploadedFile | null;
  onPerturbationComplete?: (result: PerturbationResult) => void;
  onPredictionRefresh?: (file: UploadedFile, prediction: string) => void;
  model?: string;
  dataset?: string;
  originalDataset?: string;
  /** Speaker Verification perturbation mode. When set, this component runs
   *  the single-recording /tasks/verification/perturbation flow instead of
   *  the generic /perturb flow, and every other task's rendering is unaffected. */
  verification?: VerificationPerturbationContext;
}

// Helper function to generate correct audio URL for original files
const getAudioUrl = (selectedFile: UploadedFile, dataset?: string, originalDataset?: string): string => {
  
  // Check if this is an uploaded file
  const isUploadedFile = selectedFile.file_path && (
    selectedFile.file_path.includes('uploads/') || 
    selectedFile.file_path.includes('uploads\\') ||
    selectedFile.message === "Perturbed file" ||
    selectedFile.message === "File uploaded successfully" ||
    selectedFile.message === "File uploaded and processed successfully"
  ) && selectedFile.message !== "Selected from dataset";
  
  if (isUploadedFile) {
    // For uploaded files, use the file_id (unique filename) with upload endpoint
    const url = `${API_BASE}/upload/file/${selectedFile.file_id}`;
    return url;
  } else {
    // For dataset files, use the filename with dataset endpoint
    // Use original dataset if available and it's a real dataset
    const datasetToUse = originalDataset && originalDataset !== "custom" ? originalDataset : dataset;
    
    if (datasetToUse && datasetToUse !== "custom") {
      // This is a dataset file from built-in or custom datasets
      const filename = encodeURIComponent(selectedFile.filename);
      
      // Handle custom datasets vs built-in datasets
      if (datasetToUse.startsWith('custom:')) {
        // Custom dataset: use the original route /{dataset}/file/{filename}
        const url = `${API_BASE}/${encodeURIComponent(datasetToUse)}/file/${filename}`;
        return url;
      } else {
        // Built-in dataset: use /{dataset}/file/{filename}
        const url = `${API_BASE}/${encodeURIComponent(datasetToUse)}/file/${filename}`;
        return url;
      }
    } else {
      // Fallback to upload endpoint when dataset is "custom" (generic case)
      const url = `${API_BASE}/upload/file/${selectedFile.file_id}`;
      return url;
    }
  }
};

// Helper function to generate correct audio URL for perturbed files
const getPerturbedAudioUrl = (perturbedFilePath: string): string => {
  // Extract filename from path (handle both forward and backward slashes)
  const filename = perturbedFilePath.split('/').pop() || perturbedFilePath.split('\\').pop();
  return `${API_BASE}/upload/file/${filename}`;
};

export const PerturbationTools: React.FC<PerturbationToolsProps> = ({
  selectedFile,
  onPerturbationComplete,
  onPredictionRefresh,
  model,
  dataset,
  originalDataset,
  verification,
}) => {
  // Perturbation parameters
  const [noiseLevel, setNoiseLevel] = useState([10])
  const [maskRange, setMaskRange] = useState<[number, number]>([20, 40])
  const [pitchShift, setPitchShift] = useState([2])
  const [timeStretch, setTimeStretch] = useState([110]) // 110% = 1.1x
  
  // Perturbation selection
  const [selectedPerturbations, setSelectedPerturbations] = useState({
    noise: false,
    timeMasking: false,
    pitchShift: false,
    timeStretch: false,
  })
  
  // State management
  const [isLoading, setIsLoading] = useState(false)
  const [perturbationResult, setPerturbationResult] = useState<PerturbationResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Verification-perturbation-mode state (unused unless `verification` is set)
  const [verificationResult, setVerificationResult] = useState<PerturbationApiResponse | null>(null)
  const [verificationError, setVerificationError] = useState<string | null>(null)
  const [isRunningVerificationPerturbation, setIsRunningVerificationPerturbation] = useState(false)
  const verificationAbortRef = React.useRef<AbortController | null>(null)

  // Clear perturbation results when selected file changes
  useEffect(() => {
    setPerturbationResult(null);
    setError(null);
  }, [selectedFile]);

  const activeRobustnessType: "noise" | "time_masking" | "pitch_shift" | "time_stretch" | null =
    !verification
      ? null
      : selectedPerturbations.noise
      ? "noise"
      : selectedPerturbations.timeMasking
      ? "time_masking"
      : selectedPerturbations.pitchShift
      ? "pitch_shift"
      : selectedPerturbations.timeStretch
      ? "time_stretch"
      : null;

  const activeRobustnessParamValue =
    activeRobustnessType === "noise"
      ? noiseLevel[0]
      : activeRobustnessType === "time_masking"
      ? `${maskRange[0]}-${maskRange[1]}`
      : activeRobustnessType === "pitch_shift"
      ? pitchShift[0]
      : activeRobustnessType === "time_stretch"
      ? timeStretch[0]
      : null;

  const verificationDepsKey = verification
    ? [verification.model, verification.selectedRecordingId ?? ""].join("|")
    : "";

  // Any change to the model, selected source, perturbation type, or its
  // parameter value invalidates the current result — abort any in-flight
  // request and clear stale state.
  useEffect(() => {
    if (!verification) return;
    if (verificationAbortRef.current) {
      verificationAbortRef.current.abort();
      verificationAbortRef.current = null;
    }
    setVerificationResult(null);
    setVerificationError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [verification, verificationDepsKey, activeRobustnessType, activeRobustnessParamValue]);

  const verificationValidationMessage = (): string | null => {
    if (!verification) return null;
    if (!verification.selectedRecordingId) {
      return "Select a recording from the Audio Dataset table or the embedding graph to perturb.";
    }
    if (!activeRobustnessType) {
      return "Select exactly one perturbation type.";
    }
    if (activeRobustnessType === "noise" && noiseLevel[0] <= 0) {
      return "Noise level must be greater than 0%.";
    }
    if (activeRobustnessType === "time_masking" && maskRange[0] >= maskRange[1]) {
      return "Mask start must be before mask end.";
    }
    if (activeRobustnessType === "pitch_shift" && pitchShift[0] === 0) {
      return "Choose a non-zero pitch shift.";
    }
    if (activeRobustnessType === "time_stretch" && timeStretch[0] === 100) {
      return "Choose a stretch factor other than 100%.";
    }
    return null;
  };

  const buildVerificationParams = (): Record<string, number> | null => {
    if (activeRobustnessType === "noise") return { noise_level: noiseLevel[0] / 100 };
    if (activeRobustnessType === "time_masking") return { mask_start_percent: maskRange[0], mask_end_percent: maskRange[1] };
    if (activeRobustnessType === "pitch_shift") return { pitch_shift_semitones: pitchShift[0] };
    if (activeRobustnessType === "time_stretch") return { stretch_factor: timeStretch[0] / 100 };
    return null;
  };

  const runVerificationPerturbation = async () => {
    if (!verification) return;
    const validationMessage = verificationValidationMessage();
    if (validationMessage) {
      setVerificationError(validationMessage);
      return;
    }
    const params = buildVerificationParams();
    if (!activeRobustnessType || !params || !verification.selectedRecordingId) return;

    if (verificationAbortRef.current) {
      verificationAbortRef.current.abort();
    }
    const controller = new AbortController();
    verificationAbortRef.current = controller;

    setIsRunningVerificationPerturbation(true);
    setVerificationError(null);
    setVerificationResult(null);

    try {
      const response = await fetch(`${API_BASE}/tasks/verification/perturbation`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: verification.model,
          recording_id: verification.selectedRecordingId,
          perturbation: { type: activeRobustnessType, params },
        }),
        signal: controller.signal,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Perturbation failed (${response.status}).`);
      }
      const result = payload as PerturbationApiResponse;
      setVerificationResult(result);
      verification.onPerturbationApplied(result);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setVerificationError(caught instanceof Error ? caught.message : "Perturbation failed.");
    } finally {
      setIsRunningVerificationPerturbation(false);
      if (verificationAbortRef.current === controller) {
        verificationAbortRef.current = null;
      }
    }
  };

  const handlePerturbationToggle = (perturbationType: keyof typeof selectedPerturbations) => {
    setSelectedPerturbations(prev => {
      if (verification) {
        // Exactly one perturbation is allowed per request.
        const next = { noise: false, timeMasking: false, pitchShift: false, timeStretch: false };
        next[perturbationType] = !prev[perturbationType];
        return next;
      }
      const newState = {
        ...prev,
        [perturbationType]: !prev[perturbationType]
      };
      return newState;
    });
  }

  const handleAddPerturbations = async () => {
    if (!selectedFile) {
      setError("No file selected");
      return;
    }

    if (!Object.values(selectedPerturbations).some(Boolean)) {
      setError("Please select at least one perturbation type");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const perturbations = [];
      
      if (selectedPerturbations.noise) {
        perturbations.push({
          type: "noise",
          params: {
            noise_level: noiseLevel[0] / 100.0 // Convert percentage to decimal
          }
        });
      }
      
      if (selectedPerturbations.timeMasking) {
        perturbations.push({
          type: "time_masking",
          params: {
            mask_start_percent: maskRange[0],
            mask_end_percent: maskRange[1]
          }
        });
      }
      
      if (selectedPerturbations.pitchShift) {
        perturbations.push({
          type: "pitch_shift",
          params: {
            pitch_shift_semitones: pitchShift[0]
          }
        });
      }
      
      if (selectedPerturbations.timeStretch) {
        perturbations.push({
          type: "time_stretch",
          params: {
            stretch_factor: timeStretch[0] / 100.0 // Convert percentage to decimal
          }
        });
      }

      // Determine if this is an uploaded file or dataset file
      const isUploadedFile = selectedFile.file_path && (
        selectedFile.file_path.includes('uploads/') || 
        selectedFile.file_path.startsWith('uploads/') ||
        selectedFile.message === "Perturbed file" ||
        selectedFile.message === "File uploaded successfully" ||
        selectedFile.message === "File uploaded and processed successfully"
      ) && selectedFile.message !== "Selected from dataset";

      const reqBody = {
        perturbations: perturbations,
        file_path: isUploadedFile ? selectedFile.file_path : selectedFile.filename,
        ...(isUploadedFile ? {} : {
          dataset: originalDataset || dataset
        })
      };

      const response = await fetch(`${API_BASE}/perturb`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(reqBody),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const result = await response.json();
      setPerturbationResult(result);
      
      // Notify parent component
      if (onPerturbationComplete) {
        onPerturbationComplete(result);
      }
      
      // Auto-refresh prediction for the perturbed file
      if (result.success && model && onPredictionRefresh) {
        try {
          
          // Create a file object for the perturbed file
          const perturbedFile: UploadedFile = {
            file_id: result.filename,
            filename: result.filename,
            file_path: result.perturbed_file,
            message: "Perturbed file",
            duration: result.duration_ms / 1000,
            sample_rate: result.sample_rate
          };
          
          
          // Run inference on the perturbed file

          const inferenceResponse = await fetch(`${API_BASE}/inferences/run`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              model: model,
              file_path: result.perturbed_file
            }),
          });
          
          if (inferenceResponse.ok) {
            const prediction = await inferenceResponse.json();
            const predictionText = typeof prediction === 'string' ? prediction : prediction?.text || JSON.stringify(prediction);
            onPredictionRefresh(perturbedFile, predictionText);
          } else {
            const errorText = await inferenceResponse.text();
            console.error("Error during auto-prediction:", errorText);
          }
        } catch (error) {
          console.error("DEBUG: Error running auto-prediction:", error);
        }
      } else {
        console.log("DEBUG: Skipping auto-prediction - success:", result.success, "model:", model, "callback:", !!onPredictionRefresh);
      }
      
      console.log("Perturbation response:", result);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error occurred";
      setError(errorMessage);
      console.error("Error adding perturbations:", err);
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <div className="space-y-4">
      {/* Error Display */}
      {(verification ? verificationError : error) && (
        <Card className="border-destructive/20 bg-destructive/5">
          <CardContent className="pt-4">
            <div className="flex items-center gap-2 text-destructive text-xs">
              <XCircle className="h-4 w-4" />
              {verification ? verificationError : error}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Speaker Verification perturbation context — status/help text lives
          here, scoped to this component's verification branch. */}
      {verification && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Speaker Verification Perturbation</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-xs">
            <p className="text-muted-foreground">
              Applies one perturbation to the selected recording and re-extracts its embedding
              to compare against the original. The perturbed clip is saved as a new recording you
              can select, play, and perturb again.
            </p>
            <div>
              <span className="font-medium">Selected recording: </span>
              {verification.selectedRecordingId
                ? (verification.selectedRecordingLabel ?? verification.selectedRecordingId)
                : "none selected — click a row in the Audio Dataset table or a point in the embedding graph."}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Perturbation Configuration */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Perturbation Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Noise Perturbation */}
          <div className="space-y-3 p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="noise-checkbox"
                checked={selectedPerturbations.noise}
                onCheckedChange={() => handlePerturbationToggle('noise')}
                className="border-blue-400 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600"
              />
              <Volume2 className="h-4 w-4 text-blue-600" />
              <label htmlFor="noise-checkbox" className="text-sm font-medium">
                Add Gaussian Noise
              </label>
            </div>
            
            {selectedPerturbations.noise && (
              <div className="space-y-2 pl-6">
                <div className="flex items-center justify-between">
                  <span className="text-xs">Noise Level</span>
                  <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">{noiseLevel[0]}%</Badge>
                </div>
                <Slider 
                  value={noiseLevel} 
                  onValueChange={setNoiseLevel} 
                  max={50} 
                  step={1} 
                  className="w-full [&_[role=slider]]:border-blue-500 [&_[role=slider]]:bg-blue-600" 
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>0%</span>
                  <span>25%</span>
                  <span>50%</span>
                </div>
              </div>
            )}
          </div>

          {/* Time Masking Perturbation */}
          <div className="space-y-3 p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="masking-checkbox"
                checked={selectedPerturbations.timeMasking}
                onCheckedChange={() => handlePerturbationToggle('timeMasking')}
                className="border-blue-400 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600"
              />
              <Scissors className="h-4 w-4 text-blue-600" />
              <label htmlFor="masking-checkbox" className="text-sm font-medium">
                Apply Time Masking
              </label>
            </div>
            
            {selectedPerturbations.timeMasking && (
              <div className="space-y-3 pl-6">
                <div className="flex items-center justify-between">
                  <span className="text-xs">Mask Region</span>
                  <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">
                    {maskRange[0]}% - {maskRange[1]}%
                  </Badge>
                </div>
                
                {/* Visual representation of time mask */}
                <div className="relative h-8 bg-gray-100 rounded border">
                  <div 
                    className="absolute top-0 h-full bg-blue-200 rounded transition-all duration-300 border border-blue-400"
                    style={{
                      left: `${maskRange[0]}%`,
                      width: `${maskRange[1] - maskRange[0]}%`
                    }}
                  >
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-xs font-medium text-blue-800">
                        Masked
                      </span>
                    </div>
                  </div>
                  <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
                    Audio Timeline
                  </div>
                </div>
                
                {/* Dual-pointer range slider */}
                <div className="space-y-2">
                  <RangeSlider
                    value={maskRange}
                    onValueChange={setMaskRange}
                    min={0}
                    max={100}
                    step={1}
                    className="w-full"
                    formatLabel={(value) => `${value}%`}
                  />
                </div>
                
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>0% (Start)</span>
                  <span>50% (Middle)</span>
                  <span>100% (End)</span>
                </div>
              </div>
            )}
          </div>

          {/* Pitch Shift Perturbation */}
          <div className="space-y-3 p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="pitch-checkbox"
                checked={selectedPerturbations.pitchShift}
                onCheckedChange={() => handlePerturbationToggle('pitchShift')}
                className="border-blue-400 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600"
              />
              <Plus className="h-4 w-4 text-blue-600" />
              <label htmlFor="pitch-checkbox" className="text-sm font-medium">
                Apply Pitch Shift
              </label>
            </div>
            
            {selectedPerturbations.pitchShift && (
              <div className="space-y-2 pl-6">
                <div className="flex items-center justify-between">
                  <span className="text-xs">Pitch Shift</span>
                  <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">
                    {pitchShift[0] > 0 ? "+" : ""}{pitchShift[0]} semitones
                  </Badge>
                </div>
                <Slider 
                  value={pitchShift} 
                  onValueChange={setPitchShift} 
                  min={-6} 
                  max={6} 
                  step={1} 
                  className="w-full [&_[role=slider]]:border-blue-500 [&_[role=slider]]:bg-blue-600"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>-6</span>
                  <span>0</span>
                  <span>+6</span>
                </div>
              </div>
            )}
          </div>

          {/* Time Stretch Perturbation - Hidden for Whisper models */}
          {!model?.includes('whisper') && (
            <div className="space-y-3 p-3 border rounded-lg">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="time-checkbox"
                  checked={selectedPerturbations.timeStretch}
                  onCheckedChange={() => handlePerturbationToggle('timeStretch')}
                  className="border-blue-400 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600"
                />
                <Play className="h-4 w-4 text-blue-600" />
                <label htmlFor="time-checkbox" className="text-sm font-medium">
                  Apply Time Stretch
                </label>
              </div>
              
              {selectedPerturbations.timeStretch && (
                <div className="space-y-2 pl-6">
                  <div className="flex items-center justify-between">
                    <span className="text-xs">Time Stretch</span>
                    <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">
                      {timeStretch[0]}% {timeStretch[0] < 100 ? "(Faster)" : timeStretch[0] > 100 ? "(Slower)" : "(Normal)"}
                    </Badge>
                  </div>
                  <Slider 
                    value={timeStretch} 
                    onValueChange={setTimeStretch} 
                    min={50} 
                    max={200} 
                    step={5} 
                    className="w-full [&_[role=slider]]:border-blue-500 [&_[role=slider]]:bg-blue-600" 
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>50% (2x faster)</span>
                    <span>100%</span>
                    <span>200% (2x slower)</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Apply Perturbations / Run Perturbation Button */}
      <Card>
        <CardContent className="pt-4">
          {verification ? (
            <>
              <Button
                onClick={runVerificationPerturbation}
                disabled={isRunningVerificationPerturbation || !!verificationValidationMessage()}
                className="w-full h-10 bg-blue-600 hover:bg-blue-700 text-white font-medium shadow-md"
                size="lg"
              >
                {isRunningVerificationPerturbation ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Zap className="h-4 w-4 mr-2" />
                )}
                {isRunningVerificationPerturbation ? "Applying perturbation…" : "Run Perturbation"}
              </Button>
              <p className="text-xs text-muted-foreground mt-2 text-center">
                {verificationValidationMessage() ?? "Ready to run the perturbation."}
              </p>
            </>
          ) : (
            <>
              <Button
                onClick={handleAddPerturbations}
                disabled={isLoading || !selectedFile || !Object.values(selectedPerturbations).some(Boolean)}
                className="w-full h-10 bg-blue-600 hover:bg-blue-700 text-white font-medium shadow-md"
                size="lg"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Zap className="h-4 w-4 mr-2" />
                )}
                {isLoading ? "Applying Perturbations..." : "Apply Perturbations"}
              </Button>
              <p className="text-xs text-muted-foreground mt-2 text-center">
                {!selectedFile
                  ? "Select a file to apply perturbations"
                  : !Object.values(selectedPerturbations).some(Boolean)
                  ? "Select at least one perturbation type"
                  : `Apply perturbations to ${selectedFile.filename}`
                }
              </p>
            </>
          )}
        </CardContent>
      </Card>

      {/* Perturbation Result — Speaker Verification only. The perturbed clip
          is already saved as a new session asset (see the status card above /
          the auto-selected recording); this card never shows a
          confidence/probability framing, only the raw similarity/threshold. */}
      {verification && verificationResult && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              Perturbation Result
              <Badge variant={verificationResult.same_speaker ? "secondary" : "destructive"} className="text-[10px]">
                {verificationResult.same_speaker ? "Same speaker" : "Different speaker"}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-xs">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <span className="text-muted-foreground">Model</span>
                <div className="font-medium">{verificationResult.model_label}</div>
              </div>
              <div>
                <span className="text-muted-foreground">Perturbation</span>
                <div className="font-medium">
                  {verificationResult.perturbation.type.replace("_", " ")} —{" "}
                  {Object.entries(verificationResult.perturbation.params)
                    .map(([key, value]) => `${key}: ${value}`)
                    .join(", ")}
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <span className="text-muted-foreground">Cosine similarity</span>
                <div className="font-mono font-medium">{verificationResult.similarity.toFixed(4)}</div>
              </div>
              <div>
                <span className="text-muted-foreground">Model threshold</span>
                <div className="font-mono font-medium">{verificationResult.threshold.toFixed(4)}</div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 border-t pt-2">
              <div>
                <span className="text-muted-foreground">Embedding shift</span>
                <div className="font-mono font-medium">{verificationResult.embedding_shift.toFixed(4)}</div>
              </div>
              <div>
                <span className="text-muted-foreground">New recording</span>
                <div className="font-medium truncate">{verificationResult.session_asset.display_filename}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Waveform Visualization */}
      {!verification && (selectedFile || perturbationResult) && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Audio Waveforms</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {/* Original Audio Waveform */}
            {selectedFile && (
              <div className="space-y-2">
                <div className="text-xs font-medium flex items-center gap-2">
                  Original Audio
                  <Badge variant="outline" className="text-[10px]">O</Badge>
                </div>
                <WaveformViewer 
                  audioUrl={getAudioUrl(selectedFile, dataset, originalDataset)}
                />
              </div>
            )}

            {/* Perturbed Audio Waveform */}
            {perturbationResult && perturbationResult.success && (
              <div className="space-y-2">
                <div className="text-xs font-medium flex items-center gap-2">
                  Perturbed Audio
                  <Badge variant="secondary" className="text-[10px]">P</Badge>
                </div>
                <WaveformViewer 
                  audioUrl={getPerturbedAudioUrl(perturbationResult.perturbed_file)}
                />
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {!verification && perturbationResult && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              Perturbation Results
              {perturbationResult.success ? (
                <CheckCircle className="h-4 w-4 text-green-500" />
              ) : (
                <XCircle className="h-4 w-4 text-red-500" />
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xs space-y-2">
              <div className="font-medium">Applied Perturbations:</div>
              {perturbationResult.applied_perturbations.map((perturbation, idx) => (
                <div key={idx} className="p-2 bg-muted/50 rounded space-y-1">
                  <div className="flex items-center justify-between">
                    <Badge 
                      variant={perturbation.status === "applied" ? "default" : "destructive"} 
                      className="text-[10px]"
                    >
                      {perturbation.type.replace("_", " ")}
                    </Badge>
                    <span className={`text-[10px] ${
                      perturbation.status === "applied" ? "text-green-600" : "text-red-600"
                    }`}>
                      {perturbation.status}
                    </span>
                  </div>
                  {perturbation.status === "failed" && perturbation.error && (
                    <div className="text-[10px] text-red-500">
                      Error: {perturbation.error}
                    </div>
                  )}
                </div>
              ))}
              
              <div className="mt-3 p-2 bg-blue-50 rounded border border-blue-200">
                <div className="text-xs font-medium text-blue-800 mb-1">Generated File</div>
                <div className="text-xs text-blue-700">
                  {perturbationResult.filename}
                </div>
                <div className="text-xs text-blue-600">
                  Duration: {(perturbationResult.duration_ms / 1000).toFixed(2)}s
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
