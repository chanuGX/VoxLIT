import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { Toolbar } from "./Toolbar";
import { ExplainabilityPanel } from "./ExplainabilityPanel";
import { PlaceholderExplainability } from "./PlaceholderExplainability";
import { EmbeddingPanel } from "../panels/EmbeddingPanel";
import { AudioDatasetPanel } from "../panels/AudioDatasetPanel";
import { DatapointEditorPanel } from "../panels/DatapointEditorPanel";
import { EmbeddingProvider } from "../../contexts/EmbeddingContext";
import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { API_BASE } from '@/lib/api';
import {
  TaskDefinition,
  UploadedFile,
  Wav2Vec2Prediction,
  WhisperPrediction,
  DatasetRecordingRef,
  SessionAssetMetadata,
  LocalFilePreview,
} from '@/tasks/types';
import {
  TASK_SLOTS,
  getModelLabel,
  VERIFICATION_DEMO_DATASET_ID,
  VERIFICATION_CUSTOM_DATASET_PREFIX,
  VERIFICATION_CUSTOM_DATASET_STORAGE_KEY,
  isValidCustomDatasetName,
} from '@/tasks/registry';
import { toast } from 'sonner';

/** Reads the saved custom dataset name from sessionStorage, validating it
 *  locally first -- a malformed value is removed immediately rather than
 *  ever being used to construct a `verification-custom:` selector. */
function readSavedVerificationDataset(): string | null {
  const saved = sessionStorage.getItem(VERIFICATION_CUSTOM_DATASET_STORAGE_KEY);
  if (!saved) return null;
  if (!isValidCustomDatasetName(saved)) {
    sessionStorage.removeItem(VERIFICATION_CUSTOM_DATASET_STORAGE_KEY);
    return null;
  }
  return saved;
}

interface TaskWorkbenchProps {
  task: TaskDefinition;
}

/**
 * The shared three-column workbench used by every task page:
 * left = Audio Embeddings, center = task explainability + dataset table,
 * right = Datapoint Editor. Task-specific behavior (models, datasets,
 * capabilities, results card) comes entirely from the task registry.
 */
export const TaskWorkbench = ({ task }: TaskWorkbenchProps) => {
  const [apiData, setApiData] = useState<unknown>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<UploadedFile | null>(null);
  const [model, setModel] = useState(task.defaultModel ?? "");
  // Verification-only: if a custom dataset was saved from a previous visit,
  // the initial value is ALREADY the custom selector, synchronously, before
  // any effect runs -- this is what keeps the demo dataset's auto-select-all
  // effect (below) from ever firing during restoration, since `dataset`
  // never actually equals VERIFICATION_DEMO_DATASET_ID in that case.
  const [dataset, setDataset] = useState(() => {
    if (task.id === 'verification') {
      const saved = readSavedVerificationDataset();
      if (saved) return `${VERIFICATION_CUSTOM_DATASET_PREFIX}${saved}`;
    }
    return task.defaultDataset ?? "";
  });
  // Verification-only: true while a saved custom dataset selection is being
  // confirmed against the caller's actual owned dataset list.
  const [verificationRestoring, setVerificationRestoring] = useState(
    () => task.id === 'verification' && readSavedVerificationDataset() !== null
  );
  const [batchInferenceStatus, setBatchInferenceStatus] = useState<'idle' | 'running' | 'done'>('idle');
  const [availableFiles, setAvailableFiles] = useState<string[]>([]);
  const [selectedEmbeddingFile, setSelectedEmbeddingFile] = useState<string | null>(null);
  const [perturbationResult, setPerturbationResult] = useState<any>(null);

  // Speaker-Verification-only: the currently active local device-file
  // preview shown in the Datapoint Editor. Any backend-driven selection
  // (dataset row, graph point, new session asset) clears this so the most
  // recently clicked item always wins — see handleFileSelection,
  // handleEmbeddingSelection, handleVerificationAssetCreated below.
  const [localPreview, setLocalPreview] = useState<LocalFilePreview | null>(null);

  // Speaker Verification-only shared state (harmless/unused for every other
  // task, since nothing else ever sets these away from their initial values).
  const [pairSelectionLabels, setPairSelectionLabels] = useState<string[]>([]);
  const [uploadedRawFiles, setUploadedRawFiles] = useState<Record<string, File>>({});
  const [datasetRecordings, setDatasetRecordings] = useState<DatasetRecordingRef[] | null>(null);
  // Session-scoped assets (uploads + perturbation outputs), separate from
  // `datasetRecordings` (demo-only, owned by AudioDatasetPanel's safe-listing
  // effect) so neither source clobbers the other; merged for consumers below.
  const [sessionAssets, setSessionAssets] = useState<DatasetRecordingRef[]>([]);
  const [selectedBatchIds, setSelectedBatchIds] = useState<string[]>([]);
  const [reprojectFn, setReprojectFn] = useState<((method: string, n: number) => void) | null>(null);
  const [labelResolverFn, setLabelResolverFn] = useState<((label: string) => string | undefined) | null>(null);
  // Verification-only: bumped when files are uploaded into the currently
  // active custom dataset, so AudioDatasetPanel re-fetches its recordings
  // without `dataset`/`effectiveDataset` changing -- deliberately NOT part
  // of the key used to remount WorkbenchCenter/EmbeddingPanel below, since
  // adding files to the still-active dataset is not a source change.
  const [customDatasetRefreshToken, setCustomDatasetRefreshToken] = useState(0);

  // Fetch previously created session assets once on load so they survive a
  // page refresh — verification only.
  useEffect(() => {
    if (task.id !== 'verification') return;
    const ac = new AbortController();
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/tasks/verification/session-assets`, {
          credentials: 'include',
          signal: ac.signal,
        });
        if (!res.ok) return;
        const data = await res.json() as { assets: SessionAssetMetadata[] };
        setSessionAssets(
          data.assets.map((a) => ({
            recording_id: a.asset_id,
            display_filename: a.display_filename,
            extension: a.extension,
            size_bytes: a.size_bytes,
            origin: a.origin,
            duration_seconds: a.duration_seconds,
          }))
        );
      } catch (e) {
        const name = (e as { name?: string } | null)?.name;
        if (name !== 'AbortError') console.error(e);
      }
    })();
    return () => ac.abort();
  }, [task.id]);

  // Demo recordings (safe-listed by AudioDatasetPanel) plus session assets —
  // the merged "safe recording" list passed to any consumer that needs to
  // resolve an opaque id (rec_... or asset_...) to a display label/size, or
  // to play it back. Verification-only; empty/null for every other task.
  const verificationRecordings = useMemo(() => {
    const merged = [...(datasetRecordings ?? []), ...sessionAssets];
    // Defense-in-depth dedup -- crec_/rec_/asset_ are disjoint namespaces so
    // this should never actually trigger, but guarantees one entry per id
    // regardless.
    return Array.from(new Map(merged.map((r) => [r.recording_id, r])).values());
  }, [datasetRecordings, sessionAssets]);

  // Real "N uploaded" count for the top-bar-driven Upload button — counts
  // only session assets whose origin is an actual upload, never the 92
  // built-in demo recordings or perturbation-generated assets.
  const uploadedCount = useMemo(
    () => sessionAssets.filter((a) => a.origin === 'upload').length,
    [sessionAssets]
  );

  // Appends a newly created session asset (top-bar upload or a perturbation
  // result) to shared state and makes it the selected/active recording, so
  // it's immediately visible in the table and playable in the Datapoint
  // Editor without a refresh or manual import.
  const handleVerificationAssetCreated = useCallback((asset: SessionAssetMetadata) => {
    setLocalPreview(null);
    const ref: DatasetRecordingRef = {
      recording_id: asset.asset_id,
      display_filename: asset.display_filename,
      extension: asset.extension,
      size_bytes: asset.size_bytes,
      origin: asset.origin,
      duration_seconds: asset.duration_seconds,
    };
    setSessionAssets((prev) => [...prev, ref]);
    const fileLike: UploadedFile = {
      file_id: ref.recording_id,
      filename: ref.display_filename,
      file_path: ref.display_filename,
      message: "Selected from dataset",
    };
    setSelectedFile(fileLike);
    setSelectedEmbeddingFile(ref.recording_id);
  }, []);

  // useState's setter treats a bare function argument as a functional
  // updater, so storing a callback value itself must go through the wrapper
  // form — never setReprojectFn(handler) directly, since handler may itself
  // be a function.
  const registerReprojectHandler = useCallback((handler: ((method: string, n: number) => void) | null) => {
    setReprojectFn(() => handler);
  }, []);
  const registerLabelResolver = useCallback((resolver: ((label: string) => string | undefined) | null) => {
    setLabelResolverFn(() => resolver);
  }, []);

  // Whenever any batch-defining input changes, the current pair selection is
  // no longer meaningful — clear it. (batchResult/projection clearing is
  // handled by BatchAnalysisPanel itself, which is a descendant of
  // EmbeddingProvider and can reach setEmbeddingDataDirect; TaskWorkbench
  // itself renders that provider and cannot consume its own context.)
  useEffect(() => {
    setPairSelectionLabels([]);
  }, [model, dataset, selectedBatchIds, uploadedFiles]);

  // Auto-select every loaded demo recording for batch input, once, the first
  // time the demo dataset's recordings arrive with nothing already checked —
  // never re-fires afterward, so it never clobbers a user's own selection.
  const verificationAutoSelectRef = useRef(false);
  useEffect(() => {
    if (task.id !== 'verification') return;
    if (verificationAutoSelectRef.current) return;
    if (dataset !== VERIFICATION_DEMO_DATASET_ID) return;
    if (!datasetRecordings || datasetRecordings.length === 0) return;
    if (selectedBatchIds.length > 0) return;
    verificationAutoSelectRef.current = true;
    setSelectedBatchIds(datasetRecordings.map((r) => r.recording_id));
  }, [task.id, dataset, datasetRecordings, selectedBatchIds]);

  // Verification-only: confirm a restored custom-dataset selection against
  // the caller's actual owned dataset list. The whole fetch-and-parse flow
  // is inside one try/catch, and `ac.signal.aborted` is checked before every
  // subsequent state update (not only via catching AbortError), since an
  // abort can race between `fetch()` resolving and `res.json()` running.
  // Only two outcomes ever clear the saved selection: a successful response
  // proving the dataset is genuinely absent, or a definitive non-5xx
  // failure (e.g. an auth/session problem). A 5xx or a transport failure is
  // treated as transient -- the saved selection is kept and a retry-able
  // error is surfaced, never silently discarded.
  useEffect(() => {
    if (task.id !== 'verification') return;
    const saved = readSavedVerificationDataset();
    if (!saved) {
      setVerificationRestoring(false);
      return;
    }
    const ac = new AbortController();
    (async () => {
      let res: Response;
      try {
        res = await fetch(`${API_BASE}/tasks/verification/dataset/custom`, {
          credentials: 'include',
          signal: ac.signal,
        });
      } catch (e) {
        if ((e as { name?: string } | null)?.name === 'AbortError') return;
        toast.error('Could not confirm your saved dataset — retry or reselect it from Manage Datasets.');
        setVerificationRestoring(false);
        return;
      }
      if (ac.signal.aborted) return;

      if (res.status >= 500) {
        toast.error('Could not confirm your saved dataset — retry or reselect it from Manage Datasets.');
        setVerificationRestoring(false);
        return;
      }
      if (!res.ok) {
        sessionStorage.removeItem(VERIFICATION_CUSTOM_DATASET_STORAGE_KEY);
        setDataset(task.defaultDataset ?? '');
        setVerificationRestoring(false);
        return;
      }

      try {
        const data = await res.json() as { datasets: Array<{ dataset_name: string }> };
        if (ac.signal.aborted) return;
        if (!data.datasets.some((d) => d.dataset_name === saved)) {
          sessionStorage.removeItem(VERIFICATION_CUSTOM_DATASET_STORAGE_KEY);
          setDataset(task.defaultDataset ?? '');
        }
        setVerificationRestoring(false);
      } catch (e) {
        if (ac.signal.aborted) return;
        toast.error('Could not confirm your saved dataset — retry or reselect it from Manage Datasets.');
        setVerificationRestoring(false);
      }
    })();
    return () => ac.abort();
  }, [task.id]);

  // Verification-only: prune selectedBatchIds against the currently
  // eligible recording set after every change, unconditionally. Without
  // this, switching from Dataset A (with ids selected) straight to Dataset
  // B would leave Dataset A's now-foreign ids sitting in selectedBatchIds
  // -- and since crec_ mappings deliberately remain valid across a switch
  // (needed for the multi-tab guarantee), those stray ids could otherwise
  // silently leak into a batch run for Dataset B.
  useEffect(() => {
    if (task.id !== 'verification') return;
    setSelectedBatchIds((prev) => {
      const eligible = new Set(verificationRecordings.map((r) => r.recording_id));
      const next = prev.filter((id) => eligible.has(id));
      return next.length === prev.length ? prev : next;
    });
  }, [task.id, verificationRecordings]);

  // Verification-only: clears TaskWorkbench-owned state that may reference
  // the previous dataset's recordings on EVERY genuine dataset-identity
  // switch (not just deletion) -- a plain dropdown/modal switch previously
  // left this stale. Uses a ref (not a dependency-array "changed since
  // mount" trick) so it never fires on the very first render. Child-owned
  // state (batch results, saliency, Pair Verification workspace, graph
  // selection) is cleared separately via the key={effectiveDataset} remount
  // below, not here. pairSelectionLabels/perturbationResult/predictionMap
  // already clear via existing effects that key off `dataset`/`selectedFile`
  // /`selectedEmbeddingFile` (see above/below) -- no need to duplicate that
  // here.
  const prevVerificationDatasetRef = useRef<string | null>(null);
  useEffect(() => {
    if (task.id !== 'verification') return;
    const isFirstRun = prevVerificationDatasetRef.current === null;
    const changed = !isFirstRun && prevVerificationDatasetRef.current !== dataset;
    prevVerificationDatasetRef.current = dataset;
    if (!changed) return;

    setSelectedFile(null);
    setSelectedEmbeddingFile(null);
    setLocalPreview(null);
    setDatasetRecordings(null);
  }, [task.id, dataset]);

  // Verification-only: handles CustomDatasetManager's created/uploaded/
  // deleted events. 'created' is a no-op here -- Toolbar's
  // handleDatasetCreated already performs the one state transition that
  // matters (selecting the new dataset), which itself flows through the
  // dataset-transition effect above.
  const activeCustomDatasetName = dataset.startsWith(VERIFICATION_CUSTOM_DATASET_PREFIX)
    ? dataset.slice(VERIFICATION_CUSTOM_DATASET_PREFIX.length)
    : null;

  const handleCustomDatasetChanged = useCallback(
    (event: { type: 'created' | 'uploaded' | 'deleted'; datasetName: string }) => {
      if (event.type === 'deleted' && event.datasetName === activeCustomDatasetName) {
        sessionStorage.removeItem(VERIFICATION_CUSTOM_DATASET_STORAGE_KEY);
        setDataset(task.defaultDataset ?? '');
        return;
      }
      if (event.type === 'uploaded' && event.datasetName === activeCustomDatasetName) {
        setCustomDatasetRefreshToken((t) => t + 1);
      }
    },
    [activeCustomDatasetName, task.defaultDataset]
  );

  // Prediction state
  const [wav2vecPrediction, setWav2vecPrediction] = useState<Wav2Vec2Prediction | null>(null);
  const [whisperPrediction, setWhisperPrediction] = useState<WhisperPrediction | null>(null);
  const [isLoadingPredictions, setIsLoadingPredictions] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [perturbedPredictions, setPerturbedPredictions] = useState<Wav2Vec2Prediction | WhisperPrediction | null>(null);
  const [isLoadingPerturbed, setIsLoadingPerturbed] = useState(false);

  // Refs to track ongoing requests and prevent duplicates
  const wav2vecRequestRef = useRef<AbortController | null>(null);
  const whisperRequestRef = useRef<AbortController | null>(null);

  const { resultKind } = task.capabilities;

  // Clear perturbation result and predictions when selected file changes
  useEffect(() => {
    setPerturbationResult(null);
    setWav2vecPrediction(null);
    setWhisperPrediction(null);
    setPerturbedPredictions(null);
    setPredictionError(null);
  }, [selectedFile, selectedEmbeddingFile]);

  // Fetch perturbed predictions when perturbation result is available
  useEffect(() => {
    const fetchPerturbedPredictions = async () => {
      if (!perturbationResult?.success || !model || !resultKind) {
        setPerturbedPredictions(null);
        return;
      }

      setIsLoadingPerturbed(true);
      setPredictionError(null);

      try {
        let requestBody: any = {
          file_path: perturbationResult.perturbed_file
        };

        let endpoint: string;
        if (resultKind === "classification") {
          endpoint = `${API_BASE}/inferences/wav2vec2-detailed`;
          requestBody.include_attention = false; // Disable attention for better performance
        } else if (resultKind === "transcription") {
          endpoint = `${API_BASE}/inferences/whisper-accuracy`;
          requestBody.model = model;
        } else {
          return; // Unsupported result kind
        }

        const response = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          credentials: 'include',
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch perturbed prediction: ${response.status}`);
        }

        const prediction = await response.json();
        setPerturbedPredictions(prediction);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        setPredictionError(errorMessage);
        console.error("Error fetching perturbed predictions:", err);
      } finally {
        setIsLoadingPerturbed(false);
      }
    };

    fetchPerturbedPredictions();
  }, [perturbationResult, model, resultKind]);

  // Fetch classification prediction (wav2vec2) when the task uses it and a file is selected
  useEffect(() => {
    const fetchWav2vecPrediction = async () => {
      if (resultKind !== "classification" || (!selectedFile && !selectedEmbeddingFile)) {
        setWav2vecPrediction(null);
        setPredictionError(null);
        setIsLoadingPredictions(false);
        return;
      }

      // Cancel any existing request
      if (wav2vecRequestRef.current) {
        wav2vecRequestRef.current.abort();
      }

      // Create new abort controller
      const abortController = new AbortController();
      wav2vecRequestRef.current = abortController;

      setIsLoadingPredictions(true);
      setPredictionError(null);

      try {
        let requestBody: any = {};

        if (selectedFile) {
          // Check if this is an uploaded file - more precise detection
          const isUploadedFile = selectedFile.file_path && (
            selectedFile.file_path.includes('uploads/') ||
            selectedFile.file_path.startsWith('uploads/') ||
            selectedFile.message === "Perturbed file" ||
            selectedFile.message === "File uploaded successfully" ||
            selectedFile.message === "File uploaded and processed successfully"
          ) && !selectedFile.message.includes("Selected from");

          if (isUploadedFile) {
            // This is an uploaded file, use file_path
            requestBody.file_path = selectedFile.file_path;
          } else {
            // This is a dataset file (including custom datasets), use dataset and dataset_file
            requestBody.dataset = dataset;
            requestBody.dataset_file = selectedFile.filename;
          }
        } else if (selectedEmbeddingFile && dataset) {
          // Use embedding file selection
          requestBody.dataset = dataset;
          requestBody.dataset_file = selectedEmbeddingFile;
        }

        // Add option to disable attention for better performance
        requestBody.include_attention = false;  // Set to false by default to improve performance

        const response = await fetch(`${API_BASE}/inferences/wav2vec2-detailed`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          credentials: 'include',
          body: JSON.stringify(requestBody),
          signal: abortController.signal
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch prediction: ${response.status}`);
        }

        const prediction = await response.json();
        setWav2vecPrediction(prediction);

        // Update predictionMap for uploaded files
        if (selectedFile && prediction) {
          const isUploadedFile = selectedFile.file_path && (
            selectedFile.file_path.includes('uploads/') ||
            selectedFile.file_path.startsWith('uploads/') ||
            selectedFile.message === "Perturbed file" ||
            selectedFile.message === "File uploaded successfully" ||
            selectedFile.message === "File uploaded and processed successfully"
          ) && selectedFile.message !== "Selected from embeddings" && selectedFile.message !== "Selected from dataset";

          if (isUploadedFile) {
            const predictionText = typeof prediction === 'string' ? prediction :
              prediction?.predicted_emotion || prediction?.prediction || prediction?.emotion || JSON.stringify(prediction);
            handlePredictionUpdate(selectedFile.file_id, predictionText);
          }
        }
      } catch (err) {
        // Ignore abort errors
        if (err.name === 'AbortError') return;

        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        setPredictionError(errorMessage);
        console.error("Error fetching wav2vec2 prediction:", err);
      } finally {
        setIsLoadingPredictions(false);
        // Clear the request ref if this is the current request
        if (wav2vecRequestRef.current === abortController) {
          wav2vecRequestRef.current = null;
        }
      }
    };

    fetchWav2vecPrediction();

    // Cleanup function
    return () => {
      if (wav2vecRequestRef.current) {
        wav2vecRequestRef.current.abort();
        wav2vecRequestRef.current = null;
      }
    };
  }, [selectedFile, selectedEmbeddingFile, model, dataset, resultKind]);

  // Fetch transcription prediction (whisper) when the task uses it and a file is selected
  useEffect(() => {
    const fetchWhisperPrediction = async () => {
      if (resultKind !== "transcription" || (!selectedFile && !selectedEmbeddingFile)) {
        setWhisperPrediction(null);
        setPredictionError(null);
        setIsLoadingPredictions(false);
        return;
      }

      // Cancel any existing request
      if (whisperRequestRef.current) {
        whisperRequestRef.current.abort();
      }

      // Create new abort controller
      const abortController = new AbortController();
      whisperRequestRef.current = abortController;

      setIsLoadingPredictions(true);
      setPredictionError(null);

      try {
        let requestBody: any = {
          model: model
        };

        let isUploadedFile = false;

        if (selectedFile) {
          // Check if this is an uploaded file - more precise detection
          isUploadedFile = selectedFile.file_path && (
            selectedFile.file_path.includes('uploads/') ||
            selectedFile.file_path.startsWith('uploads/') ||
            selectedFile.message === "Perturbed file" ||
            selectedFile.message === "File uploaded successfully" ||
            selectedFile.message === "File uploaded and processed successfully"
          ) && !selectedFile.message.includes("Selected from");

          if (isUploadedFile) {
            // This is an uploaded file, use file_path
            requestBody.file_path = selectedFile.file_path;
          } else {
            // This is a dataset file (including custom datasets), use dataset and dataset_file
            requestBody.dataset = dataset;
            requestBody.dataset_file = selectedFile.filename;
          }
        } else if (selectedEmbeddingFile && dataset) {
          // Use embedding file selection - this is a dataset file
          requestBody.dataset = dataset;
          requestBody.dataset_file = selectedEmbeddingFile;
          isUploadedFile = false;
        }

        // Choose the correct endpoint based on file type
        let endpoint: string;
        const isCustomDataset = dataset?.startsWith('custom:');

        if (isUploadedFile || isCustomDataset) {
          // For uploaded files or custom datasets, use basic inference endpoint (no ground truth available)
          endpoint = `${API_BASE}/inferences/run`;
        } else {
          // For regular dataset files, use accuracy endpoint to get ground truth and metrics
          endpoint = `${API_BASE}/inferences/whisper-accuracy`;
        }

        const response = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          credentials: 'include',
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch whisper prediction: ${response.status}`);
        }

        const prediction = await response.json();

        let whisperResult: WhisperPrediction;

        if (isUploadedFile || isCustomDataset) {
          // For uploaded files or custom datasets, convert basic prediction to expected format
          whisperResult = {
            predicted_transcript: typeof prediction === 'string' ? prediction : prediction?.text || JSON.stringify(prediction),
            ground_truth: "",
            accuracy_percentage: null,
            word_error_rate: null,
            character_error_rate: null,
            levenshtein_distance: null,
            exact_match: null,
            character_similarity: null,
            word_count_predicted: 0,
            word_count_truth: 0
          };
        } else {
          // For regular dataset files, the accuracy endpoint returns all the metrics
          whisperResult = {
            predicted_transcript: prediction.predicted_transcript || "",
            ground_truth: prediction.ground_truth || "",
            accuracy_percentage: prediction.accuracy_percentage !== null ? prediction.accuracy_percentage : null,
            word_error_rate: prediction.word_error_rate !== null ? prediction.word_error_rate : null,
            character_error_rate: prediction.character_error_rate !== null ? prediction.character_error_rate : null,
            levenshtein_distance: prediction.levenshtein_distance !== null ? prediction.levenshtein_distance : null,
            exact_match: prediction.exact_match !== null ? prediction.exact_match : null,
            character_similarity: prediction.character_similarity !== null ? prediction.character_similarity : null,
            word_count_predicted: prediction.word_count_predicted || 0,
            word_count_truth: prediction.word_count_truth || 0
          };
        }

        setWhisperPrediction(whisperResult);

        // Update predictionMap for uploaded files and custom datasets
        if (selectedFile && (isUploadedFile || isCustomDataset)) {
          handlePredictionUpdate(selectedFile.file_id, whisperResult.predicted_transcript);
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        setPredictionError(errorMessage);
        console.error("Error fetching whisper prediction:", err);
      } finally {
        setIsLoadingPredictions(false);
        // Clear the request ref if this is the current request
        if (whisperRequestRef.current === abortController) {
          whisperRequestRef.current = null;
        }
      }
    };

    fetchWhisperPrediction();

    // Cleanup function
    return () => {
      if (whisperRequestRef.current) {
        whisperRequestRef.current.abort();
        whisperRequestRef.current = null;
      }
    };
  }, [selectedFile, selectedEmbeddingFile, model, dataset, resultKind]);

  // Determine effective dataset based on uploaded files and custom datasets
  const effectiveDataset = (() => {
    // If a custom dataset is selected (Transcription/Emotion's custom: or
    // Verification's verification-custom:), use it as-is -- a selected
    // Verification custom dataset must never be replaced by the legacy
    // "custom" sentinel merely because session assets/uploaded files exist.
    if (dataset.startsWith('custom:') || dataset.startsWith(VERIFICATION_CUSTOM_DATASET_PREFIX)) {
      return dataset;
    }
    // Legacy behavior: if there are uploaded files and no custom dataset, show as "custom"
    if (uploadedFiles && uploadedFiles.length > 0) {
      return "custom";
    }
    // Otherwise use the selected dataset
    return dataset;
  })();

  const [predictionMap, setPredictionMap] = useState<Record<string, string>>({});

  const handlePredictionUpdate = (fileId: string, prediction: string) => {
    setPredictionMap(prev => {
      const updated = { ...prev, [fileId]: prediction };
      return updated;
    });
  };

  const handleUploadSuccess = (uploadResponse: UploadedFile, rawFile?: File) => {
    setUploadedFiles(prev => {
      const newFiles = [...prev, uploadResponse];
      return newFiles;
    });
    // Always select the newly uploaded file
    setSelectedFile(uploadResponse);
    if (rawFile) {
      setUploadedRawFiles(prev => ({ ...prev, [uploadResponse.file_id]: rawFile }));
    }
  };

  const handleFileSelection = (file: UploadedFile) => {
    setLocalPreview(null);
    setSelectedFile(file);
    // Sync embedding selection with audio dataset selection
    setSelectedEmbeddingFile(file.filename);
  };

  const handleEmbeddingSelection = (filename: string) => {
    setLocalPreview(null);
    // Speaker Verification's graph reports backend labels (upload-000,
    // rec_<hash>, ...), never real filenames — translate through the
    // index-aligned resolver BatchAnalysisPanel registered before this ever
    // reaches selectedFile/Datapoint-Editor/Audio-Playback state.
    const resolvedId = task.id === 'verification' ? labelResolverFn?.(filename) : undefined;
    const effectiveId = resolvedId ?? filename;

    setSelectedEmbeddingFile(effectiveId);

    // Try to find and select corresponding file in audio dataset
    const matchingUploadedFile = uploadedFiles.find(f => f.file_id === effectiveId || f.filename === effectiveId);
    if (matchingUploadedFile) {
      setSelectedFile(matchingUploadedFile);
      return;
    }

    // For dataset files, create a file-like object for the UI
    // The AudioDatasetPanel should handle highlighting the corresponding row
    const fileLike: UploadedFile = {
      file_id: effectiveId,
      filename: effectiveId,
      file_path: effectiveId,
      message: "Selected from embeddings"
    };
    setSelectedFile(fileLike);
  };

  const handlePerturbationComplete = (result: any) => {
    setPerturbationResult(result);

    // Clear any existing perturbed predictions since we have a new perturbation
    setPerturbedPredictions(null);
  };

  const handlePredictionRefresh = (file: UploadedFile, prediction: string) => {
    if (file.message === "Perturbed file") {
      // Add the perturbed file to uploaded files
      setUploadedFiles(prevFiles => {
        const existingFile = prevFiles.find(f => f.file_id === file.file_id);
        if (existingFile) {
          return prevFiles.map(f =>
            f.file_id === file.file_id
              ? { ...f, prediction: prediction }
              : f
          );
        } else {
          return [...prevFiles, { ...file, prediction: prediction }];
        }
      });

      // Update predictionMap for perturbed file
      setPredictionMap(prev => {
        const updated = { ...prev, [file.filename]: prediction };
        return updated;
      });
    }

    // Update selected file if it's the same file
    if (selectedFile && selectedFile.file_id === file.file_id) {
      setSelectedFile(prev => prev ? { ...prev, prediction: prediction } : null);
    }
  };

  const handleBatchInferenceStart = useCallback(() => {
    setBatchInferenceStatus('running');
  }, []);

  const handleBatchInferenceComplete = useCallback(() => {
    setBatchInferenceStatus('done');
  }, []);

  // Clear predictions when model or dataset changes
  useEffect(() => {
    setPredictionMap({});
    setBatchInferenceStatus('idle');
  }, [model, dataset]);

  const handleBatchInference = async (selectedModel: string, selectedDataset: string) => {
    // Don't run batch inference for legacy "custom" (uploaded files only)
    if (selectedDataset === 'custom') return;

    // Clear predictions when dataset/model changes to avoid showing old predictions
    setPredictionMap({});

    setBatchInferenceStatus('running');
    try {
      // This will be implemented by AudioDatasetPanel to run inference on all files
      // For now, just set the status to indicate batch inference is requested
      setBatchInferenceStatus('done');
    } catch (error) {
      console.error('Batch inference failed:', error);
      setBatchInferenceStatus('idle');
    }
  };

  // Per-task results card for the Datapoint Editor (from the registry slots).
  // Placeholder tasks have no slot component and render no results card.
  const PredictionResults = TASK_SLOTS[task.id].PredictionResults;
  const WorkbenchCenter = TASK_SLOTS[task.id].WorkbenchCenter;
  const renderPredictionResults = PredictionResults
    ? (showPerturbed: boolean) => (
        <PredictionResults
          selectedFile={selectedFile}
          selectedEmbeddingFile={selectedEmbeddingFile}
          model={model}
          modelLabel={getModelLabel(task, model)}
          wav2vecPrediction={wav2vecPrediction}
          whisperPrediction={whisperPrediction}
          perturbedPredictions={perturbedPredictions}
          isLoading={isLoadingPredictions}
          isLoadingPerturbed={isLoadingPerturbed}
          error={predictionError}
          showPerturbed={showPerturbed}
        />
      )
    : undefined;

  return (
    <EmbeddingProvider>
      <div className="h-screen flex flex-col bg-background">
        {/* Top Navigation Bar */}
        <Toolbar
          task={task}
          selectedFile={selectedFile}
          uploadedFiles={uploadedFiles}
          onFileSelect={setSelectedFile}
          model={model}
          setModel={setModel}
          dataset={dataset}
          setDataset={setDataset}
          onBatchInference={handleBatchInference}
          onUploadSuccess={handleUploadSuccess}
          onVerificationAssetUpload={handleVerificationAssetCreated}
          onCustomDatasetChanged={handleCustomDatasetChanged}
        />

        {/* Main Content Area */}
        <div className="flex-1 overflow-hidden bg-background">
          <PanelGroup direction="horizontal" className="h-full">
            {/* Left Panel: Embeddings & Scalar Plots */}
            <Panel defaultSize={25} minSize={20}>
              <EmbeddingPanel
                key={task.id === 'verification' ? effectiveDataset : undefined}
                model={model}
                dataset={dataset}
                batchAnalysis={task.capabilities.batchAnalysis}
                availableFiles={availableFiles}
                selectedFile={selectedEmbeddingFile}
                onFileSelect={handleEmbeddingSelection}
                verificationMode={task.id === 'verification'}
                // Only Transcription and Emotion run through the legacy
                // MODEL_FUNCTIONS dispatch that /inferences/embeddings needs;
                // they are exactly the tasks with a resultKind.
                legacyEmbeddings={task.capabilities.resultKind !== null}
                pairSelection={pairSelectionLabels}
                onPairSelectionChange={setPairSelectionLabels}
                onReproject={reprojectFn}
              />
            </Panel>

            <PanelResizeHandle className="w-1 bg-border hover:bg-primary/20 transition-colors" />

            {/* Center Panel: Task Explainability */}
            <Panel defaultSize={50} minSize={30}>
              <PanelGroup direction="vertical">
                <Panel defaultSize={70} minSize={40}>
                  {WorkbenchCenter ? (
                    <WorkbenchCenter
                      key={task.id === 'verification' ? effectiveDataset : undefined}
                      model={model}
                      modelLabel={getModelLabel(task, model)}
                      dataset={effectiveDataset}
                      originalDataset={dataset}
                      availableFiles={availableFiles}
                      uploadedFiles={uploadedFiles}
                      uploadedRawFiles={uploadedRawFiles}
                      selectedFile={selectedFile}
                      selectedEmbeddingFile={selectedEmbeddingFile}
                      onFileSelect={handleFileSelection}
                      pairSelection={pairSelectionLabels}
                      onClearPairSelection={() => setPairSelectionLabels([])}
                      datasetRecordings={verificationRecordings}
                      selectedBatchIds={selectedBatchIds}
                      onSelectedBatchIdsChange={setSelectedBatchIds}
                      onReprojectHandlerChange={registerReprojectHandler}
                      onLabelResolverChange={registerLabelResolver}
                      onVerificationAssetCreated={handleVerificationAssetCreated}
                      localPreview={localPreview}
                      onLocalFileSelect={setLocalPreview}
                    />
                  ) : task.status === "active" ? (
                    <ExplainabilityPanel
                      task={task}
                      selectedFile={selectedFile}
                      selectedEmbeddingFile={selectedEmbeddingFile}
                      model={model}
                      dataset={effectiveDataset}
                      originalDataset={dataset}
                      onPerturbationComplete={handlePerturbationComplete}
                      onPredictionRefresh={handlePredictionRefresh}
                    />
                  ) : (
                    <PlaceholderExplainability taskName={task.name} />
                  )}
                </Panel>

                <PanelResizeHandle className="h-1 bg-border hover:bg-primary/20 transition-colors" />

                {/* Bottom Panel: Audio Dataset Table */}
                <Panel defaultSize={30} minSize={20}>
                  <AudioDatasetPanel
                    apiData={apiData}
                    uploadedFiles={uploadedFiles}
                    selectedFile={selectedFile}
                    onFileSelect={handleFileSelection}
                    onUploadSuccess={handleUploadSuccess}
                    model={model}
                    dataset={effectiveDataset}
                    originalDataset={dataset}
                    batchInferenceStatus={batchInferenceStatus}
                    onBatchInferenceStart={handleBatchInferenceStart}
                    onBatchInferenceComplete={handleBatchInferenceComplete}
                    onAvailableFilesChange={setAvailableFiles}
                    onPredictionUpdate={handlePredictionUpdate}
                    predictionMap={predictionMap}
                    hideUploadControl={task.id === 'verification'}
                    selectionVariant={task.id === 'verification' ? 'verification' : 'default'}
                    checkedIds={selectedBatchIds}
                    onCheckedIdsChange={setSelectedBatchIds}
                    onVerificationRecordingsChange={setDatasetRecordings}
                    sessionAssets={sessionAssets}
                    verificationUploadedCount={task.id === 'verification' ? uploadedCount : undefined}
                    refreshToken={customDatasetRefreshToken}
                    isRestoringDataset={task.id === 'verification' ? verificationRestoring : undefined}
                  />
                </Panel>
              </PanelGroup>
            </Panel>

            <PanelResizeHandle className="w-1 bg-border hover:bg-primary/20 transition-colors" />

            {/* Right Panel: Audio Player & Datapoint Editor */}
            <Panel defaultSize={25} minSize={20}>
              <DatapointEditorPanel
                selectedFile={selectedFile}
                selectedEmbeddingFile={selectedEmbeddingFile}
                dataset={effectiveDataset}
                originalDataset={dataset}
                perturbationResult={perturbationResult}
                predictionMap={predictionMap}
                renderPredictionResults={renderPredictionResults}
                datasetRecordings={verificationRecordings}
                localPreview={localPreview}
              />
            </Panel>
          </PanelGroup>
        </div>
      </div>
    </EmbeddingProvider>
  );
};