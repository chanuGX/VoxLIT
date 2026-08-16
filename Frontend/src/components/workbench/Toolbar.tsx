import { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { Upload, HelpCircle } from "lucide-react";
import { toast } from "sonner";
import { API_BASE } from '@/lib/api';
import { CustomDatasetManager } from '@/components/dataset/CustomDatasetManager';
import { TaskDefinition, UploadedFile, SessionAssetMetadata } from '@/tasks/types';

interface ToolbarProps {
  task: TaskDefinition;
  selectedFile?: UploadedFile | null;
  uploadedFiles?: UploadedFile[];
  onFileSelect?: (file: UploadedFile) => void;
  model: string;
  setModel: (model: string) => void;
  dataset: string;
  setDataset: (dataset: string) => void;
  onBatchInference?: (model: string, dataset: string) => void;
  /** Single upload entry point — also used by AudioDatasetPanel's own
   *  (hideable) uploader so both feed the same shared uploadedFiles state. */
  onUploadSuccess?: (uploadResponse: UploadedFile, rawFile?: File) => void;
  /** Speaker Verification only: routes uploads through the session-scoped
   *  asset endpoint instead of the legacy /upload route. */
  onVerificationAssetUpload?: (asset: SessionAssetMetadata) => void;
}

interface CustomDataset {
  dataset_name: string;
  formatted_name: string;
  total_files: number;
}

/**
 * Registry-driven toolbar: model and dataset dropdowns come from the task
 * definition in src/tasks/registry.tsx — never hardcode options here.
 */
export const Toolbar = ({ task, selectedFile, uploadedFiles, onFileSelect, model, setModel, dataset, setDataset, onBatchInference, onUploadSuccess, onVerificationAssetUpload }: ToolbarProps) => {
  const [customDatasets, setCustomDatasets] = useState<CustomDataset[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const uploadFile = async (file: File) => {
    // Speaker Verification uploads are session-scoped assets, not legacy
    // uploads — a different endpoint, a different response shape, and no
    // `model` field. Every other task is byte-identical to before.
    if (task.id === 'verification') {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const response = await fetch(`${API_BASE}/tasks/verification/session-assets/upload`, {
          method: 'POST',
          credentials: 'include',
          body: formData,
        });
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Upload failed');
        }
        const data = await response.json();
        toast.success(`Uploaded: ${file.name}`);
        onVerificationAssetUpload?.(data);
      } catch (error) {
        console.error('Upload error:', error);
        const msg = error instanceof Error ? error.message : 'Unknown error';
        toast.error(`Failed to upload ${file.name}: ${msg}`);
      }
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        credentials: 'include',
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }
      const data = await response.json();
      toast.success(`Uploaded: ${file.name}`);
      onUploadSuccess?.(data, file);
    } catch (error) {
      console.error('Upload error:', error);
      const msg = error instanceof Error ? error.message : 'Unknown error';
      toast.error(`Failed to upload ${file.name}: ${msg}`);
    }
  };

  const handleFileInputChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      const allowedExtensions = ['.wav', '.mp3', '.m4a', '.flac'];
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
        const isValidFile = file.type.startsWith('audio/') || allowedExtensions.includes(fileExtension);
        if (isValidFile) {
          await uploadFile(file);
        } else {
          toast.error(`Invalid file type: ${file.name}. Supported formats: WAV, MP3, M4A, FLAC`);
        }
      }
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const fetchCustomDatasets = async () => {
    try {
      const response = await fetch(`${API_BASE}/upload/dataset/list`, {
        credentials: 'include'
      });

      if (response.ok) {
        const data = await response.json();
        setCustomDatasets(data.datasets || []);
      }
    } catch (err) {
      console.error('Error fetching custom datasets:', err);
    }
  };

  useEffect(() => {
    if (task.allowCustomDatasets) {
      fetchCustomDatasets();
    }
  }, [task.allowCustomDatasets]);

  const handleDatasetCreated = (datasetName: string) => {
    fetchCustomDatasets(); // Refresh the list
    setDataset(datasetName); // Automatically select the new dataset
  };

  const handleDatasetSelected = (datasetName: string) => {
    setDataset(datasetName);
  };

  const onModelChange = (value: string) => {
    setModel(value);

    // Dataset lists are per-task, so the current dataset stays valid on model
    // change — just re-run batch inference for the new model.
    const isCurrentCustomDataset = dataset.startsWith('custom:');
    if (dataset && !isCurrentCustomDataset && dataset !== 'custom' && onBatchInference) {
      onBatchInference(value, dataset);
    }
  };

  const onDatasetChange = (value: string) => {
    setDataset(value);

    // Check if this is a custom dataset (formatted as custom:session_id:dataset_name)
    const isCustomDataset = value.startsWith('custom:');

    // Trigger batch inference when dataset changes (except for custom datasets)
    if (!isCustomDataset && value !== 'custom' && onBatchInference) {
      onBatchInference(model, value);
    }
  };

  const hasAvailableModels = task.models.some((m) => m.available);
  const hasAvailableDatasets = task.datasets.some((d) => d.available);

  return (
    <TooltipProvider>
      <div className="h-12 bg-white border-b border-border px-5 flex items-center justify-between">
        {/* Left side: Brand, task name, model and dataset selectors */}
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-2.5">
            <Link to="/" className="text-base font-bold text-foreground hover:text-primary transition-colors">
              VoxLIT
            </Link>
            <Badge variant="outline" className="text-[10px] bg-primary/10 text-primary border-primary/20">
              v1.0
            </Badge>
            <Badge variant="secondary" className="text-[10px]">
              {task.name}
            </Badge>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <div className="flex items-center gap-1">
                <span className="text-xs font-medium text-foreground">Model:</span>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
                  </TooltipTrigger>
                  <TooltipContent className="space-y-1">
                    <p className="text-xs">Choose the model for this task.</p>
                    <p className="text-xs">Models are configured in the task registry.</p>
                  </TooltipContent>
                </Tooltip>
              </div>
              <Select value={model} onValueChange={onModelChange} disabled={!hasAvailableModels}>
                <SelectTrigger className="w-40 h-7 border-border text-xs">
                  <SelectValue placeholder="To be added" />
                </SelectTrigger>
                <SelectContent>
                  {task.models.map((m) => (
                    <SelectItem key={m.id} value={m.id} disabled={!m.available}>
                      <span className="flex items-center gap-1.5">
                        {m.label}
                        {!m.available && (
                          <span className="text-[9px] text-muted-foreground border border-border px-1 rounded">
                            coming soon
                          </span>
                        )}
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-1.5">
              <div className="flex items-center gap-1">
                <span className="text-xs font-medium text-foreground">Dataset:</span>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
                  </TooltipTrigger>
                  <TooltipContent className="space-y-1">
                    <p className="text-xs">Select the audio dataset to analyze.</p>
                    <p className="text-xs">Built-in datasets come from the task registry; custom datasets are your uploads.</p>
                  </TooltipContent>
                </Tooltip>
              </div>
              <Select value={dataset} onValueChange={onDatasetChange} disabled={!hasAvailableDatasets && customDatasets.length === 0}>
                <SelectTrigger className="w-44 h-7 border-border text-xs">
                  <SelectValue placeholder="To be added" />
                </SelectTrigger>
                <SelectContent>
                  {/* Built-in datasets from the task registry */}
                  {task.datasets.map((ds) => (
                    <SelectItem key={ds.id} value={ds.id} disabled={!ds.available}>
                      <span className="flex items-center gap-1.5">
                        {ds.label}
                        {!ds.available && (
                          <span className="text-[9px] text-muted-foreground border border-border px-1 rounded">
                            coming soon
                          </span>
                        )}
                      </span>
                    </SelectItem>
                  ))}

                  {/* Custom datasets */}
                  {task.allowCustomDatasets && customDatasets.length > 0 && (
                    <>
                      <SelectItem disabled value="separator">
                        ── Custom Datasets ──
                      </SelectItem>
                      {customDatasets.map((customDataset) => (
                        <SelectItem key={customDataset.formatted_name} value={customDataset.formatted_name}>
                          {customDataset.dataset_name}
                        </SelectItem>
                      ))}
                    </>
                  )}
                </SelectContent>
              </Select>
            </div>

            {uploadedFiles && uploadedFiles.length > 0 && (
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-medium text-foreground">File:</span>
                <Select
                  value={selectedFile?.file_id || ""}
                  onValueChange={(fileId) => {
                    const file = uploadedFiles.find(f => f.file_id === fileId);
                    if (file && onFileSelect) {
                      onFileSelect(file);
                    }
                  }}
                >
                  <SelectTrigger className="w-48 h-7 border-border text-xs">
                    <SelectValue placeholder="Select uploaded file" />
                  </SelectTrigger>
                  <SelectContent>
                    {uploadedFiles.map((file) => (
                      <SelectItem key={file.file_id} value={file.file_id}>
                        {file.filename}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>
        </div>

        {/* Right side: Action buttons */}
        <div className="flex items-center gap-2.5">
          {task.allowCustomDatasets && (
            <CustomDatasetManager
              onDatasetCreated={handleDatasetCreated}
              onDatasetSelected={handleDatasetSelected}
            />
          )}

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="default" size="sm" className="h-7 text-xs shadow-aws-sm" onClick={handleUploadClick}>
                <Upload className="h-3.5 w-3.5 mr-1.5" />
                Upload
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Upload audio files for analysis</p>
            </TooltipContent>
          </Tooltip>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,.flac,.wav,.mp3,.m4a"
            multiple
            onChange={handleFileInputChange}
            className="hidden"
          />
        </div>
      </div>
    </TooltipProvider>
  );
};
