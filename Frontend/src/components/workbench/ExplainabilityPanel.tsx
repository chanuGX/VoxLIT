import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SaliencyVisualization } from "../visualization/SaliencyVisualization";
import { AttentionVisualization } from "../visualization/AttentionVisualization";
import { PerturbationTools } from "../analysis/PerturbationTools";
import { TaskDefinition, UploadedFile } from "@/tasks/types";

interface PerturbationResult {
  perturbed_file: string;
  filename: string;
  duration_ms: number;
  sample_rate: number;
  applied_perturbations: Array<{
    type: string;
    params: Record<string, unknown>;
    status: string;
    error?: string;
  }>;
  success: boolean;
  error?: string;
}

interface ExplainabilityPanelProps {
  task: TaskDefinition;
  selectedFile?: UploadedFile | null;
  selectedEmbeddingFile?: string | null;
  model?: string;
  dataset?: string;
  originalDataset?: string;
  onPerturbationComplete?: (result: PerturbationResult) => void;
  onPredictionRefresh?: (file: UploadedFile, prediction: string) => void;
}

/**
 * Center explainability panel: tabs are driven by the task's capability flags
 * in the registry (saliency / attention / perturbation).
 *
 * Prediction fetching for the Datapoint Editor lives in TaskWorkbench — this
 * panel only hosts the explainability visualizations.
 */
export const ExplainabilityPanel = ({
  task,
  selectedFile,
  selectedEmbeddingFile,
  model,
  dataset,
  originalDataset,
  onPerturbationComplete,
  onPredictionRefresh,
}: ExplainabilityPanelProps) => {
  const tabs = [
    task.capabilities.saliency && "saliency",
    task.capabilities.attention && "attention",
    task.capabilities.perturbation && "perturbation",
  ].filter(Boolean) as string[];

  if (tabs.length === 0) {
    return null;
  }

  const gridCols =
    tabs.length === 3 ? "grid-cols-3" : tabs.length === 2 ? "grid-cols-2" : "grid-cols-1";

  return (
    <div className="h-full bg-panel-background border-t border-border">
      <Tabs defaultValue={tabs[0]} className="h-full flex flex-col">
        <div className="flex-shrink-0 bg-panel-header border-b border-border px-3 py-2">
          <TabsList className={`h-7 grid w-full ${gridCols} bg-muted`}>
            {task.capabilities.saliency && (
              <TabsTrigger value="saliency" className="text-xs">Saliency</TabsTrigger>
            )}
            {task.capabilities.attention && (
              <TabsTrigger value="attention" className="text-xs">Attention</TabsTrigger>
            )}
            {task.capabilities.perturbation && (
              <TabsTrigger value="perturbation" className="text-xs">Perturbation</TabsTrigger>
            )}
          </TabsList>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto bg-background scrollbar-thin">
          {task.capabilities.saliency && (
            <TabsContent value="saliency" className="m-0">
              <div className="p-3">
                <SaliencyVisualization
                  selectedFile={selectedFile || selectedEmbeddingFile}
                  model={model}
                  dataset={dataset}
                />
              </div>
            </TabsContent>
          )}

          {task.capabilities.attention && (
            <TabsContent value="attention" className="m-0">
              <div className="p-3">
                <AttentionVisualization
                  selectedFile={selectedFile || selectedEmbeddingFile}
                  model={model}
                  dataset={dataset}
                />
              </div>
            </TabsContent>
          )}

          {task.capabilities.perturbation && (
            <TabsContent value="perturbation" className="m-0">
              <div className="p-3">
                <PerturbationTools
                  selectedFile={selectedFile}
                  onPerturbationComplete={onPerturbationComplete}
                  onPredictionRefresh={onPredictionRefresh}
                  model={model}
                  dataset={dataset}
                  originalDataset={originalDataset}
                />
              </div>
            </TabsContent>
          )}
        </div>
      </Tabs>
    </div>
  );
};
