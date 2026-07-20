import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { PredictionResultsProps, Wav2Vec2Prediction } from "@/tasks/types";

/**
 * Emotion-task results card for the Datapoint Editor
 * (extracted from the wav2vec2 branch of the old PredictionDisplay).
 */
export const ClassificationResults = ({
  selectedFile,
  selectedEmbeddingFile,
  modelLabel,
  wav2vecPrediction,
  perturbedPredictions,
  isLoading,
  isLoadingPerturbed,
  error,
  showPerturbed = false,
}: PredictionResultsProps) => {
  if (!selectedFile && !selectedEmbeddingFile) {
    return (
      <Card>
        <CardContent className="p-3 text-center text-muted-foreground">
          <div className="text-xs">No file selected</div>
        </CardContent>
      </Card>
    );
  }

  const perturbed = perturbedPredictions as Wav2Vec2Prediction | null | undefined;

  return (
    <Card>
      <CardHeader className="bg-panel-header">
        <CardTitle className="text-xs">
          Classification Results
          {modelLabel && (
            <Badge variant="outline" className="ml-1.5 text-[10px] bg-primary/10 text-primary border-primary/20">
              {modelLabel}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1.5">
        {isLoading && (
          <div className="text-xs text-muted-foreground flex items-center gap-2">
            <div className="w-3 h-3 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
            Loading prediction...
          </div>
        )}

        {error && (
          <div className="text-xs text-destructive p-2 bg-destructive/5 rounded-sm border border-destructive/20">
            Error: {error}
          </div>
        )}

        {wav2vecPrediction && !isLoading && (
          <div className="space-y-3">
            {!showPerturbed ? (
              // Original Tab - Show only original predictions
              <div className="space-y-2">
                <div className="text-xs-tight font-medium flex items-center gap-2">
                  Original Audio Prediction
                  <span className="text-xs-tight text-gray-500 border border-gray-300 px-1 rounded">Original</span>
                </div>
                {Object.entries(wav2vecPrediction.probabilities)
                  .sort(([, a], [, b]) => b - a)
                  .map(([emotion, probability]) => {
                    const isPredicted = emotion === wav2vecPrediction.predicted_emotion;
                    return (
                      <div key={emotion} className="flex items-center justify-between text-xs-tight">
                        <div className="flex items-center gap-2">
                          <span className="capitalize">{emotion}</span>
                          {isPredicted && <span className="text-xs-tight text-gray-600 font-medium">Predicted</span>}
                        </div>
                        <div className="flex items-center gap-2 flex-1 max-w-[120px]">
                          <Progress value={probability * 100} className="h-2" />
                          <span className="text-muted-foreground min-w-[2rem]">
                            {(probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
              </div>
            ) : (
              // Perturbed Tab - Show perturbed predictions with comparison
              <div className="space-y-2">
                <div className="text-xs-tight font-medium flex items-center gap-2">
                  Perturbed Audio Prediction
                  <span className="text-xs-tight text-gray-500 border border-gray-300 px-1 rounded">Perturbed</span>
                  {isLoadingPerturbed && (
                    <div className="w-3 h-3 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
                  )}
                </div>
                {!isLoadingPerturbed && perturbed ? (
                  <div className="space-y-2">
                    {Object.entries(perturbed.probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([emotion, probability]) => {
                        const isPredicted = emotion === perturbed.predicted_emotion;
                        const originalProb = wav2vecPrediction.probabilities[emotion] || 0;
                        const change = (probability - originalProb) * 100;
                        const isSignificantChange = Math.abs(change) > 1; // Only highlight changes > 1%
                        return (
                          <div key={emotion} className="flex items-center justify-between text-xs-tight">
                            <div className="flex items-center gap-2">
                              <span className="capitalize">{emotion}</span>
                              {isPredicted && <span className="text-xs-tight text-gray-700 font-medium">Predicted</span>}
                            </div>
                            <div className="flex items-center gap-2 flex-1 max-w-[140px]">
                              <Progress value={probability * 100} className="h-2" />
                              <span className="text-muted-foreground min-w-[2rem]">
                                {(probability * 100).toFixed(1)}%
                              </span>
                              <span className={`text-[10px] min-w-[3rem] font-medium ${
                                !isSignificantChange ? "text-muted-foreground" :
                                change > 0 ? "text-green-600" : change < 0 ? "text-red-600" : "text-muted-foreground"
                              }`}>
                                {change > 0 ? "+" : ""}{change.toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        );
                      })}

                    {/* Show predicted emotion change summary */}
                    <div className="mt-3 p-2 bg-blue-50 rounded border border-blue-200">
                      <div className="text-xs font-medium text-blue-800">Prediction Change</div>
                      <div className="text-xs text-blue-700 mt-1">
                        Original: <span className="font-medium">{wav2vecPrediction.predicted_emotion}</span>
                        {" → "}
                        Perturbed: <span className="font-medium">{perturbed.predicted_emotion}</span>
                        {wav2vecPrediction.predicted_emotion !== perturbed.predicted_emotion && (
                          <span className="text-red-600 font-medium ml-2">Changed!</span>
                        )}
                      </div>
                    </div>
                  </div>
                ) : isLoadingPerturbed ? (
                  <div className="text-xs-tight text-blue-500 flex items-center gap-2">
                    <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                    Loading perturbed prediction...
                  </div>
                ) : (
                  <div className="text-xs-tight text-gray-500 p-2 bg-gray-50 rounded border">
                    No perturbed audio data available. Apply perturbations to compare predictions.
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};
