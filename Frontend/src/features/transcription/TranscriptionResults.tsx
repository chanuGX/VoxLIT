import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PredictionResultsProps, WhisperPrediction } from "@/tasks/types";

/**
 * Transcription-task results card for the Datapoint Editor
 * (extracted from the whisper branch of the old PredictionDisplay).
 */
export const TranscriptionResults = ({
  selectedFile,
  selectedEmbeddingFile,
  modelLabel,
  whisperPrediction,
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

  const perturbed = perturbedPredictions as WhisperPrediction | null | undefined;

  return (
    <Card>
      <CardHeader className="bg-panel-header">
        <CardTitle className="text-xs">
          Transcription Results
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

        {whisperPrediction && !isLoading && (
          <div className="space-y-3">
            {!showPerturbed ? (
              // Original Tab - Show only original transcription and metrics
              <div className="space-y-3">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="text-xs font-semibold flex items-center gap-2">
                      Original Transcription Metrics
                      <span className="text-xs-tight text-blue-600 border border-blue-300 px-1 rounded">Original</span>
                    </div>
                    {whisperPrediction.ground_truth && whisperPrediction.ground_truth.trim() !== "" ? (
                      // Show metrics when ground truth is available
                      whisperPrediction.accuracy_percentage !== null && whisperPrediction.word_error_rate !== null ? (
                        <div className="mt-2 grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                          <div className="p-2 bg-gray-50 rounded border text-gray-700">
                            <div className="text-[10px] text-gray-500">WER</div>
                            <div className="font-medium">{whisperPrediction.word_error_rate.toFixed(3)}</div>
                          </div>
                          <div className="p-2 bg-gray-50 rounded border text-gray-700">
                            <div className="text-[10px] text-gray-500">CER</div>
                            <div className="font-medium">{whisperPrediction.character_error_rate.toFixed(3)}</div>
                          </div>
                          <div className="p-2 bg-gray-50 rounded border text-gray-700">
                            <div className="text-[10px] text-gray-500">Accuracy</div>
                            <div className="font-medium">{whisperPrediction.accuracy_percentage.toFixed(1)}%</div>
                          </div>
                          <div className="p-2 bg-gray-50 rounded border text-gray-700">
                            <div className="text-[10px] text-gray-500">Words (Pred)</div>
                            <div className="font-medium">{whisperPrediction.word_count_predicted}</div>
                          </div>
                          <div className="p-2 bg-gray-50 rounded border text-gray-700">
                            <div className="text-[10px] text-gray-500">Words (Truth)</div>
                            <div className="font-medium">{whisperPrediction.word_count_truth}</div>
                          </div>
                          <div className="p-2 bg-gray-50 rounded border text-gray-700">
                            <div className="text-[10px] text-gray-500">Levenshtein</div>
                            <div className="font-medium">{whisperPrediction.levenshtein_distance}</div>
                          </div>
                        </div>
                      ) : (
                        // Ground truth exists but metrics aren't calculated yet
                        <div className="mt-2 p-3 bg-blue-50 rounded border border-blue-200 text-xs text-blue-700">
                          <div className="font-medium">Ground Truth Available</div>
                          <div className="mt-1">Accuracy metrics are being calculated...</div>
                        </div>
                      )
                    ) : (
                      // Show message when ground truth is not available
                      <div className="mt-2 p-3 bg-yellow-50 rounded border border-yellow-200 text-xs text-yellow-700">
                        <div className="font-medium">No Ground Truth Available</div>
                        <div className="mt-1">Accuracy metrics are not available for this dataset-model combination.</div>
                      </div>
                    )}
                  </div>
                </div>

                {whisperPrediction.ground_truth ? (
                  // When ground truth is available, show both in grid layout
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <div className="text-xs font-medium">Predicted Transcript</div>
                      <div className="text-xs p-2 bg-green-50 rounded border font-mono whitespace-pre-wrap">
                        {whisperPrediction.predicted_transcript ? `"${whisperPrediction.predicted_transcript}"` : <span className="italic text-gray-400">No prediction</span>}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-medium">Ground Truth</div>
                      <div className="text-xs p-2 bg-gray-50 rounded border font-mono whitespace-pre-wrap">
                        {`"${whisperPrediction.ground_truth}"`}
                      </div>
                    </div>
                  </div>
                ) : (
                  // When no ground truth is available, show predicted transcript in full width with larger format
                  <div className="w-full">
                    <div className="text-xs font-medium mb-2">Predicted Transcript</div>
                    <div className="text-xs p-4 bg-green-50 rounded-lg border border-green-200 font-mono whitespace-pre-wrap leading-relaxed">
                      {whisperPrediction.predicted_transcript ? `"${whisperPrediction.predicted_transcript}"` : <span className="italic text-gray-400">No prediction available</span>}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              // Perturbed Tab - Show perturbed transcription with comparison
              <div className="space-y-3">
                <div className="text-xs font-semibold flex items-center gap-2">
                  Perturbed Transcription Results
                  <span className="text-xs-tight text-blue-600 border border-blue-300 px-1 rounded">Perturbed</span>
                  {isLoadingPerturbed && (
                    <div className="w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                  )}
                </div>

                {!isLoadingPerturbed && perturbed ? (
                  <div className="space-y-4">
                    {/* Perturbed metrics if available */}
                    {typeof perturbed === 'object' && perturbed.word_error_rate !== null && (
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                        <div className="p-2 bg-blue-50 rounded border text-gray-700">
                          <div className="text-[10px] text-gray-500">WER</div>
                          <div className="font-medium">{perturbed.word_error_rate?.toFixed(3) || 'N/A'}</div>
                        </div>
                        <div className="p-2 bg-blue-50 rounded border text-gray-700">
                          <div className="text-[10px] text-gray-500">CER</div>
                          <div className="font-medium">{perturbed.character_error_rate?.toFixed(3) || 'N/A'}</div>
                        </div>
                        <div className="p-2 bg-blue-50 rounded border text-gray-700">
                          <div className="text-[10px] text-gray-500">Accuracy</div>
                          <div className="font-medium">{perturbed.accuracy_percentage?.toFixed(1) || 'N/A'}%</div>
                        </div>
                        <div className="p-2 bg-blue-50 rounded border text-gray-700">
                          <div className="text-[10px] text-gray-500">Words (P)</div>
                          <div className="font-medium">{perturbed.word_count_predicted || 'N/A'}</div>
                        </div>
                        <div className="p-2 bg-blue-50 rounded border text-gray-700">
                          <div className="text-[10px] text-gray-500">Words (T)</div>
                          <div className="font-medium">{perturbed.word_count_truth || 'N/A'}</div>
                        </div>
                        <div className="p-2 bg-blue-50 rounded border text-gray-700">
                          <div className="text-[10px] text-gray-500">Levenshtein</div>
                          <div className="font-medium">{perturbed.levenshtein_distance || 'N/A'}</div>
                        </div>
                      </div>
                    )}

                    {/* Show perturbed transcript */}
                    <div className="w-full">
                      <div className="text-xs font-medium mb-2">Perturbed Transcript</div>
                      <div className="text-xs p-4 bg-blue-50 rounded-lg border border-blue-200 font-mono whitespace-pre-wrap leading-relaxed">
                        {perturbed.predicted_transcript ? `"${perturbed.predicted_transcript}"` : <span className="italic text-gray-400">No prediction available</span>}
                      </div>
                    </div>

                    {/* Comparison summary if both predictions are available */}
                    {whisperPrediction && perturbed.predicted_transcript && (
                      <div className="mt-3 p-3 bg-blue-50 rounded border border-blue-200">
                        <div className="text-xs font-medium text-blue-800 mb-2">Transcription Comparison</div>
                        <div className="space-y-2 text-xs text-blue-700">
                          <div>
                            <span className="font-medium">Original:</span>
                            <span className="ml-2 font-mono">"{whisperPrediction.predicted_transcript || 'N/A'}"</span>
                          </div>
                          <div>
                            <span className="font-medium">Perturbed:</span>
                            <span className="ml-2 font-mono">"{perturbed.predicted_transcript || 'N/A'}"</span>
                          </div>
                          {whisperPrediction.predicted_transcript !== perturbed.predicted_transcript && (
                            <div className="text-red-600 font-medium">
                              ⚠ Transcription changed due to perturbation!
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ) : isLoadingPerturbed ? (
                  <div className="text-xs-tight text-blue-500 flex items-center gap-2">
                    <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                    Loading perturbed transcription...
                  </div>
                ) : (
                  <div className="text-xs-tight text-gray-500 p-2 bg-gray-50 rounded border">
                    No perturbed audio data available. Apply perturbations to compare transcriptions.
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
