import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import { API_BASE } from '@/lib/api';

export interface EmbeddingPoint {
  filename: string;
  coordinates: number[];
  embedding?: number[];
  embedding_dim?: number;
  /** Backend-derived cluster color (Speaker Verification only). */
  color?: string;
  /** Extra hover line, e.g. "cluster-2 • fit 0.81" (Speaker Verification only). */
  hoverExtra?: string;
  /** Predicted cluster id, index-aligned with the backend result (Speaker Verification only). */
  clusterId?: string;
}

export interface EmbeddingData {
  model: string;
  dataset: string;
  reduction_method: string;
  n_components: number;
  embeddings: Array<{
    filename: string;
    embedding: number[];
    embedding_dim: number;
  }>;
  reduced_embeddings?: EmbeddingPoint[];
  total_files: number;
  original_dimension: number;
  /** From BatchProjectionResponse when set directly (Speaker Verification only). */
  reduction_method_used?: string;
  effective_components?: number;
  /** Monotonically-increasing id stamped by EmbeddingProvider on every real
   *  publish (fetchEmbeddings success or setEmbeddingDataDirect call) --
   *  never derived from content fields, so two publishes with identical
   *  model/reduction_method/n_components/total_files (e.g. re-selecting a
   *  different batch) still get distinct ids. Used by EmbeddingPlot to know
   *  when the actual projection changed vs. an unrelated re-render (row/
   *  point selection), e.g. to decide whether to reset the 3D camera. */
  revision: number;
}

interface EmbeddingContextType {
  embeddingData: EmbeddingData | null;
  isLoading: boolean;
  error: string | null;
  fetchEmbeddings: (
    model: string,
    dataset: string,
    files: string[],
    reductionMethod?: string,
    nComponents?: number
  ) => Promise<void>;
  clearEmbeddings: () => void;
  /** Directly sets embeddingData, bypassing fetchEmbeddings/the legacy
   *  /inferences/embeddings endpoint entirely. Used by Speaker Verification
   *  to publish batch/project results into the same context EmbeddingPanel
   *  already reads. `revision` is stamped internally on every call -- never
   *  supplied by the caller. */
  setEmbeddingDataDirect: (data: Omit<EmbeddingData, 'revision'> | null) => void;
  /** Predicted cluster currently "focused" for highlighting in the embedding
   *  graph (Speaker Verification only). Styling-only -- never affects
   *  embeddingData, clustering, or the projection itself. */
  focusedClusterId: string | null;
  setFocusedClusterId: (id: string | null) => void;
}

const EmbeddingContext = createContext<EmbeddingContextType | undefined>(undefined);

export const useEmbedding = () => {
  const context = useContext(EmbeddingContext);
  if (context === undefined) {
    throw new Error('useEmbedding must be used within an EmbeddingProvider');
  }
  return context;
};

export const EmbeddingProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [embeddingData, setEmbeddingData] = useState<EmbeddingData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [focusedClusterId, setFocusedClusterId] = useState<string | null>(null);
  const revisionCounterRef = useRef(0);

  // Single path every real publish goes through, so `revision` always gets
  // a fresh id -- never skipped, never derived from (and therefore never
  // collidable with) the data's own content fields.
  const publish = useCallback((data: Omit<EmbeddingData, 'revision'> | null) => {
    if (data === null) {
      setEmbeddingData(null);
      return;
    }
    revisionCounterRef.current += 1;
    setEmbeddingData({ ...data, revision: revisionCounterRef.current });
  }, []);

  const fetchEmbeddings = useCallback(async (
    model: string,
    dataset: string,
    files: string[],
    reductionMethod: string = 'pca',
    nComponents: number = 3
  ) => {
    if (!files || files.length === 0) {
      setError('No files provided');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/inferences/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          model,
          dataset,
          files,
          reduction_method: reductionMethod,
          n_components: nComponents,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch embeddings: ${response.status}`);
      }

      const data = await response.json();
      publish(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch embeddings';
      setError(errorMessage);
      console.error('Error fetching embeddings:', err);
    } finally {
      setIsLoading(false);
    }
  }, [publish]);

  const clearEmbeddings = useCallback(() => {
    setEmbeddingData(null);
    setError(null);
  }, []);

  const setEmbeddingDataDirect = useCallback((data: Omit<EmbeddingData, 'revision'> | null) => {
    publish(data);
    setError(null);
  }, [publish]);

  return (
    <EmbeddingContext.Provider value={{
      embeddingData,
      isLoading,
      error,
      fetchEmbeddings,
      clearEmbeddings,
      setEmbeddingDataDirect,
      focusedClusterId,
      setFocusedClusterId,
    }}>
      {children}
    </EmbeddingContext.Provider>
  );
};
