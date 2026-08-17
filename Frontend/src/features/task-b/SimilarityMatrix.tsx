import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { DiarizationSegment, speakerColor } from "./types";

interface SimilarityMatrixProps {
  segments: DiarizationSegment[];
  embeddings: Record<string, number[]>;
  speakers: string[];
  onPlayPair: (aId: string, bId: string) => void;
}

const CANVAS_SIZE = 480;
const STRIP = 6; // speaker color strip width along the axes

/** Map cosine similarity (clamped to 0..1) to a light→dark blue ramp. */
const cellColor = (value: number): string => {
  const t = Math.max(0, Math.min(1, value));
  const from = [248, 250, 252]; // slate-50
  const to = [30, 58, 138]; // blue-900
  const channel = (i: number) => Math.round(from[i] + (to[i] - from[i]) * t);
  return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
};

interface HoverInfo {
  x: number;
  y: number;
  a: DiarizationSegment;
  b: DiarizationSegment;
  similarity: number;
}

export const SimilarityMatrix = ({
  segments,
  embeddings,
  speakers,
  onPlayPair,
}: SimilarityMatrixProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [groupBySpeaker, setGroupBySpeaker] = useState(false);
  const [hover, setHover] = useState<HoverInfo | null>(null);

  // Only segments long enough to have an embedding participate.
  const ordered = useMemo(() => {
    const embeddable = segments.filter((s) => embeddings[s.id]);
    if (!groupBySpeaker) return embeddable; // segments arrive in time order
    return [...embeddable].sort(
      (a, b) =>
        speakers.indexOf(a.speaker) - speakers.indexOf(b.speaker) ||
        a.start - b.start
    );
  }, [segments, embeddings, speakers, groupBySpeaker]);

  const n = ordered.length;

  // Cosine similarity = dot product (backend L2-normalizes embeddings).
  const matrix = useMemo(() => {
    const vectors = ordered.map((s) => embeddings[s.id]);
    const values = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      values[i * n + i] = 1;
      for (let j = i + 1; j < n; j++) {
        let dot = 0;
        const a = vectors[i];
        const b = vectors[j];
        for (let k = 0; k < a.length; k++) dot += a[k] * b[k];
        values[i * n + j] = dot;
        values[j * n + i] = dot;
      }
    }
    return values;
  }, [ordered, embeddings, n]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || n === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = CANVAS_SIZE * dpr;
    canvas.height = CANVAS_SIZE * dpr;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.scale(dpr, dpr);
    context.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    const inner = CANVAS_SIZE - STRIP;
    const cell = inner / n;

    for (let i = 0; i < n; i++) {
      // Speaker strips: top (columns) and left (rows)
      context.fillStyle = speakerColor(speakers, ordered[i].speaker);
      context.fillRect(STRIP + i * cell, 0, Math.ceil(cell), STRIP);
      context.fillRect(0, STRIP + i * cell, STRIP, Math.ceil(cell));

      for (let j = 0; j < n; j++) {
        context.fillStyle = cellColor(matrix[i * n + j]);
        context.fillRect(
          STRIP + j * cell,
          STRIP + i * cell,
          Math.ceil(cell),
          Math.ceil(cell)
        );
      }
    }
  }, [matrix, ordered, speakers, n]);

  const cellFromEvent = (
    event: React.MouseEvent<HTMLCanvasElement>
  ): { row: number; col: number } | null => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left - STRIP;
    const y = event.clientY - rect.top - STRIP;
    const inner = CANVAS_SIZE - STRIP;
    if (x < 0 || y < 0 || x >= inner || y >= inner || n === 0) return null;
    return {
      row: Math.min(n - 1, Math.floor((y / inner) * n)),
      col: Math.min(n - 1, Math.floor((x / inner) * n)),
    };
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          {n} segments · darker = more similar · same-speaker blocks should look
          dark ("checkerboard")
        </p>
        <Button
          size="sm"
          variant="outline"
          onClick={() => setGroupBySpeaker((v) => !v)}
        >
          {groupBySpeaker ? "Order by time" : "Group by speaker"}
        </Button>
      </div>

      <div className="relative inline-block">
        <canvas
          ref={canvasRef}
          style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }}
          className="cursor-crosshair rounded border"
          onMouseMove={(event) => {
            const cellPosition = cellFromEvent(event);
            if (!cellPosition) return setHover(null);
            const rect = event.currentTarget.getBoundingClientRect();
            setHover({
              x: event.clientX - rect.left,
              y: event.clientY - rect.top,
              a: ordered[cellPosition.row],
              b: ordered[cellPosition.col],
              similarity: matrix[cellPosition.row * n + cellPosition.col],
            });
          }}
          onMouseLeave={() => setHover(null)}
          onClick={(event) => {
            const cellPosition = cellFromEvent(event);
            if (!cellPosition) return;
            onPlayPair(
              ordered[cellPosition.row].id,
              ordered[cellPosition.col].id
            );
          }}
        />
        {hover && (
          <div
            className="pointer-events-none absolute z-10 rounded border bg-background px-2 py-1 text-xs shadow"
            style={{
              left: Math.min(hover.x + 12, CANVAS_SIZE - 180),
              top: hover.y + 12,
            }}
          >
            <div>
              <span className="font-medium">{hover.a.id}</span> ({hover.a.speaker},{" "}
              {hover.a.start.toFixed(1)}–{hover.a.end.toFixed(1)}s)
            </div>
            <div>
              <span className="font-medium">{hover.b.id}</span> ({hover.b.speaker},{" "}
              {hover.b.start.toFixed(1)}–{hover.b.end.toFixed(1)}s)
            </div>
            <div>
              similarity:{" "}
              <span className="font-medium">{hover.similarity.toFixed(3)}</span>
              {" · click to play both"}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};