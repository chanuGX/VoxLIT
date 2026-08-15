import {
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ProjectionPoint, speakerColor } from "./types";

interface EmbeddingScatterProps {
  points: ProjectionPoint[];
  speakers: string[];
  hoveredId: string | null;
  selectedId: string | null;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
}

export const EmbeddingScatter = ({
  points,
  speakers,
  hoveredId,
  selectedId,
  onHover,
  onSelect,
}: EmbeddingScatterProps) => (
  <ResponsiveContainer width="100%" height={320}>
    <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
      <XAxis type="number" dataKey="x" name="PC1" tick={{ fontSize: 10 }} />
      <YAxis type="number" dataKey="y" name="PC2" tick={{ fontSize: 10 }} />
      <Tooltip
        cursor={{ strokeDasharray: "3 3" }}
        content={({ active, payload }) => {
          if (!active || !payload?.length) return null;
          const point = payload[0].payload as ProjectionPoint;
          return (
            <div className="rounded border bg-background px-2 py-1 text-xs shadow">
              <div className="font-medium">{point.speaker}</div>
              <div>{point.id}</div>
              <div>
                confidence:{" "}
                {point.confidence !== null ? point.confidence.toFixed(3) : "–"}
              </div>
            </div>
          );
        }}
      />
      <Legend wrapperStyle={{ fontSize: 11 }} />
      {speakers.map((speaker) => (
        <Scatter
          key={speaker}
          name={speaker}
          data={points.filter((point) => point.speaker === speaker)}
          fill={speakerColor(speakers, speaker)}
          onMouseEnter={(entry) => onHover((entry as ProjectionPoint).id)}
          onMouseLeave={() => onHover(null)}
          onClick={(entry) => onSelect((entry as ProjectionPoint).id)}
        >
          {points
            .filter((point) => point.speaker === speaker)
            .map((point) => {
              const isActive = point.id === hoveredId || point.id === selectedId;
              return (
                <Cell
                  key={point.id}
                  stroke={isActive ? "hsl(var(--foreground))" : "none"}
                  strokeWidth={isActive ? 2 : 0}
                  fillOpacity={point.confidence !== null && point.confidence < 0.2 ? 0.45 : 0.9}
                />
              );
            })}
        </Scatter>
      ))}
    </ScatterChart>
  </ResponsiveContainer>
);