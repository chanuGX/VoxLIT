import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import Plot from "react-plotly.js";
// Same resolution react-plotly.js itself uses internally (its own entry
// does `require('plotly.js/dist/plotly')`) -- importing that exact file
// lets the bundler dedupe against the copy react-plotly.js already
// includes, instead of pulling in a second, differently-resolved copy of
// the whole plotly.js package.
import Plotly from "plotly.js/dist/plotly";
import { useEmbedding } from "../../contexts/EmbeddingContext";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ZoomIn, ZoomOut, RotateCcw, Layers3, Target } from "lucide-react";

export interface ExternalEmbeddingPoint {
  label: string;
  coordinates: number[];
  color: string;
  hoverExtra?: string;
  clusterId?: string;
}

// Cluster colors are either "#rrggbb" hex (the 12-color base palette) or an
// "hsl(h, s%, l%)" string (the golden-angle fallback for cluster index >= 12,
// see clusterColors.ts) -- blending toward white for cluster-focus dimming
// must handle both formats.
const mixWithWhite = (color: string, amount: number): string => {
  let r: number, g: number, b: number;
  const hexMatch = /^#([0-9a-f]{6})$/i.exec(color);
  if (hexMatch) {
    const hex = hexMatch[1];
    r = parseInt(hex.slice(0, 2), 16);
    g = parseInt(hex.slice(2, 4), 16);
    b = parseInt(hex.slice(4, 6), 16);
  } else {
    const hslMatch = /^hsl\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)$/i.exec(color);
    if (!hslMatch) return color;
    const h = parseFloat(hslMatch[1]) / 360;
    const s = parseFloat(hslMatch[2]) / 100;
    const l = parseFloat(hslMatch[3]) / 100;
    if (s === 0) {
      r = g = b = Math.round(l * 255);
    } else {
      const hue2rgb = (p: number, q: number, t: number) => {
        let tt = t;
        if (tt < 0) tt += 1;
        if (tt > 1) tt -= 1;
        if (tt < 1 / 6) return p + (q - p) * 6 * tt;
        if (tt < 1 / 2) return q;
        if (tt < 2 / 3) return p + (q - p) * (2 / 3 - tt) * 6;
        return p;
      };
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      r = Math.round(hue2rgb(p, q, h + 1 / 3) * 255);
      g = Math.round(hue2rgb(p, q, h) * 255);
      b = Math.round(hue2rgb(p, q, h - 1 / 3) * 255);
    }
  }
  const mix = (channel: number) => Math.round(channel + (255 - channel) * amount);
  return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
};

interface EmbeddingPlotProps {
  selectedMethod?: string;
  is3D?: boolean;
  onPointSelect?: (filename: string, coordinates: number[]) => void;
  onAngleRangeSelect?: (selectedFiles: string[]) => void;
  selectedFile?: string | null;
  selectionMode?: 'box' | 'lasso';
  onSelectionChange?: (selectedFiles: string[]) => void;
  externalData?: ExternalEmbeddingPoint[];
  externalSelectedLabels?: string[];
  /** Speaker Verification only: before a real batch result exists, render an
   *  empty-state message instead of the shared mock-data fallback. */
  verificationMode?: boolean;
}

type PlaneType = 'none' | 'xy' | 'xz' | 'yz';

// Separate component for the actual plot content
interface EmbeddingPlotContentProps {
  selectedMethod: string;
  is3D: boolean;
  onPointSelect?: (filename: string, coordinates: number[]) => void;
  onAngleRangeSelect?: (selectedFiles: string[]) => void;
  selectedFile?: string | null;
  selectionMode?: 'box' | 'lasso';
  onSelectionChange?: (selectedFiles: string[]) => void;
  externalData?: ExternalEmbeddingPoint[];
  externalSelectedLabels?: string[];
  verificationMode?: boolean;
}

const EmbeddingPlotContent = ({ selectedMethod, is3D, onPointSelect, onAngleRangeSelect, selectedFile, selectionMode = 'box', onSelectionChange, externalData, externalSelectedLabels, verificationMode }: EmbeddingPlotContentProps) => {
  const { embeddingData, isLoading, error, focusedClusterId, setFocusedClusterId } = useEmbedding();
  const isExternal = externalData !== undefined;
  const plotRef = useRef<any>(null);
  const [selectedPlane, setSelectedPlane] = useState<PlaneType>('none');
  const [angleMin, setAngleMin] = useState<number>(40);
  const [angleMax, setAngleMax] = useState<number>(50);
  const [selectedByAngle, setSelectedByAngle] = useState<string[]>([]);

  // Reset plane selection when switching to 2D
  useEffect(() => {
    if (!is3D) {
      setSelectedPlane('none');
      setSelectedByAngle([]);
    }
  }, [is3D]);

  // Confirmed via instrumented tracing (temporary console logging of every
  // handlePointClick/onPointSelect/downstream-selection-callback invocation,
  // plus raw DOM pointerdown/mousedown/pointerup/mouseup/click listeners on
  // the plot's canvas): for one physical 3D click, the DOM only ever
  // dispatches a single pointerdown/mousedown/pointerup/mouseup/click
  // sequence, but plotly.js's own gl3d picking module calls our onClick prop
  // (plotly_click) more than once for that one gesture -- both calls report
  // the identical curveNumber/pointIndex/customdata, and both land between
  // the DOM's mousedown and mouseup (i.e. while the button is still held),
  // not after it -- consistent with gl3d re-emitting a "click" hit on more
  // than one animation frame during the press before the button is
  // released. This is not a duplicate handler registration (traces/layout
  // are already memoized below and had no effect on the duplicate count),
  // not an overlapping-trace issue (only one canvas, one curveNumber), and
  // not fixable by debouncing distinct clicks over time. The correct fix is
  // to gate on the actual press/release gesture: only the first
  // plotly_click within a given pointerdown-to-pointerup window is acted
  // on, and the gate re-arms on release so the next real click still works.
  const clickGestureHandledRef = useRef(false);
  useEffect(() => {
    const rearm = () => { clickGestureHandledRef.current = false; };
    window.addEventListener('pointerup', rearm);
    window.addEventListener('mouseup', rearm);
    return () => {
      window.removeEventListener('pointerup', rearm);
      window.removeEventListener('mouseup', rearm);
    };
  }, []);

  // Tracks whether the current mousedown-to-mouseup gesture already hit a
  // point (and so was already handled by handlePointClick below) -- read by
  // the empty-space-click-clears-focus effect further down, so a real point
  // click never also runs through the empty-space path as a second,
  // redundant decision. Reset at the start of each new gesture.
  const pointClickedForGestureRef = useRef(false);

  // Reduction-method label used for 3D axis titles, plane dropdown labels,
  // plane trace names, and the plane description text -- computed once,
  // reused everywhere instead of each call site recomputing it.
  const methodLabel = selectedMethod === 'umap' ? 'UMAP' : selectedMethod === 'tsne' ? 't-SNE' : 'PCA';

  // --- 3D camera / 2D pan-range persistence -------------------------------
  // Selecting a point (gold highlight), changing pair selection, or a table
  // row click must never move the camera/view -- only a genuinely new
  // projection (new model/reduction-method/dimension/batch) should.
  //
  // Confirmed empirically (temporary instrumentation + direct DOM reads of
  // gd.layout/_fullLayout during a real 3D rotate): gl3d's orbit camera
  // controller updates the live WebGL scene directly and does not reliably
  // propagate back through react-plotly.js's onUpdate/onInitialized
  // (figure.layout.scene.camera stayed at its unrotated default the entire
  // time, even mid-rotation), so there is nothing meaningful to capture
  // there. The actual fix is simpler: Plotly's own uirevision-driven
  // preservation already keeps the live interactive camera/range intact
  // across a Plotly.react() call, *as long as the incoming layout doesn't
  // explicitly re-specify that property*. Explicitly re-specifying a
  // "last known" camera/range on every render (the original approach here)
  // is itself what overrides the live interactive state -- Plotly always
  // treats an explicit value as an instruction to set it. So: only ever
  // set camera/range explicitly on the one render that's genuinely
  // resetting for a new projection; omit them on every other render and
  // let uirevision do the preserving.
  const plotContainerRef = useRef<HTMLDivElement>(null);
  const DEFAULT_CAMERA = { eye: { x: 1.5, y: 1.5, z: 1.5 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 0, z: 1 } };
  // Last range this file's own right-drag pan (item 3 below) applied --
  // read only to seed the *next* drag's starting point, and by the layout
  // memo purely to decide whether a pan is in progress; never re-applied
  // into the layout explicitly (see reasoning above).
  const panRangeRef = useRef<{ x: [number, number]; y: [number, number] } | null>(null);

  // Identity of the current projection: embeddingData.revision only changes
  // on a genuine new publish (batch/reproject or fetchEmbeddings success --
  // see EmbeddingContext.tsx), never on a click/selection re-render, and
  // never collides across two different publishes even when they share the
  // same model/method/dimensions/count. is3D is concatenated separately so
  // toggling 2D/3D resets the view immediately, without waiting on the
  // async reproject/fetch that will itself also bump revision shortly after.
  const revisionKey = `${embeddingData?.revision ?? 'none'}|${is3D ? '3d' : '2d'}`;
  const lastRevisionRef = useRef<string | null>(null);
  let didResetViewThisRender = false;
  if (lastRevisionRef.current !== null && lastRevisionRef.current !== revisionKey) {
    // Reset synchronously during render (not in a useEffect) so the SAME
    // render that detects a new projection already reflects the reset view.
    // An effect-based reset would only take effect on some later, unrelated
    // render, since mutating a ref never itself schedules a re-render.
    panRangeRef.current = null;
    didResetViewThisRender = true;
  }
  lastRevisionRef.current = revisionKey;

  // Generate mock data as fallback
  const generateMockData = () => {
    const n = 50;
    const x = [];
    const y = [];
    const colors = [];
    const text = [];
    
    for (let i = 0; i < n; i++) {
      x.push(Math.random() * 20 - 10);
      y.push(Math.random() * 20 - 10);
      colors.push(['neutral', 'happy', 'sad', 'angry'][Math.floor(Math.random() * 4)]);
      text.push(`Sample ${i + 1}`);
    }
    
    return { x, y, colors, text };
  };

  // Handle point selection
  const handlePointClick = useCallback((event: any) => {
    // Only the first plotly_click within a given pointerdown-to-pointerup
    // gesture is acted on -- see the clickGestureHandledRef comment above.
    if (clickGestureHandledRef.current) {
      return;
    }
    if (event.points && event.points.length > 0) {
      const point = event.points[0];
      // Use customdata[0] which contains the raw filename/label (not the HTML-formatted text).
      // Traces with no customdata (origin, plane, connector) must not be treated as selectable points.
      const label = point.customdata?.[0];
      if (label === undefined) {
        return;
      }
      clickGestureHandledRef.current = true;
      const coordinates = is3D ? [point.x, point.y, point.z] : [point.x, point.y];

      // A point click always clears any active cluster focus -- this reuses
      // Plotly's own hit-testing/gesture-dedup rather than a second
      // independent heuristic, so there's no threshold to disagree with it.
      // Also marks this gesture as "already handled" so the empty-space
      // click-clear effect below skips its own (redundant) decision.
      pointClickedForGestureRef.current = true;
      setFocusedClusterId(null);

      if (onPointSelect) {
        onPointSelect(label, coordinates);
      }
    }
  }, [onPointSelect, is3D, setFocusedClusterId]);

  // Handle 2D box/lasso selection
  const handleSelection = useCallback((event: any) => {
    if (!is3D && onSelectionChange && event?.points) {
      // Use customdata[0] which contains the raw filename/label (not the HTML-formatted text).
      // Points from traces with no customdata (origin, plane, connector) are excluded.
      const selected = event.points
        .filter((p: any) => p.customdata?.[0] !== undefined)
        .map((p: any) => p.customdata[0]);
      onSelectionChange(selected);
    }
  }, [is3D, onSelectionChange]);

  // Handle deselection
  const handleDeselect = useCallback(() => {
    if (!is3D && onSelectionChange) {
      onSelectionChange([]);
    }
  }, [is3D, onSelectionChange]);

  // Use real embedding data if available, otherwise fall back to mock data
  const getPlotData = () => {
    if (isExternal) {
      const points = externalData!;
      const x = points.map(point => point.coordinates[0]);
      const y = points.map(point => point.coordinates[1]);
      const z = is3D && points.length > 0 && points[0].coordinates.length > 2
        ? points.map(point => point.coordinates[2])
        : undefined;
      const text = points.map(point => point.label);
      const colors = points.map(point => point.color);
      return { x, y, z, colors, text };
    }

    if (embeddingData && embeddingData.reduced_embeddings && embeddingData.reduced_embeddings.length > 0) {
      const x = embeddingData.reduced_embeddings.map(point => point.coordinates[0]);
      const y = embeddingData.reduced_embeddings.map(point => point.coordinates[1]);
      const z = is3D && embeddingData.reduced_embeddings[0].coordinates.length > 2 
        ? embeddingData.reduced_embeddings.map(point => point.coordinates[2]) 
        : undefined;
      const text = embeddingData.reduced_embeddings.map(point => point.filename);
      
      // Enhanced color mapping with spatial clustering
      const colors = embeddingData.reduced_embeddings.map((point, index) => {
        const filename = point.filename.toLowerCase();
        
        // First try emotion-based coloring from RAVDESS dataset
        if (filename.includes('01-01') || filename.includes('neutral')) return 'neutral';
        if (filename.includes('01-03') || filename.includes('happy') || filename.includes('joy')) return 'happy';
        if (filename.includes('01-04') || filename.includes('sad') || filename.includes('sadness')) return 'sad';
        if (filename.includes('01-05') || filename.includes('angry') || filename.includes('anger')) return 'angry';
        if (filename.includes('01-06') || filename.includes('fear') || filename.includes('afraid')) return 'fear';
        if (filename.includes('01-07') || filename.includes('disgust')) return 'disgust';
        if (filename.includes('01-08') || filename.includes('surprise')) return 'surprise';
        if (filename.includes('01-02') || filename.includes('calm')) return 'calm';
        
        // For Common Voice or other datasets, use spatial clustering
        const coords = point.coordinates;
        if (coords.length >= 2) {
          const [px, py] = coords;
          
          // Calculate quartiles for better spatial distribution
          const sortedX = x.slice().sort((a, b) => a - b);
          const sortedY = y.slice().sort((a, b) => a - b);
          const q1X = sortedX[Math.floor(sortedX.length * 0.25)];
          const q3X = sortedX[Math.floor(sortedX.length * 0.75)];
          const q1Y = sortedY[Math.floor(sortedY.length * 0.25)];
          const q3Y = sortedY[Math.floor(sortedY.length * 0.75)];
          
          // Assign colors based on spatial regions
          if (px > q3X && py > q3Y) return 'region1'; // Top-right
          if (px < q1X && py > q3Y) return 'region2'; // Top-left
          if (px < q1X && py < q1Y) return 'region3'; // Bottom-left
          if (px > q3X && py < q1Y) return 'region4'; // Bottom-right
          if (px >= q1X && px <= q3X && py >= q1Y && py <= q3Y) return 'center'; // Center
          if (px >= q1X && px <= q3X) return 'mid_vertical'; // Middle band
          if (py >= q1Y && py <= q3Y) return 'mid_horizontal'; // Middle band
        }
        
        return 'unknown';
      });
      
      return { x, y, z, colors, text };
    }
    
    const mockData = generateMockData();
    if (is3D) {
      // Generate mock Z coordinates
      const z = mockData.x.map(() => Math.random() * 20 - 10);
      return { ...mockData, z };
    }
    return mockData;
  };

  // Create transparent plane surfaces for 3D visualization
  const createPlane = (planeType: PlaneType, bounds: { x: [number, number], y: [number, number], z: [number, number] }) => {
    if (!is3D || planeType === 'none') return null;

    // Make bounds bigger for more visible plane
    const [xMin, xMax] = [bounds.x[0] * 1.3, bounds.x[1] * 1.3];
    const [yMin, yMax] = [bounds.y[0] * 1.3, bounds.y[1] * 1.3];
    const [zMin, zMax] = [bounds.z[0] * 1.3, bounds.z[1] * 1.3];

    const planeAlpha = 0.35; // Increased opacity (35% instead of 20%)
    
    switch (planeType) {
      case 'xy': // X-Y plane through origin (Z = 0)
        return {
          type: 'surface' as const,
          x: [[xMin, xMax], [xMin, xMax]],
          y: [[yMin, yMin], [yMax, yMax]],
          z: [[0, 0], [0, 0]], // Always pass through Z = 0 (origin)
          opacity: planeAlpha,
          colorscale: [[0, 'rgba(59, 130, 246, 0.5)'], [1, 'rgba(59, 130, 246, 0.5)']], // Blue with higher opacity
          showscale: false,
          hoverinfo: 'skip',
          name: `${methodLabel} 1–2 plane (${methodLabel} 3 = 0)`
        };
      case 'xz': // X-Z plane through origin (Y = 0)
        return {
          type: 'surface' as const,
          x: [[xMin, xMax], [xMin, xMax]],
          y: [[0, 0], [0, 0]], // Always pass through Y = 0 (origin)
          z: [[zMin, zMin], [zMax, zMax]],
          opacity: planeAlpha,
          colorscale: [[0, 'rgba(16, 185, 129, 0.5)'], [1, 'rgba(16, 185, 129, 0.5)']], // Green with higher opacity
          showscale: false,
          hoverinfo: 'skip',
          name: `${methodLabel} 1–3 plane (${methodLabel} 2 = 0)`
        };
      case 'yz': // Y-Z plane through origin (X = 0)
        return {
          type: 'surface' as const,
          x: [[0, 0], [0, 0]], // Always pass through X = 0 (origin)
          y: [[yMin, yMax], [yMin, yMax]],
          z: [[zMin, zMin], [zMax, zMax]],
          opacity: planeAlpha,
          colorscale: [[0, 'rgba(239, 68, 68, 0.5)'], [1, 'rgba(239, 68, 68, 0.5)']], // Red with higher opacity
          showscale: false,
          hoverinfo: 'skip',
          name: `${methodLabel} 2–3 plane (${methodLabel} 1 = 0)`
        };
      default:
        return null;
    }
  };

  // Calculate angle between point and selected plane relative to origin (0,0,0)
  const calculateAngleToPlane = (x: number, y: number, z: number, plane: PlaneType): number => {
    if (plane === 'none') return 0;
    
    const point = [x, y, z];
    const origin = [0, 0, 0];
    
    // Calculate vector from origin to point
    const vector = [x - origin[0], y - origin[1], z - origin[2]];
    const vectorMagnitude = Math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2);
    
    if (vectorMagnitude === 0) return 0; // Point at origin
    
    // Define plane normal vectors
    let planeNormal: number[];
    switch (plane) {
      case 'xy': planeNormal = [0, 0, 1]; break; // Z axis (normal to XY plane)
      case 'xz': planeNormal = [0, 1, 0]; break; // Y axis (normal to XZ plane)  
      case 'yz': planeNormal = [1, 0, 0]; break; // X axis (normal to YZ plane)
      default: planeNormal = [0, 0, 1]; break;
    }
    
    // Calculate dot product
    const dotProduct = vector[0] * planeNormal[0] + vector[1] * planeNormal[1] + vector[2] * planeNormal[2];
    
    // Calculate angle between vector and plane normal (0° = perpendicular to plane, 90° = in plane)
    const angleToNormal = Math.acos(Math.abs(dotProduct) / vectorMagnitude) * (180 / Math.PI);
    
    // Convert to angle from plane (90° - angle to normal)
    return 90 - angleToNormal;
  };

  // Single generic point source for the plane/angle tool: external batch points when
  // present, otherwise the context-driven embedding points. Keeps the plane/angle
  // geometry and controls identical for both internal and external callers.
  const activePoints = useMemo(() => {
    if (externalData) {
      return externalData.map(point => ({ label: point.label, coordinates: point.coordinates }));
    }
    if (embeddingData?.reduced_embeddings) {
      return embeddingData.reduced_embeddings.map(point => ({ label: point.filename, coordinates: point.coordinates }));
    }
    return [];
  }, [externalData, embeddingData]);

  // Fallback starting range for the very first 2D right-drag pan, before any
  // interaction has happened. Plotly's own autorange-computed values live
  // only in the private `_fullLayout` (verified in plotly.js's
  // plots/cartesian/autorange.js: `doAutoRange` writes to the internal `ax`
  // object, not the public `gd.layout`), so this is derived purely from our
  // own already-computed point coordinates instead -- simple min/max with
  // 10% padding, matching the spirit of the 3D plane bounds calculation
  // elsewhere in this file. Superseded by panRangeRef (this file's own
  // right-drag pan, below) the moment a first drag establishes one.
  const fallbackRange2D = useMemo(() => {
    if (activePoints.length === 0) return null;
    const xs = activePoints.map((p) => p.coordinates[0]);
    const ys = activePoints.map((p) => p.coordinates[1]);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xPad = (xMax - xMin) * 0.1 || 1;
    const yPad = (yMax - yMin) * 0.1 || 1;
    return {
      x: [xMin - xPad, xMax + xPad] as [number, number],
      y: [yMin - yPad, yMax + yPad] as [number, number],
    };
  }, [activePoints]);

  // Right-drag panning for 2D. Plotly's cartesian dragmode applies
  // uniformly regardless of mouse button -- there's no built-in "left
  // selects, right pans" split for scatter traces (unlike gl3d's own
  // 'orbit' dragmode, which already natively supports left-drag-rotate,
  // right-drag-pan, and wheel-zoom for free, so 3D needs none of this).
  // Driven via the public, documented Plotly.relayout(gd, updateObj) API,
  // since Plotly's own dragmode can't be told to do this. Gated on !is3D so
  // 3D's native orbit interactions and the plane-controls panel are
  // untouched.
  useEffect(() => {
    if (is3D) return;
    const container = plotContainerRef.current;
    if (!container) return;

    const MARGIN = { l: 35, r: 35, t: 35, b: 35 };
    let dragging = false;
    let startClientX = 0;
    let startClientY = 0;
    let startRange: { x: [number, number]; y: [number, number] } | null = null;

    const handleMouseMove = (e: MouseEvent) => {
      if (!dragging || !startRange) return;
      const rect = container.getBoundingClientRect();
      const plotWidth = Math.max(1, rect.width - MARGIN.l - MARGIN.r);
      const plotHeight = Math.max(1, rect.height - MARGIN.t - MARGIN.b);

      const pixelDeltaX = e.clientX - startClientX;
      const pixelDeltaY = e.clientY - startClientY;

      const xSpan = startRange.x[1] - startRange.x[0];
      const ySpan = startRange.y[1] - startRange.y[0];

      // "Grab and drag" panning: content follows the cursor. Screen X and
      // data X increase in the same direction, so dragging right shifts the
      // range left by the equivalent data-space amount. Screen Y increases
      // downward while data Y increases upward, so Y's shift direction is
      // inverted relative to X's.
      const dataDeltaX = (pixelDeltaX / plotWidth) * xSpan;
      const dataDeltaY = (pixelDeltaY / plotHeight) * ySpan;

      const newXRange: [number, number] = [startRange.x[0] - dataDeltaX, startRange.x[1] - dataDeltaX];
      const newYRange: [number, number] = [startRange.y[0] + dataDeltaY, startRange.y[1] + dataDeltaY];

      panRangeRef.current = { x: newXRange, y: newYRange };

      const gd = plotRef.current?.el;
      if (gd) {
        Plotly.relayout(gd, { 'xaxis.range': newXRange, 'yaxis.range': newYRange });
      }
    };

    const handleMouseUp = () => {
      dragging = false;
      startRange = null;
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    const handleMouseDown = (e: MouseEvent) => {
      if (e.button !== 2) return; // left-button clicks/drags fall through to Plotly untouched
      e.preventDefault();
      e.stopPropagation();

      const baseline = panRangeRef.current ?? fallbackRange2D;
      if (!baseline) return; // nothing plotted yet -- nothing to pan

      dragging = true;
      startClientX = e.clientX;
      startClientY = e.clientY;
      startRange = baseline;

      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    };

    const handleContextMenu = (e: MouseEvent) => {
      e.preventDefault();
    };

    container.addEventListener('mousedown', handleMouseDown, true);
    container.addEventListener('contextmenu', handleContextMenu);

    return () => {
      container.removeEventListener('mousedown', handleMouseDown, true);
      container.removeEventListener('contextmenu', handleContextMenu);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [is3D, fallbackRange2D]);

  // Clear cluster focus on an empty-space click (Speaker Verification only).
  // A click on an actual point is already handled by handlePointClick above,
  // via Plotly's own hit-testing -- this only needs to cover the "miss" case,
  // since there's no Plotly-native "empty space click" event. Registered
  // capture-phase on mousedown (matching the right-drag-pan listener above)
  // so it's guaranteed to observe the gesture before Plotly's own SVG/gl3d
  // handling. The ~8px threshold matches Plotly's own cartesian MINDRAG
  // constant, so a real click vs. a drag/orbit/box-select/lasso gesture is
  // judged the same way Plotly itself judges it, not by a second, possibly
  // disagreeing heuristic.
  //
  // The mouseup decision is deferred one tick (setTimeout 0): the raw DOM
  // mouseup this listener reacts to fires *before* the browser's subsequent
  // native 'click' event, which is what Plotly's own onClick/plotly_click
  // (and therefore handlePointClick, and pointClickedForGestureRef) reacts
  // to -- so at mouseup time we can't yet know whether this gesture hit a
  // point. Waiting one macrotask lets that synchronous click dispatch
  // finish first, so pointClickedForGestureRef accurately reflects this
  // gesture before deciding whether this is a genuine empty-space click.
  useEffect(() => {
    if (!isExternal) return;
    const container = plotContainerRef.current;
    if (!container) return;

    const CLICK_MOVE_THRESHOLD = 8;
    let armed = false;
    let downX = 0;
    let downY = 0;
    let pendingTimeoutId: ReturnType<typeof setTimeout> | null = null;

    const handleMouseDown = (e: MouseEvent) => {
      if (e.button !== 0) return; // left button (primary) only -- ignores right-drag panning
      armed = true;
      downX = e.clientX;
      downY = e.clientY;
      pointClickedForGestureRef.current = false;
    };

    const handleMouseUp = (e: MouseEvent) => {
      if (!armed) return;
      armed = false;
      const dx = e.clientX - downX;
      const dy = e.clientY - downY;
      // Movement beyond the threshold means this gesture was a drag/orbit/
      // box-select/lasso, never a click -- must not clear focus.
      if (Math.hypot(dx, dy) > CLICK_MOVE_THRESHOLD) return;

      pendingTimeoutId = setTimeout(() => {
        pendingTimeoutId = null;
        if (!pointClickedForGestureRef.current) {
          setFocusedClusterId(null);
        }
      }, 0);
    };

    container.addEventListener('mousedown', handleMouseDown, true);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      container.removeEventListener('mousedown', handleMouseDown, true);
      window.removeEventListener('mouseup', handleMouseUp);
      if (pendingTimeoutId !== null) {
        clearTimeout(pendingTimeoutId);
      }
    };
  }, [isExternal, setFocusedClusterId]);

  // Select points based on angle range - memoized to prevent unnecessary recalculations
  const selectedFiles = useMemo(() => {
    if (!is3D || selectedPlane === 'none' || activePoints.length === 0) {
      return [];
    }

    return activePoints
      .filter(point => {
        if (point.coordinates.length < 3) return false;

        const [x, y, z] = point.coordinates;
        const angle = calculateAngleToPlane(x, y, z, selectedPlane);

        return angle >= angleMin && angle <= angleMax;
      })
      .map(point => point.label);
  }, [is3D, selectedPlane, activePoints, angleMin, angleMax]);

  // Update selected points when calculated files change
  useEffect(() => {
    setSelectedByAngle(selectedFiles);
    
    // Notify parent component only if selection actually changed
    if (onAngleRangeSelect && selectedFiles.join(',') !== selectedByAngle.join(',')) {
      onAngleRangeSelect(selectedFiles);
    }
  }, [selectedFiles, onAngleRangeSelect]); // Remove selectedByAngle from dependencies to prevent loops

  // Traces are memoized so an unrelated re-render (e.g. a click's own state
  // update) doesn't rebuild data/traces with a new identity every time.
  // react-plotly.js calls Plotly.react() whenever the data/layout prop
  // reference changes -- measured to have no effect on the double-click bug
  // (see clickGestureHandledRef above for the actual fix), but still
  // worthwhile so an unrelated re-render never rebuilds the plotted points
  // themselves, only marker highlight state.
  const traces = useMemo(() => {
    const plotData = getPlotData();
    const { x, y, colors, text } = plotData;
    const z = 'z' in plotData ? plotData.z : undefined;

    // Calculate bounds for plane creation
    const bounds = x.length > 0 ? {
      x: [Math.min(...x) * 1.1, Math.max(...x) * 1.1] as [number, number],
      y: [Math.min(...y) * 1.1, Math.max(...y) * 1.1] as [number, number],
      z: z && z.length > 0 ? [Math.min(...z) * 1.1, Math.max(...z) * 1.1] as [number, number] : [0, 0] as [number, number]
    } : { x: [0, 0] as [number, number], y: [0, 0] as [number, number], z: [0, 0] as [number, number] };

    // Create marker sizes based on selection
    const markerSizes = text.map(filename => {
      if (isExternal) {
        return externalSelectedLabels?.includes(filename) ? 12 : 8;
      }
      if (selectedFile === filename) return 12; // Currently selected file (medium-large)
      if (selectedByAngle.includes(filename)) return 8; // Angle range selected (medium)
      return 6; // Default (smaller)
    });

    // Create marker colors based on selection. Cluster-focus dimming is
    // deliberately NOT applied here -- it's applied imperatively via
    // Plotly.restyle() in a separate effect below, so that toggling focus
    // never changes this useMemo's output reference (see focusStyles/the
    // restyle effect further down).
    const markerColors = text.map((filename, index) => {
      if (isExternal) {
        return externalSelectedLabels?.includes(filename) ? '#FFD700' : externalData![index].color;
      }
      if (selectedFile === filename) return '#FFD700'; // Gold for selected file
      if (selectedByAngle.includes(filename)) return '#ef4444'; // Red for angle selected
      return '#3b82f6'; // Blue for all other points
    });

    // Create marker opacities based on selection
    const hasSelection = isExternal
      ? (externalSelectedLabels?.length ?? 0) > 0
      : selectedFile || selectedByAngle.length > 0;
    const markerOpacities = text.map((filename, index) => {
      if (isExternal) {
        return externalSelectedLabels?.includes(filename) ? 1.0 : 0.85;
      }
      if (!hasSelection) return 0.8; // Default opacity when no selection
      if (selectedFile === filename) return 1.0; // Full opacity for selected file
      if (selectedByAngle.includes(filename)) return 0.9; // High opacity for angle selected
      // Different transparency for 2D vs 3D unselected points
      return is3D ? 0.1 : 0.45; // More transparent in 3D, slightly visible in 2D
    });

    // Create traces array - start with main scatter plot
    const result: any[] = [];

    // Create hover text with angle information
    const hoverText = text.map((filename, index) => {
      let baseText = `<b>${filename}</b>`;

      if (isExternal && externalData![index].hoverExtra) {
        baseText += `<br>${externalData![index].hoverExtra}`;
      }

      // Add angle information if this point is selected by angle range and in 3D mode
      if (is3D && selectedPlane !== 'none' && selectedByAngle.includes(filename) && z) {
        const [px, py, pz] = [x[index], y[index], z[index]];
        const angle = calculateAngleToPlane(px, py, pz, selectedPlane);
        baseText += `<br>Angle: ${angle.toFixed(1)}°`;
        baseText += `<br>Plane: ${selectedPlane.toUpperCase()}`;
      }

      return baseText;
    });

    // Create main trace data
    const traceData: any = {
      x: x,
      y: y,
      mode: 'markers',
      type: is3D ? 'scatter3d' : 'scatter',
      marker: {
        size: markerSizes,
        color: markerColors,
        showscale: false,
        line: {
          width: 0, // Remove marker outlines
          color: 'transparent'
        },
        opacity: markerOpacities // Use dynamic opacity array
      },
      text: hoverText,
      // Store [filename/label, color] for each point
      customdata: text.map((filename, index) => [filename, isExternal ? externalData![index].color : colors[index]]),
    };

    // Add Z coordinate for 3D plots
    if (is3D && z) {
      traceData.z = z;
      traceData.hovertemplate = '%{text}<extra></extra>';
    } else {
      traceData.hovertemplate = '%{text}<extra></extra>';
    }

    result.push(traceData);

    // Add origin point (0,0,0) highlight for 3D plots or (0,0) for 2D plots
    const originTrace: any = {
      x: [0],
      y: [0],
      mode: 'markers',
      type: is3D ? 'scatter3d' : 'scatter',
      marker: {
        size: is3D ? 5 : 4, // Slightly smaller to match the new scale
        color: '#000000', // Black for origin
        symbol: 'diamond',
        line: {
          width: 1, // Thinner outline
          color: '#ffffff' // White outline for visibility
        },
        opacity: 0.8 // Slightly transparent
      },
      text: [is3D ? 'Origin (0,0,0)' : 'Origin (0,0)'],
      hovertemplate: is3D ? '<b>Origin (0,0,0)</b><extra></extra>' : '<b>Origin (0,0)</b><extra></extra>',
      name: 'Origin',
      showlegend: false
    };

    if (is3D) {
      originTrace.z = [0];
    }

    result.push(originTrace);

    // Connect the two externally-selected points (pair comparison). No customdata is set,
    // so this trace is excluded from click/box-select and from the plane/angle point source.
    if (isExternal && externalData && externalSelectedLabels?.length === 2) {
      const [labelA, labelB] = externalSelectedLabels;
      const pointA = externalData.find(point => point.label === labelA);
      const pointB = externalData.find(point => point.label === labelB);
      if (pointA && pointB) {
        const connectorTrace: any = {
          x: [pointA.coordinates[0], pointB.coordinates[0]],
          y: [pointA.coordinates[1], pointB.coordinates[1]],
          mode: 'lines',
          type: is3D ? 'scatter3d' : 'scatter',
          line: { color: '#6b7280', width: 2, dash: 'dot' },
          hoverinfo: 'skip',
          showlegend: false,
          name: 'Pair connector',
        };
        if (is3D) {
          connectorTrace.z = [pointA.coordinates[2], pointB.coordinates[2]];
        }
        result.push(connectorTrace);
      }
    }

    // Add plane if selected and in 3D mode
    if (is3D && selectedPlane !== 'none') {
      const planeTrace = createPlane(selectedPlane, bounds);
      if (planeTrace) {
        result.push(planeTrace);
      }
    }

    return result;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isExternal, externalData, externalSelectedLabels, embeddingData, is3D, selectedFile, selectedByAngle, selectedPlane, angleMin, angleMax]);

  // The main scatter/scatter3d trace is always pushed first in the traces
  // useMemo above, unconditionally, before the origin/connector/plane
  // traces (which are only sometimes present) -- so it is always index 0.
  const MAIN_TRACE_INDEX = 0;

  // Cluster-focus marker styling (Speaker Verification only), computed
  // independently of the `traces` useMemo so that toggling focus never
  // changes `traces`' own output reference -- applied imperatively via
  // Plotly.restyle() below instead of by feeding a new `data` prop into
  // <Plot>. Priority matches the pre-existing selection rule: an
  // individually/pair-selected point always stays gold and fully visible;
  // the focused cluster's own points keep their normal color; every other
  // cluster's points are lightened toward white. Index-aligned with
  // `externalData` directly (not `text`/`getPlotData()`), since the main
  // trace's point order for isExternal is exactly `externalData`'s order.
  const focusStyles = useMemo(() => {
    if (!isExternal || !externalData) return null;
    const colors = externalData.map((point) => {
      if (externalSelectedLabels?.includes(point.label)) return '#FFD700';
      if (focusedClusterId && point.clusterId !== focusedClusterId) {
        return mixWithWhite(point.color, 0.7);
      }
      return point.color;
    });
    const opacities = externalData.map((point) => {
      if (externalSelectedLabels?.includes(point.label)) return 1.0;
      if (focusedClusterId && point.clusterId !== focusedClusterId) return 0.55;
      return 0.85;
    });
    return { colors, opacities };
  }, [isExternal, externalData, externalSelectedLabels, focusedClusterId]);

  // Always mirrors the latest focusStyles into a ref, read by the stable
  // (never-recreated) applyFocusStyles callback below -- assigned directly
  // during render (not inside an effect) so it's already current by the
  // time any effect/lifecycle callback runs after this render commits.
  const focusStylesRef = useRef(focusStyles);
  focusStylesRef.current = focusStyles;

  // The actual Plotly graph div, but ONLY once react-plotly.js confirms via
  // onInitialized/onUpdate that Plotly has genuinely finished creating or
  // rebuilding it. plotRef.current?.el becomes non-null as soon as the
  // underlying <div> mounts -- well before Plotly has populated
  // graphDiv.data -- and restyling against that premature reference is
  // exactly what crashed the app (Plotly's coerceTraceIndices reading
  // `.length` off an undefined graphDiv.data). onInitialized/onUpdate fire
  // strictly after Plotly.react()'s own promise resolves (see
  // react-plotly.js factory.js: figureCallback runs inside the same
  // .then() chain as the Plotly.react() call), which is the only reliable
  // readiness signal.
  const readyGraphDivRef = useRef<any>(null);

  const arraysEqual = (a: unknown[], b: unknown[]) => {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) return false;
    }
    return true;
  };

  // Applies the latest focus styles to the main trace, but only when the
  // graph is actually ready for it -- never assumed from a non-null
  // graphDiv alone. Has a stable identity (reads only refs, no props/state
  // in its closure) so it's safe to call from onInitialized/onUpdate/the
  // toggle effect below without ever having to recreate any of them.
  //
  // react-plotly.js also listens for Plotly's own 'plotly_restyle' DOM
  // event (among others) and re-invokes onUpdate whenever it fires --
  // including for a restyle THIS function itself just issued, not only
  // ones caused by a genuine Plotly.react() redraw. Comparing the trace's
  // current marker.color/opacity against the styles about to be applied,
  // and bailing out when they already match, is what breaks that loop --
  // robust regardless of exactly when the DOM event fires relative to the
  // restyle promise resolving, unlike a timing-based "in-flight" flag would be.
  const applyFocusStyles = useCallback((graphDiv: any) => {
    const styles = focusStylesRef.current;
    if (!graphDiv || !styles) return;
    if (!Array.isArray(graphDiv.data) || graphDiv.data.length <= MAIN_TRACE_INDEX) return;

    const mainTrace = graphDiv.data[MAIN_TRACE_INDEX];
    const pointCount = Array.isArray(mainTrace?.x) ? mainTrace.x.length : undefined;
    if (pointCount === undefined || styles.colors.length !== pointCount || styles.opacities.length !== pointCount) {
      return;
    }

    const currentColors = mainTrace?.marker?.color;
    const currentOpacities = mainTrace?.marker?.opacity;
    if (
      Array.isArray(currentColors) &&
      Array.isArray(currentOpacities) &&
      arraysEqual(currentColors, styles.colors) &&
      arraysEqual(currentOpacities, styles.opacities)
    ) {
      return; // Already showing these exact styles.
    }

    Plotly.restyle(
      graphDiv,
      {
        'marker.color': [styles.colors],
        'marker.opacity': [styles.opacities],
      },
      [MAIN_TRACE_INDEX]
    ).catch((err: unknown) => {
      console.error('Cluster-focus marker restyle failed:', err);
    });
  }, []);

  const handlePlotInitialized = useCallback((_figure: any, graphDiv: any) => {
    readyGraphDivRef.current = graphDiv;
    applyFocusStyles(graphDiv);
  }, [applyFocusStyles]);

  // Reapplies focus styles after every Plotly.react()-driven redraw (e.g. a
  // genuinely new batch/projection, or an is3D toggle) -- traces never
  // encodes focus lightening itself (see focusStyles above), so any redraw
  // that rebuilds the main trace resets its colors back to the
  // selection-only base and needs focus reapplied on top.
  const handlePlotUpdate = useCallback((_figure: any, graphDiv: any) => {
    readyGraphDivRef.current = graphDiv;
    applyFocusStyles(graphDiv);
  }, [applyFocusStyles]);

  const handlePlotPurge = useCallback(() => {
    readyGraphDivRef.current = null;
  }, []);

  // Normal cluster-focus toggles (clicking a cluster card, clearing focus)
  // -- never calls Plotly.react()/setEmbeddingDataDirect/a reprojection
  // request, only restyle() via applyFocusStyles.
  useEffect(() => {
    applyFocusStyles(readyGraphDivRef.current);
  }, [focusStyles, applyFocusStyles]);

  const layout = useMemo(() => {
    // Layout configuration
    const result: any = {
      autosize: true,
      margin: { l: 35, r: 35, t: 35, b: 35 },
      plot_bgcolor: 'white',
      paper_bgcolor: 'white',
      showlegend: false,
      font: {
        size: 11,
        color: '#374151'
      },
      dragmode: is3D ? 'orbit' : (selectionMode === 'box' ? 'select' : 'lasso'),
      hovermode: 'closest',
      // Changes only when the projection genuinely changes (new model/
      // reduction-method/dimension/batch, or 2D<->3D) -- never on a mere
      // point/pair/table selection. Combined with feeding the live-captured
      // camera/range back in below, this is what keeps the view stable
      // across selection while still resetting for a real new projection.
      uirevision: revisionKey
    };

    if (is3D) {
      // Axis titles reflect the active reduction method, e.g. "PCA 1"/"UMAP 2"/"t-SNE 3".
      const axisTitle = (n: 1 | 2 | 3) => `${methodLabel} ${n}`;

      // 3D scene configuration
      result.scene = {
        xaxis: {
          showgrid: true,
          gridcolor: '#e5e7eb',
          showticklabels: true,
          tickfont: { size: 9, color: '#6b7280' },
          title: { text: axisTitle(1), font: { size: 10 } },
          backgroundcolor: 'white',
          showspikes: false,
          zeroline: true,
          zerolinecolor: '#d1d5db',
          showline: true,
          linecolor: '#9ca3af'
        },
        yaxis: {
          showgrid: true,
          gridcolor: '#e5e7eb',
          showticklabels: true,
          tickfont: { size: 9, color: '#6b7280' },
          title: { text: axisTitle(2), font: { size: 10 } },
          backgroundcolor: 'white',
          showspikes: false,
          zeroline: true,
          zerolinecolor: '#d1d5db',
          showline: true,
          linecolor: '#9ca3af'
        },
        zaxis: {
          showgrid: true,
          gridcolor: '#e5e7eb',
          showticklabels: true,
          tickfont: { size: 9, color: '#6b7280' },
          title: { text: axisTitle(3), font: { size: 10 } },
          backgroundcolor: 'white',
          showspikes: false,
          zeroline: true,
          zerolinecolor: '#d1d5db',
          showline: true,
          linecolor: '#9ca3af'
        },
        bgcolor: 'white',
        aspectmode: 'cube',
        dragmode: 'orbit'
      };
      // Only explicitly set `camera` on the render that just reset it for a
      // genuine new projection. On every other render, leave it unset and
      // let Plotly's own uirevision-driven preservation keep whatever the
      // user is currently looking at -- explicitly re-specifying `camera`
      // on every render (even to a captured "latest known" value) is what
      // actually overrides the live interactive camera: Plotly always
      // treats an explicit value as an instruction to set it, real
      // interactive rotation notwithstanding.
      if (didResetViewThisRender) {
        result.scene.camera = DEFAULT_CAMERA;
      }
    } else {
      // 2D axis configuration with enhanced zoom support
      result.xaxis = {
        showgrid: true,
        gridcolor: '#e5e7eb',
        showticklabels: false,
        title: { text: 'X', font: { size: 10 } },
        zeroline: true,
        zerolinecolor: '#d1d5db',
        zerolinewidth: 1,
        fixedrange: false // Allow zoom
      };
      result.yaxis = {
        showgrid: true,
        gridcolor: '#e5e7eb',
        showticklabels: false,
        title: { text: 'Y', font: { size: 10 } },
        zeroline: true,
        zerolinecolor: '#d1d5db',
        zerolinewidth: 1,
        fixedrange: false // Allow zoom
      };
      // Same reasoning as the 3D camera above (and empirically confirmed
      // the same way): explicitly re-specifying xaxis/yaxis.range on every
      // render -- even to a captured "latest known" value -- is what
      // overrides Plotly's own uirevision-driven preservation of the live
      // pan/zoom (whether from a native Plotly zoom or this file's own
      // right-drag Plotly.relayout calls). Only force a fresh autorange on
      // the render that just reset for a genuine new projection; leave
      // range/autorange untouched on every other render.
      if (didResetViewThisRender) {
        result.xaxis.autorange = true;
        result.yaxis.autorange = true;
      }
    }

    // Add compact annotation
    if (embeddingData || isExternal) {
      const fileCount = isExternal ? (externalData?.length ?? 0) : embeddingData!.total_files;
      result.annotations = [{
        text: `${fileCount} files • ${is3D ? '3D' : '2D'}`,
        xref: 'paper',
        yref: 'paper',
        x: 0.02,
        y: 0.98,
        xanchor: 'left',
        yanchor: 'top',
        font: { size: 9, color: '#6b7280' },
        showarrow: false,
        bgcolor: 'rgba(255,255,255,0.8)',
        bordercolor: '#e5e7eb',
        borderwidth: 1,
        borderpad: 2
      }];
    }

    return result;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [is3D, selectedPlane, selectionMode, selectedMethod, embeddingData, isExternal, externalData, revisionKey, methodLabel, didResetViewThisRender]);

  // Memoized so a focus-only rerender (which changes neither the reduction
  // method, dimensionality, nor the selected plane) doesn't hand <Plot> a
  // new config object reference -- react-plotly.js diffs data/layout/config
  // props to decide whether to call Plotly.react().
  const plotConfig = useMemo(
    () => ({
      displayModeBar: false, // Hide the mode bar completely
      displaylogo: false,
      responsive: true,
      autosizable: true,
      scrollZoom: true,
      doubleClick: 'reset+autosize' as const,
      showTips: false, // Hide hover tips
      toImageButtonOptions: {
        format: 'png' as const,
        filename: `embeddings_${selectedMethod}_${is3D ? '3D' : '2D'}${selectedPlane !== 'none' ? `_${selectedPlane}` : ''}`,
        height: 800,
        width: 800,
        scale: 2
      }
    }),
    [selectedMethod, is3D, selectedPlane]
  );

  if (verificationMode && !isExternal) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-xs text-muted-foreground text-center px-4">
          No batch analysis results to display yet.
        </div>
      </div>
    );
  }

  if (!isExternal && isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-xs text-muted-foreground flex items-center gap-2">
          <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
          Loading embeddings...
        </div>
      </div>
    );
  }

  if (!isExternal && error) {
    return (
      <div className="h-full flex items-center justify-center p-4">
        <div className="text-xs text-red-500 text-center">
          <div className="font-medium">⚠️ Error loading embeddings</div>
          <div className="mt-1">{error}</div>
        </div>
      </div>
    );
  }

  const instructionText = is3D
    ? 'Left drag: rotate • Right drag: pan • Wheel: zoom'
    : 'Left drag: select • Right drag: pan • Wheel: zoom';

  return (
    <div ref={plotContainerRef} className="w-full h-full min-h-0 relative">
      {/* Plane Selection Controls - Only show in 3D mode */}
      {is3D && (
        <div className="absolute top-2 right-2 z-10 bg-white/95 backdrop-blur-sm border border-gray-200 rounded-md p-2 shadow-sm">
          {/* Plane Selection */}
          <div className="flex items-center gap-2 mb-2">
            <Layers3 className="h-3 w-3 text-gray-600" />
            <span className="text-xs text-gray-600 font-medium">Plane:</span>
            <Select
              value={selectedPlane}
              onValueChange={(value: PlaneType) => setSelectedPlane(value)}
            >
              <SelectTrigger className="w-16 h-6 text-xs border-gray-300 hover:border-gray-400 transition-colors">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                <SelectItem value="xy">{methodLabel} 1–2</SelectItem>
                <SelectItem value="xz">{methodLabel} 1–3</SelectItem>
                <SelectItem value="yz">{methodLabel} 2–3</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          {/* Angle Range Selector */}
          {selectedPlane !== 'none' && (
            <div className="space-y-2 pt-2 border-t border-gray-200">
              <div className="flex items-center gap-1">
                <Target className="h-3 w-3 text-gray-600" />
                <span className="text-xs text-gray-600 font-medium">Angle Range:</span>
              </div>
              
              <div className="flex items-center gap-1">
                <Input
                  type="number"
                  min="0"
                  max="90"
                  step="1"
                  value={angleMin}
                  onChange={(e) => setAngleMin(Number(e.target.value))}
                  className="w-14 h-6 text-xs text-center px-1"
                />
                <span className="text-xs text-gray-500">-</span>
                <Input
                  type="number"
                  min="0"
                  max="90"
                  step="1"
                  value={angleMax}
                  onChange={(e) => setAngleMax(Number(e.target.value))}
                  className="w-14 h-6 text-xs text-center px-1"
                />
                <span className="text-xs text-gray-500">°</span>
              </div>
              
              {selectedByAngle.length > 0 && (
                <div className="text-[10px] text-red-600 bg-red-50 px-2 py-1 rounded">
                  🔴 {selectedByAngle.length} points selected
                </div>
              )}
            </div>
          )}
          
          {selectedPlane !== 'none' && (
            <div className="text-[10px] text-gray-500 mt-1">
              {selectedPlane === 'xy' && `🔵 Blue plane: ${methodLabel} 1–2`}
              {selectedPlane === 'xz' && `🟢 Green plane: ${methodLabel} 1–3`}
              {selectedPlane === 'yz' && `🔴 Red plane: ${methodLabel} 2–3`}
            </div>
          )}
        </div>
      )}

      {/* Interaction hint, bottom-left so it never collides with the
          top-left "N files" Plotly annotation or the top-right plane panel. */}
      <div className="absolute bottom-2 left-2 z-10 bg-white/95 backdrop-blur-sm border border-gray-200 rounded-md px-2 py-1 text-[10px] text-gray-500 pointer-events-none">
        {instructionText}
      </div>

      <Plot
        ref={plotRef}
        data={traces}
        layout={layout}
        onClick={handlePointClick}
        onSelected={handleSelection}
        onDeselect={handleDeselect}
        onInitialized={handlePlotInitialized}
        onUpdate={handlePlotUpdate}
        onPurge={handlePlotPurge}
        config={plotConfig}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
    </div>
  );
};

export const EmbeddingPlot = ({ selectedMethod = "pca", is3D = false, onPointSelect, onAngleRangeSelect, selectedFile, selectionMode = 'box', onSelectionChange, externalData, externalSelectedLabels, verificationMode }: EmbeddingPlotProps) => {
  return (
    <div className="w-full h-full min-h-0 relative">
      <EmbeddingPlotContent
        selectedMethod={selectedMethod}
        is3D={is3D}
        onPointSelect={onPointSelect}
        onAngleRangeSelect={onAngleRangeSelect}
        selectedFile={selectedFile}
        selectionMode={selectionMode}
        onSelectionChange={onSelectionChange}
        externalData={externalData}
        externalSelectedLabels={externalSelectedLabels}
        verificationMode={verificationMode}
      />
    </div>
  );
};