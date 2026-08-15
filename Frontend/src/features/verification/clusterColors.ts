const BASE_PALETTE = [
  "#2563eb",
  "#dc2626",
  "#16a34a",
  "#d97706",
  "#7c3aed",
  "#0891b2",
  "#db2777",
  "#65a30d",
  "#ea580c",
  "#4f46e5",
  "#0d9488",
  "#be123c",
];

export function colorForClusterIndex(index: number): string {
  if (index < BASE_PALETTE.length) {
    return BASE_PALETTE[index];
  }
  const hue = (index * 137.508) % 360;
  return `hsl(${hue.toFixed(1)}, 65%, 45%)`;
}

export function buildClusterColorMap(clusterLabels: string[]): Record<string, string> {
  const order: string[] = [];
  const seen = new Set<string>();
  for (const id of clusterLabels) {
    if (!seen.has(id)) {
      seen.add(id);
      order.push(id);
    }
  }
  return Object.fromEntries(order.map((id, index) => [id, colorForClusterIndex(index)]));
}
