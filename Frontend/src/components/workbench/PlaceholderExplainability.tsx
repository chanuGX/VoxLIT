import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Construction } from "lucide-react";

interface PlaceholderExplainabilityProps {
  taskName: string;
}

/**
 * Center-panel placeholder for tasks whose explainability tooling has not been
 * implemented yet. Mirrors the tabbed layout of ExplainabilityPanel so the
 * page reads the same across tasks.
 */
export const PlaceholderExplainability = ({ taskName }: PlaceholderExplainabilityProps) => {
  return (
    <div className="h-full bg-panel-background border-t border-border">
      <Tabs value="placeholder" className="h-full flex flex-col">
        <div className="flex-shrink-0 bg-panel-header border-b border-border px-3 py-2">
          <TabsList className="h-7 grid w-full grid-cols-1 bg-muted">
            <TabsTrigger value="placeholder" disabled className="text-xs">
              Explainability
            </TabsTrigger>
          </TabsList>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto bg-background p-3 scrollbar-thin flex flex-col">
          <Card className="flex-1">
            <CardContent className="h-full flex flex-col items-center justify-center gap-3 text-center p-6">
              <Construction className="h-8 w-8 text-muted-foreground" />
              <div className="text-sm font-medium text-foreground">
                Explainability for {taskName}
              </div>
              <Badge variant="outline" className="text-[10px] bg-muted text-muted-foreground">
                To be implemented
              </Badge>
              <p className="text-xs text-muted-foreground max-w-sm leading-relaxed">
                Task-specific explainability tools (and their models and datasets) will appear
                here once this task is implemented. Register components and flip the task to
                &quot;active&quot; in <span className="font-mono">src/tasks/registry.tsx</span>.
              </p>
            </CardContent>
          </Card>
        </div>
      </Tabs>
    </div>
  );
};
