import { ReactNode, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  AudioWaveform,
  ArrowRight,
  Brain,
  ChevronDown,
  Database,
  Github,
  Layers,
  ScatterChart,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import { TASKS } from "@/tasks/registry";
import { TaskDefinition } from "@/tasks/types";

/** Fades and rises children into view the first time they are scrolled to. */
const Reveal = ({ children, delay = 0 }: { children: ReactNode; delay?: number }) => {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setVisible(true);
      },
      { threshold: 0.15 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      style={{ transitionDelay: `${delay}ms` }}
      className={`h-full transition-all duration-500 ease-out ${
        visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"
      }`}
    >
      {children}
    </div>
  );
};

/** Decorative animated equalizer bars — the audio motif of the hero. */
const EqualizerBars = () => {
  // Deterministic pseudo-random heights so the row looks organic but stable
  const heights = Array.from({ length: 24 }, (_, i) => 14 + ((i * 37) % 5) * 7 + (i % 3) * 4);
  return (
    <div className="flex items-end justify-center gap-1.5 h-14" aria-hidden="true">
      {heights.map((h, i) => (
        <div
          key={i}
          className="eq-bar w-1 rounded-full bg-primary/40"
          style={{ height: `${h}px`, animationDelay: `${i * 0.07}s` }}
        />
      ))}
    </div>
  );
};

const capabilityBadges = (task: TaskDefinition) => {
  const caps: string[] = [];
  if (task.capabilities.saliency) caps.push("Saliency");
  if (task.capabilities.attention) caps.push("Attention");
  if (task.capabilities.perturbation) caps.push("Perturbation");
  return caps;
};

const TaskCard = ({ task }: { task: TaskDefinition }) => {
  const isActive = task.status === "active";
  const availableModels = task.models.filter((m) => m.available);
  const availableDatasets = task.datasets.filter((d) => d.available);
  const caps = capabilityBadges(task);

  return (
    <Card className={`h-full flex flex-col transition-shadow hover:shadow-aws-md ${isActive ? "" : "opacity-80"}`}>
      <CardHeader className="bg-panel-header">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm">{task.name}</CardTitle>
          {isActive ? (
            <Badge variant="outline" className="text-[10px] bg-green-50 text-green-700 border-green-200">
              Available
            </Badge>
          ) : (
            <Badge variant="outline" className="text-[10px] bg-muted text-muted-foreground">
              Coming soon
            </Badge>
          )}
        </div>
        <CardDescription className="text-xs leading-relaxed">{task.shortDescription}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col flex-1 gap-3 pt-3">
        <div className="flex flex-wrap gap-1.5">
          {availableModels.length > 0 ? (
            availableModels.map((m) => (
              <Badge key={m.id} variant="secondary" className="text-[10px]">
                <Brain className="h-2.5 w-2.5 mr-1" />
                {m.label}
              </Badge>
            ))
          ) : (
            <Badge variant="outline" className="text-[10px] text-muted-foreground border-dashed">
              <Brain className="h-2.5 w-2.5 mr-1" />
              Models to be added
            </Badge>
          )}
          {availableDatasets.length > 0 ? (
            availableDatasets.map((d) => (
              <Badge key={d.id} variant="secondary" className="text-[10px]">
                <Database className="h-2.5 w-2.5 mr-1" />
                {d.label}
              </Badge>
            ))
          ) : (
            <Badge variant="outline" className="text-[10px] text-muted-foreground border-dashed">
              <Database className="h-2.5 w-2.5 mr-1" />
              Datasets to be added
            </Badge>
          )}
        </div>
        {caps.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {caps.map((c) => (
              <Badge key={c} variant="outline" className="text-[10px] bg-primary/5 text-primary border-primary/20">
                {c}
              </Badge>
            ))}
          </div>
        )}
        <div className="mt-auto">
          <Button asChild variant={isActive ? "default" : "outline"} size="sm" className="w-full h-8 text-xs shadow-aws-sm">
            <Link to={task.route}>
              {isActive ? "Open workbench" : "Preview workbench"}
              <ArrowRight className="h-3.5 w-3.5 ml-1.5" />
            </Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

const Home = () => {
  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="h-12 bg-white border-b border-border px-5 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <AudioWaveform className="h-4 w-4 text-primary" />
          <span className="text-base font-bold text-foreground">VoxLIT</span>
          <Badge variant="outline" className="text-[10px] bg-primary/10 text-primary border-primary/20">
            v1.0
          </Badge>
          <span className="hidden sm:inline text-xs text-muted-foreground ml-2">
            Learning Interpretability Tool for Voice Models
          </span>
        </div>
        <Button asChild variant="ghost" size="sm" className="h-7 text-xs">
          <a href="https://github.com/chanuGX/VoxLIT" target="_blank" rel="noreferrer">
            <Github className="h-3.5 w-3.5 mr-1.5" />
            GitHub
          </a>
        </Button>
      </header>

      <main className="flex-1">
        {/* Hero — fills most of the first viewport so the tasks section peeks below */}
        <section className="px-5 py-10 min-h-[calc(85vh-3rem)] flex flex-col items-center justify-center gap-8 border-b border-border bg-panel-background">
          <div className="max-w-5xl mx-auto text-center space-y-4">
            <Badge variant="outline" className="text-[10px] bg-primary/5 text-primary border-primary/20">
              <Sparkles className="h-2.5 w-2.5 mr-1" />
              Extending Google&apos;s LIT to audio models
            </Badge>
            <h1 className="text-3xl font-bold text-foreground">
              Interpret how voice models make decisions
            </h1>
            <p className="text-sm text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              VoxLIT brings the interpretability paradigm of the Learning Interpretability Tool to
              speech: explore embeddings, saliency, attention, and perturbation robustness of
              transformer-based audio models — interactively, one datapoint at a time.
            </p>
          </div>

          <EqualizerBars />

          {/* Scroll cue */}
          <button
            onClick={() =>
              document.getElementById("analysis-tasks")?.scrollIntoView({ behavior: "smooth" })
            }
            className="flex flex-col items-center gap-1 text-xs text-muted-foreground hover:text-primary transition-colors"
          >
            Explore the analysis tasks
            <ChevronDown className="h-5 w-5 animate-bounce" />
          </button>
        </section>

        {/* Task grid */}
        <section id="analysis-tasks" className="px-5 pt-6 pb-10 scroll-mt-14">
          <div className="max-w-5xl mx-auto space-y-4">
            <div>
              <h2 className="text-lg font-semibold text-foreground">Analysis tasks</h2>
              <p className="text-xs text-muted-foreground mt-1">
                Each task opens a dedicated workbench with its own models, datasets, and
                explainability tools.
              </p>
            </div>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {TASKS.map((task, index) => (
                <Reveal key={task.id} delay={index * 80}>
                  <TaskCard task={task} />
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        {/* How it works */}
        <section className="px-5 py-10 border-t border-border bg-panel-background">
          <div className="max-w-5xl mx-auto space-y-4">
            <h2 className="text-lg font-semibold text-foreground">The workbench</h2>
            <div className="grid gap-4 md:grid-cols-3">
              <Reveal>
                <Card className="h-full">
                  <CardContent className="p-4 space-y-2">
                    <ScatterChart className="h-5 w-5 text-primary" />
                    <div className="text-sm font-medium">Audio Embeddings</div>
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      Project high-dimensional audio representations into 2D/3D with PCA, UMAP, or
                      t-SNE. Select points to inspect individual samples.
                    </p>
                  </CardContent>
                </Card>
              </Reveal>
              <Reveal delay={100}>
                <Card className="h-full">
                  <CardContent className="p-4 space-y-2">
                    <Layers className="h-5 w-5 text-primary" />
                    <div className="text-sm font-medium">Explainability</div>
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      Task-specific analyses in the center panel: gradient saliency over the
                      waveform, attention patterns, and robustness under audio perturbations.
                    </p>
                  </CardContent>
                </Card>
              </Reveal>
              <Reveal delay={200}>
                <Card className="h-full">
                  <CardContent className="p-4 space-y-2">
                    <SlidersHorizontal className="h-5 w-5 text-primary" />
                    <div className="text-sm font-medium">Datapoint Editor</div>
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      Inspect sample metadata, model predictions with ground-truth metrics, and
                      listen to original vs. perturbed audio side by side.
                    </p>
                  </CardContent>
                </Card>
              </Reveal>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="px-5 py-4 border-t border-border">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-muted-foreground">
          <span>
            Inspired by the{" "}
            <a
              href="https://pair-code.github.io/lit/"
              target="_blank"
              rel="noreferrer"
              className="underline hover:text-primary"
            >
              Learning Interpretability Tool (LIT)
            </a>
          </span>
          <span>Datasets: Mozilla Common Voice · RAVDESS</span>
        </div>
      </footer>
    </div>
  );
};

export default Home;
