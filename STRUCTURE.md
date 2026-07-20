# VoxLIT Project Structure

VoxLIT is organized around **five analysis tasks**, each with its own page, reached
from the homepage. Two are implemented (Speech Transcription, Emotion Recognition);
three are placeholders owned by one team member each, to be finalized after the
mentor meeting.

## The one file everyone edits

**`Frontend/src/tasks/registry.tsx`** is the single source of truth for the UI:

| To do this… | …edit |
|---|---|
| Rename a task (after it's finalized) | the task's `name` / `shortDescription` |
| Add a model to any task | add a `ModelOption` (set `available: true` once the backend handles the id) |
| Add a dataset to any task | add a `DatasetOption` (id must match the backend dataset registry) |
| Activate a placeholder task | flip `status` to `"active"`, set `capabilities`, register components in `TASK_SLOTS` |

The backend has a small mirror registry in **`Backend/app/tasks/registry.py`**
(model/dataset ids only — served at `GET /tasks` for debugging). The only contract
between the two is the id strings.

Task `id`s and routes (`task-a`, `task-b`, `task-c`, …) are **stable forever** —
only display names change on rename.

## Frontend layout (`Frontend/src/`)

```
tasks/                  Central registry + shared types (types.ts is frozen)
pages/                  Home (landing page), TaskPage (thin wrapper), NotFound
components/
  workbench/            SHARED page shell: TaskWorkbench (3-column layout),
                        Toolbar (registry-driven dropdowns), ExplainabilityPanel
                        (capability-gated tabs), PlaceholderExplainability
  panels/               SHARED: EmbeddingPanel, AudioDatasetPanel, DatapointEditorPanel
  audio/ visualization/ SHARED: players, waveform, saliency/attention plots,
  analysis/ dataset/    perturbation tools, custom dataset manager
  ui/                   SHARED: shadcn/ui primitives
features/
  transcription/        TranscriptionResults card (Datapoint Editor slot)
  emotion/              ClassificationResults card
  task-a|b|c/           ← each member builds their task's components here
contexts/ hooks/ lib/   SHARED
```

Every task page renders `TaskWorkbench`, which keeps the **Audio Embeddings**
panel (left), **Audio Dataset** table (bottom center), and **Datapoint Editor**
(right) identical across tasks. Task-specific parts:

* the **center explainability panel** (tabs from `capabilities`, or a
  "to be implemented" placeholder), and
* the **results card** in the Datapoint Editor (from `TASK_SLOTS`).

## Backend layout (`Backend/app/`)

```
main.py               Registers legacy routers + mounts each task router at /tasks/<id>
api/routes/           SHARED/frozen — powers transcription + emotion (do not edit)
services/             SHARED/frozen — model loading, saliency, perturbation, datasets
core/                 SHARED — Redis, session, settings
tasks/
  registry.py         Backend task registry (models/datasets per task)
  router.py           GET /tasks
  transcription/      TASK_INFO only (uses legacy endpoints)
  emotion/            TASK_INFO only
  task_a|b|c/         ← each member's router.py + service.py
```

### Adding your task's backend (per member)

1. Implement endpoints in `app/tasks/task_x/router.py` (already mounted at
   `/tasks/task-x`) and model loading in `app/tasks/task_x/service.py`
   (copy the thread-safe `_load_once` lazy-cache idiom — see the docstring).
2. Add your models/datasets to `app/tasks/registry.py`.
3. If you want the shared `/{dataset}/metadata` + `/{dataset}/file/...` routes for
   a new built-in dataset, add one entry each to `DATASET_PATHS`/`DATASET_BASE_DIRS`
   in `app/services/dataset_service.py` and `DATASET_DIRS` in
   `app/api/routes/inferences.py`, and drop the files under `Backend/data/`.

**Total shared-file footprint of a new task:** one line in
`app/tasks/__init__.py` (already done for task_a/b/c), one entry in each
registry, and your own folders. Merge conflicts between members are limited to
adjacent one-line edits in the registries.

## Running the app

```bash
# Terminal 1 — Redis
cd Backend && docker-compose up

# Terminal 2 — API (must run from Backend/)
cd Backend && .venv\Scripts\activate && uvicorn app.main:app --reload

# Terminal 3 — Frontend (http://localhost:8080)
cd Frontend && npm run dev
```
