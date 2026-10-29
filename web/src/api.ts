// Typed client for the AI Made Easy server. Every call goes to /api; errors carry the
// server's user-facing message.
import type {
  AutoMLStarted,
  GroupThresholds,
  GroupThresholdsUsed,
  Counterfactuals,
  BackendList,
  BackendTest,
  BlockDef,
  BudgetReport,
  DataFacts,
  DataProfile,
  DesignDiff,
  DeviceProfile,
  Explanation,
  ExportFormat,
  FamilyInfo,
  FeatureImportance,
  FetchedDataset,
  GroupColumn,
  GroupResult,
  LRFindResult,
  ModelVersion,
  PipelineRecord,
  Project,
  RecipeInfo,
  RecipeSuggestion,
  ResumeInfo,
  RunEvent,
  RunRecord,
  RunSamples,
  SavedProject,
  SplitPreview,
  Summary,
  SweepAgain,
  SweepAnalysis,
  SweepParam,
  SweepRecord,
  TableLayout,
  TaskInfo,
  TrackingStatus,
  TryForm,
  Validation,
  WizardTask,
} from "./types";

export class ApiError extends Error {}

/** Links the browser downloads itself (no fetch headers) carry the access token. */
function withToken(url: string): string {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}` : url;
}

const TOKEN_KEY = "aime.token";

function headers(json: boolean): HeadersInit {
  const h: Record<string, string> = {};
  if (json) h["Content-Type"] = "application/json";
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) h["Authorization"] = `Bearer ${token}`;
  return h;
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const raw = body instanceof FormData || body instanceof Blob;  // sent as is (uploads)
  const res = await fetch(`/api${path}`, {
    method,
    headers: headers(body !== undefined && !raw),
    body: body === undefined ? undefined : raw ? body : JSON.stringify(body),
  });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      message = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* keep the status line */
    }
    throw new ApiError(message);
  }
  const type = res.headers.get("content-type") ?? "";
  return (type.includes("json") ? res.json() : res.text()) as Promise<T>;
}

const get = <T>(path: string) => request<T>("GET", path);
const post = <T>(path: string, body?: unknown) => request<T>("POST", path, body ?? {});

export const api = {
  setToken: (token: string) => localStorage.setItem(TOKEN_KEY, token),
  info: () => get<{ version: string; auth: boolean; blocks: number }>("/info"),

  blocks: () => get<{ blocks: BlockDef[] }>("/blocks"),
  samples: () => get<{ samples: string[] }>("/samples"),
  tasks: () => get<{ tasks: TaskInfo[] }>("/tasks"),
  families: () => get<{ families: FamilyInfo[] }>("/families"),
  describe: (graph: Project) =>
    post<{ family: FamilyInfo; task: TaskInfo | null; frameworks: string[];
           framework: string | null }>("/describe", { graph }),
  sample: (name: string) => get<Project>(`/samples/${encodeURIComponent(name)}`),

  validate: (graph: Project) => post<Validation>("/validate", { graph }),
  fix: (graph: Project, index: number) =>
    post<{ label: string; graph: Project }>("/fix", { graph, index }),
  targets: () => get<{ targets: { id: string; label: string }[] }>("/targets"),
  generate: (graph: Project, target: string) =>
    post<{ code: string }>("/generate", { graph, target }),
  summary: (graph: Project) => post<Summary>("/summary", { graph }),
  devices: () => get<{ devices: DeviceProfile[] }>("/devices"),
  budget: (graph: Project) => post<BudgetReport>("/budget", { graph }),
  tableLayout: (graph: Project, node: string) =>
    post<TableLayout>("/table_layout", { graph, node }),
  expand: (graph: Project, nodeId: string) =>
    post<{ graph: Project }>("/expand", { graph, node_id: nodeId }),

  projects: () => get<{ projects: SavedProject[] }>("/projects"),
  project: (name: string) => get<Project>(`/projects/${encodeURIComponent(name)}`),
  saveProject: (name: string, graph: Project) =>
    request<{ saved: string }>("PUT", `/projects/${encodeURIComponent(name)}`, graph),
  deleteProject: (name: string) =>
    request<{ deleted: string }>("DELETE", `/projects/${encodeURIComponent(name)}`),

  train: (graph: Project, project: string, framework = "auto", on = "local") =>
    post<{ run_id: string; backend: string }>("/runs", { graph, project, framework, on }),
  attachRun: (id: string) => post<Record<string, unknown>>(`/runs/${id}/attach`),
  importResults: (zip: Blob) => post<{ run_id: string; state: string }>("/runs/import", zip),

  backends: () => get<BackendList>("/backends"),
  saveBackend: (name: string, kind: string, options: Record<string, unknown>) =>
    post<Record<string, unknown>>("/backends", { name, kind, options }),
  testBackend: (on: string) => post<BackendTest>("/backends/test", { on }),
  deleteBackend: (name: string) =>
    request<{ deleted: string }>("DELETE", `/backends/${encodeURIComponent(name)}`),
  runs: (project?: string) =>
    get<{ runs: RunRecord[] }>(`/runs${project ? `?project=${encodeURIComponent(project)}` : ""}`),
  run: (id: string) => get<RunRecord>(`/runs/${id}`),
  runExplain: (id: string) => get<FeatureImportance>(`/runs/${id}/explain`),
  runDiff: (id: string, graph: Project) => post<DesignDiff>(`/runs/${id}/diff`, { graph }),
  runSamples: (id: string) => get<RunSamples>(`/runs/${id}/samples`),
  tryForm: (id: string) => get<TryForm>(`/runs/${id}/try`),
  /** 3.11: the smallest changes to a table row that change the prediction. */
  whatWouldChange: (id: string, body: { row: Record<string, unknown>; desired?: string;
                                        target?: number; locked?: string[] }) =>
    post<Counterfactuals>(`/runs/${id}/counterfactuals`, body),
  tryRun: (id: string, inputs: unknown[], explain = false) =>
    post<{ predictions: Record<string, unknown>[]; latency_ms: number }>(`/runs/${id}/try`,
                                                                          { inputs, explain }),
  /** Predictions for every input in a file, as CSV text. */
  tryFile: (id: string, file: File, changes = false) => {
    const form = new FormData();
    form.append("file", file);
    return post<string>(`/runs/${id}/try/file${changes ? "?changes=true" : ""}`, form);
  },
  metrics: (id: string) => get<{ epochs: { epoch: number; metrics: Record<string, number> }[] }>(
    `/runs/${id}/metrics`),
  stopRun: (id: string) => post<Record<string, unknown>>(`/runs/${id}/stop`),
  deleteRun: (id: string) => request<{ deleted: string }>("DELETE", `/runs/${id}`),
  tagRun: (id: string, tags: string[], note?: string) =>
    request<RunRecord>("PATCH", `/runs/${id}`, { tags, note }),
  compare: (ids: string[]) => post<Record<string, unknown>>("/runs/compare", { run_ids: ids }),

  sweepParams: (graph: Project) => post<{ params: SweepParam[] }>("/sweeps/params", { graph }),
  startSweep: (graph: Project, spec: Record<string, unknown>, project: string) =>
    post<{ sweep_id: string }>("/sweeps", { graph, spec, project }),
  sweeps: () => get<{ sweeps: SweepRecord[] }>("/sweeps"),
  sweep: (id: string) => get<SweepRecord>(`/sweeps/${id}`),
  stopSweep: (id: string) => post<Record<string, unknown>>(`/sweeps/${id}/stop`),
  bestGraph: (id: string) => get<{ graph: Project }>(`/sweeps/${id}/best`),
  sweepAnalysis: (id: string) => get<SweepAnalysis>(`/sweeps/${id}/analysis`),
  sweepAgain: (id: string) => get<SweepAgain>(`/sweeps/${id}/again`),
  // 3.8: find a learning rate, train more, results by group
  findLearningRate: (graph: Project, steps?: number) =>
    post<{ run_id: string }>("/lr-find", { graph, steps }),
  learningRateResult: (id: string) => get<LRFindResult>(`/lr-find/${id}`),
  applyLearningRate: (graph: Project, lr: number) =>
    post<{ graph: Project }>("/lr-find/apply", { graph, lr }),
  trainMore: (id: string, epochs: number, lrScale = 1, trees?: number) =>
    post<{ run_id: string; continues: string }>(`/runs/${id}/more`,
                                                { epochs, lr_scale: lrScale, trees }),
  groupColumns: (id: string) => get<{ columns: GroupColumn[] }>(`/runs/${id}/groups/columns`),
  resultsByGroup: (id: string, column: string) =>
    post<{ run_id: string }>(`/runs/${id}/groups`, { column }),
  groupResult: (id: string) => get<GroupResult>(`/groups/${id}`),
  useGroupThresholds: (scoringRun: string) =>
    post<GroupThresholdsUsed>(`/groups/${scoringRun}/use`, {}),
  groupThresholds: (runId: string) =>
    get<{ run_id: string; thresholds: GroupThresholds | null }>(`/runs/${runId}/group-thresholds`),
  stopGroupThresholds: (runId: string) =>
    request<{ run_id: string; removed: boolean }>("DELETE", `/runs/${runId}/group-thresholds`),
  // 3.9: resume an interrupted run
  resumeInfo: (id: string) => get<ResumeInfo>(`/runs/${id}/resume`),
  resumeRun: (id: string) => post<{ run_id: string; after_epoch: number }>(`/runs/${id}/resume`, {}),

  startPipeline: (graph: Project, project = "", on = "local") =>
    post<{ pipeline_id: string; stages: string[] }>("/pipelines", { graph, project, on }),
  pipelines: () => get<{ pipelines: PipelineRecord[] }>("/pipelines"),
  pipeline: (id: string) => get<PipelineRecord>(`/pipelines/${id}`),
  stopPipeline: (id: string) => post<Record<string, unknown>>(`/pipelines/${id}/stop`),
  resumePipeline: (id: string) =>
    post<{ pipeline_id: string; resumed_from: string }>(`/pipelines/${id}/resume`),

  wizardTasks: () => get<{ tasks: WizardTask[] }>("/wizard/tasks"),
  uploadData: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return post<FetchedDataset>("/data/upload", form);
  },
  detect: (path: string, target = "", task = "") =>
    post<DataFacts>("/wizard/detect", { path, target, task }),
  recommend: (body: { task: string; facts?: DataFacts | null; budget?: Record<string, unknown>;
                      modality?: string }) =>
    post<{ facts: DataFacts; suggestions: RecipeSuggestion[] }>("/wizard/recommend", body),
  buildRecipe: (body: { recipe: string; task: string; facts?: DataFacts | null;
                        knobs?: Record<string, unknown>; budget?: Record<string, unknown> }) =>
    post<{ graph: Project }>("/wizard/build", body),
  recipes: (task?: string) =>
    get<{ recipes: RecipeInfo[] }>(`/recipes${task ? `?task=${encodeURIComponent(task)}` : ""}`),
  explain: (graph: Project) => post<Explanation>("/explain", { graph }),
  startAutoml: (spec: Record<string, unknown>, project = "") =>
    post<AutoMLStarted>("/automl", { spec, project }),

  formats: (runId: string) => get<{ formats: ExportFormat[] }>(`/runs/${runId}/formats`),
  deployUrl: (runId: string, formats: string[]) =>
    withToken(`/api/runs/${runId}/deploy?formats=${encodeURIComponent(formats.join(","))}`),
  ggufUrl: (runId: string, quant: string) =>
    withToken(`/api/runs/${runId}/gguf?quant=${encodeURIComponent(quant)}`),
  models: () => get<{ models: ModelVersion[] }>("/models"),
  registerModel: (runId: string, name: string, description = "") =>
    post<ModelVersion>("/models", { run_id: runId, name, description }),
  setStage: (name: string, version: number, stage: string) =>
    request<ModelVersion>("PATCH", `/models/${encodeURIComponent(name)}/${version}`, { stage }),
  deleteModel: (name: string, version: number) =>
    request<Record<string, unknown>>("DELETE", `/models/${encodeURIComponent(name)}/${version}`),
  modelFormats: (name: string, version: number) =>
    get<{ formats: ExportFormat[] }>(`/models/${encodeURIComponent(name)}/${version}/formats`),
  modelDeployUrl: (name: string, version: number, formats: string[]) =>
    withToken(`/api/models/${encodeURIComponent(name)}/${version}/deploy?formats=` +
              encodeURIComponent(formats.join(","))),
  mirrorModel: (name: string, version: number, to: string[]) =>
    post<ModelVersion>(`/models/${encodeURIComponent(name)}/${version}/mirror`, { to }),
  pushToHub: (repoId: string, model: string, version: number, isPrivate = true) =>
    post<{ repo_id: string; url: string }>("/hub/push", { repo_id: repoId, model, version,
                                                          private: isPrivate }),
  deploySpace: (spaceId: string, model: string, version: number, isPrivate = true,
                formats: string[] = []) =>
    post<{ space_id: string; url: string }>("/hub/space", { space_id: spaceId, model, version,
                                                            private: isPrivate, formats }),
  pullFromHub: (repoId: string, name = "") =>
    post<ModelVersion>("/hub/pull", { repo_id: repoId, name }),

  tracking: (graph?: Project) => graph ? post<TrackingStatus>("/tracking", { graph })
    : get<TrackingStatus>("/tracking"),
  fetchDataset: (source: string, force = false) =>
    post<FetchedDataset>("/data/fetch", { source, force }),
  exportRunsUrl: (format: "csv" | "parquet", epochs = false) =>
    withToken(`/api/export/runs?format=${format}${epochs ? "&epochs=true" : ""}`),
  importMlflow: (uri: string, inputShape: string) =>
    post<{ graph: Project; summary: string; ok: boolean; warnings: string[];
           unsupported: string[] }>("/import/mlflow", { uri, input_shape: inputShape }),

  importModel: (form: FormData) =>
    post<{ graph: Project; summary: string; ok: boolean; warnings: string[];
           unsupported: string[] }>("/import", form),

  profile: (graph: Project, nodeId?: string) =>
    post<DataProfile>("/data/profile", { graph, node_id: nodeId ?? null }),
  dataIssues: (graph: Project) =>
    post<{ issues: { severity: string; message: string; node_id: string }[] }>(
      "/data/issues", { graph }),
  split: (graph: Project) => post<SplitPreview>("/data/split", { graph }),
  augment: (graph: Project) =>
    post<{ images: { source: string; original: string; eval: string; train: string[] }[];
           train_transforms: string[]; skipped: string[] }>("/data/augment", { graph }),
};

/** Live run events: a WebSocket that replays the epochs so far, then streams new events.
 *  Falls back to polling when WebSockets are unavailable (e.g. behind some proxies). */
export function streamRun(runId: string, onEvent: (event: RunEvent) => void): () => void {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const token = localStorage.getItem(TOKEN_KEY);
  const url = `${proto}://${location.host}/api/runs/${runId}/events` +
    (token ? `?token=${encodeURIComponent(token)}` : "");
  let closed = false;
  let received = false;
  let timer = 0;
  const poll = async () => {
    if (closed) return;
    try {
      const [run, metrics] = await Promise.all([api.run(runId), api.metrics(runId)]);
      for (const e of metrics.epochs) onEvent({ type: "epoch", ...e } as RunEvent);
      if (run.status !== "running") {
        onEvent({ type: "status", status: run.status, final_metrics: run.final_metrics });
        return;
      }
    } catch {
      /* retry */
    }
    timer = window.setTimeout(poll, 1500);
  };
  const socket = new WebSocket(url);
  socket.onmessage = (msg) => {
    received = true;
    onEvent(JSON.parse(msg.data) as RunEvent);
  };
  socket.onclose = () => {
    if (!received && !closed) poll();
  };
  return () => {
    closed = true;
    window.clearTimeout(timer);
    socket.close();
  };
}
