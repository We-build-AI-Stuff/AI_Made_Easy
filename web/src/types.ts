// Shapes of the AI Made Easy REST API (see ai_made_easy/server/app.py).

export interface ParamSpec {
  name: string;
  type: "int" | "float" | "bool" | "str" | "enum" | "shape" | string;
  default: unknown;
  options?: string[];
  minimum?: number | null;
  maximum?: number | null;
  help?: string;
}

export interface PortSpec {
  name: string;
  dtype?: string;
  multi?: boolean;
  role?: string; // tensor | logits | boxes | masks | tokens | graph | variable | ...
}

export interface BlockDef {
  type_id: string;
  display_name: string;
  category: string;
  color: string;
  composite: boolean;
  description: string;
  frameworks: string[];
  params: ParamSpec[];
  inputs: PortSpec[];
  outputs: PortSpec[];
  requires?: string[];
  extra?: string;
  missing?: string[]; // required packages not installed on the server
}

export interface TaskInfo {
  id: string;
  label: string;
  description: string;
  family: string;
  output_role: string;
  classification: boolean;
  trainer_kind: string;
  losses: string[];
  metrics: string[];
}

export interface FamilyInfo {
  id: string;
  label: string;
  description: string;
  frameworks: string[];
  targets: string[];
  trainable: boolean;
}

export interface ProjectNode {
  id: string;
  type: string;
  params: Record<string, unknown>;
  position: [number, number];
}

export interface ProjectEdge {
  from: string; // "node/port"
  to: string;
}

export interface FeatureImportance {
  run_id: string;
  method?: string;
  metric?: string;
  baseline_loss?: number;
  baseline?: string;
  rows?: number;
  features: { name: string; importance: number; std: number }[];
}

export interface DesignDiff {
  same: boolean;
  summary: string;
  added: { id: string; type: string }[];
  removed: { id: string; type: string }[];
  retyped: { id: string; from: string; to: string }[];
  changed: { id: string; type: string; param: string; from: unknown; to: unknown }[];
  wired: { from: string; to: string }[];
  unwired: { from: string; to: string }[];
}

export interface Project {
  name: string;
  nodes: ProjectNode[];
  edges: ProjectEdge[];
  meta?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface Issue {
  severity: "error" | "warning";
  node: string | null;
  message: string;
  fix: { label: string; description: string } | null;
}

export interface InlineBlock {
  id: string; type: string; name: string; color: string; summary: string; model: boolean;
}

/** The design a pipeline stage trains, drawn inside the stage (composite view). */
export interface StageDesign {
  ref: string; name: string; kind: string; blocks: InlineBlock[]; error: string;
}

export interface Validation {
  valid: boolean;
  issues: Issue[];
  shapes: Record<string, number[]>;
  designs?: Record<string, StageDesign>;
}

export interface Summary {
  total_params: number;
  total_params_display: string;
  total_flops: number;
  layers: { name: string; type: string; output_shape: number[]; params: number; flops: number }[];
}

export interface DeviceProfile {
  id: string;
  label: string;
  kind: string;
  memory_gb: number;
  tflops_fp32: number;
  tflops_fp16: number;
  bandwidth_gbs: number;
}

export interface Budget {
  device: string;
  max_train_memory_gb: number;
  max_latency_ms: number;
  max_params_m: number;
  max_model_mb: number;
}

export interface BudgetReport {
  budget: Budget;
  estimate: {
    params: number; trainable_params: number; flops: number; batch_size: number;
    mixed_precision: boolean; model_mb: number; train_memory_gb: number;
    latency_ms: number | null; step_time_ms: number | null; device: DeviceProfile | null;
  };
  checks: { kind: string; used: number; limit: number; unit: string; over: boolean }[];
}

export interface RunRecord {
  run_id: string;
  name: string;
  project: string;
  framework: string;
  kind: string;
  status: string;
  created_at: number;
  finished_at: number | null;
  final_metrics: Record<string, number>;
  best_metrics: Record<string, number>;
  epochs_done: number;
  tags: string[];
  note: string;
  parent: string;
  trial: Record<string, unknown>;
  error: string;
  data_fingerprint: string;
  backend?: string;   // "" = this computer; else ssh:<host>, kaggle, a saved backend, …
  remote?: Record<string, unknown>;
  params?: Record<string, unknown>;
  graph?: Project;
  duration?: number | null;
  boosted?: boolean;  // a boosted-tree model: Train more adds trees (3.9)
}

export interface BackendInfo {
  name: string;
  kind: string;
  remote: boolean;
  live: boolean;      // false: curves arrive per poll / when the run reports back
  options: Record<string, unknown>;
}

export interface BackendList {
  backends: BackendInfo[];
  kinds: string[];
  options: Record<string, Record<string, string>>;   // kind -> option -> help
}

export interface BackendTest {
  ok: boolean;
  message: string;
  backend: string;
  kind: string;
  accelerator: { gpu?: string; device?: string; memory_gb?: number | null };
}

export interface EpochEvent {
  type: "epoch";
  epoch: number;
  total?: number;
  metrics: Record<string, number>;
}

export interface RunEvent {
  type: "log" | "epoch" | "error" | "done" | "env" | "status" | "resources" | "samples";
  [key: string]: unknown;
}

export interface SweepParam {
  node: string;
  param: string;
  key: string;
  block: string;
  current: unknown;
  kind: "int" | "float" | "choice";
  low?: number;
  high?: number;
  log?: boolean;
  values?: unknown[];
}

export interface SweepDimension {
  node: string;
  param: string;
  kind: string;
  low?: number | null;
  high?: number | null;
  log?: boolean;
  values?: unknown[];
}

export interface SweepTrial {
  number: number;
  values: Record<string, unknown>;
  run_id: string;
  state: string;
  score: number | null;
  message: string;
  objectives?: Record<string, number>;   // multi-objective AutoML: design costs
}

/** Which settings mattered in a sweep (3.6). */
export interface SweepAxis {
  key: string;
  label: string;
  kind: "number" | "choice";
  min?: number;
  max?: number;
  log?: boolean;
  choices?: string[];
}

export interface SweepAnalysis {
  sweep_id: string;
  metric: string;
  direction: "min" | "max";
  trials: number;
  axes: SweepAxis[];
  score_axis: { min: number; max: number } | null;
  rows: { number: number; run_id: string; values: Record<string, unknown>; score: number; best: boolean }[];
  importance: { key: string; label: string; importance: number; explained: number; trend: string }[];
  note: string;
  method?: "forest" | "eta2";
  interactions?: { keys: string[]; labels: string[]; share: number }[];
  heatmap?: { x: (number | string)[]; y: (number | string)[]; x_key: string; y_key: string;
              x_label: string; y_label: string; z: number[][] } | null;
}

export interface SweepAgain {
  sweep_id: string;
  spec: SweepRecord["spec"] & Record<string, unknown>;
  graph: Project;
  changes: { key: string; label: string; before: unknown[]; after: unknown[] }[];
  note: string;
}

export interface SweepRecord {
  sweep_id: string;
  name: string;
  spec: { dimensions: SweepDimension[]; metric: string; direction: string; strategy: string;
          max_trials: number; skip_over_budget?: boolean; automl?: { task: string };
          candidates?: string[]; objectives?: string[]; device?: string };
  project: string;
  state: string;
  created_at: number;
  finished_at: number | null;
  trials: SweepTrial[];
  best: { number: number; run_id: string; score: number; values: Record<string, unknown> } | null;
  message: string;
}

export interface Finding {
  severity: "error" | "warning" | "info";
  message: string;
  hint: string;
  classes: string[];
}

export interface ColumnProfile {
  name: string;
  kind: string;
  dtype: string;
  count: number;
  missing: number;
  unique: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  median: number | null;
  max: number | null;
  top: [string, number][];
  histogram: number[];
  role: string;
}

export interface DataProfile {
  type_id: string;
  kind: string;
  summary: string;
  source: string;
  rows: number;
  truncated: boolean;
  columns: ColumnProfile[];
  classes: { name: string; count: number }[];
  task: string;
  findings: Finding[];
  details: string[];
  error: string;
  ok: boolean;
}

export interface SplitPreview {
  totals: Record<string, number>;
  per_class: Record<string, Record<string, number>>;
  method: string;
  notes: string[];
}

export interface ModelMirror {
  target: "mlflow" | "wandb";
  name: string;
  version: string;
  uri: string;
}

export interface ModelVersion {
  name: string;
  version: number;
  run_id: string;
  framework: string;
  task: string;
  stage: string;
  metrics: Record<string, number>;
  description: string;
  created_at: number;
  mirrors?: ModelMirror[];   // where else it is registered (MLflow, W&B)
}

export interface TrackingStatus {
  selected: string[];
  env: string;
  mlflow_uri: string;
  trackers: { id: string; label: string; installed: boolean }[];
}

export interface FetchedDataset {
  source: string;
  path: string;
  facts: DataFacts;
}

export interface RunSamples {
  run_id: string;
  classes: string[];
  per_class: (number | null)[];
  per_class_metric: string;
  samples: { name: string; data_url: string }[];
  texts?: { name: string; text: string }[];
}

export interface TableLayout {
  error?: string;
  variable?: string;
  rows?: string[];
  columns?: string[];
  parents?: string[];
  values?: number[][];
  fits?: boolean;
  learned?: boolean;
}

export interface WizardTask extends TaskInfo {
  family_label: string;
  modalities: string[];
  demo_modalities: string[];
  data_kinds: string[];
  recipes: number;
  automl: boolean;
  metric: string;
}

export interface DataFacts {
  kind: string;
  source: string;
  rows: number;
  target: string;
  features: string[];
  categorical: string[];
  classes: string[];
  n_classes: number;
  image_size: number;
  channels: number;
  tasks: string[];
  details: string[];
  warnings: string[];
  error: string;
  summary: string;
  [key: string]: unknown;
}

export interface RecipeKnob {
  name: string;
  kind: "float" | "int" | "choice";
  default: number | string;
  low: number | null;
  high: number | null;
  log: boolean;
  values: (number | string)[];
  help: string;
}

export interface RecipeInfo {
  id: string;
  title: string;
  tasks: string[];
  tier: string;
  tier_label: string;
  modality: string;
  description: string;
  knobs: RecipeKnob[];
  extra: string;
  missing: string[];
  downloads: string;
  family: string;
  strengths: string;
}

export interface RecipeSuggestion {
  recipe: RecipeInfo;
  task: string;
  score: number;
  reasons: string[];
  cautions: string[];
  errors: string[];
  over_budget: string[];
  downloads: { node: string; model: string; download_mb: number; license: string;
               params_m: number; label: string }[];
  ready: boolean;
  estimate: { params?: number; flops?: number; train_memory_gb?: number; latency_ms?: number | null };
  graph: Project | null;
}

export interface Explanation {
  recipe: { id: string; title: string; tier: string; task: string; description?: string;
            strengths?: string; knobs?: Record<string, unknown>; adapted?: string[] } | null;
  task: { id: string; label: string; description: string } | null;
  blocks: { id: string; type: string; name: string; category: string; why: string;
            source: "recipe" | "block" }[];
}

export interface AutoMLStarted {
  sweep_id: string;
  candidates: string[];
  metric: string;
  direction: string;
  objectives?: string[];
  device?: string;
}

export interface PipelineStage {
  id: string;
  kind: string;
  label: string;
  state: string;
  run_id: string;
  metrics: Record<string, number>;
  message: string;
  output: Record<string, string>;
  inputs: string[];
  started_at: number | null;
  finished_at: number | null;
}

export interface PipelineRecord {
  pipeline_id: string;
  name: string;
  project: string;
  state: string;
  created_at: number;
  finished_at: number | null;
  order: string[];
  stages: Record<string, PipelineStage>;
  resumed_from: string;
  message: string;
}

/** A saved project in the Open dialog: family and task labels (schema 3 records them). */
export interface SavedProject {
  name: string;
  modified: number;
  family: string | null;
  task: string | null;
}

/** An export format for a trained run: whether the design can use it and whether the
 * project's budget device runs it. */
export interface ExportFormat {
  id: string;
  label: string;
  available?: boolean;
  reason?: string;
  suggested?: boolean;
  device?: string;
}

/** What a finished run takes, to build its Try it form (3.5). */
/** 3.11: what would change a table row's prediction (POST /api/runs/{id}/counterfactuals). */
export interface Counterfactual {
  changes: { column: string; from: unknown; to: unknown }[];
  distance: number;
  prediction: Record<string, unknown>;
}

export interface Counterfactuals {
  run_id: string;
  prediction: Record<string, unknown>;
  goal: string;
  single: Counterfactual[];
  combined: Counterfactual | null;
  locked: string[];
  predictions_checked: number;
  found: boolean;
}

export interface TryForm {
  run_id: string;
  task: string | null;
  modality: string | null;
  input_kind: string;
  input_help: string;
  example: unknown[] | null;
  signature: { input_shape?: number[]; classes?: string[] | null; columns?: string[] | null };
  /** An image classifier that can show its Grad-CAM heatmap (3.6). */
  can_explain?: boolean;
  /** File types "Predict a file" takes, e.g. ".csv" or ".zip"; "" when it can't (3.6). */
  batch_formats?: string;
}

// 3.8 ------------------------------------------------------------------------------------
export interface LRFindResult {
  run_id: string;
  state: string;
  error: string;
  lrs?: number[];
  losses?: number[];
  suggested?: number | null;
  min_loss_lr?: number | null;
  reason?: string;
  stopped?: string;
}

export interface GroupColumn {
  name: string;
  distinct: number;
  numeric: boolean;
  usable: boolean;
  how: "values" | "quartiles";
}

export interface GroupRow {
  value: string;
  count: number;
  gap?: number;
  flagged?: boolean;
  [metric: string]: number | string | boolean | undefined;
}

export interface GroupResult {
  run_id: string;
  state: string;
  error: string;
  column?: string;
  how?: string;
  metric?: string | null;
  higher_is_better?: boolean;
  overall?: Record<string, number>;
  groups?: GroupRow[];
  flagged?: string[];
  fairness?: Fairness | null;
  mitigation?: Mitigation | null;
}

/** 3.11: a threshold per group for a binary classifier, with the groups' rates before
 * and after on rows the thresholds were not fitted on. */
export interface Mitigation {
  fitted_on: string;
  target_tpr: number | null;
  thresholds: Record<string, number>;
  before: MitigationSide;
  after: MitigationSide;
  fairness_before: Fairness | null;
  fairness_after: Fairness | null;
  helps?: boolean;
  summary?: string;
}

export interface MitigationSide {
  overall: { count: number; accuracy: number };
  groups: { value: string; count: number; accuracy: number;
            rates: { selection: (number | null)[]; tpr: (number | null)[]; fpr: (number | null)[] } }[];
}

/** 3.10: gaps in predicted-class and error rates between groups (classifiers). */
export interface Fairness {
  classes: string[];
  parity_gap: number;
  parity_ratio: number;
  odds_gap: number;
  worst: { parity?: string; odds?: string };
  flagged: string[];
  summary: string;
}

/** 3.9: whether a stopped, failed or interrupted run can continue (GET /api/runs/{id}/resume). */
export interface ResumeInfo {
  run_id: string;
  resumable: boolean;
  after_epoch?: number;
  reason?: string;
}
