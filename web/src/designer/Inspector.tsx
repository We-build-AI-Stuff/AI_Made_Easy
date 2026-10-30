import { useEffect, useState } from "react";

import { api } from "../api";
import { CodeView } from "../components/CodeView";
import { LRFinder } from "../components/LRFinder";
import { formatValue, shapeLabel } from "../project";
import { useStore } from "../state";
import type { Budget, BudgetReport, DeviceProfile, ParamSpec, Summary, TableLayout } from "../types";

function ParamField({ spec, value, onChange }: {
  spec: ParamSpec; value: unknown; onChange: (v: unknown) => void;
}) {
  const [text, setText] = useState(formatValue(value));
  useEffect(() => setText(value === undefined ? "" : String(value)), [value]);
  const id = `param-${spec.name}`;
  let input;
  if (spec.type === "bool") {
    input = <input id={id} type="checkbox" checked={Boolean(value)}
                   onChange={(e) => onChange(e.target.checked)} />;
  } else if (spec.type === "enum") {
    input = (
      <select id={id} value={String(value)} onChange={(e) => onChange(e.target.value)}>
        {(spec.options ?? []).map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    );
  } else {
    const numeric = spec.type === "int" || spec.type === "float";
    const commit = () => {
      if (!numeric) return onChange(text);
      const n = spec.type === "int" ? parseInt(text, 10) : parseFloat(text);
      if (Number.isNaN(n)) return setText(String(value));
      onChange(n);
    };
    input = (
      <input id={id} value={text} inputMode={numeric ? "decimal" : undefined}
             onChange={(e) => setText(e.target.value)} onBlur={commit}
             onKeyDown={(e) => e.key === "Enter" && commit()} />
    );
  }
  const range = spec.minimum != null || spec.maximum != null
    ? ` (${spec.minimum ?? "−∞"} … ${spec.maximum ?? "∞"})` : "";
  return (
    <div className="field">
      <label htmlFor={id}>{spec.name}<span className="muted">{range}</span></label>
      {input}
      {spec.help && <div className="help">{spec.help}</div>}
    </div>
  );
}

function tableSummary(value: unknown): string {
  const text = String(value ?? "").trim();
  if (!text) return "learned from data";
  try {
    const rows = JSON.parse(text) as unknown[];
    const cols = Array.isArray(rows[0]) ? (rows[0] as unknown[]).length : 1;
    return `${rows.length} × ${cols} table`;
  } catch {
    return "invalid table";
  }
}

function TableField({ spec, value, nodeId, onChange }: {
  spec: ParamSpec; value: unknown; nodeId: string; onChange: (v: unknown) => void;
}) {
  const { project, notify } = useStore();
  const [layout, setLayout] = useState<TableLayout | null>(null);
  const [cells, setCells] = useState<string[][]>([]);
  const open = async () => {
    try {
      const info = await api.tableLayout(project(), nodeId);
      if (info.error) return notify(info.error, true);
      setLayout(info);
      setCells((info.values ?? []).map((r) => r.map((v) => String(+v.toFixed(8)))));
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  const numbers = cells.map((r) => r.map((c) => parseFloat(c) || 0));
  const bad = (layout?.columns ?? []).filter((_c, j) =>
    Math.abs(numbers.reduce((s, r) => s + r[j], 0) - 1) > 1e-4 || numbers.some((r) => r[j] < 0));
  const normalize = () => setCells(numbers.map((r) => r.map((_v, j) => {
    const total = numbers.reduce((s, row) => s + Math.max(row[j], 0), 0);
    return String(+(total ? Math.max(r[j], 0) / total : 1 / numbers.length).toFixed(8));
  })));
  const close = (result?: string) => {
    if (result !== undefined) onChange(result);
    setLayout(null);
  };
  return (
    <div className="field">
      <label>{spec.name}</label>
      <div className="toolbar">
        <span className="muted">{tableSummary(value)}</span>
        <button onClick={open} data-testid="edit-table">Edit table…</button>
      </div>
      {spec.help && <div className="help">{spec.help}</div>}
      {layout && (
        <div className="modal-backdrop">
          <div className="modal" style={{ maxWidth: "90vw", overflow: "auto" }}
               data-testid="table-editor">
            <h3>P({layout.variable}{layout.parents?.length ? ` | ${layout.parents.join(", ")}` : ""})</h3>
            <p className="muted">Each column is one combination of parent states and must sum to 1.</p>
            <table>
              <thead><tr><th />{layout.columns?.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>{layout.rows?.map((r, i) => (
                <tr key={r}><th>{r}</th>{layout.columns?.map((c, j) => (
                  <td key={c}><input value={cells[i]?.[j] ?? ""} inputMode="decimal"
                                     style={{ width: 72 }} aria-label={`${r} | ${c}`}
                                     onChange={(e) => setCells(cells.map((row, a) => row.map(
                                       (v, b) => (a === i && b === j ? e.target.value : v))))} />
                  </td>))}</tr>))}</tbody>
            </table>
            <p className={bad.length ? "sev-warning" : "muted"}>
              {bad.length ? `⚠ ${bad.length} column(s) do not sum to 1 (e.g. ${bad[0]})`
                : "✓ every column sums to 1"}</p>
            <div className="actions">
              <button onClick={normalize}>Normalize columns</button>
              <button onClick={() => close("")}>Learn from data</button>
              <button onClick={() => close()}>Cancel</button>
              <button className="primary" data-testid="save-table"
                      onClick={() => close(JSON.stringify(numbers))}>Save</button>
            </div>
          </div>
        </div>)}
    </div>
  );
}

function Properties() {
  const { selected, nodes, blockMap, setParam, commit, validation, dataIssues, name, setName,
          blocks, notify } = useStore();
  const [finding, setFinding] = useState(false);
  const node = nodes.find((n) => n.id === selected);
  if (!node) {
    return (
      <div className="form">
        <div className="field">
          <label htmlFor="project-name">Project name</label>
          <input id="project-name" value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <p className="muted">
          {nodes.length} blocks on the canvas · {blocks.length} in the library. Select a block to
          edit its parameters; drag blocks from the library or double-click them to add.
        </p>
      </div>
    );
  }
  const def = blockMap.get(node.data.typeId);
  const issues = validation?.issues.filter((i) => i.node === node.id) ?? [];
  const data = dataIssues.filter((i) => i.node_id === node.id);
  return (
    <div className="form">
      <div>
        <strong>{def?.display_name ?? node.data.typeId}</strong>
        <div className="muted">{node.id} · {node.data.typeId}
          {validation?.shapes[node.id] && <> · output {shapeLabel(validation.shapes[node.id])}</>}
        </div>
        {def?.description && <p className="dim">{def.description}</p>}
      </div>
      {[...issues.map((i) => [i.severity, i.message]), ...data.map((i) => [i.severity, i.message])]
        .map(([sev, msg], k) => <div key={k} className={`sev-${sev}`}>{sev === "error" ? "✕" : "⚠"} {msg}</div>)}
      {(def?.params ?? []).map((spec) => spec.type === "table" ? (
        <TableField key={spec.name} spec={spec} nodeId={node.id}
                    value={node.data.params[spec.name] ?? spec.default}
                    onChange={(v) => { commit(); setParam(node.id, spec.name, v); }} />
      ) : (
        <ParamField key={spec.name} spec={spec} value={node.data.params[spec.name] ?? spec.default}
                    onChange={(v) => { commit(); setParam(node.id, spec.name, v); }} />
      ))}
      {!def?.params.length && <p className="muted">This block has no parameters.</p>}
      {def?.category === "Optimizer" && def.params.some((p) => p.name === "lr") && (
        <button data-testid="find-lr" onClick={() => setFinding(true)}
                disabled={Boolean(validation?.issues.some((i) => i.severity === "error"))}
                title="Run a short learning-rate range test and suggest a rate">
          Find learning rate…
        </button>)}
      {finding && (
        <LRFinder onClose={() => setFinding(false)} onUse={(lr) => {
          const rounded = Number(lr.toPrecision(3));
          commit();
          setParam(node.id, "lr", rounded);
          setFinding(false);
          notify(`Learning rate set to ${rounded}`);
        }} />)}
    </div>
  );
}

const LIMITS: { key: keyof Budget; label: string; unit: string }[] = [
  { key: "max_train_memory_gb", label: "Max training memory", unit: "GB" },
  { key: "max_latency_ms", label: "Max latency", unit: "ms" },
  { key: "max_params_m", label: "Max parameters", unit: "M" },
  { key: "max_model_mb", label: "Max model size", unit: "MB" },
];

function flops(n: number): string {
  for (const [unit, size] of [["T", 1e12], ["G", 1e9], ["M", 1e6], ["K", 1e3]] as const) {
    if (n >= size) return `${(n / size).toFixed(2)} ${unit}FLOPs`;
  }
  return n ? `${n} FLOPs` : "—";
}

function BudgetPanel({ report }: { report: BudgetReport | null }) {
  const { meta, setMeta } = useStore();
  const [devices, setDevices] = useState<DeviceProfile[]>([]);
  useEffect(() => { api.devices().then((r) => setDevices(r.devices)).catch(() => undefined); }, []);
  const budget = { device: "", max_train_memory_gb: 0, max_latency_ms: 0, max_params_m: 0,
                   max_model_mb: 0, ...((meta.budget as Partial<Budget>) ?? {}) };
  const set = (patch: Partial<Budget>) => setMeta((m) => ({ ...m, budget: { ...budget, ...patch } }));
  const memory = report?.checks.find((c) => c.kind === "train_memory");
  return (
    <fieldset className="budget" data-testid="budget-panel">
      <legend>Budget</legend>
      <label>Device <select value={budget.device} aria-label="Target device"
                            onChange={(e) => set({ device: e.target.value })}>
        <option value="">No target device</option>
        {devices.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
      </select></label>
      {LIMITS.map((l) => (
        <label key={l.key}>{l.label} <input type="number" min={0} step="any" style={{ width: 80 }}
          value={Number(budget[l.key]) || ""} placeholder="no limit" aria-label={l.label}
          onChange={(e) => set({ [l.key]: Number(e.target.value) || 0 })} /> {l.unit}</label>
      ))}
      {memory && (
        <div className={`budget-bar${memory.over ? " over" : ""}`} data-testid="memory-bar"
             title={`${memory.used.toFixed(2)} of ${memory.limit} GB`}>
          <span style={{ width: `${Math.min(100, (100 * memory.used) / memory.limit)}%` }} />
          <em>{memory.used.toFixed(2)} / {memory.limit} GB training memory</em>
        </div>
      )}
      {report?.checks.filter((c) => c.over && c.kind !== "train_memory").map((c) => (
        <p key={c.kind} className="error-text">{c.kind.replace("_", " ")} {c.used.toFixed(2)} {c.unit} is
          over the {c.limit} {c.unit} budget</p>
      ))}
    </fieldset>
  );
}

function SummaryTab() {
  const { project, validation } = useStore();
  const [summary, setSummary] = useState<Summary | null>(null);
  const [report, setReport] = useState<BudgetReport | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!validation?.valid) {
      setSummary(null);
      setReport(null);
      setError("Fix the errors to see the model summary.");
      return;
    }
    api.summary(project()).then((s) => { setSummary(s); setError(""); })
      .catch((e) => { setSummary(null); setError(String(e.message ?? e)); });
    api.budget(project()).then(setReport).catch(() => setReport(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [validation]);
  if (!summary) return <p className="muted">{error}</p>;
  const est = report?.estimate;
  return (
    <>
      <div className="tiles" data-testid="cost-tiles">
        <div><strong>{summary.total_params_display}</strong><span>parameters</span></div>
        <div><strong>{flops(summary.total_flops)}</strong><span>forward / sample</span></div>
        <div><strong>{est ? `${est.train_memory_gb.toFixed(2)} GB` : "—"}</strong>
          <span>training memory{est ? ` · batch ${est.batch_size}` : ""}</span></div>
        <div><strong>{est?.latency_ms != null ? `${est.latency_ms.toFixed(2)} ms` : "—"}</strong>
          <span>latency{est?.device ? "" : " (pick a device)"}</span></div>
      </div>
      <BudgetPanel report={report} />
      <table>
        <thead><tr><th>Layer</th><th>Output</th><th className="num">Params</th>
          <th className="num">FLOPs</th></tr></thead>
        <tbody>
          {summary.layers.map((l, i) => (
            <tr key={i}><td>{l.name}</td><td>{shapeLabel(l.output_shape)}</td>
              <td className="num">{l.params.toLocaleString()}</td>
              <td className="num">{flops(l.flops)}</td></tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function CodeTab() {
  const { project, validation, name } = useStore();
  const [targets, setTargets] = useState<{ id: string; label: string }[]>([]);
  const [target, setTarget] = useState("pytorch_model");
  const [code, setCode] = useState("");
  useEffect(() => { api.targets().then((r) => setTargets(r.targets)).catch(() => undefined); }, []);
  useEffect(() => {
    api.generate(project(), target).then((r) => setCode(r.code))
      .catch((e) => setCode(`# ${e.message ?? e}`));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [validation, target]);
  const download = () => {
    const blob = new Blob([code], { type: "text/x-python" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${name || "model"}_${target}.py`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  return (
    <>
      <div className="toolbar" style={{ marginBottom: 8 }}>
        <select value={target} onChange={(e) => setTarget(e.target.value)} aria-label="Code target">
          {targets.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
        </select>
        <button onClick={download} disabled={!code}>Download</button>
        <button onClick={() => navigator.clipboard.writeText(code)} disabled={!code}>Copy</button>
      </div>
      <CodeView code={code} />
    </>
  );
}

export function Inspector() {
  const [tab, setTab] = useState("properties");
  return (
    <div className="inspector-body">
      <div className="tabs">
        {["properties", "summary", "code"].map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
            {t[0].toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>
      <div className="tab-body">
        {tab === "properties" && <Properties />}
        {tab === "summary" && <SummaryTab />}
        {tab === "code" && <CodeTab />}
      </div>
    </div>
  );
}
