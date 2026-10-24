// The designer's bottom panels (split out of Designer.tsx in 3.10): Problems, Training
// (train, the live curves and log) and Why (what the selected block does here).
import { useEffect, useState } from "react";

import { api, streamRun } from "../api";
import { RunOnSelect, useRunOn } from "../components/Backends";
import { FrameworkSelect } from "../components/FrameworkSelect";
import { LineChart, metricSeries } from "../components/LineChart";
import { useStore } from "../state";
import type { Explanation, RunEvent } from "../types";

export function Problems() {
  const { validation, dataIssues, project, load, notify, selectNode, nodes } = useStore();
  const issues = validation?.issues ?? [];
  const select = (id: string | null) => id && selectNode(id);
  const fix = async (index: number) => {
    try {
      const result = await api.fix(project(), index);
      load(result.graph);
      notify(`Applied: ${result.label}`);
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  if (!nodes.length) return <div className="empty">The canvas is empty.</div>;
  if (!issues.length && !dataIssues.length) {
    return <div className="sev-ok" style={{ padding: 8 }}>✓ No problems — the design is valid.</div>;
  }
  return (
    <div data-testid="problems">
      {issues.map((issue, i) => (
        <div key={`v${i}`} className="issue" onClick={() => select(issue.node)}>
          <span className={`sev-${issue.severity}`}>{issue.severity === "error" ? "✕" : "⚠"}</span>
          <span style={{ flex: 1 }}>{issue.message}
            {issue.node && <span className="muted"> — {issue.node}</span>}</span>
          {issue.fix && <button onClick={(e) => { e.stopPropagation(); fix(i); }}
                                title={issue.fix.description}>{issue.fix.label}</button>}
        </div>
      ))}
      {dataIssues.map((issue, i) => (
        <div key={`d${i}`} className="issue" onClick={() => select(issue.node_id)}>
          <span className="sev-warning">⚠</span>
          <span style={{ flex: 1 }}>{issue.message} <span className="muted">— data</span></span>
        </div>
      ))}
    </div>
  );
}

export function Training() {
  const { project, name, validation, notify } = useStore();
  const [on] = useRunOn();
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState("idle");
  const [epochs, setEpochs] = useState<{ epoch: number; metrics: Record<string, number> }[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const [sample, setSample] = useState<{ data_url?: string; text?: string; epoch?: number } | null>(null);

  useEffect(() => {
    if (!runId) return;
    setEpochs([]);
    setLog([]);
    setSample(null);
    return streamRun(runId, (event: RunEvent) => {
      if (event.type === "epoch") {
        setEpochs((es) => [...es.filter((e) => e.epoch !== event.epoch),
                           { epoch: Number(event.epoch), metrics: event.metrics as Record<string, number> }]);
      } else if (event.type === "log" || event.type === "error") {
        setLog((l) => [...l.slice(-300), String(event.line ?? event.traceback ?? "")]);
      } else if (event.type === "done") {
        setStatus(event.status ? String(event.status)
          : Number(event.returncode) === 0 ? "finished" : "failed");
      } else if (event.type === "status") {
        setStatus(String(event.status));
      } else if (event.type === "samples") {
        setSample({ data_url: event.data_url as string | undefined,
                    text: event.text as string | undefined,
                    epoch: event.epoch == null ? undefined : Number(event.epoch) });
      }
    });
  }, [runId]);

  const train = async () => {
    try {
      const { run_id } = await api.train(project(), name, "auto", on);
      setRunId(run_id);
      setStatus("running");
      notify(`Training started${on === "local" ? "" : ` on ${on}`} — run ${run_id}`);
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  const stop = async () => {
    if (runId) await api.stopRun(runId).catch(() => undefined);
  };
  const last = epochs[epochs.length - 1];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr)", gap: 12,
                  height: "100%" }}>
      <div>
        <div className="toolbar" style={{ marginBottom: 6 }}>
          <button className="primary" onClick={train}
                  disabled={status === "running" || !validation?.valid}
                  data-testid="train">▶ Train</button>
          <button onClick={stop} disabled={status !== "running"}>Stop</button>
          <FrameworkSelect />
          <RunOnSelect />
          <span className="dim">{runId ? `run ${runId} · ${status}` : "No run yet"}
            {last && ` · epoch ${last.epoch}`}</span>
        </div>
        <LineChart series={metricSeries(epochs, (k) => k.includes("loss"))} height={150} />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6, minHeight: 0 }}>
        {sample && (
          <div data-testid="live-samples" style={{ overflow: "auto", maxHeight: 200 }}>
            <div className="dim">Samples{sample.epoch != null && ` · epoch ${sample.epoch}`}</div>
            {sample.data_url
              ? <img src={sample.data_url} alt="generated samples"
                     style={{ maxWidth: "100%", maxHeight: 170, imageRendering: "pixelated" }} />
              : <pre className="code" style={{ whiteSpace: "pre-wrap" }}>{sample.text}</pre>}
          </div>)}
        <pre className="code" style={{ overflow: "auto", maxHeight: sample ? 90 : 200 }}
             data-testid="train-log">
          {log.join("\n") || "Training output appears here."}
        </pre>
      </div>
    </div>
  );
}

/** Why each block is in the design: the recipe's notes, else the block's description. */
export function Why() {
  const { project, validation, selected, selectNode, nodes } = useStore();
  const [why, setWhy] = useState<Explanation | null>(null);
  useEffect(() => {
    if (!nodes.length) return setWhy(null);
    api.explain(project()).then(setWhy).catch(() => setWhy(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [validation]);
  if (!why) return <div className="muted">Add blocks to see why each one is there.</div>;
  const r = why.recipe;
  return (
    <div data-testid="why">
      {r ? <p style={{ margin: "0 0 6px" }}><b>{r.title}</b> recipe ({r.tier}){r.description && ` — ${r.description}`}
        {r.adapted?.length ? <span className="dim"> Adapted to your data: {r.adapted.join("; ")}.</span> : null}</p>
        : <p className="muted" style={{ margin: "0 0 6px" }}>Not built from a recipe: showing each block's
          description. Use New to start from a recipe.</p>}
      {why.task && <p className="dim" style={{ margin: "0 0 6px" }}>Task: {why.task.label} — {why.task.description}</p>}
      {why.blocks.map((b) => (
        <div key={b.id} className={`why-block ${selected === b.id ? "current" : ""}`}
             onClick={() => selectNode(b.id)}>
          <b>{b.name}</b> <span className="muted">{b.id}</span>
          <div className={b.source === "recipe" ? "" : "muted"}>{b.why}</div>
        </div>))}
    </div>
  );
}
