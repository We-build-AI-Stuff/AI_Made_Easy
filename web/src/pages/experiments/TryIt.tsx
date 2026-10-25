// Try a finished run (3.5): a form built from what the model takes, answered by the run's
// own predictor on the server's machine. Nothing is sent anywhere else.
import { useEffect, useState } from "react";

import { api } from "../../api";
import { Modal } from "../../components/Modal";
import type { Counterfactual, Counterfactuals, RunRecord, TryForm } from "../../types";

type Prediction = Record<string, unknown>;

const FILE_KINDS: Record<string, { accept: string; what: string }> = {
  images: { accept: "image/*", what: "an image" },
  audio: { accept: ".wav,audio/wav", what: "a WAV clip" },
  speech: { accept: ".wav,audio/wav", what: "a WAV clip" },
};
const TEXT_KINDS = new Set(["texts", "prompt"]);

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

/** A typed value from a field: numbers stay numbers, everything else is text. */
function value(raw: string): string | number {
  const n = Number(raw);
  return raw.trim() !== "" && Number.isFinite(n) ? n : raw;
}

export function TryResult({ prediction }: { prediction: Prediction }) {
  const probs = prediction.probabilities as Record<string, number> | undefined;
  const cam = typeof prediction.gradcam === "string" ? prediction.gradcam : null;
  if (probs && typeof probs === "object") {
    const top = Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 5);
    return (
      <div className="try-result" data-testid="try-result">
        {cam && <img className="try-cam" data-testid="try-gradcam" alt="Grad-CAM: where the model looked"
                     title="Grad-CAM: red is where the model looked for this class"
                     src={`data:image/png;base64,${cam}`} />}
        <div><strong>{String(prediction.label)}</strong>
          {typeof prediction.confidence === "number"
            && <span className="muted"> · {(prediction.confidence * 100).toFixed(1)}% sure</span>}</div>
        {top.map(([name, p]) => (
          <div key={name} className="try-bar">
            <span>{name}</span>
            <span className="bar"><span style={{ width: `${Math.max(1, p * 100)}%` }} /></span>
            <span className="num">{(p * 100).toFixed(1)}%</span>
          </div>))}
      </div>);
  }
  const images = Object.entries(prediction).filter(([, v]) => typeof v === "string"
    && (v as string).length > 200 && /^[A-Za-z0-9+/=]+$/.test((v as string).slice(0, 100)));
  return (
    <div className="try-result" data-testid="try-result">
      {images.map(([k, v]) => <img key={k} alt={k} src={`data:image/png;base64,${v}`} />)}
      <pre className="code">{JSON.stringify(Object.fromEntries(Object.entries(prediction)
        .filter(([k]) => !images.some(([i]) => i === k))), null, 1)}</pre>
    </div>);
}

export function TryModal({ run, onClose }: { run: RunRecord; onClose: () => void }) {
  const [form, setForm] = useState<TryForm | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [fields, setFields] = useState<Record<string, string>>({});
  const [text, setText] = useState("");
  const [json, setJson] = useState("");
  const [file, setFile] = useState<{ name: string; url: string } | null>(null);
  const [result, setResult] = useState<{ predictions: Prediction[]; latency_ms: number } | null>(null);
  const [explain, setExplain] = useState(false);
  const [batch, setBatch] = useState<{ name: string; url: string; rows: number } | null>(null);

  useEffect(() => {
    api.tryForm(run.run_id).then((f) => {
      setForm(f);
      setJson(JSON.stringify(f.example ?? [], null, 1));
    }).catch((e) => setError(String((e as Error).message)));
  }, [run.run_id]);

  const kind = form?.input_kind ?? "";
  const columns = kind === "records" ? form?.signature.columns ?? [] : [];
  const fileKind = FILE_KINDS[kind];

  const inputs = (): unknown[] => {
    if (columns.length) return [Object.fromEntries(columns.map((c) => [c, value(fields[c] ?? "")]))];
    if (TEXT_KINDS.has(kind)) return [text];
    if (fileKind) return file ? [file.url] : [];
    const parsed = JSON.parse(json || "[]");
    return Array.isArray(parsed) ? parsed : [parsed];
  };

  const predict = async () => {
    setBusy(true);
    setError("");
    try {
      setResult(await api.tryRun(run.run_id, inputs(), explain && !!form?.can_explain));
    } catch (e) {
      setError(String((e as Error).message));
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  const predictFile = async (file: File) => {
    setBusy(true);
    setError("");
    try {
      const csv = await api.tryFile(run.run_id, file);
      if (batch) URL.revokeObjectURL(batch.url);
      const stem = file.name.replace(/\.[^.]+$/, "");
      setBatch({ name: `${stem}_predictions.csv`, rows: Math.max(0, csv.trim().split("\n").length - 1),
                 url: URL.createObjectURL(new Blob([csv], { type: "text/csv" })) });
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal title={`Try run ${run.run_id}`} onClose={onClose} wide actions={
      <button className="primary" onClick={predict} disabled={!form || busy}
              data-testid="try-predict">{busy ? "Predicting…" : "Predict"}</button>}>
      <p className="dim">The trained model answers here, on the machine running AI Made Easy; nothing
        is sent anywhere else.</p>
      {!form && !error && <p className="muted">Loading the model…</p>}
      {form && (
        <div className="form" data-testid="try-form" data-kind={kind}>
          {columns.length > 0 && columns.map((c) => (
            <div key={c} className="field"><label htmlFor={`try-${c}`}>{c}</label>
              <input id={`try-${c}`} value={fields[c] ?? ""}
                     onChange={(e) => setFields((f) => ({ ...f, [c]: e.target.value }))} /></div>))}
          {TEXT_KINDS.has(kind) && (
            <div className="field"><label htmlFor="try-text">Text</label>
              <textarea id="try-text" rows={4} value={text} onChange={(e) => setText(e.target.value)}
                        placeholder={kind === "prompt" ? "Start a sentence…" : "Write a text to classify…"} /></div>)}
          {fileKind && (
            <div className="field"><label htmlFor="try-file">Choose {fileKind.what}</label>
              <input id="try-file" type="file" accept={fileKind.accept} data-testid="try-file"
                     onChange={async (e) => {
                       const f = e.target.files?.[0];
                       setFile(f ? { name: f.name, url: await readAsDataUrl(f) } : null);
                     }} />
              {file && kind === "images" && <img className="try-preview" src={file.url} alt={file.name} />}</div>)}
          {!columns.length && !TEXT_KINDS.has(kind) && !fileKind && (
            <div className="field"><label htmlFor="try-json">Inputs (JSON list)</label>
              <textarea id="try-json" rows={6} className="code" value={json}
                        onChange={(e) => setJson(e.target.value)} data-testid="try-json" />
              <div className="help">{form.input_help}</div></div>)}
          {form.can_explain && (
            <label className="check">
              <input type="checkbox" checked={explain} data-testid="try-explain"
                     onChange={(e) => setExplain(e.target.checked)} />
              Show where the model looked (Grad-CAM)</label>)}
          {form.batch_formats && (
            <div className="field"><label htmlFor="try-batch">Or predict a whole file ({form.batch_formats})</label>
              <input id="try-batch" type="file" accept={form.batch_formats} data-testid="try-batch"
                     disabled={busy}
                     onChange={(e) => { const f = e.target.files?.[0]; if (f) void predictFile(f); }} />
              <div className="help">One prediction per {kind === "images" || kind === "audio" || kind === "speech"
                ? "file in the zip" : "row"}; the predictions come back as a CSV.</div>
              {batch && (
                <p data-testid="try-batch-result">{batch.rows} predictions ·{" "}
                  <a href={batch.url} download={batch.name}>download {batch.name}</a></p>)}
            </div>)}
        </div>)}
      {error && <p className="error" role="alert">{error}</p>}
      {result && (
        <div data-testid="try-results">
          {result.predictions.map((p, i) => <TryResult key={i} prediction={p} />)}
          <p className="muted">{result.latency_ms} ms</p>
        </div>)}
      {result && columns.length > 0 && (
        <WhatWouldChange runId={run.run_id} row={inputs()[0] as Record<string, unknown>}
                         columns={columns} prediction={result.predictions[0]} />)}
    </Modal>
  );
}

const shown = (v: unknown) => (typeof v === "number" ? String(Math.round(v * 1e4) / 1e4) : String(v));

function Change({ c }: { c: Counterfactual }) {
  const p = c.prediction;
  const probs = p.probabilities as Record<string, number> | undefined;
  const label = p.label !== undefined ? String(p.label) : shown(p.prediction);
  return (
    <li>
      {c.changes.map((ch, i) => (
        <span key={ch.column}>{i > 0 && " and "}<b>{ch.column}</b> {shown(ch.from)} → {shown(ch.to)}</span>))}
      <span className="muted"> · predicts {label}
        {probs && p.label !== undefined && ` (${(probs[String(p.label)] * 100).toFixed(0)}%)`}</span>
    </li>);
}

/** 3.11: the smallest edits to this row that change the model's answer. */
function WhatWouldChange({ runId, row, columns, prediction }: {
  runId: string; row: Record<string, unknown>; columns: string[]; prediction: Prediction }) {
  const probs = prediction.probabilities as Record<string, number> | undefined;
  const classes = probs ? Object.keys(probs).filter((c) => c !== String(prediction.label)) : [];
  const [desired, setDesired] = useState("");
  const [target, setTarget] = useState("");
  const [locked, setLocked] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [found, setFound] = useState<Counterfactuals | null>(null);
  if (!probs && typeof prediction.prediction !== "number") return null;
  const search = async () => {
    setBusy(true);
    setError("");
    try {
      setFound(await api.whatWouldChange(runId, {
        row, locked, desired: probs ? desired || undefined : undefined,
        target: probs ? undefined : Number(target) }));
    } catch (e) {
      setError(String((e as Error).message));
      setFound(null);
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="what-would-change" data-testid="what-would-change">
      <h4>What would change it?</h4>
      <p className="dim">The model is asked about edited copies of this row, with values your
        training table has. It shows what the model does, not why.</p>
      <div className="form">
        {probs ? (
          <div className="field"><label htmlFor="cf-desired">Make it predict</label>
            <select id="cf-desired" value={desired} onChange={(e) => setDesired(e.target.value)}>
              <option value="">{classes.length ? `${[...classes].sort((a, b) => probs[b] - probs[a])[0]} (the runner-up)` : "—"}</option>
              {classes.map((c) => <option key={c} value={c}>{c}</option>)}
            </select></div>
        ) : (
          <div className="field"><label htmlFor="cf-target">Reach the value</label>
            <input id="cf-target" type="number" value={target} onChange={(e) => setTarget(e.target.value)} /></div>
        )}
        <fieldset className="field"><legend>Keep as they are</legend>
          {columns.map((c) => (
            <label key={c} className="check">
              <input type="checkbox" checked={locked.includes(c)} data-testid={`cf-lock-${c}`}
                     onChange={(e) => setLocked((l) => e.target.checked ? [...l, c] : l.filter((x) => x !== c))} />
              {c}</label>))}
        </fieldset>
        <button onClick={search} disabled={busy || (!probs && target.trim() === "")}
                data-testid="cf-search">{busy ? "Searching…" : "Find changes"}</button>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {found && (
        <div data-testid="cf-results">
          <p>Goal: {found.goal}.</p>
          {found.single.length > 0 && (
            <><p>One change is enough, smallest first:</p>
              <ul>{found.single.map((c) => <Change key={c.changes[0].column} c={c} />)}</ul></>)}
          {found.combined && (
            <><p>No single change is enough; together these are:</p>
              <ul><Change c={found.combined} /></ul></>)}
          {!found.found && <p>No change of up to three columns reaches the goal with values from
            the training table.</p>}
          <p className="muted">{found.predictions_checked} edited rows checked.</p>
        </div>)}
    </section>);
}
