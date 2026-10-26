// 3.8: two actions on a finished run, Train more and Results by group.
import { useEffect, useState } from "react";

import { api } from "../../api";
import { metric, Modal } from "../../components/Modal";
import { useStore } from "../../state";
import type { GroupColumn, GroupResult, Mitigation, RunRecord } from "../../types";

const FINAL = ["finished", "failed", "stopped", "interrupted"];

/** Continue a finished network for more epochs, as a new run. */
export function TrainMoreModal({ run, onClose, onStarted }: {
  run: RunRecord; onClose: () => void; onStarted: () => void;
}) {
  const { notify } = useStore();
  const trees = Boolean(run.boosted);           // 3.9: a boosted model grows more trees
  const [epochs, setEpochs] = useState(trees ? 100 : 10);
  const [scale, setScale] = useState(1);
  const start = async () => {
    try {
      const r = await api.trainMore(run.run_id, epochs, scale, trees ? epochs : undefined);
      notify(`Run ${r.run_id} continues ${run.run_id}`);
      onStarted();
      onClose();
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  return (
    <Modal title={`Train run ${run.run_id} more`} onClose={onClose}
           actions={<button className="primary" data-testid="start-more" onClick={start}
                            disabled={!(epochs >= 1)}>Train more</button>}>
      {trees
        ? <p className="dim">A new run keeps this model's trees and adds more, fitted on the
            same training rows with the same class balancing. Watch the test score: past some
            point more trees only fit the training data closer.</p>
        : <p className="dim">A new run starts from this run's best weights and trains for more
            epochs on the same data. The optimizer starts afresh (its running averages aren't
            saved with the weights), so a slightly lower learning rate is often smoother.</p>}
      <div className="form">
        <div className="field"><label htmlFor="more-epochs">{trees ? "More trees" : "More epochs"}</label>
          <input id="more-epochs" type="number" min={1} value={epochs}
                 onChange={(e) => setEpochs(parseInt(e.target.value, 10))} /></div>
        {!trees && <div className="field"><label htmlFor="more-scale">Learning-rate scale</label>
          <input id="more-scale" type="number" min={0.01} step={0.1} value={scale}
                 onChange={(e) => setScale(parseFloat(e.target.value))} />
          <div className="help">Multiplies the design's learning rate (1 = the same).</div></div>}
      </div>
    </Modal>
  );
}

/** A run's test metrics for each value of a column of its table. */
export function GroupsModal({ run, onClose }: { run: RunRecord; onClose: () => void }) {
  const [columns, setColumns] = useState<GroupColumn[] | null>(null);
  const [column, setColumn] = useState("");
  const [result, setResult] = useState<GroupResult | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    api.groupColumns(run.run_id).then((r) => {
      const usable = r.columns.filter((c) => c.usable);
      setColumns(usable);
      setColumn(usable.find((c) => !c.numeric)?.name ?? usable[0]?.name ?? "");
    }).catch((e) => setError(String((e as Error).message)));
  }, [run.run_id]);
  useEffect(() => {
    if (!result || FINAL.includes(result.state)) return;
    const timer = window.setTimeout(async () => {
      try { setResult(await api.groupResult(result.run_id)); }
      catch (e) { setError(String((e as Error).message)); }
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [result]);
  const score = async () => {
    setError("");
    try {
      const { run_id } = await api.resultsByGroup(run.run_id, column);
      setResult({ run_id, state: "running", error: "" });
    } catch (e) {
      setError(String((e as Error).message));
    }
  };
  const busy = Boolean(result && !FINAL.includes(result.state));
  const key = result?.metric ?? "";
  return (
    <Modal title={`Results of run ${run.run_id} by group`} onClose={onClose} wide
           actions={<button className="primary" data-testid="score-groups" onClick={score}
                            disabled={!column || busy}>Score by group</button>}>
      <p className="dim">The run's test rows are scored for each value of a column (numbers
        with many values are split into quartiles). Groups clearly worse than the whole test
        set are flagged: a model can be good on average and poor for some of your data.</p>
      {error && <p className="error-text" role="alert">{error}</p>}
      {columns && (
        <div className="field"><label htmlFor="group-column">Group by</label>
          <select id="group-column" value={column} onChange={(e) => setColumn(e.target.value)}>
            {columns.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.how === "quartiles" ? "quartiles" : `${c.distinct} values`})
              </option>))}
          </select></div>)}
      {busy && <p aria-busy="true"><span className="spinner" /> Scoring the test rows…</p>}
      {result?.state && FINAL.includes(result.state) && result.state !== "finished" && (
        <p className="error-text" role="alert">Scoring {result.state}: {result.error}</p>)}
      {result?.groups && (
        <div data-testid="group-results">
          <p>{result.flagged?.length
            ? <><b className="sev-warning">Worse for {result.flagged.join(", ")}</b> by
                at least 10% {result.higher_is_better ? "lower" : "higher"} {key} than the
                whole test set.</>
            : <>No group is clearly worse than the whole test set on {key}.</>}</p>
          <table>
            <thead><tr><th>{result.column}</th><th className="num">rows</th>
              <th className="num">{key}</th><th className="num">vs all</th><th /></tr></thead>
            <tbody>
              <tr className="dim"><td>all test rows</td>
                <td className="num">{result.overall?.count}</td>
                <td className="num">{metric(result.overall?.[key])}</td><td /><td /></tr>
              {result.groups.map((g) => (
                <tr key={g.value} className={g.flagged ? "flagged" : ""}>
                  <td>{g.value}</td><td className="num">{g.count}</td>
                  <td className="num">{metric(g[key])}</td>
                  <td className="num">{typeof g.gap === "number"
                    ? `${g.gap > 0 ? "−" : "+"}${Math.abs(g.gap * 100).toFixed(0)}%` : "—"}</td>
                  <td>{g.flagged && <span className="chip err">worse</span>}</td>
                </tr>))}
            </tbody>
          </table>
          {result.fairness && (
            <p data-testid="fairness"><b className={result.fairness.flagged.length ? "sev-warning" : ""}>
              Fairness:</b> {result.fairness.summary}</p>)}
          {result.mitigation && <Thresholds m={result.mitigation} column={result.column ?? ""} />}
        </div>)}
    </Modal>
  );
}

const pct = (v: number | null | undefined) => (typeof v === "number" ? `${(v * 100).toFixed(0)}%` : "—");

/** A threshold per group: each group's rates at 0.5 and with its own threshold. */
function Thresholds({ m, column }: { m: Mitigation; column: string }) {
  const after = new Map(m.after.groups.map((g) => [g.value, g]));
  return (
    <div data-testid="thresholds">
      <h4>A threshold per group</h4>
      {m.summary && <p className={m.helps ? "" : "dim"}>{m.summary}</p>}
      <table>
        <thead><tr><th>{column}</th><th className="num">threshold</th>
          <th className="num">true positives</th><th className="num">false positives</th>
          <th className="num">accuracy</th></tr></thead>
        <tbody>
          {m.before.groups.map((g) => {
            const a = after.get(g.value);
            return (
              <tr key={g.value}>
                <td>{g.value}</td>
                <td className="num">{(m.thresholds[g.value] ?? 0.5).toFixed(2)}</td>
                <td className="num">{pct(g.rates.tpr[0])} → {pct(a?.rates.tpr[0])}</td>
                <td className="num">{pct(g.rates.fpr[0])} → {pct(a?.rates.fpr[0])}</td>
                <td className="num">{pct(g.accuracy)} → {pct(a?.accuracy)}</td>
              </tr>);
          })}
        </tbody>
      </table>
      <p className="dim">Before → after, on {m.before.overall.count} rows the thresholds were
        not fitted on. They are saved as group_thresholds.json in this scoring run: predict
        the positive class when its probability reaches the row's group threshold.</p>
    </div>
  );
}
