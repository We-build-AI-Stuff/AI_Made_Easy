// 3.10: which framework a network trains with. The choice is saved on the design
// (meta.framework), so Train, the learning-rate finder, sweeps and pipelines all use it.
import { useEffect, useState } from "react";

import { api } from "../api";
import { useStore } from "../state";

const LABELS: Record<string, string> = { pytorch: "PyTorch", keras: "Keras" };

export function FrameworkSelect() {
  const { project, validation, meta, setMeta } = useStore();
  const [usable, setUsable] = useState<string[]>([]);
  useEffect(() => {
    let current = true;   // an answer for an earlier version of the design is dropped
    api.describe(project()).then((d) => { if (current) setUsable(d.frameworks ?? []); })
      .catch(() => { if (current) setUsable([]); });
    return () => { current = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [validation]);
  const chosen = typeof meta.framework === "string" ? meta.framework : "auto";
  if (usable.length < 2 && chosen === "auto") return null;   // nothing to choose
  return (
    <label className="dim" title="The framework this design trains with (saved with it)">
      Framework{" "}
      <select data-testid="framework" value={chosen} onChange={(e) => {
        const value = e.target.value;
        setMeta((m) => {
          const next = { ...m };
          if (value === "auto") delete next.framework; else next.framework = value;
          return next;
        });
      }}>
        <option value="auto">Auto ({LABELS[usable[0]] ?? "PyTorch"})</option>
        {usable.map((f) => <option key={f} value={f}>{LABELS[f] ?? f}</option>)}
        {chosen !== "auto" && !usable.includes(chosen) && (
          <option value={chosen}>{LABELS[chosen] ?? chosen} (not every block has it)</option>)}
      </select>
    </label>
  );
}
