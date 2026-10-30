// The Layout menu: which panels show, where, and a way back to the default
// arrangement. Reads the same dock state the Designer and the palette use.
import { useEffect, useRef, useState } from "react";

import { PANEL_TITLES, useDock, type PanelId } from "../designer/dock";

export function LayoutMenu() {
  const dock = useDock();
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => { if (!root.current?.contains(e.target as Node)) setOpen(false); };
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("mousedown", away);
    window.addEventListener("keydown", key);
    return () => { window.removeEventListener("mousedown", away);
                   window.removeEventListener("keydown", key); };
  }, [open]);
  const pick = (fn: () => void) => { setOpen(false); fn(); };
  const panelIds: PanelId[] = ["library", "inspector", "console"];
  return (
    <div className="menu" ref={root}>
      <button aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(!open)}
              data-testid="layout-menu" title="Panels: collapse, restore, reset">Layout ▾</button>
      {open && (
        <div className="menu-list" role="menu" aria-label="Layout">
          {panelIds.map((p) => (
            <button key={p} role="menuitem" className="menu-item"
                    data-testid={`layout-${p}`}
                    onClick={() => pick(() => dock.toggle(p))}>
              <span>{dock.layout.collapsed[p] ? `Restore the ${PANEL_TITLES[p]}`
                : `Collapse the ${PANEL_TITLES[p]}`}</span>
              <kbd className="muted">{dock.layout.sides.left.includes(p) ? "left"
                : dock.layout.sides.right.includes(p) ? "right" : "bottom"}</kbd>
            </button>))}
          <div className="menu-sep" role="separator" />
          <button role="menuitem" className="menu-item" data-testid="layout-reset"
                  onClick={() => pick(dock.reset)}>
            Reset layout<span className="muted">Library left · Inspector right</span>
          </button>
        </div>
      )}
    </div>
  );
}
