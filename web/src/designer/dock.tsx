// A dockable workspace (3.12): the Block Library, the Inspector and the bottom
// Console (Problems / Training / Why) can each be dragged to any screen edge or
// corner, resized, collapsed to a rail and restored — the layout is remembered.
//
// The workspace is flex regions around the canvas:
//
//     ┌──────────┬─────────────────────────┬──────────┐
//     │ left     │        canvas           │ right    │
//     │ column   │                         │ column   │
//     │ (stack)  ├─────────────────────────┤ (stack)  │
//     │          │   bottom strip (row)    │          │
//     └──────────┴─────────────────────────┴──────────┘
//
// A panel dragged to "left" or "right" joins that column (stacked, splitters
// between); dragged to "bottom" it joins the bottom strip side by side — its
// first seat is the bottom-left corner, the last the bottom-right. On screens
// under ~880 px the columns become overlay drawers, so nothing ever overlaps
// the canvas.
import {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
  type CSSProperties, type PointerEvent as ReactPointerEvent, type ReactNode,
} from "react";

export type PanelId = "library" | "inspector" | "console";
export type Side = "left" | "right" | "bottom";

export interface DockLayout {
  sides: Record<Side, PanelId[]>;
  sideWidth: Record<"left" | "right", number>;
  bottomHeight: number;
  /** Height (side-docked) or width (bottom-docked) of one panel, in px. */
  panelSize: Record<PanelId, number>;
  collapsed: Record<PanelId, boolean>;
}

export const PANEL_TITLES: Record<PanelId, string> = {
  library: "Block Library",
  inspector: "Inspector",
  console: "Console",
};
const PANELS: PanelId[] = ["library", "inspector", "console"];
const SIDES: Side[] = ["left", "right", "bottom"];
const KEY = "aime.layout.v2";

export const DEFAULT_LAYOUT: DockLayout = {
  sides: { left: ["library"], right: ["inspector"], bottom: ["console"] },
  sideWidth: { left: 260, right: 330 },
  bottomHeight: 250,
  panelSize: { library: 400, inspector: 400, console: 520 },
  collapsed: { library: false, inspector: false, console: false },
};

const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v));

function load(): DockLayout {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? "null") as Partial<DockLayout>;
    if (!raw?.sides) return DEFAULT_LAYOUT;
    const sides: Record<Side, PanelId[]> = { left: [], right: [], bottom: [] };
    for (const side of SIDES) {
      const known: PanelId[] = [];
      for (const p of raw.sides?.[side] ?? []) {
        if (PANELS.includes(p) && !known.includes(p)) known.push(p);
      }
      sides[side] = known;
    }
    for (const p of PANELS) {          // every panel exactly once, on its default edge
      if (!SIDES.some((s) => sides[s].includes(p))) {
        sides[SIDES.find((s) => DEFAULT_LAYOUT.sides[s].includes(p)) ?? "bottom"].push(p);
      }
    }
    return {
      sides,
      sideWidth: {
        left: clamp(Number(raw.sideWidth?.left) || DEFAULT_LAYOUT.sideWidth.left, 180, 560),
        right: clamp(Number(raw.sideWidth?.right) || DEFAULT_LAYOUT.sideWidth.right, 180, 560),
      },
      bottomHeight: clamp(Number(raw.bottomHeight) || DEFAULT_LAYOUT.bottomHeight, 120, 720),
      panelSize: {
        library: clamp(Number(raw.panelSize?.library) || 400, 120, 4000),
        inspector: clamp(Number(raw.panelSize?.inspector) || 400, 120, 4000),
        console: clamp(Number(raw.panelSize?.console) || 520, 180, 4000),
      },
      collapsed: {
        library: Boolean(raw.collapsed?.library),
        inspector: Boolean(raw.collapsed?.inspector),
        console: Boolean(raw.collapsed?.console),
      },
    };
  } catch {
    return DEFAULT_LAYOUT;
  }
}

// ------------------------------------------------------------------ state

interface DockApi {
  layout: DockLayout;
  move: (panel: PanelId, side: Side, index: number) => void;
  resizeSide: (side: "left" | "right", px: number) => void;
  resizeBottom: (px: number) => void;
  resizePanel: (panel: PanelId, px: number) => void;
  toggle: (panel: PanelId) => void;
  reset: () => void;
}

const DockContext = createContext<DockApi | null>(null);

export function useDock(): DockApi {
  const dock = useContext(DockContext);
  if (!dock) throw new Error("useDock outside <DockProvider>");
  return dock;
}

export function useDockState(): DockApi {
  const [layout, setLayout] = useState<DockLayout>(load);
  useEffect(() => { localStorage.setItem(KEY, JSON.stringify(layout)); }, [layout]);

  const move = useCallback((panel: PanelId, side: Side, index: number) => {
    setLayout((l) => {
      const sides = { left: [...l.sides.left], right: [...l.sides.right],
                      bottom: [...l.sides.bottom] };
      for (const s of SIDES) sides[s] = sides[s].filter((p) => p !== panel);
      sides[side].splice(clamp(index, 0, sides[side].length), 0, panel);
      return { ...l, sides, collapsed: { ...l.collapsed, [panel]: false } };
    });
  }, []);
  const resizeSide = useCallback((side: "left" | "right", px: number) => {
    setLayout((l) => ({ ...l, sideWidth: { ...l.sideWidth, [side]: clamp(px, 180, 560) } }));
  }, []);
  const resizeBottom = useCallback((px: number) => {
    setLayout((l) => ({ ...l, bottomHeight: clamp(px, 120, 720) }));
  }, []);
  const resizePanel = useCallback((panel: PanelId, px: number) => {
    setLayout((l) => ({ ...l, panelSize: { ...l.panelSize, [panel]: clamp(px, 120, 4000) } }));
  }, []);
  const toggle = useCallback((panel: PanelId) => {
    setLayout((l) => ({ ...l, collapsed: { ...l.collapsed, [panel]: !l.collapsed[panel] } }));
  }, []);
  const reset = useCallback(() => setLayout(DEFAULT_LAYOUT), []);
  return useMemo(() => ({ layout, move, resizeSide, resizeBottom, resizePanel, toggle, reset }),
                  [layout, move, resizeSide, resizeBottom, resizePanel, toggle, reset]);
}

export function DockProvider({ value, children }: { value: DockApi; children: ReactNode }) {
  return <DockContext.Provider value={value}>{children}</DockContext.Provider>;
}

// ------------------------------------------------------------------ dragging

interface DragState {
  panel: PanelId;
  x: number;
  y: number;
  /** Panel rects at drag start, to place the insertion line between them. */
  rects: { side: Side; panel: PanelId; box: DOMRect }[];
  zone: { side: Side; index: number } | null;
}

/** Which edge the pointer is over, and where in that edge's stack it would land. */
function zoneAt(x: number, y: number, rect: DOMRect, rects: DragState["rects"]):
    DragState["zone"] {
  const relX = (x - rect.left) / rect.width;
  const relY = (y - rect.top) / rect.height;
  let side: Side | null = null;
  if (relX < 0.2) side = "left";
  else if (relX > 0.8) side = "right";
  else if (relY > 0.78) side = "bottom";
  if (!side) return null;
  const members = rects.filter((r) => r.side === side)
    .sort((a, b) => side === "bottom" ? a.box.left - b.box.left : a.box.top - b.box.top);
  let index = members.length;
  for (let i = 0; i < members.length; i++) {
    const box = members[i].box;
    const before = side === "bottom"
      ? x < box.left + box.width / 2 : y < box.top + box.height / 2;
    if (before) { index = i; break; }
  }
  return { side, index };
}

// ------------------------------------------------------------------ host

export function DockHost({ canvas, panels, classNames }: {
  canvas: ReactNode;
  panels: Record<PanelId, ReactNode>;
  /** Extra classes per panel — the tour and tests target ".library" etc. */
  classNames?: Partial<Record<PanelId, string>>;
}) {
  const { layout, move } = useDock();
  const workspace = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [compact, setCompact] = useState(() => window.matchMedia("(max-width: 880px)").matches);
  const [drawers, setDrawers] = useState<Record<Side, boolean>>(
    { left: false, right: false, bottom: false });
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 880px)");
    const change = () => setCompact(mq.matches);
    mq.addEventListener("change", change);
    return () => mq.removeEventListener("change", change);
  }, []);

  const startDrag = (panel: PanelId) => (e: ReactPointerEvent<HTMLElement>) => {
    if (compact || e.button !== 0) return;
    const root = workspace.current;
    if (!root) return;
    const rects = Array.from(root.querySelectorAll<HTMLElement>("[data-dock-panel]"))
      .map((el) => ({ side: (el.dataset.dockSide ?? "left") as Side,
                      panel: el.dataset.dockPanel as PanelId,
                      box: el.getBoundingClientRect() }));
    const track = (ev: PointerEvent) => {
      const rect = root.getBoundingClientRect();
      setDrag((d) => d ? { ...d, x: ev.clientX, y: ev.clientY,
                           zone: zoneAt(ev.clientX, ev.clientY, rect, rects) } : d);
    };
    const up = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", track);
      window.removeEventListener("pointerup", up);
      const zone = zoneAt(ev.clientX, ev.clientY, root.getBoundingClientRect(), rects);
      if (zone) move(panel, zone.side, zone.index);
      setDrag(null);
    };
    window.addEventListener("pointermove", track);
    window.addEventListener("pointerup", up);
    setDrag({ panel, x: e.clientX, y: e.clientY, rects, zone: null });
  };

  const renderPanel = (panel: PanelId, side: Side) => (
    <DockPanel key={panel} panel={panel} side={side} onDragStart={startDrag(panel)}
               className={classNames?.[panel]}>
      {panels[panel]}
    </DockPanel>
  );

  /** A side column: open panels with splitters between, then the collapsed rails. */
  const sideColumn = (side: "left" | "right", drawerOpen: boolean) => {
    const ids = layout.sides[side];
    if (!ids.length) return null;
    const open = ids.filter((p) => !layout.collapsed[p]);
    const rails = ids.filter((p) => layout.collapsed[p]);
    const width = drawerOpen ? undefined : layout.sideWidth[side];
    return (
      <div className={`dock-side dock-${side}${drawerOpen ? " drawer-open" : ""}`}
           style={{ width }} data-dock-side={side}
           aria-label={side === "left" ? "Left panels" : "Right panels"}>
        {open.flatMap((p, i) => i < open.length - 1
          ? [<DockPanel key={p} panel={p} side={side} onDragStart={startDrag(p)}
                        className={classNames?.[p]}>{panels[p]}</DockPanel>,
             <PanelSplitter key={`split-${p}`} after={p} />]
          : [<DockPanel key={p} panel={p} side={side} onDragStart={startDrag(p)}
                        className={classNames?.[p]}>{panels[p]}</DockPanel>])}
        {rails.map((p) => <CollapsedRail key={p} panel={p} side={side} />)}
      </div>
    );
  };

  const bottom = (() => {
    const ids = layout.sides.bottom;
    if (!ids.length) return { strip: null, resizer: null, toggle: null };
    const open = ids.filter((p) => !layout.collapsed[p]);
    const rails = ids.filter((p) => layout.collapsed[p]);
    const strip = (
      <section className={`dock-bottom${compact && drawers.bottom ? " drawer-open" : ""}`}
               data-dock-side="bottom"
               style={{ height: compact ? undefined : layout.bottomHeight }}>
        <div className="dock-bottom-row">
          {open.flatMap((p, i) => i < open.length - 1
            ? [renderPanel(p, "bottom"), <PanelSplitter key={`split-${p}`} after={p} vertical />]
            : [renderPanel(p, "bottom")])}
          {rails.map((p) => <CollapsedRail key={p} panel={p} side="bottom" />)}
        </div>
      </section>
    );
    const resizer = !compact && open.length > 0 ? <BottomResizer /> : null;
    const toggle = compact ? (
      <button className={`drawer-toggle toggle-bottom${drawers.bottom ? " open" : ""}`}
              onClick={() => setDrawers((d) => ({ ...d, bottom: !d.bottom }))}
              aria-expanded={drawers.bottom}>
        {open.length ? PANEL_TITLES[open[0]] : "Panels"}
      </button>
    ) : null;
    return { strip, resizer, toggle };
  })();

  return (
    <div className="workspace dock-workspace" ref={workspace}>
      {!compact && sideColumn("left", false)}
      {!compact && layout.sides.left.some((p) => !layout.collapsed[p]) && (
        <SideResizer side="left" />)}
      <div className="center-col">
        <div className="canvas-frame">
          {canvas}
          {compact && (bottom.toggle || layout.sides.left.length > 0
                       || layout.sides.right.length > 0) && (
            <div className="drawer-toggles">
              {layout.sides.left.length > 0 && (
                <button className={`drawer-toggle toggle-left${drawers.left ? " open" : ""}`}
                        onClick={() => setDrawers((d) => ({ ...d, left: !d.left }))}
                        aria-expanded={drawers.left}>
                  {layout.sides.left.map((p) => PANEL_TITLES[p]).join(" · ")}
                </button>)}
              {layout.sides.right.length > 0 && (
                <button className={`drawer-toggle toggle-right${drawers.right ? " open" : ""}`}
                        onClick={() => setDrawers((d) => ({ ...d, right: !d.right }))}
                        aria-expanded={drawers.right}>
                  {layout.sides.right.map((p) => PANEL_TITLES[p]).join(" · ")}
                </button>)}
              {bottom.toggle}
            </div>
          )}
        </div>
        {!compact && bottom.resizer}
        {!compact && bottom.strip}
        {compact && drawers.bottom && bottom.strip}
      </div>
      {!compact && layout.sides.right.some((p) => !layout.collapsed[p]) && (
        <SideResizer side="right" />)}
      {!compact && sideColumn("right", false)}
      {compact && drawers.left && sideColumn("left", true)}
      {compact && drawers.right && sideColumn("right", true)}
      {compact && (drawers.left || drawers.right || drawers.bottom) && (
        <div className="drawer-backdrop" onClick={() =>
          setDrawers({ left: false, right: false, bottom: false })} />)}
      {drag && <DropOverlay drag={drag} rect={workspace.current?.getBoundingClientRect()} />}
    </div>
  );
}

// ------------------------------------------------------------------ pieces

function DockPanel({ panel, side, onDragStart, className, children }: {
  panel: PanelId; side: Side; onDragStart: (e: ReactPointerEvent<HTMLElement>) => void;
  className?: string; children: ReactNode;
}) {
  const { layout, toggle } = useDock();
  const style: CSSProperties = side === "bottom"
    ? { flex: `0 0 ${layout.panelSize[panel]}px`, minWidth: 0 }
    : { flex: `0 0 ${layout.panelSize[panel]}px`, minHeight: 0 };
  return (
    <aside className={`panel dock-panel ${className ?? ""}`} style={style}
           data-dock-panel={panel} data-dock-side={side}>
      <header className="dock-head" onPointerDown={onDragStart}
              title="Drag to dock this panel on another edge">
        <span className="dock-grip" aria-hidden>⠿</span>
        <span className="dock-title">{PANEL_TITLES[panel]}</span>
        <button className="ghost dock-collapse"
                aria-label={`Collapse ${PANEL_TITLES[panel]}`}
                title="Collapse — click the rail to restore"
                data-testid={`collapse-${panel}`}
                onClick={(e) => { e.stopPropagation(); toggle(panel); }}>▾</button>
      </header>
      <div className="dock-body">{children}</div>
    </aside>
  );
}

function CollapsedRail({ panel, side }: { panel: PanelId; side: Side }) {
  const { toggle } = useDock();
  return (
    <button className={`dock-rail rail-${side}`} onClick={() => toggle(panel)}
            aria-label={`Restore ${PANEL_TITLES[panel]}`} data-testid={`restore-${panel}`}
            title={`${PANEL_TITLES[panel]} — click to restore`}>
      <span className="rail-label">{PANEL_TITLES[panel]}</span>
    </button>
  );
}

/** The bar between the canvas and a side column (drags the column's width). */
function SideResizer({ side }: { side: "left" | "right" }) {
  const { layout, resizeSide } = useDock();
  const start = (e: ReactPointerEvent<HTMLElement>) => {
    const from = e.clientX;
    const before = layout.sideWidth[side];
    const drag = (ev: PointerEvent) =>
      resizeSide(side, before + (side === "left" ? ev.clientX - from : from - ev.clientX));
    listen(drag);
  };
  return <div className={`dock-resizer resizer-${side}`} onPointerDown={start}
              role="separator" aria-orientation="vertical"
              aria-label={`Resize the ${side} panels`} />;
}

/** The bar between the canvas and the bottom strip (drags its height). */
function BottomResizer() {
  const { layout, resizeBottom } = useDock();
  const start = (e: ReactPointerEvent<HTMLElement>) => {
    const from = e.clientY;
    const before = layout.bottomHeight;
    const drag = (ev: PointerEvent) => resizeBottom(before + (from - ev.clientY));
    listen(drag);
  };
  return <div className="dock-resizer resizer-bottom" onPointerDown={start}
              role="separator" aria-orientation="horizontal"
              aria-label="Resize the bottom panels" />;
}

function listen(drag: (ev: PointerEvent) => void) {
  const up = () => {
    window.removeEventListener("pointermove", drag);
    window.removeEventListener("pointerup", up);
    document.body.classList.remove("dock-resizing");
  };
  document.body.classList.add("dock-resizing");
  window.addEventListener("pointermove", drag);
  window.addEventListener("pointerup", up);
}

/** The splitter between two stacked panels: drags the upper / left one's size. */
function PanelSplitter({ after, vertical = false }: { after: PanelId; vertical?: boolean }) {
  const { resizePanel } = useDock();
  const start = (e: ReactPointerEvent<HTMLElement>) => {
    const sibling = e.currentTarget.previousElementSibling as HTMLElement | null;
    const box = sibling?.getBoundingClientRect();
    const before = vertical ? (box?.width ?? 0) : (box?.height ?? 0);
    const from = vertical ? e.clientX : e.clientY;
    const drag = (ev: PointerEvent) =>
      resizePanel(after, before + (vertical ? ev.clientX - from : ev.clientY - from));
    listen(drag);
  };
  return <div className={`dock-splitter${vertical ? " vertical" : ""}`} onPointerDown={start}
              role="separator" aria-orientation={vertical ? "vertical" : "horizontal"} />;
}

/** Half-transparent edges shown while a panel is dragged: where it would dock. */
function DropOverlay({ drag, rect }: { drag: DragState; rect?: DOMRect }) {
  if (!rect) return null;
  const zone = drag.zone;
  const line = zone ? positionLine(zone, drag, rect) : null;
  return (
    <div className="dock-drop-layer" aria-hidden>
      <div className={`drop-zone zone-left${zone?.side === "left" ? " active" : ""}`} />
      <div className={`drop-zone zone-right${zone?.side === "right" ? " active" : ""}`} />
      <div className={`drop-zone zone-bottom${zone?.side === "bottom" ? " active" : ""}`} />
      {line && <div className="drop-line" style={line} />}
      <div className="drop-ghost" style={{ left: drag.x, top: drag.y }}>
        {PANEL_TITLES[drag.panel]}
      </div>
    </div>
  );
}

function positionLine(zone: { side: Side; index: number }, drag: DragState, rect: DOMRect):
    CSSProperties {
  const members = drag.rects.filter((r) => r.side === zone.side)
    .sort((a, b) => zone.side === "bottom" ? a.box.left - b.box.left : a.box.top - b.box.top);
  const style: CSSProperties = {};
  if (zone.side === "bottom") {
    const top = members.length ? members[0].box.top : rect.bottom - 180;
    style.top = Math.max(0, top - rect.top);
    style.height = Math.min(rect.height, (members.length ? members[0].box.bottom : rect.bottom)
      - top);
    const at = zone.index < members.length ? members[zone.index].box.left - 2
      : members.length ? members[members.length - 1].box.right + 2 : rect.left + 10;
    style.left = at - rect.left;
    style.width = 3;
  } else {
    const left = members.length ? members[0].box.left
      : (zone.side === "left" ? rect.left : rect.right - 280);
    const right = members.length ? members[0].box.right
      : (zone.side === "left" ? rect.left + 280 : rect.right);
    style.left = left - rect.left;
    style.width = right - left;
    const at = zone.index < members.length ? members[zone.index].box.top - 2
      : members.length ? members[members.length - 1].box.bottom + 2 : rect.top + 10;
    style.top = at - rect.top;
    style.height = 3;
  }
  return style;
}
