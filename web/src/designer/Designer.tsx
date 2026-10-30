import {
  Background, type Connection, Controls, type Edge, MiniMap, type Node, type NodeChange, Panel,
  ReactFlow, ReactFlowProvider, useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { type DragEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { groupBounds, groupsOf, hiddenBy, notesOf } from "../canvas";
import { PipelinePanel } from "../components/Pipeline";
import { drawDesign, type PngBox } from "../exportPng";
import { type FlowNode, edgeId, paramSummary, shapeLabel } from "../project";
import { useStore } from "../state";
import { GroupNode, NoteNode } from "./Annotations";
import { BlockNode } from "./BlockNode";
import { Problems, Training, Why } from "./BottomPanels";
import { DockHost } from "./dock";
import { Inspector } from "./Inspector";
import { DRAG_TYPE, Library } from "./Library";

const nodeTypes = { block: BlockNode, note: NoteNode, aimeGroup: GroupNode };
const COLLAPSED: [number, number] = [230, 58];   // a folded group's card
const DATA_SUFFIXES = [".csv", ".tsv", ".txt", ".parquet", ".pq", ".xlsx", ".xls", ".json",
                       ".jsonl", ".zip"];

const typing = () => !!(document.activeElement as HTMLElement | null)
  ?.closest?.("input, textarea, select, [contenteditable=true]");

function Canvas({ onDataFile }: { onDataFile?: (file: File) => void }) {
  const { nodes, edges, onNodesChange, onEdgesChange, setEdges, addBlock, setSelected, commit,
          validation, undo, redo, loads, arrange, blockMap, meta, copySelection, pasteText,
          duplicateSelection, groupSelection, ungroup, addNoteAt, updateNote, removeNote,
          notify, canvas, selectIds, collapseGroup } = useStore();
  const flow = useReactFlow();

  // notes and group frames are drawn as extra nodes around the blocks
  const [marked, setMarked] = useState<Set<string>>(new Set());
  const notes = notesOf(meta);
  const groups = groupsOf(meta);
  const display = useMemo(() => {
    const positions = new Map(nodes.map((n) => [n.id, n.position]));
    const frames: Node[] = [];
    for (const g of groups) {
      const b = groupBounds(g, positions);
      if (!b) continue;
      if (g.collapsed) {   // one card in place of its blocks; their wires end on it
        frames.push({ id: `group:${g.id}`, type: "aimeGroup", position: { x: b.x, y: b.y + 22 },
                      data: { group: g }, style: { width: COLLAPSED[0], height: COLLAPSED[1] },
                      width: COLLAPSED[0], height: COLLAPSED[1], draggable: true,
                      selectable: false });
        continue;
      }
      frames.push({ id: `group:${g.id}`, type: "aimeGroup", position: { x: b.x, y: b.y },
                    data: { group: g }, style: { width: b.width, height: b.height },
                    // a known size: these nodes are rebuilt each render, so React Flow's own
                    // measurement is lost and it would hide them (visibility: hidden)
                    width: b.width, height: b.height,
                    draggable: false, selectable: false, zIndex: -1 });
    }
    const sticky: Node[] = notes.map((n) => ({
      id: `note:${n.id}`, type: "note", position: { x: n.position[0], y: n.position[1] },
      data: { note: n }, style: { width: n.size[0], height: n.size[1] },
      width: n.size[0], height: n.size[1],
      selected: marked.has(`note:${n.id}`), dragHandle: ".note-head" }));
    const hidden = hiddenBy(meta);
    const blocks = hidden.size
      ? nodes.map((n) => (hidden.has(n.id) ? { ...n, hidden: true, selected: false } : n)) : nodes;
    return [...frames, ...blocks, ...sticky] as Node[];
  }, [nodes, groups, notes, marked, meta]);

  // a folded card moves its hidden blocks: the card is drawn where they are, so it follows
  const dragFolded = useCallback((gid: string, c: NodeChange & { type: "position" }) => {
    const group = groups.find((g) => g.id === gid);
    const positions = new Map(nodes.map((n) => [n.id, n.position]));
    const b = group && groupBounds(group, positions);
    if (!group || !b || !c.position) return [];
    const dx = c.position.x - b.x, dy = c.position.y - (b.y + 22);
    if (!dx && !dy) return [];
    return group.nodes.filter((n) => positions.has(n)).map((n) => {
      const p = positions.get(n)!;
      return { type: "position", id: n, position: { x: p.x + dx, y: p.y + dy },
               dragging: c.dragging } as NodeChange<FlowNode>;
    });
  }, [groups, nodes]);

  const handleNodes = useCallback((changes: NodeChange[]) => {
    const blocks: NodeChange<FlowNode>[] = [];
    for (const c of changes) {
      const id = "id" in c ? c.id : "";
      if (id.startsWith("note:")) {
        const nid = id.slice(5);
        if (c.type === "position" && c.position) {
          updateNote(nid, { position: [Math.round(c.position.x), Math.round(c.position.y)] });
        } else if (c.type === "remove") {
          removeNote(nid);
        } else if (c.type === "select") {
          setMarked((m) => {
            const next = new Set(m);
            if (c.selected) next.add(id); else next.delete(id);
            return next;
          });
        }
      } else if (id.startsWith("group:")) {
        if (c.type === "remove") ungroup(id.slice(6));
        else if (c.type === "position" && c.position) blocks.push(...dragFolded(id.slice(6), c));
      } else {
        blocks.push(c as NodeChange<FlowNode>);
      }
    }
    if (blocks.length) onNodesChange(blocks);
  }, [onNodesChange, updateNote, removeNote, ungroup, dragFolded]);

  // fit the view whenever a project is opened or re-arranged
  useEffect(() => {
    if (!loads) return;
    const timer = window.setTimeout(() => flow.fitView({ padding: 0.15, maxZoom: 1.2 }), 60);
    return () => window.clearTimeout(timer);
  }, [loads, flow]);
  const wrapper = useRef<HTMLDivElement>(null);

  const onConnect = useCallback((c: Connection) => {
    if (!c.source || !c.target) return;
    commit();
    const sourceHandle = c.sourceHandle ?? "out";
    const targetHandle = c.targetHandle ?? "in";
    // one wire per input port (a new connection replaces the old one), except ports that
    // accept many wires (e.g. a random variable's parents)
    const type = nodes.find((n) => n.id === c.target)?.data.typeId;
    const multi = !!blockMap.get(type ?? "")?.inputs.find((p) => p.name === targetHandle)?.multi;
    const id = edgeId(c.source, sourceHandle, c.target, targetHandle);
    setEdges((es) => [
      ...es.filter((e) => e.id !== id && (multi || !(e.target === c.target
                                                    && (e.targetHandle ?? "in") === targetHandle))),
      { id, source: c.source, sourceHandle, target: c.target, targetHandle },
    ]);
  }, [commit, setEdges, nodes, blockMap]);

  const onDrop = useCallback((e: DragEvent) => {
    const typeId = e.dataTransfer.getData(DRAG_TYPE);
    const file = e.dataTransfer.files?.[0];
    if (!typeId && file) {  // a data file: start a project from it
      e.preventDefault();
      const name = file.name.toLowerCase();
      if (!DATA_SUFFIXES.some((s) => name.endsWith(s))) {
        notify(`${file.name}: drop a table (.csv, .parquet, .xlsx, .json), a text file or a `
               + ".zip of folders", true);
      } else {
        onDataFile?.(file);
      }
      return;
    }
    if (!typeId) return;
    e.preventDefault();
    const id = addBlock(typeId, flow.screenToFlowPosition({ x: e.clientX, y: e.clientY }));
    if (id) setSelected(id);
  }, [addBlock, flow, setSelected]);

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod || typing()) return;
      const k = e.key.toLowerCase();
      if (k === "z") {
        e.preventDefault();
        if (e.shiftKey) redo(); else undo();
      } else if (k === "y") {
        e.preventDefault();
        redo();
      } else if (k === "d") {
        e.preventDefault();
        const n = duplicateSelection();
        if (n) notify(`Duplicated ${n} block${n === 1 ? "" : "s"}`);
      } else if (k === "g" && !e.shiftKey) {
        e.preventDefault();
        if (groupSelection()) notify("Grouped: ✎ renames the group");
      } else if (k === "u" && e.shiftKey) {
        e.preventDefault();
        const ids = new Set(nodes.filter((n) => n.selected).map((n) => n.id));
        groups.filter((g) => g.nodes.some((n) => ids.has(n))).forEach((g) => ungroup(g.id));
      }
    };
    // the browser's own copy / paste events: no clipboard permission needed
    const copy = (e: ClipboardEvent) => {
      if (typing()) return;
      const text = copySelection();
      if (!text || !e.clipboardData) return;
      e.clipboardData.setData("text/plain", text);
      e.preventDefault();
      const n = nodes.filter((x) => x.selected).length;
      notify(`Copied ${n} block${n === 1 ? "" : "s"}`);
    };
    const paste = (e: ClipboardEvent) => {
      if (typing()) return;
      const text = e.clipboardData?.getData("text/plain");
      if (!text) return;
      const n = pasteText(text);
      if (n) {
        e.preventDefault();
        notify(`Pasted ${n} block${n === 1 ? "" : "s"}`);
      }
    };
    window.addEventListener("keydown", key);
    window.addEventListener("copy", copy);
    window.addEventListener("paste", paste);
    return () => {
      window.removeEventListener("keydown", key);
      window.removeEventListener("copy", copy);
      window.removeEventListener("paste", paste);
    };
  }, [undo, redo, duplicateSelection, groupSelection, ungroup, copySelection, pasteText, notify,
      nodes, groups]);

  const addNote = useCallback(() => {
    const el = wrapper.current?.getBoundingClientRect();
    const at = flow.screenToFlowPosition(el ? { x: el.left + 60, y: el.top + 60 } : { x: 100, y: 100 });
    addNoteAt([Math.round(at.x), Math.round(at.y)]);
  }, [flow, addNoteAt]);

  // what the command palette, menus and shortcuts can do with this canvas
  const latest = useRef({ nodes, edges, groups, notes, validation, blockMap });
  latest.current = { nodes, edges, groups, notes, validation, blockMap };
  useEffect(() => {
    canvas.current = {
      zoomIn: () => flow.zoomIn({ duration: 150 }),
      zoomOut: () => flow.zoomOut({ duration: 150 }),
      fit: () => flow.fitView({ padding: 0.15, maxZoom: 1.2, duration: 200 }),
      focus: (id) => {
        const n = latest.current.nodes.find((x) => x.id === id);
        if (!n) return;
        selectIds([id]);
        const w = n.measured?.width ?? 180;
        const h = n.measured?.height ?? 60;
        flow.setCenter(n.position.x + w / 2, n.position.y + h / 2,
                       { zoom: Math.max(flow.getZoom(), 1), duration: 250 });
      },
      selectAll: () => selectIds(latest.current.nodes.map((n) => n.id)),
      deleteSelection: () => {
        const ids = new Set(latest.current.nodes.filter((n) => n.selected).map((n) => n.id));
        if (!ids.size) return;
        flow.deleteElements({ nodes: [...ids].map((id) => ({ id })) });
      },
      ungroupSelection: () => {
        const ids = new Set(latest.current.nodes.filter((n) => n.selected).map((n) => n.id));
        latest.current.groups.filter((g) => g.nodes.some((n) => ids.has(n))).forEach((g) => ungroup(g.id));
      },
      toggleGroups: () => {
        const ids = new Set(latest.current.nodes.filter((n) => n.selected).map((n) => n.id));
        const all = latest.current.groups;
        const picked = ids.size ? all.filter((g) => g.nodes.some((n) => ids.has(n))) : all;
        const fold = picked.some((g) => !g.collapsed);   // fold them all, else open them all
        picked.forEach((g) => collapseGroup(g.id, fold));
        return picked.length;
      },
      addNote,
      image: () => {
        const { nodes: ns, edges: es, groups: gs, notes: nts, validation: v, blockMap: bm } = latest.current;
        const boxes: PngBox[] = ns.map((n) => {
          const def = bm.get(n.data.typeId);
          const shape = v?.shapes[n.id];
          return { id: n.id, kind: "block", x: n.position.x, y: n.position.y,
                   w: n.measured?.width ?? 180, h: n.measured?.height ?? 50,
                   title: def?.display_name ?? n.data.typeId, color: def?.color ?? "#9aa0a9",
                   subtitle: `${shape ? `${shapeLabel(shape)} ` : ""}${paramSummary(def, n.data.params)}` };
        });
        const positions = new Map(ns.map((n) => [n.id, n.position]));
        for (const g of gs) {
          const b = groupBounds(g, positions);
          if (b) boxes.push({ id: `group:${g.id}`, kind: "group", x: b.x, y: b.y, w: b.width, h: b.height,
                              title: g.title, color: g.color });
        }
        for (const n of nts) {
          boxes.push({ id: `note:${n.id}`, kind: "note", x: n.position[0], y: n.position[1],
                       w: n.size[0], h: n.size[1], title: n.text, color: "" });
        }
        return drawDesign(boxes, es.map((e) => ({ from: e.source, to: e.target,
          label: v?.shapes[e.source] ? shapeLabel(v.shapes[e.source]) : undefined })));
      },
    };
    return () => { canvas.current = null; };
  }, [canvas, flow, selectIds, ungroup, addNote, collapseGroup]);
  const selectedCount = nodes.filter((n) => n.selected).length;

  // wires show the tensor shape flowing through them
  const labelled: Edge[] = useMemo(() => {
    const hidden = hiddenBy(meta);
    const out: Edge[] = [];
    const seen = new Set<string>();
    for (const e of edges) {
      const shape = validation?.shapes[e.source];
      const from = hidden.get(e.source);
      const to = hidden.get(e.target);
      if (from && from === to) continue;            // inside a collapsed group
      if (!from && !to) {
        out.push(shape ? { ...e, label: shapeLabel(shape) } : e);
        continue;
      }
      const source = from ? `group:${from.id}` : e.source;
      const target = to ? `group:${to.id}` : e.target;
      const key = `${source}>${target}`;
      if (seen.has(key)) continue;                  // one wire per pair of ends
      seen.add(key);
      // a dashed stand-in, as on the desktop: the real wire ends on a block inside the card
      out.push({ ...e, id: `${e.id}~${key}`, source, target, selectable: false, deletable: false,
                 className: "stand-in",
                 sourceHandle: from ? null : e.sourceHandle, targetHandle: to ? null : e.targetHandle,
                 label: shape ? shapeLabel(shape) : undefined });
    }
    return out;
  }, [edges, validation, meta]);

  return (
    <div className="canvas" ref={wrapper} onDragOver={(e) => { e.preventDefault(); }}
         onDrop={onDrop} data-testid="canvas">
      <ReactFlow
        nodes={display} edges={labelled} nodeTypes={nodeTypes}
        onNodesChange={handleNodes} onEdgesChange={onEdgesChange} onConnect={onConnect}
        onNodeDragStart={() => commit()}
        onSelectionChange={({ nodes: sel }) => {
          const blocks = sel.filter((n) => n.type === "block");
          setSelected(blocks.length === 1 ? blocks[0].id : null);
        }}
        deleteKeyCode={["Backspace", "Delete"]} multiSelectionKeyCode={["Meta", "Control"]}
        fitView minZoom={0.1} maxZoom={2.5}
        proOptions={{ hideAttribution: true }}>
        <Background gap={20} size={1} color="var(--border)" />
        <Controls showInteractive={false} />
        <Panel position="top-right" className="toolbar">
          <button onClick={addNote} title="Add a sticky note" data-testid="add-note">Note</button>
          <button onClick={() => groupSelection()} disabled={!selectedCount}
                  title="Frame the selected blocks (Ctrl/⌘ G)" data-testid="group">Group</button>
          <button onClick={arrange} title="Lay blocks out left to right by data flow">Arrange</button>
        </Panel>
        <MiniMap pannable zoomable nodeColor={() => "var(--accent)"} maskColor="rgba(127,127,127,.18)" />
      </ReactFlow>
    </div>
  );
}

export function Designer({ onDataFile }: { onDataFile?: (file: File) => void } = {}) {
  const { addBlock, setSelected, nodes } = useStore();
  const isPipeline = nodes.some((n) => String(n.data.typeId).startsWith("pipeline."));
  const [bottom, setBottom] = useState("problems");
  const flowRef = useRef<{ add: (t: string) => void }>({ add: () => undefined });
  return (
    <ReactFlowProvider>
      <DockHost
        canvas={<Canvas onDataFile={onDataFile} />}
        classNames={{ library: "library", inspector: "inspector", console: "bottom-panel" }}
        panels={{
          library: <Library onAdd={(t) => flowRef.current.add(t)} />,
          inspector: <Inspector />,
          console: (
            <>
              <div className="tabs">
                {["problems", "training", "why"].map((t) => (
                  <button key={t} className={bottom === t ? "active" : ""}
                          onClick={() => setBottom(t)}>
                    {t[0].toUpperCase() + t.slice(1)}
                  </button>
                ))}
              </div>
              <div className="tab-body">{bottom === "problems" ? <Problems />
                : bottom === "why" ? <Why /> : isPipeline ? <PipelinePanel /> : <Training />}
              </div>
            </>
          ),
        }} />
      <AddHelper refObj={flowRef} addBlock={addBlock} setSelected={setSelected} />
    </ReactFlowProvider>
  );
}

/** Places double-clicked library blocks at the centre of the visible canvas. */
function AddHelper({ refObj, addBlock, setSelected }: {
  refObj: { current: { add: (t: string) => void } };
  addBlock: (t: string, p: { x: number; y: number }) => string | null;
  setSelected: (id: string | null) => void;
}) {
  const flow = useReactFlow();
  refObj.current.add = (typeId: string) => {
    const el = document.querySelector(".canvas")?.getBoundingClientRect();
    const center = el ? { x: el.left + el.width / 2, y: el.top + el.height / 2 } : { x: 400, y: 300 };
    const id = addBlock(typeId, flow.screenToFlowPosition(center));
    if (id) setSelected(id);
  };
  return null;
}
