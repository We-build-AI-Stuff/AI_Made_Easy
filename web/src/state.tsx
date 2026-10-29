// App-wide state: the block catalog, the open project (React Flow nodes / edges), live
// validation, undo history and toasts.
import {
  applyEdgeChanges, applyNodeChanges, type Edge, type EdgeChange, type NodeChange,
} from "@xyflow/react";
import {
  createContext, type MutableRefObject, useCallback, useContext, useEffect, useMemo, useRef, useState,
  type ReactNode,
} from "react";

import { api } from "./api";
import { useDialogs } from "./components/Dialogs";
import {
  addGroup, addNote, cleanMeta, copyFragment, type Note, notesOf, parseClipboard, pasteFragment,
  groupsOf, setCollapsed,
} from "./canvas";
import {
  autoLayout, defaultParams, type FlowNode, fromProject, hasOverlaps, newNodeId, structureKey,
  toProject,
} from "./project";
import type { BlockDef, Project, Validation } from "./types";

export type Toast = { id: number; text: string; error?: boolean; count: number };
const MAX_TOASTS = 4;

/** What the visible canvas can do; the Designer registers it (null on other pages). */
export interface CanvasApi {
  zoomIn: () => void;
  zoomOut: () => void;
  fit: () => void;
  /** Select a block and centre the view on it. */
  focus: (id: string) => void;
  selectAll: () => void;
  deleteSelection: () => void;
  ungroupSelection: () => void;
  /** Fold / open the groups around the selection (all groups when nothing is selected). */
  toggleGroups: () => number;
  addNote: () => void;
  /** The design drawn as an image. */
  image: () => HTMLCanvasElement;
}

interface Store {
  blocks: BlockDef[];
  blockMap: Map<string, BlockDef>;
  name: string;
  setName: (name: string) => void;
  nodes: FlowNode[];
  edges: Edge[];
  onNodesChange: (changes: NodeChange<FlowNode>[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  setEdges: (fn: (edges: Edge[]) => Edge[]) => void;
  project: () => Project;
  meta: Record<string, unknown>;
  setMeta: (fn: (meta: Record<string, unknown>) => Record<string, unknown>) => void;
  load: (project: Project, fit?: boolean) => void;
  loads: number;
  arrange: () => void;
  addBlock: (typeId: string, position: { x: number; y: number }) => string | null;
  setParam: (nodeId: string, name: string, value: unknown) => void;
  /** Clipboard text for the selected blocks (null when nothing is selected). */
  copySelection: () => string | null;
  /** Paste blocks (a copied selection or a whole design); returns how many. */
  pasteText: (text: string, at?: [number, number]) => number;
  duplicateSelection: () => number;
  selectedIds: () => string[];
  selectIds: (ids: string[]) => void;
  groupSelection: (title?: string) => string | null;
  ungroup: (id: string) => void;
  renameGroup: (id: string, title: string) => void;
  /** Fold a group into one card, or open it. */
  collapseGroup: (id: string, collapsed: boolean) => void;
  addNoteAt: (position: [number, number], text?: string) => string;
  updateNote: (id: string, patch: Partial<Note>) => void;
  removeNote: (id: string) => void;
  commit: () => void;
  undo: () => void;
  redo: () => void;
  validation: Validation | null;
  dataIssues: { severity: string; message: string; node_id: string }[];
  selected: string | null;
  setSelected: (id: string | null) => void;
  selectNode: (id: string) => void;
  /** Newest last; each closes itself (errors stay longer) or with its × button. */
  toasts: Toast[];
  notify: (text: string, error?: boolean) => void;
  dismissToast: (id: number) => void;
  theme: string;
  toggleTheme: () => void;
  version: string;
  canvas: MutableRefObject<CanvasApi | null>;
  /** The design was restored from the autosave at start (the app offers to discard it). */
  restored: boolean;
  discardRestored: (keep: boolean) => void;
  /** Loads the block catalog again (after sign-in). */
  refreshCatalog: () => void;
}

const Ctx = createContext<Store | null>(null);

export function useStore(): Store {
  const store = useContext(Ctx);
  if (!store) throw new Error("useStore outside StoreProvider");
  return store;
}

const AUTOSAVE = "aime.autosave";

export function StoreProvider({ children }: { children: ReactNode }) {
  const [blocks, setBlocks] = useState<BlockDef[]>([]);
  const [name, setName] = useState("untitled");
  const [nodes, setNodes] = useState<FlowNode[]>([]);
  const [edges, setEdgesState] = useState<Edge[]>([]);
  const [meta, setMeta] = useState<Record<string, unknown>>({});
  const [schema, setSchema] = useState(3);
  const [loads, setLoads] = useState(0);
  const [validation, setValidation] = useState<Validation | null>(null);
  const [dataIssues, setDataIssues] = useState<Store["dataIssues"]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [theme, setTheme] = useState(localStorage.getItem("aime.theme") ?? "dark");
  const [version, setVersion] = useState("");
  const canvas = useRef<CanvasApi | null>(null);
  const [restored, setRestored] = useState(false);
  const edited = useRef(false);   // the user has changed or opened a design
  const past = useRef<Project[]>([]);
  const future = useRef<Project[]>([]);
  const toastTimers = useRef(new Map<number, number>());
  const toastSeq = useRef(0);
  const { prompt } = useDialogs();

  const blockMap = useMemo(() => new Map(blocks.map((b) => [b.type_id, b])), [blocks]);

  const dismissToast = useCallback((id: number) => {
    window.clearTimeout(toastTimers.current.get(id));
    toastTimers.current.delete(id);
    setToasts((ts) => ts.filter((t) => t.id !== id));
  }, []);
  const notify = useCallback((text: string, error = false) => {
    const schedule = (id: number) => {
      window.clearTimeout(toastTimers.current.get(id));
      toastTimers.current.set(id, window.setTimeout(() => dismissToast(id), error ? 8000 : 3500));
    };
    setToasts((ts) => {
      const same = ts.find((t) => t.text === text && !!t.error === error);
      if (same) {  // the same message again: count it instead of stacking copies
        schedule(same.id);
        return ts.map((t) => (t === same ? { ...t, count: t.count + 1 } : t));
      }
      const id = ++toastSeq.current;
      schedule(id);
      const next = [...ts, { id, text, error, count: 1 }];
      for (const old of next.slice(0, Math.max(0, next.length - MAX_TOASTS))) {
        window.clearTimeout(toastTimers.current.get(old.id));
        toastTimers.current.delete(old.id);
      }
      return next.slice(-MAX_TOASTS);
    });
  }, [dismissToast]);

  const project = useCallback(() => toProject(name, nodes, edges, meta, schema),
                              [name, nodes, edges, meta, schema]);

  const load = useCallback((p: Project, fit = true) => {
    const { nodes: n, edges: e } = fromProject(p);
    setNodes(hasOverlaps(n) ? autoLayout(n, e) : n);
    setEdgesState(e);
    setName(p.name || "untitled");
    setMeta((p.meta as Record<string, unknown>) ?? {});
    setSchema(Number(p.schema_version ?? 1));
    setSelected(null);
    if (fit) setLoads((c) => c + 1);
  }, []);

  // ---- history: snapshot before structural edits
  const latest = useRef(project);
  latest.current = project;
  const commit = useCallback(() => {
    edited.current = true;
    past.current.push(latest.current());
    if (past.current.length > 100) past.current.shift();
    future.current = [];
  }, []);
  const undo = useCallback(() => {
    const prev = past.current.pop();
    if (!prev) return;
    future.current.push(latest.current());
    load(prev, false);
  }, [load]);
  const redo = useCallback(() => {
    const next = future.current.pop();
    if (!next) return;
    past.current.push(latest.current());
    load(next, false);
  }, [load]);

  const onNodesChange = useCallback((changes: NodeChange<FlowNode>[]) => {
    if (changes.some((c) => c.type === "remove")) commit();
    setNodes((ns) => applyNodeChanges(changes, ns));
  }, [commit]);
  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    if (changes.some((c) => c.type === "remove")) commit();
    setEdgesState((es) => applyEdgeChanges(changes, es));
  }, [commit]);
  const setEdges = useCallback((fn: (edges: Edge[]) => Edge[]) => setEdgesState(fn), []);

  const addBlock = useCallback((typeId: string, position: { x: number; y: number }) => {
    const def = blockMap.get(typeId);
    if (!def) return null;
    commit();
    let id = "";
    setNodes((ns) => {
      id = newNodeId(typeId, new Set(ns.map((n) => n.id)));
      return [...ns.map((n) => ({ ...n, selected: false })),
              { id, type: "block", position, selected: true,
                data: { typeId, params: defaultParams(def) } }];
    });
    return id;
  }, [blockMap, commit]);

  // ---- clipboard, groups and notes (3.3; see canvas.ts)
  const latestNodes = useRef<FlowNode[]>([]);
  latestNodes.current = nodes;
  const selectedIds = useCallback(
    () => latestNodes.current.filter((n) => n.selected).map((n) => n.id), []);
  const selectIds = useCallback((ids: string[]) => {
    const chosen = new Set(ids);
    setNodes((ns) => ns.map((n) => ({ ...n, selected: chosen.has(n.id) })));
    setSelected(ids.length === 1 ? ids[0] : null);
  }, []);
  const copySelection = useCallback(() => {
    const ids = selectedIds();
    if (!ids.length) return null;
    return JSON.stringify(copyFragment(latest.current(), ids), null, 1);
  }, [selectedIds]);
  const pasteText = useCallback((text: string, at?: [number, number]) => {
    const fragment = parseClipboard(text);
    if (!fragment || !fragment.nodes.length) return 0;
    const unknown = [...new Set(fragment.nodes.map((n) => n.type).filter((t) => !blockMap.has(t)))];
    if (unknown.length) {
      notify(`Can't paste: unknown block type ${unknown.join(", ")}`, true);
      return 0;
    }
    commit();
    const { design, ids } = pasteFragment(latest.current(), fragment, [40, 40], at);
    const chosen = new Set(ids);
    const added = fromProject({
      ...design, nodes: design.nodes.filter((n) => chosen.has(n.id)),
      edges: design.edges.filter((e) => chosen.has(e.from.slice(0, e.from.lastIndexOf("/")))),
    });
    setNodes((ns) => [...ns.map((n) => ({ ...n, selected: false })),
                      ...added.nodes.map((n) => ({ ...n, selected: true }))]);
    setEdgesState((es) => [...es, ...added.edges]);
    setMeta((design.meta as Record<string, unknown>) ?? {});
    setSelected(ids.length === 1 ? ids[0] : null);
    return ids.length;
  }, [blockMap, commit, notify]);
  const duplicateSelection = useCallback(() => {
    const text = copySelection();
    return text ? pasteText(text) : 0;
  }, [copySelection, pasteText]);
  const groupSelection = useCallback((title = "Group") => {
    const ids = selectedIds();
    if (!ids.length) return null;
    commit();
    let made = "";
    setMeta((m) => {
      const { meta: next, id } = addGroup(m, ids, title);
      made = id;
      return next;
    });
    return made || "group";
  }, [commit, selectedIds]);
  const ungroup = useCallback((id: string) => {
    commit();
    setMeta((m) => cleanMeta({ ...m, groups: ((m.groups as { id: string }[]) ?? [])
      .filter((g) => g.id !== id) }, new Set(latestNodes.current.map((n) => n.id))));
  }, [commit]);
  const renameGroup = useCallback((id: string, title: string) => {
    setMeta((m) => ({ ...m, groups: ((m.groups as { id: string }[]) ?? [])
      .map((g) => (g.id === id ? { ...g, title } : g)) }));
  }, []);
  const collapseGroup = useCallback((id: string, collapsed: boolean) => {
    commit();
    if (collapsed) {   // the hidden blocks can't stay selected
      const members = new Set(groupsOf(latest.current().meta ?? {}).find((g) => g.id === id)?.nodes);
      setNodes((ns) => ns.map((n) => (members.has(n.id) && n.selected ? { ...n, selected: false } : n)));
    }
    setMeta((m) => setCollapsed(m, id, collapsed));
  }, [commit]);
  const addNoteAt = useCallback((position: [number, number], text = "") => {
    commit();
    let made = "";
    setMeta((m) => {
      const { meta: next, id } = addNote(m, text, position);
      made = id;
      return next;
    });
    return made;
  }, [commit]);
  const updateNote = useCallback((id: string, patch: Partial<Note>) => {
    setMeta((m) => ({ ...m, notes: notesOf(m).map((n) => (n.id === id ? { ...n, ...patch } : n)) }));
  }, []);
  const removeNote = useCallback((id: string) => {
    commit();
    setMeta((m) => {
      const notes = notesOf(m).filter((n) => n.id !== id);
      const next: Record<string, unknown> = { ...m, notes };
      if (!notes.length) delete next.notes;
      return next;
    });
  }, [commit]);
  // deleting blocks takes them out of their groups (empty groups go)
  const idsKey = nodes.map((n) => n.id).join("|");
  useEffect(() => {
    setMeta((m) => {
      if (!Array.isArray(m.groups)) return m;
      const next = cleanMeta(m, new Set(idsKey.split("|")));
      return JSON.stringify(next) === JSON.stringify(m) ? m : next;
    });
  }, [idsKey]);

  const setParam = useCallback((nodeId: string, param: string, value: unknown) => {
    setNodes((ns) => ns.map((n) => (n.id === nodeId
      ? { ...n, data: { ...n.data, params: { ...n.data.params, [param]: value } } } : n)));
  }, []);

  // ---- boot: catalog, version, autosaved project or the demo sample (once). The sample
  // arrives late on a busy server: it never replaces a design the user has started on.
  const booted = useRef(false);
  const loadSample = useCallback(() => api.sample("mnist_cnn.json")
    .then((p) => { if (!edited.current) load(p); }).catch(() => undefined), [load]);
  const refreshCatalog = useCallback(() => api.blocks()
    .then((r) => setBlocks(r.blocks))
    .catch(() => undefined), []);   // called again after sign-in: the boot fetch ran
  useEffect(() => {                 // before the session existed and got 401s
    if (booted.current) return;
    booted.current = true;
    let attempt = 0;
    const fetchBlocks = () => api.blocks().then((r) => setBlocks(r.blocks)).catch((e) => {
      if (++attempt < 6) window.setTimeout(fetchBlocks, 1000 * attempt); // server starting up
      else notify(`Could not load the block catalog: ${e.message ?? e}`, true);
    });
    fetchBlocks();
    api.info().then((r) => {
      setVersion(r.version);
      if (r.auth && !localStorage.getItem("aime.token")) {
        prompt("This server needs the access token it printed when it started (AIME_WEB_TOKEN).",
               { title: "Access token", label: "Access token", secret: true, ok: "Connect" })
          .then((token) => {
            if (token) {
              api.setToken(token.trim());
              window.location.reload();
            }
          });
      }
    }).catch(() => undefined);
    const saved = localStorage.getItem(AUTOSAVE);
    if (saved) {
      try {
        const p = JSON.parse(saved) as Project;
        load(p);
        if (p.nodes?.length) setRestored(true);
        return;
      } catch {
        /* fall through to the sample */
      }
    }
    loadSample();
  }, [load, notify, prompt, loadSample]);

  // ---- live validation (debounced, structure only)
  const key = structureKey(nodes, edges) + JSON.stringify(meta.budget ?? null);
  useEffect(() => {
    if (!nodes.length) {
      setValidation(null);
      setDataIssues([]);
      return;
    }
    const graph = latest.current();
    const timer = window.setTimeout(() => {
      api.validate(graph).then(setValidation).catch(() => setValidation(null));
      if (graph.nodes.some((n) => n.type.startsWith("data."))) {
        api.dataIssues(graph).then((r) => setDataIssues(r.issues)).catch(() => setDataIssues([]));
      } else {
        setDataIssues([]);
      }
    }, 350);
    return () => window.clearTimeout(timer);
  }, [key, nodes.length]);

  // ---- autosave (positions included)
  useEffect(() => {
    if (!nodes.length) return;
    const timer = window.setTimeout(
      () => localStorage.setItem(AUTOSAVE, JSON.stringify(latest.current())), 800);
    return () => window.clearTimeout(timer);
  }, [nodes, edges, name, meta]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("aime.theme", theme);
  }, [theme]);
  const toggleTheme = useCallback(() => setTheme((t) => (t === "dark" ? "light" : "dark")), []);

  const selectNode = useCallback((id: string) => {
    setNodes((ns) => ns.map((n) => ({ ...n, selected: n.id === id })));
    setSelected(id);
  }, []);

  const arrange = useCallback(() => {
    commit();
    setNodes((ns) => autoLayout(ns, latestEdges.current));
    setLoads((c) => c + 1);
  }, [commit]);
  const latestEdges = useRef<Edge[]>([]);
  latestEdges.current = edges;

  const discardRestored = useCallback((keep: boolean) => {
    setRestored(false);
    if (keep) return;
    localStorage.removeItem(AUTOSAVE);
    past.current = [];
    future.current = [];
    edited.current = false;
    loadSample();
  }, [loadSample]);

  const store: Store = {
    blocks, blockMap, name, setName, nodes, edges, onNodesChange, onEdgesChange, setEdges,
    project, meta, setMeta, load, loads, arrange, addBlock, setParam, commit, undo, redo, validation, dataIssues, selected,
    copySelection, pasteText, duplicateSelection, selectedIds, selectIds, groupSelection, ungroup,
    renameGroup, collapseGroup, addNoteAt, updateNote, removeNote,
    setSelected, selectNode, toasts, notify, dismissToast, theme, toggleTheme, version, canvas,
    restored, discardRestored, refreshCatalog,
  };
  return <Ctx.Provider value={store}>{children}</Ctx.Provider>;
}
