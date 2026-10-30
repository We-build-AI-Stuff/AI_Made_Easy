import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ACTIONS, type ActionDef, matchesKeys } from "./actions";
import { api } from "./api";
import { BackendsModal } from "./components/Backends";
import { CommandPalette, type PaletteMode, ShortcutsSheet } from "./components/CommandPalette";
import { FileMenu } from "./components/FileMenu";
import { ImportDialog } from "./components/ImportDialog";
import { OpenDialog } from "./components/OpenDialog";
import { LayoutMenu } from "./components/LayoutMenu";
import { Toasts } from "./components/Toasts";
import { ToolsModal } from "./components/Tools";
import { resetTour, Tour, tourDone } from "./components/Tour";
import { NewProjectWizard } from "./components/Wizard";
import { Designer } from "./designer/Designer";
import { DockProvider, useDockState } from "./designer/dock";
import { downloadCanvas } from "./exportPng";
import { DataPage } from "./pages/DataPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { ModelsPage } from "./pages/ModelsPage";
import { rememberProject } from "./recent";
import { useStore } from "./state";
import type { DataFacts } from "./types";

const PAGES = [["design", "Design"], ["data", "Data"], ["experiments", "Experiments"],
               ["models", "Models"]] as const;
type Page = (typeof PAGES)[number][0];
type Dialog = "" | "open" | "import" | "new" | "tools" | "backends" | "shortcuts";

/** Keys typed into a field belong to the field (Escape and the palette still work). */
const typing = () => !!(document.activeElement as HTMLElement | null)
  ?.closest?.("input, textarea, select, [contenteditable=true]");
// the Designer handles these itself (it needs the selection and the clipboard events)
const CANVAS_OWNED = new Set(["edit.undo", "edit.redo", "edit.copy", "edit.paste", "edit.duplicate",
                              "edit.group", "edit.ungroup", "edit.delete"]);
// in a text field these keep their usual meaning
const FIELD_KEYS = new Set(["edit.select_all", "edit.find", "view.zoom_in", "view.zoom_out",
                            "view.zoom_fit", "help.shortcuts"]);

// the first-run tour; automated browsers (tests) only see it when asked for
const showTourAtStart = () => !tourDone()
  && (!navigator.webdriver || localStorage.getItem("aime.tour") === "show");

export function App() {
  const store = useStore();
  const dock = useDockState();
  const { name, setName, project, notify, theme, toggleTheme, version, validation, nodes,
          dataIssues, undo, redo, duplicateSelection, groupSelection, copySelection, arrange,
          canvas, load, restored, discardRestored } = store;
  const [page, setPage] = useState<Page>("design");
  const [experimentsTab, setExperimentsTab] = useState("runs");
  const [design, setDesign] = useState("");
  useEffect(() => {
    if (!nodes.length) return setDesign("");
    api.describe(project())
      .then((d) => setDesign(d.task ? `${d.family.label} · ${d.task.label}` : d.family.label))
      .catch((e) => {  // only the status bar label: say why it is missing in the console
        console.warn("describe failed:", e);
        setDesign("");
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [validation]);
  const [dialog, setDialog] = useState<Dialog>("");
  const [palette, setPalette] = useState<PaletteMode | null>(null);
  const [tour, setTour] = useState(showTourAtStart);
  const [startFacts, setStartFacts] = useState<DataFacts | null>(null);

  const save = useCallback(async () => {
    try {
      await api.saveProject(name, project());
      rememberProject(name);
      notify(`Saved “${name}”`);
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  }, [name, project, notify]);
  const download = useCallback(() => {
    const blob = new Blob([JSON.stringify(project(), null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${name || "project"}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, [name, project]);
  const openRecent = useCallback((saved: string) => {
    api.project(saved).then((p) => { load(p); rememberProject(saved); setPage("design"); })
      .catch((e) => notify(`Could not open “${saved}”: ${(e as Error).message}`, true));
  }, [load, notify]);
  /** Canvas commands need the Design page: switch there first. */
  const onCanvas = useCallback((fn: () => void) => {
    if (page !== "design") {
      setPage("design");
      window.setTimeout(fn, 80);
    } else {
      fn();
    }
  }, [page]);

  const handlers: Record<string, () => void> = useMemo(() => ({
    "file.new": () => setDialog("new"),
    "file.open": () => setDialog("open"),
    "file.save": save,
    "file.download": download,
    "file.import_model": () => setDialog("import"),
    "file.export_png": () => onCanvas(() => {
      if (!nodes.length) return notify("The canvas is empty: nothing to export", true);
      const image = canvas.current?.image();
      if (image) downloadCanvas(image, `${name || "design"}.png`);
    }),
    "edit.undo": undo,
    "edit.redo": redo,
    "edit.copy": () => {
      const text = copySelection();
      if (!text) return notify("Select blocks to copy first", true);
      navigator.clipboard?.writeText(text).then(() => notify("Copied"))
        .catch(() => notify("Press Ctrl/⌘ C on the canvas to copy", true));
    },
    "edit.paste": () => notify("Press Ctrl/⌘ V on the canvas to paste"),
    "edit.duplicate": () => onCanvas(() => { duplicateSelection(); }),
    "edit.select_all": () => onCanvas(() => canvas.current?.selectAll()),
    "edit.delete": () => onCanvas(() => canvas.current?.deleteSelection()),
    "edit.group": () => onCanvas(() => {
      if (!groupSelection()) notify("Select the blocks to group first", true);
    }),
    "edit.ungroup": () => onCanvas(() => canvas.current?.ungroupSelection()),
    "edit.collapse_group": () => onCanvas(() => {
      if (!canvas.current?.toggleGroups()) notify("Add a group first: select blocks, Ctrl/⌘ G", true);
    }),
    "edit.note": () => onCanvas(() => canvas.current?.addNote()),
    "edit.find": () => onCanvas(() => setPalette("find")),
    "view.auto_layout": () => onCanvas(arrange),
    "view.zoom_in": () => onCanvas(() => canvas.current?.zoomIn()),
    "view.zoom_out": () => onCanvas(() => canvas.current?.zoomOut()),
    "view.zoom_fit": () => onCanvas(() => canvas.current?.fit()),
    "view.theme": toggleTheme,
    "view.panel_library": () => dock.toggle("library"),
    "view.panel_inspector": () => dock.toggle("inspector"),
    "view.panel_console": () => dock.toggle("console"),
    "view.reset_layout": () => { dock.reset(); notify("Panels back to the default layout"); },
    "go.design": () => setPage("design"),
    "go.data": () => setPage("data"),
    "go.experiments": () => setPage("experiments"),
    "go.models": () => setPage("models"),
    "run.tracking": () => setDialog("tools"),
    "run.backends": () => setDialog("backends"),
    "help.palette": () => setPalette("commands"),
    "help.shortcuts": () => setDialog("shortcuts"),
    "help.tour": () => { resetTour(); setPage("design"); setTour(true); },
  }), [save, download, onCanvas, nodes.length, notify, canvas, name, undo, redo, copySelection,
       duplicateSelection, groupSelection, arrange, toggleTheme, dock]);
  const run = useCallback((id: string) => handlers[id]?.(), [handlers]);
  const available: ActionDef[] = useMemo(() => ACTIONS.filter((a) => handlers[a.id]), [handlers]);

  // one listener for the app's shortcuts; it reads the latest handlers through a ref
  const latest = useRef({ handlers, page, overlay: false });
  latest.current = { handlers, page, overlay: !!dialog || !!palette };
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const { handlers: h, page: p, overlay } = latest.current;
      for (const a of ACTIONS) {
        if (!a.keys || CANVAS_OWNED.has(a.id) || !matchesKeys(e, a.keys)) continue;
        if (a.id !== "help.palette" && a.id !== "file.save" && overlay) return;
        if (typing() && (FIELD_KEYS.has(a.id) || !a.keys.includes("Mod"))) return;
        const canvasOnly = a.id.startsWith("view.zoom") || a.id === "edit.select_all";
        if (canvasOnly && p !== "design") return;
        e.preventDefault();
        if (a.id === "help.palette") {
          setPalette((m) => (m ? null : "commands"));
        } else {
          h[a.id]?.();
        }
        return;
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);

  const errors = validation?.issues.filter((i) => i.severity === "error").length ?? 0;
  const warnings = (validation?.issues.length ?? 0) - errors + dataIssues.length;
  return (
    <DockProvider value={dock}>
    <div className="app">
      <header className="topbar">
        <div className="brand">AI <span>Made Easy</span></div>
        <FileMenu run={run} openRecent={openRecent} />
        <LayoutMenu />
        <nav className="nav" aria-label="Pages">
          {PAGES.map(([id, label]) => (
            <button key={id} className={page === id ? "active" : ""} onClick={() => setPage(id)}
                    aria-current={page === id ? "page" : undefined}>
              {label}</button>
          ))}
        </nav>
        <div className="spacer" />
        <input className="project-name" value={name} onChange={(e) => setName(e.target.value)}
               aria-label="Project name" />
        <button onClick={save} title="Save on the server (Ctrl/⌘ S)">Save</button>
        <button onClick={() => setDialog("tools")} data-testid="open-tools"
                title="Experiment tracking, Kaggle and Hub datasets, run history">Tools</button>
        <button onClick={() => setDialog("backends")} data-testid="open-backends"
                title="Where runs train: this machine, SSH, Kaggle, Colab, Modal">Backends</button>
        <button className="ghost palette-button" onClick={() => setPalette("commands")}
                title="Command palette (Ctrl/⌘ K)" aria-label="Command palette" data-testid="open-palette">
          ⌘K</button>
        <button className="ghost" onClick={() => setDialog("shortcuts")} aria-label="Keyboard shortcuts"
                title="Keyboard shortcuts (?)">?</button>
        <button className="ghost" onClick={toggleTheme} aria-label="Toggle theme"
                title={theme === "dark" ? "Light theme" : "Dark theme"}>
          {theme === "dark" ? "☀" : "☾"}</button>
      </header>
      {restored && (
        <div className="banner" role="region" aria-label="Restored design" data-testid="restored">
          <span>Restored your unsaved design from last time.</span>
          <button onClick={() => discardRestored(true)}>Keep it</button>
          <button className="ghost" onClick={() => discardRestored(false)}>Start fresh</button>
        </div>)}
      {page === "design" && <Designer onDataFile={(file) => {
        notify(`Reading ${file.name}…`);
        api.uploadData(file).then((d) => { setStartFacts(d.facts); setDialog("new"); })
          .catch((e) => notify(String((e as Error).message), true));
      }} />}
      {page === "data" && <DataPage />}
      {page === "experiments" && <ExperimentsPage initialTab={experimentsTab} />}
      {page === "models" && <ModelsPage />}
      <footer className="statusbar">
        <span className={errors ? "sev-error" : warnings ? "sev-warning" : "sev-ok"} data-testid="status">
          {!nodes.length ? "Empty design" : errors ? `${errors} error(s)` : warnings
            ? `${warnings} warning(s)` : "No problems"}</span>
        <span>{nodes.length} blocks</span>
        {design && <span data-testid="design-kind">{design}</span>}
        <span className="spacer" />
        <span>AI Made Easy {version}</span>
      </footer>
      {dialog === "open" && <OpenDialog onClose={() => setDialog("")} />}
      {dialog === "import" && <ImportDialog onClose={() => setDialog("")} />}
      {dialog === "tools" && <ToolsModal onClose={() => setDialog("")}
        onUseData={(d) => { setStartFacts(d.facts); setDialog("new"); }} />}
      {dialog === "backends" && <BackendsModal onClose={() => setDialog("")} />}
      {dialog === "shortcuts" && <ShortcutsSheet actions={available} onClose={() => setDialog("")} />}
      {dialog === "new" && <NewProjectWizard initialFacts={startFacts}
        onClose={() => { setDialog(""); setStartFacts(null); }}
        onCreated={() => { setDialog(""); setStartFacts(null); setPage("design"); }}
        onAutoml={() => { setDialog(""); setExperimentsTab("sweeps"); setPage("experiments"); }} />}
      {palette && <CommandPalette mode={palette} actions={available} run={run}
                                  onClose={() => setPalette(null)} />}
      {tour && page === "design" && !dialog && <Tour onClose={() => setTour(false)} />}
      <Toasts />
    </div>
    </DockProvider>
  );
}
