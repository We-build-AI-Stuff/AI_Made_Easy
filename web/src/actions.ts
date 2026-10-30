// Every command the web app offers (3.3): the command palette (Ctrl/⌘ K), the shortcuts sheet
// (?) and the File menu all read this list, so a shortcut is defined once. Ids and labels follow
// the desktop's ai_made_easy/ui/actions_catalog.py where both apps have the command.
// "Mod" is ⌘ on a Mac and Ctrl elsewhere.

export interface ActionDef {
  id: string;
  label: string;
  group: "File" | "Edit" | "View" | "Canvas" | "Go" | "Help";
  keys?: string;
  /** Other words the palette matches. */
  hint?: string;
}

export const ACTIONS: ActionDef[] = [
  { id: "file.new", label: "New project…", group: "File", hint: "wizard recipe automl start" },
  { id: "file.open", label: "Open…", group: "File", keys: "Mod+O", hint: "load sample example" },
  { id: "file.save", label: "Save", group: "File", keys: "Mod+S" },
  { id: "file.download", label: "Download project JSON", group: "File", hint: "export file" },
  { id: "file.import_model", label: "Import model…", group: "File", hint: "onnx keras pytorch mlflow" },
  { id: "file.export_png", label: "Export canvas as PNG", group: "File",
    hint: "image picture screenshot" },

  { id: "edit.undo", label: "Undo", group: "Edit", keys: "Mod+Z" },
  { id: "edit.redo", label: "Redo", group: "Edit", keys: "Mod+Shift+Z" },
  { id: "edit.copy", label: "Copy blocks", group: "Edit", keys: "Mod+C" },
  { id: "edit.paste", label: "Paste blocks", group: "Edit", keys: "Mod+V" },
  { id: "edit.duplicate", label: "Duplicate blocks", group: "Edit", keys: "Mod+D" },
  { id: "edit.select_all", label: "Select all blocks", group: "Edit", keys: "Mod+A" },
  { id: "edit.delete", label: "Delete selection", group: "Edit", keys: "Delete" },
  { id: "edit.group", label: "Group selected blocks", group: "Edit", keys: "Mod+G", hint: "frame" },
  { id: "edit.ungroup", label: "Ungroup", group: "Edit", keys: "Mod+Shift+U" },
  { id: "edit.collapse_group", label: "Collapse / expand groups", group: "Edit",
    hint: "fold card hide" },
  { id: "edit.note", label: "Add a sticky note", group: "Edit", hint: "comment annotation" },

  { id: "edit.find", label: "Find a block on the canvas…", group: "Canvas", keys: "Mod+F",
    hint: "search locate" },
  { id: "view.auto_layout", label: "Arrange blocks", group: "Canvas", hint: "layout tidy auto" },
  { id: "view.zoom_in", label: "Zoom in", group: "View", keys: "Mod+=" },
  { id: "view.zoom_out", label: "Zoom out", group: "View", keys: "Mod+-" },
  { id: "view.zoom_fit", label: "Fit the design in view", group: "View", keys: "Mod+0", hint: "zoom reset" },
  { id: "view.theme", label: "Toggle light / dark theme", group: "View", hint: "dark mode colour" },
  { id: "view.panel_library", label: "Collapse / restore the Block Library", group: "View",
    hint: "hide show panel dock" },
  { id: "view.panel_inspector", label: "Collapse / restore the Inspector", group: "View",
    hint: "hide show panel properties sidebar" },
  { id: "view.panel_console", label: "Collapse / restore the bottom panels", group: "View",
    hint: "hide show training problems console" },
  { id: "view.reset_layout", label: "Reset the panel layout", group: "View",
    hint: "restore default dock panels workspace" },

  { id: "go.design", label: "Go to Design", group: "Go", keys: "Alt+1" },
  { id: "go.data", label: "Go to Data", group: "Go", keys: "Alt+2" },
  { id: "go.experiments", label: "Go to Experiments", group: "Go", keys: "Alt+3", hint: "runs sweeps" },
  { id: "go.models", label: "Go to Models", group: "Go", keys: "Alt+4", hint: "registry share deploy" },
  { id: "run.tracking", label: "Tools…", group: "Go", hint: "tracking kaggle hub datasets" },
  { id: "run.backends", label: "Backends…", group: "Go", hint: "ssh kaggle colab modal remote" },

  { id: "help.palette", label: "Command palette", group: "Help", keys: "Mod+K" },
  { id: "help.shortcuts", label: "Keyboard shortcuts", group: "Help", keys: "?" },
  { id: "help.tour", label: "Show the tour again", group: "Help", hint: "help intro onboarding" },
];

export const ACTION_MAP = new Map(ACTIONS.map((a) => [a.id, a]));

export const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

/** "Mod+Shift+E" → "⌘⇧E" on a Mac, "Ctrl+Shift+E" elsewhere. */
export function keyLabel(keys: string): string {
  if (!isMac) return keys.replace("Mod", "Ctrl");
  return keys.replace("Mod+", "⌘").replace("Shift+", "⇧").replace("Alt+", "⌥");
}

/** Does this key event match "Mod+Shift+E"-style keys? */
export function matchesKeys(e: KeyboardEvent, keys: string): boolean {
  const parts = keys.split("+");
  const key = parts.pop()!.toLowerCase();
  const mod = parts.includes("Mod");
  const shift = parts.includes("Shift");
  const alt = parts.includes("Alt");
  if (mod !== (e.metaKey || e.ctrlKey) || alt !== e.altKey) return false;
  if (key === "?") return e.key === "?";
  if (shift !== e.shiftKey) return false;
  const pressed = e.key.toLowerCase();
  if (key === "=") return pressed === "=" || pressed === "+";
  if (alt && /^\d$/.test(key)) return e.code === `Digit${key}`;   // ⌥1 types "¡" on a Mac
  return pressed === key;
}

/** Fuzzy-ish match: every word of the query appears in the label, group or hint. */
export function matchesQuery(text: string, query: string): boolean {
  const hay = text.toLowerCase();
  return query.toLowerCase().split(/\s+/).filter(Boolean).every((w) => hay.includes(w));
}
