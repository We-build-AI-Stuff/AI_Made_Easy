"""Canvas editing: undo, clipboard, groups, notes, zoom and layout, custom blocks.

The ``act_*`` slots of :class:`~ai_made_easy.ui.context.AppContext` for this area, as a
mixin: the methods use the stores, services and panels AppContext builds (split out of
context.py in 3.10; no behaviour change).
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ai_made_easy.ui.context_common import _SETTINGS
from ai_made_easy.ui.dialogs import SaveTemplateDialog


class CanvasActions:
    def act_undo(self, *_):
        self.canvas.node_graph.undo_stack().undo()

    def act_redo(self, *_):
        self.canvas.node_graph.undo_stack().redo()

    def act_delete(self, *_):
        focus = QtWidgets.QApplication.focusWidget()
        if isinstance(focus, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit,
                              QtWidgets.QAbstractSpinBox, QtWidgets.QTextEdit)):
            return
        if self.canvas.delete_selected():
            self.properties.clear()

    def act_select_all(self, *_):
        self.canvas.select_all()

    @staticmethod
    def _text_focus():
        focus = QtWidgets.QApplication.focusWidget()
        if isinstance(focus, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit,
                              QtWidgets.QTextEdit, QtWidgets.QAbstractSpinBox)):
            return focus
        return None

    def act_copy(self, *_):
        focus = self._text_focus()
        if focus is not None:
            if hasattr(focus, "copy"):
                focus.copy()
            return
        text = self.canvas.copy_selection()
        if text is None:
            self.status_message.emit("Select blocks to copy")
            return
        QtWidgets.QApplication.clipboard().setText(text)
        n = len(self.canvas.selected_ids())
        self.status_message.emit(f"Copied {n} block{'s' if n != 1 else ''}")

    def act_paste(self, *_):
        focus = self._text_focus()
        if focus is not None:
            if hasattr(focus, "paste"):
                focus.paste()
            return
        text = QtWidgets.QApplication.clipboard().text()
        try:
            n = self.canvas.paste_text(text)
        except ValueError as exc:
            self.status_message.emit(f"Can't paste: {exc}")
            return
        if n:
            self.status_message.emit(f"Pasted {n} block{'s' if n != 1 else ''}")
        else:
            self.status_message.emit("The clipboard holds no blocks: copy some first")

    def act_duplicate(self, *_):
        if self._text_focus() is not None:
            return
        n = self.canvas.duplicate_selected()
        if n:
            self.status_message.emit(f"Duplicated {n} block{'s' if n != 1 else ''}")

    def act_group(self, *_):
        title, ok = QtWidgets.QInputDialog.getText(self.window, "Group", "Name of the group:",
                                                   text="Group")
        if not ok:
            return
        if self.canvas.group_selected(title.strip() or "Group") is None:
            self.status_message.emit("Select the blocks to group first")

    def act_ungroup(self, *_):
        if not self.canvas.ungroup_selected():
            self.status_message.emit("Select a group (or a block in one) to ungroup")

    def act_collapse_group(self, *_):
        if not self.canvas.toggle_selected_groups():
            self.status_message.emit("Select a group (or a block in one) to collapse it")

    def act_add_note(self, *_):
        text, ok = QtWidgets.QInputDialog.getMultiLineText(self.window, "Note", "Note:")
        if ok:
            self.canvas.add_note(text)

    def act_find(self, *_):
        if self.window is not None:
            self.window.docks["library"].show()
        self.library.focus_search()

    def act_zoom_in(self, *_):
        self.canvas.zoom(+1)

    def act_zoom_out(self, *_):
        self.canvas.zoom(-1)

    def act_zoom_fit(self, *_):
        self.canvas.center_view()

    def act_auto_layout(self, *_):
        self.canvas.auto_layout(self.project_service.snapshot())

    def act_minimap(self, checked: bool | None = None, *_):
        """Show / hide the minimap (remembered)."""
        visible = (not self.canvas_area.minimap.isVisible()) if checked is None else bool(checked)
        self.canvas_area.set_minimap(visible)
        QtCore.QSettings(_SETTINGS).setValue("minimap", visible)
        if self.window is not None:
            self.window.actions["view.minimap"].setChecked(visible)

    def act_expand(self, *_):
        from ai_made_easy.core.data.lints import project_base

        ir = self._last_ir
        if ir is not None and any(n.type_id.startswith("pipeline.") for n in ir.nodes.values()) \
                and self.graph_service.toggle_inline(ir, project_base(self.project_store.path)):
            return               # a pipeline stage shows / hides its design inline
        self.graph_service.expand_selected()

    def act_save_selection(self, *_):
        dialog = SaveTemplateDialog(self.window)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or not dialog.template_name():
            return
        try:
            path = self.graph_service.save_selection_template(dialog.template_name())
        except Exception as exc:  # noqa: BLE001
            self.log_bus.error(str(exc))
            return
        self.library.rebuild()
        self.log_bus.info(f"saved custom block → {path}")
        self.status_message.emit(f"Saved custom block '{dialog.template_name()}'")

    def act_export_blocks(self, *_):
        from ai_made_easy.core.block_packs import PackError, export_pack, list_blocks

        if not list_blocks():
            self.status_message.emit("No custom blocks yet — use Model ▸ Save Selection as Block")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Export Custom Blocks", str(self._last_dir() / "blocks.aimeblocks"),
            "Custom block packs (*.aimeblocks)")
        if not path:
            return
        try:
            export_pack(path)
        except (PackError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self.window, "Export Custom Blocks", str(exc))
            return
        self._remember_dir(path)
        self.status_message.emit(f"Exported {len(list_blocks())} custom block(s)")

    def act_import_blocks(self, *_):
        from ai_made_easy.core.block_packs import PackError, import_pack
        from ai_made_easy.core.registry import get_registry
        from ai_made_easy.ui.canvas import templates as template_store
        from ai_made_easy.ui.canvas.node_factory import make_node_class

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window, "Import Custom Blocks", str(self._last_dir()),
            "Custom block packs (*.aimeblocks)")
        if not path:
            return
        try:
            result = import_pack(path)
        except (PackError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self.window, "Import Custom Blocks", str(exc))
            return
        template_store.register_user_templates(get_registry().register, make_node_class)
        self.library.rebuild()
        parts = [f"{len(result['imported'])} imported"]
        if result["skipped"]:
            parts.append(f"{len(result['skipped'])} already installed")
        if result["renamed"]:
            parts.append("renamed: " + ", ".join(f"{a} → {b}"
                                                 for a, b in result["renamed"].items()))
        self.status_message.emit("Custom blocks: " + "; ".join(parts))
