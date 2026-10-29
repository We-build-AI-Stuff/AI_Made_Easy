"""3.8 dialogs: find a learning rate, train a finished run more, and its results by group.

Each runs on the experiment service's headless RunManager and polls the run with a timer,
so the window stays responsive.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ai_made_easy.ui.theme import SERIES

FINAL = ("finished", "failed", "stopped", "interrupted")


def _sci(value: float) -> str:
    return f"{value:.2g}"


class LRFinderDialog(QtWidgets.QDialog):
    """Runs the learning-rate range test on the design and plots loss against the rate;
    Use applies the suggestion (``suggested``) to the optimizer block."""

    def __init__(self, parent, manager, graph) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle("Find a Learning Rate")
        self.resize(640, 460)
        self.manager, self.graph = manager, graph
        self.run_id: str | None = None
        self.suggested: float | None = None
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "A short run on this computer trains the design while the learning rate rises "
            "from 1e-7 batch by batch, and stops when the loss blows up. The suggestion is "
            "where the loss falls fastest. Nothing is saved.")
        intro.setWordWrap(True)
        intro.setObjectName("blockDesc")
        layout.addWidget(intro)
        import pyqtgraph as pg

        self.plot = pg.PlotWidget()
        self.plot.setLogMode(x=True, y=False)
        self.plot.setLabel("bottom", "learning rate")
        self.plot.setLabel("left", "smoothed loss")
        self.plot.setAccessibleName("Loss against learning rate")
        self.curve = self.plot.plot([], [], pen=pg.mkPen(SERIES[0], width=2))
        self.marker = pg.InfiniteLine(angle=90, movable=False,
                                      pen=pg.mkPen(SERIES[1],
                                                     style=QtCore.Qt.PenStyle.DashLine))
        self.marker.hide()
        self.plot.addItem(self.marker)
        layout.addWidget(self.plot, 1)
        self.status = QtWidgets.QLabel("Starting…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QtWidgets.QDialogButtonBox()
        self.use_btn = buttons.addButton("Use", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.use_btn.setEnabled(False)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.timer = QtCore.QTimer(self, interval=500, timeout=self._poll)
        QtCore.QTimer.singleShot(0, self._start)

    def _start(self) -> None:
        from ai_made_easy.core import lr_find

        try:
            self.run_id = lr_find.start(self.manager, self.graph)
        except (lr_find.LRFindError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        self.status.setText("Testing learning rates…")
        self.timer.start()

    def _poll(self) -> None:
        from ai_made_easy.core import lr_find

        result = lr_find.result(self.manager, self.run_id)
        lrs, losses = result.get("lrs") or [], result.get("losses") or []
        if lrs:
            self.curve.setData(lrs, losses)
        if result.get("state") not in FINAL:
            return
        self.timer.stop()
        if result["state"] != "finished":
            self.status.setText(f"The test {result['state']}: {result.get('error', '')}")
            return
        self.suggested = result.get("suggested")
        if self.suggested:
            import math

            self.marker.setValue(math.log10(self.suggested))   # log-mode x axis
            self.marker.show()
            self.use_btn.setText(f"Use {_sci(self.suggested)}")
            self.use_btn.setEnabled(True)
            self.status.setText(
                f"Suggested {_sci(self.suggested)} ({result.get('reason')}); the loss was "
                f"lowest at {_sci(result.get('min_loss_lr') or 0)}.")
        else:
            self.status.setText(f"No suggestion: {result.get('reason')}.")

    def reject(self) -> None:
        self.timer.stop()
        if self.run_id and self.manager.status(self.run_id).get("state") not in FINAL:
            self.manager.stop(self.run_id)
        super().reject()


class TrainMoreDialog(QtWidgets.QDialog):
    """How many more epochs and the learning-rate scale; for a boosted model (``trees``),
    how many more trees."""

    def __init__(self, parent, run_id: str, trees: bool = False) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle(f"Train {run_id} More")
        self.trees = trees
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "A new run keeps this model's trees and adds more, fitted on the same training "
            "rows with the same class balancing. Past some point more trees only fit the "
            "training data closer: watch the test score." if trees else
            "A new run starts from this run's best weights and trains for more epochs on "
            "the same data. The optimizer starts afresh, so a slightly lower learning rate "
            "is often smoother.")
        note.setWordWrap(True)
        note.setObjectName("blockDesc")
        layout.addWidget(note)
        form = QtWidgets.QFormLayout()
        self.epochs = QtWidgets.QSpinBox()
        self.epochs.setRange(1, 100_000 if trees else 10_000)
        self.epochs.setValue(100 if trees else 10)
        self.scale = QtWidgets.QDoubleSpinBox()
        self.scale.setRange(0.01, 10.0)
        self.scale.setSingleStep(0.1)
        self.scale.setValue(1.0)
        self.scale.setToolTip("Multiplies the design's learning rate (1 = the same)")
        form.addRow("More trees" if trees else "More epochs", self.epochs)
        if not trees:
            form.addRow("Learning-rate scale", self.scale)
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox()
        buttons.addButton("Train More", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class GroupsDialog(QtWidgets.QDialog):
    """Pick a column; the run's test rows are scored per value and weak groups flagged."""

    def __init__(self, parent, manager, run_id: str) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle(f"Results of {run_id} by Group")
        self.resize(620, 440)
        self.manager, self.run_id = manager, run_id
        self.scoring: str | None = None
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "The run's test rows are scored for each value of a column (numbers with many "
            "values are split into quartiles). Groups clearly worse than the whole test set "
            "are flagged.")
        note.setWordWrap(True)
        note.setObjectName("blockDesc")
        layout.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Group by"))
        self.column = QtWidgets.QComboBox()
        self.column.setAccessibleName("Column to group by")
        row.addWidget(self.column, 1)
        self.score_btn = QtWidgets.QPushButton("Score by Group")
        row.addWidget(self.score_btn)
        layout.addLayout(row)
        self.summary = QtWidgets.QLabel("")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Group", "Rows", "Metric", "vs all"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAccessibleName("Results by group")
        layout.addWidget(self.table, 1)
        self.fairness = QtWidgets.QLabel("")
        self.fairness.setWordWrap(True)
        self.fairness.setAccessibleName("Fairness between groups")
        layout.addWidget(self.fairness)
        self.thresholds_note = QtWidgets.QLabel("")
        self.thresholds_note.setWordWrap(True)
        layout.addWidget(self.thresholds_note)
        self.thresholds = QtWidgets.QTableWidget(0, 5)
        self.thresholds.setHorizontalHeaderLabels(
            ["Group", "Threshold", "True positives", "False positives", "Accuracy"])
        self.thresholds.horizontalHeader().setStretchLastSection(True)
        self.thresholds.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.thresholds.setAccessibleName("A threshold per group, before and after")
        self.thresholds.hide()
        layout.addWidget(self.thresholds)
        use_row = QtWidgets.QHBoxLayout()
        self.in_use = QtWidgets.QLabel("")
        self.in_use.setAccessibleName("Thresholds per group in use")
        use_row.addWidget(self.in_use, 1)
        self.use_btn = QtWidgets.QPushButton("Use These Thresholds")
        self.use_btn.setToolTip("Try it and the run's deployment packages decide each row's "
                                "class with its group's threshold")
        self.stop_btn = QtWidgets.QPushButton("Stop Using Them")
        for b in (self.use_btn, self.stop_btn):
            b.hide()
            use_row.addWidget(b)
        layout.addLayout(use_row)
        self.use_btn.clicked.connect(self._use)
        self.stop_btn.clicked.connect(self._stop)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.timer = QtCore.QTimer(self, interval=700, timeout=self._poll)
        self.score_btn.clicked.connect(self._score)
        self._load_columns()

    def _load_columns(self) -> None:
        from ai_made_easy.core import groups

        try:
            cols = [c for c in groups.columns(self.manager, self.run_id)["columns"]
                    if c["usable"]]
        except (groups.GroupsError, KeyError, FileNotFoundError) as exc:
            self.summary.setText(str(exc))
            self.score_btn.setEnabled(False)
            return
        for c in sorted(cols, key=lambda c: c["numeric"]):
            how = "quartiles" if c["how"] == "quartiles" else f"{c['distinct']} values"
            self.column.addItem(f"{c['name']} ({how})", c["name"])

    def _score(self) -> None:
        from ai_made_easy.core import groups

        try:
            self.scoring = groups.start(self.manager, self.run_id, self.column.currentData())
        except (groups.GroupsError, KeyError) as exc:
            self.summary.setText(str(exc))
            return
        self.score_btn.setEnabled(False)
        self.summary.setText("Scoring the test rows…")
        self.timer.start()

    def _poll(self) -> None:
        from ai_made_easy.core import groups

        result = groups.result(self.manager, self.scoring)
        if result.get("state") not in FINAL:
            return
        self.timer.stop()
        self.score_btn.setEnabled(True)
        if result["state"] != "finished" or "groups" not in result:
            self.summary.setText(f"Scoring {result['state']}: {result.get('error', '')}")
            return
        self.show_result(result)

    def show_result(self, result: dict) -> None:
        key = result.get("metric") or ""
        flagged = result.get("flagged") or []
        self.summary.setText(
            f"Worse for {', '.join(flagged)}: at least 10% "
            f"{'lower' if result.get('higher_is_better') else 'higher'} {key} than the whole "
            "test set." if flagged else f"No group is clearly worse than the whole test set "
                                        f"on {key}.")
        fair = result.get("fairness")
        self.fairness.setText(f"Fairness: {fair['summary']}" if fair else "")
        self._show_thresholds(result.get("mitigation"), result.get("column", "Group"),
                              binned=result.get("how") == "quartiles")
        self.table.setHorizontalHeaderLabels([result.get("column", "Group"), "Rows", key,
                                              "vs all"])
        rows = [{"value": "all test rows", **result.get("overall", {})}, *result["groups"]]
        self.table.setRowCount(len(rows))
        for i, g in enumerate(rows):
            gap = g.get("gap")
            cells = [g["value"], str(g.get("count", "")),
                     f"{g[key]:.4g}" if isinstance(g.get(key), (int, float)) else "—",
                     "" if gap is None else f"{'−' if gap > 0 else '+'}{abs(gap) * 100:.0f}%"
                     + ("  ⚠ worse" if g.get("flagged") else "")]
            for j, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if g.get("flagged"):
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(i, j, item)

    def _show_thresholds(self, found: dict | None, column: str, binned: bool = False) -> None:
        """A threshold per group: each group's rates at 0.5, then with its threshold."""
        self.thresholds.setVisible(bool(found))
        if not found:
            self.thresholds_note.setText("")
            self._show_use(False)
            return
        self.thresholds_note.setText(
            f"A threshold per group: {found.get('summary', '')}\nBefore → after, on "
            f"{found['before']['overall']['count']} rows the thresholds were not fitted on. "
            f"Using them, Try it and the run's packages decide each row with its {column}'s "
            "threshold (requests then include the column)." + (
                "\nThese groups are quartiles of a number, so a row can't be matched to "
                "them: score by a column of categories to use thresholds." if binned else ""))
        self._show_use(not binned)
        self.thresholds.setHorizontalHeaderLabels(
            [column, "Threshold", "True positives", "False positives", "Accuracy"])

        def pct(v) -> str:  # noqa: ANN001
            return "—" if v is None else f"{v:.0%}"

        after = {g["value"]: g for g in found["after"]["groups"]}
        self.thresholds.setRowCount(len(found["before"]["groups"]))
        for i, g in enumerate(found["before"]["groups"]):
            a = after.get(g["value"], {"rates": {"tpr": [None], "fpr": [None]},
                                       "accuracy": None})
            cells = [g["value"], f"{found['thresholds'].get(g['value'], 0.5):.2f}",
                     f"{pct(g['rates']['tpr'][0])} → {pct(a['rates']['tpr'][0])}",
                     f"{pct(g['rates']['fpr'][0])} → {pct(a['rates']['fpr'][0])}",
                     f"{pct(g['accuracy'])} → {pct(a['accuracy'])}"]
            for j, text in enumerate(cells):
                self.thresholds.setItem(i, j, QtWidgets.QTableWidgetItem(text))

    def _show_use(self, offered: bool) -> None:
        from ai_made_easy.core import groups

        used = groups.adopted(self.manager, self.run_id) if offered else None
        self.use_btn.setVisible(offered and used is None)
        self.stop_btn.setVisible(used is not None)
        self.in_use.setText(f"In use, by {used['column']}." if used else "")

    def _use(self) -> None:
        from ai_made_easy.core import groups

        try:
            groups.adopt(self.manager, self.scoring)
        except (groups.GroupsError, KeyError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "Use These Thresholds", str(exc))
            return
        self._reload_try()
        self._show_use(True)

    def _reload_try(self) -> None:
        """Try it loads the run again, with its thresholds (or without)."""
        from ai_made_easy.core import api

        api.try_service().forget(self.run_id)

    def _stop(self) -> None:
        from ai_made_easy.core import groups

        groups.drop(self.manager, self.run_id)
        self._reload_try()
        self._show_use(True)

    def reject(self) -> None:
        self.timer.stop()
        super().reject()
