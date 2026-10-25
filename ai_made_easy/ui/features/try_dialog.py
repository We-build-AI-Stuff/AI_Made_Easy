"""Try it (3.5): run a finished model on your own input, on this machine.

The form follows what the model takes: one field per column for tables, a text box, an
image or WAV file, or a JSON list for everything else. Loading and predicting run on a
worker thread, so the window stays responsive.

3.6: "Predict a file…" runs the model on every input of a file (a CSV, a text file or a zip
of images or clips) and saves the predictions as CSV; image classifiers can show their
Grad-CAM heatmap next to the prediction.

3.11: for a table row, "What would change it?" lists the smallest edits to the row that
make the model predict another class (or reach a value), with columns you can lock.
"""
from __future__ import annotations

import base64
import json
import threading
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

_FILE_KINDS = {"images": ("Images (*.png *.jpg *.jpeg *.bmp *.webp)", "an image"),
               "audio": ("WAV audio (*.wav)", "a WAV clip"),
               "speech": ("WAV audio (*.wav)", "a WAV clip")}
_TEXT_KINDS = ("texts", "prompt")


def _value(raw: str):  # noqa: ANN202
    try:
        return float(raw) if raw.strip() else raw
    except ValueError:
        return raw


def describe_prediction(pred: dict) -> str:
    """A prediction as plain text: the label and the top classes, else its JSON."""
    probs = pred.get("probabilities")
    if isinstance(probs, dict) and probs:
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
        lines = [f"{pred.get('label')}" + (f"  ({pred['confidence'] * 100:.1f}% sure)"
                                           if isinstance(pred.get("confidence"), float) else "")]
        lines += [f"  {name:<18} {'█' * max(1, round(p * 30))} {p * 100:.1f}%" for name, p in top]
        return "\n".join(lines)
    return json.dumps(pred, indent=1, default=str)


def describe_changes(found: dict) -> str:
    """Counterfactuals as plain text (both apps say the same)."""
    def one(entry: dict) -> str:
        p = entry["prediction"]
        edits = " and ".join(f"{c['column']} {_shown(c['from'])} → {_shown(c['to'])}"
                             for c in entry["changes"])
        label = p.get("label", _shown(p.get("prediction")))
        probs = p.get("probabilities") or {}
        sure = f" ({probs[str(label)] * 100:.0f}%)" if str(label) in probs else ""
        return f"  {edits} · predicts {label}{sure}"

    lines = [f"Goal: {found['goal']}."]
    if found["single"]:
        lines += ["One change is enough, smallest first:"] + [one(e) for e in found["single"]]
    elif found["combined"]:
        lines += ["No single change is enough; together these are:", one(found["combined"])]
    else:
        lines.append("No change of up to three columns reaches the goal with values from the "
                     "training table.")
    lines.append(f"{found['predictions_checked']} edited rows checked.")
    return "\n".join(lines)


def _shown(v) -> str:  # noqa: ANN001
    return f"{v:.4g}" if isinstance(v, float) else str(v)


class TryDialog(QtWidgets.QDialog):
    _loaded = QtCore.Signal(object, str)     # form | None, error
    _answered = QtCore.Signal(object, str)   # result | None, error
    _filed = QtCore.Signal(object, str)      # file result | None, error
    _changed = QtCore.Signal(object, str)    # counterfactuals | None, error

    def __init__(self, parent, history, run_id: str, service=None):  # noqa: ANN001
        super().__init__(parent)
        from ai_made_easy.core import api

        self.setWindowTitle(f"Try run {run_id}")
        self.resize(640, 560)
        self._history, self._run_id = history, run_id
        self._service = service or api.try_service()
        self.form_info: dict | None = None
        self._fields: dict[str, QtWidgets.QLineEdit] = {}
        self._file: Path | None = None

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel("The trained model answers here, on this machine; nothing is "
                                 "sent anywhere else.")
        intro.setObjectName("blockMeta")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.status = QtWidgets.QLabel("Loading the model…")
        layout.addWidget(self.status)
        self.form_box = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self.form_box)
        layout.addWidget(self.form_box)
        self.explain_box = QtWidgets.QCheckBox("Show where the model looked (Grad-CAM)")
        self.explain_box.setVisible(False)
        layout.addWidget(self.explain_box)
        results = QtWidgets.QHBoxLayout()
        self.result = QtWidgets.QPlainTextEdit(readOnly=True)
        self.result.setAccessibleName("Prediction")
        self.result.setPlaceholderText("The prediction appears here.")
        results.addWidget(self.result, 1)
        self.cam = QtWidgets.QLabel()
        self.cam.setAccessibleName("Grad-CAM heatmap")
        self.cam.setVisible(False)
        results.addWidget(self.cam)
        layout.addLayout(results, 1)
        self._build_changes(layout)
        buttons = QtWidgets.QDialogButtonBox()
        self.predict_btn = buttons.addButton("Predict",
                                             QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.predict_btn.setEnabled(False)
        self.file_btn = buttons.addButton("Predict a file…",
                                          QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        self.file_btn.setEnabled(False)
        self.file_btn.setToolTip("Every input of a file at once; the predictions are saved "
                                 "as CSV")
        self.file_btn.clicked.connect(self._choose_batch)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.predict_btn.clicked.connect(self.predict)
        layout.addWidget(buttons)

        self._loaded.connect(self._on_loaded)
        self._answered.connect(self._on_answered)
        self._filed.connect(self._on_filed)
        self._changed.connect(self._on_changed)
        threading.Thread(target=self._load, daemon=True).start()

    def _build_changes(self, layout: QtWidgets.QVBoxLayout) -> None:
        """What would change it (3.11): hidden until a table row has a prediction."""
        self.changes_box = QtWidgets.QGroupBox("What would change it?")
        box = QtWidgets.QFormLayout(self.changes_box)
        note = QtWidgets.QLabel("The model is asked about edited copies of this row, with "
                                "values your training table has. It shows what the model "
                                "does, not why.")
        note.setWordWrap(True)
        note.setObjectName("blockMeta")
        box.addRow(note)
        self.goal = QtWidgets.QComboBox()
        self.goal.setAccessibleName("Make it predict")
        box.addRow("Make it predict", self.goal)
        self.target = QtWidgets.QLineEdit()
        self.target.setAccessibleName("Reach the value")
        self.target.setPlaceholderText("a value for the prediction to reach")
        box.addRow("Reach the value", self.target)
        self.locks = QtWidgets.QListWidget()
        self.locks.setAccessibleName("Columns to keep as they are")
        self.locks.setMaximumHeight(90)
        box.addRow("Keep as they are", self.locks)
        self.change_btn = QtWidgets.QPushButton("Find Changes")
        self.change_btn.clicked.connect(self.find_changes)
        box.addRow(self.change_btn)
        self.changes = QtWidgets.QPlainTextEdit(readOnly=True)
        self.changes.setAccessibleName("Changes that change the prediction")
        self.changes.setMaximumHeight(120)
        box.addRow(self.changes)
        self.changes_box.setVisible(False)
        layout.addWidget(self.changes_box)
        self._row: dict | None = None

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        from ai_made_easy.core import tryit

        try:
            self._loaded.emit(tryit.describe(self._service, self._history, self._run_id), "")
        except Exception as exc:  # noqa: BLE001 — shown in the dialog
            self._loaded.emit(None, str(exc))

    def _on_loaded(self, info: dict | None, error: str) -> None:
        if info is None:
            self.status.setText(f"Could not load the model: {error}")
            return
        self.form_info = info
        kind = info["input_kind"]
        self.status.setText(f"Input: {info.get('input_help') or kind}")
        columns = (info.get("signature") or {}).get("columns") or []
        if kind == "records" and columns:
            for col in columns:
                field = QtWidgets.QLineEdit()
                field.setAccessibleName(col)
                self._fields[col] = field
                self.form.addRow(col, field)
        elif kind in _TEXT_KINDS:
            self.text = QtWidgets.QPlainTextEdit()
            self.text.setAccessibleName("Text")
            self.text.setPlaceholderText("Start a sentence…" if kind == "prompt"
                                         else "Write a text to classify…")
            self.form.addRow("Text", self.text)
        elif kind in _FILE_KINDS:
            pick = QtWidgets.QPushButton(f"Choose {_FILE_KINDS[kind][1]}…")
            self.file_label = QtWidgets.QLabel("No file chosen")
            pick.clicked.connect(self._choose_file)
            self.form.addRow(pick, self.file_label)
        else:
            self.json = QtWidgets.QPlainTextEdit(json.dumps(info.get("example") or [], indent=1))
            self.json.setAccessibleName("Inputs as a JSON list")
            self.form.addRow("Inputs (JSON list)", self.json)
        self.predict_btn.setEnabled(True)
        self.explain_box.setVisible(bool(info.get("can_explain")))
        self.file_btn.setEnabled(bool(info.get("batch_formats")))
        if info.get("batch_formats"):
            self.file_btn.setToolTip(f"Every input of a {info['batch_formats']} file at once; "
                                     "the predictions are saved as CSV")

    def _choose_file(self) -> None:
        kind = (self.form_info or {}).get("input_kind", "")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose a file", "",
                                                        _FILE_KINDS[kind][0])
        if path:
            self.set_file(Path(path))

    def set_file(self, path: Path) -> None:
        self._file = path
        if hasattr(self, "file_label"):
            self.file_label.setText(path.name)

    # ------------------------------------------------------------------ predicting
    def inputs(self) -> list:
        kind = (self.form_info or {}).get("input_kind", "")
        if self._fields:
            return [{c: _value(f.text()) for c, f in self._fields.items()}]
        if kind in _TEXT_KINDS:
            return [self.text.toPlainText()]
        if kind in _FILE_KINDS:
            if self._file is None:
                raise ValueError(f"choose {_FILE_KINDS[kind][1]} first")
            return [base64.b64encode(self._file.read_bytes()).decode()]
        got = json.loads(self.json.toPlainText() or "[]")
        return got if isinstance(got, list) else [got]

    def predict(self) -> None:
        try:
            inputs = self.inputs()
        except (ValueError, OSError) as exc:
            self.result.setPlainText(f"⚠ {exc}")
            return
        self.predict_btn.setEnabled(False)
        self.result.setPlainText("Predicting…")
        explain = not self.explain_box.isHidden() and self.explain_box.isChecked()
        threading.Thread(target=self._ask, args=(inputs, explain), daemon=True).start()

    def _ask(self, inputs: list, explain: bool = False) -> None:
        from ai_made_easy.core import tryit

        try:
            self._answered.emit(tryit.predict(self._service, self._history, self._run_id,
                                              inputs, explain=explain), "")
        except Exception as exc:  # noqa: BLE001 — shown in the dialog
            self._answered.emit(None, str(exc))

    def _on_answered(self, result: dict | None, error: str) -> None:
        self.predict_btn.setEnabled(True)
        if result is None:
            self.result.setPlainText(f"⚠ {error}")
            self.cam.setVisible(False)
            return
        cams = [p.pop("gradcam") for p in result["predictions"]
                if isinstance(p, dict) and "gradcam" in p]
        text = "\n\n".join(describe_prediction(p) for p in result["predictions"])
        self.result.setPlainText(f"{text}\n\n{result['latency_ms']} ms")
        self._offer_changes(result["predictions"][0] if result["predictions"] else None)
        self.cam.setVisible(bool(cams))
        if cams:
            image = QtGui.QImage.fromData(base64.b64decode(cams[0]))
            self.cam.setPixmap(QtGui.QPixmap.fromImage(image).scaled(
                224, 224, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))

    # ------------------------------------------------------------------ what would change it
    def _offer_changes(self, pred) -> None:  # noqa: ANN001
        probs = pred.get("probabilities") if isinstance(pred, dict) else None
        numeric = isinstance(pred, dict) and isinstance(pred.get("prediction"), (int, float))
        if not self._fields or not (isinstance(probs, dict) or numeric):
            self.changes_box.setVisible(False)
            return
        self._row = self.inputs()[0]
        self.goal.clear()
        if isinstance(probs, dict):
            others = sorted((c for c in probs if c != str(pred.get("label"))),
                            key=lambda c: -probs[c])
            for c in others:
                self.goal.addItem(c + ("  (the runner-up)" if c == others[0] else ""), c)
        self.goal.setVisible(isinstance(probs, dict))
        self.changes_box.layout().labelForField(self.goal).setVisible(isinstance(probs, dict))
        self.target.setVisible(numeric)
        self.changes_box.layout().labelForField(self.target).setVisible(numeric)
        if self.locks.count() != len(self._fields):
            self.locks.clear()
            for col in self._fields:
                item = QtWidgets.QListWidgetItem(col)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)
                self.locks.addItem(item)
        self.changes.clear()
        self.changes_box.setVisible(True)

    def find_changes(self) -> None:
        locked = [self.locks.item(i).text() for i in range(self.locks.count())
                  if self.locks.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        target = None
        if self.target.isVisible():
            try:
                target = float(self.target.text())
            except ValueError:
                self.changes.setPlainText("⚠ give a number for the prediction to reach")
                return
        desired = self.goal.currentData() if self.goal.isVisible() else None
        self.change_btn.setEnabled(False)
        self.changes.setPlainText("Searching…")
        threading.Thread(target=self._ask_changes, args=(self._row, desired, target, locked),
                         daemon=True).start()

    def _ask_changes(self, row: dict, desired, target, locked: list) -> None:  # noqa: ANN001
        from ai_made_easy.core import counterfactual

        try:
            self._changed.emit(counterfactual.what_would_change(
                self._service, self._history, self._run_id, row, desired=desired,
                target=target, locked=locked), "")
        except Exception as exc:  # noqa: BLE001 — shown in the dialog
            self._changed.emit(None, str(exc))

    def _on_changed(self, found: dict | None, error: str) -> None:
        self.change_btn.setEnabled(True)
        if found is None:
            self.changes.setPlainText(f"⚠ {error}")
            return
        self.changes.setPlainText(describe_changes(found))

    # ------------------------------------------------------------------ a whole file
    def _choose_batch(self) -> None:
        formats = (self.form_info or {}).get("batch_formats", "")
        pattern = " ".join(f"*{ext}" for ext in formats.split(","))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Predict a file", "",
                                                        f"Inputs ({pattern})")
        if not path:
            return
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save the predictions", str(Path(path).with_name(
                f"{Path(path).stem}_predictions.csv")), "CSV (*.csv)")
        if out:
            self.predict_file(Path(path), Path(out))

    def predict_file(self, source: Path, target: Path) -> None:
        """Predict every input in ``source`` and write the CSV to ``target`` (in the
        background; the result box says when it is done)."""
        self.file_btn.setEnabled(False)
        self.result.setPlainText(f"Predicting every input in {source.name}…")
        threading.Thread(target=self._ask_file, args=(source, target), daemon=True).start()

    def _ask_file(self, source: Path, target: Path) -> None:
        from ai_made_easy.core import tryit

        try:
            got = tryit.predict_file(self._service, self._history, self._run_id, source.name,
                                     source.read_bytes())
            target.write_text(got["csv"], encoding="utf-8")
            self._filed.emit({**got, "target": str(target)}, "")
        except Exception as exc:  # noqa: BLE001 — shown in the dialog
            self._filed.emit(None, str(exc))

    def _on_filed(self, result: dict | None, error: str) -> None:
        self.file_btn.setEnabled(True)
        if result is None:
            self.result.setPlainText(f"⚠ {error}")
            return
        self.result.setPlainText(f"{result['rows']} predictions saved to {result['target']} "
                                 f"({result['latency_ms'] / 1000:.1f} s)")
