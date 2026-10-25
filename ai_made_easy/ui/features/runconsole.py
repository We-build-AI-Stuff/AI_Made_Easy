"""Bottom panels: Output log and Training monitor.

TrainingPage renders run state (single writer: RunStore), live epoch metrics
(tiles + curves) and the post-run tools: error analysis, saliency maps, model
card and the run folder.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ai_made_easy.ui import icons
from ai_made_easy.ui.theme import SERIES
from ai_made_easy.ui.stores import LogBus, RunStore

_SERIES_COLORS = SERIES


class OutputPage(QtWidgets.QPlainTextEdit):
    """The only renderer of the LogBus."""

    def __init__(self, log_bus: LogBus, parent=None):
        super().__init__(parent)
        self.setObjectName("logView")
        self.setReadOnly(True)
        self.setMaximumBlockCount(10000)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._formats = {}
        for level, color in (("error", "#f85149"), ("warning", "#e3a008")):
            fmt = QtGui.QTextCharFormat()
            fmt.setForeground(QtGui.QColor(color))
            self._formats[level] = fmt
        log_bus.logged.connect(self._append)

    def _append(self, level: str, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        prefix = {"error": "error  ", "warning": "warning"}.get(level, "info   ")
        cursor.insertText(f"{stamp}  {prefix}  ", QtGui.QTextCharFormat())
        cursor.insertText(message + "\n", self._formats.get(level, QtGui.QTextCharFormat()))
        self.setTextCursor(cursor)
        self.ensureCursorVisible()


class _Tile(QtWidgets.QFrame):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.setObjectName("metricTile")
        box = QtWidgets.QVBoxLayout(self)
        box.setContentsMargins(10, 6, 10, 6)
        box.setSpacing(0)
        self.value = QtWidgets.QLabel("—")
        self.value.setObjectName("metricValue")
        label = QtWidgets.QLabel(name)
        label.setObjectName("metricName")
        box.addWidget(self.value)
        box.addWidget(label)


class TrainingPage(QtWidgets.QWidget):
    train_clicked = QtCore.Signal()
    stop_clicked = QtCore.Signal()
    errors_clicked = QtCore.Signal()
    saliency_clicked = QtCore.Signal()
    card_clicked = QtCore.Signal()
    folder_clicked = QtCore.Signal()

    def __init__(self, run_store: RunStore, parent=None):
        super().__init__(parent)
        self.setObjectName("dockBody")
        self._run_store = run_store
        self._epoch_x: list[float] = []
        self._series: dict[str, list[float]] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._tiles: dict[str, _Tile] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("No run yet")
        self.status.setObjectName("blockMeta")
        top.addWidget(self.status)
        top.addStretch(1)
        self.start_btn = QtWidgets.QPushButton("Train")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.setIcon(icons.icon("play", color="#ffffff"))
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setIcon(icons.icon("stop"))
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.train_clicked.emit)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        from ai_made_easy.ui.features.backends import RunOnCombo

        self.run_on = RunOnCombo()
        self.framework = QtWidgets.QComboBox()   # 3.10: saved on the design (meta)
        self.framework.setToolTip("The framework this design trains with (saved with it)")
        self.framework.setAccessibleName("Framework")
        self.framework.hide()
        top.addWidget(self.framework)
        top.addWidget(self.run_on)
        top.addWidget(self.start_btn)
        top.addWidget(self.stop_btn)
        layout.addLayout(top)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.hide()
        layout.addWidget(self.progress)

        self.tile_row = QtWidgets.QHBoxLayout()
        self.tile_row.setSpacing(8)
        self.tile_row.addStretch(1)
        layout.addLayout(self.tile_row)

        pg.setConfigOptions(antialias=True)
        self.plots = pg.GraphicsLayoutWidget()
        self.loss_plot = self.plots.addPlot(row=0, col=0, title="Loss")
        self.metric_plot = self.plots.addPlot(row=0, col=1, title="Validation metrics")
        for plot in (self.loss_plot, self.metric_plot):
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.addLegend(offset=(8, 8))
            plot.setLabel("bottom", "epoch")
        layout.addWidget(self.plots, 1)

        self.samples_box = QtWidgets.QGroupBox("Samples")
        samples_layout = QtWidgets.QHBoxLayout(self.samples_box)
        self.samples_image = QtWidgets.QLabel()
        self.samples_image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.samples_text = QtWidgets.QPlainTextEdit()
        self.samples_text.setReadOnly(True)
        samples_layout.addWidget(self.samples_image)
        samples_layout.addWidget(self.samples_text)
        self.samples_box.setMaximumHeight(280)
        self.samples_box.hide()
        layout.addWidget(self.samples_box)

        tools = QtWidgets.QHBoxLayout()
        self.errors_btn = QtWidgets.QPushButton("Error Analysis")
        self.errors_btn.setIcon(icons.icon("list"))
        self.saliency_btn = QtWidgets.QPushButton("Saliency Maps")
        self.saliency_btn.setIcon(icons.icon("eye"))
        self.card_btn = QtWidgets.QPushButton("Model Card")
        self.card_btn.setIcon(icons.icon("card"))
        self.folder_btn = QtWidgets.QPushButton("Open Run Folder")
        self.folder_btn.setIcon(icons.icon("folder"))
        for btn, sig in ((self.errors_btn, self.errors_clicked),
                         (self.saliency_btn, self.saliency_clicked),
                         (self.card_btn, self.card_clicked),
                         (self.folder_btn, self.folder_clicked)):
            btn.setEnabled(False)
            btn.clicked.connect(sig.emit)
            tools.addWidget(btn)
        tools.addStretch(1)
        layout.addLayout(tools)
        run_store.state_changed.connect(self._on_state)

    def set_theme(self, tokens: dict) -> None:
        self.plots.setBackground(tokens["PANEL"])
        for plot in (self.loss_plot, self.metric_plot):
            for axis in ("left", "bottom"):
                plot.getAxis(axis).setPen(tokens["BORDER"])
                plot.getAxis(axis).setTextPen(tokens["TEXT_DIM"])
            plot.setTitle(plot.titleLabel.text, color=tokens["TEXT_DIM"], size="9pt")

    # ------------------------------------------------------------- state
    def _on_state(self, state: str, kind: str) -> None:
        running = state == RunStore.RUNNING
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running and kind == "train")
        label = {"train": "Training", "test": "Forward-pass test", "inspect": "Saliency",
                 "onnx": "ONNX export", "jit": "TorchScript export"}.get(kind, kind or "Run")
        if running:
            self.progress.setRange(0, 0)
            self.progress.show()
            self.status.setText(f"{label} running…")
        elif state == RunStore.FINISHED:
            self.progress.hide()
            epochs = len(self._epoch_x)
            self.status.setText(f"{label} finished" + (f" after {epochs} epoch(s)"
                                                       if kind == "train" and epochs else ""))
        elif state == RunStore.FAILED:
            self.progress.hide()
            self.status.setText(f"{label} failed — see Output")
        elif state == RunStore.STOPPED:
            self.progress.hide()
            self.status.setText(f"{label} stopped")

    def set_results_available(self, workdir) -> None:  # noqa: ANN001
        wd = Path(workdir) if workdir else None
        has_predictions = bool(wd and (wd / "predictions.json").exists())
        has_checkpoint = bool(wd and any(wd.glob("*_best.pt")))
        has_overlays = bool(wd and any((wd / "eval_samples").glob("*.png")))
        self.errors_btn.setEnabled(has_predictions or has_overlays)
        self.saliency_btn.setEnabled(has_predictions and has_checkpoint)
        self.card_btn.setEnabled(bool(wd and (wd / "metrics.json").exists()) or has_predictions)
        self.folder_btn.setEnabled(bool(wd and wd.exists()))

    # -------------------------------------------------------------- data
    def last_metrics(self) -> dict[str, float]:
        return {k: v[-1] for k, v in self._series.items() if v}

    def on_samples(self, event: dict) -> None:
        """Show the newest sample grid (image) or generated text of a generative run."""
        import base64

        epoch = event.get("epoch")
        self.samples_box.setTitle("Samples" + (f" — epoch {epoch}" if epoch else ""))
        url = str(event.get("data_url") or "")
        if url.startswith("data:image/png;base64,"):
            pixmap = QtGui.QPixmap()
            pixmap.loadFromData(base64.b64decode(url.split(",", 1)[1]), "PNG")
            self.samples_image.setPixmap(pixmap.scaled(
                520, 250, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.FastTransformation))
            self.samples_image.show()
            self.samples_text.hide()
        else:
            self.samples_text.setPlainText(str(event.get("text", "")))
            self.samples_text.show()
            self.samples_image.hide()
        self.samples_box.show()

    FRAMEWORK_LABELS = {"pytorch": "PyTorch", "keras": "Keras"}

    def set_frameworks(self, usable: list[str], chosen: str) -> None:
        """Offer the frameworks every block of the design has (hidden with one)."""
        blocker = QtCore.QSignalBlocker(self.framework)
        self.framework.clear()
        names = self.FRAMEWORK_LABELS
        self.framework.addItem(f"Auto ({names.get(usable[0], 'PyTorch') if usable else 'PyTorch'})",
                               "auto")
        for fw in usable:
            self.framework.addItem(names.get(fw, fw), fw)
        if chosen != "auto" and chosen not in usable:
            self.framework.addItem(f"{names.get(chosen, chosen)} (not every block has it)",
                                   chosen)
        self.framework.setCurrentIndex(max(self.framework.findData(chosen), 0))
        del blocker
        self.framework.setVisible(len(usable) > 1 or chosen != "auto")

    def reset(self) -> None:
        self.samples_box.hide()
        self._epoch_x.clear()
        self._series.clear()
        for curve in self._curves.values():
            curve.setData([], [])
        self._curves.clear()
        self.loss_plot.clear()
        self.metric_plot.clear()
        for plot in (self.loss_plot, self.metric_plot):
            plot.addLegend(offset=(8, 8))
        for tile in self._tiles.values():
            tile.deleteLater()
        self._tiles.clear()
        self.set_results_available(None)

    def on_epoch(self, event: dict) -> None:
        epoch = int(event.get("epoch", len(self._epoch_x) + 1))
        total = int(event.get("total") or epoch)
        self._epoch_x.append(float(epoch))
        for key, value in event.get("metrics", {}).items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if key == "lr":
                continue
            self._series.setdefault(key, []).append(value)
            series = self._series[key]
            xs = self._epoch_x[-len(series):]
            curve = self._curves.get(key)
            if curve is None:
                plot = self.loss_plot if "loss" in key else self.metric_plot
                color = _SERIES_COLORS[len(self._curves) % len(_SERIES_COLORS)]
                curve = plot.plot(pen=pg.mkPen(color, width=2), name=key, symbol="o",
                                  symbolSize=4, symbolBrush=color, symbolPen=None)
                self._curves[key] = curve
            curve.setData(xs, series)
            self._tile(key).value.setText(f"{value:.4g}")
        self.progress.setRange(0, total)
        self.progress.setValue(epoch)
        self.status.setText(f"Training — epoch {epoch} of {total}")

    def _tile(self, key: str) -> _Tile:
        if key not in self._tiles:
            tile = _Tile(key.replace("_", " "))
            self.tile_row.insertWidget(len(self._tiles), tile)
            self._tiles[key] = tile
        return self._tiles[key]
