"""Desktop: the Run on selector, the Backends dialog and a run on an SSH host."""
from __future__ import annotations

import importlib.util
import time

import pytest

pytest.importorskip("PySide6")

from conftest import tiny_classifier_dict  # noqa: E402
from test_backends import fake_ssh  # noqa: E402, F401 — shared fixture
from test_ui import app, ctx  # noqa: E402, F401 — shared fixtures

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                                reason="needs torch")


def test_backends_dialog_saves_and_tests(isolated_home, app, fake_ssh):  # noqa: F811
    from ai_made_easy.core.runner import backends
    from ai_made_easy.ui.features.backends import BackendsDialog, RunOnCombo

    dialog = BackendsDialog()
    dialog.name_edit.setText("box")
    dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("ssh"))
    assert set(dialog.option_edits) == set(backends._backend_classes()["ssh"].options_help)
    dialog.option_edits["host"].setText("gpu-box")
    dialog.option_edits["retries"].setText("3")
    dialog._save()
    assert backends.resolve("box").options == {"host": "gpu-box", "retries": 3}
    names = [dialog.table.item(r, 0).text() for r in range(dialog.table.rowCount())]
    assert names[:2] == ["local", "box"]
    dialog.table.selectRow(1)
    dialog._test()
    assert dialog.last_test["ok"], dialog.last_test
    assert dialog.table.item(1, 3).text().startswith("✓ gpu-box: Python 3")

    combo = RunOnCombo()
    assert combo.findData("box") >= 0 and combo.itemData(combo.count() - 1) == "__manage"
    combo.setCurrentIndex(combo.findData("box"))
    assert combo.selected() == "box"


def test_train_on_an_ssh_host_from_the_training_panel(
        isolated_home, ctx, app, fake_ssh):  # noqa: F811
    from ai_made_easy.core.runner import backends

    backends.save("box", "ssh", host="gpu-box")
    combo = ctx.training_page.run_on
    combo.refresh()
    combo.setCurrentIndex(combo.findData("box"))
    assert ctx.open_imported(tiny_classifier_dict(epochs=2))
    deadline = time.time() + 30
    while not ctx.validation_store.valid and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    assert ctx.validation_store.valid, "the design never validated"
    finished: list = []
    epochs: list = []
    service = ctx.process_service
    service.finished.connect(lambda code, kind: finished.append((code, kind)))
    service.epoch_received.connect(epochs.append)
    ctx.act_train()
    assert service.is_running()
    deadline = time.time() + 180
    while not finished:
        assert time.time() < deadline, "the SSH run never finished"
        app.processEvents()
        time.sleep(0.05)
    assert finished == [(0, "train")]
    assert [e["epoch"] for e in epochs] == [1, 2]
    record = service.history.get(service.current_run_id)
    assert record.status == "finished" and record.backend == "box"
    assert record.final_metrics
    assert not service.is_running()
