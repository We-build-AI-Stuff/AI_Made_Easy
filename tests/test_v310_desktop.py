"""3.10 in the desktop app: the Framework choice next to Run on."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from conftest import tiny_classifier_dict  # noqa: E402
from test_v38_desktop import app, ctx  # noqa: E402,F401 — fixtures


def _until(app, condition, timeout: float = 10) -> None:  # noqa: ANN001, F811
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)


def test_the_framework_choice_is_saved_on_the_design(ctx, app):  # noqa: F811
    ctx.project_service.apply_graph_dict(tiny_classifier_dict())
    combo = ctx.training_page.framework
    _until(app, lambda: combo.count() == 3)
    assert [combo.itemData(i) for i in range(combo.count())] == ["auto", "pytorch", "keras"]
    assert combo.currentData() == "auto" and combo.itemText(0) == "Auto (PyTorch)"
    combo.setCurrentIndex(combo.findData("keras"))
    combo.activated.emit(combo.currentIndex())
    _until(app, lambda: ctx.graph_service.adapter.meta.get("framework") == "keras")
    from ai_made_easy.core.families import resolve_framework

    assert resolve_framework(ctx.graph_service.adapter.to_ir()) == "keras"   # what Train uses
    combo.setCurrentIndex(combo.findData("auto"))
    combo.activated.emit(combo.currentIndex())
    _until(app, lambda: "framework" not in ctx.graph_service.adapter.meta)
    assert "framework" not in ctx.graph_service.adapter.meta


def test_one_framework_hides_the_choice(app):  # noqa: F811
    from ai_made_easy.ui.features.runconsole import TrainingPage
    from ai_made_easy.ui.stores import RunStore

    page = TrainingPage(RunStore())
    page.set_frameworks(["pytorch"], "auto")
    assert page.framework.isHidden()
    page.set_frameworks(["pytorch"], "keras")          # chosen before a PyTorch-only block
    assert not page.framework.isHidden()
    assert "not every block has it" in page.framework.currentText()
    page.close()
