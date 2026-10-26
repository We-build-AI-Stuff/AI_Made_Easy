"""Desktop: Train runs pipeline designs; Experiments ▸ Pipelines shows their stages."""
from __future__ import annotations

import importlib.util
import time

import pytest

pytest.importorskip("PySide6")

from test_pipelines import pipeline, stage  # noqa: E402
from test_ui import app, ctx  # noqa: E402, F401 — shared fixtures

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                                reason="needs torch")


def test_train_runs_a_pipeline(isolated_home, ctx, app):  # noqa: F811
    data = pipeline([stage("t", "pipeline.train", design="embedded:student", epochs=2),
                     stage("e", "pipeline.evaluate")], [("t", "e")])
    assert ctx.open_imported(data)
    app.processEvents()
    deadline = time.time() + 30
    while not ctx.validation_store.valid and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    assert ctx.validation_store.valid, "the design never validated"
    ctx.act_train()
    service = ctx.experiment_service
    assert service.pipeline is not None
    pipeline_id = service.pipeline.pipeline_id
    deadline = time.time() + 600
    while service.pipeline is not None:
        assert time.time() < deadline
        app.processEvents()
        time.sleep(0.05)
    record = service.pipelines.get(pipeline_id)
    assert record.state == "finished", {k: v["message"] for k, v in record.stages.items()}
    page = ctx.experiments_page
    ctx._refresh_experiments()
    page.show_pipeline(pipeline_id)
    assert page.tabs.tabText(page.tabs.currentIndex()) == "Pipelines"
    assert page.stages_table.rowCount() == 2
    assert page.stages_table.item(1, 1).text() == "finished"
    assert "ece" in page.stages_table.item(1, 2).text()


def test_expand_shows_a_stage_design_inline(isolated_home, ctx, app):  # noqa: F811
    from ai_made_easy.core import api

    assert ctx.open_imported(api.read_sample("pipeline_distill_quantize.json"))
    deadline = time.time() + 30
    while ctx._last_ir is None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    assert ctx._last_ir is not None, "the canvas never settled"
    adapter = ctx.graph_service.adapter
    node = adapter.nodes_of_type("pipeline.train")[0]      # the canvas has its own ids
    before = node.view.boundingRect().height()
    node.set_selected(True)
    ctx.act_expand()                                 # Ctrl+E on a stage: show its design
    rows = adapter.inline_rows(node.id)
    assert [r[0] for r in rows][:3] == ["Input", "Dense / Linear", "ReLU"]
    assert rows[0][2] == "[64]" and node.view._aime_inline_title.startswith("digits_teacher")
    assert node.view.boundingRect().height() > before + 13 * 15
    assert len(ctx._last_ir.nodes) == 5               # nothing was added to the pipeline
    ctx.act_expand()                                 # and hide it again
    assert adapter.inline_rows(node.id) == []
    assert node.view.boundingRect().height() == pytest.approx(before)
