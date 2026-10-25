"""Try a finished run (3.5)."""
from __future__ import annotations

from ai_made_easy.core.api import ApiError, manager, try_service


def try_describe(run_id: str) -> dict:
    """What a finished run takes, to build a form: the input kind and help, an example, and
    the signature (input shape, classes, table columns). Loads the run's predictor."""
    from ai_made_easy.core import tryit

    try:
        return tryit.describe(try_service(), manager().history, run_id)
    except tryit.TryError as exc:
        raise ApiError(str(exc)) from exc


def try_run(run_id: str, inputs: list, explain: bool = False) -> dict:
    """Predictions of a finished run for ``inputs`` (the same requests as its model server:
    rows, texts, base64 images, WAV files...), computed on this machine. ``explain`` adds a
    Grad-CAM heatmap to an image classifier's predictions."""
    from ai_made_easy.core import tryit

    try:
        return tryit.predict(try_service(), manager().history, run_id, inputs, explain=explain)
    except tryit.TryError as exc:
        raise ApiError(str(exc)) from exc


def what_would_change(run_id: str, row: dict, desired: str | None = None,
                      target: float | None = None, locked: list[str] | None = None) -> dict:
    """Counterfactuals of one table row for a finished run: the smallest single changes
    that give ``desired`` (another class; default the runner-up) or reach ``target`` (a
    regressor's value), else a few changes together. ``locked`` columns stay as they
    are. Asks the run's own predictor on this machine."""
    from ai_made_easy.core import counterfactual, tryit

    try:
        return counterfactual.what_would_change(try_service(), manager().history, run_id,
                                                row, desired=desired, target=target,
                                                locked=locked or ())
    except (counterfactual.CounterfactualError, tryit.TryError) as exc:
        raise ApiError(str(exc)) from exc


def try_file(run_id: str, filename: str, data: bytes) -> dict:
    """Predictions of a finished run for every input in a file (a CSV table, a text file,
    or a zip of images or WAV clips) as CSV text: ``{"csv", "rows", "filename"}``."""
    from ai_made_easy.core import tryit

    try:
        return tryit.predict_file(try_service(), manager().history, run_id, filename, data)
    except tryit.TryError as exc:
        raise ApiError(str(exc)) from exc
