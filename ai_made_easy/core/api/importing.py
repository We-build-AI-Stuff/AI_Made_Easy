"""Importing models."""
from __future__ import annotations

from ai_made_easy.core.api import ApiError, manager


def import_model(kind: str, source: str, attr: str = "", input_shape: list[int] | None = None,
                 dtype: str = "float32", kwargs: dict | None = None, name: str = "") -> dict:
    """Import a PyTorch module, ONNX file or Keras model as an editable graph."""
    from ai_made_easy.core.importers import ModelImportError
    from ai_made_easy.core.importers import import_model as _import

    try:
        return _import(kind, source=source, attr=attr, input_shape=input_shape, dtype=dtype,
                       kwargs=kwargs, name=name, python=manager().python).to_dict()
    except ModelImportError as exc:
        raise ApiError(str(exc)) from exc
