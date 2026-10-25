"""The data workspace."""
from __future__ import annotations

from pathlib import Path

from ai_made_easy.core.graph import Graph
from ai_made_easy.core.api import ApiError, _graph, manager, profile_cache


def _dataset_node(graph: Graph, node_id: str | None):
    from ai_made_easy.core.data.lints import dataset_nodes

    nodes = dataset_nodes(graph)
    if node_id:
        nodes = [n for n in nodes if n[0] == node_id]
    if not nodes:
        raise ApiError(f"no dataset block {node_id!r}" if node_id else
                       "the design has no dataset block")
    return nodes[0]


def profile_data(graph: dict | Graph | None = None, node_id: str | None = None, *,
                 path: str | None = None, target: str | None = None,
                 base: str | None = None) -> dict:
    """Profile the dataset behind a dataset block, or any table / folder ``path``."""
    from ai_made_easy.core.data.profile import profile_path

    if path:
        if not Path(path).expanduser().exists():
            raise ApiError(f"{path} does not exist")
        return profile_path(path, target=target).to_dict()
    if graph is None:
        raise ApiError("pass a graph or a path")
    _nid, type_id, params = _dataset_node(_graph(graph), node_id)
    return profile_cache().profile(type_id, params, base).to_dict()


def data_issues(graph: dict | Graph, base: str | None = None) -> dict:
    """Data warnings (missing values, leakage, imbalance, ...) for the design's datasets."""
    from ai_made_easy.core.data.lints import data_issues as _issues

    issues = _issues(_graph(graph), profile_cache(), base)
    return {"issues": [{"severity": i.severity, "message": i.message, "node_id": i.node_id}
                       for i in issues]}


def split_preview(graph: dict | Graph, base: str | None = None) -> dict:
    """Samples per class in train / validation / test, as the training script splits."""
    from ai_made_easy.core.data.lints import task_of
    from ai_made_easy.core.data.splits import split_preview as _preview
    from ai_made_easy.core.training import data_catalog as dcat

    g = _graph(graph)
    _nid, type_id, params = _dataset_node(g, None)
    split = next((dict(n.resolved_params()) for n in g.nodes.values()
                  if n.type_id == "prep.split"), None)
    if split is None:
        split = {p.name: p.default for p in dcat.BLOCKS["prep.split"].params}
    result = _preview(type_id, params, split, base=base, task=task_of(g) or "multiclass",
                      modality=dcat.BLOCKS[type_id].modality or "")
    if isinstance(result, str):
        raise ApiError(result)
    return result.to_dict()


def data_fingerprint(graph: dict | Graph, base: str | None = None) -> dict:
    from ai_made_easy.core.data.fingerprint import fingerprint

    _nid, type_id, params = _dataset_node(_graph(graph), None)
    return {"fingerprint": fingerprint(type_id, params, base)}


def augmentation_preview(graph: dict | Graph, out_dir: str, images: int = 4,
                         variants: int = 6, base: str | None = None) -> dict:
    """Render evaluation and random training views of dataset images (PNG files)."""
    from ai_made_easy.core.data.augment import PreviewError
    from ai_made_easy.core.data.augment import augmentation_preview as _preview

    try:
        return _preview(_graph(graph), out_dir, base=base, python=manager().python,
                        images=images, variants=variants)
    except PreviewError as exc:
        raise ApiError(str(exc)) from exc
