from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _resolve_learning_rate(value: Any) -> Any:
    if isinstance(value, str):
        text = str(value).strip()
        if text.lower() == "auto":
            return "auto"
        return float(text)
    return float(value)


def _embedding_method(cfg: DictConfig) -> str:
    return str(cfg.embedding.method).strip().lower()


def _embedding_display_name(method: str) -> str:
    if str(method) == "pca":
        return "PCA"
    if str(method) == "umap":
        return "UMAP"
    return "t-SNE"


def _embedding_axis_labels(method: str) -> tuple[str, str]:
    if str(method) == "pca":
        return "PC 1", "PC 2"
    if str(method) == "umap":
        return "UMAP 1", "UMAP 2"
    return "t-SNE 1", "t-SNE 2"


def _run_pca(features: np.ndarray, cfg: DictConfig) -> tuple[np.ndarray, dict[str, Any]]:
    num_points = int(features.shape[0])
    if num_points < 2:
        raise RuntimeError(f"PCA requires at least 2 points, got {num_points}.")

    pca = PCA(
        n_components=2,
        svd_solver=str(cfg.pca.svd_solver),
        whiten=bool(cfg.pca.whiten),
        random_state=int(cfg.seed),
    )
    embedding = pca.fit_transform(features).astype(np.float32, copy=False)
    explained_variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float32)
    singular_values = np.asarray(pca.singular_values_, dtype=np.float32)
    return (
        embedding,
        {
            "svd_solver": str(cfg.pca.svd_solver),
            "whiten": bool(cfg.pca.whiten),
            "explained_variance_ratio": explained_variance_ratio.tolist(),
            "singular_values": singular_values.tolist(),
        },
    )


def _run_umap(features: np.ndarray, cfg: DictConfig) -> tuple[np.ndarray, dict[str, Any]]:
    num_points = int(features.shape[0])
    if num_points < 3:
        raise RuntimeError(f"UMAP requires at least 3 points, got {num_points}.")

    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "UMAP embedding requires the `umap-learn` package in the active environment. "
            "Install it or choose embedding.method=tsne|pca."
        ) from exc

    n_neighbors_requested = int(cfg.umap.n_neighbors)
    n_neighbors_used = max(2, min(n_neighbors_requested, int(num_points - 1)))
    n_jobs = int(cfg.umap.n_jobs)
    reducer_kwargs: dict[str, Any] = {
        "n_components": 2,
        "n_neighbors": int(n_neighbors_used),
        "min_dist": float(cfg.umap.min_dist),
        "metric": str(cfg.umap.metric),
        "init": str(cfg.umap.init),
        "random_state": int(cfg.seed),
        "verbose": bool(cfg.umap.verbose),
    }
    if n_jobs != 0:
        reducer_kwargs["n_jobs"] = int(n_jobs)

    try:
        reducer = umap.UMAP(**reducer_kwargs)
    except TypeError:
        reducer_kwargs.pop("n_jobs", None)
        reducer = umap.UMAP(**reducer_kwargs)

    embedding = reducer.fit_transform(features).astype(np.float32, copy=False)
    n_jobs_effective = 1 if "random_state" in reducer_kwargs else reducer_kwargs.get("n_jobs")
    return (
        embedding,
        {
            "n_neighbors_requested": int(n_neighbors_requested),
            "n_neighbors_used": int(n_neighbors_used),
            "min_dist": float(cfg.umap.min_dist),
            "metric": str(cfg.umap.metric),
            "init": str(cfg.umap.init),
            "verbose": bool(cfg.umap.verbose),
            "n_jobs_requested": reducer_kwargs.get("n_jobs"),
            "n_jobs_effective": n_jobs_effective,
        },
    )


def _run_tsne(features: np.ndarray, cfg: DictConfig) -> tuple[np.ndarray, dict[str, Any]]:
    num_points = int(features.shape[0])
    if num_points < 3:
        raise RuntimeError(f"t-SNE requires at least 3 points, got {num_points}.")

    perplexity_requested = float(cfg.tsne.perplexity)
    perplexity_used = min(perplexity_requested, float(num_points - 1))
    if perplexity_used <= 0:
        perplexity_used = 1.0

    learning_rate = _resolve_learning_rate(cfg.tsne.learning_rate)
    n_jobs = int(cfg.tsne.n_jobs)
    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity_used),
        early_exaggeration=float(cfg.tsne.early_exaggeration),
        learning_rate=learning_rate,
        max_iter=int(cfg.tsne.max_iter),
        init=str(cfg.tsne.init),
        metric=str(cfg.tsne.metric),
        random_state=int(cfg.seed),
        method=str(cfg.tsne.method),
        angle=float(cfg.tsne.angle),
        n_jobs=None if n_jobs == 0 else int(n_jobs),
        verbose=int(cfg.tsne.verbose),
    )
    embedding = tsne.fit_transform(features).astype(np.float32, copy=False)
    return (
        embedding,
        {
            "perplexity_requested": float(perplexity_requested),
            "perplexity_used": float(perplexity_used),
            "learning_rate": learning_rate,
            "max_iter": int(cfg.tsne.max_iter),
            "init": str(cfg.tsne.init),
            "metric": str(cfg.tsne.metric),
            "method": str(cfg.tsne.method),
            "angle": float(cfg.tsne.angle),
            "n_jobs": None if n_jobs == 0 else int(n_jobs),
        },
    )


def _run_embedding(features: np.ndarray, cfg: DictConfig) -> tuple[np.ndarray, dict[str, Any]]:
    method = _embedding_method(cfg)
    if method == "tsne":
        embedding, summary = _run_tsne(features, cfg)
    elif method == "pca":
        embedding, summary = _run_pca(features, cfg)
    elif method == "umap":
        embedding, summary = _run_umap(features, cfg)
    else:
        raise ValueError(f"Unsupported embedding.method: {cfg.embedding.method}")

    return (
        embedding,
        {
            "method": str(method),
            "display_name": _embedding_display_name(method),
            **summary,
        },
    )
