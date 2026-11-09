"""
EDERA.utils
=============

Small cross-cutting helpers shared by the pipeline entry points:

* :func:`resolve_workers` — validate ``n_workers`` and turn it into a concrete
  positive integer, so that libraries which reject ``-1`` (gensim, hnswlib) get a
  usable value (§0.1 of the update plan).
* :func:`_configure_warnings` — a single switch that silences third-party warning
  chatter (gensim/numba/umap/…) without swallowing real exceptions (§0.2).
* :func:`convert_bbox_to_internal` — convert a ``diagnostic_bbox`` given in *raw*
  input units into the internal (scaled + zero-shifted) coordinate frame used
  everywhere downstream (§0.4).
* :data:`OUTPUT_PREFIXES` — single source of truth for the ``<step><letter>_``
  filename prefixes attached to every delivered artifact (§0.3).
"""

import numbers
import os
import logging
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# §0.1 Worker resolution (integer only; -1 = every logical core)
# ---------------------------------------------------------------------------
#: Worker count used by every parallel step unless the caller says otherwise.
#: Suggested from a sweep on a real dataset: past roughly this many workers the
#: parallel steps stop getting faster while their memory keeps growing. It is a
#: suggestion, not a measured optimum for *your* data — see the README's
#: *Parallelization* section.
DEFAULT_WORKERS = 16

#: Value of ``n_workers`` meaning "every logical core on this host".
ALL_CORES = -1


def all_cores():
    """Logical core count — what ``n_workers=-1`` resolves to."""
    return os.cpu_count() or 1


def resolve_workers(n_workers=DEFAULT_WORKERS):
    """Validate ``n_workers`` and return a concrete positive worker count.

    ``n_workers`` is an **integer only**; there is no ``"auto"`` mode. One value
    drives every parallel step of both entry points (see
    :func:`EDERA.api._worker_plan`), with a single documented exception:
    :func:`EDERA.lda.compute_lda` always runs on one worker, because its
    multi-process backend is not reproducible.

    Parameters
    ----------
    n_workers : int, default :data:`DEFAULT_WORKERS` (16)
        * ``n`` — use exactly ``n`` workers.
        * ``1`` — run sequentially. This is the only fully bit-reproducible
          setting, because hnswlib's ``add_items`` is order-dependent above one
          thread.
        * ``-1`` — use every logical core on the host. Note that worker memory
          adds up: each one is a separate interpreter holding its own share of
          the data, so on a many-core machine this can need far more RAM than
          the same run at a modest worker count.

    Returns
    -------
    int
        A concrete, positive worker count. Safe to pass to libraries that
        reject ``-1`` (``gensim.LdaMulticore``, ``hnswlib``, ``joblib``).

    Raises
    ------
    ValueError
        If ``n_workers`` is ``0``, a negative other than ``-1``, or not an
        integer (including the removed ``"auto"``).

    Notes
    -----
    Some libraries additionally want ``workers - 1`` (gensim). Callers apply
    that adjustment themselves on top of the value returned here, e.g.
    ``max(1, resolve_workers(n) - 1)``.
    """
    if isinstance(n_workers, str):
        raise ValueError(
            f"n_workers must be an int; got {n_workers!r}. The 'auto' mode was "
            f"removed — pass an integer instead ({DEFAULT_WORKERS} is the "
            "default, 1 runs sequentially, -1 uses every logical core).")
    if isinstance(n_workers, bool) or not isinstance(n_workers, numbers.Integral):
        raise ValueError(
            f"n_workers must be an int; got {n_workers!r} "
            f"({type(n_workers).__name__}).")

    n_workers = int(n_workers)
    if n_workers == ALL_CORES:
        return all_cores()
    if n_workers < 1:
        raise ValueError(
            f"n_workers must be a positive int or -1 (all cores); "
            f"got {n_workers}.")
    return n_workers


# ---------------------------------------------------------------------------
# §0.2 Warning suppression (flag, default on)
# ---------------------------------------------------------------------------
# Third-party loggers that emit noisy INFO/WARNING chatter during a run.
_NOISY_LOGGERS = (
    "gensim",
    "gensim.models.ldamulticore",
    "gensim.models.ldamodel",
    "gensim.utils",
    "gensim.corpora",
    "numba",
    "numba.core",
    "umap",
    "pymetis",
    "matplotlib",
)


def _configure_warnings(flag):
    """Enable or disable third-party warning / log chatter.

    Parameters
    ----------
    flag : bool
        If True, ``warnings.filterwarnings("ignore")`` and the noisy third-party
        loggers are raised to ``ERROR`` level. If False, default warning
        behaviour is restored and those loggers go back to ``WARNING``.

    Notes
    -----
    Only *warnings* and log chatter are affected — real exceptions still
    propagate. This is deliberately global (matplotlib/gensim configure logging
    at import time), and is called at the top of each entry point.

    Python's warning filters live in the interpreter, so they are **not**
    inherited by the fresh interpreters ``joblib``'s loky backend spawns for the
    kNN / FSS / METIS / RWR / smoothing steps — those workers would keep printing
    numba and umap chatter. Setting ``PYTHONWARNINGS`` covers them, because loky
    passes the parent environment to each worker process.
    """
    if flag:
        warnings.filterwarnings("ignore")
        level = logging.ERROR
        # Propagate into joblib/loky worker interpreters (see Notes).
        os.environ["PYTHONWARNINGS"] = "ignore"
    else:
        warnings.resetwarnings()
        level = logging.WARNING
        os.environ.pop("PYTHONWARNINGS", None)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)


# ---------------------------------------------------------------------------
# §0.4 Raw -> internal bbox conversion
# ---------------------------------------------------------------------------
def convert_bbox_to_internal(bbox, res, offsets):
    """Convert a raw-unit ``diagnostic_bbox`` into internal (scaled) coordinates.

    ``diagnostic_bbox`` is specified by the user in the *original* input units
    (before per-axis resolution scaling and before the zero-shift applied in
    :func:`EDERA.io_load.load_transcripts`). Internally every coordinate is
    ``coord_internal = coord_raw * res - offset`` per axis. This helper applies
    the identical transform to the bbox edges so that a bbox given in raw units
    lines up exactly with the scaled point cloud.

    Parameters
    ----------
    bbox : tuple or None
        ``(xmin, xmax, ymin, ymax)`` or
        ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in raw input units. ``None``
        passes through unchanged.
    res : tuple of float
        ``(x_res, y_res, z_res)`` per-axis resolutions.
    offsets : tuple of float
        ``(x_min, y_min, z_min)`` post-scaling per-axis origins (the values
        subtracted in :func:`load_transcripts`).

    Returns
    -------
    tuple or None
        The bbox in internal coordinates, same length as the input.
    """
    if bbox is None:
        return None
    x_res, y_res, z_res = res
    x_off, y_off, z_off = offsets

    def conv(v, r, o):
        return v * r - o

    xmin = conv(bbox[0], x_res, x_off)
    xmax = conv(bbox[1], x_res, x_off)
    ymin = conv(bbox[2], y_res, y_off)
    ymax = conv(bbox[3], y_res, y_off)
    # Scaling never inverts axis order (res > 0), so min/max stay ordered.
    out = [xmin, xmax, ymin, ymax]
    if len(bbox) >= 6:
        out += [conv(bbox[4], z_res, z_off), conv(bbox[5], z_res, z_off)]
    return tuple(out)


# ---------------------------------------------------------------------------
# §0.3 Output filename prefixes (ordering + provenance)
# ---------------------------------------------------------------------------
# Single source of truth mapping a logical output id -> its <step><letter>_
# prefix. The number encodes the pipeline step; the letter disambiguates
# multiple outputs of the same step. Keeping this in one dict keeps the
# numbering consistent across plots and tables.
OUTPUT_PREFIXES = {
    # Step 0 — partitioning
    "partition_overview": "0a",
    # Step 1 — raw FSS QC
    "fss_edge_weight_histogram": "1a",
    "fss_edge_weight_vs_distance": "1b",
    # Step 2 — filter
    "filter_stats": "2a",
    "fss_post_filter_weight_vs_distance": "2b",
    "fss_edges_bbox": "2c",
    # Step 3 — METIS
    "lda_corpus": "3a",
    "metis_spatial": "3b",
    # Step 4 — stop-words
    "gene_stopword_stats": "4a",
    "auto_stopwords": "4b",
    # Step 5 — LDA
    "gene_embeddings_lda": "5a",
    "gene_embeddings_lda_umap": "5b",
    "top_genes_per_topic": "5c",
    "doc_topics_lda": "5d",
    "doc_ids_lda": "5e",
    # Step 6 — RWR diffusion
    "topic_raw_vs_diffused": "6a",
    # Step 7 — clustering
    "cluster_counts": "7a",
    "cluster_composition": "7b",
    "cluster_assignments": "7c",
    "cluster_spatial": "7d",
    "cluster_umap": "7e",
    "smoothing_convergence": "7f",
    "cluster_merge_log": "7g",
}


def prefixed(output_id, suffix=""):
    """Return ``"<step><letter>_<output_id><suffix>"`` for a known output id.

    Parameters
    ----------
    output_id : str
        Key into :data:`OUTPUT_PREFIXES`.
    suffix : str
        Optional extra token appended to the base name (e.g. ``f"_{res_key}"``
        or ``f"_{topic}"``). Included so per-resolution / per-topic outputs keep
        the same numeric prefix.

    Returns
    -------
    str
        The prefixed base filename (no extension). If ``output_id`` is unknown
        the name is returned unprefixed (fail-soft).
    """
    prefix = OUTPUT_PREFIXES.get(output_id)
    if prefix is None:
        return f"{output_id}{suffix}"
    return f"{prefix}_{output_id}{suffix}"


# ---------------------------------------------------------------------------
# Per-FOV row lookup
# ---------------------------------------------------------------------------
def fov_row_positions(df, fov_column):
    """Row positions of every FOV, found in one pass over ``df``.

    Returns a dict ``{fov_value: int64 ndarray of row positions}``, positions in
    increasing order, so ``array[positions]`` selects exactly the rows (in the
    same order) that the boolean mask ``df[fov_column] == fov_value`` selects.
    A FOV with no rows is absent from the dict; look it up with
    ``.get(fov, EMPTY_POSITIONS)``.

    One scan replaces one full-table comparison per FOV: with ``F`` FOVs and
    ``N`` rows the per-FOV masks cost ``O(N * F)``, which at 100M transcripts
    (1,245 FOVs) made bookkeeping, not computation, the bulk of the RWR,
    filter and METIS steps.
    """
    keys = df[fov_column].to_numpy()
    return {k: np.asarray(v, dtype=np.int64) for k, v in
            pd.Series(np.arange(len(keys), dtype=np.int64)).groupby(keys, sort=False)
            .indices.items()}


EMPTY_POSITIONS = np.zeros(0, dtype=np.int64)
