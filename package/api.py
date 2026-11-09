"""
EDERA.api
===========

The two main entry points of the pipeline plus a thin wrapper:

* :func:`compute_tx_topics` — data loading through RWR diffusion of LDA topics
  (Function 1). Returns a :class:`TxTopicsResult` and, optionally, writes a
  checkpoint directory.
* :func:`cluster_tx_topics` — multi-resolution smoothing and clustering
  (Function 2). Consumes a :class:`TxTopicsResult` *or* a checkpoint directory.
* :func:`run_pipeline` — runs Function 1 then Function 2.

This module is the only place that orchestrates the per-step modules and decides
which deliverables are mandatory vs. optional (bbox-gated). See
``../drafts_and_plans/REFACTORING_PLAN.md`` for the full step/deliverable tables.
"""

import os
import json
import pickle
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Any, Union

import numpy as np
import pandas as pd

from .config import AdvancedConfig
from . import io_load, graph, partition, lda, diffusion, cluster, plotting
from . import decode as decode_mod
from .console import console
from .utils import (
    DEFAULT_WORKERS, resolve_workers, _configure_warnings,
    convert_bbox_to_internal, prefixed, fov_row_positions, EMPTY_POSITIONS,
)


# Marker genes highlighted in the LDA gene-UMAP (internal default; edit as needed).
MARKERS = [
    "ESR1", "PGR", "ERBB2", "EPCAM", "MKI67", "TOP2A",
    "KRT19", "KRT14", "KRT5", "KRT8", "KRT18",
    "CD3D", "CD3E", "CD4", "CD8A", "CD8B",
    "CD68", "CD163", "CSF1R", "CD19", "CD79A",
    "IGHA1", "IGHG1", "JCHAIN",
    "COL1A1", "DCN", "ACTA2", "PECAM1", "VWF",
    "TPSAB1/B2", "APOE", "RGS5",
]


# ---------------------------------------------------------------------------
# Function 1 presets (F1_mode)
# ---------------------------------------------------------------------------
#: Benchmarked Function 1 settings. Only these three parameters differ between
#: the presets; every other parameter keeps its own default. ``F1_mode=None``
#: uses the "default" values.
#:
#: * ``"default"``: the general-purpose setting (synthetic scatter tissue, Xenium
#:   mouse pup). Strength filter on (4 MAD), degree >= 5, ~50 transcripts per
#:   pseudo-cell.
#: * ``"complex structures"``: touching cells and contiguous, complex topologies
#:   (synthetic irregular tissue: carcinoma nests, glands, vessels, nerves).
#:   Strength filter effectively off (n_mad_strength 1e9), degree >= 2, ~25
#:   transcripts per pseudo-cell (factorial tuning on irregular_topology_v3_1M,
#:   2026-10-20).
F1_MODES = {
    "default": dict(n_mad_strength=4.0, min_connected_fss=5, transcripts_per_cell=50),
    "complex structures": dict(n_mad_strength=1e9, min_connected_fss=2, transcripts_per_cell=25),
}


def resolve_f1_mode(F1_mode=None, **explicit):
    """Resolve the preset parameters of :data:`F1_MODES`.

    Parameters
    ----------
    F1_mode : {"default", "complex structures"} or None
        Preset to apply. Case and ``_`` / ``-`` separators are ignored, so
        ``"complex_structures"`` is accepted. ``None`` means "default".
    **explicit
        The caller's values for the preset parameters (``n_mad_strength``,
        ``min_connected_fss``, ``transcripts_per_cell``). ``None`` means "not set":
        the preset value is used. Any other value overrides the preset.

    Returns
    -------
    mode : str or None
        The normalised mode name (``None`` if none was given).
    values : dict
        The resolved value of every preset parameter.
    overridden : list of str
        Preset parameters the caller set explicitly.
    """
    mode = None
    if F1_mode is not None:
        mode = " ".join(str(F1_mode).lower().replace("_", " ").replace("-", " ").split())
        if mode not in F1_MODES:
            raise ValueError(f"F1_mode must be one of {sorted(F1_MODES)} or None; got {F1_mode!r}")
    unknown = set(explicit) - set(F1_MODES["default"])
    if unknown:
        raise TypeError(f"not F1_mode parameters: {sorted(unknown)}")
    preset = F1_MODES[mode or "default"]
    values = {k: (explicit[k] if explicit.get(k) is not None else v) for k, v in preset.items()}
    overridden = [k for k in preset if explicit.get(k) is not None]
    return mode, values, overridden


# ---------------------------------------------------------------------------
# Per-step worker policy
# ---------------------------------------------------------------------------
#: Parallel steps of both entry points. Every one of them receives the caller's
#: ``n_workers``; the deliberate exception (LDA, which uses
#: ``advanced.lda_n_jobs``) is documented in :func:`_worker_plan`.
_PARALLEL_STEPS = ("knn", "fss", "smoothing", "filter_fss", "metis", "rwr",
                   "umap", "hnsw", "decode")


def _worker_plan(n_workers):
    """Resolve the worker count for every parallel step of the pipeline.

    One integer drives the whole run: ``n_workers`` is validated by
    :func:`EDERA.utils.resolve_workers` and handed to every parallel step of
    both ``compute_tx_topics`` and ``cluster_tx_topics``.

    **The one exception is ``compute_lda``**, which takes its thread count
    from ``advanced.lda_n_jobs`` instead. With the default ``"vb"`` backend
    the topic model is bit-identical at every thread count, so decoupling it
    from ``n_workers`` costs nothing and lets it be tuned for speed alone. The
    ``"gensim"`` backend always runs on one worker, because gensim's
    ``LdaMulticore`` merges worker results in *arrival* order and is not
    reproducible (see :func:`EDERA.lda.compute_lda`).

    Parameters
    ----------
    n_workers : int
        Workers for every parallel step. See
        :func:`EDERA.utils.resolve_workers` for the accepted values
        (``-1`` = every logical core, ``1`` = sequential, default
        :data:`EDERA.utils.DEFAULT_WORKERS`).

    Returns
    -------
    dict of str -> int
        Step key to worker count. Recorded verbatim in ``config.json`` /
        ``cluster_config.json`` so every run says what it actually ran.

    Notes
    -----
    How much each step gains is dataset-dependent — it is driven by how the
    transcripts divide into FOVs, since the joblib steps parallelise over FOVs
    and their wall time cannot drop below that of the single largest one.
    Worker memory adds up, so the useful setting is normally well below the
    host's core count.
    """
    workers = resolve_workers(n_workers)
    return {step: workers for step in _PARALLEL_STEPS}


# ---------------------------------------------------------------------------
# Result container + checkpoint I/O
# ---------------------------------------------------------------------------
@dataclass
class TxTopicsResult:
    """Everything Function 2 needs from Function 1.

    Attributes
    ----------
    df : DataFrame
        Post-filter transcripts with diffused ``LDA_*`` columns (float32; the
        RWR itself runs in float64) and, with ``keep_raw_topics``, ``LDA_*_raw``
        columns,
        ``tx_id`` (the transcript's identity in the input file; rows removed by
        the FSS filter are absent), ``metis_id``, coordinates, ``target`` and
        ``fov``.
    fss_list : list of csr_matrix
        Cleaned per-FOV FSS graphs, aligned with ``fss_splits``.
    fss_splits : list
        FOV identifiers.
    gene_emb_df : DataFrame
        Gene-topic embedding (``gene_name`` + ``LDA_*``).
    doc_topics : ndarray
        Per-document topic mixtures.
    doc_ids : ndarray
        METIS document ids aligned with ``doc_topics`` rows.
    lda_topics : list of str
        Names of the topic columns.
    fov_column : str
    config : dict
        Resolved exposed parameters (provenance).
    dropped : DataFrame or None
        The input transcripts that are NOT in ``df`` (removed by the FSS
        filter): ``tx_id, x, y, z, target`` in the same internal coordinates.
        Function 2's decode uses them as neighbours only; they get no label.
        ``None`` for checkpoints written before 0.4.0.
    """
    df: pd.DataFrame
    fss_list: List[Any]
    fss_splits: List[Any]
    gene_emb_df: pd.DataFrame
    doc_topics: np.ndarray
    doc_ids: np.ndarray
    lda_topics: List[str]
    fov_column: str
    config: dict = field(default_factory=dict)
    dropped: Optional[pd.DataFrame] = None

    # --- checkpoint ------------------------------------------------------
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Write the result to a checkpoint *directory* (D2).

        Sparse FSS graphs cannot live in a parquet file, so the checkpoint is a
        directory: ``transcripts.parquet`` + ``fss_graphs.pkl`` +
        ``gene_embeddings_lda.csv`` + ``doc_topics_lda.npy`` +
        ``doc_ids_lda.csv`` + ``config.json`` and, when present,
        ``dropped_transcripts.parquet``.
        """
        os.makedirs(checkpoint_path, exist_ok=True)
        self.df.to_parquet(os.path.join(checkpoint_path, "transcripts.parquet"))
        if self.dropped is not None:
            self.dropped.to_parquet(os.path.join(checkpoint_path, "dropped_transcripts.parquet"))
        with open(os.path.join(checkpoint_path, "fss_graphs.pkl"), "wb") as fh:
            pickle.dump({"fss_dict": dict(zip(self.fss_splits, self.fss_list)),
                         "fss_splits": list(self.fss_splits)}, fh)
        self.gene_emb_df.to_csv(
            os.path.join(checkpoint_path, "gene_embeddings_lda.csv"), index=False)
        np.save(os.path.join(checkpoint_path, "doc_topics_lda.npy"), self.doc_topics)
        pd.Series(self.doc_ids, name="metis_id").to_csv(
            os.path.join(checkpoint_path, "doc_ids_lda.csv"), index=False)
        meta = dict(self.config)
        meta["fov_column"] = self.fov_column
        meta["lda_topics"] = list(self.lda_topics)
        with open(os.path.join(checkpoint_path, "config.json"), "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        console.deliverable(checkpoint_path, kind="checkpoint")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "TxTopicsResult":
        """Reload a result previously written by :meth:`save_checkpoint`."""
        df = pd.read_parquet(os.path.join(checkpoint_path, "transcripts.parquet"))
        with open(os.path.join(checkpoint_path, "fss_graphs.pkl"), "rb") as fh:
            g = pickle.load(fh)
        fss_splits = g["fss_splits"]
        fss_list = [g["fss_dict"][s] for s in fss_splits]
        gene_emb_df = pd.read_csv(
            os.path.join(checkpoint_path, "gene_embeddings_lda.csv"))
        doc_topics = np.load(os.path.join(checkpoint_path, "doc_topics_lda.npy"))
        doc_ids = pd.read_csv(
            os.path.join(checkpoint_path, "doc_ids_lda.csv"))["metis_id"].to_numpy()
        with open(os.path.join(checkpoint_path, "config.json")) as fh:
            meta = json.load(fh)
        dropped_path = os.path.join(checkpoint_path, "dropped_transcripts.parquet")
        dropped = pd.read_parquet(dropped_path) if os.path.exists(dropped_path) else None
        return cls(
            df=df, fss_list=fss_list, fss_splits=fss_splits,
            gene_emb_df=gene_emb_df, doc_topics=doc_topics, doc_ids=doc_ids,
            lda_topics=meta.get("lda_topics",
                                [c for c in df.columns
                                 if c.startswith("LDA_") and not c.endswith("_raw")]),
            fov_column=meta.get("fov_column", "fov"), config=meta, dropped=dropped,
        )


# ---------------------------------------------------------------------------
# Function 1: compute_tx_topics
# ---------------------------------------------------------------------------
def compute_tx_topics(
        data_path: str,
        x_name: str = "x",
        y_name: str = "y",
        z_name: str = "z",
        gene_name: str = "gene_id",
        fov_column: str = "fov",
        x_res: float = 0.12,
        y_res: float = 0.12,
        z_res: float = 0.5,
        knn_neighbors: int = 25,
        max_edge_length: float = 4.0,
        F1_mode: Optional[str] = None,
        n_mad_strength: Optional[float] = None,
        min_connected_fss: Optional[int] = None,
        transcripts_per_cell: Optional[int] = None,
        n_topics_lda: int = 50,
        stopwords_thr: Optional[Union[float, str, List[str]]] = "2MAD",
        stopword_min_docfreq: float = 0.02,
        damping_factor_rwr: float = 0.95,
        n_workers: int = DEFAULT_WORKERS,
        out_path: str = "OUTPUT/PATH",
        plot_format: Optional[List[str]] = None,
        diagnostic_bbox: Optional[Tuple] = None,
        random_state: int = 42,
        dpi: int = 600,
        suppress_warnings: bool = True,
        verbose: bool = True,
        checkpoint_path: Optional[str] = None,
        advanced: Optional[AdvancedConfig] = None,
        id_column: Optional[str] = None,
) -> TxTopicsResult:
    """Function 1: data loading through RWR diffusion of LDA topics.

    Steps (mandatory deliverables in **bold**; optional ones are gated by
    ``diagnostic_bbox`` being non-None):

    1. load + rename + scale; explicit ``target`` gene column;
    2. RCB auto-partition if no FOV — **partition overview scatter**;
    3. per-FOV kNN;
    4. per-FOV FSS — **edge-weight histogram**, **weight-vs-distance jointplot**;
    5. unified FSS filter (no weight==1 prune) — **filter_stats.csv**,
       **post-filter weight-vs-distance scatter**; *bbox: ROI FSS-edge plot*;
    6. METIS partition — **lda_corpus.csv**; *bbox: metis spatial scatter*;
    7. corpus + stop-word detection — **gene_stopword_stats.csv**, **stop-word scatter**;
    8. LDA — **gene/doc embeddings, top genes table, gene-UMAP**;
    9. RWR diffusion (no ``beta``) — *bbox: raw-vs-diffused topic panels*.

    Parameters are the original exposed block plus the agreed plumbing
    (``plot_format``, ``diagnostic_bbox``, ``random_state``, ``dpi``,
    ``checkpoint_path``). All advanced/internal knobs live in ``advanced``.

    Stop words are detected on per-gene **document frequency** (the fraction of
    METIS pseudo-cells a gene appears in); genes are only ever kept or flagged as
    stop words, there is no rare-gene removal. ``stopwords_thr`` accepts:

    * ``"XMAD"`` (default ``"2MAD"``) — automatic threshold
      ``median(DF) + X * MAD(DF)``, estimated over the genes with
      ``doc_freq >= stopword_min_docfreq``;
    * a float in ``(0, 1]`` — a custom document-frequency cutoff (genes at or
      above it are stop words); ``stopword_min_docfreq`` is then unused;
    * a list of gene names — exactly those genes are stop words (for example
      the stop words of another run, read from its
      ``4a_gene_stopword_stats.csv``); listed genes absent from the data are
      reported and ignored, and ``stopword_min_docfreq`` is unused;
    * ``None`` — stop-word detection is skipped entirely.

    **Presets (``F1_mode``).** Two benchmarked settings (:data:`F1_MODES`):

    * ``"default"`` — the general-purpose setting;
    * ``"complex structures"`` — touching cells and contiguous, complex
      topologies (dense epithelia, glands, vessels, nerves).

    A preset sets only ``n_mad_strength``, ``min_connected_fss`` and
    ``transcripts_per_cell``. Any of the three passed explicitly overrides the
    preset; every other parameter is unaffected. ``F1_mode=None`` uses the
    "default" values. Example::

        compute_tx_topics(path, F1_mode="default", n_topics_lda=10)
        compute_tx_topics(path, F1_mode="complex structures", transcripts_per_cell=35)

    The mode and the resolved values are written to ``config.json``.

    Set ``verbose=False`` to silence the styled per-step terminal log (banner,
    timings and the delivered-output summary printed via :mod:`EDERA.console`).

    **Parallelism.** ``n_workers`` is applied to every parallel step here —
    ``compute_knn_graph``, ``compute_fss``, ``filter_fss``,
    ``compute_metis_partition``, ``run_rwr_diffusion`` and the gene UMAP — each
    of which distributes its work across FOVs. ``compute_lda`` is the one
    exception: it uses ``advanced.lda_n_jobs`` threads, and its result does not
    depend on that number (see :func:`EDERA.lda.compute_lda`); topic models
    therefore reproduce at any ``n_workers``.

    How much the FOV-parallel steps gain depends on how your transcripts divide
    into FOVs: their wall time cannot fall below that of the single largest FOV,
    so a dataset dominated by one big FOV benefits less than one with evenly
    sized FOVs. Worker memory adds up as well — each worker is a separate
    interpreter holding its share of the data — so the useful setting is
    normally well below the host's core count.

    Parameters
    ----------
    F1_mode : {"default", "complex structures"} or None, default None
        Preset for ``n_mad_strength``, ``min_connected_fss`` and
        ``transcripts_per_cell`` (see above). ``None`` = "default".
    n_mad_strength : float, optional
        Strength filter: drop nodes with strength <= median - n_mad_strength * MAD.
        Default from ``F1_mode`` (4.0 default, 1e9 = off complex structures).
    min_connected_fss : int, optional
        Iteratively drop nodes with fewer neighbours (``0`` disables). Default
        from ``F1_mode`` (5 default, 2 complex structures).
    transcripts_per_cell : int, optional
        Target METIS pseudo-cell size. Default from ``F1_mode`` (50 default,
        25 complex structures).
    n_workers : int, default 16
        Workers for every parallel step. ``1`` runs sequentially, ``-1`` uses
        every logical core. The resolved per-step plan is recorded in
        ``config.json``. See the README's *Parallelization* section.
    id_column : str, optional
        Column of the input CSV holding your own transcript identifier (unique,
        no missing values). It is carried through every step as ``tx_id`` and
        written to every output table. Default ``None``: ``tx_id`` is the
        0-based row position in the CSV. Transcripts removed by the FSS filter
        are absent from the outputs, so a missing ``tx_id`` means "dropped".

    Returns
    -------
    TxTopicsResult
    """
    F1_mode, f1_values, f1_overridden = resolve_f1_mode(
        F1_mode, n_mad_strength=n_mad_strength, min_connected_fss=min_connected_fss,
        transcripts_per_cell=transcripts_per_cell)
    n_mad_strength = f1_values["n_mad_strength"]
    min_connected_fss = f1_values["min_connected_fss"]
    transcripts_per_cell = f1_values["transcripts_per_cell"]
    if plot_format is None:
        plot_format = ["pdf"]
    if advanced is None:
        advanced = AdvancedConfig()
    _configure_warnings(suppress_warnings)
    # Per-step worker counts (see _worker_plan): n_workers everywhere except
    # compute_lda, which uses advanced.lda_n_jobs.
    plan = _worker_plan(n_workers)
    lda_jobs = advanced.lda_n_jobs if advanced.lda_backend == "vb" else 1
    os.makedirs(out_path, exist_ok=True)
    lda_out = os.path.join(out_path, "LDA_out")
    os.makedirs(lda_out, exist_ok=True)

    plotting.setup_style(dpi=dpi, dark_bg=advanced.dark_bg)

    owns_run = console.begin(
        verbose=verbose, title="compute_tx_topics", out_path=out_path,
        meta={"data": data_path,
              "workers": f"{plan['knn']} (compute_lda: {lda_jobs} "
                         f"{advanced.lda_backend} threads)",
              "F1_mode": (f"{F1_mode or 'None (= default)'}"
                          + (f", overridden: {', '.join(f1_overridden)}" if f1_overridden else "")),
              "plots": plot_format, "seed": random_state})
    console.function_start("compute_tx_topics")

    # --- Step 1: load + scale -------------------------------------------------
    with console.step(1, 9, "Load & scale transcripts"):
        data, coord_offsets = io_load.load_transcripts(
            data_path, x_name, y_name, z_name, gene_name, fov_column,
            x_res, y_res, z_res, id_column=id_column)
        # Convert the raw-unit diagnostic bbox into internal (scaled + shifted)
        # coordinates once, so every downstream plot compares like with like (§0.4).
        bbox_internal = convert_bbox_to_internal(
            diagnostic_bbox, (x_res, y_res, z_res), coord_offsets)

    # --- Step 2: auto-partition if no FOV ------------------------------------
    with console.step(2, 9, "Auto-partition FOVs"):
        data, fov_was_synthetic = io_load.autopartition_if_no_fov(
            data, fov_column=fov_column,
            target_partition_size=advanced.target_partition_size,
            coords_2d=("x", "y"))
        # MANDATORY, always-on partition overview (global, subsampled).
        console.deliverable(plotting.plot_partition_overview(
            data, fov_column, out_path, plot_format, dpi, advanced, random_state),
            kind="plot")
        # Sort by FOV to guarantee index alignment across parallel grouped ops.
        # The row order after this point carries no meaning for the caller: the
        # original row is identified by `tx_id`, which travels with it. Keep the
        # default (unstable) sort: its tie order is part of the reproducible
        # kNN / HNSW inputs.
        data = data.sort_values(fov_column).reset_index(drop=True)

    # --- Step 3: kNN ----------------------------------------------------------
    with console.step(3, 9, "kNN graph"):
        knn_idx, knn_d, split_values = graph.compute_knn_graph(
            df=data, n_neighbors=knn_neighbors, coords_cols=("x", "y", "z"),
            split_column=fov_column, n_jobs=plan["knn"],
            inner_threads=advanced.inner_threads, random_state=random_state)
        df = data.copy()
        console.info(f"Transcripts entering FSS: {len(df):,}")

    # --- Step 4: FSS ----------------------------------------------------------
    with console.step(4, 9, "Fuzzy simplicial sets"):
        fss_list, fss_splits, fss_node_indices = graph.compute_fss(
            df=df, knn_indices=knn_idx, knn_dists=knn_d,
            n_neighbors=knn_neighbors, coords_cols=("x", "y", "z"),
            split_column=fov_column, n_jobs=plan["fss"],
                random_state=random_state)

        # fuzzy_simplicial_set returns a tuple; first element is the sparse matrix.
        fss_dict = {}
        for i, split_val in enumerate(fss_splits):
            fss_clean = fss_list[i][0].tocsr()
            fss_dict[split_val] = fss_clean
            fss_list[i] = fss_clean
        console.info(f"Constructed FSS graphs for {len(fss_splits)} FOVs "
                     "(raw, unfiltered).")

        # MANDATORY global QC plots.
        console.deliverable(plotting.plot_edge_weight_histogram(
            fss_list, out_path, plot_format, dpi, advanced, random_state),
            kind="plot")
        console.deliverable(plotting.plot_edge_weight_vs_distance(
            fss_list, fss_splits, df, fov_column, out_path, plot_format, dpi,
            advanced, random_state, max_edge_length=max_edge_length), kind="plot")

    # --- Step 5: unified FSS filter (no weight==1 prune) ---------------------
    with console.step(5, 9, "Unified FSS filter"):
        fss_filtered, fss_dict_filtered, df_filtered, filter_stats = \
            graph.filter_fss(
                fss_list=fss_list, fss_splits=fss_splits, fov_column=fov_column,
                fss_node_indices=fss_node_indices, df=df,
                max_edge_distance=max_edge_length, min_node_strength="auto",
                n_mad_strength=n_mad_strength, min_connected_fss=min_connected_fss,
                chunk_size=1_000_000, n_jobs=plan["filter_fss"])

        filter_stats_path = os.path.join(out_path, f"{prefixed('filter_stats')}.csv")
        filter_stats.to_csv(filter_stats_path, index=False)
        console.deliverable(filter_stats_path, kind="table")

        fss_list = fss_filtered
        fss_dict = fss_dict_filtered
        df = df_filtered
        console.info(f"After unified filter: {len(df):,} transcripts kept; "
                     f"total edges = {sum(m.nnz for m in fss_list):,}")

        # MANDATORY post-filter QC; OPTIONAL (bbox) ROI edge plot.
        console.deliverable(plotting.plot_post_filter_weight_vs_distance(
            fss_list, fss_splits, df, fov_column, out_path, plot_format, dpi,
            advanced, random_state), kind="plot")
        console.deliverable(plotting.plot_fss_edges_in_bbox(
            fss_dict, df, fov_column, bbox_internal, out_path, plot_format, dpi),
            kind="plot")

    # --- Step 6: METIS --------------------------------------------------------
    with console.step(6, 9, "METIS partition"):
        metis_labels_dict = partition.compute_metis_partition(
            fss_list, fss_splits, transcripts_per_cell, n_jobs=plan["metis"],
            contig=advanced.metis_contig)
        # One pass for the FOV rows, one column write (not a full-table mask
        # and a .loc write per FOV); identical ids.
        rows_by_fov = fov_row_positions(df, fov_column)
        metis_id = np.full(len(df), -1, dtype=np.int64)
        for f in fss_splits:
            metis_id[rows_by_fov.get(f, EMPTY_POSITIONS)] = metis_labels_dict[f]
        df["metis_id"] = metis_id

        lda_corpus_path = os.path.join(out_path, f"{prefixed('lda_corpus')}.csv")
        df[["metis_id", "target"]].to_csv(lda_corpus_path, index=False)
        console.deliverable(lda_corpus_path, kind="table")

        # OPTIONAL (bbox) METIS spatial scatter.
        console.deliverable(plotting.plot_metis_spatial(
            df, fov_column, bbox_internal, out_path, plot_format, dpi),
            kind="plot")

    # --- Step 7: corpus + stop-word detection --------------------------------
    with console.step(7, 9, "Corpus & stop-word detection"):
        dt_matrix, doc_ids, term_ids, df_to_doc = lda.build_corpus_from_metis(
            df, doc_col="metis_id", term_col="target", drop_unassigned=True)

        if stopwords_thr is not None:
            keep_mask, stopword_stats, thr_info = lda.auto_detect_stopwords(
                dt_matrix, term_ids, stopwords_thr=stopwords_thr,
                min_docfreq_for_stats=stopword_min_docfreq)
            stopword_stats_path = os.path.join(
                lda_out, f"{prefixed('gene_stopword_stats')}.csv")
            stopword_stats.sort_values("doc_freq", ascending=False).to_csv(
                stopword_stats_path)
            console.deliverable(stopword_stats_path, kind="table")
            counts = stopword_stats["reason"].value_counts()
            console.info("Stop-word category counts: "
                         + ", ".join(f"{k}={v}" for k, v in counts.items()))
            dt_filt = dt_matrix[:, keep_mask]
            term_filt = np.asarray(term_ids)[keep_mask]
            # MANDATORY stop-word scatter (gene-level, global). The grey
            # stats-pool line is only meaningful for the automatic MAD rule.
            console.deliverable(plotting.plot_stopword_scatter(
                stopword_stats, lda_out, plot_format, dpi,
                docfreq_cutoff=(stopword_min_docfreq
                                if thr_info["mode"] == "mad" else None),
                docfreq_threshold=thr_info["threshold"],
                thr_spec=(f"explicit list, {thr_info['n_found']} genes"
                          if thr_info["mode"] == "list" else str(stopwords_thr))),
                kind="plot")
        else:
            dt_filt = dt_matrix
            term_filt = term_ids
        console.info(f"Corpus for LDA: {dt_filt.shape[0]:,} docs x "
                     f"{dt_filt.shape[1]:,} genes")

    # --- Step 8: LDA ----------------------------------------------------------
    with console.step(8, 9, "LDA topic model"):
        lda_model, doc_topics, gene_emb_df = lda.compute_lda(
            dt_filt, term_filt, n_topics=n_topics_lda, n_jobs=lda_jobs,
            chunksize=advanced.lda_chunksize, passes=advanced.lda_passes,
            eta=advanced.lda_eta, random_state=random_state,
            max_train_docs=advanced.lda_max_train_docs,
            backend=advanced.lda_backend)
        lda_fit = lda_model.edera_fit_info

        gene_emb_path = os.path.join(
            lda_out, f"{prefixed('gene_embeddings_lda')}.csv")
        gene_emb_df.to_csv(gene_emb_path, index=False)
        console.deliverable(gene_emb_path, kind="table")
        doc_topics_path = os.path.join(
            lda_out, f"{prefixed('doc_topics_lda')}.npy")
        np.save(doc_topics_path, doc_topics)
        console.deliverable(doc_topics_path, kind="array")
        doc_ids_path = os.path.join(lda_out, f"{prefixed('doc_ids_lda')}.csv")
        pd.Series(doc_ids, name="metis_id").to_csv(doc_ids_path, index=False)
        console.deliverable(doc_ids_path, kind="table")
        top_df = lda.top_genes_per_topic(gene_emb_df, n_top=n_topics_lda)
        top_genes_path = os.path.join(
            lda_out, f"{prefixed('top_genes_per_topic')}.csv")
        top_df.to_csv(top_genes_path, index=False)
        console.deliverable(top_genes_path, kind="table")

        # MANDATORY gene-UMAP in LDA space (gene-level, global).
        umap_written, _ = plotting.plot_gene_umap_lda(
            gene_emb_df, MARKERS, lda_out, plot_format, dpi, random_state,
            n_jobs=plan["umap"], umap_parallel=advanced.umap_parallel)
        console.deliverable(umap_written, kind="plot")

    # --- Step 9: merge LDA + RWR diffusion -----------------------------------
    with console.step(9, 9, "RWR diffusion"):
        lda_topics = [c for c in gene_emb_df.columns if c.startswith("LDA_")]
        emb = gene_emb_df[lda_topics].to_numpy()
        eps = emb.min() * 0.1

        # Restart matrix: each transcript's gene embedding, looked up by
        # `target` straight into one float32 (n x K) array. Genes without an
        # embedding (stop-words) get eps, as the former left merge + fillna
        # did. A DataFrame merge would rebuild the whole transcript table
        # (every column) just to append the topics; the lookup allocates only
        # the topic matrix, which the RWR then overwrites in place.
        gene_names = gene_emb_df["gene_name"]
        if gene_names.duplicated().any():
            raise RuntimeError(
                "gene embedding lists a gene twice; the per-transcript topic "
                "lookup (and tx_id) would be ambiguous")
        gene_pos = pd.Index(gene_names).get_indexer(df["target"])
        table = np.vstack([emb, np.full((1, emb.shape[1]), eps)]).astype(
            diffusion.TOPIC_DTYPE)
        X = table[np.where(gene_pos < 0, len(emb), gene_pos)]
        del gene_pos, table
        # Raw (pre-diffusion) copies for before/after comparison. They are the
        # gene embeddings looked up by `target`, so they can be dropped
        # (advanced.keep_raw_topics=False) and rebuilt exactly when needed.
        X_raw = X.copy() if advanced.keep_raw_topics else None

        console.info(f"Diffusing {len(lda_topics)} topics "
                     f"(damping = {damping_factor_rwr:g})...")
        diffusion.diffuse_topic_matrix(
            X, fov_row_positions(df, fov_column), fss_list, fss_splits,
            damping_factor_rwr=damping_factor_rwr,
            tolerance=advanced.rwr_tolerance,
            max_steps=advanced.rwr_max_steps, n_jobs=plan["rwr"])
        df[lda_topics] = X
        del X
        if X_raw is not None:
            df[[f"{t}_raw" for t in lda_topics]] = X_raw
            del X_raw
        console.info("Parallel RWR complete. LDA scores successfully diffused.")

        # OPTIONAL (bbox) raw-vs-diffused topic panels.
        console.deliverable(plotting.plot_raw_vs_diffused_topics(
            df, gene_emb_df, fov_column, bbox_internal, lda_topics, out_path,
            plot_format, dpi, advanced.dark_bg), kind="plot")

    # --- Assemble result + optional checkpoint -------------------------------
    config = dict(
        data_path=data_path, x_name=x_name, y_name=y_name, z_name=z_name,
        gene_name=gene_name, fov_column=fov_column, id_column=id_column,
        x_res=x_res, y_res=y_res, z_res=z_res, knn_neighbors=knn_neighbors, max_edge_length=max_edge_length,
        F1_mode=F1_mode, F1_mode_overridden=f1_overridden,
        n_mad_strength=n_mad_strength, min_connected_fss=min_connected_fss,
        transcripts_per_cell=transcripts_per_cell, n_topics_lda=n_topics_lda,
        stopwords_thr=stopwords_thr, stopword_min_docfreq=stopword_min_docfreq,
        # Whether advanced.lda_max_train_docs actually engaged, and the
        # resolved backend/batch. Two checkpoints with identical nominal
        # parameters are only comparable if their topic models saw the same
        # documents with the same backend, so record the resolved values
        # rather than leaving them to be re-derived.
        n_docs_lda_total=int(dt_filt.shape[0]),
        n_docs_lda_trained=int(lda_fit["n_docs_trained"]),
        lda_backend=lda_fit["backend"], lda_n_jobs=lda_fit["n_jobs"],
        lda_batch_size=int(lda_fit["batch_size"]),
        lda_cap_rule=lda_fit["cap_rule"],
        damping_factor_rwr=damping_factor_rwr,
        n_workers=n_workers, worker_plan=plan,
        out_path=out_path, plot_format=plot_format,
        random_state=random_state, dpi=dpi, suppress_warnings=suppress_warnings,
        # Persist the scaling origins + resolutions so a raw-unit diagnostic_bbox
        # can be converted to internal coordinates in Function 2 as well (§0.4).
        coord_offsets=list(coord_offsets),
        coord_resolutions=[x_res, y_res, z_res],
        advanced=asdict(advanced),
    )

    # The input transcripts F1 removed: Function 2's decode uses them as
    # neighbours (the decode was validated with every input transcript present).
    dropped = data.loc[~data["tx_id"].isin(df["tx_id"]),
                       ["tx_id", "x", "y", "z", "target"]].reset_index(drop=True)
    del data
    config["n_dropped"] = int(len(dropped))

    result = TxTopicsResult(
        df=df, fss_list=fss_list, fss_splits=fss_splits, gene_emb_df=gene_emb_df,
        doc_topics=doc_topics, doc_ids=doc_ids, lda_topics=lda_topics,
        fov_column=fov_column, config=config, dropped=dropped)

    if checkpoint_path is not None:
        result.save_checkpoint(checkpoint_path)

    console.function_end("compute_tx_topics")
    console.end(owns_run)
    return result


# ---------------------------------------------------------------------------
# Function 2: cluster_tx_topics
# ---------------------------------------------------------------------------
def cluster_tx_topics(
        result: Optional[TxTopicsResult] = None,
        *,
        checkpoint_path: Optional[str] = None,
        sample_fraction_transfer_cluster: float = 0.2,
        sample_max_nodes: Optional[int] = 10_000_000,
        n_neighbors_hnsw: int = 100,
        leiden_resolution: Optional[List[float]] = None,
        min_ratio_merge_small_clusters: float = 0.0,
        marker_merge: bool = True,
        marker_merge_max_markers: int = 50,
        marker_merge_min_similarity: float = 0.55,
        decode: bool = True,
        smoothing_threshold: float = 0.01,
        min_delta: float = 0.01,
        max_loops: int = 3,
        n_workers: int = DEFAULT_WORKERS,
        out_path: str = "OUTPUT/PATH",
        plot_format: Optional[List[str]] = None,
        diagnostic_bbox: Optional[Tuple] = None,
        random_state: int = 42,
        dpi: int = 600,
        suppress_warnings: bool = True,
        verbose: bool = True,
        advanced: Optional[AdvancedConfig] = None,
        fss_weight: float = 0.5,
) -> pd.DataFrame:
    """Function 2: multi-resolution smoothing and clustering.

    Consumes either an in-memory ``result`` from :func:`compute_tx_topics` or a
    ``checkpoint_path`` directory written by it. Steps 2-5 run once per
    resolution; each resolution gets its own ``final_cluster_res<r>`` column.

    Steps:

    1. **Leverage sampling.** ``sample_fraction_transfer_cluster`` of the
       transcripts (at most ``sample_max_nodes``) are drawn with per-FOV
       quotas; within each FOV, with probability proportional to their
       statistical leverage in the diffused LDA topic space
       (:func:`EDERA.cluster.sample_leverage_by_fov`), which favours rare
       topic profiles. At ``1.0`` (and no cap reached) every transcript is
       clustered and label transfer is skipped.
    2. **Graph.** HNSW cosine kNN graph (``n_neighbors_hnsw``) on the sample,
       mixed with the FSS spatial graph (``fss_weight``), then symmetrized
       without self-loops.
    3. **Leiden + merge + transfer + smoothing + decode**, once per resolution:
       scikit-network Leiden (Newman modularity, seeded with
       ``random_state``); size-aware marker merge of the sampled clusters
       (``marker_merge``, on by default); optional small-cluster merge; HNSW
       majority-vote transfer to the unsampled transcripts; per-FOV FSS
       consensus smoothing; molecule-level decode (``decode``, on by default).
    4. Tables.

    The sampler and the Leiden implementation were chosen by the F2 clustering
    benchmark (``data/package_tests_dev/f2_bench/BENCHMARK_REPORT.md``).

    **Output columns.** ``final_cluster_res<r>`` is the label of the last stage
    that ran. When a later stage is on, the earlier labels are kept too:
    ``cluster_res<r>_leiden`` (Leiden, transferred and smoothed, no merge) when
    the merge or the decode runs, and ``cluster_res<r>_merged`` (merged,
    transferred and smoothed, not decoded) when both run. Every column covers
    the transcripts kept by Function 1.

    Mandatory deliverables: per-resolution cluster-count table,
    ``cluster_assignments.csv``, per-resolution composition summary and
    ``cluster_config.json``. Optional (bbox-gated): per-resolution spatial
    cluster scatter. Optional (global): per-resolution smoothing convergence
    trace and, with ``advanced.plot_cluster_umap``, the cluster UMAP.

    **Parallelism.** ``n_workers`` is applied to the two hnswlib calls
    (``build_hnsw_graph``, ``hnsw_transfer_neighbors``, both C++ thread pools),
    the per-FOV FSS consensus smoothing, the decode tiles (forked processes)
    and the diagnostic cluster UMAP. The Leiden step is single-threaded.

    Note that the hnswlib calls are **order-dependent above one thread**:
    ``add_items`` inserts concurrently, so the graph — and therefore the
    clustering — is only bit-reproducible at ``n_workers=1``. Sampling and
    Leiden are seeded with ``random_state``.

    Parameters
    ----------
    sample_fraction_transfer_cluster : float in (0, 1], default 0.2
        Fraction of transcripts clustered directly; the rest receive labels by
        transfer. Rare-type recovery depends on this fraction (and on the
        resulting number of graph nodes), not only on the sampler, and the
        decode does not recover what a smaller sample loses: with the default
        F1, 0.1 lost up to 0.11 P and most of some rare types (irregular 10M,
        Monocyte F1 0.94 -> 0.36) even with the decode on
        (``data/package_tests_dev/decode_package/REPORT.md``, D5).
    sample_max_nodes : int or None, default 10,000,000
        Cap on the number of clustered transcripts: the fraction actually used
        is ``min(sample_fraction_transfer_cluster, sample_max_nodes / N)``.
        Leiden on a 10M-node graph alone takes ~2.3 h; the graph memory is
        bound by the sample. ``None`` disables the cap.
    min_ratio_merge_small_clusters : float, default 0.0
        Clusters holding less than this fraction of the *sampled* transcripts
        are merged into the giant cluster they are most connected to. ``0``
        (default) disables the merge: at the former default of 0.03 every
        population below 3 % of the data is merged away by design.
    marker_merge : bool, default True
        Size-aware marker merge after Leiden, on the sampled labels and before
        transfer (:func:`EDERA.cluster.merge_clusters_marker`). At high
        resolution Leiden splits abundant cell types by gene; this merge joins
        those fragments while keeping rare and contact types, so the output
        depends much less on the resolution. Repeatedly joins the pair with the
        fewest marker genes while it has fewer than
        ``marker_merge_max_markers`` and its pseudo-bulk profiles have a
        Bhattacharyya coefficient of at least ``marker_merge_min_similarity``.
        The marker definition (fold, count floor) is in ``AdvancedConfig``
        (``marker_merge_*``). Every merge step is logged in
        ``7g_cluster_merge_log_res<r>.csv``. ``False`` returns the raw Leiden
        partition, as before 2026-10-22.
    marker_merge_max_markers : int, default 50
        Pairs with fewer marker genes are merged. Validated window 30-50; at 75
        and above rare types start to be merged.
    marker_merge_min_similarity : float, default 0.55
        Minimum pseudo-bulk Bhattacharyya coefficient of a merged pair.
    decode : bool, default True
        Molecule-level decode after smoothing (:mod:`EDERA.decode`): every
        transcript gets the cluster that maximises (own-gene likelihood under
        the cluster's gene profile) x (local cluster prior from its neighbours
        within ``advanced.decode_radius`` µm), iterated up to
        ``advanced.decode_rounds`` times with the profiles fixed. It undoes the
        label leakage between touching cells that the diffused topic vectors
        carry; it cannot rejoin gene-split fragments, which is what the merge
        does. Run tile by tile (``advanced.decode_tile_um``, halo
        ``advanced.decode_halo_um``), equal to one global decode. The
        transcripts removed by Function 1 take part as neighbours only, as in
        the validation, and get no label (older checkpoints: rebuilt from the
        input file, see ``TxTopicsResult.dropped``).
    n_workers : int, default 16
        Workers for every parallel step. ``1`` runs sequentially and is the only
        bit-reproducible setting (see above); ``-1`` uses every logical core.
        The resolved per-step plan is recorded in ``cluster_config.json``. See
        the README's *Parallelization* section.

    Returns
    -------
    DataFrame : the transcript table with one ``final_cluster_res<r>`` column per
    requested resolution, plus the ``cluster_res<r>_leiden`` /
    ``cluster_res<r>_merged`` columns of the earlier stages (see *Output
    columns*).
    """
    if result is None:
        if checkpoint_path is None:
            raise ValueError("Provide either `result` or `checkpoint_path`.")
        result = TxTopicsResult.from_checkpoint(checkpoint_path)
    if plot_format is None:
        plot_format = ["pdf"]
    if advanced is None:
        advanced = AdvancedConfig()
    if leiden_resolution is None:
        leiden_resolution = [0.25, 0.50, 0.75, 1.00]
    if not (0.0 < sample_fraction_transfer_cluster <= 1.0):
        raise ValueError(
            "sample_fraction_transfer_cluster must be in (0, 1]; got "
            f"{sample_fraction_transfer_cluster}")
    if not (0.0 <= min_ratio_merge_small_clusters < 1.0):
        raise ValueError(
            "min_ratio_merge_small_clusters must be in [0, 1); got "
            f"{min_ratio_merge_small_clusters}")
    if marker_merge and not (int(marker_merge_max_markers) >= 1 and 0.0 <= marker_merge_min_similarity <= 1.0):
        raise ValueError(
            "marker_merge_max_markers must be >= 1 and marker_merge_min_similarity in [0, 1]; got "
            f"{marker_merge_max_markers}, {marker_merge_min_similarity}")
    if sample_max_nodes is not None and int(sample_max_nodes) < 1:
        raise ValueError(f"sample_max_nodes must be >= 1 or None; got {sample_max_nodes}")
    _configure_warnings(suppress_warnings)
    # Per-step worker counts (see _worker_plan).
    plan = _worker_plan(n_workers)
    fss_weight = float(np.clip(fss_weight, 0.0, 1.0))
    os.makedirs(out_path, exist_ok=True)

    plotting.setup_style(dpi=dpi, dark_bg=advanced.dark_bg)

    owns_run = console.begin(
        verbose=verbose, title="cluster_tx_topics", out_path=out_path,
        meta={"data": checkpoint_path,
              "workers": f"{plan['hnsw']} (hnswlib + decode tiles; rest serial)",
              "plots": plot_format, "seed": random_state})
    console.function_start("cluster_tx_topics")

    df = result.df
    fss_list = result.fss_list
    fss_splits = result.fss_splits
    fov_column = result.fov_column

    # Reconstruct the raw->internal bbox conversion from provenance saved by
    # compute_tx_topics (§0.4); fall back to treating the bbox as already
    # internal if a legacy checkpoint lacks the offsets.
    offsets = result.config.get("coord_offsets")
    resolutions = result.config.get("coord_resolutions")
    if offsets is not None and resolutions is not None:
        bbox_internal = convert_bbox_to_internal(
            diagnostic_bbox, tuple(resolutions), tuple(offsets))
    else:
        bbox_internal = diagnostic_bbox

    # Feature matrix (diffused LDA topics) + coordinate/metadata table.
    lda_topics = [c for c in df.columns
                  if c.startswith("LDA_") and not c.endswith("_raw")]
    # One float32 topic matrix (n x K) serves sampling, the graph and the
    # transfer queries; the unsampled rows are queried from it by index rather
    # than copied out (they are ~80 % of the data).
    X_topics = df[lda_topics].to_numpy(dtype=diffusion.TOPIC_DTYPE)
    fov_values = df[fov_column].to_numpy()

    num_total_points = X_topics.shape[0]

    # The fraction actually clustered: the requested one, capped at
    # sample_max_nodes transcripts (Leiden time and graph memory follow the sample).
    sample_fraction = float(sample_fraction_transfer_cluster)
    if sample_max_nodes is not None:
        sample_fraction = min(sample_fraction, int(sample_max_nodes) / num_total_points)

    # --- Step 1: leverage sampling with per-FOV quotas ------------------------
    with console.step(1, 4, "Leverage sampling by FOV"):
        if sample_fraction < sample_fraction_transfer_cluster:
            console.info(f"sample_max_nodes {int(sample_max_nodes):,}: sample fraction "
                         f"{sample_fraction_transfer_cluster:g} -> {sample_fraction:.4f}")
        # When the fraction is 1.0 (cluster on the *whole* dataset) there is no
        # unseen majority: everyone is "sampled" and label transfer is skipped
        # (U8).
        full_sample = sample_fraction >= 1.0
        if full_sample:
            sample_indices = np.arange(num_total_points)
            unseen_indices = np.empty(0, dtype=int)
            console.info("sample_fraction_transfer_cluster >= 1.0: clustering on "
                         "ALL points; label transfer step skipped.")
        else:
            sample_indices, unseen_indices = cluster.sample_leverage_by_fov(
                X_topics, fov_values,
                sample_fraction, random_state=random_state)

        sampled_data = X_topics[sample_indices]
        n_unseen = len(unseen_indices)
        console.info(f"{sampled_data.shape[0]:,} sampled · "
                     f"{n_unseen:,} unseen "
                     f"of {num_total_points:,} points")

    # --- Step 2: HNSW graph + FSS integration, symmetrized ---------------------
    with console.step(2, 4, f"HNSW graph (k={n_neighbors_hnsw}) + FSS integration"):
        adj_matrix = cluster.build_hnsw_graph(
            sampled_data, k=n_neighbors_hnsw, ef_construction=advanced.ef_construction,
            M=advanced.hnsw_M, num_threads=plan["hnsw"])
        combined_adj = cluster.integrate_fss_with_hnsw(
            adj_matrix=adj_matrix, fss_list=fss_list, fss_splits=fss_splits,
            df=df, fov_column=fov_column, sample_indices=sample_indices,
            fss_weight=fss_weight)
        del adj_matrix
        leiden_graph = cluster.symmetrize_graph(combined_adj)
        del combined_adj
        console.info(f"graph: {leiden_graph.shape[0]:,} nodes, "
                     f"{leiden_graph.nnz // 2:,} undirected edges")

    # Row positions of every FOV, computed once for all resolutions (the same
    # rows, in the same order, as the boolean mask df[fov_column] == fov_id).
    fov_rows = fov_row_positions(df, fov_column)
    # Lookup is by hash; fall back to an elementwise comparison if a split id
    # and the column dtype disagree (e.g. after a parquet checkpoint
    # round-trip) instead of silently treating the FOV as empty.
    fss_rows = [fov_rows[f] if f in fov_rows
                else np.flatnonzero(np.asarray(df[fov_column] == f)).astype(np.int64)
                for f in fss_splits]

    # --- Step 3: per-resolution Leiden, merge, transfer, smoothing, decode ----
    # Label stages, in order. ``final_cluster_<res>`` holds the last one; the
    # earlier ones are kept as ``cluster_<res>_<stage>`` columns.
    merging = marker_merge or min_ratio_merge_small_clusters > 0
    stages = ["leiden"] + (["merged"] if merging else []) + (["decoded"] if decode else [])
    cluster_count_rows = []
    res_list = ", ".join(str(r) for r in leiden_resolution)
    with console.step(3, 4, f"Leiden + {'merge + ' if merging else ''}smoothing"
                            f"{' + decode' if decode else ''} (resolutions: {res_list})"):
        # The HNSW transfer neighbours depend only on the topic vectors, not on
        # the labels: build the index and query the unseen points once, then
        # only the vote is repeated per resolution. The topic matrix is not
        # needed after that and is released.
        transfer_neighbors = None
        unseen_data = None
        if n_unseen > 0 and not advanced.transfer_ot:
            transfer_neighbors = cluster.hnsw_transfer_neighbors(
                sampled_data, X_topics, k=advanced.transfer_k,
                ef_construction=advanced.ef_construction, M=advanced.hnsw_M,
                num_threads=plan["hnsw"], query_rows=unseen_indices)
        elif n_unseen > 0:
            unseen_data = X_topics[unseen_indices]   # OT transfer needs the matrix
        del X_topics
        neighbours = _decode_neighbours(result) if decode else None
        n_dropped = 0 if neighbours is None else len(neighbours)
        if marker_merge or decode:
            # Gene code of every transcript (all genes, stop-words included); the
            # F1-removed transcripts the decode uses as neighbours come after the
            # kept ones.
            targets = df["target"].to_numpy()
            if n_dropped:
                targets = np.concatenate([targets, neighbours["target"].to_numpy()])
            genes_all, all_gene_codes = np.unique(targets, return_inverse=True)
            del targets
        if marker_merge:
            # Gene codes of the clustered transcripts over the kept transcripts'
            # panel: all genes (marker counts) and the LDA vocabulary, i.e.
            # stop-words excluded (pseudo-bulk profiles).
            present, kept_codes = np.unique(all_gene_codes[:num_total_points], return_inverse=True)
            genes_kept = genes_all[present]
            gene_codes = kept_codes[sample_indices]
            del kept_codes
            n_panel = len(genes_kept)
            vocab = {g: i for i, g in enumerate(result.gene_emb_df["gene_name"])}
            profile_codes = np.array([vocab.get(g, -1) for g in genes_kept])[gene_codes]
        if decode:
            xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
            if n_dropped:
                xyz = np.vstack([xyz, neighbours[["x", "y", "z"]].to_numpy(dtype=np.float64)])
            del neighbours
            console.info(f"decode: {num_total_points:,} transcripts"
                         + (f" + {n_dropped:,} removed by Function 1 as neighbours only" if n_dropped else ""))

        def transfer_and_smooth(sampled_labels):
            """Transfer sampled labels to the unseen rows, then per-FOV FSS smoothing."""
            labels = np.zeros(num_total_points, dtype=int)
            labels[sample_indices] = sampled_labels
            if n_unseen == 0:
                pass  # U8: whole-dataset clustering -> nothing to transfer.
            elif advanced.transfer_ot:
                labels[unseen_indices] = cluster.transfer_labels_ot_unbalanced_batched(
                    sampled_data=sampled_data, sampled_labels=sampled_labels,
                    unseen_data=unseen_data, batch_size=advanced.ot_batch_size,
                    epsilon=advanced.ot_epsilon, tau=advanced.ot_tau,
                    max_iter=advanced.ot_max_iter, tol=advanced.ot_tol,
                    max_anchors=advanced.ot_max_anchors,
                    random_state=random_state)
            else:
                labels[unseen_indices] = cluster.vote_transfer_labels(
                    transfer_neighbors, sampled_labels)

            # Iterative per-FOV FSS smoothing.
            smoothed = labels.copy()
            tasks = [(fov_id, fss_list[i], labels[fss_rows[i]])
                     for i, fov_id in enumerate(fss_splits)]
            results = Parallel_smoothing(
                tasks, smoothing_threshold, min_delta, max_loops,
                advanced.consensus_threshold, plan["smoothing"])
            histories = []
            for i, (fov_id, refined_labels, frac_history) in enumerate(results):
                smoothed[fss_rows[i]] = refined_labels
                if frac_history:
                    histories.append(frac_history)
            return smoothed, histories

        for res in leiden_resolution:
            res_key = cluster._res_key(res)
            col = f"final_cluster_{res_key}"
            count_row = {"resolution": res, "res_key": res_key}

            sampled_labels = cluster.leiden_sknetwork(
                leiden_graph, resolution=res, random_state=random_state)
            n_leiden = int(len(np.unique(sampled_labels)))
            count_row["n_leiden"] = n_leiden
            stage_labels = {}
            if merging:
                # Leiden labels without any merge, transferred and smoothed alike.
                stage_labels["leiden"], _ = transfer_and_smooth(sampled_labels)

            # Size-aware marker merge of gene-split fragments (on by default).
            if marker_merge:
                sampled_labels, merge_log = cluster.merge_clusters_marker(
                    sampled_labels, gene_codes, n_panel, profile_codes, len(vocab),
                    max_markers=marker_merge_max_markers,
                    min_similarity=marker_merge_min_similarity,
                    fold=advanced.marker_merge_fold, floor_min=advanced.marker_merge_floor_min,
                    floor_per_gene=advanced.marker_merge_floor_per_gene)
                log_path = os.path.join(
                    out_path, f"{prefixed('cluster_merge_log', f'_{res_key}')}.csv")
                merge_log.to_csv(log_path, index=False)
                console.deliverable(log_path, kind="table")
                console.info(f"resolution {res}: marker merge {n_leiden} -> "
                             f"{len(np.unique(sampled_labels))} clusters")

            # Optional: merge small clusters (disabled at min_ratio 0).
            if min_ratio_merge_small_clusters > 0:
                import anndata as ad
                adata = ad.AnnData(obs=pd.DataFrame(
                    {"clusters": sampled_labels},
                    index=np.arange(len(sampled_labels)).astype(str)))
                sampled_labels = cluster.merge_small_clusters_paga_style(
                    adata, leiden_graph, label_key="clusters",
                    min_ratio=min_ratio_merge_small_clusters).astype(int)

            # Transfer + smoothing of the (merged) labels.
            pre_decode = "merged" if merging else "leiden"
            stage_labels[pre_decode], histories = transfer_and_smooth(sampled_labels)
            if merging:
                count_row["n_merged"] = int(len(np.unique(stage_labels["merged"])))

            # Molecule-level decode of the smoothed labels (on by default).
            if decode:
                to_decode = stage_labels[pre_decode]
                if n_dropped:
                    to_decode = np.concatenate([to_decode, np.full(n_dropped, -1, to_decode.dtype)])
                decoded, dinfo = decode_mod.decode_labels(
                    xyz, all_gene_codes, to_decode,
                    radius=advanced.decode_radius, k_max=advanced.decode_k_max,
                    lam=advanced.decode_lambda, rounds=advanced.decode_rounds,
                    tile_um=advanced.decode_tile_um, halo_um=advanced.decode_halo_um,
                    n_jobs=plan["decode"], log=console.info)
                stage_labels["decoded"] = decoded[:num_total_points]
                del to_decode, decoded
                count_row["decode_relabelled"] = round(dinfo["relabelled"], 6)
                count_row["decode_seconds"] = dinfo["seconds"]

            for stage in stages[:-1]:
                df[f"cluster_{res_key}_{stage}"] = stage_labels[stage]
            df[col] = stage_labels[stages[-1]]
            del stage_labels

            # MANDATORY per-resolution cluster-count row.
            n_clusters = int(pd.Series(df[col]).nunique())
            count_row["n_clusters"] = n_clusters
            cluster_count_rows.append(count_row)
            console.info(f"resolution {res} ({res_key}) → {n_clusters} clusters")
            # MANDATORY per-resolution composition summary.
            comp_path = os.path.join(
                out_path, f"{prefixed('cluster_composition', f'_{res_key}')}.csv")
            df[col].value_counts().sort_index().to_csv(comp_path, header=["count"])
            console.deliverable(comp_path, kind="table")

            # OPTIONAL global convergence trace (mean fraction per loop across FOVs).
            if histories:
                max_len = max(len(h) for h in histories)
                trace = [np.mean([h[i] for h in histories if len(h) > i])
                         for i in range(max_len)]
                console.deliverable(plotting.plot_convergence_trace(
                    trace, res_key, out_path, plot_format, dpi), kind="plot")

            # OPTIONAL (bbox) spatial cluster scatter.
            console.deliverable(plotting.plot_cluster_spatial(
                df, fov_column, col, bbox_internal, out_path, plot_format, dpi,
                advanced.dark_bg), kind="plot")

            # OPTIONAL cluster UMAP (subsampled + rasterized), off by default:
            # the costliest diagnostic left, and drawn once per resolution.
            if advanced.plot_cluster_umap:
                console.deliverable(plotting.plot_cluster_umap(
                    sampled_data, sampled_labels, res_key, out_path, plot_format,
                    dpi, advanced, random_state, n_jobs=plan["umap"]), kind="plot")

    # --- Step 4: mandatory tabular outputs -----------------------------------
    with console.step(4, 4, "Write cluster tables"):
        # Provenance for Function 2, mirroring Function 1's checkpoint config.json:
        # without it `fss_weight` / `transfer_ot` / the resolutions are not
        # recorded anywhere next to the outputs they produced.
        cluster_config_path = os.path.join(out_path, "cluster_config.json")
        with open(cluster_config_path, "w") as fh:
            json.dump({
                "sample_fraction_transfer_cluster": sample_fraction_transfer_cluster,
                "sample_max_nodes": sample_max_nodes,
                "sample_fraction_used": sample_fraction,
                "n_clustered": int(len(sample_indices)),
                "n_neighbors_hnsw": n_neighbors_hnsw,
                "leiden_resolution": list(leiden_resolution),
                "min_ratio_merge_small_clusters": min_ratio_merge_small_clusters,
                "marker_merge": bool(marker_merge),
                "marker_merge_max_markers": int(marker_merge_max_markers),
                "marker_merge_min_similarity": float(marker_merge_min_similarity),
                "marker_merge_rule": ("size-aware marker merge (cluster.merge_clusters_marker): "
                                      "fewest markers first, while < max_markers and pseudo-bulk "
                                      "Bhattacharyya >= min_similarity; fold / floor in advanced"),
                "decode": bool(decode),
                "decode_rule": ("molecule-level decode (decode.decode_labels): q ∝ B[gene, c] · π^λ, "
                                "fixed global profiles B, punkst kernel prior over neighbours within "
                                "advanced.decode_radius µm, tiled with a halo; F1-removed transcripts are "
                                "neighbours only (no label)"),
                "label_stages": stages,
                "decode_neighbours_dropped_by_f1": int(n_dropped),
                "sampler": "leverage, per-FOV quotas (cluster.sample_leverage_by_fov)",
                "graph": "HNSW cosine kNN + FSS, symmetrized (sum), no self-loops",
                "leiden": f"scikit-network {_pkg_version('scikit-network')} Leiden, "
                          "modularity='newman', C rand() seeded with random_state",
                "smoothing_threshold": smoothing_threshold,
                "min_delta": min_delta, "max_loops": max_loops,
                "fss_weight": fss_weight, "n_workers": n_workers,
                "worker_plan": plan,
                "random_state": random_state,
                "suppress_warnings": suppress_warnings,
                "advanced": asdict(advanced),
            }, fh, indent=2, default=str)
        console.deliverable(cluster_config_path, kind="table")

        counts_path = os.path.join(out_path, f"{prefixed('cluster_counts')}.csv")
        pd.DataFrame(cluster_count_rows).to_csv(counts_path, index=False)
        console.deliverable(counts_path, kind="table")

        final_cols = []
        for r in leiden_resolution:
            res_key = cluster._res_key(r)
            final_cols += [f"cluster_{res_key}_{stage}" for stage in stages[:-1]]
            final_cols.append(f"final_cluster_{res_key}")
        # `tx_id` is the transcript's identity in the input file. Transcripts
        # removed by the FSS filter in Function 1 have no row here.
        id_cols = ["tx_id"] if "tx_id" in df.columns else []
        if not id_cols:
            console.warn("the transcript table has no 'tx_id' column "
                         "(checkpoint written before tx_id existed); "
                         "cluster_assignments.csv cannot be matched to the "
                         "input rows. Re-run compute_tx_topics to add it.")
        assign = df[id_cols + ["x", "y", "z", fov_column, "target"]
                    + final_cols].copy()
        assign_path = os.path.join(
            out_path, f"{prefixed('cluster_assignments')}.csv")
        assign.to_csv(assign_path, index=False)
        console.deliverable(assign_path, kind="table")
        console.info("Columns added: " + ", ".join(final_cols))

    console.function_end("cluster_tx_topics")
    console.end(owns_run)
    return df


def _decode_neighbours(result):
    """The input transcripts removed by Function 1 (``tx_id, x, y, z, target``), or None.

    The decode uses them as neighbours only, as in its validation. Checkpoints
    written before 0.4.0 do not store them: they are then rebuilt from the input
    file recorded in ``config.json`` (same loader, same coordinate transform). If
    that is not possible the decode runs on the kept transcripts alone, which
    changes its result slightly (``data/package_tests_dev/decode_package``).
    """
    if result.dropped is not None:
        return result.dropped
    cfg = result.config
    path = cfg.get("data_path")
    if "tx_id" not in result.df.columns or not path or not os.path.exists(path):
        console.warn("decode: the transcripts removed by Function 1 are not in this "
                     "checkpoint and its input file is not available; decoding the kept "
                     "transcripts alone. Re-run compute_tx_topics (>= 0.4.0) to store them.")
        return None
    console.info(f"decode: checkpoint has no dropped transcripts; rebuilding them from {path}")
    data, offsets = io_load.load_transcripts(
        path, cfg["x_name"], cfg["y_name"], cfg["z_name"], cfg["gene_name"],
        cfg["fov_column"], *cfg["coord_resolutions"], id_column=cfg.get("id_column"))
    if not np.allclose(offsets, cfg["coord_offsets"]):
        console.warn("decode: the input file no longer matches the checkpoint "
                     "(different coordinate origin); decoding the kept transcripts alone.")
        return None
    return data.loc[~data["tx_id"].isin(result.df["tx_id"]),
                    ["tx_id", "x", "y", "z", "target"]].reset_index(drop=True)


def _pkg_version(name):
    """Installed version of a distribution, or 'unknown'."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def Parallel_smoothing(tasks, smoothing_threshold, min_delta, max_loops,
                       consensus_threshold, n_workers):
    """Run per-FOV iterative FSS smoothing with ``n_workers`` joblib workers.

    Thin wrapper around :func:`EDERA.cluster._refine_fov_chunk` kept here so
    the orchestration code reads top-to-bottom. Returns a list of
    ``(fov_id, refined_labels, fraction_history)`` tuples.

    Callers pass 1: the per-FOV work is small enough that spawning workers
    costs more than it saves (0.9 s serial vs 1.5 s at 8 and 1.6 s at 32
    workers). The parameter is kept so the step can be re-parallelized without
    touching this wrapper.
    """
    from joblib import Parallel, delayed
    console.info("Starting parallel FSS smoothing of transferred labels...")
    return Parallel(n_jobs=n_workers)(
        delayed(cluster._refine_fov_chunk)(
            fov_id, fss_graph, initial_labels, smoothing_threshold, min_delta,
            max_loops, consensus_threshold)
        for fov_id, fss_graph, initial_labels in console.pbar(tasks, desc="Refining FOVs")
    )


# ---------------------------------------------------------------------------
# Wrapper: run_pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
        data_path: str,
        # --- dataset column names (promoted for discoverability, U4) ---
        x_name: str = "x",
        y_name: str = "y",
        z_name: str = "z",
        gene_name: str = "gene_id",
        fov_column: str = "fov",
        # --- shared plumbing ---
        n_workers: int = DEFAULT_WORKERS,
        out_path: str = "OUTPUT/PATH",
        plot_format: Optional[List[str]] = None,
        diagnostic_bbox: Optional[Tuple] = None,
        random_state: int = 42,
        dpi: int = 600,
        suppress_warnings: bool = True,
        verbose: bool = True,
        fss_weight: float = 0.5,
        checkpoint_path: Optional[str] = None,
        advanced: Optional[AdvancedConfig] = None,
        id_column: Optional[str] = None,
        # --- Function 1 params ---
        compute_kwargs: Optional[dict] = None,
        # --- Function 2 params ---
        cluster_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Run Function 1 then Function 2 with shared plumbing.

    ``compute_kwargs`` / ``cluster_kwargs`` carry the function-specific exposed
    parameters (e.g. ``compute_kwargs={"knn_neighbors": 30}``); the shared
    ``n_workers / out_path / plot_format / diagnostic_bbox / random_state / dpi /
    suppress_warnings`` are passed to both.

    The source column names ``x_name / y_name / z_name / gene_name / fov_column``
    are exposed here directly (U4) and forwarded into Function 1; passing them a
    second time inside ``compute_kwargs`` raises a clear error.

    ``fss_weight`` is forwarded to Function 2 (it was previously dropped).

    ``n_workers`` is forwarded to both functions and drives every parallel step
    of each (see :func:`_worker_plan`); ``compute_lda`` is the one exception,
    using ``advanced.lda_n_jobs`` threads (its result does not depend on it).

    ``verbose`` (default True) drives a single styled terminal log around both
    functions — a banner, per-step timings, and a final summary listing every
    delivered file; set it False to run quietly.

    Returns
    -------
    DataFrame : the clustered transcript table.
    """
    compute_kwargs = dict(compute_kwargs or {})
    cluster_kwargs = dict(cluster_kwargs or {})

    # Explicit column-name args win; a duplicate in compute_kwargs is an error.
    col_names = dict(x_name=x_name, y_name=y_name, z_name=z_name,
                     gene_name=gene_name, fov_column=fov_column,
                     id_column=id_column)
    dupes = set(col_names) & set(compute_kwargs)
    if dupes:
        raise ValueError(
            f"Column-name argument(s) {sorted(dupes)} were passed both directly "
            "to run_pipeline and inside compute_kwargs; pass each only once.")

    shared = dict(n_workers=n_workers, out_path=out_path, plot_format=plot_format,
                  diagnostic_bbox=diagnostic_bbox, random_state=random_state,
                  dpi=dpi, suppress_warnings=suppress_warnings, verbose=verbose,
                  advanced=advanced)

    # Own the run at the wrapper level so the banner + final deliverables summary
    # are printed once around both functions (the nested begin() calls inside the
    # two entry points detect the active run and skip their own banner/summary).
    # Resolve only to display it: each entry point applies the same per-step plan.
    _banner_plan = _worker_plan(n_workers)
    owns_run = console.begin(
        verbose=verbose, title="run_pipeline", out_path=out_path,
        meta={"data": data_path,
              "workers": f"{_banner_plan['hnsw']} (compute_lda pinned to 1)",
              "plots": plot_format, "seed": random_state})
    try:
        result = compute_tx_topics(
            data_path=data_path, checkpoint_path=checkpoint_path, **col_names,
            **shared, **compute_kwargs)

        # fss_weight belongs to Function 2; forward it unless the caller overrode it.
        cluster_kwargs.setdefault("fss_weight", fss_weight)
        return cluster_tx_topics(result=result, **shared, **cluster_kwargs)
    finally:
        console.end(owns_run)
