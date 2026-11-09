"""
EDERA.config
==============

Internal / advanced configuration for the :mod:`EDERA` pipeline.

The two main entry points (:func:`EDERA.api.compute_tx_topics` and
:func:`EDERA.api.cluster_tx_topics`) expose only the *primary* scientific
knobs directly in their signatures (see
``../drafts_and_plans/REFACTORING_PLAN.md``, section 8).
Every remaining tunable lives here, in :class:`AdvancedConfig`, with a sensible
default. A power user can override any of them by constructing an
``AdvancedConfig`` and passing it as the ``advanced=`` argument, without
cluttering the main call signatures.

Note that ``dpi`` is *not* defined here: it is an exposed parameter of the main
functions (default 600), per the agreed plan.
"""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class AdvancedConfig:
    """Advanced / internal parameters with defaults matching the original script.

    Attributes
    ----------
    target_partition_size : int
        Approximate number of transcripts per pseudo-FOV when 2D Recursive
        Coordinate Bisection (RCB) auto-partitioning is triggered (i.e. when the
        input has no native ``fov`` column).
    inner_threads : int
        Number of threads used *inside* each parallel kNN worker. Kept at 1 by
        default so that the total thread count equals ``n_workers`` and the
        machine is not oversubscribed. Raise it only if ``n_workers`` is well
        below the number of physical cores. Note that ``umap`` forces this back
        to 1 whenever ``random_state`` is set (which EDERA always does), so
        raising it has no effect on ``compute_knn_graph``.
    ef_construction : int
        HNSW build-time accuracy parameter (higher = more accurate, slower build).
    hnsw_M : int
        HNSW number of bi-directional links per element.
    hnsw_max_threads : int
        **Unused — kept only so existing callers do not break.** It used to cap
        the removed ``n_workers="auto"`` mode for the two hnswlib calls. Now
        that ``n_workers`` is always an explicit integer it is honoured
        verbatim, with no ceiling: set ``n_workers`` itself to control how many
        threads hnswlib gets.
    transfer_k : int
        Number of nearest neighbours used during HNSW majority-vote label
        transfer from the clustered subsample to the unseen majority.
    transfer_ot : bool
        If True, use optimal transport (OT) instead of HNSW for label transfer.
    ot_batch_size : int
        Number of unseen points processed per batch in the unbalanced-OT label
        transfer (memory control).
    ot_max_anchors : int or None
        Upper bound on the number of *reference* points entering the OT problem,
        drawn as a label-stratified subsample of the clustered subsample. The OT
        kernel is dense and sized ``ot_max_anchors x ot_batch_size``, so this cap
        — not ``sample_fraction_transfer_cluster`` — is what bounds memory
        (roughly ``3 * ot_max_anchors * ot_batch_size * 4`` bytes at the peak;
        ~6 GB at the defaults). ``None`` uses every sampled point and is only
        safe for small subsamples.
    ot_epsilon : float
        Entropic regularisation coefficient of the OT transfer (lower = sharper).
    ot_tau : float
        Marginal-relaxation parameter of the unbalanced OT transfer.
    ot_max_iter : int
        Maximum generalised-Sinkhorn iterations in the OT transfer.
    ot_tol : float
        Convergence tolerance for the OT scaling loop.
    consensus_threshold : float
        Minimum weighted neighbour-vote fraction required to flip a label during
        per-FOV FSS consensus smoothing.
    marker_merge_fold : float
        Size-aware marker merge (``cluster_tx_topics(marker_merge=True)``): a
        gene is a marker of a cluster pair when it is at least this many times
        more frequent in one of the two clusters. Default 2.0.
    marker_merge_floor_min : float
        Minimum transcript count of a marker gene in the richer cluster. Default 5.
    marker_merge_floor_per_gene : float
        The count floor also scales with the mean per-gene count of the smaller
        cluster: ``floor = max(floor_min, floor_per_gene * n_smaller / n_genes)``.
        Default 1.952, i.e. 0.2 % of the smaller cluster on a 976-gene panel
        (the calibration panel), and proportionally less on larger panels.
    decode_radius : float
        Molecule-level decode (``cluster_tx_topics(decode=True)``,
        :mod:`EDERA.decode`): radius in µm of the neighbourhood that forms a
        transcript's local cluster prior. Default 5.0 (selected among 3/5/8 on
        the development tissues).
    decode_k_max : int
        Maximum neighbours per transcript within ``decode_radius``. Default 128.
    decode_lambda : float
        Exponent λ of the local prior against the own-gene likelihood. Default 1.0.
    decode_rounds : int
        Decode rounds (prior, then posterior); all of them always run, in
        every tile. Default 10.
    decode_tile_um : float
        Side in µm of the tiles the decode runs on, one forked worker per tile.
        Default 400.
    decode_halo_um : float
        Margin in µm of neighbouring transcripts decoded with each tile and then
        discarded. Default 30, which makes the tiled result equal to one global
        decode at the default radius and rounds.
    lda_backend : str
        ``"vb"`` (default): numba-compiled online variational Bayes
        (:mod:`EDERA.lda_vb`), multi-threaded and bit-identical at every
        thread count. ``"gensim"``: the previous single-process gensim
        ``LdaModel``, kept as a reference; ~50-100x slower at 10M+.
    lda_n_jobs : int
        Threads for the ``"vb"`` LDA fit and inference. Deliberately separate
        from ``n_workers``: the result does not depend on it, so it can be
        tuned for speed alone. Ignored by the gensim backend (always 1).
    lda_passes : int
        Passes (epochs) of online LDA training.
    lda_chunksize : int or "auto"
        Documents per online update. ``"auto"`` (default) = training
        documents / 50, i.e. 50 updates per pass, which minimised held-out
        perplexity on the 1M, 10M and 100M real corpora. The previous fixed
        2000 under-fits small corpora and is noisy on large ones.
    lda_eta : str
        Prior on the per-topic gene distribution (e.g. ``"symmetric"``).
    lda_max_train_docs : int, "auto" or None
        Optional cap on the number of METIS pseudo-cell documents used to
        **train** LDA. ``None`` or ``<= 0`` (default) fits every document.
        An int fits a uniform ``random_state``-seeded sample of that size;
        ``"auto"`` derives the size from the corpus' gene token counts (see
        :func:`EDERA.lda._auto_train_docs`). **Per-document topics are
        always inferred for every document**, so
        ``doc_topics``/``5d_doc_topics_lda.npy`` stay full-corpus. No cap is
        the default because the full 100M corpus (~1.8M documents) fits in
        about a minute, and training on everything measurably improves the
        seed-to-seed stability of the gene embedding over the previous
        200,000 cap. A cap makes LDA cost flat in the document count, which
        flattens the LDA segment of any scalability curve by construction.
    rwr_tolerance : float
        Frobenius-norm convergence tolerance for the RWR fixed-point iteration.
    rwr_max_steps : int
        Maximum number of RWR iterations before forced termination.
    metis_contig : bool
        If True, METIS is asked for contiguous parts (``contig=1``): every
        pseudo-cell is a connected piece of the FSS graph. Default False (the
        historical behaviour): k-way METIS with a tight balance attaches small
        fragments to neighbouring parts, so a share of pseudo-cells consist of
        two or more nearby pieces. Changes every pseudo-cell; ~2x METIS time.
    keep_raw_topics : bool
        If True (default), Function 1 keeps an ``LDA_*_raw`` copy of every
        topic column (the pre-diffusion values) in the transcript table and the
        checkpoint. The raw vector of a transcript is exactly its gene's LDA
        embedding, so the copies are redundant: ``False`` drops them (halving
        the topic memory, ~200 B per transcript at float32) without changing any result.
        The raw-vs-diffused diagnostic then rebuilds them for the bbox only.
        Set False for very large inputs.
    plot_cluster_umap : bool
        Whether to draw the per-resolution cluster UMAP. **Off by default**: even
        after being cut to a single 10,000-point ``fit_transform`` it remained the
        most expensive diagnostic in the pipeline, and it is drawn once per Leiden
        resolution, so its cost multiplies with ``len(leiden_resolution)``. It is a
        diagnostic figure only -- nothing downstream reads it. Set True to restore
        it.
    umap_parallel : bool
        If False (default), diagnostic UMAP embeddings are single-threaded and
        seeded, so the figures reproduce exactly. Set True to run them
        multi-threaded with no fixed ``random_state`` — measured at only 1.14x
        faster on the cluster UMAP, and non-deterministic, which is why it is
        no longer the default.
    umap_n_neighbors : int
        ``n_neighbors`` for the diagnostic UMAP of the clustered subsample.
    umap_min_dist : float
        ``min_dist`` for the diagnostic UMAP of the clustered subsample.
    umap_subsample_fraction : float
        **Unused — kept only so existing callers do not break.** The cluster
        UMAP used to fit on this fraction and then run UMAP's slow ``transform``
        over ``cluster_umap_subsample`` points; it now fits and plots one single
        subsample, so there is no separate fit fraction. Size the figure with
        ``cluster_umap_subsample`` instead.
    diag_hist_subsample : int
        Maximum number of FSS edges per FOV sampled for the edge-weight histogram.
    diag_jointplot_subsample : int
        Maximum number of edges sampled for the edge-weight-vs-distance jointplot.
    diag_scatter_subsample : int
        Maximum number of points sampled for the post-filter weight-vs-distance
        scatter.
    cluster_umap_subsample : int
        Number of points both *fitted* and rendered in the always-on cluster
        UMAP. Default 10,000: at the previous 200,000 this single diagnostic
        figure was 10-11 % of the whole pipeline runtime (43.1 s), almost all of
        it in UMAP's ``transform`` pass. Raise it if the figure looks too sparse.
    partition_overview_subsample : int
        Maximum number of points rendered in the always-on RCB partition overview.
    dark_bg : bool
        If True, spatial scatter panels use a black background (screen-friendly).
        Default False (white/transparent) for clean Adobe Illustrator editing.
    """

    # --- 2D auto-partitioning ------------------------------------------------
    target_partition_size: int = 100_000

    # --- parallelism ---------------------------------------------------------
    inner_threads: int = 1

    # --- HNSW graph ----------------------------------------------------------
    ef_construction: int = 200
    hnsw_M: int = 16
    hnsw_max_threads: int = 32
    transfer_k: int = 15
    transfer_ot: bool = False

    # --- optimal-transport label transfer (used when transfer_ot=True) ------
    ot_batch_size: int = 10_000
    ot_max_anchors: int = 50_000
    ot_epsilon: float = 0.05
    ot_tau: float = 1.0
    ot_max_iter: int = 100
    ot_tol: float = 1e-4

    # --- FSS consensus smoothing --------------------------------------------
    consensus_threshold: float = 0.8

    # --- size-aware marker merge (cluster_tx_topics(marker_merge=True)) ------
    marker_merge_fold: float = 2.0
    marker_merge_floor_min: float = 5.0
    marker_merge_floor_per_gene: float = 1.952

    # --- molecule-level decode (cluster_tx_topics(decode=True)) --------------
    decode_radius: float = 5.0
    decode_k_max: int = 128
    decode_lambda: float = 1.0
    decode_rounds: int = 10
    decode_tile_um: float = 400.0
    decode_halo_um: float = 30.0

    # --- LDA -----------------------------------------------------------------
    lda_backend: str = "vb"
    lda_n_jobs: int = 16
    lda_passes: int = 10
    lda_chunksize: Union[int, str] = "auto"
    lda_eta: str = "symmetric"
    lda_max_train_docs: Optional[Union[int, str]] = None

    # --- RWR diffusion -------------------------------------------------------
    rwr_tolerance: float = 1e-5
    rwr_max_steps: int = 1000
    keep_raw_topics: bool = True
    metis_contig: bool = False

    # --- diagnostic UMAP determinism / speed --------------------------------
    umap_parallel: bool = False
    plot_cluster_umap: bool = False

    # --- diagnostic UMAP of the clustered subsample -------------------------
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_subsample_fraction: float = 0.01   # unused; see the class docstring

    # --- diagnostic subsample caps ------------------------------------------
    diag_hist_subsample: int = 10_000
    diag_jointplot_subsample: int = 50_000
    diag_scatter_subsample: int = 100_000
    cluster_umap_subsample: int = 10_000
    partition_overview_subsample: int = 200_000

    # --- styling -------------------------------------------------------------
    dark_bg: bool = False
