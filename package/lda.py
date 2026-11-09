"""
EDERA.lda
===========

Topic modelling over the METIS pseudo-cell corpus:

* :func:`build_corpus_from_metis` turns the long transcript table into a sparse
  (document x gene) count matrix, where documents are METIS partitions;
* :func:`auto_detect_stopwords` flags ubiquitous ("stop-word") genes via a
  **document-frequency** criterion (automatic ``median + X*MAD`` threshold or a
  custom cutoff, see ``stopwords_thr``), or takes an explicit list of genes;
* :func:`filter_stopwords` is the simpler fixed-threshold alternative (kept for
  completeness);
* :func:`compute_lda` fits the topic model: a numba-compiled online variational
  Bayes LDA with a multi-threaded, reproducible E-step (default, see
  :mod:`EDERA.lda_vb`), or gensim.

Changes vs. the original script:

* ``compute_lda`` no longer derives its worker count from ``cpu_count()``; the
  number of workers is an explicit argument (exactly ``n_workers``, issue I2).
* ``random_state`` is threaded into ``compute_lda`` (issue I8).
* ``compute_lda`` accepts ``max_train_docs``, an optional cap on the documents
  used to *fit* the model (fixed, or ``"auto"`` = derived from the corpus'
  gene token counts). Off by default: see the parameter docs.
  Per-document topics are still inferred for the whole corpus, in row blocks,
  so ``doc_topics`` stays aligned with ``doc_ids`` and the full corpus is never
  materialised as Python objects at once. With no cap the chunked inference is
  bit-identical to the previous single-call version, since gensim's only
  randomness there is one row-major ``gamma`` draw from a persistent
  ``RandomState``.
* The plotting that the stop-word helpers used to do internally has been moved
  to :mod:`EDERA.plotting`; these functions now only compute and return.
* ``compute_lda`` defaults to the ``"vb"`` backend (:mod:`EDERA.lda_vb`), a
  numba port of scikit-learn's online VB LDA. gensim's reproducible
  ``LdaModel`` path is single-threaded and spends its time in per-document
  interpreter overhead (documents are ~18 tokens), which at ~2M documents made
  it the dominant F1 cost (4h39m at 100M). The ``"vb"`` backend fits the same
  100M corpus uncapped in about a minute on 16 threads and is bit-identical
  at every thread count. See ``data/package_tests_dev/old/LDA_PERFORMANCE_FINDINGS.md``. The
  gensim backend is kept as the reference implementation.
"""

import re
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import median_abs_deviation
from gensim import models
from gensim.matutils import Sparse2Corpus

from .console import console


def build_corpus_from_metis(df, doc_col="metis_id", term_col="target",
                            drop_unassigned=True):
    """Build a sparse (document x gene) count matrix from a long transcript df.

    Each METIS partition is a document; each gene (``target``) is a term.

    Parameters
    ----------
    df : DataFrame with the document and term columns.
    doc_col : str
        Document id column (default ``metis_id``).
    term_col : str
        Term column (default canonical gene column ``target``).
    drop_unassigned : bool
        If True, rows with ``doc_col < 0`` (unassigned transcripts) are dropped.

    Returns
    -------
    dt_matrix : csr_matrix (n_docs x n_terms) of integer counts
    doc_ids   : ndarray mapping matrix row -> original metis_id
    term_ids  : ndarray mapping matrix col -> gene name
    df_to_doc : Series mapping df.index -> matrix row (doc index)
    """
    if drop_unassigned:
        mask = df[doc_col] >= 0
        df_use = df.loc[mask, [doc_col, term_col]]
    else:
        df_use = df[[doc_col, term_col]]

    doc_ids, doc_codes = np.unique(df_use[doc_col].to_numpy(), return_inverse=True)
    term_ids, term_codes = np.unique(df_use[term_col].to_numpy(), return_inverse=True)

    n_docs, n_terms = doc_ids.size, term_ids.size
    counts = np.ones(len(df_use), dtype=np.int32)
    dt_matrix = sp.coo_matrix(
        (counts, (doc_codes, term_codes)), shape=(n_docs, n_terms)).tocsr()

    df_to_doc = pd.Series(doc_codes, index=df_use.index, name="doc_row")

    console.info(f"Built corpus: {n_docs} METIS documents x {n_terms} genes; "
          f"total transcripts (nnz) = {dt_matrix.nnz}")
    return dt_matrix, doc_ids, term_ids, df_to_doc


def filter_stopwords(dt_matrix, term_ids, min_doc_freq=None, max_doc_freq=None):
    """Filter genes by their document frequency (fixed-threshold variant).

    ``min_doc_freq`` removes genes expressed in too few documents (rare/noise);
    ``max_doc_freq`` removes near-ubiquitous genes (stop words). Both thresholds
    are fractions of documents in ``[0, 1]``; either may be None.

    Returns
    -------
    dt_filt, term_filt, doc_freq
    """
    n_docs = dt_matrix.shape[0]
    doc_freq = np.asarray((dt_matrix > 0).sum(axis=0)).ravel() / n_docs

    keep = np.ones(term_ids.size, dtype=bool)
    if min_doc_freq is not None:
        keep &= doc_freq >= min_doc_freq
    if max_doc_freq is not None:
        keep &= doc_freq <= max_doc_freq

    n_kept = int(keep.sum())
    n_removed = int((~keep).sum())
    dt_filt = dt_matrix[:, keep]
    term_filt = term_ids[keep]
    console.info(f"Stop-word filter: kept {n_kept}/{term_ids.size} genes "
          f"(removed {n_removed}; min_df={min_doc_freq}, max_df={max_doc_freq})")
    return dt_filt, term_filt, doc_freq


_MAD_SPEC_RE = re.compile(r"^\s*(\d*\.?\d+)?\s*mad\s*$", re.IGNORECASE)


def parse_stopwords_thr(stopwords_thr):
    """Parse the ``stopwords_thr`` spec into ``(mode, value)``.

    ``stopwords_thr`` accepts:

    * a **float** (or numeric string) -> a *custom* document-frequency cutoff;
      returns ``("fixed", value)`` with ``value`` in ``(0, 1]``;
    * a **``"XMAD"`` string** (e.g. ``"3MAD"``, ``"2.5mad"``, ``"MAD"`` = 1x)
      -> an *automatic* ``median + X*MAD`` threshold; returns ``("mad", X)``;
    * a **list** (or tuple, set, array) of gene names -> exactly those genes
      are stop words; returns ``("list", frozenset(names))``. Useful to reuse
      the stop words of another run, or to set them by hand.

    Raises
    ------
    ValueError
        If the spec is neither a ``XMAD`` string, a document frequency in
        ``(0, 1]``, nor a collection of gene names.
    """
    if isinstance(stopwords_thr, (list, tuple, set, frozenset, np.ndarray, pd.Index, pd.Series)):
        names = [str(g) for g in np.asarray(list(stopwords_thr), dtype=object).ravel()]
        return "list", frozenset(names)
    if isinstance(stopwords_thr, str):
        m = _MAD_SPEC_RE.match(stopwords_thr)
        if m is not None:
            # The regex only matches unsigned numbers, so the coefficient is
            # non-negative by construction ("-1MAD" falls through to the float
            # branch below and raises).
            return "mad", float(m.group(1)) if m.group(1) is not None else 1.0
        try:
            value = float(stopwords_thr)
        except ValueError:
            raise ValueError(
                "stopwords_thr must be a document-frequency float in (0, 1] or "
                f"a 'XMAD' string (e.g. '3MAD'); got {stopwords_thr!r}") from None
    else:
        value = float(stopwords_thr)

    if not 0 < value <= 1:
        raise ValueError(
            "A custom stopwords_thr is a document-frequency fraction and must "
            f"lie in (0, 1]; got {value!r}")
    return "fixed", value


def auto_detect_stopwords(dt_matrix, term_ids, stopwords_thr="2MAD",
                          min_docfreq_for_stats=0.02):
    """Automatic stop-word detection on per-gene **document frequency**.

    A gene's document frequency ``DF_g`` is the fraction of METIS pseudo-cells
    in which it is detected. Stop words are the near-ubiquitous genes (technical
    noise / housekeeping background): a gene is flagged when

        ``DF_g >= threshold``

    where ``threshold`` comes from ``stopwords_thr`` (see
    :func:`parse_stopwords_thr`) — either a custom document-frequency cutoff, or
    the automatic robust cutoff ``median(DF) + X * MAD(DF)``.

    The robust location/scale are estimated **only over "sufficiently frequent"
    genes** — those with ``doc_freq >= min_docfreq_for_stats`` — so that the long
    tail of near-absent genes does not drag the median down and over-flag (U9).
    *All* genes are still scored against the resulting threshold; the
    low-doc-frequency genes are merely excluded from *estimating* it. If too few
    genes (< 10) survive the cutoff, the statistics fall back to using all genes
    and a warning is emitted.

    Genes are only ever ``"kept"`` or ``"stopword"``: there is no rare-gene
    floor, low-frequency genes are never dropped here. Per-gene entropy /
    information content are still computed, but purely as diagnostics for the
    stop-word scatter (4b) — they take no part in the decision.

    Fully vectorized (``O(nnz)`` via scipy.sparse + ``np.bincount``); no dense
    materialisation.

    Parameters
    ----------
    dt_matrix : csr_matrix (n_docs x n_terms) of counts.
    term_ids : iterable of gene names aligned to columns.
    stopwords_thr : float, str or list of str
        Custom document-frequency cutoff, ``"XMAD"`` for the automatic
        ``median + X*MAD`` rule (default ``"2MAD"``), or an explicit list of
        gene names (listed genes absent from the corpus are reported and
        ignored).
    min_docfreq_for_stats : float
        Fraction-of-documents cutoff; only genes at or above it enter the
        median/MAD estimation (ignored for a custom threshold or a list).

    Returns
    -------
    keep_mask : bool ndarray of length n_terms
    stats     : DataFrame indexed by gene name with df_count, doc_freq, entropy,
                info_content, used_for_stats (bool), kept (bool), reason (str)
    info      : dict describing the applied threshold (``mode``, ``spec``,
                ``threshold`` (NaN for a list), for MAD mode ``coef``,
                ``median``, ``mad``, ``n_stats``, for a list ``n_requested``,
                ``n_found`` and ``missing``)
    """
    mode, value = parse_stopwords_thr(stopwords_thr)

    n_docs = dt_matrix.shape[0]
    if not sp.issparse(dt_matrix):
        dt_matrix = sp.csr_matrix(dt_matrix)

    # Document frequency — the detection statistic.
    df_counts = np.asarray((dt_matrix > 0).sum(axis=0)).ravel()
    doc_freq = df_counts / max(n_docs, 1)

    # Per-gene Shannon entropy across documents (vectorized via COO + bincount).
    # Diagnostic only: it is the y-axis of the 4b scatter, not a criterion.
    col_sums = np.asarray(dt_matrix.sum(axis=0)).ravel()
    safe_sums = np.where(col_sums > 0, col_sums, 1.0)
    coo = dt_matrix.tocoo()
    p = coo.data / safe_sums[coo.col]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        contrib = -p * np.log(p)
    H = np.bincount(coo.col, weights=contrib, minlength=dt_matrix.shape[1])
    H_norm = H / np.log(max(n_docs, 2))            # normalize to [0, 1]
    info_content = 1.0 - H_norm

    info = {"mode": mode, "spec": stopwords_thr}
    if mode == "list":
        # Explicit list: no threshold and no statistics; listed genes absent
        # from the corpus are reported and ignored.
        used_for_stats = np.zeros(len(doc_freq), dtype=bool)
        names = np.asarray([str(t) for t in term_ids], dtype=object)
        is_listed = np.isin(names, list(value))
        missing = sorted(value - set(names[is_listed]))
        if missing:
            warnings.warn(f"{len(missing)} of {len(value)} listed stop words are "
                          f"not in the corpus and are ignored: {', '.join(missing[:20])}"
                          + (" ..." if len(missing) > 20 else ""))
        info.update(n_requested=len(value), n_found=int(is_listed.sum()),
                    missing=missing, spec=sorted(value))
        df_threshold = float("nan")
    elif mode == "mad":
        # Estimate median/MAD only over "sufficiently frequent" genes so that the
        # near-absent tail does not distort the robust location/scale (U9).
        used_for_stats = doc_freq >= min_docfreq_for_stats
        if int(used_for_stats.sum()) < 10:
            warnings.warn(
                f"Only {int(used_for_stats.sum())} genes have doc_freq >= "
                f"{min_docfreq_for_stats}; falling back to all genes for the "
                "stop-word median/MAD statistics.")
            used_for_stats = np.ones_like(used_for_stats, dtype=bool)
        df_stat_pool = doc_freq[used_for_stats]

        df_med = float(np.median(df_stat_pool))
        df_mad = float(median_abs_deviation(df_stat_pool))
        df_threshold = df_med + value * df_mad
        info.update(coef=value, median=df_med, mad=df_mad,
                    n_stats=int(used_for_stats.sum()))
    else:
        # Custom cutoff: no statistics are estimated, so no gene is "used".
        used_for_stats = np.zeros(len(doc_freq), dtype=bool)
        df_threshold = value
    info["threshold"] = float(df_threshold)

    is_stopword = is_listed if mode == "list" else doc_freq >= df_threshold
    keep_mask = ~is_stopword

    reason = np.full(len(term_ids), "kept", dtype=object)
    reason[is_stopword] = "stopword"

    stats = pd.DataFrame({
        "df_count": df_counts,
        "doc_freq": doc_freq,
        "entropy": H_norm,
        "info_content": info_content,
        "used_for_stats": used_for_stats,
        "kept": keep_mask,
        "reason": reason,
    }, index=[str(t) for t in term_ids])

    if mode == "mad":
        detail = (f"doc-freq threshold = {df_threshold:.4f} "
                  f"(median {df_med:.4f} + {value:g}*MAD {df_mad:.4f}; "
                  f"stats over {int(used_for_stats.sum())} genes with doc_freq "
                  f">= {min_docfreq_for_stats})")
    elif mode == "list":
        detail = (f"explicit list: {info['n_found']} of {info['n_requested']} "
                  "listed genes found in the corpus")
    else:
        detail = f"custom doc-freq threshold = {df_threshold:.4f}"
    console.info(f"auto_detect_stopwords: kept {int(keep_mask.sum())}/"
                 f"{len(term_ids)} genes "
                 f"(stopwords: {int(is_stopword.sum())}; {detail})")

    return keep_mask, stats, info


def _corpus_chunks(dt_matrix, rows_per_chunk):
    """Yield ``(start, corpus_list)`` for row blocks of a doc-term matrix.

    ``Sparse2Corpus`` reads documents as *columns*, so each block is
    transposed before conversion. Chunking exists so the full corpus is never
    materialised at once: a gensim corpus is a Python list of
    ``(term_id, count)`` tuple lists, which at ~2M documents costs tens of GB
    of interpreter objects — the dominant memory term in this step.
    """
    n_docs = dt_matrix.shape[0]
    for start in range(0, n_docs, rows_per_chunk):
        block = dt_matrix[start:start + rows_per_chunk]
        yield start, list(Sparse2Corpus(block.T))


def _infer_doc_topics(lda_model, dt_matrix, rows_per_chunk):
    """Per-document topic mixtures for **every** row of ``dt_matrix``.

    A single variational E-step per document, run in row blocks. Separated
    from training so that a capped fit (``max_train_docs``) still produces a
    full-corpus ``doc_topics``: downstream consumers index it per pseudo-cell,
    so a truncated version would silently misalign with ``doc_ids``.
    """
    n_docs, n_topics = dt_matrix.shape[0], lda_model.num_topics
    doc_topics = np.empty((n_docs, n_topics), dtype=np.float32)
    for start, chunk in _corpus_chunks(dt_matrix, rows_per_chunk):
        gamma, _ = lda_model.inference(chunk)
        doc_topics[start:start + len(chunk)] = (
            gamma / gamma.sum(axis=1, keepdims=True))
    return doc_topics


def _sample_train_rows(n_docs, max_train_docs, random_state):
    """Row indices to train on: all of them, or a sorted uniform sample.

    Uniform (not stratified) on purpose. METIS pseudo-cells are built to a
    target transcript count, so they are near-equal in size and a uniform
    sample over documents is already close to uniform over transcripts and
    over space — including across the decorrelated tiles of a size-padded
    dataset. Sorted so the sampled corpus keeps corpus order, which keeps the
    fit reproducible under a fixed ``random_state``.
    """
    if not max_train_docs or max_train_docs <= 0 or n_docs <= max_train_docs:
        return None
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_docs, size=int(max_train_docs), replace=False))


#: ``max_train_docs="auto"``: expected tokens the 5th-percentile gene must get
#: in the training sample. Calibrated on the real CosMx 100M corpus
#: (``data/package_tests_dev/old/LDA_PERFORMANCE_FINDINGS.md``, Resolution): at ~3000 tokens the
#: capped model keeps ~97% of the full-corpus seed-to-seed stability of the
#: gene geometry F2 consumes.
AUTO_CAP_GENE_TOKENS = 3000
AUTO_CAP_GENE_QUANTILE = 5

#: ``chunksize="auto"``: online updates per pass. Held-out perplexity is best at
#: ~50 on the 1M, 10M and 100M real corpora; a fixed 2000-document chunk means
#: ~9 updates per pass at 1M (under-fitted) and ~900 at 100M (noisy).
AUTO_UPDATES_PER_PASS = 50


def _auto_train_docs(dt_matrix):
    """Data-derived training cap for ``max_train_docs="auto"``.

    What the pipeline consumes is each gene's topic row, whose precision is
    set by how many tokens of that gene the training sample holds. Returns the
    smallest document count at which the ``AUTO_CAP_GENE_QUANTILE``-th
    percentile gene expects ``AUTO_CAP_GENE_TOKENS`` tokens, or ``None`` (no
    cap) if the full corpus does not reach that. O(n_terms) from column sums.
    """
    n_docs = dt_matrix.shape[0]
    tokens = np.asarray(dt_matrix.sum(axis=0)).ravel()
    tokens = tokens[tokens > 0]
    if n_docs == 0 or tokens.size == 0:
        return None
    q = float(np.percentile(tokens, AUTO_CAP_GENE_QUANTILE))
    n_star = int(np.ceil(n_docs * AUTO_CAP_GENE_TOKENS / q))
    return None if n_star >= n_docs else n_star


def _resolve_chunksize(chunksize, n_train):
    """Documents per online update: an int, or ``"auto"`` = n_train / 50."""
    if chunksize == "auto":
        return max(1, int(round(n_train / AUTO_UPDATES_PER_PASS)))
    return int(chunksize)


def _fit_vb(dt_train, n_topics, n_jobs, chunksize, passes, eta, random_state):
    """Online VB LDA via :class:`EDERA.lda_vb.OnlineLDA`, set up like gensim.

    Settings follow gensim's ``LdaModel`` defaults so the two backends fit the
    same model: symmetric ``1/K`` priors, ``chunksize`` documents per update,
    ``passes`` epochs, at most 50 E-step iterations per document with a
    ``1e-3`` mean-change tolerance, and the step size ``(1 + t)^-0.5``
    (gensim's ``offset=1, decay=0.5``), on gensim's schedule: the first
    update fully replaces the random initial topics and the step also decays
    with the pass count. With sklearn's schedule instead, held-out perplexity
    is unchanged but ARI against ground truth on the simulated TME dataset
    dropped from 0.46 to 0.36-0.41 (3 seeds; gensim itself: 0.48).
    """
    from .lda_vb import OnlineLDA

    if eta == "symmetric":
        topic_word_prior = 1.0 / n_topics
    else:
        try:
            topic_word_prior = float(eta)
        except (TypeError, ValueError):
            raise ValueError(
                "The 'vb' LDA backend supports eta='symmetric' or a float; "
                f"got {eta!r}. Use lda_backend='gensim' for other eta modes."
            ) from None
    model = OnlineLDA(
        n_components=n_topics,
        doc_topic_prior=1.0 / n_topics,
        topic_word_prior=topic_word_prior,
        learning_decay=0.5,
        learning_offset=1.0,
        batch_size=chunksize,
        max_iter=passes,
        max_doc_update_iter=50,
        mean_change_tol=1e-3,
        n_jobs=n_jobs,
        random_state=random_state,
        schedule="gensim",
    )
    return model.fit(dt_train)


def compute_lda(dt_matrix, term_names, n_topics=30, n_jobs=1, chunksize="auto",
                passes=10, eta="symmetric", random_state=42,
                max_train_docs=None, backend="vb"):
    """Train an LDA topic model on a sparse doc-term matrix.

    Backends (``backend``):

    * ``"vb"`` (default) — :class:`EDERA.lda_vb.OnlineLDA`, online VB with
      the per-document E-step compiled by numba and run on ``n_jobs``
      threads. **Bit-identical at every ``n_jobs``**: the sufficient
      statistics are reduced in document order, whatever the scheduling.
    * ``"gensim"`` — the previous implementation, selected by worker count:
      ``LdaModel`` (single process, reproducible) for one worker,
      ``LdaMulticore`` (not reproducible) for more.

    ``LdaMulticore`` pushes chunks onto an async job queue and merges worker
    results in *arrival* order, so its output follows process timing rather
    than ``random_state``. That makes it non-deterministic **even at
    ``workers=1``**, as soon as the corpus exceeds one chunk per pass
    (``n_docs > chunksize``); a single-chunk corpus hides the bug entirely.
    See ``data/package_tests_dev/old/PROFILING_FINDINGS.md`` §8 for the three-process
    measurement that pins this down. ``LdaModel`` on the same input reproduces
    bit-identically and also sheds ``LdaMulticore``'s IPC overhead.

    Parameters
    ----------
    dt_matrix : csr_matrix (n_docs x n_terms)
    term_names : iterable of gene names aligned to columns of ``dt_matrix``
    n_topics : int
        Number of LDA topics to fit.
    n_jobs : int
        Worker count for this step. ``"vb"``: E-step threads (does not change
        the result). gensim: ``1`` selects the reproducible ``LdaModel``;
        ``>2`` selects ``LdaMulticore`` with ``n_jobs - 1`` workers and gives
        up reproducibility (the pipeline passes 1 for gensim).
    chunksize : int or "auto"
        Documents per online update. ``"auto"`` (default) uses
        ``n_train / 50``, i.e. 50 updates per pass, which minimised held-out
        perplexity on the 1M-100M real corpora.
    passes, eta : online-VB epochs and topic-word prior, shared by both
        backends.
    random_state : int
        Reproducibility seed.
    max_train_docs : int, "auto" or None
        Cap on documents used for **training**; ``None``/``<=0`` (default)
        fits everything. Above an int cap, a uniform ``random_state``-seeded
        sample is fitted. ``"auto"`` derives the cap from the data (see
        :func:`_auto_train_docs`) and resolves to no cap when the corpus is
        not large enough to afford one. Fitting everything is the default
        because it is cheap with the ``"vb"`` backend (~40 s for the 1.8M
        document 100M corpus on 16 threads) and still measurably improves the
        gene embedding: its seed-to-seed stability keeps rising with the
        training size up to the full corpus. ``doc_topics`` is **always**
        returned for every row of ``dt_matrix``, capped or not.
    backend : {"vb", "gensim"}

    Returns
    -------
    lda_model   : :class:`EDERA.lda_vb.OnlineLDA`, or gensim ``LdaModel``
                  (or ``LdaMulticore``, which subclasses it). Its
                  ``edera_fit_info`` dict records the resolved backend,
                  threads, training-document count, cap and batch size.
    doc_topics  : ndarray (n_docs x n_topics), per-document topic mixture,
                  covering every document even when training was capped
    gene_emb_df : DataFrame (n_terms x n_topics) of gene-topic weights, with a
                  leading ``gene_name`` column.
    """
    if not sp.issparse(dt_matrix):
        dt_matrix = sp.csr_matrix(dt_matrix)
    dt_matrix = dt_matrix.tocsr()          # row slicing for the cap / chunking

    n_docs_total = dt_matrix.shape[0]
    cap_rule = "fixed"
    if max_train_docs == "auto":
        cap_rule = "auto"
        max_train_docs = _auto_train_docs(dt_matrix)
    train_rows = _sample_train_rows(n_docs_total, max_train_docs, random_state)
    dt_train = dt_matrix if train_rows is None else dt_matrix[train_rows]

    if train_rows is not None:
        # A gene absent from the sample would keep only its eta prior and hand
        # every transcript of that gene a flat topic vector, so report the
        # sample's vocabulary coverage rather than assume it.
        n_absent = int((np.asarray((dt_train > 0).sum(axis=0)).ravel() == 0).sum())
        console.info(
            f"LDA training cap: fitting on {dt_train.shape[0]:,} of "
            f"{n_docs_total:,} documents (uniform sample, seed {random_state}); "
            f"per-document topics still inferred for all {n_docs_total:,}")
        if n_absent:
            warnings.warn(
                f"{n_absent} of {dt_matrix.shape[1]} genes have zero counts in "
                f"the {dt_train.shape[0]:,}-document LDA training sample and "
                "will carry only their eta prior. Raise "
                "advanced.lda_max_train_docs if these genes matter.")

    n_train = dt_train.shape[0]
    chunksize = _resolve_chunksize(chunksize, n_train)
    fit_info = dict(backend=backend, n_jobs=int(n_jobs), n_docs_total=n_docs_total,
                    n_docs_trained=int(n_train), max_train_docs=max_train_docs,
                    cap_rule=cap_rule if train_rows is not None else "none",
                    batch_size=chunksize, passes=int(passes))

    if backend == "vb":
        console.info(f"Training LDA with K={n_topics} topics on "
                     f"{dt_train.shape[0]:,} documents x {dt_train.shape[1]} "
                     f"terms using online VB (numba, {n_jobs} threads, "
                     f"{passes} passes, batch {chunksize})...")
        t0 = time.time()
        lda_model = _fit_vb(dt_train, n_topics, n_jobs, chunksize, passes,
                            eta, random_state)
        console.info(f"LDA fitted in {time.time() - t0:.1f}s")
        del dt_train

        # components_ is the variational lambda; normalising its rows gives
        # the same topic-word distribution as gensim's get_topics().
        comp = lda_model.components_
        word_topic_matrix = (comp / comp.sum(axis=1, keepdims=True)).T
        gene_emb_df = pd.DataFrame(
            word_topic_matrix, columns=[f"LDA_{i}" for i in range(n_topics)])
        gene_emb_df.insert(0, "gene_name", [str(t) for t in term_names])

        # Full-corpus inference, threaded and deterministic (constant init).
        t0 = time.time()
        doc_topics = lda_model.transform(dt_matrix).astype(np.float32)
        console.info(f"Inferred topics for {doc_topics.shape[0]:,} documents "
                     f"in {time.time() - t0:.1f}s")
        lda_model.edera_fit_info = fit_info
        return lda_model, doc_topics, gene_emb_df
    if backend != "gensim":
        raise ValueError(
            f"backend must be 'vb' or 'gensim'; got {backend!r}")

    # Materialize once so the corpus is cheap to re-iterate across LDA passes.
    # gensim Sparse2Corpus reads documents as columns -> transpose.
    corpus = list(Sparse2Corpus(dt_train.T))
    id2word = dict(enumerate([str(t) for t in term_names]))

    # gensim LdaMulticore rejects -1 and internally spawns `workers` processes;
    # it recommends workers = n_cores - 1. n_jobs is already a resolved positive
    # int here, so subtract one (floored at 1).
    lda_workers = max(1, int(n_jobs) - 1)
    backend = "LdaModel (single-process, reproducible)" if lda_workers == 1 \
        else f"LdaMulticore ({lda_workers} workers, NOT reproducible)"
    console.info(f"Training LDA with K={n_topics} topics on "
          f"{dt_train.shape[0]} documents x {dt_train.shape[1]} terms "
          f"using {backend} ({passes} passes)...")
    t0 = time.time()
    if lda_workers == 1:
        # Single-process path. LdaMulticore would merge its async chunk results
        # in arrival order and wobble run to run even here; LdaModel does not.
        lda_model = models.LdaModel(
            corpus=corpus,
            num_topics=n_topics,
            id2word=id2word,
            chunksize=chunksize,
            passes=passes,
            random_state=random_state,
            eta=eta,
        )
    else:
        lda_model = models.LdaMulticore(
            corpus=corpus,
            num_topics=n_topics,
            id2word=id2word,
            workers=lda_workers,
            chunksize=chunksize,
            passes=passes,
            random_state=random_state,
            eta=eta,
        )
    console.info(f"LDA fitted in {time.time() - t0:.1f}s")

    # Gene-topic matrix: (n_terms, n_topics). This is the only LDA output the
    # pipeline consumes downstream (merged onto transcripts by gene in
    # compute_tx_topics step 9), and the only one the training cap affects.
    word_topic_matrix = lda_model.get_topics().T
    gene_emb_df = pd.DataFrame(
        word_topic_matrix, columns=[f"LDA_{i}" for i in range(n_topics)])
    gene_emb_df.insert(0, "gene_name", [str(t) for t in term_names])

    # Per-document topic distribution via variational inference, over the FULL
    # corpus regardless of the training cap: doc_topics is indexed by
    # pseudo-cell alongside doc_ids, so it must not be a subsample.
    del corpus                             # free the training corpus first
    t0 = time.time()
    doc_topics = _infer_doc_topics(lda_model, dt_matrix, max(chunksize, 1))
    console.info(f"Inferred topics for {doc_topics.shape[0]:,} documents in "
                 f"{time.time() - t0:.1f}s")

    lda_model.edera_fit_info = fit_info
    return lda_model, doc_topics, gene_emb_df


def top_genes_per_topic(gene_emb_df, n_top=30):
    """Return a tidy table of the top ``n_top`` genes per LDA topic."""
    lda_cols = [c for c in gene_emb_df.columns if c.startswith("LDA_")]
    return pd.DataFrame({
        "topic": lda_cols,
        "top_genes": [
            ", ".join(gene_emb_df.sort_values(col, ascending=False)
                      ["gene_name"].head(n_top))
            for col in lda_cols
        ],
    })
