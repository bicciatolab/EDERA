"""
EDERA.diffusion
=================

Spatial diffusion of per-transcript LDA topic vectors by Random Walk with
Restart (RWR) over each FOV's FSS graph.

The RWR fixed-point iteration is unchanged from the original script::

    X^(t+1) = c * P * X^(t) + (1 - c) * X^(0)
    P = D^(-1) * A_FSS,  c = damping_factor_rwr

where ``c`` is ``damping_factor_rwr`` (restart probability ``1 - c``). The
``beta`` sharpening exponent that the original worker accepted (and that was
hardwired to 1.0, i.e. inert) has been removed (issue I4).

Parallelism uses **exactly** ``n_workers`` outer processes (issue I2); the
``getattr(globals(), 'n_workers', ...)`` lookup of the original is gone.
"""

import numpy as np
from scipy.sparse import diags
from joblib import Parallel, delayed

from .console import console
from .utils import fov_row_positions, EMPTY_POSITIONS


# Topic matrices are stored as float32 (half the memory of float64); the RWR
# iteration itself always runs in float64 inside each worker. Iterating in
# float32 would be unsafe: the convergence test is an absolute Frobenius norm
# over the whole FOV chunk, and float32 rounding noise on a multi-million-row
# chunk is itself ~1e-5, so large FOVs could fail to converge and run to
# ``max_steps``.
TOPIC_DTYPE = np.float32


def _diffuse_fov_chunk(fov_id, P_mat, X_0, damping_factor_rwr, alpha, tolerance,
                       max_steps):
    """Diffuse LDA topics over a single FOV graph by RWR (worker function).

    Parameters
    ----------
    fov_id : hashable
        FOV identifier (passed through for re-assembly).
    P_mat : csr_matrix
        The FOV's (unnormalised) FSS adjacency; row-normalised internally.
    X_0 : ndarray (n_nodes x n_topics)
        Initial per-transcript topic vectors (the restart distribution). Any
        float dtype; the walk runs in float64 and the result is returned in
        ``X_0``'s dtype.
    damping_factor_rwr : float
        Propagation weight ``c``.
    alpha : float
        Restart weight ``1 - c``.
    tolerance : float
        Frobenius-norm convergence tolerance.
    max_steps : int
        Maximum iterations.

    Returns
    -------
    fov_id, diffused_X, n_steps
    """
    out_dtype = X_0.dtype
    X_0 = np.asarray(X_0, dtype=np.float64)
    # Row-stochastic normalization: P = D^(-1) A.
    row_sums = np.array(P_mat.sum(axis=1), dtype=np.float64).flatten()
    row_sums[row_sums == 0] = 1e-10
    D_inv = diags(1.0 / row_sums)
    P = D_inv @ P_mat

    current_lda = X_0.copy()

    # RWR iterator.
    step = 0
    for step in range(max_steps):
        next_lda = damping_factor_rwr * P.dot(current_lda) + alpha * X_0
        diff = np.linalg.norm(next_lda - current_lda, ord="fro")
        if diff < tolerance:
            current_lda = next_lda
            break
        current_lda = next_lda

    return fov_id, current_lda.astype(out_dtype, copy=False), step + 1


def diffuse_topic_matrix(X, rows, fss_list, fss_splits, damping_factor_rwr,
                         tolerance=1e-5, max_steps=1000, n_jobs=1):
    """Diffuse a topic matrix **in place**, FOV by FOV, in parallel.

    Parameters
    ----------
    X : ndarray (n x K), writeable
        Restart topic vectors (one row per transcript), overwritten with the
        diffused values. Its dtype is kept (float32 in the pipeline).
    rows : dict
        ``{fov_id: int64 row positions}`` (:func:`EDERA.utils.fov_row_positions`).
    fss_list, fss_splits, damping_factor_rwr, tolerance, max_steps, n_jobs :
        as :func:`run_rwr_diffusion`.

    Returns
    -------
    X : the same array.

    Each FOV's restart rows are gathered when its task is dispatched and its
    result is written back as soon as it completes. FOV row sets are disjoint,
    so overwriting a finished FOV never touches rows of a FOV still waiting to
    be dispatched; no second full-size matrix is ever held.
    """
    alpha = 1 - damping_factor_rwr

    def tasks():
        for i, split_val in enumerate(fss_splits):
            X_0 = np.ascontiguousarray(X[rows.get(split_val, EMPTY_POSITIONS)])
            yield split_val, fss_list[i], X_0

    results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(_diffuse_fov_chunk)(
            fov_id, P_mat, X_0, damping_factor_rwr, alpha, tolerance, max_steps
        ) for fov_id, P_mat, X_0 in tasks()
    )
    for fov_id, diffused_lda, steps in console.pbar(results, total=len(fss_splits),
                                                    desc="Diffusing FOVs"):
        X[rows.get(fov_id, EMPTY_POSITIONS)] = diffused_lda
    return X


def run_rwr_diffusion(df, fss_list, fss_splits, fov_column, lda_topics,
                      damping_factor_rwr, tolerance=1e-5, max_steps=1000,
                      n_jobs=1):
    """Diffuse LDA topic columns of ``df`` across all FOVs in parallel.

    DataFrame wrapper around :func:`diffuse_topic_matrix` (kept for callers
    that hold the topics as columns). Mutates ``df`` in place: the
    ``lda_topics`` columns are replaced by their diffused values, stored as
    float32. ``*_raw`` copies must be created by the caller *before* calling
    this function if a before/after comparison is needed.

    Parameters
    ----------
    df : DataFrame with the ``lda_topics`` columns and ``fov_column``.
    fss_list : list of csr_matrix aligned with ``fss_splits``.
    fss_splits : list of FOV identifiers.
    fov_column : str
    lda_topics : list of str
        Names of the topic columns to diffuse.
    damping_factor_rwr : float
        Propagation weight ``c``.
    tolerance, max_steps : RWR convergence controls.
    n_jobs : int
        Outer worker processes (exactly ``n_workers``).

    Returns
    -------
    df : the same DataFrame, with diffused topic columns.
    """
    rows = fov_row_positions(df, fov_column)
    X = np.array(df[lda_topics].to_numpy(), dtype=TOPIC_DTYPE)
    diffuse_topic_matrix(X, rows, fss_list, fss_splits, damping_factor_rwr,
                         tolerance=tolerance, max_steps=max_steps, n_jobs=n_jobs)
    df[lda_topics] = X
    console.info("Parallel RWR complete. LDA scores successfully diffused.")
    return df
