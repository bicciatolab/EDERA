"""
EDERA.cluster
===============

Clustering of the diffused LDA topic space and spatial label smoothing
(Function 2 of the pipeline):

1. draw a leverage-score subsample of the transcripts, with per-FOV quotas
   (:func:`sample_leverage_by_fov`);
2. build an HNSW topic graph on the subsample, mix in the FSS spatial graph,
   symmetrize it (:func:`symmetrize_graph`) and run scikit-network's Leiden per
   resolution (:func:`leiden_sknetwork`);
3. merge Leiden clusters that are fragments of one cell type with the
   size-aware marker merge (:func:`merge_clusters_marker`, on by default), then
   optionally merge clusters smaller than a fraction of the data (disabled by
   default, ``min_ratio_merge_small_clusters = 0``);
4. transfer labels to the unseen majority by HNSW majority vote;
5. iteratively smooth labels per FOV by FSS weighted-consensus voting.

Steps 1 and 2 were chosen by the F2 clustering benchmark
(``data/package_tests_dev/f2_bench``): on simulated tissue with planted rare cell types,
leverage sampling recovered rare types better than FOV-stratified uniform and
geometric-sketch sampling, and scikit-network's Leiden matched igraph's quality
in a fraction of the time (igraph run to convergence was 14-18x slower).

Changes vs. the original script:

* Multi-resolution output: each Leiden resolution produces its own column,
  named with a float-safe key ``final_cluster_res<r>`` (issue I1). The original
  ``'final_cluster_' + res`` string concatenation (which raised on float ``res``)
  is gone, and downstream code no longer assumes a single ``final_cluster``.
* Parallelism uses **exactly** ``n_workers`` (issue I2); the
  ``getattr(globals(), 'n_workers', ...)`` lookups are gone.
* The gene label is read from the canonical ``target`` column, not the
  merge-side-effect ``gene_name`` column (issue I7).
* ``random_state`` is threaded into the stratified split (issue I8).
* ``_refine_fov_chunk`` now also returns the per-loop smoothed-fraction history,
  enabling the optional convergence-trace plot.
"""

import numpy as np
import pandas as pd
from scipy.stats import mode
from scipy.sparse import csr_matrix

from .console import console


def _res_key(res):
    """Float-safe resolution key, e.g. 0.25 -> 'res0.25', 1.0 -> 'res1'."""
    return f"res{res:g}"


def build_hnsw_graph(data, k=15, ef_construction=200, M=16, num_threads=1):
    """Build an HNSW index and return a sparse adjacency matrix.

    Uses cosine space (appropriate for LDA topic features). Edge weights are
    ``1 - cosine_distance`` (i.e. cosine similarity).

    Parameters
    ----------
    data : ndarray (n x d)
    k : int
        Number of neighbours queried per node.
    ef_construction : int
        HNSW build-time accuracy parameter.
    M : int
        HNSW number of bi-directional links per element.
    num_threads : int
        Number of threads used for index construction and queries. Pass a
        resolved positive integer (hnswlib does not accept ``-1``).
    """
    import hnswlib
    num_elements, dim = data.shape

    p = hnswlib.Index(space="cosine", dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
    p.set_num_threads(max(1, int(num_threads)))
    p.add_items(data, num_threads=max(1, int(num_threads)))
    p.set_ef(k + 50)  # query-time accuracy

    labels, distances = p.knn_query(data, k=k, num_threads=max(1, int(num_threads)))

    # hnswlib cosine distance = 1 - cosine_similarity -> convert to similarity.
    weights = 1.0 - distances

    rows = np.repeat(np.arange(num_elements), k)
    cols = labels.reshape(-1)
    weights = weights.reshape(-1)

    adj = csr_matrix((weights, (rows, cols)),
                     shape=(num_elements, num_elements))
    return adj


# ---------------------------------------------------------------------------
# Leverage-score sampling (step 1)
# ---------------------------------------------------------------------------
def _quota_per_fov(fov_values, fov_counts, n_target):
    """Per-FOV sample sizes proportional to FOV size, summing exactly to
    ``n_target`` (largest-remainder rounding)."""
    raw = fov_counts * (n_target / fov_counts.sum())
    base = np.floor(raw).astype(int)
    remainder = n_target - base.sum()
    if remainder > 0:
        base[np.argsort(-(raw - base))[:remainder]] += 1
    return dict(zip(fov_values, base))


def _leverage_scores(features):
    """Statistical leverage ``l_i = ||U_i||^2`` from a thin QR of ``features``."""
    Q, _ = np.linalg.qr(np.asarray(features, dtype=np.float64), mode="reduced")
    return np.einsum("ij,ij->i", Q, Q)


def sample_leverage_by_fov(features, fov, fraction, random_state=42):
    """Leverage-score subsample with per-FOV quotas.

    Each FOV contributes a share of the sample proportional to its size, so
    every acquisition region is represented. Within a FOV, transcripts are
    drawn without replacement with probability proportional to their
    statistical leverage in the diffused LDA topic space (the rows of ``Q``
    from a thin QR decomposition), which favours transcripts whose topic
    profile is unusual *within that FOV* — i.e. rare cell types.

    This is the ``leverage`` sampler of the F2 benchmark
    (``data/package_tests_dev/f2_bench/chain/sampling.py``), ported unchanged, so a given
    ``random_state`` draws the same transcripts as the benchmark seed.

    Parameters
    ----------
    features : ndarray (n x K)
        Diffused LDA topic matrix.
    fov : array-like (n,)
        FOV id of every row.
    fraction : float in (0, 1)
        Fraction of rows to sample; the sample size is ``round(fraction * n)``.
    random_state : int

    Returns
    -------
    (sample_indices, unseen_indices) : sorted int64 arrays partitioning
    ``range(n)``.
    """
    fov = np.asarray(fov)
    n = len(fov)
    n_target = int(round(fraction * n))
    rng = np.random.default_rng(random_state)

    fov_values, fov_inverse, fov_counts = np.unique(
        fov, return_inverse=True, return_counts=True)
    quotas = _quota_per_fov(fov_values, fov_counts, n_target)
    # Row positions of every FOV in one pass, in increasing order (the same
    # rows, in the same order, as ``np.flatnonzero(fov == f)``).
    rows_by_fov = np.split(np.argsort(fov_inverse, kind="stable"),
                           np.cumsum(fov_counts)[:-1])

    picked = []
    for f, pool in zip(fov_values, rows_by_fov):
        q = quotas[f]
        if q <= 0:
            continue
        if q >= pool.size:
            picked.append(pool)
            continue
        ell = _leverage_scores(np.asarray(features)[pool])
        s = ell.sum()
        p = ell / s if s > 0 else np.full(pool.size, 1.0 / pool.size)
        picked.append(pool[rng.choice(pool.size, size=q, replace=False, p=p)])
    idx = np.unique(np.concatenate(picked)) if picked else np.empty(0, dtype=np.int64)

    # Top up (or trim) to exactly n_target from the remainder.
    if idx.size > n_target:
        idx = rng.choice(idx, size=n_target, replace=False)
    elif idx.size < n_target:
        rest = np.setdiff1d(np.arange(n), idx, assume_unique=False)
        idx = np.concatenate([idx, rng.choice(rest, size=n_target - idx.size, replace=False)])
    sample_indices = np.sort(idx.astype(np.int64))
    unseen = np.ones(n, dtype=bool)
    unseen[sample_indices] = False
    return sample_indices, np.flatnonzero(unseen).astype(np.int64)


# ---------------------------------------------------------------------------
# Graph symmetrization and Leiden (step 2)
# ---------------------------------------------------------------------------
def symmetrize_graph(adj):
    """Symmetric, self-loop-free float64 CSR of a directed kNN(+FSS) graph.

    ``A_sym = triu(A, 1) + tril(A, -1).T``, mirrored into a full symmetric
    matrix, diagonal dropped. Reciprocal edges are *summed*, which is exactly
    what modularity on the original multigraph would count, with half the
    edges. The self-loops that ``build_hnsw_graph`` puts on every node (each
    point is its own nearest neighbour) are removed.
    """
    import scipy.sparse as sp
    A = adj.tocsr().astype(np.float64)
    upper = (sp.triu(A, k=1) + sp.tril(A, k=-1).transpose()).tocsr()
    A_sym = (upper + upper.transpose()).tocsr()
    A_sym.setdiag(0)
    A_sym.eliminate_zeros()
    return A_sym


def _seed_c_rand(seed):
    """Seed the C library's ``rand()``.

    scikit-network's Leiden (0.33) picks refinement moves with the C
    ``rand()`` and ignores its ``random_state`` argument, so without this two
    fits in the same process return different partitions (ARI as low as 0.59
    on a 370k-node graph).
    """
    import ctypes
    import ctypes.util
    lib = ctypes.util.find_library("c")
    libc = ctypes.CDLL(lib) if lib else ctypes.CDLL(None)
    libc.srand(int(seed) & 0x7FFFFFFF)


def leiden_sknetwork(adjacency, resolution, random_state=42):
    """Leiden community detection with scikit-network.

    Newman modularity on the symmetric graph from :func:`symmetrize_graph`,
    no cap on aggregation levels, default tolerances (tighter ones gave the
    same partitions at 1.6-2.5x the run time in the F2 benchmark). Run time is
    roughly independent of ``resolution``.

    Returns an int ndarray of labels, clusters numbered by decreasing size.
    """
    from sknetwork.clustering import Leiden
    _seed_c_rand(random_state)
    algo = Leiden(resolution=resolution, modularity="newman", n_aggregations=-1,
                  random_state=random_state)
    return np.asarray(algo.fit_predict(adjacency), dtype=int)


def _marker_counts_row(counts, sizes, a, others, fold, floor_min, floor_per_gene):
    """Size-aware marker-gene counts between cluster ``a`` and clusters ``others``.

    A gene is a marker of a pair when it is at least ``fold`` times more frequent
    (pseudocounted frequencies) in one cluster and that cluster holds at least
    ``max(floor_min, floor_per_gene * n_smaller / n_genes)`` of its transcripts.
    """
    n_genes = counts.shape[1]
    fa = (counts[a] + 1.0) / (sizes[a] + n_genes)
    fb = (counts[others] + 1.0) / (sizes[others, None] + n_genes)
    floor = np.maximum(floor_min, floor_per_gene * np.minimum(sizes[a], sizes[others]) / n_genes)[:, None]
    up = (fa[None, :] >= fold * fb) & (counts[a][None, :] >= floor)
    down = (fb >= fold * fa[None, :]) & (counts[others] >= floor)
    return (up.sum(1) + down.sum(1)).astype(np.int64)


def merge_clusters_marker(labels, genes, n_genes, profile_genes, n_profile_genes,
                          max_markers=50, min_similarity=0.55, fold=2.0,
                          floor_min=5.0, floor_per_gene=1.952):
    """Join Leiden clusters that are fragments of one cell type (size-aware marker merge).

    At high resolution Leiden splits abundant cell types *by gene*: each fragment
    is enriched for one anchor gene that has its own LDA topic, while rare and
    contact types only get their own cluster at those resolutions. This merge
    joins the fragments and keeps the distinct clusters.

    Agglomerative, on the clustered (sampled) transcripts: repeatedly join the
    pair of clusters with the **fewest marker genes**, among pairs whose
    pseudo-bulk profiles are similar, while that pair has fewer than
    ``max_markers`` markers.

    * Marker gene of a pair: at least ``fold`` times more frequent in one
      cluster (pseudocounted frequencies), with at least
      ``max(floor_min, floor_per_gene * n_smaller / n_genes)`` transcripts there
      (``n_smaller``: transcripts of the smaller cluster). The floor is relative
      to the mean per-gene count of the smaller cluster, so it scales with both
      cluster size and panel size; at ``n_genes = 976`` it is 0.2 % of the
      smaller cluster. A fixed floor would leave small clusters without markers,
      so that they would merge by default.
    * Similarity gate: Bhattacharyya coefficient of the two pseudo-bulk gene
      profiles (stop-words excluded) at least ``min_similarity``. Positive
      evidence of similarity is required, not only the absence of markers.
    * Merged clusters pool their counts; the pair statistics of the merged
      cluster are recomputed before the next step.

    Validated on simulated scatter and irregular tissue (1M and 10M transcripts)
    and on a Xenium 5K mouse section:
    ``data/package_tests_dev/v3_cluster_improve/merge_verify/REPORT.md``.

    Parameters
    ----------
    labels : int array (n_sampled,)
        Leiden labels of the clustered transcripts.
    genes : int array (n_sampled,)
        Gene code (0 .. n_genes-1) of each clustered transcript, all genes.
    n_genes : int
        Panel size (number of distinct genes).
    profile_genes : int array (n_sampled,)
        Gene code in the pseudo-bulk vocabulary (genes with an LDA embedding,
        i.e. stop-words excluded), -1 for genes outside it.
    n_profile_genes : int
        Size of that vocabulary.
    max_markers, min_similarity, fold, floor_min, floor_per_gene
        Rule parameters (see above).

    Returns
    -------
    merged : int array (n_sampled,)
        Merged labels, renumbered 0 .. k-1 by the smallest original label of
        each group.
    log : DataFrame
        One row per merge step: ``into``, ``absorbed`` (original labels),
        ``markers``, ``similarity``, ``size_into``, ``size_absorbed``.
    """
    lab = np.unique(labels, return_inverse=True)[1]
    k = int(lab.max()) + 1 if len(lab) else 0
    cols = ["step", "into", "absorbed", "markers", "similarity", "size_into", "size_absorbed"]
    if k < 2:
        return lab, pd.DataFrame(columns=cols)
    counts = np.bincount(lab * n_genes + genes, minlength=k * n_genes).reshape(k, n_genes).astype(float)
    use = profile_genes >= 0
    prof = np.bincount(lab[use] * n_profile_genes + profile_genes[use],
                       minlength=k * n_profile_genes).reshape(k, n_profile_genes).astype(float)
    sizes = counts.sum(1)
    markers = np.zeros((k, k), dtype=np.int64)
    all_k = np.arange(k)
    for a in range(k):                       # row by row: bounded memory on large panels
        markers[a] = _marker_counts_row(counts, sizes, a, all_k, fold, floor_min, floor_per_gene)
    sq = np.sqrt(prof / np.maximum(prof.sum(1, keepdims=True), 1e-300))
    sim = sq @ sq.T
    alive = np.ones(k, dtype=bool)
    parent = np.arange(k)
    rows = []
    never = np.iinfo(np.int64).max
    while alive.sum() > 1:
        ok = alive[:, None] & alive[None, :] & (sim >= min_similarity)
        np.fill_diagonal(ok, False)
        if not ok.any():
            break
        cand = np.where(ok, markers, never)
        i, j = np.unravel_index(np.argmin(cand), cand.shape)
        if cand[i, j] >= max_markers:
            break
        i, j = min(i, j), max(i, j)
        rows.append(dict(step=len(rows) + 1, into=int(i), absorbed=int(j), markers=int(cand[i, j]),
                         similarity=float(sim[i, j]), size_into=int(sizes[i]), size_absorbed=int(sizes[j])))
        counts[i] += counts[j]
        prof[i] += prof[j]
        sizes[i] += sizes[j]
        alive[j] = False
        parent[j] = i
        live = np.flatnonzero(alive)
        row = _marker_counts_row(counts, sizes, i, live, fold, floor_min, floor_per_gene)
        markers[i, live] = row
        markers[live, i] = row
        sq[i] = np.sqrt(prof[i] / max(prof[i].sum(), 1e-300))
        srow = sq[live] @ sq[i]
        sim[i, live] = srow
        sim[live, i] = srow
    root = parent.copy()
    for c in range(k):
        r = c
        while root[r] != r:
            r = root[r]
        root[c] = r
    merged = np.unique(root, return_inverse=True)[1][lab]
    return merged.astype(int), pd.DataFrame(rows, columns=cols)


def merge_small_clusters(adata, label_key="leiden", min_ratio=0.03):
    """Merge clusters smaller than ``min_ratio`` of all points into the nearest
    large cluster (by cosine distance between LDA centroids).

    .. note::
       **Alternative / not wired into the pipeline.** The active merge path is
       :func:`merge_small_clusters_paga_style`, which merges by graph
       connectivity rather than geometric centroids. This centroid-based variant
       is retained as a documented alternative (a possible ``merge_strategy``
       switch), not dead code to be called silently.

    Returns
    -------
    labels : ndarray of merged labels.
    """
    labels = adata.obs[label_key].values.copy()
    counts = pd.Series(labels).value_counts(normalize=True)

    orphan_clusters = counts[counts < min_ratio].index
    giant_clusters = counts[counts >= min_ratio].index

    if len(orphan_clusters) == 0:
        return labels

    console.info(f"Merging {len(orphan_clusters)} small clusters into "
          f"{len(giant_clusters)} giants...")

    centroids = {g: adata.X[labels == g].mean(axis=0) for g in giant_clusters}

    for orphan in orphan_clusters:
        orphan_mask = (labels == orphan)
        orphan_centroid = adata.X[orphan_mask].mean(axis=0)

        best_giant = None
        min_dist = np.inf
        for g, g_centroid in centroids.items():
            dist = 1 - (np.dot(orphan_centroid, g_centroid)
                        / (np.linalg.norm(orphan_centroid)
                           * np.linalg.norm(g_centroid)))
            if dist < min_dist:
                min_dist = dist
                best_giant = g

        labels[orphan_mask] = best_giant

    return labels
# ***

def merge_small_clusters_paga_style(adata, adj_matrix, label_key="leiden", min_ratio=0.03):
    """Merge small clusters into the most connected giant cluster.

    Instead of comparing geometric centroids, this uses the HNSW adjacency matrix 
    to measure actual graph connectivity (sum of edge weights) between an orphan 
    cluster and the giant clusters. This mimics PAGA by respecting the true 
    manifold topology of the data.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    adj_matrix : scipy.sparse.csr_matrix
        The HNSW or spatial adjacency graph (N x N).
    label_key : str
        The column in adata.obs containing the current cluster labels.
    min_ratio : float
        The minimum fraction of total points required to survive as a giant.

    Returns
    -------
    labels : ndarray of merged labels.
    """
    labels = adata.obs[label_key].values.copy()
    counts = pd.Series(labels).value_counts(normalize=True)

    orphan_clusters = counts[counts < min_ratio].index
    giant_clusters = counts[counts >= min_ratio].index

    if len(orphan_clusters) == 0:
        return labels

    if len(giant_clusters) == 0:
        # No cluster reaches min_ratio: every orphan hits the disconnected
        # fallback below and the whole space collapses onto the largest cluster.
        # That is almost never intended, so say so instead of failing silently.
        console.warn(
            f"merge_small_clusters_paga_style: no cluster reaches "
            f"min_ratio={min_ratio:g} (largest is {counts.iloc[0]:.3f} of the "
            f"data), so all {len(orphan_clusters)} clusters collapse into "
            f"'{counts.index[0]}'. Lower min_ratio_merge_small_clusters or the "
            "Leiden resolution.")

    console.info(f"Merging {len(orphan_clusters)} small clusters into "
          f"{len(giant_clusters)} giants using graph connectivity...")

    # Precompute masks for giant clusters to speed up column indexing
    giant_masks = {g: (labels == g) for g in giant_clusters}
    
    # Ensure the adjacency matrix is in CSR format for efficient row slicing
    adj = adj_matrix.tocsr()

    for orphan in orphan_clusters:
        orphan_mask = (labels == orphan)
        
        # Extract the sub-matrix of edges originating strictly from the orphan nodes
        orphan_edges = adj[orphan_mask, :]
        
        best_giant = None
        max_connectivity = -1.0
        
        for g in giant_clusters:
            g_mask = giant_masks[g]
            
            # Sum the edge weights connecting the orphan nodes to this specific giant
            inter_cluster_weight = orphan_edges[:, g_mask].sum()
            
            # Normalize by the size of the giant cluster. 
            # This is the core PAGA philosophy: it prevents an artificial bias 
            # where an orphan merges into a giant simply because the giant is massive.
            g_size = g_mask.sum()
            normalized_connectivity = inter_cluster_weight / g_size if g_size > 0 else 0.0
            
            if normalized_connectivity > max_connectivity:
                max_connectivity = normalized_connectivity
                best_giant = g
                
        # Fallback: if the orphan is entirely disconnected (an extreme outlier),
        # default to the largest overall giant cluster. `counts` comes from
        # value_counts (descending), so counts.index[0] is explicitly the
        # largest cluster; picking it by that construction rather than relying on
        # giant_clusters ordering implicitly.
        if best_giant is None:
            best_giant = counts.index[0]

        # Reassign the labels of the orphan points
        labels[orphan_mask] = best_giant
        
        # Update the giant mask immediately so subsequent orphans can merge 
        # into this newly expanded giant if appropriate.
        giant_masks[best_giant] = (labels == best_giant)

    return labels


def hnsw_transfer_neighbors(sampled_data, unseen_data, k=15, ef_construction=200,
                            M=16, num_threads=1, batch_size=2_000_000, query_rows=None):
    """k nearest clustered (sampled) points of every unseen point, by HNSW.

    The neighbours depend only on the topic vectors, not on the labels, so
    :func:`cluster_tx_topics` computes them once and reuses them for every
    Leiden resolution (:func:`vote_transfer_labels`). Queries are independent,
    so querying in ``batch_size`` blocks returns exactly what one call would;
    the indices are stored as int32 (``k`` x 4 bytes per unseen point).

    With ``query_rows`` given, ``unseen_data`` is the full topic matrix and only
    its rows ``query_rows`` are queried (gathered batch by batch), so the
    unseen points never need to be copied out as one array.

    ``num_threads`` (resolved positive int) controls build/query parallelism.
    As before, the index (and therefore the result) is bit-reproducible only
    at one thread: hnswlib's multi-threaded insertion is order-dependent.

    Returns
    -------
    ndarray (n_unseen x k) of int32 row positions into ``sampled_data``.
    """
    import hnswlib
    console.info(f"Building transfer index on {len(sampled_data)} sampled points...")
    dim = sampled_data.shape[1]
    nthreads = max(1, int(num_threads))

    p = hnswlib.Index(space="cosine", dim=dim)
    p.init_index(max_elements=len(sampled_data), ef_construction=ef_construction,
                 M=M)
    p.set_num_threads(nthreads)
    p.add_items(sampled_data, num_threads=nthreads)
    p.set_ef(k + 50)

    n_query = len(unseen_data) if query_rows is None else len(query_rows)
    console.info(f"Querying {n_query} unseen points...")
    assert len(sampled_data) < np.iinfo(np.int32).max
    neighbors = np.empty((n_query, k), dtype=np.int32)
    for start in range(0, n_query, batch_size):
        stop = min(start + batch_size, n_query)
        batch = (unseen_data[start:stop] if query_rows is None
                 else unseen_data[query_rows[start:stop]])
        idx, _ = p.knn_query(batch, k=k, num_threads=nthreads)
        neighbors[start:stop] = idx
    return neighbors


def vote_transfer_labels(neighbors, sampled_labels, batch_size=2_000_000):
    """Majority vote (``scipy.stats.mode``, smallest label on ties) of the
    sampled labels at ``neighbors`` (from :func:`hnsw_transfer_neighbors`),
    row by row, in blocks to bound memory.

    Returns
    -------
    ndarray of int labels, one per row of ``neighbors``.
    """
    console.info("Voting on labels...")
    sampled_labels = np.asarray(sampled_labels)
    out = np.empty(len(neighbors), dtype=int)
    for start in range(0, len(neighbors), batch_size):
        stop = min(start + batch_size, len(neighbors))
        lab, _ = mode(sampled_labels[neighbors[start:stop]], axis=1, keepdims=False)
        out[start:stop] = lab
    return out


def transfer_labels_hnsw(sampled_data, sampled_labels, unseen_data, k=15,
                         ef_construction=200, M=16, num_threads=1):
    """Assign labels to unseen data by HNSW majority vote of k nearest
    clustered neighbours (one-shot form of :func:`hnsw_transfer_neighbors` +
    :func:`vote_transfer_labels`; kept for callers that transfer once).

    ``num_threads`` (resolved positive int) controls build/query parallelism.

    Returns
    -------
    ndarray of int labels for ``unseen_data``.
    """
    neighbors = hnsw_transfer_neighbors(
        sampled_data, unseen_data, k=k, ef_construction=ef_construction, M=M,
        num_threads=num_threads)
    return vote_transfer_labels(neighbors, sampled_labels)


def _subsample_ot_anchors(sampled_data, dense_labels, num_classes, max_anchors, rng):
    """Draw a label-stratified anchor subsample for the OT reference side.

    The OT cost matrix is dense and scales with the number of anchors, so the
    reference set must be bounded independently of
    ``sample_fraction_transfer_cluster``. Each class keeps at least one anchor;
    the remaining budget is shared proportionally to class size, so the anchor
    composition still approximates the clustered subsample.

    Returns
    -------
    sub_data, sub_dense_labels
    """
    counts = np.bincount(dense_labels, minlength=num_classes)
    present = counts > 0

    # Proportional allocation, floored at 1 anchor per non-empty class.
    quota = np.zeros(num_classes, dtype=int)
    quota[present] = np.maximum(
        1, np.floor(counts[present] / counts.sum() * max_anchors).astype(int))
    # Never ask for more anchors than a class actually has.
    quota = np.minimum(quota, counts)

    picks = []
    for cls in np.flatnonzero(present):
        idx = np.flatnonzero(dense_labels == cls)
        take = int(quota[cls])
        picks.append(idx if take >= idx.size
                     else rng.choice(idx, size=take, replace=False))

    sel = np.sort(np.concatenate(picks))
    return sampled_data[sel], dense_labels[sel]


def transfer_labels_ot_unbalanced_batched(sampled_data, sampled_labels, unseen_data,
                                          batch_size=10000, epsilon=0.05,
                                          tau=1.0, max_iter=100, tol=1e-4,
                                          max_anchors=50_000, random_state=42):
    """Assign labels to unseen data using an efficient, mini-batch adaptation
    of Unbalanced Entropic Optimal Transport (inspired by scOT-LT).

    Memory
    ------
    The Sinkhorn kernel is dense and sized ``(n_anchors x batch_size)``. Batching
    bounds the *query* side only, so the **anchor** side is capped separately by
    ``max_anchors`` (label-stratified subsample); without that cap a run with
    ``sample_fraction_transfer_cluster=0.2`` over a few million transcripts would
    try to allocate tens of GB per batch. Working arrays are ``float32``, and the
    peak is roughly ``3 * n_anchors * batch_size * 4`` bytes.

    Parameters
    ----------
    sampled_data : ndarray (N_sampled x D)
        The anchor/reference feature matrix in LDA topic space.
    sampled_labels : ndarray (N_sampled,)
        Integer cluster labels for the sampled reference data. Need **not** be
        contiguous: labels are remapped to a dense ``0..K-1`` range internally
        (e.g. after ``merge_small_clusters_paga_style`` leaves gaps) and mapped
        back on return.
    unseen_data : ndarray (N_unseen x D)
        The query feature matrix representing the unannotated majority data.
    batch_size : int
        Number of unseen points processed per batch to guarantee low memory usage.
    epsilon : float
        Entropic regularization coefficient. Lower values yield sharper boundaries;
        higher values yield smoother, more diffused mappings. Must be > 0.
    tau : float
        Marginal relaxation parameter for Unbalanced OT. Higher values approach
        standard balanced Sinkhorn; lower values tolerate higher compositional
        mismatches. Must be > 0.
    max_iter : int
        Maximum number of scaling iterations for the generalized Sinkhorn loop.
    tol : float
        Convergence tolerance on the scaling vector ``u`` between two successive
        Sinkhorn iterations; the loop stops early once the update falls below it.
    max_anchors : int or None
        Upper bound on the number of reference points entering the OT problem.
        ``None`` (or a value at/above ``len(sampled_data)``) uses every anchor —
        only safe for small subsamples.
    random_state : int
        Seed for the anchor subsample and for the microscopic tie-breaking noise
        (reproducibility).

    Returns
    -------
    final_labels : ndarray of int labels for ``unseen_data`` (in the original
        label space of ``sampled_labels``).
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0; got {epsilon}")
    if tau <= 0:
        raise ValueError(f"tau must be > 0; got {tau}")
    if batch_size is None or int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")
    batch_size = int(batch_size)
    console.info(f"Initializing scOT-LT style label transfer on {len(unseen_data)} target points...")

    num_unseen = len(unseen_data)

    # Remap possibly non-contiguous labels to a dense 0..K-1 range so the
    # class-indicator matrix has no all-zero (empty) rows, then invert at the end.
    sampled_labels = np.asarray(sampled_labels)
    unique_labels, dense_labels = np.unique(sampled_labels, return_inverse=True)
    num_classes = len(unique_labels)

    rng = np.random.default_rng(random_state)

    # --- bound the anchor side ------------------------------------------------
    if max_anchors is not None and len(sampled_data) > int(max_anchors):
        sampled_data, dense_labels = _subsample_ot_anchors(
            sampled_data, dense_labels, num_classes, int(max_anchors), rng)
        console.info(f"OT anchors capped at {len(sampled_data):,} of the clustered "
                     f"subsample (max_anchors={int(max_anchors):,}).")
    num_sampled = len(sampled_data)

    # Fail loudly *before* allocating, with an actionable message: a silent MemoryError
    # deep inside the Sinkhorn loop is much harder to act on.
    peak_gb = 3 * num_sampled * min(batch_size, max(num_unseen, 1)) * 4 / 1024 ** 3
    if peak_gb > 32:
        raise MemoryError(
            f"Unbalanced-OT label transfer would allocate ~{peak_gb:.1f} GB "
            f"({num_sampled:,} anchors x {batch_size:,} batch). Lower "
            "`advanced.ot_max_anchors` and/or `advanced.ot_batch_size`, or set "
            "`advanced.transfer_ot=False` to use the HNSW majority vote.")

    # Pre-normalize representations to compute cosine distance via matrix multiplication
    sampled_data = np.asarray(sampled_data, dtype=np.float32)
    unseen_data = np.asarray(unseen_data, dtype=np.float32)
    norm_sampled = sampled_data / (np.linalg.norm(sampled_data, axis=1, keepdims=True) + 1e-12)
    norm_unseen = unseen_data / (np.linalg.norm(unseen_data, axis=1, keepdims=True) + 1e-12)

    # Class indicators for vectorized probability grouping, as a sparse
    # (Num_Classes x N_sampled) matrix — dense would cost K x N_sampled floats.
    class_indicators = csr_matrix(
        (np.ones(num_sampled, dtype=np.float32),
         (dense_labels, np.arange(num_sampled))),
        shape=(num_classes, num_sampled))

    # Pre-calculate generalized Sinkhorn scaling exponents for Unbalanced OT
    alpha1 = tau / (tau + epsilon)
    alpha2 = tau / (tau + epsilon)

    # Target uniform marginal distributions
    a = np.full(num_sampled, 1.0 / num_sampled, dtype=np.float32)

    unseen_labels_list = []

    # Process the unseen majority in blocks
    for start_idx in range(0, num_unseen, batch_size):
        end_idx = min(start_idx + batch_size, num_unseen)
        batch_unseen = norm_unseen[start_idx:end_idx]
        B = len(batch_unseen)

        # Calculate Cosine Distance Matrix: C_ij = 1 - cos_sim(i, j)
        # Dimensions: (N_sampled x B). Built in place from the similarity product
        # so only one dense (N_sampled x B) block is ever live.
        C = np.dot(norm_sampled, batch_unseen.T)
        np.subtract(1.0, C, out=C)
        np.clip(C, 0.0, 2.0, out=C)  # Numerical safety boundary guard

        # Compute Gibbs Kernel in place (the distances are not needed afterwards).
        K = np.exp(np.divide(C, -epsilon, out=C), out=C)

        # Target marginal distribution for current batch
        b = np.full(B, 1.0 / B, dtype=np.float32)

        # Scaling vectors initialization
        u = np.full(num_sampled, 1.0 / num_sampled, dtype=np.float32)
        v = np.full(B, 1.0 / B, dtype=np.float32)

        # Generalized Sinkhorn-Knopp scaling loop for Unbalanced Optimal Transport
        for _ in range(max_iter):
            u_prev = u

            # Unbalanced relaxation scaling updates
            u = (a / (np.dot(K, v) + 1e-12)) ** alpha1
            v = (b / (np.dot(K.T, u) + 1e-12)) ** alpha2

            # Convergence check
            if np.linalg.norm(u - u_prev) < tol:
                break

        # Reconstruct optimal coupling matrix P in place on K
        # P[i, j] = u[i] * K[i, j] * v[j]
        K *= u[:, None]
        K *= v[None, :]

        # Map coupling weights straight to cluster spaces via a sparse product
        # (Num_Classes x N_sampled) @ (N_sampled x B) -> (Num_Classes x B).
        # Accumulated in float64: this block is small, and the tie-breaking noise
        # below would vanish under float32 rounding.
        class_scores_batch = np.asarray(class_indicators @ K, dtype=np.float64)

        # Inject microscopic uniform noise to break structural integer ties randomly
        # (seeded from random_state for reproducibility).
        class_scores_batch += rng.uniform(0, 1e-15, size=class_scores_batch.shape)

        # Extract the highest probability label per target entity
        batch_labels = np.argmax(class_scores_batch, axis=0)
        unseen_labels_list.append(batch_labels)

    if not unseen_labels_list:          # nothing to transfer
        return np.empty(0, dtype=int)

    dense_out = np.concatenate(unseen_labels_list)
    # Map dense 0..K-1 predictions back to the original label space.
    final_labels = unique_labels[dense_out]
    console.info("Optimal Transport label transfer completed successfully.")
    return final_labels.astype(int)


def integrate_fss_with_hnsw(adj_matrix, fss_list, fss_splits, df, fov_column, sample_indices, fss_weight):
    """Integrates the FSS spatial graph with the HNSW topic graph.

    Extracts the subgraph of the FSS matrix corresponding to `sample_indices`
    and returns a convex combination:
    ``(1 - fss_weight) * adj_matrix + fss_weight * fss_adj``.

    ``sample_indices`` are **positional** row indices into ``df`` (which was
    ``reset_index(drop=True)`` and sorted by FOV upstream), so ``np.where`` on
    the FOV mask yields the matching per-FOV local node order used to build each
    ``fss_list[i]``. The assertions below fail loudly if that alignment is ever
    violated — a silent mis-map here would corrupt clustering.

    ``fss_weight`` is clamped to ``[0, 1]``. At the extremes the result reduces
    to the (row-normalised) HNSW graph (``fss_weight=0``) or the pure FSS graph
    (``fss_weight=1``).
    """
    import scipy.sparse as sp
    n_sampled = len(sample_indices)
    fss_weight = float(np.clip(fss_weight, 0.0, 1.0))

    # Positional df row -> position within `sample_indices` (-1 = not sampled).
    # A lookup array replaces both a dict membership test and a per-row Python
    # loop: the previous version was O(n_rows) *per FOV* in interpreted code.
    sample_indices = np.asarray(sample_indices)
    sample_pos_of_row = np.full(len(df), -1, dtype=np.int64)
    sample_pos_of_row[sample_indices] = np.arange(n_sampled)

    # One pass over the FOV column instead of one boolean scan per FOV.
    fov_values = np.asarray(df[fov_column])
    rows_by_fov = pd.Series(np.arange(len(df))).groupby(fov_values).indices

    rows = []
    cols = []
    data = []

    for i, fov_id in enumerate(fss_splits):
        if fov_id in rows_by_fov:
            fov_orig_indices = rows_by_fov[fov_id]
        else:
            # Dict lookup is by hash; fall back to elementwise comparison if the
            # split id and the column dtype disagree (e.g. after a parquet
            # checkpoint round-trip) rather than silently dropping the FOV.
            fov_orig_indices = np.flatnonzero(np.asarray(df[fov_column] == fov_id))

        fss_fov = fss_list[i].tocsr()
        # Per-FOV alignment guard: the FSS graph's node count must match the
        # number of df rows for this FOV (they share the same reset row order).
        assert fss_fov.shape[0] == fov_orig_indices.shape[0], (
            f"FOV {fov_id}: FSS has {fss_fov.shape[0]} nodes but df has "
            f"{fov_orig_indices.shape[0]} rows; row order is misaligned.")

        # Local (within-FOV) positions of the sampled nodes, and where each one
        # lands in the sampled adjacency — both fully vectorised.
        fov_sample_pos = sample_pos_of_row[fov_orig_indices]
        sampled_mask_in_fov = fov_sample_pos >= 0
        if not np.any(sampled_mask_in_fov):
            continue

        local_sampled_indices = np.flatnonzero(sampled_mask_in_fov)
        assert local_sampled_indices.max() < fss_fov.shape[0], (
            f"FOV {fov_id}: local sampled index exceeds FSS node count.")
        global_sampled_indices = fov_sample_pos[local_sampled_indices]

        sub_fss = fss_fov[local_sampled_indices, :][:, local_sampled_indices]

        sub_fss_coo = sub_fss.tocoo()

        rows.append(global_sampled_indices[sub_fss_coo.row])
        cols.append(global_sampled_indices[sub_fss_coo.col])
        data.append(sub_fss_coo.data)

    if len(rows) > 0:
        fss_rows = np.concatenate(rows)
        fss_cols = np.concatenate(cols)
        fss_data = np.concatenate(data)
        fss_adj_sampled = sp.csr_matrix((fss_data, (fss_rows, fss_cols)), shape=(n_sampled, n_sampled))
    else:
        fss_adj_sampled = sp.csr_matrix((n_sampled, n_sampled))

    from sklearn.preprocessing import normalize
    
    # Ensure both matrices are on an identical 0-1 scale per row before mixing
    adj_matrix = normalize(adj_matrix, norm='l1', axis=1)
    if fss_adj_sampled.nnz > 0:
        fss_adj_sampled = normalize(fss_adj_sampled, norm='l1', axis=1)

    combined_adj = (1.0 - fss_weight) * adj_matrix + fss_weight * fss_adj_sampled
    return combined_adj


def cluster_refinement_fss(fss_graph, labels, consensus_threshold=0.6):
    """One pass of FSS weighted-majority label smoothing.

    A point switches to the majority label of its FSS neighbours if that
    majority's weight fraction is at least ``consensus_threshold``. Implemented
    as a sparse one-hot matrix product (fast for large graphs).

    Returns
    -------
    new_labels, smoothed_fraction
    """
    A = fss_graph.tocsr().copy()
    A.setdiag(0)            # a point must not vote for itself
    A.eliminate_zeros()

    num_points = len(labels)
    unique_labels, inverse_labels = np.unique(labels, return_inverse=True)
    num_classes = len(unique_labels)

    # One-hot encode current labels (sparse): (N_points x N_classes).
    row_ind = np.arange(num_points)
    col_ind = inverse_labels
    data = np.ones(num_points)
    Y_onehot = csr_matrix((data, (row_ind, col_ind)),
                          shape=(num_points, num_classes))

    # Weighted vote: A @ Y sums neighbour edge weights per class.
    votes = A @ Y_onehot
    votes_dense = votes.toarray()

    total_weights = votes_dense.sum(axis=1)
    total_weights[total_weights == 0] = 1  # avoid divide-by-zero for isolates

    max_votes = votes_dense.max(axis=1)
    majority_indices = votes_dense.argmax(axis=1)

    majority_labels = unique_labels[majority_indices]
    majority_fraction = max_votes / total_weights

    to_change = (labels != majority_labels) & (majority_fraction >= consensus_threshold)
    new_labels = labels.copy()
    new_labels[to_change] = majority_labels[to_change]

    smoothed_fraction = np.sum(to_change) / num_points
    return new_labels, smoothed_fraction


def _refine_fov_chunk(fov_id, fss_graph, initial_labels, smoothing_threshold,
                      min_delta, max_loops, consensus_threshold):
    """Iteratively refine labels for a single FOV (worker function).

    Returns
    -------
    fov_id, current_labels, fraction_history
        ``fraction_history`` is the list of per-loop smoothed fractions (used for
        the optional convergence-trace plot).
    """
    smoothed_fraction = 1.0
    prev_smoothed_fraction = 1.0
    loop = 1
    fraction_history = []

    current_labels = initial_labels.copy()

    while smoothed_fraction > smoothing_threshold and loop <= max_loops:
        current_labels, current_fraction = cluster_refinement_fss(
            fss_graph=fss_graph, labels=current_labels,
            consensus_threshold=consensus_threshold)
        fraction_history.append(current_fraction)

        # Convergence check.
        if loop > 1:
            if abs(prev_smoothed_fraction - current_fraction) < min_delta:
                break

        prev_smoothed_fraction = current_fraction
        smoothed_fraction = current_fraction
        loop += 1

    return fov_id, current_labels, fraction_history
