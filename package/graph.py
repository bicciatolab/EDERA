"""
EDERA.graph
=============

Spatial neighbourhood-graph construction and cleaning:

* per-FOV k-nearest-neighbour (kNN) graphs;
* per-FOV fuzzy simplicial sets (FSS), with ``local_connectivity=0.0`` exactly
  as in the original pipeline;
* a unified post-FSS filter that prunes long edges, low-strength nodes (MAD
  based) and low-degree nodes (iteratively).

Changes vs. the original script:

* Parallelism uses **exactly** ``n_workers`` outer processes; the inner thread
  count of the kNN workers is an explicit argument (``inner_threads``, default 1)
  instead of being derived from ``cpu_count()`` (issue I2).
* ``random_state`` is an explicit argument threaded into ``nearest_neighbors``
  and ``fuzzy_simplicial_set`` (issue I8).
* The **weight==1 edge pruning** step has been removed from the filter
  (issue I9): ``local_connectivity=0.0`` is kept, but weight-1 edges now flow
  through unchanged. The ``drop_weight_one`` argument and the
  ``removed_weight_one`` statistic no longer exist.
"""

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from joblib import Parallel, delayed
from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set

from .console import console
from .utils import fov_row_positions, EMPTY_POSITIONS


# ---------------------------------------------------------------------------
# kNN graph
# ---------------------------------------------------------------------------
def _compute_knn_chunk(split_value, group_df, coords_cols, n_neighbors,
                       inner_threads, random_state):
    """Compute the kNN graph for a single FOV chunk (worker function)."""
    from umap.umap_ import nearest_neighbors
    X_group = group_df[coords_cols].values
    knn_indices_group, knn_dists_group, _ = nearest_neighbors(
        X_group,
        n_neighbors=n_neighbors,
        metric="euclidean",
        metric_kwds={},
        angular=False,
        random_state=random_state,
        n_jobs=inner_threads,
    )

    # Adjust indices to be global (relative to the full DataFrame).
    knn_indices_group += group_df.index.min()
    split_list = [split_value] * group_df.shape[0]

    return knn_indices_group, knn_dists_group, split_list


def compute_knn_graph(df, n_neighbors, coords_cols=("x", "y", "z"),
                      split_column=None, n_jobs=1, inner_threads=1,
                      random_state=42):
    """Compute a kNN graph, optionally per-FOV in parallel.

    Parameters
    ----------
    df : DataFrame with the coordinate columns.
    n_neighbors : int
        Number of nearest neighbours per node.
    coords_cols : sequence of str
        Coordinate columns to use.
    split_column : str or None
        If given, the graph is computed independently per group (FOV) in
        parallel; otherwise a single graph is computed for the whole table.
    n_jobs : int
        Number of outer worker processes (exactly ``n_workers``).
    inner_threads : int
        Threads used inside each worker's ``nearest_neighbors`` call.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    knn_indices, knn_dists, split_values
    """
    coords_cols = list(coords_cols)
    pool_processes = n_jobs

    if split_column is not None:
        grouped_tasks = list(df.groupby(split_column))

        try:
            results = list(console.pbar(
                Parallel(n_jobs=pool_processes, return_as="generator")(
                    delayed(_compute_knn_chunk)(
                        split_value, group, coords_cols, n_neighbors,
                        inner_threads, random_state,
                    ) for split_value, group in grouped_tasks
                ),
                total=len(grouped_tasks),
                desc=f"Processing {split_column} chunks",
            ))
        except Exception:
            results = Parallel(n_jobs=pool_processes)(
                delayed(_compute_knn_chunk)(
                    split_value, group, coords_cols, n_neighbors,
                    inner_threads, random_state,
                ) for split_value, group in console.pbar(
                    grouped_tasks, desc=f"Dispatching {split_column} chunks")
            )

        knn_indices_list = [res[0] for res in results]
        knn_dists_list = [res[1] for res in results]
        split_values = [res[2] for res in results]

        knn_indices = np.vstack(knn_indices_list)
        knn_dists = np.vstack(knn_dists_list)

    else:
        X = df[coords_cols].values
        # No FOV split: run a single kNN using exactly n_jobs threads.
        knn_indices, knn_dists, _ = nearest_neighbors(
            X,
            n_neighbors=n_neighbors,
            metric="euclidean",
            metric_kwds={},
            angular=False,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        split_values = None

    return knn_indices, knn_dists, split_values


# ---------------------------------------------------------------------------
# Fuzzy simplicial set (FSS)
# ---------------------------------------------------------------------------
def _compute_fss_chunk(X, knn_indices, knn_dists, n_neighbors, random_state):
    """Run UMAP's ``fuzzy_simplicial_set`` on a precomputed kNN graph.

    ``local_connectivity=0.0`` is kept exactly as in the original pipeline. No
    pre-capping of long edges is done here; all distance/noise filtering happens
    in the unified post-FSS filter.
    """
    fss = fuzzy_simplicial_set(
        X=X,
        n_neighbors=n_neighbors,
        random_state=random_state,
        metric="euclidean",
        knn_dists=knn_dists,
        knn_indices=knn_indices,
        local_connectivity=0.0,
        set_op_mix_ratio=1.0,
    )
    return fss


def compute_fss(df, knn_indices, knn_dists, n_neighbors,
                coords_cols=("x", "y", "z"), split_column=None, n_jobs=1,
                random_state=42):
    """Compute FSS graphs per FOV (or globally) from a precomputed kNN.

    Parameters mirror :func:`compute_knn_graph`. Returns
    ``(fss_list, splits, node_indices)`` where ``fss_list[i]`` is the raw FSS
    tuple for split ``splits[i]`` and ``node_indices[i]`` maps local node order
    back to the original DataFrame index.
    """
    coords_cols = list(coords_cols)
    pool_processes = n_jobs

    if split_column is not None:
        grouped_tasks = list(df.groupby(split_column))

        def prep_chunk(split_value, group_data):
            X = group_data[coords_cols].values
            f_knn_idx = knn_indices[group_data.index].copy()
            f_knn_d = knn_dists[group_data.index].copy()
            # Convert global indices to local subset indices for FSS.
            f_knn_idx -= group_data.index.min()
            return X, f_knn_idx, f_knn_d, split_value, group_data.index.to_numpy()

        task_params = [prep_chunk(val, group) for val, group in grouped_tasks]

        try:
            results = list(console.pbar(
                Parallel(n_jobs=pool_processes, return_as="generator")(
                    delayed(_compute_fss_chunk)(
                        X, f_knn_idx, f_knn_d, n_neighbors, random_state,
                    ) for X, f_knn_idx, f_knn_d, _, _ in task_params
                ),
                total=len(task_params),
                desc=f"Processing {split_column} FSS chunks",
            ))
        except Exception:
            results = Parallel(n_jobs=pool_processes)(
                delayed(_compute_fss_chunk)(
                    X, f_knn_idx, f_knn_d, n_neighbors, random_state,
                ) for X, f_knn_idx, f_knn_d, _, _ in console.pbar(
                    task_params, desc=f"Dispatching {split_column} FSS")
            )

        fss_list = list(results)
        splits = [tp[3] for tp in task_params]
        node_indices = [tp[4] for tp in task_params]

    else:
        X = df[coords_cols].values
        fss = _compute_fss_chunk(X, knn_indices.copy(), knn_dists.copy(),
                                 n_neighbors, random_state)
        fss_list = [fss]
        splits = None
        node_indices = [df.index.to_numpy()]

    return fss_list, splits, node_indices


# ---------------------------------------------------------------------------
# Unified post-FSS filter (weight==1 pruning removed, issue I9)
# ---------------------------------------------------------------------------
def _filter_fss_chunk(fov_id, mat, coords, df_index, max_edge_distance,
                      min_node_strength, n_mad_strength, min_connected_fss,
                      chunk_size):
    """Filter a single FOV's FSS graph (worker function).

    Pruning, in order:
      1. distance prune (one absolute cap);
      2. FSS-strength noise prune (drop nodes with strength <= threshold);
      3. iterative low-degree prune (drop nodes with < ``min_connected_fss``
         neighbours, cascading until convergence).

    The original weight==1 prune has been removed (issue I9).
    """
    if mat.shape[0] != coords.shape[0]:
        n_common = min(mat.shape[0], coords.shape[0])
        console.info(f"FOV {fov_id}: aligning mismatched chunk sizes "
              f"(fss nodes={mat.shape[0]}, coords rows={coords.shape[0]}) "
              f"to {n_common} via prefix trim.")
        mat = mat[:n_common, :n_common]
        coords = coords[:n_common]
        df_index = df_index[:n_common]
    n_nodes_before = mat.shape[0]

    coo = mat.tocoo(copy=True)
    nnz_before = coo.nnz
    new_data = coo.data.copy()
    removed_distance = 0

    # Chunked edge evaluation to keep peak memory modest on large graphs.
    for start in range(0, nnz_before, chunk_size):
        end = min(start + chunk_size, nnz_before)
        r = coo.row[start:end]
        c = coo.col[start:end]

        prune_mask = np.zeros(end - start, dtype=bool)
        if max_edge_distance is not None:
            d = np.linalg.norm(coords[r] - coords[c], axis=1)
            dist_mask = d > max_edge_distance
            prune_mask |= dist_mask
            removed_distance += int(dist_mask.sum())

        if prune_mask.any():
            chunk = new_data[start:end]
            chunk[prune_mask] = 0.0
            new_data[start:end] = chunk

    coo.data = new_data
    pruned = coo.tocsr()
    pruned.setdiag(0)
    pruned.eliminate_zeros()
    # Strictly undirected (METIS requirement); kills non-mutual leftovers.
    pruned = pruned.minimum(pruned.T)
    pruned.eliminate_zeros()

    # Node strength on the pruned graph drives noise filtering.
    strength = np.asarray(pruned.sum(axis=1)).ravel()

    if min_node_strength is None:
        thresh = 0.0  # drop only orphan nodes (strength == 0)
    elif isinstance(min_node_strength, str) and min_node_strength.lower() == "auto":
        nz = strength[strength > 0]
        if nz.size:
            med = float(np.median(nz))
            mad = float(median_abs_deviation(nz))
            thresh = max(0.0, med - n_mad_strength * mad)
        else:
            thresh = 0.0
    else:
        thresh = float(min_node_strength)

    keep_nodes = strength > thresh
    pruned = pruned[keep_nodes, :][:, keep_nodes]
    df_index_kept = df_index[keep_nodes]

    # Iterative low-degree pruning: remove nodes with fewer than
    # min_connected_fss neighbours until no more are dropped. Cascading is
    # necessary because removing a node reduces its neighbours' degrees.
    removed_low_degree = 0
    if min_connected_fss is not None and min_connected_fss > 0:
        while True:
            pruned = pruned.tocsr()
            degree = np.diff(pruned.indptr)
            keep = degree >= min_connected_fss
            n_drop = int((~keep).sum())
            if n_drop == 0:
                break
            removed_low_degree += n_drop
            pruned = pruned[keep, :][:, keep]
            df_index_kept = df_index_kept[keep]

    stats = {
        "fov": fov_id,
        "n_nodes_before": n_nodes_before,
        "n_nodes_after": pruned.shape[0],
        "n_nodes_dropped": n_nodes_before - pruned.shape[0],
        "nnz_before": nnz_before,
        "nnz_after": pruned.nnz,
        "removed_distance": removed_distance,
        "strength_threshold": thresh,
        "removed_low_degree": removed_low_degree,
    }

    return fov_id, pruned, df_index_kept, stats


def filter_fss(fss_list, fss_splits, df, fov_column, fss_node_indices=None,
               max_edge_distance=None, min_node_strength=None,
               n_mad_strength=3.0, min_connected_fss=10, chunk_size=1_000_000,
               n_jobs=1):
    """Apply the unified post-FSS filter across all FOVs in parallel.

    Parameters
    ----------
    fss_list : list of csr_matrix
    fss_splits : list
        FOV identifiers aligned with ``fss_list``.
    df : DataFrame
    fov_column : str
    fss_node_indices : list or None
        Per-FOV mapping of node order to original DataFrame index.
    max_edge_distance : float or None
        Absolute physical-distance cap on edges.
    min_node_strength : None, 'auto', or float
        Node-strength noise threshold. ``'auto'`` uses a MAD-based cut.
    n_mad_strength : float
        MAD multiplier for the ``'auto'`` strength threshold.
    min_connected_fss : int
        Minimum neighbour count for the iterative low-degree prune.
    chunk_size : int
        Edge-evaluation chunk size (memory control).
    n_jobs : int
        Outer worker processes (exactly ``n_workers``).

    Returns
    -------
    fss_filtered, fss_dict_filtered, df_filtered, stats_df
    """
    pool_processes = n_jobs

    if fss_node_indices is not None and len(fss_node_indices) != len(fss_list):
        raise ValueError(
            f"fss_node_indices has {len(fss_node_indices)} entries, "
            f"but fss_list has {len(fss_list)} graphs")

    # Row positions of every FOV in one pass (not one full-table mask per FOV)
    # and the float32 coordinates converted once; the rows each FOV gets, and
    # their order, are unchanged.
    rows_by_fov = fov_row_positions(df, fov_column)
    index_values = df.index.to_numpy()
    xyz32 = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    unique_index = df.index.is_unique

    tasks = []
    for i, (fov_id, mat) in enumerate(zip(fss_splits, fss_list)):
        pos = rows_by_fov.get(fov_id, EMPTY_POSITIONS)
        if fss_node_indices is None:
            df_index = index_values[pos]
            coords = xyz32[pos]
        else:
            df_index = np.asarray(fss_node_indices[i])
            if df_index.size and unique_index:
                label_pos = df.index.get_indexer(df_index)
                valid = bool((label_pos >= 0).all())
            elif df_index.size:
                label_pos = None
                valid = bool(np.isin(df_index, index_values).all())
            else:
                valid = False
            if valid:
                coords = (xyz32[label_pos] if label_pos is not None else
                          df.loc[df_index, ["x", "y", "z"]].to_numpy(dtype=np.float32))
            else:
                # Fallback: use current FOV slice order when stored indices are
                # stale, absolute, or otherwise no longer valid labels.
                df_index = index_values[pos]
                coords = xyz32[pos]
        tasks.append((fov_id, mat, coords, df_index))

    try:
        results = list(console.pbar(
            Parallel(n_jobs=pool_processes, return_as="generator")(
                delayed(_filter_fss_chunk)(
                    fov_id, mat, coords, df_index, max_edge_distance,
                    min_node_strength, n_mad_strength, min_connected_fss,
                    chunk_size,
                ) for fov_id, mat, coords, df_index in tasks
            ),
            total=len(tasks),
            desc="Filtering FSS graphs",
        ))
    except Exception:
        results = Parallel(n_jobs=pool_processes)(
            delayed(_filter_fss_chunk)(
                fov_id, mat, coords, df_index, max_edge_distance,
                min_node_strength, n_mad_strength, min_connected_fss,
                chunk_size,
            ) for fov_id, mat, coords, df_index in console.pbar(
                tasks, desc="Dispatching FSS filter")
        )

    # Restore original FOV order from fss_splits in case of generator ordering.
    by_fov = {res[0]: res for res in results}
    ordered = [by_fov[fov] for fov in fss_splits]

    fss_filtered = [r[1] for r in ordered]
    fss_dict_filtered = {r[0]: r[1] for r in ordered}
    keep_index = np.concatenate([r[2] for r in ordered])
    stats_df = pd.DataFrame([r[3] for r in ordered])

    df_filtered = df.loc[keep_index].copy().reset_index(drop=True)
    return fss_filtered, fss_dict_filtered, df_filtered, stats_df
