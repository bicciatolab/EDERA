"""
EDERA.io_load
===============

Data input for the pipeline: loading the long transcript table, renaming and
scaling coordinates, carrying the gene label in an explicit canonical column,
and — when no native ``fov``/chunk column exists — partitioning the 2D plane
into balanced, spatially-connected pseudo-FOVs via Recursive Coordinate
Bisection (RCB).

Changes vs. the original script:

* The gene column is renamed to the canonical name ``target`` at load time and
  used explicitly everywhere downstream (issue I7); no code relies on a
  ``gene_name`` column that only exists as a merge side-effect.
* The embedded partition scatter plot has been removed from
  :func:`autopartition_if_no_fov`; the always-on partition overview is now drawn
  by :mod:`EDERA.plotting`.
* The subsampling RNG is seeded from the single pipeline ``random_state`` (I8).
"""

import time

import numpy as np
import pandas as pd

from .console import console


def load_transcripts(data_path, x_name, y_name, z_name, gene_name, fov_column,
                     x_res, y_res, z_res, id_column=None):
    """Load and normalise a long transcript table.

    Reads the CSV at ``data_path``, keeps and renames the required columns to the
    canonical names ``x, y, z, target`` (and ``fov`` if present), scales each
    coordinate axis by its physical resolution, and zero-shifts each axis to the
    origin. Every transcript also gets a ``tx_id`` that travels with its row
    through every later step (sorting, filtering, merging), so any output row can
    be matched back to its input row.

    Parameters
    ----------
    data_path : str
        Path to the input transcript CSV.
    x_name, y_name, z_name : str
        Source column names for the spatial coordinates.
    gene_name : str
        Source column name for the gene/target label. Renamed to ``target``.
    fov_column : str
        Name of the FOV/chunk column. If absent it is created later by
        :func:`autopartition_if_no_fov`.
    x_res, y_res, z_res : float
        Per-axis physical resolution (multiplied into the coordinates).
    id_column : str or None
        Source column holding a caller-supplied transcript identifier, copied
        to ``tx_id``. It must be present, free of missing values and unique.
        ``None`` (default) uses the 0-based row position in the CSV.

    Returns
    -------
    data : pandas.DataFrame
        Columns ``tx_id, x, y, z, target`` and, if present in the source,
        ``fov``. ``tx_id`` is int64 when taken from the row position.
    offsets : tuple of float
        The per-axis origins ``(x_min, y_min, z_min)`` subtracted after scaling.
        Persisted so a raw-unit ``diagnostic_bbox`` can be converted to internal
        coordinates exactly (see :func:`EDERA.utils.convert_bbox_to_internal`).
    """
    data = pd.read_csv(data_path)

    # Always-required columns + their renames. `fov` is optional: if the source
    # CSV lacks it, autopartition_if_no_fov() will partition 2D space instead.
    # The gene label becomes the canonical column `target` (explicit provenance).
    column_map = {x_name: "x", y_name: "y", z_name: "z", gene_name: "target"}
    if fov_column in data.columns:
        column_map[fov_column] = "fov"

    # Row identity is fixed here, before anything can reorder or drop rows: from
    # this point on the table is sorted by FOV and filtered, and its index is
    # reset, so the CSV row position cannot be recovered afterwards.
    if id_column is None:
        tx_id = np.arange(len(data), dtype=np.int64)
    else:
        if id_column not in data.columns:
            raise ValueError(f"id_column '{id_column}' not found in {data_path}")
        tx_id = data[id_column]
        if tx_id.isna().any():
            raise ValueError(f"id_column '{id_column}' has missing values")
        if not tx_id.is_unique:
            raise ValueError(f"id_column '{id_column}' is not unique "
                             f"({int(tx_id.duplicated().sum()):,} repeated values)")
        tx_id = tx_id.to_numpy()

    data = data[list(column_map)].rename(columns=column_map)
    data.insert(0, "tx_id", tx_id)

    # Scale x, y, z by per-axis resolutions and zero-shift to the origin.
    data["x"] = data["x"] * x_res
    data["y"] = data["y"] * y_res
    data["z"] = data["z"] * z_res
    x_off = float(data["x"].min())
    y_off = float(data["y"].min())
    z_off = float(data["z"].min())
    data["x"] = data["x"] - x_off
    data["y"] = data["y"] - y_off
    data["z"] = data["z"] - z_off

    return data, (x_off, y_off, z_off)


def partition_2d_rcb(df, target_partition_size=100_000, coords=("x", "y")):
    """Recursive Coordinate Bisection on 2D coordinates (z ignored here).

    At each step the longer-extent axis is split at its median, recursively,
    until each region holds at most ``target_partition_size`` points. Every
    partition is an axis-aligned rectangle (convex, hence spatially connected),
    and the partitions have approximately balanced point counts. Complexity is
    ``O(N log K)`` with ``K = ceil(N / target_partition_size)``.

    Parameters
    ----------
    df : DataFrame containing the two coordinate columns named in ``coords``.
    target_partition_size : int
        Approximate number of points per partition (default 100_000).
    coords : tuple of 2 str
        Coordinate column names to use (defaults to ``("x", "y")``).

    Returns
    -------
    labels : int32 ndarray of shape (len(df),) with values 0..K-1.
    """
    x = df[coords[0]].to_numpy(dtype=np.float64)
    y = df[coords[1]].to_numpy(dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    labels = np.empty(n, dtype=np.int32)
    stack = [np.arange(n, dtype=np.int64)]
    leaf_id = 0

    while stack:
        idx = stack.pop()
        m = len(idx)
        if m <= target_partition_size or m < 2:
            labels[idx] = leaf_id
            leaf_id += 1
            continue

        x_c = x[idx]
        y_c = y[idx]
        # Split along the longer-extent axis at the median.
        if (x_c.max() - x_c.min()) >= (y_c.max() - y_c.min()):
            vals = x_c
        else:
            vals = y_c
        med = np.median(vals)

        left_mask = vals <= med
        right_mask = ~left_mask
        # Degenerate split (e.g. heavy ties): bail out and accept as a leaf.
        if not left_mask.any() or not right_mask.any():
            labels[idx] = leaf_id
            leaf_id += 1
            continue

        stack.append(idx[left_mask])
        stack.append(idx[right_mask])

    return labels


def autopartition_if_no_fov(data, fov_column="fov", target_partition_size=100_000,
                            coords_2d=("x", "y")):
    """Ensure a ``fov_column`` exists, creating one via 2D RCB if necessary.

    If ``data`` already contains ``fov_column`` it is used as the chunking key.
    Otherwise the 2D plane is partitioned into balanced, spatially-connected
    pseudo-FOVs. The always-on partition overview plot is drawn separately by
    :mod:`EDERA.plotting` (this function no longer plots).

    Parameters
    ----------
    data : DataFrame
    fov_column : str
        Name of the chunking column to look for / create.
    target_partition_size : int
        Approximate transcripts per pseudo-FOV.
    coords_2d : tuple of 2 str
        Coordinate columns for the 2D partition.

    Returns
    -------
    data : DataFrame (with ``fov_column`` guaranteed present)
    fov_was_synthetic : bool
        True if this function created the column, False if it was already there.
    """
    if fov_column in data.columns:
        n_fovs = data[fov_column].nunique()
        console.info(f"Column '{fov_column}' present: {n_fovs} unique values; "
              f"using as chunking key.")
        return data, False

    console.info(f"Column '{fov_column}' missing; partitioning {len(data):,} points "
          f"in 2D ({coords_2d}) via RCB...")
    t0 = time.time()
    labels = partition_2d_rcb(data, target_partition_size, coords=coords_2d)
    elapsed = time.time() - t0

    data = data.copy()
    data[fov_column] = labels

    counts = pd.Series(labels).value_counts().sort_index()
    console.info(f"  Built {counts.size} partitions in {elapsed:.2f}s. "
          f"Sizes: min={counts.min()}, max={counts.max()}, "
          f"median={int(counts.median())}, mean={counts.mean():.0f}")

    return data, True
