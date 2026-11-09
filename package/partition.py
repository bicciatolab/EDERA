"""
EDERA.partition
=================

METIS partitioning of the cleaned FSS graphs into pseudo-cells / micro-domains.
Each resulting partition is later treated as a *document* for LDA, and each gene
as a *term*.

Spatial FSS graphs are typically highly disconnected (many small "islands" of
transcripts), which makes a single global METIS call fail. The strategy here is
therefore to split each FOV graph into connected components first, partition
each large component separately, and assign small components to a single
document.

Change vs. the original script: parallelism uses **exactly** ``n_workers`` outer
processes (issue I2); the ``cpu_count()``-derived default has been removed.
"""

import numpy as np
from scipy.sparse.csgraph import connected_components
import pymetis
from joblib import Parallel, delayed

from .console import console


def _compute_metis_chunk(fss_matrix, split_value, transcripts_per_cell, contig=False):
    """Partition a single FOV's FSS graph into pseudo-cells (worker function)."""
    # Local copy cleaned for METIS (it rejects self-loops and explicit zeros).
    fss_matrix = fss_matrix.copy()
    fss_matrix.setdiag(0)
    fss_matrix.eliminate_zeros()

    total_transcripts = fss_matrix.shape[0]
    membership = np.zeros(total_transcripts, dtype=int)

    # Sub-partition by connected components first to avoid METIS failing on
    # severely disconnected graphs. On very large components (~1M+ nodes,
    # tens of thousands of parts) METIS can still print "Cannot bisect a graph
    # with 0 vertices" and return a few empty parts; the offset below handles
    # that.
    n_comps, comp_labels = connected_components(csgraph=fss_matrix, directed=False)

    current_offset = 0
    comp_sizes = np.bincount(comp_labels)

    for comp_id in range(n_comps):
        comp_size = comp_sizes[comp_id]
        if comp_size == 0:
            continue

        comp_indices = np.where(comp_labels == comp_id)[0]

        # Components too small to meaningfully partition become a single cell.
        if comp_size < int(transcripts_per_cell * 1.5):
            membership[comp_indices] = current_offset
            current_offset += 1
        else:
            n_parts = max(1, comp_size // transcripts_per_cell)
            sub_fss = fss_matrix[comp_indices, :][:, comp_indices]

            # Strict positional args: the C-extension rejects kwargs.
            adjacency = pymetis.CSRAdjacency(sub_fss.indptr, sub_fss.indices)

            try:
                if contig:
                    _, sub_membership = pymetis.part_graph(
                        nparts=n_parts + 1, adjacency=adjacency,
                        options=pymetis.Options(contig=1))
                else:
                    _, sub_membership = pymetis.part_graph(
                        nparts=n_parts + 1, adjacency=adjacency)
                sub_membership = np.asarray(sub_membership)
                membership[comp_indices] = sub_membership + current_offset
                # Advance the offset past the largest id used. METIS can leave
                # some of the requested parts empty (it then prints "Cannot
                # bisect a graph with 0 vertices"), so the ids are not
                # 0..n_unique-1: advancing by the number of unique ids made the
                # next component reuse this component's top ids, merging
                # unrelated transcripts into one pseudo-cell. Without empty
                # parts both rules give the same offset.
                current_offset += max(2, int(sub_membership.max()) + 1)
            except Exception as e:
                console.warn(e)
                # Failsafe: if a tight subgraph crashes METIS, group as one cell.
                membership[comp_indices] = current_offset
                current_offset += 2

    return membership, split_value, current_offset


def compute_metis_partition(fss_list, fss_splits, transcripts_per_cell, n_jobs=1,
                            contig=False):
    """Partition every FOV's FSS graph into pseudo-cells in parallel.

    A global offset is added per FOV so that ``metis_id`` values are unique
    across the whole dataset.

    Parameters
    ----------
    fss_list : list of csr_matrix
    fss_splits : list
        FOV identifiers aligned with ``fss_list``.
    transcripts_per_cell : int
        Target number of transcripts per pseudo-cell (METIS partition).
    n_jobs : int
        Outer worker processes (exactly ``n_workers``).
    contig : bool
        Ask METIS for contiguous parts (``AdvancedConfig.metis_contig``).

    Returns
    -------
    dict : ``{fov_id: membership_array}`` with globally unique partition ids.
    """
    pool_processes = n_jobs
    tasks = list(zip(fss_list, fss_splits))

    try:
        results = list(console.pbar(
            Parallel(n_jobs=pool_processes, return_as="generator")(
                delayed(_compute_metis_chunk)(
                    fss_mat, split_val, transcripts_per_cell, contig
                ) for fss_mat, split_val in tasks
            ),
            total=len(tasks),
            desc="Computing Metis partitions",
        ))
    except Exception as e:
        console.warn(e)
        results = Parallel(n_jobs=pool_processes)(
            delayed(_compute_metis_chunk)(
                fss_mat, split_val, transcripts_per_cell, contig
            ) for fss_mat, split_val in console.pbar(tasks, desc="Computing Metis partitions")
        )

    metis_labels_dict = {}
    offsets = 0
    for labels, split_val, n_parts in results:
        metis_labels_dict[split_val] = labels + offsets
        offsets += n_parts

    return metis_labels_dict
