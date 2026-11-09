"""
EDERA
=======

Segmentation-free, transcript-level topic-modelling and spatial-clustering
pipeline for high-plex spatial transcriptomics (e.g. CosMx, Xenium).

The package refactors the original monolithic script
``scripts_and_analysis/full_pipeline_unnamed_segm_rwr.py`` into modular steps
while preserving the algorithmic logic. See ``README.md`` (here) and
``../drafts_and_plans/REFACTORING_PLAN.md`` for details.

Pipeline overview
-----------------
1. Build a per-FOV spatial neighbourhood graph (kNN -> fuzzy simplicial set).
2. Filter graph noise (long edges, low node strength, low degree).
3. Partition each FSS into pseudo-cells via METIS (documents for LDA).
4. Learn a latent topic structure with LDA.
5. Diffuse per-transcript topic vectors spatially with Random Walk with Restart.
6. Cluster the diffused topic space (HNSW -> Leiden), merge gene-split
   fragments, transfer labels, smooth them by FSS consensus and decode them
   at the molecule level, at one or more Leiden resolutions.

Public API
----------
- :func:`compute_tx_topics`  : Function 1 (load -> RWR diffusion).
- :func:`cluster_tx_topics`  : Function 2 (multi-resolution clustering).
- :func:`run_pipeline`       : Function 1 then Function 2.
- :class:`TxTopicsResult`    : result container / checkpoint I/O.
- :class:`AdvancedConfig`    : advanced/internal parameters.

Example
-------
>>> from EDERA import run_pipeline
>>> df = run_pipeline(
...     data_path="transcripts.csv",
...     n_workers=16, out_path="run01",
...     plot_format=["pdf", "html"], dpi=600,
...     diagnostic_bbox=(0, 5000, 0, 5000),
...     compute_kwargs={"F1_mode": "complex structures", "n_topics_lda": 30},
...     cluster_kwargs={"leiden_resolution": [0.5, 1.0]},
... )
"""

from .config import AdvancedConfig
from .api import (
    compute_tx_topics,
    cluster_tx_topics,
    run_pipeline,
    TxTopicsResult,
    F1_MODES,
)

__all__ = [
    "compute_tx_topics",
    "cluster_tx_topics",
    "run_pipeline",
    "TxTopicsResult",
    "AdvancedConfig",
    "F1_MODES",
]

__version__ = "0.4.0"
