"""
EDERA.decode
==============

Molecule-level decode of the Function 2 clusters (``cluster_tx_topics(decode=True)``).

punkst's pixel-decode principle, applied to EDERA clusters. Each transcript i (gene g_i)
gets the posterior over clusters c

    q_i[c]  ∝  B[g_i, c] · π_i[c]^λ

  B[g, c] = (n_gc + a) / (n_c + a·G)       gene profile of cluster c, estimated ONCE from the
                                            input labels and then kept fixed
  π_i     = (1 − δ) · W_i q / Σ(W_i q) + δ / C
            local prior from the 3D neighbours j of i (self excluded) within r µm, at most
            k_max of them, weighted by punkst's kernel w = clamp(1 − (d/r)^ν, 0.05, 0.95),
            ν = ln 0.5 / ln 0.7. δ is punkst's θ floor (0.2 / K).

q starts as the one-hot input labels. Each round recomputes the prior from the current q and
then the posterior, for exactly ``rounds`` rounds. Final label = argmax q. The own gene
enters multiplicatively, so a transcript with a type-specific gene keeps its type even when
most of its neighbours belong to another cluster.
This undoes the label leakage across touching cells that the RWR-averaged topic vectors carry.

Fixed profiles mirror punkst, whose pixel decode keeps β fixed. Re-estimating B from q
collapses the partition (−0.1 to −0.5 in the validation).

The data are decoded tile by tile. Every tile is decoded together with a ``halo_um`` margin
of neighbouring transcripts, and only its core is kept. Because the profiles are global, a
30 µm halo makes the tiled result equal to one global decode (agreement 1.0 on the 10M
simulated tissues). The validated global decode stopped early once < 0.1 % of all labels
changed in a round; applied per tile, that rule stops quiet tiles too soon and changed rare
types by up to 0.2 F1, so the tiled decode always runs every round (on the validation data the
global decode ran all 10 rounds anyway). Tiles run in forked workers; each worker is
single-threaded (k-d tree with 1 worker, serial numba kernels), so ``n_jobs`` is the number
of cores used.

Validation: ``data/Benchmarks/competitor_bench_v2/punkst_gap/REPORT.md`` (adoption rule on
held-out 10M tissues; frozen r 5 µm, k_max 128, λ 1, 10 rounds) and ``DECODE_VS_PUNKST.md``
(what is and is not taken from punkst). The simulator matches the decode's generative model,
so the synthetic gains are an upper bound.
"""

from __future__ import annotations

import multiprocessing
import time

import numba as nb
import numpy as np
import scipy.sparse as sp

#: Exponent of punkst's distance kernel: w = 0.5 at d = 0.7 r.
NU = np.log(0.5) / np.log(0.7)
#: Pseudocount of the cluster gene profiles B.
PSEUDOCOUNT = 0.5
#: punkst's θ floor δ: share of the local prior spread evenly over the clusters.
PRIOR_FLOOR = 0.2

_G = {}                                   # tile inputs, shared with the forked workers

# The kernels are compiled once per process (about a second) and NOT cached on disk: the
# package is importable both as ``EDERA`` and, from the tests, as ``package``, and numba's
# file cache keyed to one module name breaks imports under the other.


def neighbour_graph(xyz, r, k_max=128, chunk=1_000_000, workers=1):
    """CSR (float32 punkst-kernel weights) of the ≤ ``k_max`` nearest neighbours within ``r`` (3D, self excluded)."""
    from scipy.spatial import cKDTree
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    n = len(xyz)
    tree = cKDTree(xyz)
    counts, idx, wts = [], [], []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d, j = tree.query(xyz[s:e], k=k_max + 1, distance_upper_bound=r, workers=workers)
        ok = np.isfinite(d) & (j != np.arange(s, e)[:, None])
        ok &= np.cumsum(ok, axis=1) <= k_max          # rows where self was not returned
        counts.append(ok.sum(1))
        idx.append(j[ok].astype(np.int32))
        wts.append(np.clip(1.0 - (d[ok] / r) ** NU, 0.05, 0.95).astype(np.float32))
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.concatenate(counts), out=indptr[1:])
    return sp.csr_matrix((np.concatenate(wts), np.concatenate(idx), indptr), shape=(n, n))


@nb.njit
def _prior(indptr, indices, w, Q, out):
    """out[i] = Σ_j w_ij Q[j] over the neighbours j of i."""
    n, k = Q.shape
    for i in range(n):
        for c in range(k):
            out[i, c] = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            wij = w[p]
            for c in range(k):
                out[i, c] += wij * Q[j, c]


@nb.njit
def _posterior(Pi, logB, g, lam, delta, Q, lab):
    """Q[i] = softmax(logB[g_i] + λ log π_i), in place; lab = argmax."""
    n, k = Pi.shape
    for i in range(n):
        s = 0.0
        for c in range(k):
            s += Pi[i, c]
        mx = -np.inf
        best = 0
        for c in range(k):
            p = (1.0 - delta) * Pi[i, c] / s + delta / k if s > 0 else 1.0 / k
            v = logB[g[i], c] + lam * np.log(p)
            Q[i, c] = v
            if v > mx:
                mx = v
                best = c
        z = 0.0
        for c in range(k):
            e = np.exp(Q[i, c] - mx)
            Q[i, c] = e
            z += e
        for c in range(k):
            Q[i, c] /= z
        lab[i] = best


def global_log_profiles(labels, genes, n_genes, n_clusters, a=PSEUDOCOUNT):
    """log B (G x C) from every labelled transcript: (n_gc + a) / (n_c + a G)."""
    ok = labels >= 0
    cnt = np.bincount(genes[ok] * n_clusters + labels[ok], minlength=n_genes * n_clusters)
    cnt = cnt.reshape(n_genes, n_clusters).astype(np.float64)
    return np.log((cnt + a) / (cnt.sum(0, keepdims=True) + a * n_genes))


def _decode_fixed(xyz, genes, labels, logB, r, k_max, lam, rounds, delta):
    """Decode one block with fixed profiles; ``labels`` are 0..C-1 (−1 = no label, starts at q = 0)."""
    W = neighbour_graph(xyz, r, k_max=k_max, workers=1)
    n, k = len(labels), logB.shape[1]
    Q = np.zeros((n, k), np.float32)
    ok = labels >= 0
    Q[np.where(ok)[0], labels[ok]] = 1.0
    Pi = np.empty_like(Q)
    lab = labels.astype(np.int64).copy()
    w = W.data.astype(np.float32)
    for _ in range(rounds):
        _prior(W.indptr, W.indices, w, Q, Pi)
        _posterior(Pi, logB, genes, float(lam), float(delta), Q, lab)
    return lab


def _tile_task(t):
    g = _G
    i, j = divmod(t, g["ny"])
    cand = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii, jj = i + di, j + dj
            if 0 <= ii < g["nx"] and 0 <= jj < g["ny"]:
                tid = ii * g["ny"] + jj
                cand.append(g["order"][g["start"][tid]:g["start"][tid + 1]])
    cand = np.concatenate(cand)
    x0 = g["x0"] + i * g["tile"]
    y0 = g["y0"] + j * g["tile"]
    h = g["halo"]
    xs, ys = g["xyz"][cand, 0], g["xyz"][cand, 1]
    sel = cand[(xs >= x0 - h) & (xs < x0 + g["tile"] + h) & (ys >= y0 - h) & (ys < y0 + g["tile"] + h)]
    core = sel[g["tile_id"][sel] == t]
    if len(core) == 0:
        return t, core, core
    lab = _decode_fixed(g["xyz"][sel], g["genes"][sel], g["labels"][sel], g["logB"], g["r"],
                        g["k_max"], g["lam"], g["rounds"], g["delta"])
    keep = g["tile_id"][sel] == t
    return t, core, lab[keep]


def _warm_up():
    """Compile the numba kernels in the parent so forked workers do not each compile them."""
    rng = np.random.default_rng(0)
    xyz = rng.random((200, 3))
    labels = (np.arange(200) % 3).astype(np.int64)
    genes = (np.arange(200) % 7).astype(np.int64)
    _decode_fixed(xyz, genes, labels, global_log_profiles(labels, genes, 7, 3), 0.3, 8, 1.0, 2,
                  PRIOR_FLOOR)


def decode_labels(xyz, gene_codes, labels, *, radius=5.0, k_max=128, lam=1.0, rounds=10,
                  tile_um=400.0, halo_um=30.0, n_jobs=1, log=None):
    """Decode a labelling at the molecule level (see the module docstring).

    Parameters
    ----------
    xyz : (N, 3) array
        Transcript coordinates in µm (EDERA's internal ``x, y, z``).
    gene_codes : (N,) int array
        Gene of every transcript, coded 0..G-1. Every gene enters, stop words included.
    labels : (N,) int array
        Cluster labels to decode (any ids; −1 = no label: such a transcript starts with
        q = 0 and receives a label from its neighbours).
    radius, k_max, lam, rounds :
        Neighbourhood radius (µm), neighbour cap, prior exponent λ and number of rounds.
    tile_um, halo_um :
        Tile side and halo (µm) of the tiled decode.
    n_jobs : int
        Tile workers (forked processes, single-threaded each). ``1`` decodes in-process.
    log : callable or None
        Progress messages (e.g. ``console.info``).

    Returns
    -------
    decoded : (N,) int64 array, labels in the input's ids, every transcript labelled
    info : dict with ``n_tiles``, ``relabelled`` (share of labelled transcripts whose label
        changed) and ``seconds``.
    """
    t0 = time.time()
    log = log or (lambda msg: None)
    labels = np.asarray(labels)
    ids, comp = np.unique(labels[labels >= 0], return_inverse=True)
    lab_c = np.full(len(labels), -1, np.int64)
    lab_c[labels >= 0] = comp
    genes = np.asarray(gene_codes, np.int64)
    n_genes = int(genes.max()) + 1
    logB = global_log_profiles(lab_c, genes, n_genes, len(ids))
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    x0, y0 = xyz[:, 0].min(), xyz[:, 1].min()
    nx = int(np.floor((xyz[:, 0].max() - x0) / tile_um)) + 1
    ny = int(np.floor((xyz[:, 1].max() - y0) / tile_um)) + 1
    tx = np.minimum(((xyz[:, 0] - x0) / tile_um).astype(np.int64), nx - 1)
    ty = np.minimum(((xyz[:, 1] - y0) / tile_um).astype(np.int64), ny - 1)
    tile_id = tx * ny + ty
    order = np.argsort(tile_id, kind="stable")
    start = np.searchsorted(tile_id[order], np.arange(nx * ny + 1))
    tiles = np.flatnonzero(np.diff(start) > 0)
    _G.clear()
    _G.update(xyz=xyz, genes=genes, labels=lab_c, logB=logB, tile_id=tile_id, order=order, start=start,
              nx=nx, ny=ny, x0=x0, y0=y0, tile=float(tile_um), halo=float(halo_um), r=float(radius),
              k_max=int(k_max), lam=float(lam), rounds=int(rounds), delta=PRIOR_FLOOR)
    _warm_up()
    out = np.full(len(labels), -1, np.int64)
    sizes = np.diff(start)[tiles]
    todo = tiles[np.argsort(-sizes)]                                  # largest tiles first
    n_jobs = max(1, min(int(n_jobs), len(todo)))
    log(f"decode: {len(labels):,} transcripts, {len(ids)} clusters, {len(tiles)} tiles of "
        f"{tile_um:g} µm (halo {halo_um:g} µm, largest {sizes.max():,}), {n_jobs} workers")
    try:
        if n_jobs == 1:
            for t, core, lab in map(_tile_task, todo):
                out[core] = lab
        else:
            with multiprocessing.get_context("fork").Pool(n_jobs) as pool:
                for t, core, lab in pool.imap_unordered(_tile_task, todo, chunksize=1):
                    out[core] = lab
    finally:
        _G.clear()
    if (out < 0).any():
        raise RuntimeError(f"decode: {int((out < 0).sum())} transcripts were not decoded")
    decoded = ids[out].astype(np.int64)
    has = labels >= 0
    info = dict(n_tiles=int(len(tiles)),
                relabelled=float(np.mean(decoded[has] != labels[has])) if has.any() else 0.0,
                seconds=round(time.time() - t0, 1))
    log(f"decode: done in {info['seconds']:.0f} s; {info['relabelled']:.1%} relabelled")
    return decoded, info
