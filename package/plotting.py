"""
EDERA.plotting
================

Centralised plotting for the pipeline. Every figure produced anywhere in
:mod:`EDERA` is created here, so that three cross-cutting concerns are
handled in exactly one place:

1. **Output format** — ``plot_format`` is a *list* (e.g. ``["pdf"]`` or
   ``["png", "html"]``). ``pdf``/``png`` are rendered with matplotlib;
   ``html`` is rendered with Plotly. Multiple formats may be requested together.
2. **Adobe Illustrator editability** — :func:`setup_style` sets
   ``pdf.fonttype = 42`` / ``ps.fonttype = 42`` / ``svg.fonttype = "none"`` so
   that text stays as editable text (not outlined). Dense scatters use
   ``rasterized=True`` so only the point cloud is flattened to a raster layer at
   the chosen ``dpi`` (default 600), while axes, labels and legends stay vector.
3. **Bounding-box gating** — :func:`subset_bbox` restricts *spatial* (transcript
   position) and *graph-edge* plots to an optional ``diagnostic_bbox``. If the
   bbox is ``None`` those plots are skipped. Global QC plots (histograms, UMAPs,
   the partition overview) are never bbox-gated and always render.

All plot functions are deliberately side-effecting: they save to ``out_path``
and return the list of written file paths.
"""

import functools
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import seaborn as sns

from .console import console
from .utils import prefixed, fov_row_positions, EMPTY_POSITIONS


# ---------------------------------------------------------------------------
# Discrete categorical palette
# ---------------------------------------------------------------------------
# A single 256-colour palette drives every categorical plot. ``tab20`` (the old
# default) only has 20 colours, so any plot with >20 distinct labels reused
# colours and, worse, gave adjacent label indices near-identical colours. The
# palette below is hue-spread then *scrambled* by a coprime stride so that
# consecutive labels (e.g. 20 and 21) land ~2/3 of the way around the colour
# wheel apart, and holds 256 distinct entries (8-bit) before any repeat.
_N_DISCRETE = 256
_SCRAMBLE_STRIDE = 151          # coprime with 256 -> bijection + big hue jump/step
_UNASSIGNED_COLOR = (0.72, 0.72, 0.72, 1.0)   # fixed grey for the -1 sentinel


@functools.lru_cache(maxsize=1)
def _discrete_palette():
    """256 RGBA colours, hue-ordered then scrambled so consecutive indices sit
    far apart on the colour wheel. Deterministic; 256 unique entries."""
    import colorsys
    # Primary variation = hue (monotonic in i). Saturation/value cycle across a
    # few tiers so even hue-near colours differ in brightness/richness.
    base = []
    for i in range(_N_DISCRETE):
        h = i / _N_DISCRETE
        s = 0.65 + 0.30 * ((i // 3) % 2)      # two saturation tiers
        v = 0.95 - 0.30 * (i % 3) / 2.0        # three value tiers
        base.append(colorsys.hsv_to_rgb(h, s, v) + (1.0,))
    # Coprime stride guarantees base[k*151] and base[(k+1)*151] are 151 hue-steps
    # (~2/3 of the wheel) apart -> adjacent labels never look similar.
    return [base[(k * _SCRAMBLE_STRIDE) % _N_DISCRETE] for k in range(_N_DISCRETE)]


def discrete_color_map(values, unassigned=None):
    """``{value: rgba}`` restricted to the distinct values present in ``values``.

    <=256 distinct -> unique colours (no repeats); >256 -> colours wrap (repeat).
    Consecutive integer values get distant colours. ``unassigned`` (e.g. -1), if
    present, is forced to a fixed neutral grey and does not consume a palette slot.
    """
    pal = _discrete_palette()
    uniq = sorted(pd.unique(pd.Series(list(values)).dropna()))
    out, j = {}, 0
    for v in uniq:
        if unassigned is not None and v == unassigned:
            out[v] = _UNASSIGNED_COLOR
        else:
            out[v] = pal[j % len(pal)]
            j += 1
    return out


def _hex_color_map(color_map):
    """String-keyed hex map for Plotly ``color_discrete_map`` (drops alpha)."""
    return {str(k): mcolors.to_hex(c) for k, c in color_map.items()}


# ---------------------------------------------------------------------------
# Style and format routing
# ---------------------------------------------------------------------------
def setup_style(dpi: int = 600, dark_bg: bool = False) -> None:
    """Configure matplotlib once for reproducible, Illustrator-editable output.

    Parameters
    ----------
    dpi : int
        Resolution for both on-figure rendering and the rasterized point layers
        of dense scatter plots. Default 600.
    dark_bg : bool
        If True, default figure/axes backgrounds are black. Default False.
    """
    matplotlib.rcParams["pdf.fonttype"] = 42      # editable text in PDF
    matplotlib.rcParams["ps.fonttype"] = 42       # editable text in EPS/PS
    matplotlib.rcParams["svg.fonttype"] = "none"  # editable text in SVG
    matplotlib.rcParams["figure.dpi"] = dpi
    matplotlib.rcParams["savefig.dpi"] = dpi
    if dark_bg:
        matplotlib.rcParams["figure.facecolor"] = "black"
        matplotlib.rcParams["axes.facecolor"] = "black"


def save_mpl(fig, out_path: str, name: str, plot_format, dpi: int = 600,
             close: bool = True):
    """Save a matplotlib figure to every requested raster/vector format.

    Only ``pdf`` and ``png`` entries of ``plot_format`` are honoured here;
    ``html`` is ignored (handled by :func:`save_plotly`).

    Returns
    -------
    list of str : paths written.
    """
    os.makedirs(out_path, exist_ok=True)
    written = []
    for fmt in plot_format:
        if fmt in ("pdf", "png"):
            p = os.path.join(out_path, f"{name}.{fmt}")
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            written.append(p)
    if close:
        plt.close(fig)
    return written


def save_plotly(fig, out_path: str, name: str, plot_format):
    """Save a Plotly figure as HTML if ``html`` is in ``plot_format``.

    Returns
    -------
    list of str : paths written.
    """
    written = []
    if "html" in plot_format:
        os.makedirs(out_path, exist_ok=True)
        p = os.path.join(out_path, f"{name}.html")
        fig.write_html(p)
        written.append(p)
    return written


def subset_bbox(df: pd.DataFrame, bbox):
    """Restrict a transcript DataFrame to an optional (x, y[, z]) bounding box.

    Parameters
    ----------
    df : DataFrame with ``x``, ``y`` and (optionally) ``z`` columns.
    bbox : tuple or None
        ``(xmin, xmax, ymin, ymax)`` or ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
        If None, returns None (caller skips the spatial plot).

    Returns
    -------
    DataFrame or None
    """
    if bbox is None:
        return None
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    mask = (
        (df["x"] >= xmin) & (df["x"] <= xmax)
        & (df["y"] >= ymin) & (df["y"] <= ymax)
    )
    if len(bbox) >= 6:
        zmin, zmax = bbox[4], bbox[5]
        mask &= (df["z"] >= zmin) & (df["z"] <= zmax)
    return df[mask]


def _subsample_idx(n: int, cap: int, rng: np.random.Generator):
    """Return indices for a reproducible random subsample of size <= ``cap``."""
    if n <= cap:
        return np.arange(n)
    return rng.choice(n, size=cap, replace=False)


# ---------------------------------------------------------------------------
# Function 1 plots
# ---------------------------------------------------------------------------
def plot_partition_overview(df, fov_column, out_path, plot_format, dpi,
                            advanced, random_state):
    """MANDATORY, always-on, global subsampled scatter of RCB pseudo-FOVs.

    This is a *global chunking sanity check*, not a diagnostic spatial crop, so
    it is never bbox-gated. Subsampled for responsiveness.
    """
    rng = np.random.default_rng(random_state)
    idx = _subsample_idx(len(df), advanced.partition_overview_subsample, rng)
    df_p = df.iloc[idx]
    n_fovs = df[fov_column].nunique()

    fig = plt.figure(figsize=(9, 9))
    cof = discrete_color_map(df_p[fov_column])
    plt.scatter(df_p["x"], df_p["y"], c=df_p[fov_column].map(cof).tolist(),
                s=2, alpha=0.6, rasterized=True)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(f"Auto 2D partitions via RCB (K={n_fovs})")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    return save_mpl(fig, out_path, prefixed("partition_overview"), plot_format, dpi)


def plot_edge_weight_histogram(fss_list, out_path, plot_format, dpi, advanced,
                               random_state):
    """MANDATORY global QC: distribution of FSS edge weights (subsampled per FOV)."""
    rng = np.random.default_rng(random_state)
    edge_values = fss_list[0].data
    if len(fss_list) > 1:
        parts = []
        for m in fss_list:
            if len(m.data) > advanced.diag_hist_subsample:
                parts.append(rng.choice(m.data, size=advanced.diag_hist_subsample,
                                        replace=False))
            else:
                parts.append(m.data)
        edge_values = np.concatenate(parts)

    fig = plt.figure(figsize=(10, 6))
    sns.histplot(edge_values, bins=100, color="#5A7DC7", edgecolor="black",
                 alpha=0.7)
    plt.title("Distribution of Existing Edge Weights", fontsize=14,
              fontweight="bold")
    plt.xlabel("Edge Weight Score", fontsize=12)
    plt.ylabel("Frequency (Number of Edges)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    return save_mpl(fig, out_path, prefixed("fss_edge_weight_histogram"),
                    plot_format, dpi)


def plot_edge_weight_vs_distance(fss_list, fss_splits, df, fov_column, out_path,
                                 plot_format, dpi, advanced, random_state,
                                 max_edge_length=None):
    """MANDATORY global QC: FSS edge weight vs physical distance (jointplot).

    If ``max_edge_length`` is given, a dashed grey vertical line marks the
    distance cap that the unified filter applies (edges to its right are pruned).
    """
    rng = np.random.default_rng(random_state)
    knn_distances, edge_weights = [], []
    rows_by_fov = fov_row_positions(df, fov_column)
    xyz = df[["x", "y", "z"]].values
    for i, fss_i in enumerate(fss_list):
        fov = fss_splits[i]
        coords = xyz[rows_by_fov.get(fov, EMPTY_POSITIONS)]
        coo = fss_i.tocoo()
        if len(coo.data) > advanced.diag_jointplot_subsample:
            sel = rng.choice(len(coo.data), size=advanced.diag_jointplot_subsample,
                             replace=False)
            rows, cols, weights = coo.row[sel], coo.col[sel], coo.data[sel]
        else:
            rows, cols, weights = coo.row, coo.col, coo.data
        # Guard against any stale row/col indices exceeding the coord table.
        if coords.shape[0] == 0 or rows.size == 0:
            continue
        valid = (rows < coords.shape[0]) & (cols < coords.shape[0])
        rows, cols, weights = rows[valid], cols[valid], weights[valid]
        d = np.linalg.norm(coords[rows] - coords[cols], axis=1)
        knn_distances.extend(d)
        edge_weights.extend(weights)

    x = np.asarray(knn_distances)
    y = np.asarray(edge_weights)
    if x.size == 0:
        return []
    cap = advanced.diag_scatter_subsample
    if x.size > cap:
        sel = rng.choice(x.size, size=cap, replace=False)
        x, y = x[sel], y[sel]

    g = sns.JointGrid(x=x, y=y, height=8, space=0)
    g.plot_joint(sns.scatterplot, color="#5A7DC7", alpha=0.35, s=1, linewidth=0,
                 rasterized=True)
    g.plot_marginals(sns.histplot, bins=100, color="#5A7DC7", alpha=0.7)
    g.ax_joint.set_title("Edge Weight vs Physical Distance", fontsize=14,
                         fontweight="bold")
    g.ax_joint.set_xlabel("Distance", fontsize=12)
    g.ax_joint.set_ylabel("Fuzzy Simplicial Set Edge Weight", fontsize=12)
    if max_edge_length is not None:
        g.ax_joint.axvline(max_edge_length, color="grey", linestyle="--",
                           linewidth=1.5,
                           label=f"max_edge_length = {max_edge_length:g}")
        g.ax_joint.legend(fontsize=9, loc="upper right")
    g.ax_joint.grid(True, linestyle="--", alpha=0.6)
    g.figure.tight_layout()
    return save_mpl(g.figure, out_path, prefixed("fss_edge_weight_vs_distance"),
                    plot_format, dpi)


def plot_post_filter_weight_vs_distance(fss_list, fss_splits, df, fov_column,
                                        out_path, plot_format, dpi, advanced,
                                        random_state):
    """MANDATORY global QC: post-filter edge weight vs distance scatter."""
    rng = np.random.default_rng(random_state)
    all_dists, all_w = [], []
    rows_by_fov = fov_row_positions(df, fov_column)
    xyz32 = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    for fov_id, mat in zip(fss_splits, fss_list):
        coo = mat.tocoo()
        if coo.nnz == 0:
            continue
        pos = rows_by_fov.get(fov_id, EMPTY_POSITIONS)
        if pos.size == 0:
            continue
        coords = xyz32[pos]
        max_node = int(max(coo.row.max(), coo.col.max()))
        if coords.shape[0] <= max_node:
            continue
        n_sample = min(advanced.diag_jointplot_subsample, coo.nnz)
        sel = (rng.choice(coo.nnz, size=n_sample, replace=False)
               if coo.nnz > n_sample else np.arange(coo.nnz))
        d = np.linalg.norm(coords[coo.row[sel]] - coords[coo.col[sel]], axis=1)
        all_dists.append(d)
        all_w.append(coo.data[sel])

    if not all_dists:
        return []
    x = np.concatenate(all_dists)
    y = np.concatenate(all_w)
    fig = plt.figure(figsize=(10, 7))
    sns.scatterplot(x=x, y=y, s=2, alpha=0.3, color="#5A7DC7", linewidth=0,
                    rasterized=True)
    plt.xlabel("Physical distance")
    plt.ylabel("FSS edge weight")
    plt.title("Post-filter: edge weight vs physical distance")
    plt.tight_layout()
    return save_mpl(fig, out_path, prefixed("fss_post_filter_weight_vs_distance"),
                    plot_format, dpi)


def plot_fss_edges_in_bbox(fss_dict, df, fov_column, bbox, out_path,
                           plot_format, dpi):
    """OPTIONAL (bbox-gated): a single FOV-independent ROI FSS-edge plot (G1).

    Shows *all* transcripts and *all* FSS edges whose endpoints fall inside the
    (already internal-coordinate) ``bbox``, regardless of FOV, in **one** figure
    (``2c_fss_edges_bbox``). Skipped when ``bbox`` is None.

    Implementation notes:

    * FSS graphs are per-FOV with **local** node numbering, and FSS has no
      cross-FOV edges, so iterating per-FOV and collecting segments/nodes into
      global lists (using each node's real scaled x/y) is lossless.
    * ``bbox`` is already in internal (scaled + zero-shifted) coordinates — the
      raw→internal conversion happens once in ``api`` (§0.4) — so the in-bbox
      mask, node/edge positions and axis limits are all in the same frame.
    * Aesthetics (maintainer spec): edges coloured by weight with the seaborn
      ``rocket`` colormap, white α=0.6 nodes, black background. When ``html`` is
      in ``plot_format`` an interactive Plotly **3D** version (x/y/z) is written
      alongside the static 2D figure.
    """
    if bbox is None:
        return []
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

    all_segments = []          # 2D segments for matplotlib
    all_weights = []
    edge_p0, edge_p1 = [], []  # 3D endpoint coords for plotly
    node_x, node_y, node_z = [], [], []

    # Collect in-bbox nodes/edges across every FOV into global lists.
    rows_by_fov = fov_row_positions(df, fov_column)
    xyz = df[["x", "y", "z"]].to_numpy()
    for fov_id, mat in fss_dict.items():
        pos = rows_by_fov.get(fov_id, EMPTY_POSITIONS)
        if pos.size == 0:
            continue
        coords_xyz = xyz[pos]
        coords_xy = coords_xyz[:, :2]
        if coords_xyz.shape[0] != mat.shape[0]:
            # Row order must match the FSS node order; skip a stale FOV rather
            # than mis-map (should not happen post-filter).
            continue
        in_roi = (
            (coords_xy[:, 0] >= xmin) & (coords_xy[:, 0] <= xmax)
            & (coords_xy[:, 1] >= ymin) & (coords_xy[:, 1] <= ymax)
        )
        if int(in_roi.sum()) == 0:
            continue

        node_x.append(coords_xyz[in_roi, 0])
        node_y.append(coords_xyz[in_roi, 1])
        node_z.append(coords_xyz[in_roi, 2])

        coo = mat.tocoo()
        edge_in_roi = in_roi[coo.row] & in_roi[coo.col]
        if edge_in_roi.any():
            rows = coo.row[edge_in_roi]
            cols = coo.col[edge_in_roi]
            all_segments.append(np.stack([coords_xy[rows], coords_xy[cols]], axis=1))
            all_weights.append(coo.data[edge_in_roi])
            edge_p0.append(coords_xyz[rows])
            edge_p1.append(coords_xyz[cols])

    if not node_x:
        return []

    node_x = np.concatenate(node_x)
    node_y = np.concatenate(node_y)
    node_z = np.concatenate(node_z)
    segs = np.concatenate(all_segments, axis=0) if all_segments else np.empty((0, 2, 2))
    weights = np.concatenate(all_weights) if all_weights else np.empty(0)

    written = []

    # --- Static 2D matplotlib (pdf / png) -----------------------------------
    if any(f in plot_format for f in ("pdf", "png")):
        # Figure size from the converted bbox extents, with a sane scale + cap so
        # a large ROI does not produce a 100-inch canvas.
        x_ext = max(xmax - xmin, 1e-9)
        y_ext = max(ymax - ymin, 1e-9)
        scale = 8.0 / max(x_ext, y_ext)          # longest side -> ~8 inches
        figw = float(np.clip(x_ext * scale, 4, 20))
        figh = float(np.clip(y_ext * scale, 4, 20))

        rocket = sns.color_palette("rocket", as_cmap=True)
        fig, ax = plt.subplots(figsize=(figw, figh), facecolor="black")
        ax.set_facecolor("black")
        if len(segs):
            lc = LineCollection(segs, array=weights, cmap=rocket,
                                linewidths=0.5, alpha=1.0, clim=(0, 1))
            ax.add_collection(lc)
            cbar = fig.colorbar(lc, ax=ax, label="FSS edge weight")
            cbar.set_label("FSS edge weight", color="white")
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.get_yticklabels(), color="white")
        ax.scatter(node_x, node_y, s=1, alpha=0.6, c="white", zorder=3,
                   rasterized=True)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title("ROI — FSS edges (all FOVs, coloured by weight)", color="white")
        ax.set_xlabel("x", color="white"); ax.set_ylabel("y", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        plt.tight_layout()
        written += save_mpl(fig, out_path, prefixed("fss_edges_bbox"),
                            plot_format, dpi)

    # --- Interactive 3D Plotly (html) ---------------------------------------
    if "html" in plot_format:
        written += _plot_fss_edges_bbox_3d(
            edge_p0, edge_p1, weights, node_x, node_y, node_z, out_path,
            plot_format)
    return written


def _rocket_plotly_colorscale(n=32):
    """Sample the seaborn ``rocket`` colormap into a Plotly colorscale list.

    Plotly has no built-in ``rocket`` scale, so we sample the matplotlib cmap
    to reuse the exact same palette as the static 2c plot / the 6a panels.
    """
    import matplotlib.colors as mcolors
    cmap = sns.color_palette("rocket", as_cmap=True)
    return [[i / (n - 1), mcolors.to_hex(cmap(i / (n - 1)))] for i in range(n)]


def _plot_fss_edges_bbox_3d(edge_p0, edge_p1, weights, node_x, node_y, node_z,
                            out_path, plot_format, max_edges=30_000):
    """Interactive 3D Plotly version of the ROI FSS-edge plot (rocket, black bg)."""
    import plotly.graph_objects as go

    rocket_scale = _rocket_plotly_colorscale()
    fig = go.Figure()

    if edge_p0:
        p0 = np.concatenate(edge_p0, axis=0)
        p1 = np.concatenate(edge_p1, axis=0)
        w = np.asarray(weights)
        # Cap edge count so the html stays light.
        if p0.shape[0] > max_edges:
            sel = np.random.default_rng(0).choice(p0.shape[0], max_edges,
                                                  replace=False)
            p0, p1, w = p0[sel], p1[sel], w[sel]
        # Build x/y/z with None separators between edges; per-vertex colour = the
        # edge weight (endpoints share it, so each segment is a solid colour).
        n = p0.shape[0]
        ex = np.empty(n * 3); ey = np.empty(n * 3); ez = np.empty(n * 3)
        ex[0::3], ex[1::3], ex[2::3] = p0[:, 0], p1[:, 0], np.nan
        ey[0::3], ey[1::3], ey[2::3] = p0[:, 1], p1[:, 1], np.nan
        ez[0::3], ez[1::3], ez[2::3] = p0[:, 2], p1[:, 2], np.nan
        cvals = np.repeat(w, 3)
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color=cvals, colorscale=rocket_scale, cmin=0, cmax=1,
                      width=2, colorbar=dict(title="FSS edge weight")),
            connectgaps=False, hoverinfo="skip", name="edges"))

    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode="markers",
        marker=dict(size=1, color="white", opacity=0.6), name="transcripts",
        hoverinfo="skip"))

    fig.update_layout(
        title="ROI — FSS edges (all FOVs, coloured by weight)",
        paper_bgcolor="black", font=dict(color="white"), showlegend=False,
        scene=dict(
            xaxis=dict(title="x", backgroundcolor="black", color="white"),
            yaxis=dict(title="y", backgroundcolor="black", color="white"),
            zaxis=dict(title="z", backgroundcolor="black", color="white"),
            aspectmode="data"),
        width=900, height=700)
    return save_plotly(fig, out_path, prefixed("fss_edges_bbox"), plot_format)


# Static viewpoints for pdf/png (G4): views 1-2 are 3D perspectives (elev, azim);
# view 3 is a true 2D x-y scatter (not a 3D plot seen from above).
_3D_VIEWS = [(20, 45), (20, 135)]


def _bbox_extents(sub, bbox):
    """Return (x_ext, y_ext, z_ext) for 3D aspect ratios, from bbox if given."""
    if bbox is not None and len(bbox) >= 6:
        x_ext = max(bbox[1] - bbox[0], 1e-9)
        y_ext = max(bbox[3] - bbox[2], 1e-9)
        z_ext = max(bbox[5] - bbox[4], 1e-9)
    else:
        x_ext = max(float(sub["x"].max() - sub["x"].min()), 1e-9)
        y_ext = max(float(sub["y"].max() - sub["y"].min()), 1e-9)
        z_ext = max(float(sub["z"].max() - sub["z"].min()), 1e-9)
    return x_ext, y_ext, z_ext


def plot_metis_spatial(df, fov_column, bbox, out_path, plot_format, dpi):
    """OPTIONAL (bbox-gated): METIS partition (metis_id) spatial scatter.

    HTML -> interactive Plotly 3D; PDF/PNG -> three static matplotlib 3D views
    (G4). The z-axis is visually exaggerated 3x via the box/scene aspect ratio
    while the tick labels keep the true coordinate range (G3). Skipped when
    ``bbox`` is None.
    """
    sub = subset_bbox(df, bbox)
    if sub is None or sub.empty:
        return []
    written = []
    x_ext, y_ext, z_ext = _bbox_extents(sub, bbox)
    # One discrete mapping shared by all static views and the html output, so
    # every metis_id keeps the same colour everywhere (restricted to the ids
    # present in this bbox subset).
    cof = discrete_color_map(sub["metis_id"])
    pcolors = sub["metis_id"].map(cof).tolist()

    # --- Static matplotlib (pdf / png): two 3D views + one 2D view (G4) ------
    if any(f in plot_format for f in ("pdf", "png")):
        for vi, (elev, azim) in enumerate(_3D_VIEWS, start=1):
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(sub["x"], sub["y"], sub["z"], c=pcolors, s=5,
                       rasterized=True)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.set_title(f"METIS partition (metis_id) — view {vi}")
            # G3: stretch z visually by 3x without touching the data or ticks.
            ax.set_box_aspect((x_ext, y_ext, 3 * z_ext))
            ax.view_init(elev=elev, azim=azim)
            plt.tight_layout()
            written += save_mpl(fig, out_path,
                                prefixed("metis_spatial", f"_view{vi}"),
                                plot_format, dpi)

        # View 3: a genuine 2D x-y scatter (not a 3D plot viewed from above).
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(sub["x"], sub["y"], c=pcolors, s=4, rasterized=True)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_title("METIS partition (metis_id) — view 3 (2D x-y)")
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        written += save_mpl(fig, out_path,
                            prefixed("metis_spatial", "_view3"),
                            plot_format, dpi)

    # --- Interactive Plotly 3D (html) ---------------------------------------
    if "html" in plot_format:
        import plotly.express as px
        sub2 = sub.assign(metis_id_str=sub["metis_id"].astype(str))
        fig = px.scatter_3d(
            sub2, x="x", y="y", z="z", color="metis_id_str",
            hover_data=["target"], title="Interactive Gene Distribution",
            labels={"x": "X", "y": "Y", "z": "Z", "metis_id_str": "Metis ID"},
            color_discrete_map=_hex_color_map(cof),
        )
        fig.update_traces(marker=dict(size=5, opacity=1))
        # G3: manual aspect ratio, z stretched 3x; real z values stay on axis.
        fig.update_layout(
            legend_title_text="Metis ID", width=900, height=700,
            scene=dict(aspectmode="manual",
                       aspectratio=dict(x=x_ext, y=y_ext, z=3 * z_ext)))
        written += save_plotly(fig, out_path, prefixed("metis_spatial"),
                               plot_format)
    return written


def _repel_labels(texts, ax, time_lim=15.0):
    """Pull a set of ``Text`` labels apart so gene names never overlap.

    Thin wrapper over :func:`adjustText.adjust_text` that centralises the
    repelling parameters used by every labelled gene plot. The knobs follow the
    **adjustText >= 1.0** API (``expand`` / ``force_static`` / ``force_explode``
    / ``max_move``); the pre-1.0 spellings (``expand_text``, ``expand_points``,
    ``force_points``, ``lim``) are silently swallowed by modern versions into
    ``**kwargs`` and forwarded to the arrow patch, i.e. the repelling would
    quietly run on defaults and labels would stay on top of each other.

    ``max_move=None`` lifts the default 10-pixel-per-iteration cap so crowded
    labels can actually travel to free space, and ``time_lim`` gives the
    optimiser a real budget (the library default is a single second).

    Exactly-coincident labels are jittered by adjustText through the *global*
    numpy RNG, and that code path carries a stray ``print``; the global RNG state
    is therefore seeded and restored around the call (reproducible layouts, no
    side effect on the pipeline's own randomness) and stdout is muted.
    """
    if not texts:
        return
    import contextlib
    import io

    from adjustText import adjust_text
    rng_state = np.random.get_state()
    np.random.seed(0)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5, alpha=0.6),
                expand=(1.3, 1.6),          # bbox padding around each label
                force_text=(0.5, 1.0),      # label <-> label repulsion
                force_static=(0.3, 0.6),    # label <-> data-point repulsion
                force_explode=(0.4, 0.8),   # explosion out of dense clumps
                force_pull=(0.005, 0.005),  # weak pull back to the anchor point
                max_move=None,              # no per-iteration displacement cap
                min_arrow_len=2,
                ensure_inside_axes=True,
                time_lim=time_lim,
            )
    finally:
        np.random.set_state(rng_state)


def plot_stopword_scatter(stopword_stats, out_path, plot_format, dpi,
                          docfreq_cutoff=None, docfreq_threshold=None,
                          thr_spec=None, max_labels=60):
    """MANDATORY (gene-level, global): info-content vs document-frequency scatter.

    Genes fall in exactly two categories — ``kept`` (blue) and ``stopword``
    (red); detection is on the x-axis (document frequency), information content
    is plotted as a diagnostic only.

    ``docfreq_threshold`` draws the applied stop-word cutoff; if omitted it is
    inferred from the lowest-frequency flagged gene. NaN (stop words given as an
    explicit list, so there is no cutoff) draws no line. If ``docfreq_cutoff`` is
    given, a second vertical line marks the ``min_docfreq_for_stats`` threshold
    (U9): only genes to its right entered the median/MAD estimation.

    At most ``max_labels`` stop-word names are drawn (the most frequent ones), so
    that :func:`adjustText.adjust_text` can actually pull them apart.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # Two categories only: kept (blue) and stopword (red).
    palette = {"kept": "#1f77b4", "stopword": "#d62728"}
    for cat, color in palette.items():
        sel = stopword_stats["reason"].values == cat
        if sel.any():
            ax.scatter(stopword_stats.loc[sel, "doc_freq"],
                       stopword_stats.loc[sel, "info_content"],
                       s=14, alpha=0.75, color=color,
                       label=f"{cat} (n={int(sel.sum())})", rasterized=True)
    sw = stopword_stats[stopword_stats["reason"] == "stopword"]
    if docfreq_threshold is None and not sw.empty:
        docfreq_threshold = float(sw["doc_freq"].min())
    if docfreq_threshold is not None and np.isfinite(docfreq_threshold):
        thr_label = f"doc-freq threshold = {docfreq_threshold:.3f}"
        if thr_spec is not None:
            thr_label += f" ({thr_spec})"
        ax.axvline(docfreq_threshold, color="#d62728", linestyle="--",
                   alpha=0.6, label=thr_label)
    if docfreq_cutoff is not None:
        n_stat = int(stopword_stats.get(
            "used_for_stats", pd.Series(dtype=bool)).sum())
        ax.axvline(docfreq_cutoff, color="#4a4a4a", linestyle="--", alpha=0.6,
                   label=f"doc-freq cutoff = {docfreq_cutoff:g} "
                         f"(stats n={n_stat})")
    ax.set_xlabel("Document Frequency")
    ax.set_ylabel("Information Content (1 - normalized entropy)")
    labelled = sw.sort_values("doc_freq", ascending=False).head(int(max_labels))
    if len(sw) > len(labelled):
        console.info(f"Stop-word scatter: labelling the {len(labelled)} most "
                     f"frequent of {len(sw)} stop words (max_labels="
                     f"{int(max_labels)}) to keep the names readable")
    texts = [ax.text(row["doc_freq"], row["info_content"], row.name, fontsize=7)
             for _, row in labelled.iterrows()]
    _repel_labels(texts, ax)
    ax.set_title("Auto stop-word detection (document frequency)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return save_mpl(fig, out_path, prefixed("auto_stopwords"), plot_format, dpi)


def plot_gene_umap_lda(gene_emb_df, markers, out_path, plot_format, dpi,
                       random_state, n_jobs=1, umap_parallel=False):
    """MANDATORY (gene-level, global): UMAP of LDA gene embeddings with markers.

    When ``umap_parallel`` is False (default) the embedding is single-threaded
    and seeded, so the figure reproduces exactly. Set it True to run
    multi-threaded (``n_jobs`` workers) with no fixed ``random_state`` — faster
    in principle, but **non-deterministic** (U7), and this plot was measured at
    2.2 s either way, so there is nothing to gain.
    """
    from umap import UMAP
    if umap_parallel:
        reducer = UMAP(n_components=2, n_jobs=max(1, int(n_jobs)))
    else:
        reducer = UMAP(n_components=2, random_state=random_state, n_jobs=1)
    gene_emb = gene_emb_df[[c for c in gene_emb_df.columns
                            if c.startswith("LDA_")]].to_numpy()
    emb = reducer.fit_transform(gene_emb)
    plot_df = gene_emb_df.copy()
    plot_df["UMAP_1"], plot_df["UMAP_2"] = emb[:, 0], emb[:, 1]

    fig = plt.figure(figsize=(10, 8))
    ax = sns.scatterplot(x="UMAP_1", y="UMAP_2", data=plot_df, s=30, alpha=0.8,
                         color="#5A7DC7", edgecolor="k", linewidth=0.3,
                         rasterized=True)
    texts = []
    for _, row in plot_df.iterrows():
        if row["gene_name"] in markers:
            texts.append(ax.text(row["UMAP_1"], row["UMAP_2"], row["gene_name"],
                                 fontsize=9, color="#000000"))
    _repel_labels(texts, ax)
    plt.title("UMAP of LDA gene embeddings")
    plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
    plt.tight_layout()
    written = save_mpl(fig, out_path, prefixed("gene_embeddings_lda_umap"),
                       plot_format, dpi)
    return written, plot_df


def plot_raw_vs_diffused_topics(df, gene_emb_df, fov_column, bbox, lda_topics,
                                out_path, plot_format, dpi, dark_bg,
                                n_topics_to_plot=5):
    """OPTIONAL (bbox-gated): per-topic raw-vs-diffused spatial panels.

    Skipped when ``bbox`` is None. PDF/PNG via matplotlib; if HTML is requested
    the same panels are written as static images (Plotly does not add value for
    a side-by-side raster comparison).

    The panels use the same ``rocket`` colormap and **black background** as the
    ROI FSS-edge plot (2c) for a consistent look, regardless of ``dark_bg``.
    """
    from sklearn.preprocessing import MinMaxScaler
    sub = subset_bbox(df, bbox)
    if sub is None or sub.empty:
        return []
    raw_cols = [f"{t}_raw" for t in lda_topics[:n_topics_to_plot]]
    if not set(raw_cols) <= set(sub.columns):
        # keep_raw_topics=False: rebuild the raw values for the bbox only, the
        # same way compute_tx_topics built them (gene embedding of `target`,
        # genes without an embedding at eps).
        topics = lda_topics[:n_topics_to_plot]
        eps = gene_emb_df[lda_topics].min().min() * 0.1
        emb = gene_emb_df.set_index("gene_name")[topics]
        raw = emb.reindex(sub["target"].to_numpy()).fillna(eps).to_numpy()
        sub = sub.copy()
        sub[raw_cols] = raw
    # Force black background + rocket palette to match 2c (maintainer spec).
    fg = "black"
    txt = "white"
    rocket = sns.color_palette("rocket", as_cmap=True)
    written = []
    for topic in lda_topics[:n_topics_to_plot]:
        topic_raw = f"{topic}_raw"
        scaled_raw = MinMaxScaler().fit_transform(
            sub[topic_raw].values.reshape(-1, 1)).flatten()
        scaled_diff = MinMaxScaler().fit_transform(
            sub[topic].values.reshape(-1, 1)).flatten()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor=fg)
        for ax, (weights, title) in zip(
                axes, [(scaled_raw, "Raw (Before)"), (scaled_diff, "Diffused (After)")]):
            ax.set_facecolor(fg)
            sc = ax.scatter(sub["x"], sub["y"], c=weights, cmap=rocket, s=0.5,
                            alpha=1, vmin=0, vmax=1, rasterized=True)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(topic, color=txt)
            cbar.ax.yaxis.set_tick_params(color=txt)
            plt.setp(cbar.ax.get_yticklabels(), color=txt)
            ax.set_title(f"{topic} | {title}", color=txt, pad=15)
            ax.set_xlabel("X", color=txt); ax.set_ylabel("Y", color=txt)
            ax.tick_params(colors=txt)
            for spine in ax.spines.values():
                spine.set_color(txt)
            ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        written += save_mpl(fig, out_path,
                            prefixed("topic_raw_vs_diffused", f"_{topic}"),
                            plot_format, dpi)
    return written


# ---------------------------------------------------------------------------
# Function 2 plots
# ---------------------------------------------------------------------------
def plot_convergence_trace(trace, res_key, out_path, plot_format, dpi):
    """OPTIONAL global: smoothed-fraction vs loop for one resolution."""
    if not trace:
        return []
    fig = plt.figure(figsize=(7, 5))
    loops = list(range(1, len(trace) + 1))
    plt.plot(loops, trace, marker="o", color="#5A7DC7")
    plt.xlabel("Smoothing loop")
    plt.ylabel("Smoothed fraction")
    plt.title(f"FSS smoothing convergence ({res_key})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    return save_mpl(fig, out_path,
                    prefixed("smoothing_convergence", f"_{res_key}"),
                    plot_format, dpi)


def plot_cluster_spatial(df, fov_column, cluster_col, bbox, out_path,
                         plot_format, dpi, dark_bg):
    """OPTIONAL (bbox-gated): spatial scatter of final clusters for one resolution.

    HTML -> interactive Plotly 2D; PDF/PNG -> static matplotlib. Skipped when
    ``bbox`` is None.
    """
    sub = subset_bbox(df, bbox)
    if sub is None or sub.empty:
        return []
    labels = sub[cluster_col].fillna(-1).astype(int)
    written = []
    # Discrete categorical mapping: one colour per cluster id, restricted to the
    # ids present in this bbox subset, shared by the static plot, its legend and
    # the html output. -1 (unassigned) is forced to a fixed neutral grey.
    cluster_ids = sorted(labels.unique().tolist())
    num_clusters = int((np.asarray(cluster_ids) >= 0).sum())
    color_of = discrete_color_map(labels, unassigned=-1)
    point_colors = labels.map(color_of).tolist()

    if any(f in plot_format for f in ("pdf", "png")):
        from matplotlib.lines import Line2D
        fg = "black" if dark_bg else "white"
        txt = "white" if dark_bg else "black"
        # Legend (G2) caps to the top-N clusters by size with a "+k more" entry
        # so it stays readable when there are many clusters.

        fig, ax = plt.subplots(figsize=(8, 8), facecolor=fg)
        ax.set_facecolor(fg)
        ax.scatter(sub["x"], sub["y"], c=point_colors, s=5.0, alpha=0.8,
                   rasterized=True)
        ax.set_title(f"{cluster_col} | Post-FSS Refined Clusters (n={num_clusters})",
                     color=txt, pad=15, fontsize=14)
        ax.set_xlabel("X Coordinate", color=txt)
        ax.set_ylabel("Y Coordinate", color=txt)
        ax.tick_params(colors=txt)
        for spine in ax.spines.values():
            spine.set_color(txt)
        ax.set_aspect("equal", adjustable="box")

        # Legend (top-N clusters by size when there are many).
        TOP_N = 20
        order = labels.value_counts().index.tolist()
        legend_ids = order[:TOP_N]
        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                          markerfacecolor=color_of[cid], markeredgecolor="none",
                          label=str(cid)) for cid in legend_ids]
        extra = len(order) - len(legend_ids)
        if extra > 0:
            handles.append(Line2D([0], [0], marker="o", linestyle="",
                                  markersize=7, markerfacecolor="none",
                                  markeredgecolor=txt, label=f"+{extra} more"))
        leg = ax.legend(handles=handles, title="cluster", loc="upper left",
                        bbox_to_anchor=(1.02, 1), fontsize=8, markerscale=1.2,
                        frameon=False)
        plt.setp(leg.get_title(), color=txt)
        for t in leg.get_texts():
            t.set_color(txt)
        plt.tight_layout()
        written += save_mpl(fig, out_path,
                            prefixed("cluster_spatial", f"_{cluster_col}"),
                            plot_format, dpi)

    if "html" in plot_format:
        import plotly.express as px
        sub2 = sub.assign(cluster_str=labels.astype(str).values)
        fig = px.scatter(sub2, x="x", y="y", color="cluster_str",
                         title=f"{cluster_col} spatial clusters",
                         color_discrete_map=_hex_color_map(color_of))
        fig.update_traces(marker=dict(size=3, opacity=0.8))
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(legend_title_text="cluster")
        written += save_plotly(fig, out_path,
                               prefixed("cluster_spatial", f"_{cluster_col}"),
                               plot_format)
    return written


def plot_cluster_umap(sampled_data, sampled_clusters, res_key, out_path,
                      plot_format, dpi, advanced, random_state, n_jobs=1):
    """MANDATORY, always-on: UMAP of the clustered subsample, coloured by cluster.

    Not bbox-gated (it is an embedding, not spatial positions). Subsampled and
    rasterized to bound file size.

    One subsample of ``advanced.cluster_umap_subsample`` points (default 10,000)
    is both fitted and plotted, in a single ``fit_transform``. This replaces an
    earlier fit-on-1%-then-``transform``-200,000-points scheme that cost 43.1 s —
    10-11 % of the entire pipeline — for a diagnostic figure, nearly all of it in
    UMAP's slow ``transform`` pass. Fitting the points that are actually drawn is
    also the better embedding: nothing is projected through a smaller fit.

    When ``advanced.umap_parallel`` is True the embedding runs multi-threaded
    with no fixed ``random_state`` — non-deterministic (U7), and measured at only
    1.14x faster, hence not the default. The default (False) path is
    single-threaded and seeded, so the figure reproduces exactly.
    """
    import umap.umap_ as umap

    rng = np.random.default_rng(random_state)
    n = sampled_data.shape[0]

    # One subsample, fitted and plotted.
    plot_idx = _subsample_idx(n, advanced.cluster_umap_subsample, rng)

    if getattr(advanced, "umap_parallel", False):
        reducer = umap.UMAP(n_neighbors=advanced.umap_n_neighbors,
                            min_dist=advanced.umap_min_dist, n_components=2,
                            n_jobs=max(1, int(n_jobs)))
    else:
        reducer = umap.UMAP(n_neighbors=advanced.umap_n_neighbors,
                            min_dist=advanced.umap_min_dist, n_components=2,
                            random_state=random_state, n_jobs=1)
    emb = reducer.fit_transform(sampled_data[plot_idx])
    plot_df = pd.DataFrame({
        "UMAP_1": emb[:, 0], "UMAP_2": emb[:, 1],
        "cluster": np.asarray(sampled_clusters)[plot_idx],
    }).sample(frac=1, random_state=random_state)

    fig = plt.figure(figsize=(10, 8))
    color_of = discrete_color_map(plot_df["cluster"])
    sns.scatterplot(data=plot_df, x="UMAP_1", y="UMAP_2", hue="cluster",
                    palette=color_of, s=3, alpha=0.8, edgecolor=None,
                    rasterized=True)
    plt.title(f"UMAP of clustered subsample ({res_key})")
    plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left",
               markerscale=4)
    plt.tight_layout()
    return save_mpl(fig, out_path, prefixed("cluster_umap", f"_{res_key}"),
                    plot_format, dpi)
