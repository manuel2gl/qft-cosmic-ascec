"""Dendrogram and threshold-diagnostic plots.

Two PNGs are written per clustering run:

1. The **dendrogram** — the UPGMA tree with a horizontal line at the cut
   height that produced the partition.
2. The **threshold diagnostic** — the sorted merge-height curve, the applied
   cut, and the alternate (Mojena, standard) thresholds for reference, with
   a Pearson similarity-floor legend.

matplotlib is imported lazily inside the function so the rest of the
clustering pipeline does not need a display to run.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from cosmic_ascec.clustering.thresholds import pearson_similarity_pct

# 8 professional muted-dark colors; repeated with stride 3 (coprime with 8)
# so consecutive clusters never share adjacent palette entries.
_CLUSTER_PALETTE = [
    '#4878a8',  # steel blue
    '#b85450',  # brick red
    '#5a9e72',  # sage green
    '#8b6ab5',  # dusty violet
    '#c47d3a',  # warm ochre
    '#3a8f8f',  # dark teal
    '#b5607a',  # mauve
    '#7a8f3a',  # olive
]
_PALETTE_STRIDE = 3   # gcd(3, 8) == 1 → non-consecutive repetition
_ABOVE_CUT_COLOR = '#aaaaaa'


def _build_cluster_colors(linkage_matrix: np.ndarray, cut_height: float):
    """Return (link_color_func, color_map, cluster_labels) or None."""
    try:
        from scipy.cluster.hierarchy import fcluster
    except ImportError:
        return None

    n = linkage_matrix.shape[0] + 1
    cluster_labels = fcluster(linkage_matrix, t=cut_height, criterion='distance')
    uniq = sorted(set(int(c) for c in cluster_labels))
    p = len(_CLUSTER_PALETTE)
    color_map = {cid: _CLUSTER_PALETTE[(i * _PALETTE_STRIDE) % p] for i, cid in enumerate(uniq)}

    descendants: dict[int, set[int]] = {i: {i} for i in range(n)}
    for i in range(n, n + linkage_matrix.shape[0]):
        a, b = int(linkage_matrix[i - n][0]), int(linkage_matrix[i - n][1])
        descendants[i] = descendants[a] | descendants[b]

    def _link_color_func(k: int) -> str:
        k = int(k)
        leaves = descendants.get(k, set())
        cids = {int(cluster_labels[idx]) for idx in leaves}
        if len(cids) == 1:
            return color_map[next(iter(cids))]
        return _ABOVE_CUT_COLOR

    return _link_color_func, color_map, cluster_labels




def plot_annotated_dendrogram(
    linkage_matrix: np.ndarray,
    optimal_k: int,
    cut_height: float,
    filename: str,
    title_suffix: str = "",
    conf_labels: Optional[Sequence[str]] = None,
    mojena_threshold: Optional[float] = None,
    mojena_k: Optional[int] = None,
    n_eff: Optional[float] = None,
    knee_threshold: Optional[float] = None,
    knee_k: Optional[int] = None,
) -> None:
    """
    Save two plot files:
      1. Dendrogram with horizontal cut line -> filename (e.g., dendrogram.png)
      2. Mojena diagnostic -> threshold_diagnostic.png in the same directory

    Verbatim port of cosmic-v01's ``plot_annotated_dendrogram`` (4543-4723).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as _mticker
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import dendrogram

    # Check if all distances are effectively zero
    lm = linkage_matrix.copy()
    if np.all(lm[:, 2] == 0.0):
        lm[:, 2] += 1e-12

    # --- File 1: Dendrogram ---
    n_conf = len(conf_labels) if conf_labels is not None else lm.shape[0] + 1
    fig_width = max(12.0, min(n_conf * 0.18, 200 * 0.18))
    leaf_font = 11 if n_conf <= 50 else (9 if n_conf <= 120 else 8)

    fig1, ax1 = plt.subplots(1, 1, figsize=(fig_width, 8))

    color_data = _build_cluster_colors(lm, cut_height)
    dendro_kw: dict = dict(labels=conf_labels, leaf_rotation=90,
                            leaf_font_size=leaf_font, ax=ax1)
    if color_data is not None:
        dendro_kw['link_color_func'] = color_data[0]
    dendrogram(lm, **dendro_kw)
    for line in ax1.get_lines():
        line.set_linewidth(2.5)

    ax1.axhline(y=cut_height, color='#e74c3c', linestyle='--', linewidth=1.5)
    ax1.set_title(f"Hierarchical Clustering Dendrogram ({title_suffix})", fontsize=16)
    ax1.set_xlabel("Configuration", fontsize=18)
    ax1.set_ylabel("UPGMA linkage distance", fontsize=18)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.yaxis.set_major_formatter(_mticker.FormatStrFormatter('%.1f'))
    ax1.set_ylim(bottom=0)
    fig1.tight_layout()
    fig1.savefig(filename, dpi=150)
    plt.close(fig1)

    # --- File 2: Diagnostic plot (sorted merge heights + knee, applied cut, Mojena) ---
    heights_sorted = np.sort(linkage_matrix[:, 2])
    n_merges = len(heights_sorted)
    if n_merges < 3:
        return

    diag_filename = os.path.join(os.path.dirname(filename), "threshold_diagnostic.png")
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    merge_idx = np.arange(1, n_merges + 1)

    # Interpolate the exact cut-crossing point so both fills share a clean vertex.
    xf = merge_idx.astype(float)
    hf = heights_sorted.astype(float)
    _cross = np.where(
        (heights_sorted[:-1] <= cut_height) & (heights_sorted[1:] > cut_height)
    )[0]
    if len(_cross):
        _i = int(_cross[0])
        _frac = (cut_height - heights_sorted[_i]) / (heights_sorted[_i + 1] - heights_sorted[_i])
        _xc = merge_idx[_i] + _frac * (merge_idx[_i + 1] - merge_idx[_i])
        xf = np.insert(xf, _i + 1, _xc)
        hf = np.insert(hf, _i + 1, cut_height)

    # Shadow below the cut (blue) and above the cut (red).
    ax2.fill_between(xf, 0, np.minimum(hf, cut_height),
                     alpha=0.15, color='#3498db', linewidth=0, edgecolor='none')
    ax2.fill_between(xf, cut_height, hf, where=(hf >= cut_height), interpolate=True,
                     alpha=0.20, color='#e74c3c', linewidth=0, edgecolor='none')

    # Merge-height curve — not added to the legend.
    ax2.plot(merge_idx, heights_sorted, 'o-', color='#3498db',
             linewidth=0.9, markersize=3)

    n_above_cut = int(np.sum(heights_sorted > cut_height))
    ax2.axhline(y=cut_height, color='#e74c3c', linestyle='--', linewidth=2,
                label=rf'Applied cut $\tau$={cut_height:.2f} ($n_c$={n_above_cut + 1})')

    # Two redundancy guards so no legend entry just duplicates the applied cut at
    # the displayed (2-decimal) precision:
    #   * Standard τ=2.0 is dropped when the applied cut already sits at 2.0
    #     (legacy / knee-capped) — otherwise "Applied" and "Standard" both read
    #     "τ=2.00", the uninformative pair we want to avoid.
    #   * Knee is dropped when it coincides with the applied cut (the common
    #     non-capped case where the applied cut *is* the knee). It is kept, with
    #     its raw uncapped value, precisely when detection was capped to τ=2.0 —
    #     so the empirical line is replaced by the actual knee it was capped from.
    STANDARD_T = 2.0
    show_standard = round(cut_height, 2) != round(STANDARD_T, 2)
    show_knee = (knee_threshold is not None
                 and round(float(knee_threshold), 2) != round(cut_height, 2))

    if show_standard:
        n_standard = int(np.sum(heights_sorted > STANDARD_T)) + 1
        ax2.plot([], [], ' ',
                 label=rf'Standard $\tau$=2.00 ($n_c$={n_standard})')

    if show_knee:
        _knee_k = knee_k if knee_k is not None else int(np.sum(heights_sorted > knee_threshold)) + 1
        ax2.plot([], [], ' ',
                 label=rf'Knee $\tau$={knee_threshold:.2f} ($n_c$={_knee_k})')

    if mojena_threshold is not None:
        _moj_k = mojena_k if mojena_k is not None else int(np.sum(heights_sorted > mojena_threshold)) + 1
        ax2.plot([], [], ' ',
                 label=rf'Mojena $\tau$={mojena_threshold:.2f} ($n_c$={_moj_k})')

    ax2.set_xlabel("Merge Step (sorted)", fontsize=15)
    ax2.set_ylabel("UPGMA linkage distance", fontsize=15)
    ax2.set_title("Threshold Diagnostic (Merge Height Distribution)", fontsize=16)
    ax2.tick_params(labelsize=13)
    ax2.yaxis.set_major_formatter(_mticker.FormatStrFormatter('%.1f'))
    # Small margin on both ends so the first and last points are not clipped.
    ax2.set_xlim(0, n_merges + 1)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    # The main legend has no title; the similarity-floor legend carries the
    # "Similarity floor" title and is therefore one row taller. To make both
    # boxes the same height with their content vertically centred (no empty
    # title row), we grow the MAIN legend's borderpad to match the trust box.
    _MAIN_BP = 0.6
    _MAIN_LS = 0.6
    _TRUST_BP = 0.6
    _TRUST_LS = 0.6

    leg_main = ax2.legend(loc='upper left', fontsize=12,
                          labelspacing=_MAIN_LS, borderpad=_MAIN_BP)
    fig2.tight_layout()

    # Similarity-floor entries (Pearson floor for each threshold).
    trust_segments = []
    if n_eff is not None and n_eff > 0:
        def _fmt_trust_segment(label, t_val):
            pct = pearson_similarity_pct(t_val, n_eff)
            if pct is None:
                return f"{label} \u2192 N/A"
            return f"{label} \u2192 {pct:.1f}%"

        # Mirror the main-legend redundancy guards (show_standard / show_knee) so
        # the two boxes list the same thresholds.
        trust_segments.append(_fmt_trust_segment("Applied", cut_height))
        if show_standard:
            trust_segments.append(_fmt_trust_segment("Standard", STANDARD_T))
        if show_knee:
            trust_segments.append(_fmt_trust_segment("Knee", float(knee_threshold)))
        if mojena_threshold is not None:
            trust_segments.append(_fmt_trust_segment("Mojena", float(mojena_threshold)))

    if trust_segments:
        _trust_kwargs = dict(
            handles=[Patch(visible=False) for _ in trust_segments],
            labels=trust_segments,
            loc='upper left',
            fontsize=12,
            title='Similarity floor',
            title_fontsize=12,
            handlelength=0, handletextpad=0,
            borderaxespad=0,
            borderpad=_TRUST_BP,
            labelspacing=_TRUST_LS,
        )

        # 1. Probe the trust box height (fixed; independent of position).
        fig2.canvas.draw()
        probe = ax2.legend(**_trust_kwargs)
        fig2.canvas.draw()
        h_trust = probe.get_window_extent().height
        probe.remove()

        # 2. Grow the main legend's borderpad until it matches that height.
        #    Symmetric padding keeps the main entries vertically centred.
        fontsize_px = 12 * fig2.dpi / 72
        main_bp = _MAIN_BP
        for _ in range(12):
            leg_main = ax2.legend(loc='upper left', fontsize=12,
                                  labelspacing=_MAIN_LS, borderpad=main_bp)
            fig2.canvas.draw()
            delta_px = h_trust - leg_main.get_window_extent().height
            if abs(delta_px) <= 0.5:
                break
            new_bp = max(0.1, main_bp + delta_px / (2 * fontsize_px))
            if abs(new_bp - main_bp) < 0.002:
                break
            main_bp = new_bp

        # 3. Main box is final; anchor the trust box just past its right edge.
        ax2.add_artist(leg_main)
        leg_bbox_axes = leg_main.get_window_extent().transformed(
            ax2.transAxes.inverted())
        trust_anchor_x = min(leg_bbox_axes.x1 + 0.005, 0.55)
        trust_anchor_y = leg_bbox_axes.y1

        ax2.legend(
            bbox_to_anchor=(trust_anchor_x, trust_anchor_y),
            bbox_transform=ax2.transAxes,
            **_trust_kwargs,
        )

    fig2.savefig(diag_filename, dpi=150)
    plt.close(fig2)


__all__ = ["plot_annotated_dendrogram"]
