"""Dendrogram cut-height selection — where to slice the UPGMA tree.

Hierarchical clustering produces a *tree*, not a partition; turning it into
clusters requires a cut height ``τ``. Three modes are supported:

* **Knee** (default, automatic) — find the elbow of the sorted merge-height
  curve, i.e. the height at which merges stop joining within-cluster
  structures and start joining genuinely different families. In ``"auto"``
  the detected knee is capped at the empirical ceiling τ=2.0; ``"knee"``
  forces the raw knee through even when it lands above that ceiling.
* **Mojena** — Mojena's upper-tail rule (mean + k · stddev) over merge heights.
* **Explicit** — user passes a float directly.

Two workflow modes (``opt-pearson`` / ``opt-spread``) rebuild ``τ`` from a
sibling COSMIC stage's resolved threshold, so a refinement step can keep the
same partition the preliminary step found.

The Pearson-similarity helpers attach an "% similarity to representative"
score to each cluster representative so the printable report can show how
tight each cluster is.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

from cosmic_ascec.clustering.energies import EnergyMode, sorting_energy
from cosmic_ascec.clustering.scaling import (
    effective_n_features,
    median_pairwise_distance,
)

# A user threshold is "auto", "knee", a float, or an (mode, params) tuple.
UserThreshold = Union[str, float, Tuple[str, Mapping[str, Any]]]

# Empirical ceiling on the automatic cut. A knee above this means the
# merge-height curve never really flattens, so "auto" distrusts it and falls
# back to 2.0; --th=knee bypasses the ceiling on purpose.
EMPIRICAL_TAU_CEILING = 2.0


# --------------------------------------------------------------------------- #
# Pearson similarity (d <-> r <-> %)                                           #
# --------------------------------------------------------------------------- #


def pearson_r_from_distance(d: float, n_eff: Optional[float]) -> Optional[float]:
    """Pearson correlation between two (weighted) z-standardised feature
    vectors, derived from their Euclidean distance d via
    r = 1 - d^2 / (2 n_eff). Returns None if n_eff is not positive; clamps
    the result to [-1, 1] for numerical safety.

    Verbatim port of cosmic-v01's ``_pearson_r_from_distance`` (4044-4056).
    """
    if n_eff is None or n_eff <= 0:
        return None
    r = 1.0 - (float(d) * float(d)) / (2.0 * float(n_eff))
    if r > 1.0:
        return 1.0
    if r < -1.0:
        return -1.0
    return r


def pearson_similarity_pct(d: float, n_eff: Optional[float]) -> Optional[float]:
    """Pearson-based similarity as a percentage in [0, 100]. Negative r is
    clamped at 0%. Returns None if n_eff is not positive.

    Verbatim port of cosmic-v01's ``_pearson_similarity_pct`` (4059-4065).
    """
    r = pearson_r_from_distance(d, n_eff)
    if r is None:
        return None
    return max(0.0, r) * 100.0


def attach_pearson_to_rep(
    mols: Sequence[MutableMapping[str, Any]],
    scaled_matrix: np.ndarray,
    cluster_labels: Sequence[Any],
    tau: Optional[float],
    mode: EnergyMode,
) -> None:
    """For each mol in *mols*, find the lowest-energy member of its cluster
    (the property-cluster representative) and record the Pearson similarity
    of the mol to that representative. Uses the weighted z-scored vectors in
    *scaled_matrix* (same row order as *mols*).

    Verbatim port of cosmic-v01's ``_attach_pearson_to_rep`` (4068-4089).
    cosmic-v01's ``_sorting_energy`` global becomes the explicit *mode*
    argument (**D-007**).
    """
    n_eff = effective_n_features(scaled_matrix)
    rep_by_label: Dict[Any, Tuple[int, MutableMapping[str, Any], Tuple[float, str]]] = {}
    for idx, lbl in enumerate(cluster_labels):
        current = rep_by_label.get(lbl)
        mol = mols[idx]
        if current is None or (sorting_energy(mol, mode), mol['filename']) < current[2]:
            rep_by_label[lbl] = (idx, mol, (sorting_energy(mol, mode), mol['filename']))
    for idx, mol in enumerate(mols):
        lbl = cluster_labels[idx]
        rep_idx, rep_mol, _ = rep_by_label[lbl]
        d = float(np.linalg.norm(scaled_matrix[idx] - scaled_matrix[rep_idx]))
        mol['_pearson_rep_filename'] = rep_mol['filename']
        mol['_pearson_rep_distance'] = d
        mol['_pearson_n_eff'] = n_eff
        mol['_pearson_rep_r'] = pearson_r_from_distance(d, n_eff)
        mol['_pearson_rep_pct'] = pearson_similarity_pct(d, n_eff)
        mol['_pearson_threshold_tau'] = float(tau) if tau is not None else None


def threshold_entry(
    tau: float,
    scaled_matrix: np.ndarray,
    source: str,
    group_label: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build a resolved-threshold summary entry with its Pearson equivalent
    and the median pairwise distance (scale anchor for --th=opt-spread).

    Verbatim port of cosmic-v01's ``_threshold_entry`` (lines 4092-4105).
    """
    n_eff = effective_n_features(scaled_matrix)
    d_med = median_pairwise_distance(scaled_matrix)
    return {
        'tau': float(tau),
        'n_eff': n_eff,
        'd_med': d_med,
        'r_thresh': pearson_r_from_distance(tau, n_eff),
        'pct_thresh': pearson_similarity_pct(tau, n_eff),
        'source': source,
        'group_label': group_label,
    }


# --------------------------------------------------------------------------- #
# Mojena diagnostic                                                            #
# --------------------------------------------------------------------------- #


def compute_mojena_threshold(linkage_matrix: np.ndarray, verbose: bool = False) -> Tuple[float, int]:
    """
    Mojena-style stopping rule (robust variant): median(h) + alpha *
    1.4826 * MAD(h). The 1.4826 factor is the MAD->sigma consistency constant
    for normal distributions. Computed purely for the diagnostic plot --
    never drives the cut (see resolve_clustering_threshold).

    Verbatim port of cosmic-v01's ``compute_mojena_threshold`` (4312-4355).
    """
    from scipy.cluster.hierarchy import fcluster
    from scipy.stats import median_abs_deviation

    MOJENA_ALPHA = 2.0

    heights = linkage_matrix[:, 2]
    n_samples = len(heights) + 1

    if n_samples <= 2 or np.all(heights < 1e-12):
        return float(heights[-1]) if len(heights) > 0 else 0.0, 1

    median_h = float(np.median(heights))
    mad_h = float(median_abs_deviation(heights))

    if mad_h > 1e-12:
        mojena_t = median_h + MOJENA_ALPHA * 1.4826 * mad_h
    else:
        mojena_t = float(np.mean(heights)) + MOJENA_ALPHA * float(np.std(heights))

    mojena_t = max(float(heights[0]) * 1.01, min(mojena_t, float(heights[-1]) * 0.99))

    labels = fcluster(linkage_matrix, t=mojena_t, criterion='distance')
    mojena_k = len(set(labels))

    if verbose:
        print(f"  Mojena diagnostic (robust, alpha={MOJENA_ALPHA}):")
        print(f"    median={median_h:.4f}, MAD={mad_h:.4f}")
        print(f"    Mojena threshold = {mojena_t:.4f} (n_c={mojena_k})")

    return mojena_t, mojena_k


# --------------------------------------------------------------------------- #
# Knee detection                                                               #
# --------------------------------------------------------------------------- #


def compute_knee_threshold(
    linkage_matrix: np.ndarray, verbose: bool = False
) -> Tuple[Optional[float], Optional[int], bool, str, Optional[int], Optional[float]]:
    """
    Auto-detect the dendrogram cut height by finding the elbow of the
    sorted merge-height curve: the point of maximum perpendicular distance
    to the secant line between the first and last merges, in unit-normalized
    coordinates (so the x and y axes have comparable scales).

    Returns ``(t_cut, k_resulting, ok, reason, knee_index, h_knee)``.

    Verbatim port of cosmic-v01's ``compute_knee_threshold`` (4358-4407).
    """
    from scipy.cluster.hierarchy import fcluster
    h = np.sort(linkage_matrix[:, 2])
    n = len(h)
    if n < 8:
        return None, None, False, f"too few merges ({n} < 8)", None, None
    if h[-1] <= 0 or h[-1] / max(h[0], 1e-9) < 5.0:
        return None, None, False, "no dynamic range in merge heights", None, None

    x_norm = np.arange(n, dtype=float) / (n - 1)
    y_norm = (h - h.min()) / (h.max() - h.min() + 1e-12)
    d = np.abs(x_norm - y_norm) / np.sqrt(2.0)
    k_star = int(np.argmax(d))
    h_knee = float(h[k_star])

    if k_star >= n - 1:
        t_cut = 0.5 * (h[-2] + h[-1])
    else:
        t_cut = 0.5 * (h[k_star] + h[k_star + 1])
    t_cut = float(t_cut)

    labels = fcluster(linkage_matrix, t=t_cut, criterion='distance')
    k_res = len(set(labels))
    n_samples = n + 1
    k_cap = max(2, n_samples // 2)
    if k_res < 2 or k_res > k_cap:
        return t_cut, k_res, False, f"knee gives unreasonable n_c={k_res}", k_star, h_knee

    if verbose:
        print(f"  Knee diagnostic: knee_index={k_star}/{n-1}, "
              f"h_knee={h_knee:.4f}, τ_cut={t_cut:.4f}, n_c={k_res}")
    return t_cut, int(k_res), True, "ok", k_star, h_knee


# --------------------------------------------------------------------------- #
# opt-* parameter transfer from a sibling cosmic run                           #
# --------------------------------------------------------------------------- #


_OPT_DETAIL_RE = re.compile(
    r"tau\s*=\s*([-+0-9.eE]+)\s*,\s*"
    r"r\s*>=?\s*([-+0-9.eE]+)\s*,\s*"
    r"N_f\s*=\s*([-+0-9.eE]+)\s*,\s*"
    r"d_med\s*=\s*([-+0-9.eE]+)\s*,\s*"
    r"source\s*=\s*([A-Za-z0-9_+\-]+)"
)

# Fallback: the human-readable threshold line the current code always writes,
# e.g. "COSMIC threshold: tau = 0.5243 (knee detection)" or "... (legacy)".
# Captures tau and the parenthesised method. This is what makes --th=opt work
# on summaries generated by the modular code (which omit the rich _OPT_DETAIL_RE
# line that only the old monolith emitted). For plain --th=opt only tau is
# needed; r / N_f / d_med stay None (they were only used by the deprecated
# opt-pearson / opt-spread transforms).
_OPT_TAU_FALLBACK_RE = re.compile(
    r"tau\s*=\s*([-+0-9.eE]+)\s*\(\s*([A-Za-z][^)]*?)\s*\)"
)


def parse_opt_params_from_summary(summary_path: str) -> Optional[Dict[str, Any]]:
    """Pull (tau, r, N_f, d_med, source) out of a sibling cosmic's
    clustering_summary.txt.

    Prefers the rich ``tau=…, r>=…, N_f=…, d_med=…, source=…`` detail line when
    present; otherwise falls back to the human-readable
    ``COSMIC threshold: tau = X (method)`` line that the current code writes,
    returning tau with r / n_eff / d_med as None. Returns None only if the file
    is missing or contains no parseable threshold at all.
    """
    if not os.path.isfile(summary_path):
        return None
    fallback: Optional[Dict[str, Any]] = None
    try:
        with open(summary_path, encoding='utf-8') as fh:
            for line in fh:
                m = _OPT_DETAIL_RE.search(line)
                if m:
                    return {
                        "tau": float(m.group(1)),
                        "r": float(m.group(2)),
                        "n_eff": float(m.group(3)),
                        "d_med": float(m.group(4)),
                        "source": m.group(5),
                    }
                if fallback is None:
                    fm = _OPT_TAU_FALLBACK_RE.search(line)
                    if fm:
                        method = fm.group(2).strip().split()[0].lower()
                        fallback = {
                            "tau": float(fm.group(1)),
                            "r": None,
                            "n_eff": None,
                            "d_med": None,
                            "source": method,
                        }
    except OSError:
        return None
    return fallback


def resolve_opt_params_from_sibling_cosmic(current_cwd: str) -> Optional[Dict[str, Any]]:
    """Look for the canonical post-opt cosmic dir as a sibling of *current_cwd*
    and parse its clustering_summary.txt for the resolved threshold parameters.

    Verbatim port of cosmic-v01's ``_resolve_opt_params_from_sibling_cosmic``
    (lines 4444-4473).
    """
    try:
        parent = os.path.dirname(os.path.abspath(current_cwd))
        entries = sorted(
            d for d in os.listdir(parent)
            if d.lower().startswith("cosmic")
            and os.path.isdir(os.path.join(parent, d))
        )
    except OSError:
        return None

    bare = next((d for d in entries if d.lower() == "cosmic"), None)
    candidates = [bare] if bare else []
    candidates.extend(d for d in entries if d != bare)

    self_abs = os.path.abspath(current_cwd)
    for name in candidates:
        cand_dir = os.path.join(parent, name)
        if os.path.abspath(cand_dir) == self_abs:
            continue
        params = parse_opt_params_from_summary(
            os.path.join(cand_dir, "clustering_summary.txt"))
        if params is not None:
            params["source_dir"] = cand_dir
            return params
    return None


# --------------------------------------------------------------------------- #
# Threshold resolution                                                         #
# --------------------------------------------------------------------------- #


def resolve_clustering_threshold(
    linkage_matrix: np.ndarray,
    user_threshold: UserThreshold,
    scaled_matrix: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[float, int, str]:
    """
    Decide the actual cut height for fcluster().

    *user_threshold* may be "auto", "knee", a float, or an ``(mode, params)``
    tuple (``opt-pearson`` / ``opt-spread``). Returns
    ``(t_cut, k_resulting, source)`` where source is one of "user", "knee",
    "knee-capped", "knee-uncapped", "legacy", "opt-pearson", "opt-spread".
    "knee-capped" means the knee was detected but its cut height exceeded the
    empirical ceiling τ=2.0, so τ=2.0 was applied instead (the raw knee is
    still surfaced on the diagnostic plot). "knee-uncapped" is the same
    situation under ``"knee"``, where the user asked for the raw knee to be
    applied regardless of the ceiling.

    Port of cosmic-v01's ``resolve_clustering_threshold`` (4476-4540), plus the
    forced-knee mode.
    """
    from scipy.cluster.hierarchy import fcluster

    if isinstance(user_threshold, tuple) and len(user_threshold) == 2:
        mode, params = user_threshold
        tau_opt = float(params.get("tau", 0.0))
        if mode == "opt-pearson":
            r_opt = float(params["r"])
            n_eff_ref = (effective_n_features(scaled_matrix)
                         if scaled_matrix is not None else None)
            if n_eff_ref is None or n_eff_ref <= 0:
                t = tau_opt
            else:
                t = float(np.sqrt(max(0.0, 2.0 * n_eff_ref * (1.0 - r_opt))))
            k = len(set(fcluster(linkage_matrix, t=t, criterion='distance')))
            if verbose:
                print(f"  --th=opt-pearson: r_opt={r_opt:.4f}, "
                      f"N_f_ref={n_eff_ref:.2f} → τ_ref={t:.4f}, n_c={k}")
            return t, k, "opt-pearson"
        if mode == "opt-spread":
            d_med_opt = float(params.get("d_med", 0.0))
            d_med_ref = (median_pairwise_distance(scaled_matrix)
                         if scaled_matrix is not None else 0.0)
            if d_med_opt > 0.0 and d_med_ref > 0.0:
                t = tau_opt * (d_med_ref / d_med_opt)
            else:
                t = tau_opt
            k = len(set(fcluster(linkage_matrix, t=t, criterion='distance')))
            if verbose:
                print(f"  --th=opt-spread: τ_opt={tau_opt:.4f}, "
                      f"d_med_opt={d_med_opt:.4f}, d_med_ref={d_med_ref:.4f} "
                      f"→ τ_ref={t:.4f}, n_c={k}")
            return t, k, "opt-spread"

    # --th=knee: use the detected knee whatever its height, i.e. skip the
    # empirical τ=2.0 ceiling that "auto" applies.
    force_knee = isinstance(user_threshold, str) and user_threshold.lower() == "knee"

    if not force_knee and user_threshold != "auto":
        t = float(user_threshold)
        k = len(set(fcluster(linkage_matrix, t=t, criterion='distance')))
        return t, k, "user"

    t, k, ok, reason, _, _ = compute_knee_threshold(linkage_matrix, verbose=verbose)
    if ok:
        if t > EMPIRICAL_TAU_CEILING:
            if force_knee:
                if verbose:
                    print(f"  --th=knee: knee τ={t:.4f} exceeds the empirical "
                          f"ceiling {EMPIRICAL_TAU_CEILING}; applying it anyway "
                          f"(n_c={k}).")
                return t, k, "knee-uncapped"
            # Knee landed above the empirical ceiling τ=2.0. A knee that high
            # means the merge-height curve never really flattens, so the elbow
            # is unreliable; fall back to the standard empirical cut but keep
            # "knee" provenance so the diagnostic plot still shows where the
            # detected knee was.
            k_cap = len(set(fcluster(linkage_matrix, t=EMPIRICAL_TAU_CEILING,
                                     criterion='distance')))
            if verbose:
                print(f"  Knee τ={t:.4f} exceeds empirical ceiling "
                      f"{EMPIRICAL_TAU_CEILING}; using empirical "
                      f"τ={EMPIRICAL_TAU_CEILING} (n_c={k_cap}).")
            return EMPIRICAL_TAU_CEILING, k_cap, "knee-capped"
        return t, k, "knee"
    if force_knee:
        # The user explicitly asked for the knee, so say out loud (not only
        # under -v) that no usable knee exists and 2.0 is being used instead.
        print(f"WARNING: --th=knee requested but knee detection failed "
              f"({reason}); falling back to legacy τ={EMPIRICAL_TAU_CEILING}.")
    elif verbose:
        print(f"  Knee detection skipped ({reason}); falling back to legacy "
              f"τ={EMPIRICAL_TAU_CEILING}.")
    k_l = len(set(fcluster(linkage_matrix, t=EMPIRICAL_TAU_CEILING,
                           criterion='distance')))
    return EMPIRICAL_TAU_CEILING, k_l, "legacy"


__all__ = [
    "EMPIRICAL_TAU_CEILING",
    "UserThreshold",
    "attach_pearson_to_rep",
    "compute_knee_threshold",
    "compute_mojena_threshold",
    "parse_opt_params_from_summary",
    "pearson_r_from_distance",
    "pearson_similarity_pct",
    "resolve_clustering_threshold",
    "resolve_opt_params_from_sibling_cosmic",
    "threshold_entry",
]
