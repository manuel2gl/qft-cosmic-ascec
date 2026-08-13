"""``--data`` feature-vector dump — per-configuration matrices from a cache.

``cosmic --data data_cache_*.pkl`` is a stand-alone utility that reads a
cached list of parsed property dicts and writes the feature matrix three ways:

* ``features.csv`` — labelled, units in the header (human-readable).
* ``matrix.csv`` — bare numbers (spreadsheet import).
* ``matrix.npy`` — numpy binary (downstream analysis).

Columns that are entirely NaN are dropped, and a ``cluster`` column is
appended when ``clustering_summary.txt`` sits next to the cache (so the
exported matrix can be coloured by cluster membership).
"""

from __future__ import annotations

import os
import pickle
import re
from typing import Optional

import numpy as np

from cosmic_ascec.clustering.energies import HARTREE_TO_EV
from cosmic_ascec.clustering.features.feature_spec import (
    CLUSTERING_NUMERICAL_FEATURES,
    FEATURE_MAPPING,
    ROTATIONAL_CONSTANT_SUBFEATURES,
    labelled_column as _extract_labeled,
)


def run_data_extraction(pkl_path: str, out_dir: Optional[str] = None) -> int:
    """Dump per-configuration feature vectors from a data_cache_*.pkl file.

    Writes features.csv (labeled, units in header), matrix.csv (numeric, no
    header), and matrix.npy next to the cache (or to out_dir).  Returns 0 on
    success, nonzero on error.

    Verbatim port of cosmic-v01's ``run_data_extraction`` (lines 3741-3889).
    """
    if not os.path.isfile(pkl_path):
        print(f"Error: cache file not found: {pkl_path}")
        return 1

    try:
        with open(pkl_path, 'rb') as fh:
            cache = pickle.load(fh)
    except Exception as exc:
        print(f"Error: could not read cache file {pkl_path}: {exc}")
        return 1

    if isinstance(cache, dict) and 'successful' in cache:
        entries = cache.get('successful') or []
        skipped = cache.get('skipped') or []
    elif isinstance(cache, list):
        entries = cache
        skipped = []
    else:
        print(f"Error: unexpected cache format: {type(cache).__name__}")
        return 1

    if not entries:
        print("Error: cache contains no successful entries.")
        return 1

    cache_dir = os.path.dirname(os.path.abspath(pkl_path))
    target_dir = out_dir or cache_dir
    os.makedirs(target_dir, exist_ok=True)

    feature_columns = list(CLUSTERING_NUMERICAL_FEATURES) + list(ROTATIONAL_CONSTANT_SUBFEATURES)

    def _natural_key(entry):
        fname = entry.get('filename', '') or ''
        m = re.search(r'(\d+)', fname)
        return (int(m.group(1)) if m else 0, fname)

    entries = sorted(entries, key=_natural_key)

    filenames = []
    rows = []
    for entry in entries:
        filenames.append(entry.get('filename', '') or '')
        row = []
        for feat in feature_columns:
            val = float('nan')
            if feat.startswith('rotational_constants_'):
                axis_idx = {'A': 0, 'B': 1, 'C': 2}[feat[-1]]
                rc = entry.get('rotational_constants')
                if rc is not None and hasattr(rc, '__len__') and len(rc) > axis_idx:
                    try:
                        val = float(rc[axis_idx])
                    except (TypeError, ValueError):
                        val = float('nan')
            else:
                key = FEATURE_MAPPING.get(feat, feat)
                raw = entry.get(key)
                if raw is not None:
                    try:
                        val = float(raw)
                    except (TypeError, ValueError):
                        val = float('nan')
                if feat in ('homo_lumo_gap', 'homo_energy') and np.isfinite(val):
                    # Both orbital features are cached in eV; emit Hartree so the
                    # dump matches the unit FEATURE_UNITS declares for them.
                    val = val / HARTREE_TO_EV
            row.append(val)
        rows.append(row)

    matrix = np.array(rows, dtype=float)

    # Parse clustering_summary.txt (same dir as cache) for cluster labels.
    summary_path = os.path.join(cache_dir, 'clustering_summary.txt')
    cluster_map = {}
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, encoding='utf-8') as fh:
                current = None
                for line in fh:
                    m = re.match(r'^Cluster\s+(\d+)', line)
                    if m:
                        current = int(m.group(1))
                        continue
                    m = re.match(r'^\s*-\s+(\S+)', line)
                    if m and current is not None:
                        cluster_map[m.group(1)] = current
                        continue
                    if not line.strip():
                        current = None
        except Exception as exc:
            print(f"Warning: failed to parse {summary_path}: {exc}")
    else:
        print(f"Note: clustering_summary.txt not found at {summary_path}; cluster column will be blank.")

    features_csv = os.path.join(target_dir, 'features.csv')
    matrix_csv = os.path.join(target_dir, 'matrix.csv')
    matrix_npy = os.path.join(target_dir, 'matrix.npy')

    # Drop columns that are entirely NaN so Excel's import wizard can classify
    # every remaining column as Number.
    keep_mask = ~np.all(np.isnan(matrix), axis=0) if matrix.size else np.ones(len(feature_columns), dtype=bool)
    dropped_cols = [_extract_labeled(feature_columns[i]) for i in range(len(feature_columns)) if not keep_mask[i]]
    kept_features = [feature_columns[i] for i in range(len(feature_columns)) if keep_mask[i]]
    kept_matrix = matrix[:, keep_mask] if matrix.size else matrix

    # Drop the trailing cluster column when no labels were resolved.
    include_cluster = bool(cluster_map)

    header = ['filename'] + [_extract_labeled(f) for f in kept_features]
    if include_cluster:
        header.append('cluster')
    with open(features_csv, 'w', encoding='utf-8') as fh:
        fh.write(','.join(header) + '\n')
        for fname, row in zip(filenames, kept_matrix):
            fields = [fname]
            for v in row:
                fields.append('' if not np.isfinite(v) else f'{v:.9g}')
            if include_cluster:
                fields.append(str(cluster_map.get(fname, '')))
            fh.write(','.join(fields) + '\n')

    np.savetxt(matrix_csv, kept_matrix, fmt='%.9g', delimiter=',')
    np.save(matrix_npy, kept_matrix)

    labeled_populated = sum(1 for fn in filenames if fn in cluster_map)

    print(f"Wrote {features_csv}")
    print(f"Wrote {matrix_csv}")
    print(f"Wrote {matrix_npy}")
    print(f"Rows (configurations): {kept_matrix.shape[0]}")
    print(f"Feature columns: {kept_matrix.shape[1]} ({', '.join(kept_features)})")
    if dropped_cols:
        print(f"Dropped all-NaN columns: {', '.join(dropped_cols)}")
    if include_cluster:
        print(f"Cluster labels populated for {labeled_populated}/{len(filenames)} rows.")
    else:
        print("Cluster column omitted (no cluster labels found).")
    if skipped:
        print(f"Cache also recorded {len(skipped)} skipped file(s) (not included in output).")
    return 0


__all__ = ["run_data_extraction"]
