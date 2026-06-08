"""Composite energies — ``G_composite = E_eref + (G_prev − E_prev)``.

COSMIC's optional energy-refinement step recomputes each motif's electronic
energy at a higher level of theory (``E_eref``, expensive) but reuses the
thermal correction from the cheaper previous stage (``G_prev − E_prev``).
The composite Gibbs energy is what then drives the Boltzmann population for
the final ranking.

This module attaches that composite energy to each structure in the dataset.
The previous-stage energies are located by matching basename against the
sibling cosmic stage's output directory.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, List, MutableMapping, Sequence

Record = MutableMapping[str, Any]


def apply_composite_energies(
    dataset: Sequence[Record],
    prev_out_dir: str,
) -> int:
    """Apply composite energies: G_composite = E_eref + (G_prev - E_prev).

    Reads QM output files from prev_out_dir/orca_out_*/ (or gaussian_out_*/, opt_out_*/)
    to get the previous-stage electronic and Gibbs energies, then computes the thermal
    correction and adds composite_gibbs to each matched molecule in dataset.

    Args:
        dataset: list of mol dicts (already extracted from eref outputs)
        prev_out_dir: path to the previous COSMIC base directory (e.g. "COSMIC_2")

    Returns:
        Number of structures that received a composite_gibbs value.

    Verbatim port of cosmic-v01's ``apply_composite_energies`` (4726-4849).
    """
    from cosmic_ascec.clustering.features.extractor import (
        extract_properties_from_logfile,
    )

    def _out_suffix_count(name: str):
        m = re.search(r'_(\d+)$', name)
        return int(m.group(1)) if m else 10**9

    def _out_type_rank(name: str):
        lower = name.lower()
        if lower.startswith("orca_out_"):
            return 0
        if lower.startswith("gaussian_out_"):
            return 1
        if lower.startswith("calc_out_"):
            return 2
        if lower.startswith("xtb_out_"):
            return 3
        if lower.startswith("opt_out_"):
            return 4
        return 9

    # Collect all .out/.log files from prev_out_dir output subfolders.
    # Deterministic ordering is important when multiple out folders exist:
    # prefer older/lower-count folders because they usually contain the
    # previous-stage thermal corrections used for composite energies.
    output_subdir_patterns = ["orca_out_*", "opt_out_*", "gaussian_out_*", "calc_out_*", "xtb_out_*"]
    output_subdirs: List[str] = []
    for pattern in output_subdir_patterns:
        for subdir in glob.glob(os.path.join(prev_out_dir, pattern)):
            if os.path.isdir(subdir):
                output_subdirs.append(subdir)

    output_subdirs = sorted(
        output_subdirs,
        key=lambda p: (
            _out_suffix_count(os.path.basename(p)),
            _out_type_rank(os.path.basename(p)),
            os.path.basename(p),
        ),
    )

    prev_files: List[str] = []
    for subdir in output_subdirs:
        prev_files.extend(sorted(glob.glob(os.path.join(subdir, "*.out"))))
        prev_files.extend(sorted(glob.glob(os.path.join(subdir, "*.log"))))

    if not prev_files:
        print(f"  Warning: No output files found in {prev_out_dir}/ for composite energy calculation")
        return 0

    # Build lookup: base_stem → {final_electronic_energy, gibbs_free_energy}
    prev_data: Dict[str, Dict[str, float]] = {}
    for fpath in prev_files:
        stem = os.path.splitext(os.path.basename(fpath))[0]
        # Keep the first valid match only, preserving preference for earlier folders.
        if stem in prev_data:
            continue
        props = extract_properties_from_logfile(fpath)
        if props:
            elec = props.get('final_electronic_energy')
            gibbs = props.get('gibbs_free_energy')
            if elec is not None and gibbs is not None:
                prev_data[stem] = {'elec': elec, 'gibbs': gibbs}

    if not prev_data:
        print(f"  Warning: Could not extract energies from {prev_out_dir}/ files")
        return 0

    # Build a stem alias map for umotif→motif renaming that happens between
    # refinement and energy refinement.  The motif/umotif XYZ files written by
    # cosmic contain the source stem in the comment line (line 2 of each frame),
    # e.g. "motif_02_opt (G = ...)".  If a direct stem match fails, we consult
    # this map to resolve the original prev-stage stem.
    stem_alias: dict = {}  # eref_stem → prev_stem
    umotif_dirs = sorted(glob.glob(os.path.join(prev_out_dir, "umotifs_*")))
    motif_dirs = sorted(glob.glob(os.path.join(prev_out_dir, "motifs_*")))
    source_dirs = umotif_dirs or motif_dirs
    if source_dirs:
        latest_dir = source_dirs[-1]
        for xyz_file in glob.glob(os.path.join(latest_dir, "*.xyz")):
            xyz_basename = os.path.splitext(os.path.basename(xyz_file))[0]  # e.g. umotif_01
            try:
                with open(xyz_file, 'r') as xf:
                    lines = xf.readlines()
                    if len(lines) >= 2:
                        # Comment line format: "motif_02_opt (G = -458.216632 Hartree ...)"
                        comment = lines[1].strip()
                        source_stem = comment.split()[0] if comment else ''
                        if source_stem and source_stem != xyz_basename:
                            # Map both umotif_01 and umotif_01_opt to source stem
                            stem_alias[xyz_basename] = source_stem
                            stem_alias[xyz_basename + '_opt'] = source_stem
                            stem_alias[xyz_basename + '_calc'] = source_stem
            except Exception:
                pass

    n_matched = 0
    for mol in dataset:
        stem = os.path.splitext(os.path.basename(mol.get('filename', '')))[0]
        # Try direct match first, then fall back to alias map
        lookup_stem = stem
        if stem not in prev_data and stem in stem_alias:
            lookup_stem = stem_alias[stem]
        if lookup_stem in prev_data:
            e_prev = prev_data[lookup_stem]['elec']
            g_prev = prev_data[lookup_stem]['gibbs']
            e_eref = mol.get('final_electronic_energy')
            if e_eref is not None:
                thermal_correction = g_prev - e_prev
                mol['composite_gibbs'] = e_eref + thermal_correction
                # Retain the components so the final Boltzmann report can break
                # down the composite into its DFT (previous-stage Gibbs) and
                # CCSD(T) (eref electronic) contributions.
                mol['composite_dft_gibbs'] = g_prev
                mol['composite_ccsdt_elec'] = e_eref
                mol['composite_thermal'] = thermal_correction
                n_matched += 1

    return n_matched


__all__ = ["apply_composite_energies"]
