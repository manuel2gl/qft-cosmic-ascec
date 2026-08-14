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

from cosmic_ascec import levels as _levels

Record = MutableMapping[str, Any]


# Clustering features that the high-level energy-refinement (eref) single point
# does NOT produce but the previous (geometry-refinement) stage does. The eref
# step is an electronic single point (no Opt/Freq), so it carries an improved
# electronic energy, dipole, and frontier-orbital data, but lacks the thermal
# (Gibbs) and vibrational features. These are geometry-invariant between the two
# stages (the single point reuses the refined geometry verbatim), so they can be
# carried over from the matched previous-stage output without re-running freq.
#
# The rule is "prefer eref, complement with the previous stage": a key is only
# backfilled when it is missing from the eref record, so CCSD(T)'s electronic
# features are always kept and never overwritten by the cheaper DFT values.
_BACKFILL_FEATURE_KEYS = (
    "gibbs_free_energy",
    "first_vib_freq",
    "last_vib_freq",
    "homo_energy",
    "homo_lumo_gap",
    "dipole_moment",
    "vnn_nuclear_repulsion",
    "rotational_constants",
    "num_hydrogen_bonds",
    "average_hbond_distance",
    "std_hbond_distance",
    "average_hbond_angle",
)


def apply_composite_energies(
    dataset: Sequence[Record],
    prev_out_dir: str,
) -> int:
    """Apply composite energies: G_composite = E_eref + (G_prev - E_prev).

    Reads QM output files from prev_out_dir/orca_out_*/ (or gaussian_out_*/, opt_out_*/)
    to get the previous-stage electronic and Gibbs energies, then computes the thermal
    correction and adds composite_gibbs to each matched molecule in dataset.

    In addition to the composite Gibbs energy, this also **backfills the
    clustering features that the eref single point cannot produce** (Gibbs,
    vibrational frequencies, …) from the matched previous-stage output — see
    :data:`_BACKFILL_FEATURE_KEYS`. Without this, the final post-eref clustering
    would run on a degraded feature vector (only the electronic features survive
    a single point), losing the frequency-derived discriminators and collapsing
    genuinely distinct motifs. Backfilling restores the full feature set so the
    final partition stays consistent with the geometry-refinement stage while
    still ranking by the high-level composite energy.

    Args:
        dataset: list of mol dicts (already extracted from eref outputs)
        prev_out_dir: path to the previous COSMIC base directory (e.g. "COSMIC_2")

    Returns:
        Number of structures that received a composite_gibbs value.

    Based on cosmic-v01's ``apply_composite_energies`` (4726-4849); the
    feature-backfill block (eref is feature-poor, the previous DFT stage is not)
    is a v05 addition.
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
                # Retain the full property dict so the feature-backfill step can
                # recover the vibrational / thermal features the eref single
                # point does not produce.
                prev_data[stem] = {'elec': elec, 'gibbs': gibbs, 'props': props}

    if not prev_data:
        print(f"  Warning: Could not extract energies from {prev_out_dir}/ files")
        return 0

    # Build a stem alias map for umotif→motif renaming that happens between
    # refinement and energy refinement.  The motif/umotif XYZ files written by
    # cosmic contain the source stem in the comment line (line 2 of each frame),
    # e.g. "motif_02_opt (G = ...)".  If a direct stem match fails, we consult
    # this map to resolve the original prev-stage stem.
    stem_alias: dict = {}  # eref_stem → prev_stem
    # Most-refined folder present wins, so a prev_out_dir holding several stage
    # folders resolves to the one that actually fed this pass. ALL_FOLDER_GLOBS
    # is ordered u_motifs, umotifs (legacy), motifs, candidates; the legacy
    # spelling has to stay because runs made before the rename are still on disk
    # and must keep resolving.
    source_dirs = []
    for _pattern in _levels.ALL_FOLDER_GLOBS:
        source_dirs = sorted(glob.glob(os.path.join(prev_out_dir, _pattern)))
        if source_dirs:
            break
    if source_dirs:
        latest_dir = source_dirs[-1]
        for xyz_file in glob.glob(os.path.join(latest_dir, "*.xyz")):
            xyz_basename = os.path.splitext(os.path.basename(xyz_file))[0]  # e.g. umotif_01
            try:
                with open(xyz_file, 'r', encoding='utf-8', errors='replace') as xf:
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
    n_backfilled = 0
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
                # CCSD(T) (eref electronic) contributions. The previous-stage
                # electronic energy is kept too, so the report can show a
                # relative electronic energy for the geometry-refinement level
                # without reconstructing it as ``g_prev - thermal_correction``.
                mol['composite_dft_gibbs'] = g_prev
                mol['composite_ccsdt_elec'] = e_eref
                mol['composite_thermal'] = thermal_correction
                mol['composite_prev_elec'] = e_prev
                n_matched += 1

            # Backfill the clustering features the eref single point cannot
            # provide (Gibbs, vibrational frequencies, …) from the matched
            # previous-stage output. Only fill keys missing from the eref
            # record, so the high-level electronic features are kept verbatim.
            prev_props = prev_data[lookup_stem].get('props') or {}
            backfilled_here = []
            for key in _BACKFILL_FEATURE_KEYS:
                if mol.get(key) is not None:
                    continue  # eref already provides it — keep the better value
                prev_value = prev_props.get(key)
                if prev_value is not None:
                    mol[key] = prev_value
                    backfilled_here.append(key)
            if backfilled_here:
                mol['_backfilled_features'] = backfilled_here
                n_backfilled += 1

    if n_backfilled:
        print(f"  Backfilled feature-poor eref structures from {os.path.basename(prev_out_dir)}: "
              f"{n_backfilled}/{len(dataset)} (Gibbs / vibrational features carried over)")

    return n_matched


__all__ = ["apply_composite_energies"]
