"""Per-cluster ``.dat`` report writer.

For every cluster COSMIC writes one ``extracted_data/<cluster>.dat`` file: a
plain-text dossier with the RMSD context, the Pearson trust score, a
per-feature deviation analysis across the cluster, the electronic /
molecular / vibrational / hydrogen-bond descriptors for each member, the
pairwise heavy-atom RMSD matrix, and the final geometry of each structure.

The exact byte layout of this file (separators, field widths, ``N/A``
strings) is a long-standing contract with downstream tooling — preserve it
when editing.
"""

from __future__ import annotations

import os
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from cosmic_ascec.clustering.console import calculate_deviation_percentage, vprint
from cosmic_ascec.clustering.energies import (
    HARTREE_TO_EV,
    hartree_to_ev,
    hartree_to_kcal_mol,
)
from cosmic_ascec.clustering.features.geometric import (
    HB_MAX_DISTANCE,
    HB_MIN_ANGLE,
    HB_MIN_DISTANCE,
    atomic_number_to_symbol,
)
from cosmic_ascec.clustering.rmsd import calculate_rmsd
from cosmic_ascec.clustering.scaling import feature_value
from cosmic_ascec.clustering.thresholds import (
    pearson_similarity_pct as _pearson_similarity_pct,
)

Record = MutableMapping[str, Any]


def write_cluster_dat_file(
    dat_file_prefix: str,
    cluster_members_data: Sequence[Record],
    output_base_dir: str,
    rmsd_threshold_value: Optional[float] = None,
    hbond_count_for_original_cluster: Optional[int] = None,
    weights: Optional[Mapping[str, float]] = None,
    tolerances: Optional[Mapping[str, float]] = None,
    rmsd_only: bool = False,
    rmsd_heavy: bool = False,
) -> None:
    """
    Writes combined .dat file for cluster members, including comparison and
    RMSD context sections.

    Ported from cosmic-v01's ``write_cluster_dat_file`` (lines 2697-3155), with
    a condensed Pearson trust-score block (intro + similarity floor only) and a
    deviation analysis restricted to the active reduced feature vector
    (unused / N/A features are omitted rather than printed as ``N/A``).
    """
    if weights is None:
        weights = {}
    if tolerances is None:
        tolerances = {}

    num_configurations = len(cluster_members_data)

    dat_output_dir = os.path.join(output_base_dir, "extracted_data")
    os.makedirs(dat_output_dir, exist_ok=True)

    output_filename = os.path.join(dat_output_dir, f"{dat_file_prefix}.dat")

    def write_deviation_line(file_obj, label, values):
        valid_values = [value for value in values if value is not None]
        if len(valid_values) != len(cluster_members_data) or not valid_values:
            file_obj.write(f"  {label} %Dev: N/A\n")
            return
        file_obj.write(f"  {label} %Dev: {calculate_deviation_percentage(valid_values):.2f}%\n")

    def write_scalar_descriptor_line(file_obj, label, value, formatter):
        if value is None:
            file_obj.write(f"        {label}: N/A\n")
            return
        file_obj.write(f"        {label}: {formatter(value)}\n")

    with open(output_filename, 'w', newline='\n', encoding='utf-8') as f:
        f.write("=" * 90 + "\n\n")

        rmsd_context_printed = False

        if rmsd_threshold_value is not None and cluster_members_data and '_first_rmsd_context_listing' in cluster_members_data[0] and cluster_members_data[0]['_first_rmsd_context_listing'] is not None:
            initial_rmsd_context = cluster_members_data[0]['_first_rmsd_context_listing']
            if rmsd_only:
                f.write(f"RMSD Clustering Context (cut at {rmsd_threshold_value:.3f} Å):\n")
                f.write("Configurations in this RMSD cluster:\n")
            else:
                f.write("Initial Clustering RMSD Context (Before Refinement):\n")
                f.write("Configurations from the original property cluster:\n")
            for item in initial_rmsd_context:
                rmsd_val_str = f"({item['rmsd_to_rep']:.3f} Å)" if item['rmsd_to_rep'] is not None else "(N/A)"
                f.write(f"    - {item['filename']} {rmsd_val_str}\n")
            f.write("\n")

            original_prop_cluster_label = cluster_members_data[0].get('_initial_cluster_label', 'N/A')
            parent_global_cluster_id_for_display = cluster_members_data[0].get('_parent_global_cluster_id', 'N/A')

            hbond_context = f" (H-bonds {hbond_count_for_original_cluster})" if hbond_count_for_original_cluster is not None else ""
            if rmsd_only:
                f.write("RMSD values relative to lowest energy representative of this RMSD cluster")
            else:
                f.write(f"RMSD values relative to lowest energy representative of initial property group")
            f.write("\n\n")
            rmsd_context_printed = True

        if rmsd_threshold_value is not None and cluster_members_data and \
           cluster_members_data[0].get('_rmsd_pass_origin') == 'second_pass_formed' and \
           cluster_members_data[0].get('_second_rmsd_context_listing') is not None:

            second_rmsd_context = cluster_members_data[0]['_second_rmsd_context_listing']
            second_rmsd_rep_filename = cluster_members_data[0].get('_second_rmsd_rep_filename', 'N/A')

            f.write("Second RMSD Clustering Context:\n")

            parent_global_cluster_id_for_display = cluster_members_data[0].get('_parent_global_cluster_id', 'N/A')

            if num_configurations == 1:
                f.write(f"This single configuration was either an outlier or remained a singleton after a second RMSD clustering step (threshold: {rmsd_threshold_value:.3f} Å).\n")
                f.write(f"  It originated from original Cluster {parent_global_cluster_id_for_display}.\n")

                self_rmsd_info = next((item for item in second_rmsd_context if item['filename'] == cluster_members_data[0]['filename']), None)
                if self_rmsd_info and self_rmsd_info['rmsd_to_rep'] is not None:
                    f.write(f"RMSD to its second-level cluster's representative ({second_rmsd_rep_filename}): {self_rmsd_info['rmsd_to_rep']:.3f} Å\n")
                else:
                    f.write(f"RMSD to its second-level cluster's representative ({second_rmsd_rep_filename}): N/A\n")

            else:
                f.write(f"This cluster was formed by a second RMSD clustering step (threshold: {rmsd_threshold_value:.3f} Å).\n")
                f.write(f"  It originated from original Cluster {parent_global_cluster_id_for_display}.\n")
                f.write(f"Representative for this second-level cluster: {second_rmsd_rep_filename}\n")
                f.write("RMSD values relative to this second-level cluster's representative:\n")

                for item in second_rmsd_context:
                    rmsd_val_str = f"({item['rmsd_to_rep']:.3f} Å)" if item['rmsd_to_rep'] is not None else "(N/A)"
                    f.write(f"    - {item['filename']} {rmsd_val_str}\n")
            f.write("\n")
            rmsd_context_printed = True

        # Separator after RMSD context (if any was printed)
        if rmsd_context_printed:
            f.write("=" * 90 + "\n\n")

        # 4. Cluster Summary Header (ALWAYS print this)
        f.write(f"Cluster (represented by: {dat_file_prefix}) ({num_configurations} configurations)\n\n")
        for mol_data in cluster_members_data:
            f.write(f"    - {mol_data['filename']}\n")
        f.write("\n")

        # 4b. Pearson similarity to initial property-cluster representative
        _pearson_available = any(
            m.get('_pearson_rep_r') is not None for m in cluster_members_data
        )
        if _pearson_available:
            rep_filename = next(
                (m.get('_pearson_rep_filename') for m in cluster_members_data
                 if m.get('_pearson_rep_filename')),
                None,
            )
            tau_val = next(
                (m.get('_pearson_threshold_tau') for m in cluster_members_data
                 if m.get('_pearson_threshold_tau') is not None),
                None,
            )
            n_eff_val = next(
                (m.get('_pearson_n_eff') for m in cluster_members_data
                 if m.get('_pearson_n_eff')),
                None,
            )

            f.write("Pearson Similarity to Cluster Representative\n")
            f.write("--------------------------------------------\n")
            f.write("  Each cluster has a representative (the lowest-energy member).\n")
            f.write("  For every member we measure how similar it is to that representative\n")
            f.write("  using the Pearson identity derived from the weighted, z-standardised\n")
            f.write("  feature vectors.\n")
            f.write("\n")

            if tau_val is not None and n_eff_val:
                pct_th = _pearson_similarity_pct(tau_val, n_eff_val)
            else:
                pct_th = None

            if pct_th is not None:
                f.write(f"      Similarity floor: {pct_th:.1f}%\n")
                f.write("\n")

            f.write("  Per-member similarity to representative:\n")
            _name_w = max(
                (len(m.get('filename', '') or '') for m in cluster_members_data),
                default=16,
            )
            _name_w = max(_name_w, 16)
            for mol_data in cluster_members_data:
                fname = mol_data.get('filename', '') or ''
                pct = mol_data.get('_pearson_rep_pct')
                d = mol_data.get('_pearson_rep_distance')
                r_val = mol_data.get('_pearson_rep_r')
                is_rep = fname == rep_filename
                marker = "  (representative)" if is_rep else ""
                if pct is None or d is None or r_val is None:
                    f.write(f"    {fname:<{_name_w}} :  similarity = N/A{marker}\n")
                else:
                    f.write(
                        f"    {fname:<{_name_w}} :  "
                        f"d = {d:6.3f}   r = {r_val:6.3f}   "
                        f"similarity = {pct:6.1f} %{marker}\n"
                    )
            f.write("\n")

        # 5. Deviation Analysis (ONLY for clusters with >1 configuration)
        if num_configurations > 1:
            # Dynamically detect which features are NOT available in all cluster members
            _zero_weight = {k for k, v in (weights or {}).items() if v == 0.0}

            # All deviation entries: (display_name, data_extractor, feature_key_for_filter).
            # Order and membership track feature_spec.FEATURE_COLUMNS (the cosmic
            # vector contract) — keep them in sync; do not reintroduce lumo_energy
            # or radius_of_gyration, which were dropped from the v04 vector.
            _deviation_entries = [
                ("Electronic Energy (Hartree)", lambda d: d.get('final_electronic_energy'), "electronic_energy"),
                ("Gibbs Free Energy (Hartree)", lambda d: d.get('gibbs_free_energy'), "gibbs_free_energy"),
                ("HOMO Energy (Hartree)", lambda d: d.get('homo_energy'), "homo_energy"),
                ("HOMO-LUMO Gap (Hartree)", lambda d: d.get('homo_lumo_gap'), "homo_lumo_gap"),
                ("Dipole Moment (Debye)", lambda d: d.get('dipole_moment'), "dipole_moment"),
                ("Nuclear Repulsion (Hartree)", lambda d: d.get('vnn_nuclear_repulsion'), "vnn_nuclear_repulsion"),
                ("First Vibrational Frequency (cm^-1)", lambda d: d.get('first_vib_freq'), "first_vib_freq"),
                ("Last Vibrational Frequency (cm^-1)", lambda d: d.get('last_vib_freq'), "last_vib_freq"),
                ("Number of Hydrogen Bonds", lambda d: d.get('num_hydrogen_bonds'), "num_hydrogen_bonds"),
                # Read through feature_value so a member with no hydrogen bonds
                # shows the 0.0 the clustering vector actually used, instead of
                # the whole row being reported as an unused feature.
                ("Average H-Bond Distance (Å)", lambda d: feature_value(d, 'average_hbond_distance'), "average_hbond_distance"),
                ("Std H-Bond Distance (Å)", lambda d: feature_value(d, 'std_hbond_distance'), "std_hbond_distance"),
                ("Average H-Bond Angle (°)", lambda d: feature_value(d, 'average_hbond_angle'), "average_hbond_angle"),
                ("Rotational Constant A (cm^-1)", lambda d: d['rotational_constants'][0] if d.get('rotational_constants') is not None and isinstance(d.get('rotational_constants'), np.ndarray) and len(d.get('rotational_constants')) == 3 else None, "rotational_constants_A"),
                ("Rotational Constant B (cm^-1)", lambda d: d['rotational_constants'][1] if d.get('rotational_constants') is not None and isinstance(d.get('rotational_constants'), np.ndarray) and len(d.get('rotational_constants')) == 3 else None, "rotational_constants_B"),
                ("Rotational Constant C (cm^-1)", lambda d: d['rotational_constants'][2] if d.get('rotational_constants') is not None and isinstance(d.get('rotational_constants'), np.ndarray) and len(d.get('rotational_constants')) == 3 else None, "rotational_constants_C"),
            ]

            # Dynamically detect features not available in all cluster members
            _feat_display_map = {
                'electronic_energy': 'Electronic Energy', 'gibbs_free_energy': 'Gibbs Free Energy',
                'homo_energy': 'HOMO Energy', 'homo_lumo_gap': 'HOMO-LUMO Gap',
                'dipole_moment': 'Dipole Moment',
                'vnn_nuclear_repulsion': 'Nuclear Repulsion',
                'first_vib_freq': 'First Vibrational Frequency',
                'last_vib_freq': 'Last Vibrational Frequency',
                'num_hydrogen_bonds': 'Number of Hydrogen Bonds',
                'average_hbond_distance': 'Average H-Bond Distance',
                'std_hbond_distance': 'Std H-Bond Distance',
                'average_hbond_angle': 'Average H-Bond Angle',
                'rotational_constants_A': 'Rotational Constant A',
                'rotational_constants_B': 'Rotational Constant B',
                'rotational_constants_C': 'Rotational Constant C',
            }
            _missing_features = set()
            for _, extractor, feat_key in _deviation_entries:
                values = [extractor(d) for d in cluster_members_data]
                if not all(v is not None for v in values):
                    _missing_features.add(feat_key)
            _all_excluded = _missing_features | _zero_weight

            if _all_excluded:
                _excluded_display = [_feat_display_map.get(k, k) for k in sorted(_all_excluded)]
                f.write(f"\nDynamic reduced feature vector.\n")
                f.write(f"Features not used: {', '.join(_excluded_display)}\n")

            f.write("\nDeviation Analysis (Max-Min / |Mean|):\n")
            for display_name, extractor, feat_key in _deviation_entries:
                # Only report features that are actually part of the reduced
                # vector: skip both zero-weight and missing (N/A) features.
                if feat_key in _all_excluded:
                    continue
                values = [extractor(d) for d in cluster_members_data]
                write_deviation_line(f, display_name, values)

            # --- Weights and tolerances display order ---
            weight_display_order = [
                ("electronic_energy", "Electronic Energy", "final_electronic_energy"),
                ("gibbs_free_energy", "Gibbs Free Energy", "gibbs_free_energy"),
                ("homo_energy", "HOMO Energy", "homo_energy"),
                ("homo_lumo_gap", "HOMO-LUMO Gap", "homo_lumo_gap"),
                ("dipole_moment", "Dipole Moment", "dipole_moment"),
                ("vnn_nuclear_repulsion", "Nuclear Repulsion", "vnn_nuclear_repulsion"),
                ("first_vib_freq", "First Vibrational Frequency", "first_vib_freq"),
                ("last_vib_freq", "Last Vibrational Frequency", "last_vib_freq"),
                ("num_hydrogen_bonds", "Number of Hydrogen Bonds", "num_hydrogen_bonds"),
                ("average_hbond_distance", "Average H-Bond Distance", "average_hbond_distance"),
                ("std_hbond_distance", "Std H-Bond Distance", "std_hbond_distance"),
                ("average_hbond_angle", "Average H-Bond Angle", "average_hbond_angle"),
                ("rotational_constants_A", "Rotational Constant A", "rotational_constants"),
                ("rotational_constants_B", "Rotational Constant B", "rotational_constants"),
                ("rotational_constants_C", "Rotational Constant C", "rotational_constants"),
            ]
            # Filter out all excluded features (freq-dependent + zero-weight)
            weight_display_order = [(k, dn, dk) for k, dn, dk in weight_display_order if k not in _all_excluded]

            # Print clustering weights applied
            f.write("\nClustering Weights Applied:\n")
            for feature_key, feature_display_name, data_key in weight_display_order:
                weight_value = weights.get(feature_key, 1.0)
                f.write(f"  {feature_display_name}: {weight_value:.2f}\n")

            f.write("\n")

            # Print clustering tolerances applied (if any non-default tolerances exist)
            has_custom_tolerances = any(tolerances.get(key, 0.0) != 0.0 for key, _, _ in weight_display_order)
            if has_custom_tolerances:
                f.write("Clustering Absolute Tolerances Applied:\n")
                tolerances_printed = False
                for feature_key, feature_display_name, data_key in weight_display_order:
                    tol_value = tolerances.get(feature_key, 0.0)
                    if tol_value != 0.0:
                        if abs(tol_value) < 1e-5:
                            tol_str = f"{tol_value:.7f}".rstrip('0').rstrip('.')
                        elif abs(tol_value) < 1e-3:
                            tol_str = f"{tol_value:.6f}".rstrip('0').rstrip('.')
                        elif abs(tol_value) < 0.1:
                            tol_str = f"{tol_value:.5f}".rstrip('0').rstrip('.')
                        else:
                            tol_str = f"{tol_value:.4f}".rstrip('0').rstrip('.')
                        f.write(f"  {feature_display_name}: {tol_str}\n")
                        tolerances_printed = True

                if not tolerances_printed:
                    f.write("  None\n")
                f.write("\n")

        # Separator before the detailed descriptor comparison section
        f.write("=" * 90 + "\n\n")

        # 6. Detailed Descriptors Comparison for each structure
        f.write("Electronic configuration descriptors:\n")
        for mol_data in cluster_members_data:
            f.write(f"    {mol_data['filename']}:\n")
            write_scalar_descriptor_line(
                f,
                "Final Electronic Energy",
                mol_data.get('final_electronic_energy'),
                lambda value: f"{value:.6f} Hartree ({hartree_to_kcal_mol(value):.2f} kcal/mol, {hartree_to_ev(value):.2f} eV)"
            )
            write_scalar_descriptor_line(
                f,
                "Gibbs Free Energy",
                mol_data.get('gibbs_free_energy'),
                lambda value: f"{value:.6f} Hartree ({hartree_to_kcal_mol(value):.2f} kcal/mol, {hartree_to_ev(value):.2f} eV)"
            )
            write_scalar_descriptor_line(f, "HOMO Energy (Hartree)", mol_data.get('homo_energy'), lambda value: f"{value:.6f}")
            write_scalar_descriptor_line(f, "HOMO-LUMO Gap (Hartree)", mol_data.get('homo_lumo_gap'), lambda value: f"{value / HARTREE_TO_EV:.6f}")
            write_scalar_descriptor_line(f, "Nuclear Repulsion (Hartree)", mol_data.get('vnn_nuclear_repulsion'), lambda value: f"{value:.6f}")
        f.write("\n")

        f.write("Molecular configuration descriptors:\n")
        for mol_data in cluster_members_data:
            f.write(f"    {mol_data['filename']}:\n")
            write_scalar_descriptor_line(f, "Dipole Moment (Debye)", mol_data.get('dipole_moment'), lambda value: f"{value:.6f}")
            rc = mol_data.get('rotational_constants')
            if rc is not None and isinstance(rc, np.ndarray) and rc.ndim == 1 and len(rc) == 3:
                f.write(f"        Rotational Constants (cm^-1): {rc[0]:.6f}, {rc[1]:.6f}, {rc[2]:.6f}\n")
            else:
                f.write("        Rotational Constants (cm^-1): N/A\n")
            write_scalar_descriptor_line(f, "Average H-Bond Distance (Å)", mol_data.get('average_hbond_distance'), lambda value: f"{value:.6f}")
            write_scalar_descriptor_line(f, "Std H-Bond Distance (Å)", mol_data.get('std_hbond_distance'), lambda value: f"{value:.6f}")
            write_scalar_descriptor_line(f, "Average H-Bond Angle (°)", mol_data.get('average_hbond_angle'), lambda value: f"{value:.6f}")
            write_scalar_descriptor_line(f, "Number of Hydrogen Bonds", mol_data.get('num_hydrogen_bonds'), lambda value: f"{int(value)}")
        f.write("\n")

        f.write("Vibrational frequency summary:\n")
        for mol_data in cluster_members_data:
            f.write(f"    {mol_data['filename']}:\n")
            if mol_data.get('_has_freq_calc', False):
                f.write(f"        Number of imaginary frequencies: {mol_data.get('num_imaginary_freqs', 'N/A')}\n")
                write_scalar_descriptor_line(f, "First Vibrational Frequency (cm^-1)", mol_data.get('first_vib_freq'), lambda value: f"{value:.2f}")
                write_scalar_descriptor_line(f, "Last Vibrational Frequency (cm^-1)", mol_data.get('last_vib_freq'), lambda value: f"{value:.2f}")
            else:
                f.write("        Number of imaginary frequencies: N/A\n")
                f.write("        First Vibrational Frequency (cm^-1): N/A\n")
                f.write("        Last Vibrational Frequency (cm^-1): N/A\n")
        f.write("\n")

        f.write("Hydrogen bond analysis:\n")
        f.write(f"Criterion: H...A distance between {HB_MIN_DISTANCE:.1f} Å and {HB_MAX_DISTANCE:.1f} Å, D-H...A angle >= {HB_MIN_ANGLE:.1f}°, with H covalently bonded to a donor (O, N, F).\n")
        for mol_data in cluster_members_data:
            f.write(f"    {mol_data['filename']}:\n")
            f.write(f"        Number of hydrogen bonds: {mol_data.get('num_hydrogen_bonds', 0)}\n")

            if mol_data.get('hbond_details'):
                for hbond in mol_data['hbond_details']:
                    f.write(f"            Hydrogen bond: {hbond['donor_atom_label']}-{hbond['hydrogen_atom_label']}...{hbond['acceptor_atom_label']} (Dist: {hbond['H...A_distance']:.3f} Å, D-H: {hbond['D-H_covalent_distance']:.3f} Å, Angle: {hbond['D-H...A_angle']:.2f}°)\n")
            else:
                f.write("        No hydrogen bonds detected based on the criterion.\n")
        f.write("\n")

        # RMSD comparison section (for clusters with multiple configurations)
        if num_configurations > 1:
            f.write(f"RMSD Analysis ({'Heavy Atoms' if rmsd_heavy else 'All Atoms'}):\n")
            f.write("Pairwise RMSD values between configurations (Å):\n")

            # Check if all configurations have geometry data
            all_have_geometry = all(
                mol['final_geometry_coords'] is not None and
                mol['final_geometry_atomnos'] is not None
                for mol in cluster_members_data
            )

            if all_have_geometry:
                # Calculate pairwise RMSD matrix
                rmsd_matrix = []
                for i in range(num_configurations):
                    row = []
                    for j in range(num_configurations):
                        if i == j:
                            row.append(0.0)
                        elif i < j:
                            # Calculate RMSD
                            mol_i = cluster_members_data[i]
                            mol_j = cluster_members_data[j]
                            rmsd_val = calculate_rmsd(
                                mol_i['final_geometry_atomnos'],
                                mol_i['final_geometry_coords'],
                                mol_j['final_geometry_atomnos'],
                                mol_j['final_geometry_coords'],
                                heavy_only=rmsd_heavy
                            )
                            row.append(rmsd_val if rmsd_val is not None else float('nan'))
                        else:
                            # Mirror the upper triangle
                            row.append(rmsd_matrix[j][i])
                    rmsd_matrix.append(row)

                # Display RMSD values in a readable format
                for i, mol_data in enumerate(cluster_members_data):
                    f.write(f"    {mol_data['filename']}:\n")
                    for j, mol_other in enumerate(cluster_members_data):
                        if i != j:
                            rmsd_val = rmsd_matrix[i][j]
                            if np.isnan(rmsd_val):
                                f.write(f"        vs {mol_other['filename']}: N/A (calculation failed)\n")
                            else:
                                f.write(f"        vs {mol_other['filename']}: {rmsd_val:.3f} Å\n")
            else:
                f.write("    RMSD calculation unavailable: Missing geometry data for one or more configurations.\n")
            f.write("\n")

        # Separator before the individual structure details
        f.write("=" * 90 + "\n\n")

        # 7. Individual Structure Details
        for i, mol_data in enumerate(cluster_members_data):
            if i > 0:  # Add shorter separator only before subsequent structures
                f.write("=" * 50 + "\n\n")

            f.write(f"Processed file: {mol_data['filename']}\n")
            f.write(f"Method: {mol_data.get('method', 'N/A')}\n")
            f.write(f"Functional: {mol_data.get('functional', 'N/A')}\n")
            f.write(f"Basis Set: {mol_data.get('basis_set', 'N/A')}\n")
            f.write(f"Charge: {mol_data.get('charge', 'N/A')}\n")
            f.write(f"Multiplicity: {mol_data.get('multiplicity', 'N/A')}\n")
            f.write(f"Number of atoms: {mol_data.get('num_atoms', 'N/A')}\n")

            # Final Geometry
            if mol_data.get('final_geometry_atomnos') is not None and mol_data.get('final_geometry_coords') is not None:
                f.write("Final Geometry:\n")
                atomnos = mol_data['final_geometry_atomnos']
                atomcoords = mol_data['final_geometry_coords']
                for j in range(len(atomnos)):
                    symbol = atomic_number_to_symbol(atomnos[j])
                    f.write(f"{symbol:<2} {atomcoords[j][0]:10.6f} {atomcoords[j][1]:10.6f} {atomcoords[j][2]:10.6f}\n")
            else:
                f.write("Final Geometry: N/A\n")
            f.write("\n")

    vprint(f"Wrote combined data for Cluster '{dat_file_prefix}' to '{os.path.basename(output_filename)}'")


__all__ = ["write_cluster_dat_file"]
