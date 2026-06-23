"""Quality-control filters — imaginary-frequency and non-converged structures.

Before COSMIC reports a cluster's representative as a discovered motif it must
be a *true minimum*. Two QC filters enforce that:

* :func:`filter_imaginary_freq_structures` — a structure with imaginary
  frequencies is not a minimum. If it shares a cluster with true minima it is
  discarded (the cluster already has a better representative); if it forms its
  own cluster it is saved to ``skipped_structures/`` because it may flag a
  motif that hasn't been properly converged yet.
* :func:`filter_non_converged_structures` plus
  :func:`save_non_converged_critical_structures` — flag and write out
  structures that have reduced/unmatched feature vectors (or, in freq mode,
  no Gibbs energy). These cannot become representatives but are kept so the
  user can recalculate them.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

from cosmic_ascec.clustering import console
from cosmic_ascec.clustering.console import print_step, version, vprint
from cosmic_ascec.clustering.energies import hartree_to_kcal_mol
from cosmic_ascec.clustering.features.geometric import extract_xyz_from_output

Record = MutableMapping[str, Any]
Cluster = List[Record]


def filter_imaginary_freq_structures(
    clusters_list: Sequence[Cluster],
    output_base_dir: str,
    input_source: Any = None,
    total_processed: Optional[int] = None,
    write_summary: bool = False,
    precomputed_skipped: Optional[Dict[str, List[Record]]] = None,
) -> Tuple[List[Cluster], Dict[str, List[Record]]]:
    """
    Filters clusters to handle structures with imaginary frequencies.

    Removes imaginary frequency structures from mixed clusters or saves them for
    recalculation if they form isolated clusters that may represent missing
    motifs.

    Returns ``(filtered_clusters, skipped_info_dict)``.

    Verbatim port of cosmic-v01's ``filter_imaginary_freq_structures``
    (lines 2123-2480).
    """
    filtered_clusters: List[Cluster] = []

    # Use precomputed skipped structures if provided, otherwise collect them
    if precomputed_skipped:
        skipped_clustered_with_normal = precomputed_skipped.get('clustered_with_normal', [])
        skipped_need_recalc = precomputed_skipped.get('need_recalculation', [])
    else:
        skipped_clustered_with_normal = []
        skipped_need_recalc = []

    for cluster in clusters_list:
        if not cluster:
            continue

        # Separate structures with and without imaginary frequencies
        has_imag = [m for m in cluster if m.get('_has_imaginary_freqs', False)]
        no_imag = [m for m in cluster if not m.get('_has_imaginary_freqs', False)]

        if has_imag and no_imag:
            filtered_clusters.append(no_imag)
            skipped_clustered_with_normal.extend(has_imag)
            if console.VERBOSE:
                for m in has_imag:
                    print(f"  INFO: Discarding {m['filename']} (imaginary freq) - clustered with true minima")
        elif has_imag and not no_imag:
            skipped_need_recalc.extend(has_imag)
            if console.VERBOSE:
                for m in has_imag:
                    print(f"  INFO: Saving {m['filename']} to skipped_structures/ - may represent missing motif")
        else:
            filtered_clusters.append(cluster)

    if (skipped_clustered_with_normal or skipped_need_recalc) and write_summary:
        skipped_dir = os.path.join(output_base_dir, "skipped_structures")
        os.makedirs(skipped_dir, exist_ok=True)

        clustered_dir = os.path.join(skipped_dir, "clustered_with_minima")
        need_recalc_dir = os.path.join(skipped_dir, "need_recalculation")
        os.makedirs(clustered_dir, exist_ok=True)
        os.makedirs(need_recalc_dir, exist_ok=True)

        def center_text_skipped(text, width=75):
            """Center text within specified width."""
            return text.center(width)

        summary_lines = []
        summary_lines.append("=" * 75)
        summary_lines.append("")
        summary_lines.append(center_text_skipped("***************************"))
        summary_lines.append(center_text_skipped("* C O S M I C *"))
        summary_lines.append(center_text_skipped("***************************"))
        summary_lines.append("")
        summary_lines.append("                             √≈≠==≈                                  ")
        summary_lines.append("   √≈≠==≠≈√   √≈≠==≠≈√         ÷++=                      ≠===≠       ")
        summary_lines.append("     ÷++÷       ÷++÷           =++=                     ÷×××××=      ")
        summary_lines.append("     =++=       =++=     ≠===≠ ÷++=      ≠====≠         ÷-÷ ÷-÷      ")
        summary_lines.append("     =++=       =++=    =××÷=≠=÷++=    ≠÷÷÷==÷÷÷≈      ≠××≠ =××=     ")
        summary_lines.append("     =++=       =++=   ≠××=    ÷++=   ≠×+×    ×+÷      ÷+×   ×+××    ")
        summary_lines.append("     =++=       =++=   =+÷     =++=   =+-×÷==÷×-×≠    =×+×÷=÷×+-÷    ")
        summary_lines.append("     ≠×+÷       ÷+×≠   =+÷     =++=   =+---×××××÷×   ≠××÷==×==÷××≠   ")
        summary_lines.append("      =××÷     =××=    ≠××=    ÷++÷   ≠×-×           ÷+×       ×+÷   ")
        summary_lines.append("       ≠=========≠      ≠÷÷÷=≠≠=×+×÷-  ≠======≠≈√  -÷×+×≠     ≠×+×÷- ")
        summary_lines.append("          ≠===≠           ≠==≠  ≠===≠     ≠===≠    ≈====≈     ≈====≈ ")
        summary_lines.append("")
        summary_lines.append("")
        summary_lines.append(center_text_skipped("Universidad de Antioquia - Medellín - Colombia"))
        summary_lines.append("")
        summary_lines.append("")
        summary_lines.append(center_text_skipped("Skipped Structures Summary"))
        summary_lines.append("")
        summary_lines.append(center_text_skipped(version))
        summary_lines.append("")
        summary_lines.append("")
        summary_lines.append(center_text_skipped("Química Física Teórica - QFT"))
        summary_lines.append("")
        summary_lines.append("")
        summary_lines.append("=" * 75 + "\n")

        # Statistics section
        total_skipped = len(skipped_clustered_with_normal) + len(skipped_need_recalc)
        summary_lines.append("Summary statistics")
        summary_lines.append("=" * 75)
        summary_lines.append("")
        summary_lines.append(f"Total structures with imaginary frequencies: {total_skipped}")
        summary_lines.append(f"  - Clustered with true minima (can be ignored): {len(skipped_clustered_with_normal)}")
        summary_lines.append(f"  - Not clustered structures (need review): {len(skipped_need_recalc)}")
        summary_lines.append("")

        if len(skipped_clustered_with_normal) > 0:
            percentage_ignored = (len(skipped_clustered_with_normal) / total_skipped * 100)
            summary_lines.append(f"Percentage of skipped structures that can be ignored: {percentage_ignored:.1f}%")
        if len(skipped_need_recalc) > 0:
            percentage_recalc = (len(skipped_need_recalc) / total_skipped * 100)
            summary_lines.append(f"Percentage of structures needing review: {percentage_recalc:.1f}%")

        if total_processed is not None and total_processed > 0:
            percentage_of_total = (len(skipped_need_recalc) / total_processed * 100)
            summary_lines.append("")
            summary_lines.append(f"Impact of critical structures on total dataset: {len(skipped_need_recalc)}/{total_processed} configurations ({percentage_of_total:.1f}%)")
            summary_lines.append("")

            if percentage_of_total < 10:
                summary_lines.append("Assessment: Low impact (<10%)")
                summary_lines.append("  Small fraction of dataset. Recalculation recommended but not critical.")
            elif percentage_of_total < 20:
                summary_lines.append("Assessment: Moderate impact (10-20%)")
                summary_lines.append("  Noticeable portion affected. Recalculation recommended for complete coverage.")
            else:
                summary_lines.append("Assessment: High impact (>20%)")
                summary_lines.append("  Significant portion affected. Suggests systematic optimization issues.")
                summary_lines.append("  Review calculation settings before recalculating.")

        summary_lines.append("")
        summary_lines.append("=" * 75 + "\n")

        if skipped_need_recalc:
            summary_lines.append("Structures needing recalculation (potential missing motifs)")
            summary_lines.append("=" * 75)
            summary_lines.append("")
            summary_lines.append("Structures were not clustered with true minima. They may")
            summary_lines.append("represent missing motifs or transition states. Recalculation recommended")
            summary_lines.append("to verify if they correspond to true minima.")
            summary_lines.append("")
            summary_lines.append(f"Total structures: {len(skipped_need_recalc)}")
            summary_lines.append("Files saved in: need_recalculation/")
            summary_lines.append("")
            summary_lines.append("RECOMMENDATION:")
            summary_lines.append("  1. Review calculation setup and input parameters")
            summary_lines.append("  2. Re-run geometry optimization with tighter convergence criteria")
            summary_lines.append("  3. Check if structure is a transition state or saddle point")
            summary_lines.append("  4. Verify starting geometry was reasonable")
            summary_lines.append("")

            if total_processed is not None and total_processed > 0:
                percentage_of_total = (len(skipped_need_recalc) / total_processed * 100)

                if percentage_of_total >= 20:
                    summary_lines.append(f"⚠ CRITICAL: {len(skipped_need_recalc)} structures ({percentage_of_total:.1f}% of dataset)")
                    summary_lines.append("High percentage suggests SYSTEMATIC optimization issues.")
                    summary_lines.append("Before recalculating, review:")
                    summary_lines.append("  • Convergence criteria")
                    summary_lines.append("  • Starting geometries")
                    summary_lines.append("  • Basis set and functional")
                    summary_lines.append("  • Optimization thresholds")
                    summary_lines.append("")
                elif percentage_of_total >= 10:
                    summary_lines.append(f"⚠ WARNING: {len(skipped_need_recalc)} structures ({percentage_of_total:.1f}% of dataset)")
                    summary_lines.append("Noticeable portion may indicate systematic issues.")
                    summary_lines.append("Consider reviewing structures before bulk recalculation.")
                    summary_lines.append("")
                else:
                    summary_lines.append(f"ℹ NOTE: {len(skipped_need_recalc)} structures ({percentage_of_total:.1f}% of dataset)")
                    summary_lines.append("Relatively small portion. Likely isolated problematic cases.")
                    summary_lines.append("")
            else:
                if len(skipped_need_recalc) > 5:
                    summary_lines.append(f"IMPORTANT: {len(skipped_need_recalc)} structures need recalculation.")
                    summary_lines.append("Consider reviewing calculation settings.")
                    summary_lines.append("")
            summary_lines.append("File list:")
            for m in skipped_need_recalc:
                num_imag = m.get('num_imaginary_freqs', 'Unknown')
                gibbs_energy = m.get('gibbs_free_energy')
                if gibbs_energy is not None:
                    energy_str = f"G = {gibbs_energy:.6f} Hartree ({hartree_to_kcal_mol(gibbs_energy):.2f} kcal/mol)"
                else:
                    energy_str = "G = N/A"
                summary_lines.append(f"  - {m['filename']}")
                summary_lines.append(f"    Imaginary frequencies: {num_imag}")
                summary_lines.append(f"    {energy_str}")
            summary_lines.append("")
            summary_lines.append("=" * 75 + "\n")

        if skipped_clustered_with_normal:
            summary_lines.append("Structures clustered with true minima")
            summary_lines.append("=" * 75)
            summary_lines.append("")
            summary_lines.append("Structures clustered with true minima")
            summary_lines.append("Better representations exist, so these can be safely ignored.")
            summary_lines.append("")
            summary_lines.append(f"Total structures: {len(skipped_clustered_with_normal)}")
            summary_lines.append("Files saved in: clustered_with_minima/")
            summary_lines.append("")
            summary_lines.append("File list:")
            for m in skipped_clustered_with_normal:
                num_imag = m.get('num_imaginary_freqs', 'Unknown')
                gibbs_energy = m.get('gibbs_free_energy')
                if gibbs_energy is not None:
                    energy_str = f"G = {gibbs_energy:.6f} Hartree ({hartree_to_kcal_mol(gibbs_energy):.2f} kcal/mol)"
                else:
                    energy_str = "G = N/A"
                summary_lines.append(f"  - {m['filename']}")
                summary_lines.append(f"    Imaginary frequencies: {num_imag}")
                summary_lines.append(f"    {energy_str}")
            summary_lines.append("")
            summary_lines.append("=" * 75)

        summary_file = os.path.join(skipped_dir, "skipped_summary.txt")
        with open(summary_file, 'w', newline='\n', encoding='utf-8') as f:
            f.write("\n".join(summary_lines))

        if input_source:
            import glob as glob_module

            if isinstance(input_source, list):
                available_files = {os.path.basename(f): f for f in input_source}
            else:
                # Robust file finding that includes subdirectories (e.g. orca_out_*)
                log_files = []
                out_files = []

                # Check root
                log_files.extend(glob_module.glob(os.path.join(str(input_source), "*.log")))
                out_files.extend(glob_module.glob(os.path.join(str(input_source), "*.out")))

                # Check subdirectories
                for item in os.listdir(str(input_source)):
                    item_path = os.path.join(str(input_source), item)
                    if os.path.isdir(item_path):
                        log_files.extend(glob_module.glob(os.path.join(item_path, "*.log")))
                        out_files.extend(glob_module.glob(os.path.join(item_path, "*.out")))

                all_files = log_files + out_files
                available_files = {os.path.basename(f): f for f in all_files}

            for m in skipped_clustered_with_normal:
                source_file = available_files.get(m['filename'])
                if source_file and os.path.exists(source_file):
                    # Copy output file
                    dest_file = os.path.join(clustered_dir, m['filename'])
                    shutil.copy2(source_file, dest_file)

                    # Extract XYZ geometry
                    natoms, coords, symbols = extract_xyz_from_output(source_file)
                    if natoms is not None and coords is not None and symbols is not None:
                        # Save individual XYZ file
                        basename = os.path.splitext(m['filename'])[0]
                        xyz_file = os.path.join(clustered_dir, f"{basename}.xyz")
                        with open(xyz_file, 'w', encoding='utf-8') as f:
                            f.write(f"{natoms}\n")
                            f.write(f"{basename} - clustered with minima\n")
                            for symbol, coord in zip(symbols, coords):
                                f.write(f"{symbol:2s}  {coord[0]:15.8f}  {coord[1]:15.8f}  {coord[2]:15.8f}\n")

            # Process structures needing recalculation - extract XYZ geometries
            xyz_data_list = []  # For combined file
            for m in skipped_need_recalc:
                source_file = available_files.get(m['filename'])
                if source_file and os.path.exists(source_file):
                    # Copy output file
                    dest_file = os.path.join(need_recalc_dir, m['filename'])
                    shutil.copy2(source_file, dest_file)

                # Extract XYZ geometry (ALWAYS do this if source file exists)
                if source_file and os.path.exists(source_file):
                    natoms, coords, symbols = extract_xyz_from_output(source_file)
                    if natoms is not None and coords is not None and symbols is not None:
                        # Save individual XYZ file
                        basename = os.path.splitext(m['filename'])[0]
                        xyz_file = os.path.join(need_recalc_dir, f"{basename}.xyz")
                        with open(xyz_file, 'w', encoding='utf-8') as f:
                            f.write(f"{natoms}\n")
                            f.write(f"{basename} - needs recalculation\n")
                            for symbol, coord in zip(symbols, coords):
                                f.write(f"{symbol:2s}  {coord[0]:15.8f}  {coord[1]:15.8f}  {coord[2]:15.8f}\n")

                        # Store for combined file
                        xyz_data_list.append({
                            'natoms': natoms,
                            'symbols': symbols,
                            'coords': coords,
                            'basename': basename
                        })

            # Create combined XYZ file only if there are 2+ structures needing recalculation
            if xyz_data_list:
                if len(xyz_data_list) >= 2:
                    combined_xyz = os.path.join(need_recalc_dir, "combined_need_recalc.xyz")
                    with open(combined_xyz, 'w', encoding='utf-8') as f:
                        for data in xyz_data_list:
                            f.write(f"{data['natoms']}\n")
                            f.write(f"{data['basename']} - needs recalculation\n")
                            for symbol, coord in zip(data['symbols'], data['coords']):
                                f.write(f"{symbol:2s}  {coord[0]:15.8f}  {coord[1]:15.8f}  {coord[2]:15.8f}\n")

                    vprint(f"  Created combined XYZ file: {combined_xyz}")

                    # Create combined MOL file from combined XYZ
                    combined_mol = os.path.join(need_recalc_dir, "combined_need_recalc.mol")
                    try:
                        result = subprocess.run(['obabel', '-ixyz', combined_xyz, '-omol', '-O', combined_mol],
                                              capture_output=True, check=True)
                        vprint(f"  Created combined MOL file: {combined_mol}")
                    except Exception:
                        pass  # obabel not available or failed
                elif len(xyz_data_list) == 1:
                    # Single structure - create MOL file from its XYZ
                    basename = xyz_data_list[0]['basename']
                    xyz_file = os.path.join(need_recalc_dir, f"{basename}.xyz")
                    mol_file = os.path.join(need_recalc_dir, f"{basename}.mol")
                    try:
                        result = subprocess.run(['obabel', '-ixyz', xyz_file, '-omol', '-O', mol_file],
                                              capture_output=True, check=True)
                        vprint(f"  Single structure, created MOL file: {mol_file}")
                    except Exception:
                        pass  # obabel not available or failed
                    vprint(f"  Extracted {len(xyz_data_list)} individual XYZ files")
                else:
                    vprint(f"  Single file needing recalculation - no combined file created")
                    vprint(f"  Extracted {len(xyz_data_list)} individual XYZ file")

        total_skipped = len(skipped_clustered_with_normal) + len(skipped_need_recalc)
        print()
        print()
        print_step(f"Processed {total_skipped} structures with imaginary frequencies:")
        print(f"  - {len(skipped_clustered_with_normal)} clustered with true minima (can be ignored)")
        if skipped_clustered_with_normal:
            for item in skipped_clustered_with_normal:
                print(f"         {item['filename']}  -  {item['num_imaginary_freqs']} imaginary freq(s)")
        print(f"  - {len(skipped_need_recalc)} may represent missing motifs (need recalculation)")
        if skipped_need_recalc:
            for item in skipped_need_recalc:
                print(f"         {item['filename']}  -  {item['num_imaginary_freqs']} imaginary freq(s)")

        if len(skipped_need_recalc) > 0 and console.VERBOSE:
            print(f"  → Review 'skipped_structures/skipped_summary.txt' for details")

        print()  # Blank line after "Processed"

    skipped_info = {
        'clustered_with_normal': skipped_clustered_with_normal,
        'need_recalculation': skipped_need_recalc
    }

    return filtered_clusters, skipped_info


def _is_non_converged_structure(mol_data: Record, dataset_has_freq: bool = True) -> bool:
    """Return True when a structure should be treated as non-converged/critical.

    Verbatim port of cosmic-v01's ``_is_non_converged_structure``
    (lines 2483-2501).
    """
    # Structures explicitly flagged as unmatched reduced → critical
    if mol_data.get('_reduced_unmatched', False):
        return True
    # In freq mode, structures missing Gibbs energy that are NOT absorbed
    # reduced-vector matches are still critical
    if dataset_has_freq and not mol_data.get('_is_full_feature', True):
        # But if it was matched (absorbed), it's fine
        if mol_data.get('_initial_cluster_label') is not None and not mol_data.get('_reduced_unmatched', False):
            return False
        return True
    return False


def filter_non_converged_structures(
    clusters_list: Sequence[Cluster], dataset_has_freq: bool = True
) -> Tuple[List[Cluster], List[Record]]:
    """
    Remove non-converged structures from cluster candidates.

    Verbatim port of cosmic-v01's ``filter_non_converged_structures``
    (lines 2504-2527).
    """
    filtered_clusters: List[Cluster] = []
    critical_non_converged: List[Record] = []

    for cluster in clusters_list:
        if not cluster:
            continue

        non_converged = [m for m in cluster if _is_non_converged_structure(m, dataset_has_freq)]
        converged = [m for m in cluster if not _is_non_converged_structure(m, dataset_has_freq)]

        if non_converged:
            critical_non_converged.extend(non_converged)

        if converged:
            filtered_clusters.append(converged)

    return filtered_clusters, critical_non_converged


def save_non_converged_critical_structures(
    non_converged_structures: Sequence[Record],
    output_base_dir: str,
    input_source: Any = None,
    total_processed: Optional[int] = None,
) -> None:
    """Write critical non-converged structures to
    skipped_structures/critical_non_converged/.

    Verbatim port of cosmic-v01's ``save_non_converged_critical_structures``
    (lines 2530-2602).
    """
    if not non_converged_structures:
        return

    skipped_dir = os.path.join(output_base_dir, "skipped_structures")
    critical_dir = os.path.join(skipped_dir, "critical_non_converged")
    os.makedirs(critical_dir, exist_ok=True)

    # Build filename->source path map from input source.
    available_files: Dict[str, str] = {}
    try:
        if isinstance(input_source, list):
            available_files = {os.path.basename(f): f for f in input_source}
        elif input_source:
            source_root = str(input_source)
            root_candidates = glob.glob(os.path.join(source_root, "*.out")) + glob.glob(os.path.join(source_root, "*.log"))
            for p in root_candidates:
                available_files[os.path.basename(p)] = p

            for item in os.listdir(source_root):
                item_path = os.path.join(source_root, item)
                if os.path.isdir(item_path):
                    sub_candidates = glob.glob(os.path.join(item_path, "*.out")) + glob.glob(os.path.join(item_path, "*.log"))
                    for p in sub_candidates:
                        available_files[os.path.basename(p)] = p
    except Exception:
        pass

    # Copy outputs and export XYZ geometries for redo mode.
    for m in non_converged_structures:
        filename = m.get('filename')
        if not filename:
            continue

        source_file = available_files.get(filename)
        if source_file and os.path.exists(source_file):
            try:
                shutil.copy2(source_file, os.path.join(critical_dir, filename))
            except Exception:
                pass

            natoms, coords, symbols = extract_xyz_from_output(source_file)
            if natoms is not None and coords is not None and symbols is not None:
                basename = os.path.splitext(filename)[0]
                xyz_file = os.path.join(critical_dir, f"{basename}.xyz")
                try:
                    with open(xyz_file, 'w', encoding='utf-8') as f:
                        f.write(f"{natoms}\n")
                        f.write(f"{basename} - critical non-converged\n")
                        for symbol, coord in zip(symbols, coords):
                            f.write(f"{symbol:2s}  {coord[0]:15.8f}  {coord[1]:15.8f}  {coord[2]:15.8f}\n")
                except Exception:
                    pass

    # Write a compact summary file for auditability.
    summary_path = os.path.join(critical_dir, "non_converged_summary.txt")
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Critical Non-converged Structures\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total structures: {len(non_converged_structures)}\n")
            if total_processed is not None and total_processed > 0:
                pct = (len(non_converged_structures) / total_processed) * 100.0
                f.write(f"Impact on dataset: {len(non_converged_structures)}/{total_processed} ({pct:.1f}%)\n")
            f.write("\nFiles:\n")
            for m in non_converged_structures:
                g = m.get('gibbs_free_energy')
                f.write(f"  - {m.get('filename', 'UNKNOWN')} (Gibbs: {'N/A' if g is None else f'{g:.6f}'})\n")
    except Exception:
        pass

    print_step(f"Critical non-converged structures: {len(non_converged_structures)} (saved to skipped_structures/critical_non_converged)")


__all__ = [
    "filter_imaginary_freq_structures",
    "filter_non_converged_structures",
    "save_non_converged_critical_structures",
]
