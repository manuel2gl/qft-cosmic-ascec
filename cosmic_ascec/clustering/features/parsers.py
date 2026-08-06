"""QM-output property parsers — verbatim port of cosmic-v01's three front-ends.

R5 (D-039 faithful decomposition) re-ports the COSMIC clustering pipeline's
per-package property parsers. cosmic-v01 parses every clustering candidate
through one of three standalone functions — never through ascec-v04's
annealing QM adapters — and returns a free-form ``extracted_props`` dict. The
pre-R5 v05 routed clustering feature extraction through the
``quantum_chemistry`` adapters' ``parse_result`` / ``QMResult.extras``; that was
a divergence (the annealing adapters are not on cosmic-v01's clustering path).
This module restores cosmic-v01's own parsers verbatim.

Ported verbatim from ``cosmic-v01.py``:

* ``extract_properties_with_xtb`` — lines 857-1154.
* ``extract_properties_with_cclib`` — lines 1156-1533.
* ``extract_properties_with_opi`` — lines 1535-1831.
* ``_extract_method_from_file_opi`` … ``_extract_rotconsts_from_file_opi`` —
  lines 1833-1958.
* ``extract_homo_lumo_from_orca_text`` — lines 4215-4281.

The only adaptations are the mechanical ones D-039 permits: cross-module
imports replace cosmic-v01's single-script scope. ``print`` statements are part
of cosmic-v01's stdout contract and are kept verbatim. R5c unified the
``VERBOSE`` / ``vprint`` verbosity mechanism: this module routes through
:mod:`cosmic_ascec.clustering.console` (the single ``VERBOSE`` flag), rather
than carrying its own R5-era copy.
"""

from __future__ import annotations

import os
import re
import warnings

import numpy as np

from cosmic_ascec.clustering import console
from cosmic_ascec.clustering.console import vprint
from cosmic_ascec.clustering.features.geometric import (
    calculate_nuclear_repulsion,
    calculate_radius_of_gyration,
    calculate_rotational_constants,
    detect_hydrogen_bonds,
    extract_xyz_from_output,
)

# cclib imported at top level for multiprocessing compatibility — cosmic-v01.py 9-14.
try:
    from cclib.io import ccread
    CCLIB_AVAILABLE = True
except ImportError:
    CCLIB_AVAILABLE = False
    ccread = None

# OPI (ORCA Python Interface) for ORCA 6.1+ support — cosmic-v01.py 16-23.
from pathlib import Path as PathLib  # stdlib, always available
try:
    from opi.output.core import Output as OPIOutput  # type: ignore
    OPI_AVAILABLE = True
except ImportError:
    OPI_AVAILABLE = False
    OPIOutput = None

# Energy conversion constant — cosmic-v01.py line 44.
BOHR_TO_ANGSTROM = 0.529177210903  # Angstrom per Bohr


def extract_properties_with_xtb(logfile_path):
    """Extract properties from standalone xTB text outputs."""
    extracted_props = {
        'filename': os.path.basename(logfile_path),
        'method': "GFN2-xTB",
        'functional': "GFN2-xTB",
        'basis_set': "N/A",
        'charge': None,
        'multiplicity': None,
        'num_atoms': 0,
        'final_geometry_atomnos': None,
        'final_geometry_coords': None,
        'final_electronic_energy': None,
        'gibbs_free_energy': None,
        'homo_energy': None,
        'lumo_energy': None,
        'homo_lumo_gap': None,
        'dipole_moment': None,
        'rotational_constants': None,
        'radius_of_gyration': None,
        'vnn_nuclear_repulsion': None,
        'num_imaginary_freqs': None,
        'first_vib_freq': None,
        'last_vib_freq': None,
        'num_hydrogen_bonds': 0,
        'hbond_details': [],
        'average_hbond_distance': None,
        'min_hbond_distance': None,
        'max_hbond_distance': None,
        'std_hbond_distance': None,
        'average_hbond_angle': None,
        'min_hbond_angle': None,
        'max_hbond_angle': None,
        'std_hbond_angle': None,
        '_has_freq_calc': False,
        '_initial_cluster_label': None,
        '_parent_global_cluster_id': None,
        '_first_rmsd_context_listing': None,
        '_second_rmsd_sub_cluster_id': None,
        '_second_rmsd_context_listing': None,
        '_second_rmsd_rep_filename': None,
        '_rmsd_pass_origin': None
    }

    try:
        with open(logfile_path, 'r', encoding='utf-8', errors='ignore') as f_log:
            content = f_log.read()
        lines = content.splitlines()

        # --- Method detection ------------------------------------------------
        # Match the actual method used (in the setup block), not citations.
        # xTB prints e.g. "G F N 2 - x T B" or "G F N - F F" as a banner,
        # and "GFN2-xTB" / "GFN-FF" in the settings block.
        # Look for the spaced banner first (most reliable), then settings line.
        if re.search(r'G\s+F\s+N\s*-?\s*F\s+F', content):
            extracted_props['method'] = 'GFN-FF'
            extracted_props['functional'] = 'GFN-FF'
        elif re.search(r'G\s+F\s+N\s+0', content):
            extracted_props['method'] = 'GFN0-xTB'
            extracted_props['functional'] = 'GFN0-xTB'
        elif re.search(r'G\s+F\s+N\s+1\s', content):
            extracted_props['method'] = 'GFN1-xTB'
            extracted_props['functional'] = 'GFN1-xTB'
        # else: default GFN2-xTB (already set)

        # --- Charge ----------------------------------------------------------
        # xTB prints ":  net charge   0  :" in the setup block
        charge_match = re.search(r'net\s+charge\s+([-]?\d+)', content)
        if charge_match:
            extracted_props['charge'] = int(charge_match.group(1))

        # --- Multiplicity ----------------------------------------------------
        # xTB prints ":  unpaired electrons   0  :" — multiplicity = 2S+1
        unpaired_match = re.search(r'unpaired\s+electrons\s+(\d+)', content)
        if unpaired_match:
            extracted_props['multiplicity'] = int(unpaired_match.group(1)) + 1

        # --- Energy (last TOTAL ENERGY in the summary box) -------------------
        energy_matches = re.findall(r'TOTAL ENERGY\s+([-+]?\d+\.\d+)\s*Eh', content)
        if energy_matches:
            extracted_props['final_electronic_energy'] = float(energy_matches[-1])

        # --- HOMO-LUMO gap (last occurrence = post-optimization) -------------
        gap_matches = re.findall(r'HOMO-?LUMO\s+(?:GAP|gap)\s*[:=]?\s*([-+]?\d+\.\d+)', content, re.IGNORECASE)
        if gap_matches:
            extracted_props['homo_lumo_gap'] = float(gap_matches[-1])

        # --- HOMO / LUMO orbital energies ------------------------------------
        # xTB format:  "  24   2.0000   -0.4357648   -11.8578 (HOMO)"
        #              "  25            0.0026252     0.0714 (LUMO)"
        # Take last occurrence (post-optimization Property Printout).
        homo_matches = re.findall(
            r'([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+\(HOMO\)', content)
        if homo_matches:
            # Each match is (Energy/Eh, Energy/eV); take Eh from the last match
            extracted_props['homo_energy'] = float(homo_matches[-1][0])
        lumo_matches = re.findall(
            r'([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+\(LUMO\)', content)
        if lumo_matches:
            extracted_props['lumo_energy'] = float(lumo_matches[-1][0])

        # --- Dipole moment ---------------------------------------------------
        # xTB format:
        #   molecular dipole:
        #                x           y           z       tot (Debye)
        #    full:    -0.011      -0.003       0.029       0.079
        # Must match exactly 4 float columns (not 6 as in quadrupole).
        dipole_matches = re.findall(
            r'^\s*full:\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*$',
            content, re.MULTILINE)
        if dipole_matches:
            # Take the last "full:" line (post-optimization), tot column
            extracted_props['dipole_moment'] = float(dipole_matches[-1][3])

        # --- Rotational constants --------------------------------------------
        rot_match = re.search(
            r'rotational constants/cm.*?:\s*([-+]?\d+\.\d+E[+-]?\d+)\s+([-+]?\d+\.\d+E[+-]?\d+)\s+([-+]?\d+\.\d+E[+-]?\d+)',
            content, re.IGNORECASE)
        if rot_match:
            extracted_props['rotational_constants'] = np.array([
                float(rot_match.group(1)),
                float(rot_match.group(2)),
                float(rot_match.group(3))
            ])

        # --- Vibrational frequencies -----------------------------------------
        freq_matches = re.findall(r'([-+]?\d+\.\d+)\s*cm\^-?1', content, re.IGNORECASE)
        if freq_matches:
            freqs = [float(v) for v in freq_matches]
            extracted_props['_has_freq_calc'] = True
            imag = [v for v in freqs if v < 0.0]
            real = [v for v in freqs if v > 0.0]
            extracted_props['num_imaginary_freqs'] = len(imag)
            if real:
                extracted_props['first_vib_freq'] = min(real)
                extracted_props['last_vib_freq'] = max(real)

        # --- Geometry --------------------------------------------------------
        # Strategy for xTB:
        # 0) Parse "final structure:" block directly from the .out file (most reliable)
        # 1) <basename>.xtbopt.xyz (from --namespace)
        # 2) xtbopt.xyz
        # 3) <basename>.xtbopt.log (trajectory — take last frame)
        # 4) xtbopt.log (trajectory — take last frame)
        # 5) <basename>.xyz (original input — pre-optimization, last resort)
        file_path = PathLib(logfile_path)
        basename = file_path.stem
        workdir = file_path.parent

        sym_to_num = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Mo': 42, 'Ru': 44, 'Rh': 45,
            'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52,
            'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Pt': 78, 'Au': 79,
            'Hg': 80, 'Pb': 82, 'Bi': 83
        }

        def _read_xyz_geometry(xyz_path):
            """Read a single-frame .xyz file and return (atomnos, coords)."""
            with open(xyz_path, 'r', encoding='utf-8', errors='ignore') as xf:
                raw = [ln.rstrip('\n') for ln in xf]
            if len(raw) < 2:
                return None, None
            try:
                nat = int(raw[0].strip())
            except Exception:
                return None, None
            atoms = raw[2:2 + nat]
            if len(atoms) != nat:
                return None, None
            atomnos = []
            coords = []
            for ln in atoms:
                parts = ln.split()
                if len(parts) < 4:
                    return None, None
                atomnos.append(sym_to_num.get(parts[0], 0))
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            return np.array(atomnos), np.array(coords, dtype=float)

        def _read_last_xyz_frame(traj_path):
            """Read the last frame from an xTB trajectory .log file (multi-frame xyz)."""
            with open(traj_path, 'r', encoding='utf-8', errors='ignore') as xf:
                raw = [ln.rstrip('\n') for ln in xf]
            if len(raw) < 3:
                return None, None
            try:
                nat = int(raw[0].strip())
            except Exception:
                return None, None
            frame_size = nat + 2  # natoms line + comment line + atom lines
            total_frames = len(raw) // frame_size
            if total_frames < 1:
                return None, None
            # Last frame starts at this offset
            last_start = (total_frames - 1) * frame_size
            atoms = raw[last_start + 2: last_start + 2 + nat]
            if len(atoms) != nat:
                return None, None
            atomnos = []
            coords = []
            for ln in atoms:
                parts = ln.split()
                if len(parts) < 4:
                    return None, None
                atomnos.append(sym_to_num.get(parts[0], 0))
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            return np.array(atomnos), np.array(coords, dtype=float)

        # 0) Parse "final structure:" block from the .out file itself
        # Format:
        #   ================
        #    final structure:
        #   ================
        #   18
        #    xtb: 6.7.0 (08769fc)
        #   O            2.572886...   0.054433...  -0.806688...
        #   H            1.695956...  -0.135886...  -1.264428...
        final_struct_match = re.search(r'final structure:\s*={3,}\s*\n\s*(\d+)\s*\n[^\n]*\n((?:\s*[A-Z][a-z]?\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s*\n)+)', content)
        if final_struct_match:
            nat = int(final_struct_match.group(1))
            atom_block = final_struct_match.group(2).strip().splitlines()
            if len(atom_block) >= nat:
                atomnos = []
                coords = []
                for ln in atom_block[:nat]:
                    parts = ln.split()
                    if len(parts) >= 4:
                        atomnos.append(sym_to_num.get(parts[0], 0))
                        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(atomnos) == nat:
                    extracted_props['final_geometry_atomnos'] = np.array(atomnos)
                    extracted_props['final_geometry_coords'] = np.array(coords, dtype=float)
                    extracted_props['num_atoms'] = nat

        # Fallback 1) Try single-frame xyz files
        if extracted_props['final_geometry_atomnos'] is None:
            xyz_candidates = [
                workdir / f"{basename}.xtbopt.xyz",
                workdir / "xtbopt.xyz",
            ]
            for cand in xyz_candidates:
                if cand.exists():
                    atomnos, coords = _read_xyz_geometry(cand)
                    if atomnos is not None and coords is not None:
                        extracted_props['final_geometry_atomnos'] = atomnos
                        extracted_props['final_geometry_coords'] = coords
                        extracted_props['num_atoms'] = len(atomnos)
                        break

        # Fallback 2) Try trajectory .log files (last frame = optimized geom)
        if extracted_props['final_geometry_atomnos'] is None:
            traj_candidates = [
                workdir / f"{basename}.xtbopt.log",
                workdir / "xtbopt.log",
            ]
            for cand in traj_candidates:
                if cand.exists():
                    atomnos, coords = _read_last_xyz_frame(cand)
                    if atomnos is not None and coords is not None:
                        extracted_props['final_geometry_atomnos'] = atomnos
                        extracted_props['final_geometry_coords'] = coords
                        extracted_props['num_atoms'] = len(atomnos)
                        break

        # Last resort: original input xyz
        if extracted_props['final_geometry_atomnos'] is None:
            input_xyz = workdir / f"{basename}.xyz"
            if input_xyz.exists():
                atomnos, coords = _read_xyz_geometry(input_xyz)
                if atomnos is not None and coords is not None:
                    extracted_props['final_geometry_atomnos'] = atomnos
                    extracted_props['final_geometry_coords'] = coords
                    extracted_props['num_atoms'] = len(atomnos)

        if extracted_props['final_geometry_atomnos'] is not None and extracted_props['final_geometry_coords'] is not None:
            try:
                extracted_props['radius_of_gyration'] = calculate_radius_of_gyration(
                    extracted_props['final_geometry_atomnos'], extracted_props['final_geometry_coords']
                )
            except Exception:
                pass

            # xTB does not print V_NN directly; compute it from geometry+Z.
            try:
                extracted_props['vnn_nuclear_repulsion'] = calculate_nuclear_repulsion(
                    extracted_props['final_geometry_atomnos'], extracted_props['final_geometry_coords']
                )
            except Exception:
                pass

            try:
                hbond_results = detect_hydrogen_bonds(
                    extracted_props['final_geometry_atomnos'], extracted_props['final_geometry_coords']
                )
                extracted_props.update(hbond_results)
            except Exception:
                pass

        return extracted_props
    except Exception as e:
        print(f"  GENERIC_ERROR: Failed to extract properties from {os.path.basename(logfile_path)} with xTB parser: {e}")
        return None


# Feature keys a plain XYZ file can populate. Eight of the fifteen clustering
# columns are pure functions of (atomic numbers, coordinates); the other seven
# require a QM calculation and stay None unless ``--sp`` fills them in.
XYZ_DERIVED_FEATURES = (
    'vnn_nuclear_repulsion',
    'num_hydrogen_bonds',
    'average_hbond_distance',
    'std_hbond_distance',
    'average_hbond_angle',
    'rotational_constants_A',
    'rotational_constants_B',
    'rotational_constants_C',
)


def extract_properties_with_xyz(xyz_path):
    """Extract every geometry-determined property from a plain XYZ file.

    This is COSMIC's third input front-end, alongside the QM-output parsers.
    It computes the descriptors that follow from coordinates alone and leaves
    everything requiring a wavefunction as ``None``:

    * ``rotational_constants`` — eigenvalues of the mass-weighted inertia
      tensor, via :func:`calculate_rotational_constants`. These cannot be
      derived from the radius of gyration: R_g fixes only ``trace(I) =
      2·M·R_g²``, one constraint on three unknowns, so the full tensor is
      required.
    * ``vnn_nuclear_repulsion`` — Coulomb shape descriptor Σ Z_i Z_j / r_ij.
    * the hydrogen-bond block — count, mean/std H···A distance, mean D-H···A
      angle.
    * ``radius_of_gyration`` — reported in the per-cluster ``.dat`` files but
      deliberately not a clustering column (v04 dropped it as redundant with
      the rotational-constant block).

    The comment line is not parsed. It is free-form, most producers write
    nothing meaningful in it, and ranking structures on a scraped string would
    be less honest than declining to rank them — ``--sp`` is the supported way
    to obtain energies.

    The file must hold exactly one structure; multi-frame input is split up
    front by :func:`~cosmic_ascec.clustering.features.xyz_input.explode_multiframe_xyz`.

    Returns cosmic-v01's ``extracted_props`` dict, or ``None`` if no structure
    could be read.
    """
    from cosmic_ascec.clustering.features.xyz_input import read_xyz_frames

    extracted_props = {
        'filename': os.path.basename(xyz_path),
        'method': "N/A (geometry only)",
        'functional': "N/A (geometry only)",
        'basis_set': "N/A",
        'charge': None,
        'multiplicity': None,
        'num_atoms': 0,
        'final_geometry_atomnos': None,
        'final_geometry_coords': None,
        'final_electronic_energy': None,
        'gibbs_free_energy': None,
        'homo_energy': None,
        'lumo_energy': None,
        'homo_lumo_gap': None,
        'dipole_moment': None,
        'rotational_constants': None,
        'radius_of_gyration': None,
        'vnn_nuclear_repulsion': None,
        'num_imaginary_freqs': None,
        'first_vib_freq': None,
        'last_vib_freq': None,
        'num_hydrogen_bonds': 0,
        'hbond_details': [],
        'average_hbond_distance': None,
        'min_hbond_distance': None,
        'max_hbond_distance': None,
        'std_hbond_distance': None,
        'average_hbond_angle': None,
        'min_hbond_angle': None,
        'max_hbond_angle': None,
        'std_hbond_angle': None,
        '_has_freq_calc': False,
        '_has_imaginary_freqs': False,
        '_initial_cluster_label': None,
        '_parent_global_cluster_id': None,
        '_first_rmsd_context_listing': None,
        '_second_rmsd_sub_cluster_id': None,
        '_second_rmsd_context_listing': None,
        '_second_rmsd_rep_filename': None,
        '_rmsd_pass_origin': None,
        '_from_xyz': True,
    }

    try:
        frames = read_xyz_frames(xyz_path, max_frames=1)
        if not frames:
            print(f"  WARNING: no structure could be read from {os.path.basename(xyz_path)}. Skipping.")
            return None

        frame = frames[0]
        atomnos = frame.atomnos
        coords = frame.coords

        if atomnos.size == 0:
            print(f"  WARNING: {os.path.basename(xyz_path)} contains no atoms. Skipping.")
            return None

        unknown = int(np.count_nonzero(atomnos == 0))
        if unknown:
            print(
                f"  WARNING: {os.path.basename(xyz_path)}: {unknown} atom(s) with an "
                f"unrecognised element symbol; they carry no mass or charge in the descriptors."
            )

        extracted_props['final_geometry_atomnos'] = atomnos
        extracted_props['final_geometry_coords'] = coords
        extracted_props['num_atoms'] = frame.natoms

        rot_consts = calculate_rotational_constants(atomnos, coords)
        if rot_consts is not None:
            extracted_props['rotational_constants'] = rot_consts

        extracted_props['radius_of_gyration'] = calculate_radius_of_gyration(atomnos, coords)
        extracted_props['vnn_nuclear_repulsion'] = calculate_nuclear_repulsion(atomnos, coords)
        extracted_props.update(detect_hydrogen_bonds(atomnos, coords))

        return extracted_props

    except Exception as e:
        print(f"  GENERIC_ERROR: Failed to extract properties from {os.path.basename(xyz_path)} with XYZ parser: {e}")
        return None


def extract_properties_with_cclib(logfile_path):
    """
    Extract properties using cclib (original implementation).
    """
    if not CCLIB_AVAILABLE:
        print(f"  ERROR: cclib is not installed. Please install it with: pip install cclib")
        return None

    data = None

    try:
        raw_data = ccread(logfile_path)  # type: ignore[misc]

        if raw_data:
            # Determine the actual ccData object to work with
            if hasattr(raw_data, 'data') and isinstance(raw_data.data, list) and len(raw_data.data) > 0:  # type: ignore
                # This is a ccCollection, extract the last ccData object for optimization results
                data = raw_data.data[-1]  # type: ignore
            elif type(raw_data).__name__.startswith('ccData'):  # Broaden check to include subclasses like ccData_optdone_bool
                data = raw_data
            else:
                # This branch catches cases where raw_data is not a recognized ccData type or ccCollection
                # Only show in verbose mode - this is expected for incomplete/failed calculations
                vprint(f"  WARNING: Unexpected cclib return type for {os.path.basename(logfile_path)}. Type: {type(raw_data)}. Returning None.")
                return None
        else:
            # ccread returned None - expected for incomplete/corrupted files
            # Only show in verbose mode to reduce noise
            vprint(f"  WARNING: cclib returned None for {os.path.basename(logfile_path)}. This means the file could not be parsed. Returning None.")
            return None

        # Final explicit check: ensure 'data' is indeed a ccData object before proceeding
        if not type(data).__name__.startswith('ccData'):
            vprint(f"  ERROR: 'data' is not a ccData object after initial processing for {os.path.basename(logfile_path)}. Actual type: {type(data)}. Returning None.")
            return None

    except Exception as e:
        # This catches any errors during ccread or initial data extraction from ccCollection
        # Keep this as a regular print since it indicates a more serious parsing error
        vprint(f"(CCL_ERROR) Failed to parse or process {os.path.basename(logfile_path)} with cclib: {e}")
        return None

    extracted_props = {
        'filename': os.path.basename(logfile_path),
        'method': "Unknown",
        'functional': "Unknown",
        'basis_set': "Unknown",
        'charge': None,
        'multiplicity': None,
        'num_atoms': 0,
        'final_geometry_atomnos': None,
        'final_geometry_coords': None,
        'final_electronic_energy': None,
        'gibbs_free_energy': None,
        'homo_energy': None,
        'lumo_energy': None,
        'homo_lumo_gap': None,
        'dipole_moment': None,
        'rotational_constants': None,
        'radius_of_gyration': None,
        'vnn_nuclear_repulsion': None,
        'num_imaginary_freqs': None,  # Changed default to None
        'first_vib_freq': None,
        'last_vib_freq': None,
        'num_hydrogen_bonds': 0,  # Will be set by detect_hydrogen_bonds
        'hbond_details': [],     # Will be set by detect_hydrogen_bonds
        'average_hbond_distance': None,
        'min_hbond_distance': None,
        'max_hbond_distance': None,
        'std_hbond_distance': None,
        'average_hbond_angle': None,
        'min_hbond_angle': None,
        'max_hbond_angle': None,
        'std_hbond_angle': None,
        '_has_freq_calc': False,  # New internal flag
        '_initial_cluster_label': None,  # To store the integer label of the initial property cluster (from fcluster)
        '_parent_global_cluster_id': None,  # New: Stores the 'Y' from "Validated from Cluster Y"
        '_first_rmsd_context_listing': None,  # To store details of the original property cluster (RMSD to its rep)
        '_second_rmsd_sub_cluster_id': None,  # New label for sub-cluster after 2nd RMSD
        '_second_rmsd_context_listing': None,  # RMSD to representative of 2nd-level sub-cluster
        '_second_rmsd_rep_filename': None,  # Filename of representative of 2nd-level sub-cluster
        '_rmsd_pass_origin': None  # New flag to indicate if cluster came from 1st or 2nd RMSD pass
    }

    try:
        with open(logfile_path, 'r', encoding='utf-8', errors='ignore') as f_log:
            lines = f_log.readlines()

        # Determine file type based on extension
        file_extension = os.path.splitext(logfile_path)[1].lower()

        # These lines will now correctly access attributes from a ccData object
        metadata = getattr(data, 'metadata', {})
        if metadata:
            methods_list = metadata.get("methods", [])
            extracted_props['method'] = methods_list[0] if methods_list else "Unknown"
            extracted_props['functional'] = metadata.get("functional", "Unknown")
            extracted_props['basis_set'] = metadata.get("basis_set", "Unknown")
        else:
            extracted_props['method'] = "Unknown"
            extracted_props['functional'] = "Unknown"
            extracted_props['basis_set'] = "Unknown"

        extracted_props['charge'] = getattr(data, 'charge', None)
        extracted_props['multiplicity'] = getattr(data, 'mult', None)

        if hasattr(data, 'atomnos') and data.atomnos is not None and len(data.atomnos) > 0:  # type: ignore
            extracted_props['num_atoms'] = len(data.atomnos)  # type: ignore
            extracted_props['final_geometry_atomnos'] = data.atomnos  # type: ignore
        # Silently skip if atomnos missing (expected for incomplete calculations)

        if hasattr(data, 'atomcoords') and data.atomcoords is not None and len(data.atomcoords) > 0:  # type: ignore
            coords_candidate = None
            if isinstance(data.atomcoords, list) and len(data.atomcoords) > 0:  # type: ignore
                coords_candidate = np.asarray(data.atomcoords[-1])  # type: ignore
            elif isinstance(data.atomcoords, np.ndarray) and data.atomcoords.ndim == 3:  # type: ignore
                coords_candidate = data.atomcoords[-1]  # type: ignore
            elif isinstance(data.atomcoords, np.ndarray) and data.atomcoords.ndim == 2 and data.atomcoords.shape[1] == 3:  # type: ignore
                coords_candidate = data.atomcoords  # type: ignore

            if coords_candidate is not None and coords_candidate.ndim == 2 and coords_candidate.shape[1] == 3:
                extracted_props['final_geometry_coords'] = coords_candidate
            else:
                # Silently skip if coords have unexpected shape (expected for incomplete calculations)
                extracted_props['final_geometry_coords'] = None
        # Silently skip if atomcoords missing (expected for incomplete calculations)

        # --- Conditional ELECTRONIC ENERGY EXTRACTION based on file type ---
        if file_extension == '.out':
            # ORCA .out specific parsing for Final Electronic Energy
            # Take the last FINAL SINGLE POINT ENERGY in the file
            # (for OptFreq jobs this is always the converged geometry's energy)
            temp_final_electronic_energy = None
            for line in reversed(lines):
                if "FINAL SINGLE POINT ENERGY" in line:
                    try:
                        temp_final_electronic_energy = float(line.split()[-1])
                        break
                    except (ValueError, IndexError):
                        pass
            extracted_props['final_electronic_energy'] = temp_final_electronic_energy
        elif file_extension == '.log':
            # Original .log file parsing (SCF Done)
            # Iterate to capture the LAST SCF Done energy
            for line in lines:
                if "SCF Done" in line:
                    parts = line.strip().split()
                    if "=" in parts:
                        idx = parts.index("=")
                        try:
                            extracted_props['final_electronic_energy'] = float(parts[idx + 1])
                        except (ValueError, IndexError):
                            pass

        # --- Conditional GIBBS FREE ENERGY EXTRACTION based on file type ---
        if file_extension == '.out':
            # ORCA .out specific parsing for Gibbs Free Energy
            # Iterate all lines and keep the LAST value (multi-step jobs)
            for line in lines:
                if "Final Gibbs free energy" in line:
                    try:
                        extracted_props['gibbs_free_energy'] = float(line.split()[-2])
                    except (ValueError, IndexError):
                        pass
        elif file_extension == '.log':
            # Original .log file parsing
            # Iterate to capture the LAST Gibbs Free Energy
            for line in lines:
                if "Sum of electronic and thermal Free Energies=" in line:
                    try:
                        extracted_props['gibbs_free_energy'] = float(line.strip().split()[-1])
                    except ValueError:
                        pass

        # --- Conditional DIPOLE MOMENT EXTRACTION based on file type ---
        if file_extension == '.out':
            # ORCA .out specific parsing (regex)
            # Iterate all lines and keep the LAST dipole (multi-step jobs)
            dipole_re = re.compile(r"Total Dipole Moment\s*:\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)")
            for line in lines:
                match = dipole_re.search(line)
                if match:
                    try:
                        x = float(match.group(1))
                        y = float(match.group(2))
                        z = float(match.group(3))
                        extracted_props['dipole_moment'] = np.linalg.norm([x, y, z])
                    except ValueError:
                        pass
        elif file_extension == '.log':
            # Original .log file parsing (X=, Y=, Z=)
            # Iterate to capture the LAST Dipole Moment
            current_dipole = None  # Use a temporary variable to hold the latest found value
            for i, line in enumerate(lines):
                if "Dipole moment" in line:
                    for j in range(i+1, min(i+6, len(lines))):
                        if ('X=' in lines[j]) and ('Y=' in lines[j]) and ('Z=' in lines[j]):
                            parts = lines[j].replace('=', ' ').split()
                            try:
                                x_index = parts.index('X') + 1
                                y_index = parts.index('Y') + 1
                                z_index = parts.index('Z') + 1
                                x = float(parts[x_index])
                                y = float(parts[y_index])
                                z = float(parts[z_index])
                                current_dipole = np.linalg.norm([x, y, z])
                            except (ValueError, IndexError):
                                current_dipole = None
                            break  # Break from inner loop after finding XYZ
            extracted_props['dipole_moment'] = current_dipole  # Assign the last found value
        # --- END Conditional DIPOLE MOMENT EXTRACTION ---

        # Extract HOMO/LUMO energies and gap using cclib ---
        # This part relies on cclib, which generally extracts the final values correctly for Gaussian.
        # Check for both homos and moenergies attributes with proper error handling
        try:
            if (hasattr(data, "homos") and hasattr(data, "moenergies") and
                data.homos is not None and data.moenergies is not None and  # type: ignore
                len(data.homos) > 0 and len(data.moenergies) > 0):  # type: ignore
                # Additional check to ensure moenergies[0] exists and is accessible
                if isinstance(data.moenergies[0], (list, np.ndarray)) and len(data.moenergies[0]) > 0:  # type: ignore
                    homo_index = data.homos[0]  # type: ignore
                    if homo_index >= 0 and (homo_index + 1) < len(data.moenergies[0]):  # type: ignore
                        homo_energy = data.moenergies[0][homo_index]  # type: ignore
                        lumo_energy = data.moenergies[0][homo_index + 1]  # type: ignore
                        extracted_props['homo_energy'] = homo_energy
                        extracted_props['lumo_energy'] = lumo_energy
                        extracted_props['homo_lumo_gap'] = lumo_energy - homo_energy
                    # Silently skip if indices out of bounds (expected for incomplete calculations)
                # Silently skip if moenergies[0] not accessible (expected for incomplete calculations)
            # Silently skip if homos/moenergies missing (expected for incomplete calculations)
        except (AttributeError, IndexError, TypeError):
            # Silently handle expected data access errors
            pass
        except Exception:
            # Silently handle other exceptions
            pass

        # Custom parsing for HOMO-LUMO Gap if cclib fails for .out files (e.g., semiempirical)
        if extracted_props['homo_lumo_gap'] is None and file_extension == '.out':
            # Iterate all lines and keep the LAST gap value (multi-step jobs)
            homo_lumo_gap_re = re.compile(r":: HOMO-LUMO gap\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*eV\s*::")
            for line in lines:
                match = homo_lumo_gap_re.search(line)
                if match:
                    try:
                        extracted_props['homo_lumo_gap'] = float(match.group(1))
                    except ValueError:
                        pass

        if file_extension == '.out' and (
            extracted_props['homo_energy'] is None or
            extracted_props['lumo_energy'] is None or
            extracted_props['homo_lumo_gap'] is None
        ):
            homo_energy, lumo_energy, homo_lumo_gap = extract_homo_lumo_from_orca_text(lines)
            if extracted_props['homo_energy'] is None:
                extracted_props['homo_energy'] = homo_energy
            if extracted_props['lumo_energy'] is None:
                extracted_props['lumo_energy'] = lumo_energy
            if extracted_props['homo_lumo_gap'] is None:
                extracted_props['homo_lumo_gap'] = homo_lumo_gap

        # --- Conditional ROTATIONAL CONSTANTS EXTRACTION based on file type ---
        # First, try cclib for all file types
        # This part also relies on cclib for rotconsts, which should get the final ones.
        if hasattr(data, "rotconsts") and data.rotconsts is not None and len(data.rotconsts) > 0:  # type: ignore
            rot_consts_candidate = None
            try:
                if isinstance(data.rotconsts, list) and len(data.rotconsts) > 0:  # type: ignore
                    rot_consts_candidate = np.asarray(data.rotconsts[0])  # type: ignore
                elif isinstance(data.rotconsts, np.ndarray) and data.rotconsts.ndim > 0:  # type: ignore
                    if data.rotconsts.ndim > 1:  # type: ignore
                        rot_consts_candidate = data.rotconsts[-1]  # type: ignore
                    else:
                        rot_consts_candidate = data.rotconsts  # type: ignore

                if rot_consts_candidate is not None and rot_consts_candidate.ndim == 1 and len(rot_consts_candidate) == 3:
                    # cclib reports rotational constants in GHz; normalize to cm^-1
                    extracted_props['rotational_constants'] = rot_consts_candidate.astype(float) / 29.9792458
                else:
                    # Silently skip if unexpected format (will try custom parse below)
                    extracted_props['rotational_constants'] = None
            except Exception:
                # Silently skip exceptions (will try custom parse below)
                extracted_props['rotational_constants'] = None

        # Fallback to custom parsing for .out files if cclib fails
        if extracted_props['rotational_constants'] is None and file_extension == '.out':
            # Iterate all lines and keep the LAST rotational constants (multi-step jobs)
            rot_const_re = re.compile(r"Rotational constants in cm-1:\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)")
            for line in lines:
                match = rot_const_re.search(line)
                if match:
                    try:
                        extracted_props['rotational_constants'] = np.array([float(match.group(1)),
                                                                             float(match.group(2)),
                                                                             float(match.group(3))])
                    except ValueError:
                        pass
        # --- END Conditional ROTATIONAL CONSTANTS EXTRACTION ---

        # Fallback: compute rotational constants from geometry if not available from output
        if extracted_props['rotational_constants'] is None and \
           extracted_props['final_geometry_atomnos'] is not None and \
           extracted_props['final_geometry_coords'] is not None and \
           len(extracted_props['final_geometry_atomnos']) > 0:
            computed_rc = calculate_rotational_constants(
                extracted_props['final_geometry_atomnos'],
                extracted_props['final_geometry_coords']
            )
            if computed_rc is not None:
                extracted_props['rotational_constants'] = computed_rc

        if extracted_props['final_geometry_atomnos'] is not None and extracted_props['final_geometry_coords'] is not None:
            if len(extracted_props['final_geometry_atomnos']) > 0 and extracted_props['final_geometry_coords'].shape[0] > 0:
                try:
                    radius_gyr = calculate_radius_of_gyration(extracted_props['final_geometry_atomnos'], extracted_props['final_geometry_coords'])
                    extracted_props['radius_of_gyration'] = radius_gyr
                except Exception:
                    pass

                # V_NN nuclear repulsion: try cclib attribute first, fall back to geometric.
                vnn = None
                for attr in ("nuclear_repulsion", "nuclear_repulsion_energy"):
                    val = getattr(data, attr, None)
                    if val is not None:
                        try:
                            vnn = float(val)
                            break
                        except (TypeError, ValueError):
                            pass
                if vnn is None:
                    try:
                        vnn = calculate_nuclear_repulsion(
                            extracted_props['final_geometry_atomnos'],
                            extracted_props['final_geometry_coords'],
                        )
                    except Exception:
                        vnn = None
                extracted_props['vnn_nuclear_repulsion'] = vnn
            # Silently skip if empty atomnos or coords (expected for incomplete calculations)

        # Extract vibrational frequencies
        if hasattr(data, "vibfreqs") and data.vibfreqs is not None and len(data.vibfreqs) > 0:  # type: ignore
            extracted_props['_has_freq_calc'] = True
            try:
                if len(data.vibfreqs) > 0 and isinstance(data.vibfreqs[0], (int, float, np.number)):  # type: ignore
                    imag_freqs = [freq for freq in data.vibfreqs if freq < 0]  # type: ignore
                    real_freqs = [freq for freq in data.vibfreqs if freq > 0]  # type: ignore

                    extracted_props['num_imaginary_freqs'] = len(imag_freqs)
                    extracted_props['_has_imaginary_freqs'] = len(imag_freqs) > 0

                    if len(imag_freqs) > 0:
                        print(f"  INFO: {os.path.basename(logfile_path)} contains {len(imag_freqs)} imaginary freq(s) - will be filtered after clustering")

                    if real_freqs:
                        extracted_props['first_vib_freq'] = min(real_freqs)
                        extracted_props['last_vib_freq'] = max(real_freqs)
                    else:
                        extracted_props['first_vib_freq'] = None
                        extracted_props['last_vib_freq'] = None
                # Silently skip if frequencies not valid numeric data (expected for incomplete calculations)
            except Exception:
                # Silently skip exceptions during frequency extraction
                pass
        else:
            # Explicitly set flag if no freq calc detected
            extracted_props['_has_freq_calc'] = False
            extracted_props['_has_imaginary_freqs'] = False
            # Ensure these are None if no frequencies are found, which they should be by default
            extracted_props['num_imaginary_freqs'] = None
            extracted_props['first_vib_freq'] = None
            extracted_props['last_vib_freq'] = None

        # Detect Hydrogen Bonds
        if extracted_props['final_geometry_atomnos'] is not None and extracted_props['final_geometry_coords'] is not None:
            if len(extracted_props['final_geometry_atomnos']) > 0 and extracted_props['final_geometry_coords'].shape[0] > 0:
                try:
                    # Call the updated detect_hydrogen_bonds function
                    hbond_results = detect_hydrogen_bonds(extracted_props['final_geometry_atomnos'], extracted_props['final_geometry_coords'])

                    extracted_props.update(hbond_results)  # Update all hbond related keys
                except Exception:
                    # Silently skip if hydrogen bond detection fails (expected for incomplete data)
                    pass
            # Silently skip if empty atomnos or coords (expected for incomplete calculations)

        return extracted_props

    except Exception as e:
        print(f"  GENERIC_ERROR: Failed to extract properties from {os.path.basename(logfile_path)} (after cclib parse): {e}")
        return None


def extract_properties_with_opi(logfile_path):
    """
    Extract properties using OPI (ORCA Python Interface) for ORCA 6.1+.
    """
    if not OPI_AVAILABLE:
        print(f"  ERROR: OPI is not installed. Please install it with: pip install orca-pi")
        return None

    assert OPIOutput is not None  # guaranteed by OPI_AVAILABLE check above

    # Initialize extracted properties dictionary with default values
    extracted_props = {
        'filename': os.path.basename(logfile_path),
        'method': "Unknown",
        'functional': "Unknown",
        'basis_set': "Unknown",
        'charge': None,
        'multiplicity': None,
        'num_atoms': 0,
        'final_geometry_atomnos': None,
        'final_geometry_coords': None,
        'final_electronic_energy': None,
        'gibbs_free_energy': None,
        'homo_energy': None,
        'lumo_energy': None,
        'homo_lumo_gap': None,
        'dipole_moment': None,
        'rotational_constants': None,
        'radius_of_gyration': None,
        'vnn_nuclear_repulsion': None,
        'num_imaginary_freqs': None,
        'first_vib_freq': None,
        'last_vib_freq': None,
        'num_hydrogen_bonds': 0,
        'hbond_details': [],
        'average_hbond_distance': None,
        'min_hbond_distance': None,
        'max_hbond_distance': None,
        'std_hbond_distance': None,
        'average_hbond_angle': None,
        'min_hbond_angle': None,
        'max_hbond_angle': None,
        'std_hbond_angle': None,
        '_has_freq_calc': False,
        '_initial_cluster_label': None,
        '_parent_global_cluster_id': None,
        '_first_rmsd_context_listing': None,
        '_second_rmsd_sub_cluster_id': None,
        '_second_rmsd_context_listing': None,
        '_second_rmsd_rep_filename': None,
        '_rmsd_pass_origin': None
    }

    try:
        # Read file content for text-based parsing
        with open(logfile_path, 'r', encoding='utf-8', errors='ignore') as f_log:
            lines = f_log.readlines()

        file_extension = os.path.splitext(logfile_path)[1].lower()

        # Extract basename from file path
        file_path = PathLib(logfile_path)
        basename = file_path.stem
        working_dir = file_path.parent

        # --- Attempt OPI JSON-based parsing first ---
        opi_parsed = False
        opi_output = None
        try:
            opi_output = OPIOutput(
                basename=basename,
                working_dir=working_dir,
                version_check=False
            )

            # Check if output terminated normally
            if not opi_output.terminated_normally():
                vprint(f"  WARNING: ORCA did not terminate normally for {os.path.basename(logfile_path)}")
                return None

            # Try to create JSON files if they don't exist (requires orca_2json and .property.txt)
            prop_json = working_dir / f"{basename}.property.json"
            prop_txt = working_dir / f"{basename}.property.txt"

            if not prop_json.exists() and prop_txt.exists():
                try:
                    opi_output.create_property_json()
                except Exception:
                    pass  # orca_2json might not be available

            # Try parsing (no arguments in installed OPI version)
            if prop_json.exists():
                opi_output.parse()
                if opi_output.results_properties and opi_output.results_properties.geometries:
                    opi_parsed = True
                    vprint(f"  OPI JSON parsing succeeded for {os.path.basename(logfile_path)}")
        except Exception as e:
            vprint(f"  OPI JSON parsing not available for {os.path.basename(logfile_path)}: {e}")

        # --- Text-based fallback (always works with just .out files) ---
        # This is the primary path when only .out files exist (no JSON)

        # Check terminated normally via text if OPI wasn't used
        if opi_output is None:
            terminated = False
            for line in reversed(lines):
                if 'ORCA TERMINATED NORMALLY' in line:
                    terminated = True
                    break
            if not terminated:
                vprint(f"  WARNING: ORCA did not terminate normally for {os.path.basename(logfile_path)}")
                return None

        # Method info (extract from input section in output file)
        extracted_props['method'] = _extract_method_from_file_opi(lines)
        extracted_props['functional'] = extracted_props['method']
        extracted_props['basis_set'] = _extract_basis_from_file_opi(lines)

        # Charge and multiplicity from input section: "* xyz CHARGE MULT"
        _extract_charge_mult_from_file_opi(lines, extracted_props)

        # --- Geometry extraction ---
        if opi_parsed and opi_output is not None:
            # Try OPI geometry (JSON-based, coordinates in Bohr)
            try:
                geom = opi_output.results_properties.geometries[-1]
                if hasattr(geom.geometry, 'coordinates') and geom.geometry.coordinates:
                    coords_data = geom.geometry.coordinates.cartesians
                    atomnos = []
                    coordinates = []

                    atomic_numbers = {
                        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
                        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
                        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
                        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
                        'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
                        'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Mo': 42, 'Ru': 44, 'Rh': 45,
                        'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52,
                        'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Pt': 78, 'Au': 79,
                        'Hg': 80, 'Pb': 82, 'Bi': 83
                    }

                    coord_units = getattr(geom.geometry.coordinates, 'units', None)
                    need_conversion = True
                    if coord_units and 'angst' in str(coord_units).lower():
                        need_conversion = False

                    for atom_data in coords_data:
                        element = atom_data[0]
                        atomnos.append(atomic_numbers.get(element, 0))
                        if need_conversion:
                            coordinates.append([
                                atom_data[1] * BOHR_TO_ANGSTROM,
                                atom_data[2] * BOHR_TO_ANGSTROM,
                                atom_data[3] * BOHR_TO_ANGSTROM
                            ])
                        else:
                            coordinates.append([atom_data[1], atom_data[2], atom_data[3]])

                    extracted_props['final_geometry_atomnos'] = np.array(atomnos)
                    extracted_props['final_geometry_coords'] = np.array(coordinates)
                    extracted_props['num_atoms'] = len(atomnos)
            except Exception:
                pass

        # Text-based geometry fallback: "CARTESIAN COORDINATES (ANGSTROEM)"
        if extracted_props['final_geometry_coords'] is None:
            natoms, coords, symbols = extract_xyz_from_output(logfile_path)
            if natoms and coords is not None and symbols:
                # Convert symbols to atomic numbers
                sym_to_num = {
                    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
                    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
                    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
                    'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
                    'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
                    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Mo': 42, 'Ru': 44, 'Rh': 45,
                    'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52,
                    'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Pt': 78, 'Au': 79,
                    'Hg': 80, 'Pb': 82, 'Bi': 83
                }
                atomnos = np.array([sym_to_num.get(s, 0) for s in symbols])
                extracted_props['final_geometry_atomnos'] = atomnos
                extracted_props['final_geometry_coords'] = coords
                extracted_props['num_atoms'] = natoms

        # --- Energy extraction from text ---
        # FINAL SINGLE POINT ENERGY - take the last occurrence in the file
        # (for OptFreq jobs this is always the converged geometry's energy)
        temp_final_energy = None
        for line in reversed(lines):
            if "FINAL SINGLE POINT ENERGY" in line:
                try:
                    parts = line.strip().split()
                    temp_final_energy = float(parts[-1])
                    break
                except (ValueError, IndexError):
                    pass
        extracted_props['final_electronic_energy'] = temp_final_energy

        # --- Gibbs Free Energy from text ---
        for line in lines:
            if "Final Gibbs free energy" in line:
                try:
                    parts = line.strip().split()
                    extracted_props['gibbs_free_energy'] = float(parts[-2])
                except (ValueError, IndexError):
                    pass

        # --- HOMO-LUMO gap from text ---
        # Format: ":: HOMO-LUMO gap             12.261913785019 eV    ::"
        homo_lumo_gap_re = re.compile(r'::\s*HOMO-LUMO gap\s+(-?\d+\.?\d*)\s+eV\s*::')
        last_gap = None
        for line in lines:
            match = homo_lumo_gap_re.search(line)
            if match:
                try:
                    last_gap = float(match.group(1))
                except ValueError:
                    pass
        if last_gap is not None:
            extracted_props['homo_lumo_gap'] = last_gap

        homo_energy, lumo_energy, homo_lumo_gap = extract_homo_lumo_from_orca_text(lines)
        if extracted_props['homo_energy'] is None:
            extracted_props['homo_energy'] = homo_energy
        if extracted_props['lumo_energy'] is None:
            extracted_props['lumo_energy'] = lumo_energy
        if extracted_props['homo_lumo_gap'] is None:
            extracted_props['homo_lumo_gap'] = homo_lumo_gap

        # --- Dipole moment from text ---
        # Format: "Magnitude (Debye)      :      2.38138"
        last_dipole = None
        for line in lines:
            if "Magnitude (Debye)" in line:
                try:
                    parts = line.split(':')
                    if len(parts) > 1:
                        last_dipole = float(parts[-1].strip())
                except (ValueError, IndexError):
                    pass
        # Also try "Total Dipole Moment" for norm calculation
        if last_dipole is None:
            dipole_re = re.compile(r"Total Dipole Moment\s*:\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)")
            for line in lines:
                match = dipole_re.search(line)
                if match:
                    try:
                        dx, dy, dz = float(match.group(1)), float(match.group(2)), float(match.group(3))
                        last_dipole = np.sqrt(dx*dx + dy*dy + dz*dz)
                    except ValueError:
                        pass
        if last_dipole is not None:
            extracted_props['dipole_moment'] = last_dipole

        # --- Vibrational frequencies from text ---
        _extract_vibfreqs_from_file_opi(lines, extracted_props, logfile_path)

        # --- Rotational constants from text ---
        extracted_props['rotational_constants'] = _extract_rotconsts_from_file_opi(lines)

        # Fallback: compute rotational constants from geometry if not available from output
        if extracted_props['rotational_constants'] is None and \
           extracted_props['final_geometry_atomnos'] is not None and \
           extracted_props['final_geometry_coords'] is not None and \
           len(extracted_props['final_geometry_atomnos']) > 0:
            computed_rc = calculate_rotational_constants(
                extracted_props['final_geometry_atomnos'],
                extracted_props['final_geometry_coords']
            )
            if computed_rc is not None:
                extracted_props['rotational_constants'] = computed_rc

        # --- Calculate radius of gyration ---
        if extracted_props['final_geometry_atomnos'] is not None and extracted_props['final_geometry_coords'] is not None:
            if len(extracted_props['final_geometry_atomnos']) > 0 and extracted_props['final_geometry_coords'].shape[0] > 0:
                try:
                    radius_gyr = calculate_radius_of_gyration(extracted_props['final_geometry_atomnos'], extracted_props['final_geometry_coords'])
                    extracted_props['radius_of_gyration'] = radius_gyr
                except Exception:
                    pass

                # V_NN nuclear repulsion: ORCA prints "Nuclear Repulsion ... ENuc ..." — try
                # the text first, fall back to geometric computation.
                vnn = _extract_nuclear_repulsion_from_file_opi(lines)
                if vnn is None:
                    try:
                        vnn = calculate_nuclear_repulsion(
                            extracted_props['final_geometry_atomnos'],
                            extracted_props['final_geometry_coords'],
                        )
                    except Exception:
                        vnn = None
                extracted_props['vnn_nuclear_repulsion'] = vnn

        # --- Detect Hydrogen Bonds ---
        if extracted_props['final_geometry_atomnos'] is not None and extracted_props['final_geometry_coords'] is not None:
            if len(extracted_props['final_geometry_atomnos']) > 0 and extracted_props['final_geometry_coords'].shape[0] > 0:
                try:
                    hbond_results = detect_hydrogen_bonds(extracted_props['final_geometry_atomnos'], extracted_props['final_geometry_coords'])
                    extracted_props.update(hbond_results)
                except Exception:
                    pass

        return extracted_props

    except Exception as e:
        print(f"  GENERIC_ERROR: Failed to extract properties from {os.path.basename(logfile_path)} with OPI: {e}")
        return None


def _extract_method_from_file_opi(lines):
    """Extract method/functional from ORCA output file lines."""
    try:
        for line in lines:
            # Match any input line: |  1> !, |  2> !, |  3> !, etc.
            if re.search(r'\|\s*\d+>\s*!', line) or re.search(r'\|\s*\d+>\s*#', line):
                parts = line.split('!') if '!' in line else line.split('#')
                if len(parts) > 1:
                    keywords = parts[1].strip().split()
                    for kw in keywords:
                        kw_upper = kw.upper()
                        # DFT functionals and methods
                        if any(x in kw_upper for x in ['B3LYP', 'PBE', 'TPSS', 'M06', 'B97', 'R2SCAN', 'HF', 'MP2', 'CCSD', 'DLPNO']):
                            return kw
                        # Semiempirical methods (GFN-xTB, PM3, AM1, etc.)
                        if any(x in kw_upper for x in ['GFN', 'XTB', 'PM3', 'PM6', 'PM7', 'AM1', 'MNDO', 'ZINDO']):
                            return kw
        # Also check for "Your calculation utilizes the semiempirical" line
        for line in lines:
            if 'Your calculation utilizes the semiempirical' in line:
                parts = line.split('semiempirical')
                if len(parts) > 1:
                    method = parts[1].strip().split()[0]
                    return method
    except:
        pass
    return "Unknown"


def _extract_basis_from_file_opi(lines):
    """Extract basis set from ORCA output file lines."""
    try:
        for line in lines:
            # Match any input line: |  1> !, |  2> !, |  3> !, etc.
            if re.search(r'\|\s*\d+>\s*!', line) or re.search(r'\|\s*\d+>\s*#', line):
                parts = line.split('!') if '!' in line else line.split('#')
                if len(parts) > 1:
                    keywords = parts[1].strip().split()
                    for kw in keywords:
                        kw_upper = kw.upper()
                        if any(x in kw_upper for x in ['DEF2', 'CC-PV', 'AUG-CC', 'STO', 'SVP', 'TZVP', 'QZVP', '6-31', '6-311']):
                            return kw
    except:
        pass
    return "Unknown"


def _extract_charge_mult_from_file_opi(lines, extracted_props):
    """Extract charge and multiplicity from ORCA input section in output file."""
    try:
        for line in lines:
            # Match "* xyz CHARGE MULT" or "* xyzfile CHARGE MULT ..."
            match = re.search(r'\*\s*(?:xyz|xyzfile|int|gzmt)\s+(-?\d+)\s+(\d+)', line, re.IGNORECASE)
            if match:
                extracted_props['charge'] = int(match.group(1))
                extracted_props['multiplicity'] = int(match.group(2))
                return
    except:
        pass


def _extract_vibfreqs_from_file_opi(lines, extracted_props, logfile_path):
    """Extract vibrational frequencies from ORCA output file lines."""
    try:
        # Look for "VIBRATIONAL FREQUENCIES" section
        freq_section_start = None
        for i, line in enumerate(lines):
            if 'VIBRATIONAL FREQUENCIES' in line and '---' not in line:
                freq_section_start = i

        if freq_section_start is None:
            return

        freqs = []
        for i in range(freq_section_start + 3, len(lines)):
            line = lines[i].strip()
            # Skip empty lines and lines without frequency data
            if not line:
                continue
            # Stop at section boundaries (multiple dashes or equals, or specific keywords)
            if '------' in line or '======' in line or 'NORMAL MODES' in line:
                break
            parts = line.split()
            # Format: "   0:         0.00 cm**-1" or "   6:       123.45 cm**-1"
            if len(parts) >= 3 and parts[0].endswith(':') and 'cm' in parts[2]:
                try:
                    freq_val = float(parts[1])
                    if freq_val != 0.0:
                        freqs.append(freq_val)
                except ValueError:
                    continue

        if freqs:
            extracted_props['_has_freq_calc'] = True
            imag_freqs = [f for f in freqs if f < 0]
            real_freqs = [f for f in freqs if f > 0]

            extracted_props['num_imaginary_freqs'] = len(imag_freqs)
            extracted_props['_has_imaginary_freqs'] = len(imag_freqs) > 0

            if len(imag_freqs) > 0:
                print(f"  INFO: {os.path.basename(logfile_path)} contains {len(imag_freqs)} imaginary freq(s) - will be filtered after clustering")

            if real_freqs:
                extracted_props['first_vib_freq'] = min(real_freqs)
                extracted_props['last_vib_freq'] = max(real_freqs)
    except Exception as e:
        if console.VERBOSE:
            print(f"  DEBUG: Error extracting frequencies from {os.path.basename(logfile_path)}: {e}")


def _extract_rotconsts_from_file_opi(lines):
    """Extract rotational constants from ORCA output file lines, normalized to cm^-1."""
    try:
        last_match = None
        for line in lines:
            is_cm = 'Rotational constants in cm-1:' in line
            is_mhz = 'Rotational constants in MHz:' in line
            if is_cm or is_mhz:
                parts = line.split(':')
                if len(parts) > 1:
                    values = parts[1].strip().split()
                    if len(values) >= 3:
                        arr = np.array([float(values[0]), float(values[1]), float(values[2])])
                        # MHz → cm^-1 : divide by c[cm/s] × 1e-6 = 29979.2458
                        last_match = arr / 29979.2458 if is_mhz else arr
        return last_match
    except:
        pass
    return None


def _extract_nuclear_repulsion_from_file_opi(lines):
    """Extract V_NN (Hartree) from an ORCA output. Returns the last match or None.

    ORCA prints lines such as ``Nuclear Repulsion      ENuc            ...   Eh``.
    """
    try:
        nuc_re = re.compile(
            r"Nuclear\s+Repulsion[^:\n]*?(-?\d+\.\d+)\s*Eh", re.IGNORECASE
        )
        last_match = None
        for line in lines:
            m = nuc_re.search(line)
            if m:
                last_match = float(m.group(1))
        return last_match
    except Exception:
        return None


def extract_homo_lumo_from_orca_text(lines):
    """Extract HOMO/LUMO from an ORCA ORBITAL ENERGIES section — cosmic-v01.py 4215-4281."""
    last_section = None
    current_section = []
    in_section = False

    for line in lines:
        if 'ORBITAL ENERGIES' in line:
            in_section = True
            current_section = []
            continue

        if not in_section:
            continue

        stripped = line.strip()
        if not stripped:
            if current_section:
                last_section = current_section[:]
                in_section = False
            continue

        if '----' in stripped or 'NO   OCC' in stripped or 'NO  OCC' in stripped:
            continue

        if stripped.startswith('*') or stripped.startswith('JOB NUMBER') or stripped.startswith('MULLIKEN'):
            if current_section:
                last_section = current_section[:]
            in_section = False
            continue

        current_section.append(stripped)

    if in_section and current_section:
        last_section = current_section[:]

    if not last_section:
        return None, None, None

    parsed_rows = []
    for row in last_section:
        parts = row.split()
        if len(parts) < 4:
            continue
        try:
            occupation = float(parts[1])
            energy_ev = float(parts[3])
        except ValueError:
            continue
        parsed_rows.append((occupation, energy_ev))

    if not parsed_rows:
        return None, None, None

    homo_energy = None
    lumo_energy = None
    for occupation, energy_ev in parsed_rows:
        if occupation > 0:
            homo_energy = energy_ev
        elif occupation == 0 and lumo_energy is None:
            lumo_energy = energy_ev
            break

    if homo_energy is None or lumo_energy is None:
        return None, None, None

    return homo_energy, lumo_energy, lumo_energy - homo_energy


__all__ = [
    "CCLIB_AVAILABLE",
    "OPI_AVAILABLE",
    "extract_homo_lumo_from_orca_text",
    "extract_properties_with_cclib",
    "extract_properties_with_opi",
    "extract_properties_with_xtb",
]
