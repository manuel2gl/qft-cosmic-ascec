#!/usr/bin/env python3

import argparse
import os
import numpy as np
import warnings
# matplotlib import moved to functions that use it for faster startup
# cclib imported at top level for multiprocessing compatibility
try:
    from cclib.io import ccread
    CCLIB_AVAILABLE = True
except ImportError:
    CCLIB_AVAILABLE = False
    ccread = None

# OPI (ORCA Python Interface) for ORCA 6.1+ support
from pathlib import Path as PathLib  # stdlib, always available
try:
    from opi.output.core import Output as OPIOutput # type: ignore
    OPI_AVAILABLE = True
except ImportError:
    OPI_AVAILABLE = False
    OPIOutput = None

# sklearn and scipy imports moved to functions that use them for faster startup
import glob
import re
import subprocess
import shutil
import pickle # Added for caching
import multiprocessing as mp  # Added for parallel processing
from functools import partial  # Added for parallel processing
import sys  # Added for custom argument parsing
# scipy.spatial.transform import moved to function that uses it


# Physical constants for Boltzmann distribution
BOLTZMANN_CONSTANT_HARTREE_PER_K = 3.1668114e-6  # Hartree/K (k_B in atomic units)
DEFAULT_TEMPERATURE_K = 298.15  # K (Room temperature)

# Energy conversion constants
HARTREE_TO_KCAL_MOL = 627.509474  # kcal/mol per Hartree
HARTREE_TO_EV = 27.211386245988  # eV per Hartree
BOHR_TO_ANGSTROM = 0.529177210903  # Angstrom per Bohr

# Script version
version = "* COSMIC-v01: Feb-2026 *"

def print_version_banner():
    """Print the ASCII art banner with UDEA logo and version information."""
    banner = """
===========================================================================

                        ***************************                        
                        *       C O S M I C       *                          
                        ***************************                        

                             √≈≠==≈                                  
   √≈≠==≠≈√   √≈≠==≠≈√         ÷++=                      ≠===≠       
     ÷++÷       ÷++÷           =++=                     ÷×××××=      
     =++=       =++=     ≠===≠ ÷++=      ≠====≠         ÷-÷ ÷-÷      
     =++=       =++=    =××÷=≠=÷++=    ≠÷÷÷==÷÷÷≈      ≠××≠ =××=     
     =++=       =++=   ≠××=    ÷++=   ≠×+×    ×+÷      ÷+×   ×+××    
     =++=       =++=   =+÷     =++=   =+-×÷==÷×-×≠    =×+×÷=÷×+-÷    
     ≠×+÷       ÷+×≠   =+÷     =++=   =+---×××××÷×   ≠××÷==×==÷××≠   
      =××÷     =××=    ≠××=    ÷++÷   ≠×-×           ÷+×       ×+÷   
       ≠=========≠      ≠÷÷÷=≠≠=×+×÷-  ≠======≠≈√  -÷×+×≠     ≠×+×÷- 
          ≠===≠           ≠==≠  ≠===≠     ≠===≠    ≈====≈     ≈====≈ 


               Universidad de Antioquia - Medellín - Colombia              


           Clustering Analysis for Quantum Chemistry Calculations          

                        {version}                       


                        Química Física Teórica - QFT                       


===========================================================================
""".format(version=version)
    print(banner)

# Global variables
VERBOSE = False  # Control verbosity of output
_CPU_COUNT_CACHE = None  # Cache CPU count to avoid repeated calls
_DATASET_HAS_FREQ = True  # Set by perform_clustering_and_analysis; True = freq mode (original), False = opt-only


def _sorting_energy(mol_data):
    """Return the energy value to use for sorting/selection of representatives.

    In freq mode (_DATASET_HAS_FREQ=True): uses Gibbs free energy (original behaviour).
    In opt-only mode (_DATASET_HAS_FREQ=False): uses electronic energy.
    Returns float('inf') when no valid energy is found so the structure sorts last.
    """
    if _DATASET_HAS_FREQ:
        g = mol_data.get('gibbs_free_energy')
        if g is not None:
            return g
        return float('inf')
    else:
        e = mol_data.get('final_electronic_energy')
        if e is not None:
            return e
        return float('inf')


def hartree_to_kcal_mol(energy_hartree):
    """Convert energy from Hartree to kcal/mol"""
    return energy_hartree * HARTREE_TO_KCAL_MOL

def hartree_to_ev(energy_hartree):
    """Convert energy from Hartree to eV"""
    return energy_hartree * HARTREE_TO_EV

def detect_motif_input_level(filenames):
    """
    Detect the input naming level to determine the appropriate output prefix.
    
    Naming convention:
    - conf_### or opt_conf_### → First step (preoptimization) → outputs motif_##
    - motif_## or motif_##_opt → Second step (optimization) → outputs umotif_##
    - umotif_## → Third step (if applicable) → outputs umotif_## (no further prefix)
    
    Args:
        filenames: List of input file names to analyze
        
    Returns:
        tuple: (output_prefix, folder_prefix, is_second_step)
            - output_prefix: 'motif' or 'umotif'
            - folder_prefix: 'motifs' or 'umotifs'
            - is_second_step: True if input files are already motifs
    """
    if not filenames:
        return 'motif', 'motifs', False
    
    # Count files matching each pattern
    motif_pattern = re.compile(r'^motif_\d+', re.IGNORECASE)
    umotif_pattern = re.compile(r'^umotif_\d+', re.IGNORECASE)
    
    motif_count = 0
    umotif_count = 0
    
    for filename in filenames:
        base_name = os.path.splitext(os.path.basename(filename))[0]
        if umotif_pattern.match(base_name):
            umotif_count += 1
        elif motif_pattern.match(base_name):
            motif_count += 1
    
    total_files = len(filenames)
    
    # If majority of files are umotif, keep using umotif
    if umotif_count > total_files * 0.5:
        return 'umotif', 'umotifs', True
    
    # If majority of files are motif, use umotif for output
    if motif_count > total_files * 0.5:
        return 'umotif', 'umotifs', True
    
    # Default: first step, use motif
    return 'motif', 'motifs', False

def vprint(message, **kwargs):
    """Print message only if verbose mode is enabled"""
    if VERBOSE:
        print(message, **kwargs)

def print_step(message, **kwargs):
    """Print concise step information (always shown)"""
    print(message, **kwargs)

def get_cpu_count_fast():
    """
    Get CPU count using fast methods with caching.
    Tries multiple detection methods for maximum compatibility.
    
    Returns:
        int: Number of available CPU cores
    """
    global _CPU_COUNT_CACHE
    
    if _CPU_COUNT_CACHE is not None:
        return _CPU_COUNT_CACHE
    
    # Try os.cpu_count() first (fastest method)
    try:
        cpu_count = os.cpu_count()
        if cpu_count is not None and cpu_count > 0:
            _CPU_COUNT_CACHE = cpu_count
            return cpu_count
    except (OSError, AttributeError):
        pass
    
    # Try /proc/cpuinfo (Linux)
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpu_count = sum(1 for line in f if line.startswith('processor'))
            if cpu_count > 0:
                _CPU_COUNT_CACHE = cpu_count
                return cpu_count
    except (FileNotFoundError, IOError):
        pass
    
    # Try nproc command (Linux)
    try:
        result = subprocess.run(['nproc'], capture_output=True, text=True, timeout=1.0)
        if result.returncode == 0:
            cpu_count = int(result.stdout.strip())
            if cpu_count > 0:
                _CPU_COUNT_CACHE = cpu_count
                return cpu_count
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass
    
    # Use multiprocessing.cpu_count() as fallback
    try:
        cpu_count = mp.cpu_count()
        if cpu_count > 0:
            _CPU_COUNT_CACHE = cpu_count
            return cpu_count
    except (OSError, AttributeError):
        pass
    
    # Final fallback
    _CPU_COUNT_CACHE = 24
    return 24

# Element masses dictionary (in atomic mass units)
element_masses = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.012, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723,
    "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906,
    "Mo": 95.96, "Ru": 101.07, "Rh": 102.906, "Pd": 106.42, "Ag": 107.868,
    "Cd": 112.414, "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.60,
    "I": 126.904, "Xe": 131.293, "Cs": 132.905, "Ba": 137.327, "La": 138.905,
    "Ce": 140.116, "Pr": 140.908, "Nd": 144.242, "Sm": 150.36, "Eu": 151.964,
    "Gd": 157.25, "Tb": 158.925, "Dy": 162.500, "Ho": 164.930, "Er": 167.259,
    "Tm": 168.934, "Yb": 173.04, "Lu": 174.967, "Hf": 178.49, "Ta": 180.948,
    "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084,
    "Au": 196.967, "Hg": 200.592, "Tl": 204.383, "Pb": 207.2, "Bi": 208.980,
    "Th": 232.038, "Pa": 231.036, "U": 238.028
}

def atomic_number_to_symbol(atomic_number):
    """
    Convert atomic number to element symbol.
    
    Args:
        atomic_number (int): Atomic number
        
    Returns:
        str: Element symbol
    """
    periodic_table_symbols = [
        "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
        "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb",
        "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
        "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
        "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
        "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
        "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
        "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg",
        "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ]
    if 0 <= atomic_number < len(periodic_table_symbols):
        return periodic_table_symbols[atomic_number]
    else:
        return str(atomic_number)

def calculate_deviation_percentage(values):
    """
    Calculate percentage deviation (max-min / abs(mean)) for a list of values.
    
    Args:
        values (list): List of numerical values
        
    Returns:
        float: Percentage deviation
    """
    if not values or len(values) < 2:
        return 0.0
    
    numeric_values = [v for v in values if v is not None]
    if not numeric_values:
        return 0.0

    min_val = min(numeric_values)
    max_val = max(numeric_values)

    if min_val == 0.0 and max_val == 0.0:
        return 0.0

    if max_val == min_val:
        return 0.0

    mean_val = np.mean(numeric_values)
    if mean_val == 0.0:
        return 100.0 if max_val != min_val else 0.0
    
    return ((max_val - min_val) / abs(mean_val)) * 100.0


def calculate_rms_deviation(values):
    """
    Calculate Root Mean Square Deviation (RMSD) for a list of values.
    
    Args:
        values (list): List of numerical values
        
    Returns:
        float: RMSD value
    """
    if not values:
        return 0.0
    
    mean_val = np.mean(values)
    squared_deviations = [(val - mean_val)**2 for val in values]
    rmsd = np.sqrt(np.mean(squared_deviations))
    
    return rmsd

def extract_xyz_from_output(output_file):
    """
    Extract XYZ coordinates from ORCA or Gaussian output file.
    Works for both file types by detecting the format.
    
    Args:
        output_file: Path to .out or .log file
        
    Returns:
        tuple: (natoms, coords_array, symbols_list) or (None, None, None) if failed
    """
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        # Detect file type
        is_orca = False
        is_gaussian = False
        for line in lines[:100]:
            if 'O   R   C   A' in line:
                is_orca = True
                break
            if 'Gaussian' in line:
                is_gaussian = True
                break
        
        if is_orca:
            # Extract from ORCA "CARTESIAN COORDINATES (ANGSTROEM)" section (last occurrence)
            coord_start = None
            for i in range(len(lines)-1, -1, -1):
                if 'CARTESIAN COORDINATES (ANGSTROEM)' in lines[i]:
                    coord_start = i + 2
                    break
            
            if coord_start is None:
                return None, None, None
            
            coords = []
            symbols = []
            for i in range(coord_start, len(lines)):
                line = lines[i].strip()
                if not line or line.startswith('---') or line.startswith('='):
                    break
                parts = line.split()
                if len(parts) >= 4:
                    symbol = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    symbols.append(symbol)
                    coords.append([x, y, z])
            
            if coords:
                return len(coords), np.array(coords), symbols
        
        elif is_gaussian:
            # Extract from Gaussian "Standard orientation" section (last occurrence)
            coord_start = None
            for i in range(len(lines)-1, -1, -1):
                if 'Standard orientation:' in lines[i]:
                    coord_start = i + 5
                    break
            
            if coord_start is None:
                return None, None, None
            
            coords = []
            symbols = []
            for i in range(coord_start, len(lines)):
                line = lines[i].strip()
                if line.startswith('---'):
                    break
                parts = line.split()
                if len(parts) >= 6:
                    atomic_num = int(parts[1])
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                    # Convert atomic number to symbol
                    symbol = atomic_number_to_symbol(atomic_num)
                    symbols.append(symbol)
                    coords.append([x, y, z])
            
            if coords:
                return len(coords), np.array(coords), symbols
        
        return None, None, None
        
    except Exception as e:
        return None, None, None

def calculate_radius_of_gyration(atomnos, atomcoords):
    """
    Calculate radius of gyration for a molecule.
    
    Args:
        atomnos (array): Atomic numbers
        atomcoords (array): Atomic coordinates (N, 3)
        
    Returns:
        float: Radius of gyration in Angstroms, or None if error
    """
    try:
        symbols = [atomic_number_to_symbol(n) for n in atomnos]
        masses = np.array([element_masses.get(s, 0.0) for s in symbols])
        
        coords = np.asarray(atomcoords)

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected atomcoords to be (N, 3), but got shape {coords.shape}")

        total_mass = np.sum(masses)
        if total_mass == 0.0:
            return None
        
        center_of_mass = np.sum(coords.T * masses, axis=1) / total_mass
        
        rg_squared = np.sum(masses * np.sum((coords - center_of_mass)**2, axis=1)) / total_mass
        return np.sqrt(rg_squared)
    except Exception as e:
        print(f"Error in calculate_radius_of_gyration: {e}")
        return None

def calculate_rotational_constants(atomnos, atomcoords):
    """
    Calculate rotational constants (A, B, C) from the moment of inertia tensor.

    Uses the principal moments of inertia Ia <= Ib <= Ic (in amu*Å²)
    and converts to rotational constants A >= B >= C (in GHz) via:

        X = h / (8 π² Ix)

    where h is Planck's constant and Ix is a principal moment.

    Args:
        atomnos (array): Atomic numbers.
        atomcoords (array): Atomic coordinates (N, 3) in Angstroms.

    Returns:
        numpy.ndarray of shape (3,) with [A, B, C] in GHz, or None on error.
    """
    try:
        symbols = [atomic_number_to_symbol(n) for n in atomnos]
        masses = np.array([element_masses.get(s, 0.0) for s in symbols])

        coords = np.asarray(atomcoords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            return None

        total_mass = np.sum(masses)
        if total_mass == 0.0:
            return None

        # Centre of mass
        com = np.sum(coords.T * masses, axis=1) / total_mass
        r = coords - com  # centred coordinates

        # Inertia tensor (amu·Å²)
        Ixx = np.sum(masses * (r[:, 1]**2 + r[:, 2]**2))
        Iyy = np.sum(masses * (r[:, 0]**2 + r[:, 2]**2))
        Izz = np.sum(masses * (r[:, 0]**2 + r[:, 1]**2))
        Ixy = -np.sum(masses * r[:, 0] * r[:, 1])
        Ixz = -np.sum(masses * r[:, 0] * r[:, 2])
        Iyz = -np.sum(masses * r[:, 1] * r[:, 2])

        inertia_tensor = np.array([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz]
        ])

        # Principal moments (ascending)
        eigenvalues = np.linalg.eigvalsh(inertia_tensor)

        # Conversion factor: amu·Å² → kg·m²
        # 1 amu = 1.66053906660e-27 kg, 1 Å = 1e-10 m
        amu_ang2_to_kg_m2 = 1.66053906660e-27 * (1e-10)**2  # 1.66053906660e-47

        # h / (8 π²) in SI (J·s / rad²) → divide by I in kg·m² → Hz → GHz
        h = 6.62607015e-34  # J·s
        factor = h / (8.0 * np.pi**2)  # J·s / rad²

        rot_consts = []
        for I_val in eigenvalues:
            if I_val <= 0.0:
                rot_consts.append(0.0)
            else:
                I_si = I_val * amu_ang2_to_kg_m2
                freq_hz = factor / I_si
                rot_consts.append(freq_hz * 1e-9)  # Hz → GHz

        # Convention: A >= B >= C
        rot_consts.sort(reverse=True)
        return np.array(rot_consts, dtype=float)
    except Exception:
        return None


def detect_hydrogen_bonds(atomnos, atomcoords):
    """
    Detect hydrogen bonds based on distance and angle criteria.
    
    Distance criterion: 1.2-3.2 Å between H and acceptor
    Angle criterion: D-H...A angle >= 30°
    
    Args:
        atomnos (array): Atomic numbers
        atomcoords (array): Atomic coordinates (N, 3)
        
    Returns:
        dict: Dictionary with hydrogen bond information
    """
    try:
        coords = np.asarray(atomcoords)

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected atom coords to be (N, 3), but got shape {coords.shape}")

        # Hydrogen bond criteria
        potential_donor_acceptor_z = {7, 8, 9}  # N, O, F
        hydrogen_atom_num = 1
        HB_min_dist_actual = 1.2  # Minimum H...A distance (Å)
        HB_max_dist_actual = 3.2  # Maximum H...A distance (Å)
        COVALENT_DH_SEARCH_DIST = 1.5  # D-H covalent bond search limit (Å)
        HB_min_angle_actual = 30.0  # Minimum D-H...A angle (degrees)
        HB_max_angle_actual = 180.0  # Maximum D-H...A angle (degrees)

        symbols = [atomic_number_to_symbol(n) for n in atomnos]
        atom_labels = [f"{sym}{i+1}" for i, sym in enumerate(symbols)]
        all_potential_hbonds_details = []
        
        # First pass: Identify covalently bonded donor (D) for each hydrogen (H)
        h_covalent_donors = {}  # {h_idx: (donor_idx, D-H_distance)}
        for i_h, h_atom_num in enumerate(atomnos):
            if h_atom_num != hydrogen_atom_num:
                continue

            coord_h = coords[i_h]
            
            min_dist_dh = float('inf')
            donor_idx_for_h = -1
            
            for i_d, d_atom_num in enumerate(atomnos):
                if i_d == i_h:
                    continue

                if d_atom_num in potential_donor_acceptor_z:
                    dist_dh = np.linalg.norm(coords[i_d] - coord_h)

                    if dist_dh < min_dist_dh:
                        min_dist_dh = dist_dh
                        donor_idx_for_h = i_d
            
            if donor_idx_for_h != -1 and min_dist_dh <= COVALENT_DH_SEARCH_DIST:
                h_covalent_donors[i_h] = (donor_idx_for_h, min_dist_dh)

        # Second pass: Detect potential H-bonds and calculate angles
        for i_h, h_atom_num in enumerate(atomnos):
            if h_atom_num != hydrogen_atom_num:
                continue

            if i_h not in h_covalent_donors:
                continue

            donor_idx, actual_dh_covalent_distance = h_covalent_donors[i_h]
            coord_h = coords[i_h]
            coord_d = coords[donor_idx]

            for i_a, a_atom_num in enumerate(atomnos):
                if i_a == i_h or i_a == donor_idx:
                    continue

                if a_atom_num in potential_donor_acceptor_z:
                    coord_a = coords[i_a]

                    dist_ha = np.linalg.norm(coord_h - coord_a)
                    
                    if HB_min_dist_actual <= dist_ha <= HB_max_dist_actual:
                        vec_h_d = coord_d - coord_h
                        vec_h_a = coord_a - coord_h
                        
                        norm_vec_h_d = np.linalg.norm(vec_h_d)
                        norm_vec_h_a = np.linalg.norm(vec_h_a)

                        if norm_vec_h_d == 0 or norm_vec_h_a == 0:
                            angle_deg = 0.0
                        else:
                            dot_product = np.dot(vec_h_d, vec_h_a)
                            cos_angle = dot_product / (norm_vec_h_d * norm_vec_h_a)
                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                            angle_rad = np.arccos(cos_angle)
                            angle_deg = np.degrees(angle_rad)

                        all_potential_hbonds_details.append({
                            'donor_atom_label': atom_labels[donor_idx],
                            'hydrogen_atom_label': atom_labels[i_h],
                            'acceptor_atom_label': atom_labels[i_a],
                            'H...A_distance': dist_ha,
                            'D-H...A_angle': angle_deg,
                            'D-H_covalent_distance': actual_dh_covalent_distance
                        })
        
        # Filter bonds meeting angle criterion for statistics
        filtered_hbonds_for_stats = [
            b for b in all_potential_hbonds_details 
            if HB_min_angle_actual <= b['D-H...A_angle'] <= HB_max_angle_actual
        ]
        extracted_props = {
            'num_hydrogen_bonds': len(filtered_hbonds_for_stats),
            'hbond_details': all_potential_hbonds_details,
            'average_hbond_distance': None,
            'min_hbond_distance': None,
            'max_hbond_distance': None,
            'std_hbond_distance': None,
            'average_hbond_angle': None,
            'min_hbond_angle': None,
            'max_hbond_angle': None,
            'std_hbond_angle': None
        }

        hbonds_for_geometry_stats = filtered_hbonds_for_stats if filtered_hbonds_for_stats else all_potential_hbonds_details

        if hbonds_for_geometry_stats:
            distances = [bond['H...A_distance'] for bond in hbonds_for_geometry_stats]
            angles = [bond['D-H...A_angle'] for bond in hbonds_for_geometry_stats]
            
            extracted_props['average_hbond_distance'] = np.mean(distances)
            extracted_props['min_hbond_distance'] = np.min(distances)
            extracted_props['max_hbond_distance'] = np.max(distances)
            extracted_props['std_hbond_distance'] = np.std(distances) if len(distances) > 1 else 0.0

            extracted_props['average_hbond_angle'] = np.mean(angles)
            extracted_props['min_hbond_angle'] = np.min(angles)
            extracted_props['max_hbond_angle'] = np.max(angles)
            extracted_props['std_hbond_angle'] = np.std(angles) if len(angles) > 1 else 0.0

        return extracted_props

    except Exception as e:
        return {
            'num_hydrogen_bonds': 0,
            'hbond_details': [],
            'average_hbond_distance': None, 'min_hbond_distance': None, 
            'max_hbond_distance': None, 'std_hbond_distance': None,
            'average_hbond_angle': None, 'min_hbond_angle': None, 
            'max_hbond_angle': None, 'std_hbond_angle': None
        }

def detect_file_type(logfile_path):
    """
    Detect whether the output file is from ORCA or Gaussian.
    
    Args:
        logfile_path: Path to output file (.out or .log)
        
    Returns:
        'orca', 'gaussian', or None if not detected
    """
    try:
        with open(logfile_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read first 100 lines to detect file type
            for i, line in enumerate(f):
                if i > 100:
                    break
                if 'O   R   C   A' in line:
                    return 'orca'
                if 'Gaussian' in line:
                    return 'gaussian'
    except Exception:
        pass
    return None

def detect_orca_version(logfile_path):
    """
    Detect ORCA version from output file.
    
    Args:
        logfile_path: Path to ORCA output file (.out or .log)
        
    Returns:
        Tuple of (major, minor) version numbers, or None if not detected
    """
    try:
        with open(logfile_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Scan header section until we find version or hit input file section
            # This is robust to header size changes in future ORCA versions
            for i, line in enumerate(f):
                # Stop if we've reached the input file section (version should be before this)
                if 'INPUT FILE' in line or i > 500:
                    break
                # Look for version pattern like "Program Version 6.1.0" or "Program Version 7.0.0"
                match = re.search(r'Program Version\s+(\d+)\.(\d+)', line, re.IGNORECASE)
                if match:
                    major = int(match.group(1))
                    minor = int(match.group(2))
                    return (major, minor)
    except Exception:
        pass
    return None

def choose_parser(logfile_path):
    """
    Automatically choose the appropriate parser based on file type and available libraries.
    
    Args:
        logfile_path: Path to ORCA or Gaussian output file
        
    Returns:
        'opi' or 'cclib', or raises error if no suitable parser
    """
    # First, detect the file type (ORCA or Gaussian)
    file_type = detect_file_type(logfile_path)
    
    # For Gaussian files, always use cclib (OPI only supports ORCA)
    if file_type == 'gaussian':
        if CCLIB_AVAILABLE:
            return 'cclib'
        else:
            raise RuntimeError(
                "Gaussian output file detected, but cclib is not installed. "
                "Please install cclib: pip install cclib"
            )
    
    # For ORCA files, check version and choose appropriate parser
    if file_type == 'orca':
        version = detect_orca_version(logfile_path)
        
        # If version detected
        if version:
            major, minor = version
            
            # ORCA 6.1+ requires OPI (use tuple comparison for correct 7.0, 8.0 etc.)
            if (major, minor) >= (6, 1):
                if OPI_AVAILABLE:
                    return 'opi'
                else:
                    raise RuntimeError(
                        f"ORCA version {major}.{minor} detected, which requires OPI. "
                        "Please install OPI: pip install orca-pi"
                    )
            
            # ORCA 6.0 not supported by either
            elif major == 6 and minor == 0:
                raise RuntimeError(
                    f"ORCA version 6.0 is not supported. Please use ORCA 5.0.x or ORCA 6.1+. "
                    "cclib supports up to ORCA 5.0.x, OPI supports ORCA 6.1+"
                )
            
            # ORCA <=5.0 can use cclib
            else:
                if CCLIB_AVAILABLE:
                    return 'cclib'
                elif OPI_AVAILABLE:
                    warnings.warn(
                        f"ORCA version {major}.{minor} detected. OPI will be used, but cclib is recommended for ORCA <=5.0"
                    )
                    return 'opi'
                else:
                    raise RuntimeError("Neither cclib nor OPI available. Please install one of them.")
    
    # If file type not detected, use available parser (prefer cclib for compatibility)
    if CCLIB_AVAILABLE:
        return 'cclib'
    elif OPI_AVAILABLE:
        return 'opi'
    else:
        raise RuntimeError("Neither cclib nor OPI available. Please install one of them.")

def process_file_parallel_wrapper(file_path):
    """
    Wrapper function for parallel processing of files.
    Returns a tuple of (success, result, filename) to handle both successful extractions and skipped files.
    """
    try:
        filename = os.path.basename(file_path)
        # Avoid noisy parser fallback errors for empty output files.
        if os.path.getsize(file_path) == 0:
            print(f"  WARNING: {filename} is empty (0 bytes). Skipping.")
            return (False, None, filename)
        extracted_props = extract_properties_from_logfile(file_path)
        if extracted_props:
            return (True, extracted_props, filename)  # Success
        else:
            return (False, None, filename)  # Skipped (likely imaginary frequencies)
    except Exception as e:
        print(f"  ERROR: Failed to process {os.path.basename(file_path)}: {e}")
        return (False, None, os.path.basename(file_path))  # Error

def extract_properties_from_logfile(logfile_path):
    """
    Extract molecular properties from quantum chemistry log file.
    Supports both cclib (ORCA <=5.0) and OPI (ORCA >=6.1).
    
    Args:
        logfile_path (str): Path to .log or .out file
        
    Returns:
        dict: Extracted properties, or None if parsing fails
    """
    if not CCLIB_AVAILABLE and not OPI_AVAILABLE:
        print(f"  ERROR: Neither cclib nor OPI is installed. Please install one: pip install cclib OR pip install orca-pi")
        return None
    
    # Determine which parser to use
    try:
        parser_type = choose_parser(logfile_path)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return None
    
    # Use the appropriate parser
    if parser_type == 'opi':
        return extract_properties_with_opi(logfile_path)
    else:
        return extract_properties_with_cclib(logfile_path)

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
            elif type(raw_data).__name__.startswith('ccData'): # Broaden check to include subclasses like ccData_optdone_bool
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
        'num_imaginary_freqs': None, # Changed default to None
        'first_vib_freq': None,
        'last_vib_freq': None,
        'num_hydrogen_bonds': 0, # Will be set by detect_hydrogen_bonds
        'hbond_details': [],     # Will be set by detect_hydrogen_bonds
        'average_hbond_distance': None,
        'min_hbond_distance': None,
        'max_hbond_distance': None,
        'std_hbond_distance': None,
        'average_hbond_angle': None,
        'min_hbond_angle': None,
        'max_hbond_angle': None,
        'std_hbond_angle': None,
        '_has_freq_calc': False, # New internal flag
        '_initial_cluster_label': None, # To store the integer label of the initial property cluster (from fcluster)
        '_parent_global_cluster_id': None, # New: Stores the 'Y' from "Validated from Cluster Y"
        '_first_rmsd_context_listing': None, # To store details of the original property cluster (RMSD to its rep)
        '_second_rmsd_sub_cluster_id': None, # New label for sub-cluster after 2nd RMSD
        '_second_rmsd_context_listing': None, # RMSD to representative of 2nd-level sub-cluster
        '_second_rmsd_rep_filename': None, # Filename of representative of 2nd-level sub-cluster
        '_rmsd_pass_origin': None # New flag to indicate if cluster came from 1st or 2nd RMSD pass
    }

    try:
        with open(logfile_path, 'r') as f_log:
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
            current_dipole = None # Use a temporary variable to hold the latest found value
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
                            break # Break from inner loop after finding XYZ
            extracted_props['dipole_moment'] = current_dipole # Assign the last found value
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
                    extracted_props['rotational_constants'] = rot_consts_candidate.astype(float)
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
                    
                    extracted_props.update(hbond_results) # Update all hbond related keys
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
        if VERBOSE:
            print(f"  DEBUG: Error extracting frequencies from {os.path.basename(logfile_path)}: {e}")

def _extract_rotconsts_from_file_opi(lines):
    """Extract rotational constants from ORCA output file lines."""
    try:
        last_match = None
        for line in lines:
            if 'Rotational constants in cm-1:' in line or 'Rotational constants in MHz:' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    values = parts[1].strip().split()
                    if len(values) >= 3:
                        last_match = np.array([float(values[0]), float(values[1]), float(values[2])])
        return last_match
    except:
        pass
    return None

# --- START: RMSD functions provided by user ---
def calculate_rmsd(atomnos1, coords1, atomnos2, coords2):
    """
    Calculates the RMSD between two sets of coordinates using Kabsch alignment,
    considering only heavy (non-hydrogen) atoms.

    Args:
        atomnos1 (list or np.array): Atomic numbers for structure 1.
        coords1 (np.array): Nx3 array of atomic coordinates for structure 1.
        atomnos2 (list or np.array): Atomic numbers for structure 2.
        coords2 (np.array): Nx3 array of atomic coordinates for structure 2.

    Returns:
        float: The calculated RMSD value, or None if an error occurs.
    """
    from scipy.spatial.transform import Rotation as R  # Import only when needed
    
    # Filter out hydrogen atoms (atomic number 1)
    heavy_indices1 = [i for i, z in enumerate(atomnos1) if z != 1]
    heavy_coords1 = coords1[heavy_indices1]
    # heavy_atomnos1 = [atomnos1[i] for i in heavy_indices1] # Not strictly needed for RMSD calculation itself

    heavy_indices2 = [i for i, z in enumerate(atomnos2) if z != 1]
    heavy_coords2 = coords2[heavy_indices2]
    # heavy_atomnos2 = [atomnos2[i] for i in heavy_indices2] # Not strictly needed for RMSD calculation itself

    if len(heavy_indices1) == 0 or len(heavy_indices2) == 0:
        return None

    if len(heavy_indices1) != len(heavy_indices2):
        # This check is crucial and implies mismatched heavy atom counts
        # It's better to explicitly check if lengths are different, not just shapes, as shape might match if coords is empty.
        return None

    # Ensure coordinates are numpy arrays and float64
    coords1_filtered = np.asarray(heavy_coords1, dtype=np.float64)
    coords2_filtered = np.asarray(heavy_coords2, dtype=np.float64)

    # Check for empty arrays after filtering, though len() check above should catch
    if coords1_filtered.shape[0] == 0 or coords2_filtered.shape[0] == 0:
        return None

    try:
        # Step 1: Center the coordinates (move to origin)
        center1 = np.mean(coords1_filtered, axis=0)
        centered_coords1 = coords1_filtered - center1

        center2 = np.mean(coords2_filtered, axis=0)
        centered_coords2 = coords2_filtered - center2

        # Step 2: Perform Kabsch alignment to find the optimal rotation
        # R.align_vectors(a, b) finds rotation and RMSD to transform a to align with b.
        # So here, we want to align centered_coords2 (source) to centered_coords1 (target).
        result = R.align_vectors(centered_coords2, centered_coords1)
        rotation = result[0]
        rmsd_value = result[1]  # The RMSD is always the second value returned

        # The 'rmsd_value' returned by align_vectors IS the minimized RMSD.
        # You don't need to re-apply the rotation and calculate it again.
        # If you did, it should yield the same 'rmsd_value'.
        # The VMD results strongly suggest that this 'rmsd_value' is the one you want.
        
        return rmsd_value

    except Exception as e:
        print(f"  ERROR during heavy atom RMSD calculation: {e}")
        return None

def post_process_clusters_with_rmsd(initial_clusters, rmsd_validation_threshold):
    """
    Refines property-based clusters by performing an RMSD validation within each.
    Configurations failing the RMSD check are extracted into new single-configuration clusters.

    Args:
        initial_clusters (list): A list of lists/tuples, where each inner list/tuple
                                 represents a property-based cluster and contains
                                 dictionaries of extracted data for each configuration.
        rmsd_validation_threshold (float): The maximum allowed RMSD value (in Angstroms) for
                                 configurations to remain in the same cluster.

    Returns:
        tuple: A tuple containing:
               - list: A new list of validated multi-configuration clusters (members passed RMSD to rep).
               - list: A list of individual outlier configurations (data dicts) that were split out.
    """
    validated_main_clusters = []
    individual_outliers = []

    print(f"  Initiating first pass RMSD validation with threshold: {rmsd_validation_threshold:.3f} Å...")

    for cluster_idx, current_property_cluster in enumerate(initial_clusters):
        if not current_property_cluster:
            continue

        if len(current_property_cluster) == 1:
            # Single member clusters are passed directly to validated_main_clusters

            current_property_cluster[0]['_rmsd_pass_origin'] = 'first_pass_validated'
            validated_main_clusters.append(current_property_cluster)
            continue

        print(f"    Validating initial property cluster {current_property_cluster[0].get('_parent_global_cluster_id', 'N/A')} with {len(current_property_cluster)} configurations...")

        # Select the lowest energy configuration as the representative for this property cluster

        representative_conf = min(current_property_cluster,
                                  key=lambda x: (_sorting_energy(x), x['filename']))

        current_validated_sub_cluster = [representative_conf] # Start new validated cluster with representative
        processed_members_filenames = {representative_conf['filename']}


        representative_conf['_rmsd_pass_origin'] = 'first_pass_validated'

        coords_rep = representative_conf.get('final_geometry_coords')
        atomnos_rep = representative_conf.get('final_geometry_atomnos')

        if coords_rep is None or atomnos_rep is None:
            print(f"    WARNING: Representative {representative_conf['filename']} has missing geometry. Skipping RMSD validation for this property cluster. All members kept together for now.")
            # If skipping, mark all as first_pass_validated
            for conf_member in current_property_cluster:
                conf_member['_rmsd_pass_origin'] = 'first_pass_validated'
            validated_main_clusters.append(current_property_cluster)
            continue

        other_members = [conf for conf in current_property_cluster if conf != representative_conf]

        for conf_member in other_members:
            if conf_member['filename'] in processed_members_filenames:
                continue

            coords_member = conf_member.get('final_geometry_coords')
            atomnos_member = conf_member.get('final_geometry_atomnos')

            if coords_member is None or atomnos_member is None:
                print(f"    WARNING: {conf_member['filename']} has missing geometry data. Treating as an individual outlier for now.")
                conf_member['_parent_global_cluster_id'] = representative_conf['_parent_global_cluster_id']
                conf_member['_rmsd_pass_origin'] = 'second_pass_formed'
                individual_outliers.append(conf_member) # Collect this as an outlier
                processed_members_filenames.add(conf_member['filename'])
                continue
            
            rmsd_val = calculate_rmsd(
                atomnos_rep, coords_rep,
                atomnos_member, coords_member
            )

            if rmsd_val is not None and rmsd_val <= rmsd_validation_threshold:
                current_validated_sub_cluster.append(conf_member)
                conf_member['_rmsd_pass_origin'] = 'first_pass_validated'
                processed_members_filenames.add(conf_member['filename'])
            else:
                print(f"    {conf_member['filename']} (RMSD={rmsd_val:.3f} Å) is an outlier from {representative_conf['filename']} (Threshold={rmsd_validation_threshold:.3f} Å).")
                conf_member['_parent_global_cluster_id'] = representative_conf['_parent_global_cluster_id']
                conf_member['_rmsd_pass_origin'] = 'second_pass_formed'
                individual_outliers.append(conf_member) # Collect this as an outlier
                processed_members_filenames.add(conf_member['filename'])

        if current_validated_sub_cluster:
             validated_main_clusters.append(current_validated_sub_cluster)

    return validated_main_clusters, individual_outliers

def filter_imaginary_freq_structures(clusters_list, output_base_dir, input_source=None, total_processed=None, write_summary=False, precomputed_skipped=None):
    """
    Filters clusters to handle structures with imaginary frequencies.
    
    Removes imaginary frequency structures from mixed clusters or saves them for recalculation
    if they form isolated clusters that may represent missing motifs.
    
    Args:
        clusters_list: List of clusters (each cluster is a list of structure dicts)
        output_base_dir: Base output directory for creating skipped_structures folder
        input_source: Input folder/files path for copying original files
        total_processed: Total number of configurations processed (for percentage calculation)
        write_summary: Whether to write the skipped summary file (default: False)
        precomputed_skipped: Dict with 'clustered_with_normal' and 'need_recalculation' lists if already collected
        
    Returns:
        tuple: (filtered_clusters, skipped_info_dict)
            - filtered_clusters: Clusters with imaginary freq structures removed
            - skipped_info_dict: Info about skipped structures
    """
    filtered_clusters = []
    
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
            if VERBOSE:
                for m in has_imag:
                    print(f"  INFO: Discarding {m['filename']} (imaginary freq) - clustered with true minima")
        elif has_imag and not no_imag:
            skipped_need_recalc.extend(has_imag)
            if VERBOSE:
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
        with open(summary_file, 'w', newline='\n') as f:
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
                        with open(xyz_file, 'w') as f:
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
                        with open(xyz_file, 'w') as f:
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
                    with open(combined_xyz, 'w') as f:
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
                    except:
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
                    except:
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
        
        if len(skipped_need_recalc) > 0 and VERBOSE:
            print(f"  → Review 'skipped_structures/skipped_summary.txt' for details")
        
        print()  # Blank line after "Processed"
    
    skipped_info = {
        'clustered_with_normal': skipped_clustered_with_normal,
        'need_recalculation': skipped_need_recalc
    }
    
    return filtered_clusters, skipped_info


def _is_non_converged_structure(mol_data, dataset_has_freq=True):
    """Return True when a structure should be treated as non-converged/critical.

    With dynamic vector clustering, reduced-vector structures that were
    successfully matched to a fullest-tier cluster are NOT treated as
    non-converged.  Only structures explicitly flagged as unmatched
    (``_reduced_unmatched``) or lacking essential data are critical.
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


def filter_non_converged_structures(clusters_list, dataset_has_freq=True):
    """
    Remove non-converged structures from cluster candidates.

    Any non-converged member is classified as critical for recalculation and
    cannot become a representative.
    """
    filtered_clusters = []
    critical_non_converged = []

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


def save_non_converged_critical_structures(non_converged_structures, output_base_dir, input_source=None, total_processed=None):
    """Write critical non-converged structures to skipped_structures/critical_non_converged/."""
    if not non_converged_structures:
        return

    skipped_dir = os.path.join(output_base_dir, "skipped_structures")
    critical_dir = os.path.join(skipped_dir, "critical_non_converged")
    os.makedirs(critical_dir, exist_ok=True)

    # Build filename->source path map from input source.
    available_files = {}
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
                    with open(xyz_file, 'w') as f:
                        f.write(f"{natoms}\n")
                        f.write(f"{basename} - critical non-converged\n")
                        for symbol, coord in zip(symbols, coords):
                            f.write(f"{symbol:2s}  {coord[0]:15.8f}  {coord[1]:15.8f}  {coord[2]:15.8f}\n")
                except Exception:
                    pass

    # Write a compact summary file for auditability.
    summary_path = os.path.join(critical_dir, "non_converged_summary.txt")
    try:
        with open(summary_path, 'w') as f:
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

def perform_second_rmsd_clustering(cluster_members_to_refine, rmsd_threshold):
    """
    Performs second RMSD-based clustering on configurations (typically outliers from first pass).
    
    Args:
        cluster_members_to_refine: List of configuration dictionaries to re-cluster
        rmsd_threshold: RMSD threshold for this clustering step
        
    Returns:
        list: New sub-clusters (each a list of data dictionaries)
    """
    from scipy.cluster.hierarchy import linkage, fcluster  # Import only when needed
    
    if len(cluster_members_to_refine) <= 1:
        for m in cluster_members_to_refine:
            m['_second_rmsd_sub_cluster_id'] = m.get('_initial_cluster_label')
            m['_second_rmsd_context_listing'] = [{'filename': m['filename'], 'rmsd_to_rep': 0.0}]
            m['_second_rmsd_rep_filename'] = m['filename']
            m['_rmsd_pass_origin'] = 'second_pass_formed'
        return [[m] for m in cluster_members_to_refine]

    num_members = len(cluster_members_to_refine)
    rmsd_matrix = np.zeros((num_members, num_members))

    for i in range(num_members):
        for j in range(i + 1, num_members):
            conf1 = cluster_members_to_refine[i]
            conf2 = cluster_members_to_refine[j]

            coords1 = conf1.get('final_geometry_coords')
            atomnos1 = conf1.get('final_geometry_atomnos')
            coords2 = conf2.get('final_geometry_coords')
            atomnos2 = conf2.get('final_geometry_atomnos')

            rmsd = calculate_rmsd(atomnos1, coords1, atomnos2, coords2)
            if rmsd is None:
                rmsd = float('inf')
            rmsd_matrix[i, j] = rmsd_matrix[j, i] = rmsd

    condensed_distances = []
    for i in range(num_members):
        for j in range(i + 1, num_members):
            condensed_distances.append(rmsd_matrix[i, j])

    if not condensed_distances:
        for m in cluster_members_to_refine:
            m['_second_rmsd_sub_cluster_id'] = m.get('_initial_cluster_label')
            m['_second_rmsd_context_listing'] = [{'filename': m['filename'], 'rmsd_to_rep': 0.0}]
            m['_second_rmsd_rep_filename'] = m['filename']
            m['_rmsd_pass_origin'] = 'second_pass_formed'
        return [[m] for m in cluster_members_to_refine]

    linkage_matrix = linkage(condensed_distances, method='average', metric='euclidean')
    second_cluster_labels = fcluster(linkage_matrix, t=rmsd_threshold, criterion='distance')

    second_level_clusters_data = {}
    for i, label in enumerate(second_cluster_labels):
        cluster_members_to_refine[i]['_second_rmsd_sub_cluster_id'] = label
        cluster_members_to_refine[i]['_rmsd_pass_origin'] = 'second_pass_formed'
        second_level_clusters_data.setdefault(label, []).append(cluster_members_to_refine[i])

    final_sub_clusters = []
    for label, sub_cluster_members in second_level_clusters_data.items():
        if not sub_cluster_members: continue

        sub_cluster_rep = min(sub_cluster_members,
                              key=lambda x: (_sorting_energy(x), x['filename']))
        
        sub_cluster_rmsd_listing = []
        if sub_cluster_rep.get('final_geometry_coords') is not None and sub_cluster_rep.get('final_geometry_atomnos') is not None:
            for member_conf in sub_cluster_members:
                if member_conf == sub_cluster_rep:
                    rmsd_val = 0.0
                else:
                    rmsd_val = calculate_rmsd(
                        sub_cluster_rep['final_geometry_atomnos'], sub_cluster_rep['final_geometry_coords'],
                        member_conf['final_geometry_atomnos'], member_conf['final_geometry_coords']
                    )
                sub_cluster_rmsd_listing.append({'filename': member_conf['filename'], 'rmsd_to_rep': rmsd_val})
        else:
            for member_conf in sub_cluster_members:
                sub_cluster_rmsd_listing.append({'filename': member_conf['filename'], 'rmsd_to_rep': None})

        for member_conf in sub_cluster_members:
            member_conf['_second_rmsd_context_listing'] = sub_cluster_rmsd_listing
            member_conf['_second_rmsd_rep_filename'] = sub_cluster_rep['filename']
        
        final_sub_clusters.append(sub_cluster_members)
    
    return final_sub_clusters
# --- END: RMSD functions provided by user ---


def write_cluster_dat_file(dat_file_prefix, cluster_members_data, output_base_dir, rmsd_threshold_value=None, 
                           hbond_count_for_original_cluster=None, weights=None, tolerances=None):
    """
    Writes combined .dat file for cluster members, including comparison and RMSD context sections.
    
    Args:
        dat_file_prefix: Desired name for .dat file (e.g., 'cluster_12' or 'cluster_12_5')
        cluster_members_data: List of configuration dictionaries
        output_base_dir: Base output directory
        rmsd_threshold_value: RMSD threshold used
        hbond_count_for_original_cluster: H-bond count for initial group
        weights: Dictionary mapping feature names to weights for conditional printing
        tolerances: Dictionary mapping feature names to absolute tolerances
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


    with open(output_filename, 'w', newline='\n') as f:
        f.write("=" * 90 + "\n\n")

        rmsd_context_printed = False

        if rmsd_threshold_value is not None and cluster_members_data and '_first_rmsd_context_listing' in cluster_members_data[0] and cluster_members_data[0]['_first_rmsd_context_listing'] is not None:
            initial_rmsd_context = cluster_members_data[0]['_first_rmsd_context_listing']
            f.write("Initial Clustering RMSD Context (Before Refinement):\n")
            f.write("Configurations from the original property cluster:\n")
            for item in initial_rmsd_context:
                rmsd_val_str = f"({item['rmsd_to_rep']:.3f} Å)" if item['rmsd_to_rep'] is not None else "(N/A)"
                f.write(f"    - {item['filename']} {rmsd_val_str}\n")
            f.write("\n")
            
            original_prop_cluster_label = cluster_members_data[0].get('_initial_cluster_label', 'N/A')
            parent_global_cluster_id_for_display = cluster_members_data[0].get('_parent_global_cluster_id', 'N/A')

            hbond_context = f" (H-bonds {hbond_count_for_original_cluster})" if hbond_count_for_original_cluster is not None else ""
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

        # 5. Deviation Analysis (ONLY for clusters with >1 configuration)
        if num_configurations > 1:
            # Dynamically detect which features are NOT available in all cluster members
            _zero_weight = {k for k, v in (weights or {}).items() if v == 0.0}

            # All deviation entries: (display_name, data_extractor, feature_key_for_filter)
            _deviation_entries = [
                ("Electronic Energy (Hartree)", lambda d: d.get('final_electronic_energy'), "electronic_energy"),
                ("Gibbs Free Energy (Hartree)", lambda d: d.get('gibbs_free_energy'), "gibbs_free_energy"),
                ("HOMO Energy (eV)", lambda d: d.get('homo_energy'), "homo_energy"),
                ("LUMO Energy (eV)", lambda d: d.get('lumo_energy'), "lumo_energy"),
                ("HOMO-LUMO Gap (eV)", lambda d: d.get('homo_lumo_gap'), "homo_lumo_gap"),
                ("Dipole Moment (Debye)", lambda d: d.get('dipole_moment'), "dipole_moment"),
                ("Radius of Gyration (Å)", lambda d: d.get('radius_of_gyration'), "radius_of_gyration"),
                ("Rotational Constant A (GHz)", lambda d: d['rotational_constants'][0] if d.get('rotational_constants') is not None and isinstance(d.get('rotational_constants'), np.ndarray) and len(d.get('rotational_constants')) == 3 else None, "rotational_constants_A"),
                ("Rotational Constant B (GHz)", lambda d: d['rotational_constants'][1] if d.get('rotational_constants') is not None and isinstance(d.get('rotational_constants'), np.ndarray) and len(d.get('rotational_constants')) == 3 else None, "rotational_constants_B"),
                ("Rotational Constant C (GHz)", lambda d: d['rotational_constants'][2] if d.get('rotational_constants') is not None and isinstance(d.get('rotational_constants'), np.ndarray) and len(d.get('rotational_constants')) == 3 else None, "rotational_constants_C"),
                ("First Vibrational Frequency (cm^-1)", lambda d: d.get('first_vib_freq'), "first_vib_freq"),
                ("Last Vibrational Frequency (cm^-1)", lambda d: d.get('last_vib_freq'), "last_vib_freq"),
                ("Average H-Bond Distance (Å)", lambda d: d.get('average_hbond_distance'), "average_hbond_distance"),
                ("Average H-Bond Angle (°)", lambda d: d.get('average_hbond_angle'), "average_hbond_angle"),
            ]

            # Dynamically detect features not available in all cluster members
            _feat_display_map = {
                'electronic_energy': 'Electronic Energy', 'gibbs_free_energy': 'Gibbs Free Energy',
                'homo_energy': 'HOMO Energy', 'lumo_energy': 'LUMO Energy',
                'homo_lumo_gap': 'HOMO-LUMO Gap', 'dipole_moment': 'Dipole Moment',
                'radius_of_gyration': 'Radius of Gyration',
                'rotational_constants_A': 'Rotational Constant A',
                'rotational_constants_B': 'Rotational Constant B',
                'rotational_constants_C': 'Rotational Constant C',
                'first_vib_freq': 'First Vibrational Frequency',
                'last_vib_freq': 'Last Vibrational Frequency',
                'average_hbond_distance': 'Average H-Bond Distance',
                'average_hbond_angle': 'Average H-Bond Angle',
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
                if feat_key in _zero_weight:
                    continue
                values = [extractor(d) for d in cluster_members_data]
                if feat_key in _missing_features:
                    f.write(f"  {display_name} %Dev: N/A\n")
                else:
                    write_deviation_line(f, display_name, values)

            # --- Weights and tolerances display order ---
            weight_display_order = [
                ("electronic_energy", "Electronic Energy", "final_electronic_energy"),
                ("gibbs_free_energy", "Gibbs Free Energy", "gibbs_free_energy"),
                ("homo_energy", "HOMO Energy", "homo_energy"),
                ("lumo_energy", "LUMO Energy", "lumo_energy"),
                ("homo_lumo_gap", "HOMO-LUMO Gap", "homo_lumo_gap"),
                ("dipole_moment", "Dipole Moment", "dipole_moment"),
                ("radius_of_gyration", "Radius of Gyration", "radius_of_gyration"),
                ("rotational_constants_A", "Rotational Constant A", "rotational_constants"),
                ("rotational_constants_B", "Rotational Constant B", "rotational_constants"),
                ("rotational_constants_C", "Rotational Constant C", "rotational_constants"),
                ("first_vib_freq", "First Vibrational Frequency", "first_vib_freq"),
                ("last_vib_freq", "Last Vibrational Frequency", "last_vib_freq"),
                ("average_hbond_distance", "Average H-Bond Distance", "average_hbond_distance"),
                ("average_hbond_angle", "Average H-Bond Angle", "average_hbond_angle")
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
        # This section always prints the descriptors for each member of the cluster.
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
            write_scalar_descriptor_line(f, "HOMO Energy (eV)", mol_data.get('homo_energy'), lambda value: f"{value:.6f}")
            write_scalar_descriptor_line(f, "LUMO Energy (eV)", mol_data.get('lumo_energy'), lambda value: f"{value:.6f}")
            write_scalar_descriptor_line(f, "HOMO-LUMO Gap (eV)", mol_data.get('homo_lumo_gap'), lambda value: f"{value:.6f}")
        f.write("\n")

        f.write("Molecular configuration descriptors:\n")
        for mol_data in cluster_members_data:
            f.write(f"    {mol_data['filename']}:\n")
            write_scalar_descriptor_line(f, "Dipole Moment (Debye)", mol_data.get('dipole_moment'), lambda value: f"{value:.6f}")
            rc = mol_data.get('rotational_constants')
            if rc is not None and isinstance(rc, np.ndarray) and rc.ndim == 1 and len(rc) == 3:
                f.write(f"        Rotational Constants (GHz): {rc[0]:.6f}, {rc[1]:.6f}, {rc[2]:.6f}\n")
            else:
                f.write("        Rotational Constants (GHz): N/A\n")
            write_scalar_descriptor_line(f, "Radius of Gyration (Å)", mol_data.get('radius_of_gyration'), lambda value: f"{value:.6f}")
            write_scalar_descriptor_line(f, "Average H-Bond Distance (Å)", mol_data.get('average_hbond_distance'), lambda value: f"{value:.6f}")
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
        HB_min_angle_actual_for_display = 30.0 # Define explicitly for display
        f.write(f"Criterion: H...A distance between 1.2 Å and 3.2 Å, with H covalently bonded to a donor (O, N, F).\n")
        f.write(f"  (For counting, D-H...A angle must be >= {HB_min_angle_actual_for_display:.1f}°)\n")
        for mol_data in cluster_members_data:
            f.write(f"    {mol_data['filename']}:\n")
            num_counted_hb = mol_data.get('num_hydrogen_bonds', 0)
            total_potential_hb = len(mol_data.get('hbond_details', []))
            f.write(f"        Number of hydrogen bonds counted (angle >= 30.0°): {num_counted_hb} out of {total_potential_hb} potential bonds.\n")
            
            if mol_data.get('hbond_details'):
                for hbond in mol_data['hbond_details']:
                    angle_note = ""
                    if hbond['D-H...A_angle'] < HB_min_angle_actual_for_display:
                        angle_note = f" (Angle < {HB_min_angle_actual_for_display:.1f}° - Not counted as HB)"
                    f.write(f"            Hydrogen bond: {hbond['donor_atom_label']}-{hbond['hydrogen_atom_label']}...{hbond['acceptor_atom_label']} (Dist: {hbond['H...A_distance']:.3f} Å, D-H: {hbond['D-H_covalent_distance']:.3f} Å, Angle: {hbond['D-H...A_angle']:.2f}°){angle_note}\n")
            else:
                f.write("        No hydrogen bonds detected based on the criterion.\n")
        f.write("\n")

        # RMSD comparison section (for clusters with multiple configurations)
        if num_configurations > 1:
            f.write("RMSD Analysis (Heavy Atoms):\n")
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
                                mol_j['final_geometry_coords']
                            )
                            row.append(rmsd_val if rmsd_val is not None else float('nan'))
                        else:
                            # Mirror the upper triangle
                            row.append(rmsd_matrix[j][i])
                    rmsd_matrix.append(row)
                
                # Display RMSD values in a readable format
                # For each configuration, show RMSD to all others
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
        # This section now only includes general file info and geometry,
        # avoiding the repetition of descriptor summaries.
        for i, mol_data in enumerate(cluster_members_data):
            if i > 0: # Add shorter separator only before subsequent structures
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

            # Removed the repeated descriptor blocks here (Electronic, Molecular, Vibrational, Hydrogen bond)
            # as they are already printed in Section 6 for each member of the cluster.

    vprint(f"Wrote combined data for Cluster '{dat_file_prefix}' to '{os.path.basename(output_filename)}'")

def write_xyz_file(mol_data, filename):
    """
    Writes atomic coordinates to an XYZ file with energy in the comment line.
    Freq mode: Gibbs free energy (original).  Opt-only mode: electronic energy.
    """
    atomnos = mol_data.get('final_geometry_atomnos')
    atomcoords = mol_data.get('final_geometry_coords')

    if atomnos is None or atomcoords is None or len(atomnos) == 0:
        print(f"  WARNING: Cannot write XYZ for {os.path.basename(filename)}: Missing geometry data.")
        return

    base_name = os.path.splitext(os.path.basename(mol_data['filename']))[0]

    if _DATASET_HAS_FREQ:
        gibbs_free_energy = mol_data.get('gibbs_free_energy')
        gibbs_str = f"{gibbs_free_energy:.6f} Hartree ({hartree_to_kcal_mol(gibbs_free_energy):.2f} kcal/mol, {hartree_to_ev(gibbs_free_energy):.2f} eV)" if gibbs_free_energy is not None else "N/A"
        comment_line = f"{base_name} (G = {gibbs_str})"
    else:
        electronic_energy = mol_data.get('final_electronic_energy')
        elec_str = f"{electronic_energy:.6f} Hartree ({hartree_to_kcal_mol(electronic_energy):.2f} kcal/mol, {hartree_to_ev(electronic_energy):.2f} eV)" if electronic_energy is not None else "N/A"
        comment_line = f"{base_name} (E = {elec_str})"

    symbols = [atomic_number_to_symbol(n) for n in atomnos]

    with open(filename, 'w', newline='\n') as f:
        f.write(f"{len(atomnos)}\n")
        f.write(f"{comment_line}\n")
        for i in range(len(atomnos)):
            f.write(f"{symbols[i]:<2} {atomcoords[i][0]:10.6f} {atomcoords[i][1]:10.6f} {atomcoords[i][2]:10.6f}\n")

def create_unique_motifs_folder(all_clusters_data, output_base_dir, openbabel_alias="obabel", cluster_id_mapping=None, output_prefix='motif', folder_prefix='motifs', boltzmann_data=None, dataset_has_freq=True):
    """
    Creates a motifs/umotifs folder containing the lowest energy representative structure from each cluster.
    Also creates a combined XYZ file with all representatives and attempts to convert to MOL format.
    
    Args:
        all_clusters_data (list): List of clusters, where each cluster is a list of molecule data dictionaries
        output_base_dir (str): Base output directory where motifs folder will be created
        openbabel_alias (str): Alias for OpenBabel command (default: "obabel")
        cluster_id_mapping (dict): Optional mapping from cluster index to original cluster ID
        output_prefix (str): Prefix for output files - 'motif' for first step, 'umotif' for second step
        folder_prefix (str): Prefix for output folder - 'motifs' or 'umotifs'
        boltzmann_data (dict): Optional Boltzmann population data keyed by cluster_id, 
                              each entry has 'filename' and 'population' for sorting by population
        
    Returns:
        dict: Mapping from motif number to cluster ID
    """
    if not all_clusters_data:
        print("  No clusters found. Skipping motifs creation.")
        return {}
    

    num_motifs = len(all_clusters_data)
    motifs_dir = os.path.join(output_base_dir, f"{folder_prefix}_{num_motifs:02d}")
    os.makedirs(motifs_dir, exist_ok=True)
    
    # Determine display name based on prefix
    display_name = "unique motifs" if output_prefix == 'umotif' else "motifs"
    
    print()
    print()
    print_step(f"Creating {num_motifs} {display_name} from cluster representatives...")
    vprint(f"  Output directory: {motifs_dir}")
    
    representatives = []
    representative_cluster_ids = []
    
    for cluster_idx, cluster_members in enumerate(all_clusters_data):
        if not cluster_members:
            continue
        
        # CRITICAL: No motif can have imaginary frequencies or non-converged data.
        # Representatives must come from the fullest-tier structures only.
        # Reduced-vector structures absorbed into a cluster cannot be representatives.
        if dataset_has_freq:
            valid_members = [
                m for m in cluster_members
                if not m.get('_has_imaginary_freqs', False)
                and m.get('gibbs_free_energy') is not None
                and m.get('_is_full_feature', True)
            ]
        else:
            valid_members = [
                m for m in cluster_members
                if m.get('final_electronic_energy') is not None
                and m.get('_is_full_feature', True)
            ]
        
        if not valid_members:
            # All members are invalid for representative selection.
            print(f"  WARNING: Cluster {cluster_idx + 1} has no converged minima - skipping motif creation")
            continue
            
        # Find the lowest energy representative from valid (non-imaginary) members only
        representative = min(valid_members,
                           key=lambda x: (_sorting_energy(x), x['filename']))
        
        # Get the cluster ID for this representative
        cluster_id = cluster_id_mapping[cluster_idx] if cluster_id_mapping else cluster_idx + 1
        
        representatives.append(representative)
        representative_cluster_ids.append(cluster_id)
    
    # Sort representatives by Boltzmann population (if available) or Gibbs free energy as fallback
    representatives_with_ids = list(zip(representatives, representative_cluster_ids))
    
    if boltzmann_data:
        # Create a filename-to-population mapping for sorting
        filename_to_population = {}
        for cluster_id_key, data in boltzmann_data.items():
            filename_to_population[data['filename']] = data['population']
        
        def get_population_for_rep(rep_tuple):
            """Get Boltzmann population for a representative, or -inf if not found (to sort last)."""
            rep, cid = rep_tuple
            filename = rep.get('filename', '')
            return filename_to_population.get(filename, -float('inf'))
        
        # Sort by population descending (highest population = motif_01)
        sorted_representatives_with_ids = sorted(
            representatives_with_ids,
            key=lambda x: (-get_population_for_rep(x), x[0]['filename'])  # Negative for descending
        )
    else:
        # Fallback: sort by electronic energy (lowest = motif_01)
        sorted_representatives_with_ids = sorted(
            representatives_with_ids,
            key=lambda x: (_sorting_energy(x[0]), x[0]['filename'])
        )
    

    for motif_idx, (representative, cluster_id) in enumerate(sorted_representatives_with_ids, 1):
        base_name = os.path.splitext(representative['filename'])[0]
        
        # For umotif output, always use clean umotif_## naming regardless of input name
        if output_prefix == 'umotif':
            # Clean naming: umotif_01.xyz (the source is recorded in the combined XYZ comment)
            motif_filename = f"{output_prefix}_{motif_idx:02d}.xyz"
        # Check if base_name already has a motif number (for motif output from non-motif input)
        elif base_name.lower().startswith("motif_"):
            # Extract the motif number from base_name (e.g., "motif_01_opt" -> 1)
            match = re.match(r"motif_(\d+)", base_name, re.IGNORECASE)
            if match:
                original_motif_num = int(match.group(1))
                if original_motif_num == motif_idx:
                    # Energy rank matches original motif number, no duplication needed
                    motif_filename = f"{base_name}.xyz"
                else:
                    # Energy rank differs, show both to indicate reordering
                    motif_filename = f"{output_prefix}_{motif_idx:02d}_{base_name}.xyz"
            else:
                # Couldn't parse motif number, use full format
                motif_filename = f"{output_prefix}_{motif_idx:02d}_{base_name}.xyz"
        else:
            # Doesn't start with motif_, use full format
            motif_filename = f"{output_prefix}_{motif_idx:02d}_{base_name}.xyz"
        
        motif_path = os.path.join(motifs_dir, motif_filename)
        
        write_xyz_file(representative, motif_path)
        
        display_prefix = output_prefix.upper() if output_prefix == 'umotif' else 'Motif'
        if dataset_has_freq:
            gibbs_str = f"{representative['gibbs_free_energy']:.6f}" if representative.get('gibbs_free_energy') is not None else "N/A"
            vprint(f"  {display_prefix} {motif_idx:02d}: {base_name} (Gibbs Energy: {gibbs_str} Hartree, Cluster ID: {cluster_id})")
        else:
            elec_str = f"{representative['final_electronic_energy']:.6f}" if representative.get('final_electronic_energy') is not None else "N/A"
            vprint(f"  {display_prefix} {motif_idx:02d}: {base_name} (Electronic Energy: {elec_str} Hartree, Cluster ID: {cluster_id})")
    

    # Use appropriate filename based on prefix
    combined_xyz_filename = f"all_{folder_prefix}_combined.xyz"
    combined_xyz_path = os.path.join(motifs_dir, combined_xyz_filename)
    
    with open(combined_xyz_path, "w", newline='\n') as outfile:
        for motif_idx, (rep_data, cluster_id) in enumerate(sorted_representatives_with_ids, 1):
            atomnos = rep_data.get('final_geometry_atomnos')
            atomcoords = rep_data.get('final_geometry_coords')

            if atomnos is None or atomcoords is None or len(atomnos) == 0:
                print(f"    WARNING: Skipping representative {rep_data['filename']} due to missing geometry data.")
                continue

            base_name = os.path.splitext(rep_data['filename'])[0]
            if dataset_has_freq:
                gibbs_free_energy = rep_data.get('gibbs_free_energy')
                gibbs_str = f"{gibbs_free_energy:.6f} Hartree ({hartree_to_kcal_mol(gibbs_free_energy):.2f} kcal/mol, {hartree_to_ev(gibbs_free_energy):.2f} eV)" if gibbs_free_energy is not None else "N/A"
                energy_comment = f"G = {gibbs_str}"
            else:
                electronic_energy = rep_data.get('final_electronic_energy')
                elec_str = f"{electronic_energy:.6f} Hartree ({hartree_to_kcal_mol(electronic_energy):.2f} kcal/mol, {hartree_to_ev(electronic_energy):.2f} eV)" if electronic_energy is not None else "N/A"
                energy_comment = f"E = {elec_str}"
            # Use the output_prefix for naming, include source info for umotif
            if output_prefix == 'umotif':
                # For umotifs, include the source motif name in the comment for traceability
                motif_name = f"{output_prefix}_{motif_idx:02d}"
                comment_line = f"{motif_name} (from {base_name}, {energy_comment})"
            else:
                motif_name = f"{output_prefix}_{motif_idx:02d}_{base_name}"
                comment_line = f"{motif_name} ({energy_comment})"
            
            outfile.write(f"{len(atomnos)}\n")
            outfile.write(f"{comment_line}\n")
            for i in range(len(atomnos)):
                symbol = atomic_number_to_symbol(atomnos[i])
                outfile.write(f"{symbol:<2} {atomcoords[i][0]:10.6f} {atomcoords[i][1]:10.6f} {atomcoords[i][2]:10.6f}\n")
    
    # Return the mapping from motif number to cluster ID for use in Boltzmann analysis
    motif_to_cluster_mapping = {}
    for motif_idx, (rep_data, cluster_id) in enumerate(sorted_representatives_with_ids, 1):
        motif_to_cluster_mapping[motif_idx] = cluster_id
    
    vprint(f"  Created combined XYZ file: {os.path.basename(combined_xyz_path)}")
    
    # Attempt to create MOL file using OpenBabel  
    mol_filename = f"all_{folder_prefix}_combined.mol"
    mol_output_path = os.path.join(motifs_dir, mol_filename)
    openbabel_full_path = shutil.which(openbabel_alias)
    
    if openbabel_full_path:
        try:
            # Use the correct OpenBabel syntax: obabel -i<format> input_file -o<format> -O output_file
            result = subprocess.run([openbabel_alias, "-ixyz", combined_xyz_path, "-omol", "-O", mol_output_path], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                vprint(f"  Successfully created MOL file: {os.path.basename(mol_output_path)}")
            else:
                print(f"  WARNING: OpenBabel conversion to MOL failed. Error: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"  WARNING: OpenBabel conversion to MOL timed out after 30 seconds.")
        except Exception as e:
            print(f"  WARNING: Error during OpenBabel conversion to MOL: {e}")
    else:
        print(f"  WARNING: OpenBabel ({openbabel_alias}) not found. Skipping MOL conversion.")
        print("  Please ensure OpenBabel is installed and added to your system's PATH.")
    

    try:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        if len(sorted_representatives_with_ids) > 1:
            # Get representatives data for clustering using complete feature set (same as main clustering)
            representatives_data = []
            motif_labels = []

            _freq_dep = {'gibbs_free_energy', 'first_vib_freq', 'last_vib_freq'}
            _all_num_features = [
                'electronic_energy', 'gibbs_free_energy', 'homo_energy', 'lumo_energy',
                'radius_of_gyration', 'dipole_moment', 'homo_lumo_gap',
                'first_vib_freq', 'last_vib_freq', 'average_hbond_distance',
                'average_hbond_angle'
            ]
            all_potential_numerical_features = _all_num_features if dataset_has_freq else [f for f in _all_num_features if f not in _freq_dep]
            rotational_constant_subfeatures = ROTATIONAL_CONSTANT_SUBFEATURES
            
            # Check which features are globally available across all representatives
            globally_missing_features = []
            for feature in all_potential_numerical_features:
                internal_key = FEATURE_MAPPING.get(feature, feature)
                if all(d.get(internal_key) is None for d in [rep_data for rep_data, _ in sorted_representatives_with_ids]):
                    globally_missing_features.append(feature)
            
            # Check rotational constants availability
            is_rot_const_globally_missing = True
            for rep_data, _ in sorted_representatives_with_ids:
                rot_consts = rep_data.get('rotational_constants')
                if rot_consts is not None and isinstance(rot_consts, np.ndarray) and rot_consts.ndim == 1 and len(rot_consts) == 3:
                    is_rot_const_globally_missing = False
                    break
            
            if is_rot_const_globally_missing:
                globally_missing_features.extend(rotational_constant_subfeatures)
            
            active_features = [f for f in all_potential_numerical_features if f not in globally_missing_features]
            
            for motif_idx, (rep_data, motif_id) in enumerate(sorted_representatives_with_ids, 1):
                # Build feature vector using same logic as main clustering
                feature_vector = []
                
                # Add standard numerical features
                for feature_name in active_features:
                    value = rep_data.get(feature_name)
                    if value is None:
                        value = 0.0
                    feature_vector.append(value)
                
                # Add rotational constants if available
                if not is_rot_const_globally_missing:
                    rot_consts = rep_data.get('rotational_constants')
                    if rot_consts is not None and isinstance(rot_consts, np.ndarray) and len(rot_consts) == 3:
                        feature_vector.extend([rot_consts[0], rot_consts[1], rot_consts[2]])
                    else:
                        feature_vector.extend([0.0, 0.0, 0.0])
                
                if feature_vector:
                    representatives_data.append(feature_vector)
                    # Use just the motif number (e.g., "01" instead of "motif_01")
                    motif_labels.append(f"{motif_idx:02d}")
            
            if len(representatives_data) > 1:

                linkage_matrix = linkage(representatives_data, method='average', metric='euclidean')
                

                plt.figure(figsize=(12, 8))
                dendrogram(linkage_matrix, labels=motif_labels, orientation='top', 
                          distance_sort=True, show_leaf_counts=True)
                # Use appropriate title based on prefix
                dendrogram_title = 'Unique Motifs (umotifs)' if output_prefix == 'umotif' else 'Motifs'
                plt.title(f'{dendrogram_title} Dendrogram')
                plt.xlabel(dendrogram_title)
                plt.ylabel('Distance')
                plt.xticks(rotation=0)  # Keep horizontal since labels are short
                plt.tight_layout()
                
                # Save dendrogram in the motifs directory
                dendrogram_filename = f"{folder_prefix}_dendrogram.png"
                dendrogram_path = os.path.join(motifs_dir, dendrogram_filename)
                plt.savefig(dendrogram_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  Created {folder_prefix} dendrogram: {os.path.basename(dendrogram_path)}")
        
    except ImportError:
        print(f"  WARNING: matplotlib not available. Skipping {folder_prefix} dendrogram creation.")
    except Exception as e:
        print(f"  WARNING: Error creating {folder_prefix} dendrogram: {e}")
    
    display_name = "Unique motifs" if output_prefix == 'umotif' else "Motifs"
    print_step(f"{display_name} created: {len(sorted_representatives_with_ids)} representatives saved to {os.path.basename(motifs_dir)}\n")
    
    return motif_to_cluster_mapping


def combine_xyz_files(cluster_members_data, input_dir, output_base_name=None, openbabel_alias="obabel", prefix_template=None, motif_numbers=None):
    """
    Combines relevant .xyz data from cluster members into a single multi-frame .xyz file
    and attempts to convert the resulting file (or the single original .xyz) to a .mol file.
    Each frame in the combined XYZ will include Gibbs Free Energy in its comment line.
    The frames in the combined XYZ will be sorted by Gibbs free energy (lowest to highest).
    
    Args:
        prefix_template (str): Optional template for comment line prefix, e.g., "Motif_{:02d}_" for motifs
        motif_numbers (list): Optional list of motif numbers corresponding to each member (for unique motifs)
    """
    final_xyz_source_path = None # This will be the path to the XYZ file used for MOL conversion
    
    if not cluster_members_data:
        return

    if len(cluster_members_data) == 1:
        # For a single configuration, the XYZ file has already been written by write_xyz_file.
        # We just need to ensure the MOL conversion uses that file and its original name.
        single_mol_data = cluster_members_data[0]
        original_filename_base = os.path.splitext(single_mol_data['filename'])[0]
        final_xyz_source_path = os.path.join(input_dir, f"{original_filename_base}.xyz")
        # The output_base_name for MOL should be the original filename base
        final_output_mol_name_base = original_filename_base
        vprint(f"  Single configuration found in cluster. Using existing '{os.path.basename(final_xyz_source_path)}' for .mol conversion.")
        
    else:
        # For multiple configurations, create a new combined multi-frame XYZ file.
        if output_base_name is None:

            output_base_name = "combined_cluster"

        full_combined_xyz_path = os.path.join(input_dir, f"{output_base_name}.xyz")
        final_output_mol_name_base = output_base_name # Base name for the .mol file

        # Sort members by Gibbs free energy (lowest to highest), with filename as a tie-breaker
        # Also sort the motif_numbers list in the same order if provided
        if motif_numbers and len(motif_numbers) == len(cluster_members_data):

            paired_data = list(zip(cluster_members_data, motif_numbers))
            sorted_pairs = sorted(
                paired_data,
                key=lambda x: (_sorting_energy(x[0]), x[0]['filename'])
            )
            sorted_members_data = [pair[0] for pair in sorted_pairs]
            sorted_motif_numbers = [pair[1] for pair in sorted_pairs]
        else:
            sorted_members_data = sorted(
                cluster_members_data,
                key=lambda x: (_sorting_energy(x), x['filename'])
            )
            sorted_motif_numbers = None

        with open(full_combined_xyz_path, "w", newline='\n') as outfile:
            for frame_idx, mol_data in enumerate(sorted_members_data, 1): # Iterate over sorted data
                atomnos = mol_data.get('final_geometry_atomnos')
                atomcoords = mol_data.get('final_geometry_coords')
                if atomnos is None or atomcoords is None or len(atomnos) == 0:
                    print(f"    WARNING: Skipping {mol_data['filename']} in combined XYZ due to missing geometry data.")
                    continue

                base_name_for_frame = os.path.splitext(mol_data['filename'])[0]
                if _DATASET_HAS_FREQ:
                    gibbs_free_energy = mol_data.get('gibbs_free_energy')
                    gibbs_str = f"{gibbs_free_energy:.6f} Hartree ({hartree_to_kcal_mol(gibbs_free_energy):.2f} kcal/mol, {hartree_to_ev(gibbs_free_energy):.2f} eV)" if gibbs_free_energy is not None else "N/A"
                    energy_comment = f"G = {gibbs_str}"
                else:
                    electronic_energy = mol_data.get('final_electronic_energy')
                    elec_str = f"{electronic_energy:.6f} Hartree ({hartree_to_kcal_mol(electronic_energy):.2f} kcal/mol, {hartree_to_ev(electronic_energy):.2f} eV)" if electronic_energy is not None else "N/A"
                    energy_comment = f"E = {elec_str}"

                # Apply prefix template with actual motif number if provided
                if prefix_template and sorted_motif_numbers:
                    motif_num = sorted_motif_numbers[frame_idx - 1]  # frame_idx starts at 1
                    comment_line = f"{prefix_template.format(motif_num)}{base_name_for_frame} ({energy_comment})"
                elif prefix_template:
                    comment_line = f"{prefix_template.format(frame_idx)}{base_name_for_frame} ({energy_comment})"
                else:
                    comment_line = f"{base_name_for_frame} ({energy_comment})"

                outfile.write(f"{len(atomnos)}\n")
                outfile.write(f"{comment_line}\n")
                for i in range(len(atomnos)):
                    symbol = atomic_number_to_symbol(atomnos[i])
                    outfile.write(f"{symbol:<2} {atomcoords[i][0]:10.6f} {atomcoords[i][1]:10.6f} {atomcoords[i][2]:10.6f}\n")
        
        vprint(f"  Successfully created combined multi-frame .xyz file: '{os.path.basename(full_combined_xyz_path)}'")
        final_xyz_source_path = full_combined_xyz_path

    # Section for Open Babel Integration (always attempts for MOL conversion)
    if final_xyz_source_path:
        mol_output_filename = f"{final_output_mol_name_base}.mol"
        full_mol_output_path = os.path.join(input_dir, mol_output_filename)

        openbabel_full_path = shutil.which(openbabel_alias)
        openbabel_installed = False

        if openbabel_full_path:
            openbabel_installed = True
        else:
            print(f"\n  Open Babel ({openbabel_alias}) command not found or not executable. Skipping .mol conversion.")
            print("  Please ensure Open Babel is installed and added to your system's PATH, or provide the correct alias.")
            print(f"  You can change the alias using the 'openbabel_alias' parameter in the function call, e.g., combine_xyz_files(..., openbabel_alias='obabel').")

        if openbabel_installed:
            try:
                conversion_command = [openbabel_full_path, "-i", "xyz", final_xyz_source_path, "-o", "mol", "-O", full_mol_output_path]
                subprocess.run(conversion_command, check=True, capture_output=True, text=True)

                if os.path.exists(full_mol_output_path):
                    vprint(f"  Successfully converted '{os.path.basename(final_xyz_source_path)}' to '{os.path.basename(full_mol_output_path)}' using Open Babel.")

            except subprocess.CalledProcessError as e:
                print(f"  Open Babel conversion failed for '{os.path.basename(final_xyz_source_path)}'.")
                print(f"  Error details: {e.stderr.strip()}")
            except Exception as e:
                print(f"  An unexpected error occurred during Open Babel conversion for '{final_xyz_source_path}': {e}")


# New: Feature mapping and weight parsing
FEATURE_MAPPING = {
    "electronic_energy": "final_electronic_energy",
    "gibbs_free_energy": "gibbs_free_energy",
    "homo_energy": "homo_energy",
    "lumo_energy": "lumo_energy",
    "homo_lumo_gap": "homo_lumo_gap",
    "dipole_moment": "dipole_moment",
    "radius_of_gyration": "radius_of_gyration",
    "rotational_constants_A": "rotational_constants_0", # Special handling for array elements
    "rotational_constants_B": "rotational_constants_1",
    "rotational_constants_C": "rotational_constants_2",
    "first_vib_freq": "first_vib_freq",
    "last_vib_freq": "last_vib_freq",
    "average_hbond_distance": "average_hbond_distance",
    "average_hbond_angle": "average_hbond_angle",
    "num_hydrogen_bonds": "num_hydrogen_bonds" # Although not a numerical feature for clustering, good to map
}

def parse_weights_argument(weight_str):
    """
    Parses the --weights argument string into a dictionary of feature_name: weight.
    Example: "(electronic_energy=0.1)(homo_lumo_gap=0.2)"
    """
    weights = {}
    if not weight_str:
        return weights

    # Regex to find (key=value) pairs
    matches = re.findall(r'\(([^=]+)=([\d.]+)\)', weight_str)
    for key, value in matches:
        try:
            weights[key.strip()] = float(value.strip())
        except ValueError:
            print(f"WARNING: Could not parse weight for '{key}={value}'. Skipping this weight.")
    return weights

def parse_abs_tolerance_argument(tolerance_str):
    """
    Parses the --abs-tolerance argument string into a dictionary of feature_name: tolerance.
    Example: "(electronic_energy=1e-5)(dipole_moment=1e-3)"
    """
    tolerances = {}
    if not tolerance_str:
        return tolerances
    
    matches = re.findall(r'\(([^=]+)=([\d\.eE-]+)\)', tolerance_str)
    for key, value in matches:
        try:
            tolerances[key.strip()] = float(value.strip())
        except ValueError:
            print(f"WARNING: Could not parse absolute tolerance for '{key}={value}'. Skipping this tolerance.")
    return tolerances


CLUSTERING_NUMERICAL_FEATURES = [
    'electronic_energy',
    'gibbs_free_energy',
    'homo_energy',
    'lumo_energy',
    'homo_lumo_gap',
    'dipole_moment',
    'radius_of_gyration',
    'first_vib_freq',
    'last_vib_freq',
    'average_hbond_distance',
    'average_hbond_angle'
]

ROTATIONAL_CONSTANT_SUBFEATURES = [
    'rotational_constants_A',
    'rotational_constants_B',
    'rotational_constants_C'
]


def is_valid_scalar(value):
    if value is None:
        return False
    if isinstance(value, (float, np.floating)):
        return np.isfinite(value)
    return True


def group_has_any_clustering_feature_data(group_data):
    scalar_internal_keys = [FEATURE_MAPPING[feature_name] for feature_name in CLUSTERING_NUMERICAL_FEATURES]

    has_scalar_feature = any(
        is_valid_scalar(molecule_data.get(feature_key))
        for molecule_data in group_data
        for feature_key in scalar_internal_keys
    )
    has_rotational_constants = any(
        molecule_data.get('rotational_constants') is not None
        and isinstance(molecule_data.get('rotational_constants'), np.ndarray)
        and molecule_data.get('rotational_constants').ndim == 1
        and len(molecule_data.get('rotational_constants')) == 3
        for molecule_data in group_data
    )
    return has_scalar_feature or has_rotational_constants


def has_valid_rotational_constants(molecule_data):
    rot_consts = molecule_data.get('rotational_constants')
    return (
        rot_consts is not None
        and isinstance(rot_consts, np.ndarray)
        and rot_consts.ndim == 1
        and len(rot_consts) == 3
    )


def select_complete_group_scalar_features(group_data, candidate_features):
    active_features = []
    dropped_features = []
    for feature_name in candidate_features:
        internal_key = FEATURE_MAPPING.get(feature_name, feature_name)
        if all(is_valid_scalar(molecule_data.get(internal_key)) for molecule_data in group_data):
            active_features.append(feature_name)
        else:
            dropped_features.append(feature_name)
    return active_features, dropped_features


def _build_feature_vectors(mols, scalar_features, use_rotconsts, weights):
    """Build raw feature vectors for a list of molecules using given features.
    Returns (vectors_list, ordered_feature_names)."""
    vectors = []
    feature_names = []
    for mol in mols:
        vec = []
        names = []
        for feat in scalar_features:
            if weights.get(feat, 1.0) != 0.0:
                key = FEATURE_MAPPING.get(feat, feat)
                vec.append(mol.get(key))
                names.append(feat)
        if use_rotconsts:
            rc = mol.get('rotational_constants')
            for i, rc_name in enumerate(['rotational_constants_A', 'rotational_constants_B', 'rotational_constants_C']):
                if weights.get(rc_name, 1.0) != 0.0:
                    vec.append(rc[i] if rc is not None and len(rc) > i else None)
                    names.append(rc_name)
        vectors.append(vec)
        if not feature_names:
            feature_names = names
    return vectors, feature_names


def _zscore_scale(raw_np, feature_names, min_std_threshold, abs_tolerances):
    """Z-score standardize a feature matrix column-wise.  Returns scaled array."""
    from sklearn.preprocessing import StandardScaler
    scaled = np.zeros_like(raw_np)
    for col in range(raw_np.shape[1]):
        col_data = raw_np[:, col]
        fname = feature_names[col]
        max_diff = np.max(col_data) - np.min(col_data)
        if fname in abs_tolerances and max_diff < abs_tolerances[fname]:
            scaled[:, col] = 0.0
        else:
            std = np.std(col_data)
            if std < min_std_threshold:
                scaled[:, col] = 0.0
            else:
                scaler = StandardScaler()
                scaled[:, col] = scaler.fit_transform(col_data.reshape(-1, 1)).flatten()
    return scaled


def _match_reduced_to_clusters(
    reduced_mols, fullest_mols, cluster_labels_fullest,
    primary_scalar_features, use_primary_rotconsts,
    weights, threshold, min_std_threshold, abs_tolerances
):
    """Match reduced-vector structures against existing clusters from the fullest tier.

    For each group of reduced structures sharing the same available features,
    builds a combined Z-score–standardized matrix (fullest tier + reduced) using
    only the shared features, then checks if each reduced structure falls within
    *threshold* distance of a cluster representative.

    Returns (matched_by_label, unmatched):
        matched_by_label: dict {cluster_label: [mol, ...]}
        unmatched: list of mols that didn't match any cluster
    """
    from collections import defaultdict

    if not reduced_mols:
        return {}, []

    # Build cluster → members mapping and pick representatives
    clusters = defaultdict(list)
    for mol, lbl in zip(fullest_mols, cluster_labels_fullest):
        clusters[lbl].append(mol)
    representatives = {}
    for lbl, members in clusters.items():
        representatives[lbl] = min(members, key=lambda x: (_sorting_energy(x), x['filename']))

    # Index fullest mols for quick lookup
    fullest_idx = {id(mol): i for i, mol in enumerate(fullest_mols)}

    # Group reduced mols by their available feature set
    reduced_by_features = defaultdict(list)
    for mol in reduced_mols:
        key = frozenset(mol['_available_features'])
        reduced_by_features[key].append(mol)

    matched_by_label = defaultdict(list)
    unmatched = []

    all_primary_features = set(primary_scalar_features)
    if use_primary_rotconsts:
        all_primary_features.update(['rotational_constants_A', 'rotational_constants_B', 'rotational_constants_C'])

    for feat_set, tier_mols in reduced_by_features.items():
        # Shared scalar features (preserving order)
        shared_scalar = [f for f in primary_scalar_features if f in feat_set]
        use_shared_rot = (
            use_primary_rotconsts
            and 'rotational_constants_A' in feat_set
            and 'rotational_constants_B' in feat_set
            and 'rotational_constants_C' in feat_set
        )

        if not shared_scalar and not use_shared_rot:
            unmatched.extend(tier_mols)
            continue

        # Build combined matrix: fullest + reduced tier
        combined = list(fullest_mols) + tier_mols
        n_fullest = len(fullest_mols)

        raw_vecs, feat_names = _build_feature_vectors(
            combined, shared_scalar, use_shared_rot, weights
        )
        if not feat_names:
            unmatched.extend(tier_mols)
            continue

        raw_np = np.array(raw_vecs, dtype=float)
        scaled = _zscore_scale(raw_np, feat_names, min_std_threshold, abs_tolerances)

        # Match each reduced structure to nearest representative
        for idx, mol in enumerate(tier_mols):
            mol_vec = scaled[n_fullest + idx]
            best_dist = float('inf')
            best_label = None
            for lbl, rep in representatives.items():
                rep_pos = fullest_idx[id(rep)]
                dist = np.linalg.norm(mol_vec - scaled[rep_pos])
                if dist < best_dist:
                    best_dist = dist
                    best_label = lbl
            if best_dist <= threshold:
                matched_by_label[best_label].append(mol)
            else:
                unmatched.append(mol)

    return dict(matched_by_label), unmatched


def extract_homo_lumo_from_orca_text(lines):
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

def preprocess_j_argument(argv):
    """
    Preprocesses command line arguments:
    - Handle -j8 format (no space) by converting it to -j 8.
    - Extract boolean flags (--verbose, etc.) that appear after --compare
      so they are not consumed as file arguments by nargs='+'.
    """
    # First pass: extract standalone boolean flags that could be trapped by --compare nargs='+'
    _bool_flags = {'-v', '--verbose', '-r', '--reprocess-files', '-V', '--version'}
    extracted_flags = []
    remaining = []
    for arg in argv:
        if arg in _bool_flags:
            extracted_flags.append(arg)
        else:
            remaining.append(arg)

    # Second pass: handle -j8 → -j 8
    processed_argv = []
    for arg in remaining:
        if arg.startswith('-j') and len(arg) > 2 and arg[2:].isdigit():
            processed_argv.extend(['-j', arg[2:]])
        else:
            processed_argv.append(arg)

    # Put boolean flags at the front so they are parsed before --compare
    return extracted_flags + processed_argv



def compute_mojena_threshold(linkage_matrix, verbose=False):
    """
    Compute Mojena's stopping rule threshold (robust variant) for diagnostic purposes.

    Uses the robust formulation: median(h) + alpha * 1.4826 * MAD(h)
    where MAD = median(|h - median(h)|) and 1.4826 is the consistency constant
    for normal distributions (Mojena, 1977).

    Parameters
    ----------
    linkage_matrix : np.ndarray, shape (n_samples-1, 4)
        Linkage matrix from scipy.cluster.hierarchy.linkage (UPGMA).
    verbose : bool
        If True, print diagnostic details to stdout.

    Returns
    -------
    mojena_threshold : float
        The Mojena-recommended threshold (for diagnostic comparison).
    mojena_k : int
        Number of clusters at the Mojena threshold.
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

    # Clamp within merge height range
    mojena_t = max(float(heights[0]) * 1.01, min(mojena_t, float(heights[-1]) * 0.99))

    labels = fcluster(linkage_matrix, t=mojena_t, criterion='distance')
    mojena_k = len(set(labels))

    if verbose:
        print(f"  Mojena diagnostic (robust, alpha={MOJENA_ALPHA}):")
        print(f"    median={median_h:.4f}, MAD={mad_h:.4f}")
        print(f"    Mojena threshold = {mojena_t:.4f} (k={mojena_k})")

    return mojena_t, mojena_k


def plot_annotated_dendrogram(linkage_matrix, optimal_k, cut_height,
                               filename, title_suffix="", conf_labels=None,
                               mojena_threshold=None, mojena_k=None):
    """
    Save two plot files:
      1. Dendrogram with horizontal cut line → filename (e.g., dendrogram.png)
      2. Mojena diagnostic → threshold_diagnostic.png in the same directory

    Parameters
    ----------
    linkage_matrix : np.ndarray
    optimal_k : int
    cut_height : float
    filename : str
        Output path for the dendrogram PNG.
    title_suffix : str
    conf_labels : list of str or None
    mojena_threshold : float or None
        Mojena's recommended threshold for diagnostic comparison.
    mojena_k : int or None
        Number of clusters at the Mojena threshold.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    # Check if all distances are effectively zero
    lm = linkage_matrix.copy()
    if np.all(lm[:, 2] == 0.0):
        lm[:, 2] += 1e-12

    # --- File 1: Dendrogram ---
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    dendrogram(lm, labels=conf_labels, leaf_rotation=90, leaf_font_size=8, ax=ax1)

    cut_label = f'Threshold t={cut_height:.2f} (k={optimal_k})'
    ax1.axhline(y=cut_height, color='#e74c3c', linestyle='--', linewidth=1.5,
                label=cut_label)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_title(f"Hierarchical Clustering Dendrogram ({title_suffix})")
    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("UPGMA Distance (Z-standardized)")
    ax1.set_ylim(bottom=0)
    fig1.tight_layout()
    fig1.savefig(filename, dpi=150)
    plt.close(fig1)

    # --- File 2: Mojena diagnostic plot ---
    heights = linkage_matrix[:, 2]
    n_merges = len(heights)
    if n_merges < 3:
        return

    diag_filename = os.path.join(os.path.dirname(filename), "threshold_diagnostic.png")

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

    # Plot merge heights as a step function (sorted, indexed by merge step)
    merge_indices = np.arange(1, n_merges + 1)
    ax2.fill_between(merge_indices, 0, heights, alpha=0.15, color='#3498db')
    ax2.plot(merge_indices, heights, '-', color='#3498db', linewidth=1.2,
             label='Merge heights')

    # User's threshold line
    n_above_user = int(np.sum(heights > cut_height))
    ax2.axhline(y=cut_height, color='#e74c3c', linestyle='--', linewidth=2,
                label=f'Threshold t={cut_height:.2f} (k={n_above_user + 1})')

    # Mojena's threshold line (diagnostic)
    if mojena_threshold is not None and mojena_k is not None:
        ax2.axhline(y=mojena_threshold, color='#9b59b6', linestyle=':', linewidth=1.5,
                    label=f'Mojena t={mojena_threshold:.2f} (k={mojena_k})')

    # Shade between-cluster region (above user threshold)
    ax2.fill_between(merge_indices, cut_height, heights,
                     where=(heights > cut_height),
                     alpha=0.2, color='#e74c3c', label='Between-cluster merges')

    ax2.set_xlabel("Merge Step (sorted)")
    ax2.set_ylabel("UPGMA Merge Height")
    ax2.set_title("Threshold Diagnostic (Merge Height Distribution)")
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(diag_filename, dpi=150)
    plt.close(fig2)


# Modified to accept rmsd_threshold and output_base_dir
def perform_clustering_and_analysis(input_source, threshold=2.0, file_extension_pattern=None, rmsd_threshold=None, output_base_dir=None, force_reprocess_cache=False, weights=None, is_compare_mode=False, min_std_threshold=1e-6, abs_tolerances=None, num_cores=None, temperature_k=298.15, group_hb=False):
    """
    Performs hierarchical clustering and comprehensive analysis on molecular structures.
    This is the main analysis function that orchestrates the entire clustering workflow.
    """
    from sklearn.preprocessing import StandardScaler  # type: ignore # Import only when needed
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster  # type: ignore # Import only when needed
    import matplotlib.pyplot as plt  # type: ignore # Import only when needed
    

    if num_cores is None:
        num_cores = get_cpu_count_fast()
    """
    Performs hierarchical clustering and analysis on the extracted molecular properties,
    and saves .dat and .xyz files for each cluster.
    Includes an optional RMSD post-processing step and caching of extracted data.
    Output files will be saved relative to output_base_dir.
    `input_source` can be a folder path (normal mode) or a list of file paths (compare mode).
    `weights` is a dictionary mapping feature names (user-friendly) to their weights.
    `min_std_threshold` (float): Minimum standard deviation for a feature to be scaled.
                                 Features with std dev below this are treated as constant (0.0).
    `abs_tolerances` (dict): Dictionary of feature_name: absolute_tolerance. If the max difference
                             for a feature within a group is less than its tolerance, it's zeroed out.
    """
    # Default weights for all clustering features
    # These can be adjusted using the --weights flag, e.g., --weights "(first_vib_freq=1.0)(homo_lumo_gap=1.5)"

    # Available features:
    # - electronic_energy: Final electronic energy (Hartree)
    # - gibbs_free_energy: Gibbs free energy (Hartree) 
    # - homo_energy: HOMO energy (eV)
    # - lumo_energy: LUMO energy (eV)
    # - homo_lumo_gap: HOMO-LUMO gap (eV)
    # - dipole_moment: Dipole moment (Debye)
    # - radius_of_gyration: Radius of gyration (Å)
    # - rotational_constants_A/B/C: Rotational constants (GHz)
    # - first_vib_freq: First vibrational frequency (cm⁻¹)
    # - last_vib_freq: Last vibrational frequency (cm⁻¹)
    # - average_hbond_distance: Average hydrogen bond distance (Å)
    # - average_hbond_angle: Average hydrogen bond angle (degrees)
    # - num_hydrogen_bonds: Number of hydrogen bonds (not used for clustering but mapped)
    default_weights = {
        'electronic_energy': 1.0,          # Final electronic energy
        'gibbs_free_energy': 1.0,          # Gibbs free energy
        'homo_energy': 1.0,                # HOMO energy
        'lumo_energy': 1.0,                # LUMO energy
        'homo_lumo_gap': 1.0,              # HOMO-LUMO gap
        'dipole_moment': 1.0,              # Dipole moment
        'radius_of_gyration': 1.0,         # Radius of gyration
        'rotational_constants_A': 1.0,     # Rotational constant A
        'rotational_constants_B': 1.0,     # Rotational constant B
        'rotational_constants_C': 1.0,     # Rotational constant C
        'first_vib_freq': 1.0,             # First vibrational frequency
        'last_vib_freq': 1.0,              # Last vibrational frequency
        'average_hbond_distance': 1.0,     # Average hydrogen bond distance
        'average_hbond_angle': 1.0,        # Average hydrogen bond angle
    }
    
    # Track which features the user explicitly set via --weights
    _user_explicit_weights = set(weights.keys()) if weights else set()

    if weights is None:
        weights = default_weights.copy() # Use default weights if none provided
    else:
        # Merge user weights with defaults, user weights take priority
        merged_weights = default_weights.copy()
        merged_weights.update(weights)
        weights = merged_weights
    
    if abs_tolerances is None:
        abs_tolerances = {} # Ensure abs_tolerances is a dict if not provided

    # Ensure output_base_dir is set, default to current working directory if None
    if output_base_dir is None:
        output_base_dir = os.getcwd()

    # NEW: Adjust output_base_dir for comparison mode and handle unique naming
    if is_compare_mode:
        base_comparison_dir_name = "comparison"
        final_comparison_dir = base_comparison_dir_name
        counter = 0
        while os.path.exists(os.path.join(output_base_dir, final_comparison_dir)):
            counter += 1
            final_comparison_dir = f"{base_comparison_dir_name}_{counter}"
        output_base_dir = os.path.join(output_base_dir, final_comparison_dir)
        os.makedirs(output_base_dir, exist_ok=True) # Ensure this new base directory exists
        print(f"  Comparison mode: All outputs will be placed in '{output_base_dir}'")
    else:
        # For normal mode, output directly to the working directory (no subfolder)
        os.makedirs(output_base_dir, exist_ok=True) # Ensure this base directory exists
        print(f"  All outputs will be placed in the current working directory")

    
    # Provide early feedback before expensive operations
    if not is_compare_mode:
        print_step("Initializing data extraction...")

    # Define a generic cache file path with random seed (like ASCEC protocol cache)
    # Check for existing cache file first
    import glob
    existing_caches = glob.glob(os.path.join(output_base_dir, "data_cache_*.pkl"))
    
    if existing_caches:
        # Use the most recent cache file if multiple exist
        cache_file_path = max(existing_caches, key=os.path.getmtime)
        vprint(f"Found existing cache file: {os.path.basename(cache_file_path)}")
    else:
        # Create new cache file with random seed
        import random
        cache_seed = random.randint(0, 999999)
        cache_file_name = f"data_cache_{cache_seed:06d}.pkl"
        cache_file_path = os.path.join(output_base_dir, cache_file_name)

    all_extracted_data = []
    skipped_files = set()  # Initialize skipped_files early to avoid unbound variable warnings
    
    files_to_process = []
    if is_compare_mode:
        files_to_process = input_source # input_source is already the list of files
        # For compare mode, we should probably bypass cache loading/saving, or make it specific to the comparison.
        # For now, let's assume compare mode always re-processes the two files.
        print(f"Starting parallel data extraction for comparison mode from {len(files_to_process)} files...")
        
        # Use parallel processing for comparison mode
        effective_cores = num_cores  # Use all available cores
        print(f"  Using {effective_cores} CPU cores for parallel processing")
        
        with mp.Pool(processes=effective_cores) as pool:
            results = pool.map(process_file_parallel_wrapper, sorted(files_to_process))
        
        # Process results
        for success, extracted_props, filename in results:
            if success and extracted_props:
                all_extracted_data.append(extracted_props)
            elif not success:
                skipped_files.add(filename)
    else:

        # INCREMENTAL CACHE UPDATE MODE (for redo)
        update_cache_file = None
        incremental_update_done = False  # Flag to track if update happened
        
        if len(sys.argv) > 1:  # Check if there are arguments
            for i, arg in enumerate(sys.argv):
                if arg == '--update-cache' and i + 1 < len(sys.argv):
                    update_cache_file = sys.argv[i + 1]
                    break
        
        if update_cache_file and os.path.exists(update_cache_file):
            # Read list of files to update
            with open(update_cache_file, 'r') as f:
                basenames_to_update = {line.strip() for line in f if line.strip()}
            
            # Determine file extension (check if .out or .log exists for first basename)
            file_ext = None
            if basenames_to_update:
                first_basename = next(iter(basenames_to_update))
                if os.path.exists(os.path.join(str(input_source), first_basename + '.out')):
                    file_ext = '.out'
                elif os.path.exists(os.path.join(str(input_source), first_basename + '.log')):
                    file_ext = '.log'
            
            if file_ext:                
                # Load existing cache
                cache_exists = os.path.exists(cache_file_path)
                if cache_exists:
                    with open(cache_file_path, 'rb') as f:
                        cached_data = pickle.load(f)
                    if isinstance(cached_data, list):
                        successful_data = cached_data
                        skipped_files = set()
                    else:
                        successful_data = cached_data.get('successful', [])
                        skipped_files = set(cached_data.get('skipped', []))
                else:
                    successful_data = []
                    skipped_files = set()
                
                # Remove old data for files being updated
                filenames_to_update = {b + file_ext for b in basenames_to_update}
                successful_data = [d for d in successful_data if d['filename'] not in filenames_to_update]
                skipped_files = skipped_files - filenames_to_update
                
                # Reprocess only the specified files
                files_to_reprocess = [os.path.join(str(input_source), f) for f in filenames_to_update 
                                     if os.path.exists(os.path.join(str(input_source), f))]
                
                effective_cores = min(num_cores, len(files_to_reprocess)) if len(files_to_reprocess) > 0 else 1
                
                with mp.Pool(processes=effective_cores) as pool:
                    results = pool.map(process_file_parallel_wrapper, sorted(files_to_reprocess))
                
                # Update with new data
                for success, extracted_props, filename in results:
                    if success and extracted_props:
                        successful_data.append(extracted_props)
                    elif not success:
                        skipped_files.add(filename)
                
                # Save updated cache
                cache_data_to_save = {
                    'successful': successful_data,
                    'skipped': list(skipped_files)
                }
                with open(cache_file_path, 'wb') as f:
                    pickle.dump(cache_data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                all_extracted_data = successful_data
                incremental_update_done = True
                print(f"    Cache updated with {len(results)} file(s)")
            else:
                print(f"  Warning: Could not determine file extension for incremental update")
                force_reprocess_cache = True  # Fallback to full reprocess
        
        # Existing cache logic for normal mode (skip if incremental update was done)
        if not incremental_update_done:
            if os.path.exists(cache_file_path) and not force_reprocess_cache:
                print(f"Attempting to load data from cache: '{os.path.basename(cache_file_path)}'")
                try:
                    with open(cache_file_path, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    # Handle both old format (list) and new format (dict with 'successful' and 'skipped')
                    if isinstance(cached_data, list):
                        # Old format - assume all were successful, no skipped info
                        successful_data = cached_data
                        skipped_files = set()
                    elif isinstance(cached_data, dict) and 'successful' in cached_data:
                        # New format with skipped file tracking
                        successful_data = cached_data['successful']
                        skipped_files = set(cached_data.get('skipped', []))
                    else:
                        # Unknown format
                        raise ValueError("Unknown cache format")
                    
                    # Only scan filesystem if cache has data (lightweight check first)
                    if len(successful_data) > 0 or len(skipped_files) > 0:
                        # Defer expensive glob until we know cache exists and has data
                        current_files_in_folder = {os.path.basename(f) for f in glob.glob(os.path.join(str(input_source), str(file_extension_pattern)))} if input_source and file_extension_pattern else set()
                        retained_cached_data = [d for d in successful_data if d['filename'] in current_files_in_folder]
                        
                        # Files that were processed (either successfully or skipped)
                        processed_files = {d['filename'] for d in successful_data} | skipped_files
                        unprocessed_files = current_files_in_folder - processed_files
                        
                        vprint(f"  Cache contains {len(successful_data)} successful entries and {len(skipped_files)} skipped files")
                        vprint(f"  Current folder has {len(current_files_in_folder)} files")
                        vprint(f"  Unprocessed files: {len(unprocessed_files)}")
                        
                        if len(unprocessed_files) == 0:
                            # All files have been processed before
                            all_extracted_data = retained_cached_data
                            vprint(f"Data loaded from cache successfully. ({len(all_extracted_data)} entries)")
                            print_step("Using cached data")
                        else:
                            # Some files haven't been processed yet
                            if len(retained_cached_data) > 0:
                                vprint(f"  Cache partial: {len(unprocessed_files)} files need processing")
                                all_extracted_data = retained_cached_data.copy()
                            else:
                                vprint("Cache data incomplete or outdated. Re-extracting all files.")
                                all_extracted_data = []
                                skipped_files = set()
                                if os.path.exists(cache_file_path):
                                    os.remove(cache_file_path)
                    else:
                        # Cache is empty, invalidate it
                        vprint("Cache is empty. Re-extracting all files.")
                        all_extracted_data = []
                        skipped_files = set()
                        if os.path.exists(cache_file_path):
                            os.remove(cache_file_path)

                except Exception as e:
                    vprint(f"Error loading data from cache: {e}. Re-extracting all files.")
                    all_extracted_data = []
                    skipped_files = set()
                    if os.path.exists(cache_file_path):
                        os.remove(cache_file_path)
        else:
            # Incremental update was done, so we trust the current state
            # all_extracted_data is already set
            pass

        files_to_process = glob.glob(os.path.join(str(input_source), str(file_extension_pattern))) if input_source and file_extension_pattern else []
        if not files_to_process:
            print(f"No files matching '{file_extension_pattern}' found in '{input_source}'. Skipping this folder.")
            return

        # Determine if we need to process any files
        cached_filenames = {d['filename'] for d in all_extracted_data}
        current_filenames = {os.path.basename(f) for f in files_to_process}
        
        # Files that have been processed (either successfully extracted or skipped)
        processed_files = cached_filenames | skipped_files
        unprocessed_files = current_filenames - processed_files
        
        if len(unprocessed_files) == 0:
            # All files have been processed before
            files_to_actually_process = []
        elif not processed_files:
            # No cached data, process all files
            print_step(f"\nExtracting data from {len(files_to_process)} files...")
            files_to_actually_process = files_to_process
        else:
            # We have some cached data, only process unprocessed files
            print_step(f"\nProcessing {len(unprocessed_files)} new/unprocessed files...")
            files_to_actually_process = [f for f in files_to_process if os.path.basename(f) in unprocessed_files]

        # Process files if needed
        if files_to_actually_process:
            # Use parallel processing for normal mode
            effective_cores = num_cores  # Use all available cores
            print(f"  Using {effective_cores} CPU cores for parallel processing")
            
            with mp.Pool(processes=effective_cores) as pool:
                results = pool.map(process_file_parallel_wrapper, sorted(files_to_actually_process))
            
            # Process results
            for success, extracted_props, filename in results:
                if success and extracted_props:
                    all_extracted_data.append(extracted_props)
                elif not success:
                    # File was skipped (likely due to imaginary frequencies)
                    skipped_files.add(filename)
            
            # Save updated cache with both successful and skipped files
            if not is_compare_mode:
                cache_data = {
                    'successful': all_extracted_data,
                    'skipped': list(skipped_files)
                }
                vprint(f"Updating cache with {len(all_extracted_data)} successful and {len(skipped_files)} skipped entries: '{os.path.basename(cache_file_path)}'")
                try:
                    with open(cache_file_path, 'wb') as f:
                        pickle.dump(cache_data, f)
                    vprint("Cache updated successfully.")
                except Exception as e:
                    vprint(f"  Error updating cache: {e}")

    if not all_extracted_data:
        print_step("No data was successfully extracted from files. Skipping clustering.")
        return

    print_step("Data extraction complete. Proceeding to clustering.\n")
    print() # Add extra blank line for readability
    
    clean_data_for_clustering = []
    essential_base_features = ['final_geometry_atomnos', 'final_geometry_coords', 'num_hydrogen_bonds']
    
    for mol_data in all_extracted_data:
        is_essential_missing = False
        missing_essential_info = []
        for f in essential_base_features:
            if mol_data.get(f) is None:
                is_essential_missing = True
                missing_essential_info.append(f"Missing essential feature '{f}'")
        
        if not is_essential_missing:
            clean_data_for_clustering.append(mol_data)
        else:
            print(f"Skipping '{mol_data.get('filename', 'Unknown')}' for clustering due to: {'; '.join(missing_essential_info)}")


    if not clean_data_for_clustering:
        print(f"No complete data entries to cluster after filtering. Exiting clustering step.")
        return

    # --- Detect whether the dataset has frequency calculations ---
    # If ANY structure has gibbs_free_energy, we consider the dataset as having freq data.
    # Otherwise, we operate in "opt-only" mode with a reduced feature vector.
    _dataset_has_freq = any(
        is_valid_scalar(_mol.get('gibbs_free_energy'))
        for _mol in clean_data_for_clustering
    )

    # Set module-level flag so helper functions (_sorting_energy, write_xyz_file, etc.)
    # automatically use the correct mode without needing explicit parameters.
    global _DATASET_HAS_FREQ
    _DATASET_HAS_FREQ = _dataset_has_freq

    # --- H-bond grouping decision ---
    # By default, all structures go into a single pool and property-based
    # clustering decides the grouping.  H-bond detection is sensitive to small
    # geometric changes — two nearly identical structures can differ by 1-2
    # H-bonds — so pre-grouping by exact H-bond count can split genuinely
    # similar structures into separate families.
    #
    # The --group-hb flag restores the old behavior of pre-grouping by exact
    # H-bond count, which produces separate dendrograms per HB family.  This
    # can be useful for visualization when the user wants to inspect each
    # H-bond family independently.
    if is_compare_mode and len(clean_data_for_clustering) >= 2:
        hbond_groups = {0: sorted(clean_data_for_clustering,
                                  key=lambda x: (_sorting_energy(x), x['filename']))}
        print("  Comparison mode: Running clustering to generate dendrogram, then forcing a single output cluster.")
    elif group_hb:
        # Group by exact H-bond count (separate dendrograms per HB family)
        hbond_groups = {}
        for item in clean_data_for_clustering:
            hbond_groups.setdefault(item['num_hydrogen_bonds'], []).append(item)
        print(f"  H-bond pre-grouping enabled: {len(hbond_groups)} HB families detected")
    else:
        # Default: single pool — let property-based clustering decide
        hbond_groups = {0: sorted(clean_data_for_clustering,
                                  key=lambda x: (_sorting_energy(x), x['filename']))}

    # Output directory paths
    dendrogram_images_folder = os.path.join(output_base_dir, "dendrogram_images")
    extracted_data_folder = os.path.join(output_base_dir, "extracted_data")
    extracted_clusters_folder = os.path.join(output_base_dir, "extracted_clusters")
    

    
    os.makedirs(dendrogram_images_folder, exist_ok=True)
    os.makedirs(extracted_data_folder, exist_ok=True)
    os.makedirs(extracted_clusters_folder, exist_ok=True)

    if VERBOSE:
        print(f"Dendrogram images will be saved to '{dendrogram_images_folder}'")
        print(f"Extracted data files will be saved to '{extracted_data_folder}'")
        print(f"Extracted cluster XYZ/MOL files will be saved to '{extracted_clusters_folder}'")
    else:
        print_step("Setting up output directories...")

    summary_file_content_lines = []
    comparison_specific_summary_lines = [] # New list for comparison-specific details

    total_clusters_outputted = 0
    total_rmsd_outliers_first_pass = 0

    # Helper function to center text within 75 characters
    def center_text(text, width=75):
        return text.center(width)
    
    # Add ASCII art header similar to ASCEC but for COSMIC
    summary_file_content_lines.append("=" * 75)
    summary_file_content_lines.append("")
    summary_file_content_lines.append(center_text("***************************"))
    summary_file_content_lines.append(center_text("*       C O S M I C       *"))
    summary_file_content_lines.append(center_text("***************************"))
    summary_file_content_lines.append("")
    summary_file_content_lines.append("                             √≈≠==≈                                  ")
    summary_file_content_lines.append("   √≈≠==≠≈√   √≈≠==≠≈√         ÷++=                      ≠===≠       ")
    summary_file_content_lines.append("     ÷++÷       ÷++÷           =++=                     ÷×××××=      ")
    summary_file_content_lines.append("     =++=       =++=     ≠===≠ ÷++=      ≠====≠         ÷-÷ ÷-÷      ")
    summary_file_content_lines.append("     =++=       =++=    =××÷=≠=÷++=    ≠÷÷÷==÷÷÷≈      ≠××≠ =××=     ")
    summary_file_content_lines.append("     =++=       =++=   ≠××=    ÷++=   ≠×+×    ×+÷      ÷+×   ×+××    ")
    summary_file_content_lines.append("     =++=       =++=   =+÷     =++=   =+-×÷==÷×-×≠    =×+×÷=÷×+-÷    ")
    summary_file_content_lines.append("     ≠×+÷       ÷+×≠   =+÷     =++=   =+---×××××÷×   ≠××÷==×==÷××≠   ")
    summary_file_content_lines.append("      =××÷     =××=    ≠××=    ÷++÷   ≠×-×           ÷+×       ×+÷   ")
    summary_file_content_lines.append("       ≠=========≠      ≠÷÷÷=≠≠=×+×÷-  ≠======≠≈√  -÷×+×≠     ≠×+×÷- ")
    summary_file_content_lines.append("          ≠===≠           ≠==≠  ≠===≠     ≠===≠    ≈====≈     ≈====≈ ")
    summary_file_content_lines.append("")
    summary_file_content_lines.append("")
    summary_file_content_lines.append(center_text("Universidad de Antioquia - Medellín - Colombia"))
    summary_file_content_lines.append("")
    summary_file_content_lines.append("")
    summary_file_content_lines.append(center_text("Clustering Analysis for Quantum Chemistry Calculations"))
    summary_file_content_lines.append("")
    summary_file_content_lines.append(center_text(version))
    summary_file_content_lines.append("")
    summary_file_content_lines.append("")
    summary_file_content_lines.append(center_text("Química Física Teórica - QFT"))
    summary_file_content_lines.append("")
    summary_file_content_lines.append("")
    summary_file_content_lines.append("=" * 75 + "\n")
    if is_compare_mode:
        summary_file_content_lines.append(f"Comparison Results for: {', '.join([os.path.basename(f) for f in input_source])}")
    else:
        summary_file_content_lines.append(f"Clustering Results for: {os.path.basename(input_source)}")
    
    # Conditional cosmic threshold display
    if is_compare_mode:
        summary_file_content_lines.append(f"COSMIC threshold (distance): N/A")
    else:
        summary_file_content_lines.append(f"COSMIC threshold (distance): {threshold}")
    
    if rmsd_threshold is not None:
        summary_file_content_lines.append(f"RMSD validation threshold: {rmsd_threshold:.3f} Å")
    
    # Moved these to comparison_specific_summary_lines for conditional output
    # if weights:
    #     summary_file_content_lines.append(f"Applied Feature Weights: {weights}")
    # if abs_tolerances:
    #     summary_file_content_lines.append(f"Applied Absolute Tolerances: {abs_tolerances}")


    total_files_attempted = len(clean_data_for_clustering) + len(skipped_files)
    if total_files_attempted > 0:
        skipped_percentage = (len(skipped_files) / total_files_attempted) * 100
        skipped_info = f"{len(skipped_files)} ({skipped_percentage:.1f}%)"
    else:
        skipped_info = f"{len(skipped_files)}"
    
    summary_file_content_lines.append(f"Total configurations processed: {len(clean_data_for_clustering)}")
    summary_file_content_lines.append(f"Total files skipped: <TOTAL_SKIPPED_PLACEHOLDER>")
    summary_file_content_lines.append(f"Critical skipped files: <IMAG_NEED_RECALC_PLACEHOLDER>")
    summary_file_content_lines.append(f"Critical reduced-vector unmatched: <REDUCED_UNMATCHED_PLACEHOLDER>")
    summary_file_content_lines.append(f"Total number of final clusters: <TOTAL_CLUSTERS_PLACEHOLDER>")
    if rmsd_threshold is not None:
        summary_file_content_lines.append(f"Total RMSD moved configurations: <TOTAL_RMSD_OUTLIERS_PLACEHOLDER>")
    summary_file_content_lines.append("\n" + "=" * 75 + "\n")


    previous_hbond_group_processed = False
    total_imag_clustered_with_normal = 0
    total_imag_need_recalc = 0
    total_non_converged_critical = 0
    all_skipped_clustered_with_normal = []
    all_skipped_need_recalc = []
    all_non_converged_critical = []

    # --- Dynamic feature vector: all 14 features are always candidates ---
    # Per-structure availability determines the actual vector used.
    _all_scalar_features = [
        'electronic_energy', 'gibbs_free_energy', 'homo_energy', 'lumo_energy',
        'radius_of_gyration', 'dipole_moment', 'homo_lumo_gap',
        'first_vib_freq', 'last_vib_freq', 'average_hbond_distance', 'average_hbond_angle'
    ]
    _scalar_features = list(_all_scalar_features)

    # Compute available features per structure
    _vector_size_hist = {}
    for _mol in clean_data_for_clustering:
        _available = set()
        for _fname in _scalar_features:
            _key = FEATURE_MAPPING.get(_fname, _fname)
            if is_valid_scalar(_mol.get(_key)):
                _available.add(_fname)
        if has_valid_rotational_constants(_mol):
            _available.update(['rotational_constants_A', 'rotational_constants_B', 'rotational_constants_C'])
        _mol['_available_features'] = _available
        _mol['_feature_vector_size'] = len(_available)
        _vector_size_hist[_mol['_feature_vector_size']] = _vector_size_hist.get(_mol['_feature_vector_size'], 0) + 1

    # The "full" vector is the maximum feature count found in the pool
    _pool_max_features = max(_mol['_feature_vector_size'] for _mol in clean_data_for_clustering)
    for _mol in clean_data_for_clustering:
        _mol['_is_full_feature'] = (_mol['_feature_vector_size'] == _pool_max_features)

    _full_count = sum(1 for _mol in clean_data_for_clustering if _mol['_is_full_feature'])
    _non_full_count = len(clean_data_for_clustering) - _full_count
    print(f"Full-feature structures: {_full_count} ({_pool_max_features} features)")
    if _non_full_count > 0:
        print(f"  Reduced-vector structures: {_non_full_count}")
        for _vec_size in sorted([k for k in _vector_size_hist.keys() if k < _pool_max_features], reverse=True):
            _count = _vector_size_hist[_vec_size]
            print(f"    - {_count} with {_vec_size} features")

    # --- Boltzmann Population Calculation (based on initial property clusters) ---
    all_initial_property_clusters = []
    pseudo_global_cluster_id_counter = 1 # This counter is for assigning unique IDs to initial clusters for Boltzmann calc

    # Track reduced structures that could not match any fullest-tier cluster
    all_reduced_unmatched = []

    for hbond_count, group_data in sorted(hbond_groups.items()):
        if len(group_data) < 2 or not group_has_any_clustering_feature_data(group_data):
            for single_mol_data in group_data:
                single_mol_data['_initial_cluster_label'] = hbond_count
                single_mol_data['_parent_global_cluster_id'] = pseudo_global_cluster_id_counter
                all_initial_property_clusters.append([single_mol_data])
                pseudo_global_cluster_id_counter += 1
        else:
            # --- Tier-based dynamic vector clustering ---
            # Separate fullest-tier from reduced-tier structures
            fullest_tier = [m for m in group_data if m['_is_full_feature']]
            reduced_tier = [m for m in group_data if not m['_is_full_feature']]

            # If no fullest tier structures, promote the local maximum
            if not fullest_tier:
                _local_max = max(m['_feature_vector_size'] for m in group_data)
                fullest_tier = [m for m in group_data if m['_feature_vector_size'] == _local_max]
                reduced_tier = [m for m in group_data if m['_feature_vector_size'] < _local_max]

            # If fullest tier too small to cluster, treat all as singletons
            if len(fullest_tier) < 2:
                for mol in fullest_tier + reduced_tier:
                    mol['_initial_cluster_label'] = hbond_count
                    mol['_parent_global_cluster_id'] = pseudo_global_cluster_id_counter
                    all_initial_property_clusters.append([mol])
                    pseudo_global_cluster_id_counter += 1
                continue

            # Feature selection on fullest tier only
            active_numerical_features_for_group, dropped_scalar_features = select_complete_group_scalar_features(fullest_tier, list(_scalar_features))
            use_rotational_constants = all(has_valid_rotational_constants(m) for m in fullest_tier)

            # Build vectors for fullest tier
            features_for_scaling_raw, ordered_feature_names_for_scaling = _build_feature_vectors(
                fullest_tier, active_numerical_features_for_group, use_rotational_constants, weights
            )

            if not features_for_scaling_raw or all(len(f) == 0 for f in features_for_scaling_raw):
                for mol in fullest_tier + reduced_tier:
                    mol['_initial_cluster_label'] = hbond_count
                    mol['_parent_global_cluster_id'] = pseudo_global_cluster_id_counter
                    all_initial_property_clusters.append([mol])
                    pseudo_global_cluster_id_counter += 1
                continue

            features_for_scaling_raw_np = np.array(features_for_scaling_raw, dtype=float)
            features_scaled = _zscore_scale(features_for_scaling_raw_np, ordered_feature_names_for_scaling, min_std_threshold, abs_tolerances)

            linkage_matrix = linkage(features_scaled, method='average', metric='euclidean')
            initial_cluster_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')

            # Build cluster data from fullest tier
            initial_clusters_data = {}
            for i, label in enumerate(initial_cluster_labels):
                fullest_tier[i]['_initial_cluster_label'] = label
                initial_clusters_data.setdefault(label, []).append(fullest_tier[i])

            # Match reduced-tier structures against fullest-tier clusters
            if reduced_tier:
                matched, unmatched = _match_reduced_to_clusters(
                    reduced_tier, fullest_tier, initial_cluster_labels,
                    active_numerical_features_for_group, use_rotational_constants,
                    weights, threshold, min_std_threshold, abs_tolerances
                )
                for label, mols in matched.items():
                    for mol in mols:
                        mol['_initial_cluster_label'] = label
                    initial_clusters_data.setdefault(label, []).extend(mols)
                # Unmatched reduced structures → singletons flagged as critical
                for mol in unmatched:
                    mol['_reduced_unmatched'] = True
                    mol['_initial_cluster_label'] = hbond_count
                    mol['_parent_global_cluster_id'] = pseudo_global_cluster_id_counter
                    all_initial_property_clusters.append([mol])
                    all_reduced_unmatched.append(mol)
                    pseudo_global_cluster_id_counter += 1

            initial_clusters_list_unsorted = list(initial_clusters_data.values())
            initial_clusters_list_sorted_by_energy = sorted(
                initial_clusters_list_unsorted,
                key=lambda cluster: (min(_sorting_energy(m) for m in cluster),
                                     min(m['filename'] for m in cluster))
            )

            for initial_prop_cluster in initial_clusters_list_sorted_by_energy:
                parent_id = pseudo_global_cluster_id_counter
                for member_conf in initial_prop_cluster:
                    member_conf['_parent_global_cluster_id'] = parent_id
                all_initial_property_clusters.append(initial_prop_cluster)
                pseudo_global_cluster_id_counter += 1

    boltzmann_g1_data = {}
    global_min_gibbs_energy = None
    global_min_rep_filename = "N/A"
    global_min_cluster_id = "N/A"

    if _dataset_has_freq and all_initial_property_clusters:
        # Find the global minimum Gibbs energy among all representatives
        # Also store the filename and cluster ID of this global minimum representative
        valid_reps_for_emin = []
        for initial_prop_cluster in all_initial_property_clusters:
            # Select the lowest energy member as the representative for this initial property cluster
            rep_conf = min(initial_prop_cluster,
                           key=lambda x: (_sorting_energy(x), x['filename']))

            if rep_conf.get('gibbs_free_energy') is not None:
                valid_reps_for_emin.append({
                    'energy': rep_conf['gibbs_free_energy'],
                    'filename': rep_conf['filename'],
                    'cluster_id': rep_conf['_parent_global_cluster_id']
                })

        if valid_reps_for_emin:
            global_min_info = min(valid_reps_for_emin, key=lambda x: x['energy'])
            global_min_gibbs_energy = global_min_info['energy']
            global_min_rep_filename = global_min_info['filename']
            global_min_cluster_id = global_min_info['cluster_id']

            sum_factors_g1 = 0.0

            for initial_prop_cluster in all_initial_property_clusters:
                rep_conf = min(initial_prop_cluster,
                               key=lambda x: (_sorting_energy(x), x['filename']))

                if rep_conf.get('gibbs_free_energy') is None:
                    continue

                rep_gibbs_energy = rep_conf['gibbs_free_energy']
                cluster_id = rep_conf['_parent_global_cluster_id']
                cluster_size = len(initial_prop_cluster)

                delta_e = rep_gibbs_energy - global_min_gibbs_energy
                
                if BOLTZMANN_CONSTANT_HARTREE_PER_K * temperature_k == 0:
                    factor_g1 = 1.0 if delta_e == 0 else 0.0
                else:
                    factor_g1 = np.exp(-delta_e / (BOLTZMANN_CONSTANT_HARTREE_PER_K * temperature_k))
                
                boltzmann_g1_data[cluster_id] = {
                    'energy': rep_gibbs_energy,
                    'filename': rep_conf['filename'],
                    'population': factor_g1,
                    'cluster_size': cluster_size
                }
                
                sum_factors_g1 += factor_g1
            
            if sum_factors_g1 > 0:
                for cluster_id, data in boltzmann_g1_data.items():
                    data['population'] = (data['population'] / sum_factors_g1) * 100.0
            else:
                for cluster_id in boltzmann_g1_data:
                    boltzmann_g1_data[cluster_id]['population'] = 0.0
    # --- End Boltzmann Population Calculation ---


    # Collect all final clusters for unique motifs creation
    all_final_clusters = []
    cluster_id_mapping = {}  # Maps cluster index to cluster ID for motifs

    # Now iterate through the hbond_groups again to perform the clustering and write files
    # This loop is responsible for generating the actual clusters and writing their files.
    # The Boltzmann data calculated above will be passed to write_cluster_dat_file.
    cluster_global_id_counter = 1
    for hbond_count, group_data in sorted(hbond_groups.items()):
        
        hbond_group_summary_lines = []
        
        if previous_hbond_group_processed:
            hbond_group_summary_lines.append("\n" + "-" * 75 + "\n") 
        
        if group_hb and not is_compare_mode:
            hbond_group_summary_lines.append(f"Hydrogen bonds: {hbond_count}\n")
        else:
            hbond_group_summary_lines.append(f"All configurations\n")
        hbond_group_summary_lines.append(f"Configurations: {len(group_data)}")

        current_hbond_group_clusters_for_final_output = [] 

        if len(group_data) < 2 or not group_has_any_clustering_feature_data(group_data):
            vprint(f"\nSkipping detailed clustering: Less than 2 configurations or no valid numerical features left after filtering. Treating each as a single-configuration cluster.")
            
            for single_mol_data in group_data:
                single_mol_data['_rmsd_pass_origin'] = 'first_pass_validated' 
                current_hbond_group_clusters_for_final_output.append([single_mol_data]) 

        else: # Proceed with actual clustering
            # --- Tier-based dynamic vector clustering ---
            # In compare mode, use all structures together (common features);
            # otherwise, cluster fullest tier first, then match reduced structures.
            if is_compare_mode:
                _clustering_pool = group_data
                _reduced_pool = []
            else:
                _clustering_pool = [m for m in group_data if m['_is_full_feature']]
                _reduced_pool = [m for m in group_data if not m['_is_full_feature']]
                if not _clustering_pool:
                    _local_max = max(m['_feature_vector_size'] for m in group_data)
                    _clustering_pool = [m for m in group_data if m['_feature_vector_size'] == _local_max]
                    _reduced_pool = [m for m in group_data if m['_feature_vector_size'] < _local_max]

            # If clustering pool is too small, treat everything as singletons
            if len(_clustering_pool) < 2:
                for mol in group_data:
                    mol['_rmsd_pass_origin'] = 'first_pass_validated'
                    current_hbond_group_clusters_for_final_output.append([mol])
                    if not mol['_is_full_feature']:
                        mol['_reduced_unmatched'] = True
                        all_reduced_unmatched.append(mol)
                continue

            filenames_base = [os.path.splitext(item['filename'])[0] for item in _clustering_pool]

            # Feature selection on clustering pool (fullest tier or all in compare mode)
            active_numerical_features_for_group, dropped_scalar_features = select_complete_group_scalar_features(_clustering_pool, list(_scalar_features))
            use_rotational_constants = all(has_valid_rotational_constants(m) for m in _clustering_pool)

            # Build feature vectors for the clustering pool
            features_for_scaling_raw, ordered_feature_names_for_scaling = _build_feature_vectors(
                _clustering_pool, active_numerical_features_for_group, use_rotational_constants, weights
            )

            if not features_for_scaling_raw or all(len(f) == 0 for f in features_for_scaling_raw):
                vprint(f"  WARNING: No numerical features left for clustering after applying weights. Treating each as a single-configuration cluster.")
                print_step(f"{len(group_data)} config(s) - no features for clustering")
                for mol in group_data:
                    mol['_rmsd_pass_origin'] = 'first_pass_validated'
                    current_hbond_group_clusters_for_final_output.append([mol])
                continue

            # Announce clustering
            _hb_tag = f" (H-bonds={hbond_count})" if group_hb else ""
            _tier_info = f" (fullest tier: {len(_clustering_pool)}/{len(group_data)})" if _reduced_pool else ""
            print_step(f"Clustering {len(_clustering_pool)} configurations{_hb_tag}{_tier_info}...")

            if dropped_scalar_features:
                vprint(f"  Reduced scalar feature set due to missing values: {', '.join(dropped_scalar_features)}")
            if not use_rotational_constants:
                vprint(f"  Reduced vector excludes rotational constants (not available for all structures)")

            # Z-score scaling
            features_for_scaling_raw_np = np.array(features_for_scaling_raw, dtype=float)
            features_scaled = _zscore_scale(features_for_scaling_raw_np, ordered_feature_names_for_scaling, min_std_threshold, abs_tolerances)

            linkage_matrix = linkage(features_scaled, method='average', metric='euclidean')

            # --- Apply threshold clustering ---
            initial_cluster_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')
            _main_optimal_k = len(set(initial_cluster_labels))
            _main_cut_height = threshold

            _mojena_t, _mojena_k = compute_mojena_threshold(linkage_matrix, verbose=VERBOSE)

            # --- Extract configuration labels for dendrogram ---
            import re
            conf_labels = []
            for filename in filenames_base:
                match = re.search(r'(\d+)', filename)
                if match:
                    conf_labels.append(match.group(1))
                else:
                    conf_labels.append(filename)

            if is_compare_mode:
                dendrogram_title_suffix = "Comparison"
            elif group_hb:
                dendrogram_title_suffix = f"H-bonds = {hbond_count}"
            else:
                dendrogram_title_suffix = "All configurations"

            if group_hb and not is_compare_mode:
                dendrogram_filename = os.path.join(dendrogram_images_folder, f"dendrogram_H{hbond_count}.png")
            else:
                dendrogram_filename = os.path.join(dendrogram_images_folder, f"dendrogram.png")

            plot_annotated_dendrogram(
                linkage_matrix, _main_optimal_k, _main_cut_height,
                dendrogram_filename, title_suffix=dendrogram_title_suffix,
                conf_labels=conf_labels,
                mojena_threshold=_mojena_t, mojena_k=_mojena_k)
            vprint(f"Dendrogram saved as '{os.path.basename(dendrogram_filename)}'")

            # --- Match reduced-tier structures against fullest-tier clusters ---
            _matched_reduced = {}
            _unmatched_reduced = []
            if _reduced_pool:
                _matched_reduced, _unmatched_reduced = _match_reduced_to_clusters(
                    _reduced_pool, _clustering_pool, initial_cluster_labels,
                    active_numerical_features_for_group, use_rotational_constants,
                    weights, threshold, min_std_threshold, abs_tolerances
                )
                if _matched_reduced:
                    _total_matched = sum(len(v) for v in _matched_reduced.values())
                    print(f"  Reduced-vector matching: {_total_matched} matched, {len(_unmatched_reduced)} unmatched (critical)")
                if _unmatched_reduced:
                    all_reduced_unmatched.extend(_unmatched_reduced)

            if is_compare_mode and len(group_data) >= 2:
                print("  Comparison mode: Overriding clustering to output a single combined cluster.")
                for i, member in enumerate(group_data):
                    member['_initial_cluster_label'] = 1
                    member['_parent_global_cluster_id'] = 1
                    member['_rmsd_pass_origin'] = 'first_pass_validated'
                    member['_second_rmsd_sub_cluster_id'] = None
                    member['_second_rmsd_context_listing'] = None
                    member['_second_rmsd_rep_filename'] = None

                prop_rep_conf = group_data[0]
                first_rmsd_listing = []
                for member_conf in group_data:
                    if member_conf == prop_rep_conf:
                        rmsd_val = 0.0
                    elif prop_rep_conf.get('final_geometry_coords') is not None and prop_rep_conf.get('final_geometry_atomnos') is not None and \
                         member_conf.get('final_geometry_coords') is not None and \
                         member_conf.get('final_geometry_atomnos') is not None:
                        rmsd_val = calculate_rmsd(
                            prop_rep_conf['final_geometry_atomnos'], prop_rep_conf['final_geometry_coords'],
                            member_conf['final_geometry_atomnos'], member_conf['final_geometry_coords']
                        )
                    else:
                        rmsd_val = None
                    first_rmsd_listing.append({'filename': member_conf['filename'], 'rmsd_to_rep': rmsd_val})

                for member_conf in group_data:
                    member_conf['_first_rmsd_context_listing'] = first_rmsd_listing

                current_hbond_group_clusters_for_final_output.append(group_data)
            else:
                # Normal clustering: build cluster data from fullest tier + absorbed reduced
                initial_clusters_data = {}
                for i, label in enumerate(initial_cluster_labels):
                    _clustering_pool[i]['_initial_cluster_label'] = label
                    initial_clusters_data.setdefault(label, []).append(_clustering_pool[i])

                # Absorb matched reduced structures into their clusters
                for label, mols in _matched_reduced.items():
                    for mol in mols:
                        mol['_initial_cluster_label'] = label
                    initial_clusters_data.setdefault(label, []).extend(mols)

                # Unmatched reduced → singleton clusters flagged as critical
                for mol in _unmatched_reduced:
                    mol['_reduced_unmatched'] = True
                    mol['_rmsd_pass_origin'] = 'first_pass_validated'
                    current_hbond_group_clusters_for_final_output.append([mol])

                initial_clusters_list_unsorted = list(initial_clusters_data.values())
                initial_clusters_list_sorted_by_energy = sorted(
                    initial_clusters_list_unsorted,
                    key=lambda cluster: (min(_sorting_energy(m) for m in cluster),
                                         min(m['filename'] for m in cluster))
                )

                for initial_prop_cluster in initial_clusters_list_sorted_by_energy:
                    if initial_prop_cluster and initial_prop_cluster[0].get('_parent_global_cluster_id') is None:
                        parent_id = pseudo_global_cluster_id_counter
                        for member_conf in initial_prop_cluster:
                            member_conf['_parent_global_cluster_id'] = parent_id
                        pseudo_global_cluster_id_counter += 1

                if rmsd_threshold is not None:
                    print(f"  Performing first RMSD validation...")

                    validated_main_clusters, individual_outliers_from_first_pass = \
                        post_process_clusters_with_rmsd(initial_clusters_list_sorted_by_energy, rmsd_threshold)

                    current_hbond_group_clusters_for_final_output.extend(validated_main_clusters)
                    total_rmsd_outliers_first_pass += len(individual_outliers_from_first_pass)

                    if individual_outliers_from_first_pass:
                        print(f"    Attempting second RMSD clustering on {len(individual_outliers_from_first_pass)} outliers from first pass...")

                        outliers_grouped_by_parent_global_cluster = {}
                        for outlier_conf in individual_outliers_from_first_pass:
                            parent_global_id = outlier_conf.get('_parent_global_cluster_id')
                            if parent_global_id is not None:
                                outliers_grouped_by_parent_global_cluster.setdefault(parent_global_id, []).append(outlier_conf)

                        for parent_global_id_for_outlier_group, outlier_group in outliers_grouped_by_parent_global_cluster.items():
                            if len(outlier_group) > 1:
                                print(f"      Re-clustering {len(outlier_group)} outliers from original Cluster {parent_global_id_for_outlier_group}...")
                                second_level_clusters = perform_second_rmsd_clustering(outlier_group, rmsd_threshold)
                                current_hbond_group_clusters_for_final_output.extend(second_level_clusters)
                            else:
                                single_member_processed = perform_second_rmsd_clustering(outlier_group, rmsd_threshold)
                                current_hbond_group_clusters_for_final_output.extend(single_member_processed)
                else:
                    for cluster in initial_clusters_list_sorted_by_energy:
                        for member in cluster:
                            member['_rmsd_pass_origin'] = 'first_pass_validated'
                    current_hbond_group_clusters_for_final_output.extend(initial_clusters_list_sorted_by_energy)

        current_hbond_group_clusters_for_final_output.sort(key=lambda cluster: (min(_sorting_energy(m) for m in cluster),
                                                                                  min(m['filename'] for m in cluster))) 

        # Filter out structures with imaginary frequencies AFTER clustering
        current_hbond_group_clusters_for_final_output, hbond_skipped_info = filter_imaginary_freq_structures(
            current_hbond_group_clusters_for_final_output, 
            output_base_dir, 
            input_source,
            total_processed=len(clean_data_for_clustering)
        )

        # Remove non-converged structures from clusters after imaginary filtering.
        # These are always critical and must never become representatives.
        current_hbond_group_clusters_for_final_output, hbond_non_converged = filter_non_converged_structures(
            current_hbond_group_clusters_for_final_output, dataset_has_freq=_dataset_has_freq
        )
        
        # Track and accumulate skipped structures
        total_imag_clustered_with_normal += len(hbond_skipped_info.get('clustered_with_normal', []))
        total_imag_need_recalc += len(hbond_skipped_info.get('need_recalculation', []))
        all_skipped_clustered_with_normal.extend(hbond_skipped_info.get('clustered_with_normal', []))
        all_skipped_need_recalc.extend(hbond_skipped_info.get('need_recalculation', []))
        total_non_converged_critical += len(hbond_non_converged)
        all_non_converged_critical.extend(hbond_non_converged)

        all_final_clusters.extend(current_hbond_group_clusters_for_final_output)

        hbond_group_summary_lines.append(f"Number of clusters: {len(current_hbond_group_clusters_for_final_output)}\n\n")

        # Print info only if there are valid clusters after filtering
        if len(current_hbond_group_clusters_for_final_output) > 0:
            # Check if this was a single-config group (before any potential RMSD processing)
            original_group_size = len(group_data)
            if original_group_size < 2:
                print()
                print()
                print_step(f"{original_group_size} config(s) - treating as single-config clusters")
                print()  # Blank line after single-config group

        # Add blank line before multi-config group
        if len(current_hbond_group_clusters_for_final_output) > 0 and len(group_data) >= 2:
            print()
        
        for members_data in current_hbond_group_clusters_for_final_output:
            current_global_cluster_id = cluster_global_id_counter 

            summary_line_prefix = f"Cluster {current_global_cluster_id} ({len(members_data)} configurations)"

            if rmsd_threshold is not None and members_data[0].get('_rmsd_pass_origin') == 'second_pass_formed':
                parent_global_cluster_id_for_tag = members_data[0].get('_parent_global_cluster_id')

                if len(members_data) == 1:
                    summary_line_prefix += f" | RMSD Validated from Cluster {parent_global_cluster_id_for_tag}"
                else: 
                    summary_line_prefix += f" | RMSD Validated from Cluster {parent_global_cluster_id_for_tag}"

            hbond_group_summary_lines.append(summary_line_prefix + ":")
            hbond_group_summary_lines.append("Files:")
            for m_data in members_data:
                if _DATASET_HAS_FREQ:
                    if m_data['gibbs_free_energy'] is not None:
                        gibbs_str = f"{m_data['gibbs_free_energy']:.6f} Hartree ({hartree_to_kcal_mol(m_data['gibbs_free_energy']):.2f} kcal/mol, {hartree_to_ev(m_data['gibbs_free_energy']):.2f} eV)"
                    else:
                        gibbs_str = "N/A"
                    hbond_group_summary_lines.append(f"  - {m_data['filename']} (Gibbs Energy: {gibbs_str})")
                else:
                    elec = m_data.get('final_electronic_energy')
                    elec_str = f"{elec:.6f} Hartree" if elec is not None else "N/A"
                    hbond_group_summary_lines.append(f"  - {m_data['filename']} (Electronic Energy: {elec_str})")
            hbond_group_summary_lines.append("\n")
            
            # Add newline after cluster info
            if members_data == current_hbond_group_clusters_for_final_output[-1]:
                print()

            # Print cluster info - verbose shows all files, non-verbose shows summary
            if VERBOSE:
                print(f"\n{summary_line_prefix}:")
                for m_data in members_data:
                    print(f"  - {m_data['filename']}")
            else:
                print_step(f"{summary_line_prefix}")

            cluster_name_prefix = "" 
            num_configurations_in_cluster = len(members_data)

            if num_configurations_in_cluster == 1:
                cluster_name_prefix = f"cluster_{current_global_cluster_id}"
            else:
                cluster_name_prefix = f"cluster_{current_global_cluster_id}_{num_configurations_in_cluster}"

            write_cluster_dat_file(cluster_name_prefix, members_data, output_base_dir, rmsd_threshold, 
                                   hbond_count_for_original_cluster=hbond_count if group_hb else None, weights=weights, tolerances=abs_tolerances)
            vprint(f"Wrote combined data for Cluster '{cluster_name_prefix}' to '{cluster_name_prefix}.dat'")

            cluster_xyz_subfolder = os.path.join(extracted_clusters_folder, cluster_name_prefix)
            os.makedirs(cluster_xyz_subfolder, exist_ok=True)
            vprint(f"  Saving .xyz files to '{cluster_xyz_subfolder}'")

            # Store cluster ID in each member for later motif mapping
            for m_data in members_data:
                m_data['_cluster_global_id'] = current_global_cluster_id
                xyz_filename = os.path.join(cluster_xyz_subfolder, os.path.splitext(m_data['filename'])[0] + ".xyz")
                write_xyz_file(m_data, xyz_filename) 
            
            combine_xyz_files(members_data, cluster_xyz_subfolder, output_base_name=cluster_name_prefix)

            total_clusters_outputted += 1 
            cluster_global_id_counter += 1

        if len(current_hbond_group_clusters_for_final_output) > 0:
            summary_file_content_lines.extend(hbond_group_summary_lines)
            previous_hbond_group_processed = True 

    # Write combined skipped structures summary after clustering
    if all_skipped_clustered_with_normal or all_skipped_need_recalc:
        combined_skipped_info = {
            'clustered_with_normal': all_skipped_clustered_with_normal,
            'need_recalculation': all_skipped_need_recalc
        }
        filter_imaginary_freq_structures(
            [],  # Empty cluster list since we're using precomputed data
            output_base_dir,
            input_source,
            total_processed=len(clean_data_for_clustering),
            write_summary=True,
            precomputed_skipped=combined_skipped_info
        )

    # Persist non-converged critical structures for redo mode.
    if all_non_converged_critical:
        save_non_converged_critical_structures(
            all_non_converged_critical,
            output_base_dir,
            input_source,
            total_processed=len(clean_data_for_clustering)
        )

    total_skipped_all = len(skipped_files) + total_imag_clustered_with_normal + total_imag_need_recalc + total_non_converged_critical
    if total_files_attempted > 0:
        total_skipped_percentage = (total_skipped_all / total_files_attempted) * 100
        critical_total = total_imag_need_recalc + total_non_converged_critical
        critical_skipped_percentage = (critical_total / total_files_attempted) * 100
        total_skipped_str = f"{total_skipped_all} ({total_skipped_percentage:.1f}%)"
        critical_skipped_str = f"{critical_total} ({critical_skipped_percentage:.1f}%)"
    else:
        total_skipped_str = str(total_skipped_all)
        critical_skipped_str = str(total_imag_need_recalc + total_non_converged_critical)

    # Reduced-vector structures that could not match any fullest-tier cluster
    # are flagged as critical for redo (recalculation / Hessian / imaginary displacement).
    # These were collected during tier-based matching in all_reduced_unmatched.
    reduced_unmatched_critical = list(all_reduced_unmatched)
    if reduced_unmatched_critical:
        print(f"\nReduced-vector criticals: {len(reduced_unmatched_critical)} structure(s) did not match any full-feature cluster.")
        print(f"  These structures need recalculation (redo mode).")

    if total_files_attempted > 0:
        reduced_unmatched_percentage = (len(reduced_unmatched_critical) / total_files_attempted) * 100
        reduced_unmatched_str = f"{len(reduced_unmatched_critical)} ({reduced_unmatched_percentage:.1f}%)"
    else:
        reduced_unmatched_str = str(len(reduced_unmatched_critical))

    for i, line in enumerate(summary_file_content_lines):
        if "<TOTAL_CLUSTERS_PLACEHOLDER>" in line:
            summary_file_content_lines[i] = line.replace("<TOTAL_CLUSTERS_PLACEHOLDER>", str(total_clusters_outputted))
        if "<TOTAL_RMSD_OUTLIERS_PLACEHOLDER>" in line:
            summary_file_content_lines[i] = line.replace("<TOTAL_RMSD_OUTLIERS_PLACEHOLDER>", str(total_rmsd_outliers_first_pass))
        if "<TOTAL_SKIPPED_PLACEHOLDER>" in line:
            summary_file_content_lines[i] = line.replace("<TOTAL_SKIPPED_PLACEHOLDER>", total_skipped_str)
        if "<IMAG_NEED_RECALC_PLACEHOLDER>" in line:
            summary_file_content_lines[i] = line.replace("<IMAG_NEED_RECALC_PLACEHOLDER>", critical_skipped_str)
        if "<REDUCED_UNMATCHED_PLACEHOLDER>" in line:
            summary_file_content_lines[i] = line.replace("<REDUCED_UNMATCHED_PLACEHOLDER>", reduced_unmatched_str)

    # Add comparison-specific details at the very end if in comparison mode
    if is_compare_mode:
        comparison_specific_summary_lines.append("\n" + "=" * 75 + "\n")
        comparison_specific_summary_lines.append("Comparison Parameters:\n")
        if weights:
            comparison_specific_summary_lines.append("  Applied Feature Weights:")
            for key, value in weights.items():
                comparison_specific_summary_lines.append(f"    - {key}: {value}")
        if abs_tolerances:
            comparison_specific_summary_lines.append("  Applied Absolute Tolerances:")
            for key, value in abs_tolerances.items():
                # Format the float to a fixed number of decimal places to avoid scientific notation
                # Adjust precision as needed, e.g., for 1e-5 use 5 decimal places, for 0.5 use 1 decimal place
                # A general approach is to find the number of decimal places or use a high fixed number.
                # For simplicity and to cover common cases, I'll use a fixed high precision like 7.
                formatted_value = f"{value:.7f}".rstrip('0').rstrip('.') if '.' in f"{value:.7f}" else f"{int(value)}"
                comparison_specific_summary_lines.append(f"    - {key}: {formatted_value}")
        summary_file_content_lines.extend(comparison_specific_summary_lines)

    # Create cluster ID mapping for motifs BEFORE Boltzmann analysis
    for cluster_idx, cluster_members in enumerate(all_final_clusters):
        if cluster_members:
            # Get the cluster ID from the first member (all members have the same cluster ID)
            cluster_id = cluster_members[0].get('_cluster_global_id', cluster_idx + 1)
            cluster_id_mapping[cluster_idx] = cluster_id

    # --- RE-COMPUTE Boltzmann using FINAL cluster IDs ---
    # This ensures traceability: cluster IDs in Boltzmann match cluster IDs in summary
    # Boltzmann distribution is only meaningful with Gibbs free energy (freq mode).
    boltzmann_final_data = {}
    final_global_min_energy = None
    final_global_min_filename = "N/A"
    final_global_min_cluster_id = "N/A"

    # First pass: find representatives and global minimum
    final_representatives = []
    if _dataset_has_freq:
        for cluster_members in all_final_clusters:
            if not cluster_members:
                continue

            # Get the final cluster ID
            final_cluster_id = cluster_members[0].get('_cluster_global_id')
            if final_cluster_id is None:
                continue

            # Find lowest energy representative in this cluster (excluding imaginary freq structures)
            valid_members = [m for m in cluster_members if not m.get('_has_imaginary_freqs', False)]
            if not valid_members:
                valid_members = cluster_members  # Use all if none are valid

            rep = min(valid_members,
                      key=lambda x: (_sorting_energy(x), x['filename']))

            if rep.get('gibbs_free_energy') is not None:
                final_representatives.append({
                    'cluster_id': final_cluster_id,
                    'filename': rep['filename'],
                    'energy': rep['gibbs_free_energy'],
                    'cluster_size': len(cluster_members)
                })
    
    # Find global minimum
    if final_representatives:
        min_rep = min(final_representatives, key=lambda x: x['energy'])
        final_global_min_energy = min_rep['energy']
        final_global_min_filename = min_rep['filename']
        final_global_min_cluster_id = min_rep['cluster_id']
        
        # Calculate Boltzmann factors
        sum_factors = 0.0
        for rep in final_representatives:
            delta_e = rep['energy'] - final_global_min_energy
            if BOLTZMANN_CONSTANT_HARTREE_PER_K * temperature_k == 0:
                factor = 1.0 if delta_e == 0 else 0.0
            else:
                factor = np.exp(-delta_e / (BOLTZMANN_CONSTANT_HARTREE_PER_K * temperature_k))
            
            boltzmann_final_data[rep['cluster_id']] = {
                'energy': rep['energy'],
                'filename': rep['filename'],
                'population': factor,
                'cluster_size': rep['cluster_size']
            }
            sum_factors += factor
        
        # Normalize populations
        if sum_factors > 0:
            for cid in boltzmann_final_data:
                boltzmann_final_data[cid]['population'] = (boltzmann_final_data[cid]['population'] / sum_factors) * 100.0
        else:
            for cid in boltzmann_final_data:
                boltzmann_final_data[cid]['population'] = 0.0

    # Detect if inputs are from a previous motif step to determine output naming
    # This enables the workflow: conf_### → motif_## → umotif_##
    all_input_filenames = [m.get('filename', '') for m in clean_data_for_clustering]
    output_prefix, folder_prefix, is_second_step = detect_motif_input_level(all_input_filenames)
    
    if is_second_step:
        print_step(f"Detected motif inputs - using '{output_prefix}_##' naming for unique motifs")

    # Create motifs folder with representative structures from each cluster
    # Pass boltzmann_final_data to sort motifs by population (highest population = motif_01)
    motif_to_cluster_mapping = create_unique_motifs_folder(all_final_clusters, output_base_dir,
                                                          cluster_id_mapping=cluster_id_mapping,
                                                          output_prefix=output_prefix,
                                                          folder_prefix=folder_prefix,
                                                          boltzmann_data=boltzmann_final_data,
                                                          dataset_has_freq=_dataset_has_freq)

    # --- Create separate Boltzmann Distribution Analysis file ---
    boltzmann_file_content_lines = []
    if final_global_min_energy is not None:
        # Helper function to center text within 75 characters (same as summary)
        def center_text_boltzmann(text, width=75):
            return text.center(width)
        
        # Header for Boltzmann file - using the same beautiful ASCII art as summary
        boltzmann_file_content_lines.append("=" * 75)
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append(center_text_boltzmann("***************************"))
        boltzmann_file_content_lines.append(center_text_boltzmann("* C O S M I C *"))
        boltzmann_file_content_lines.append(center_text_boltzmann("***************************"))
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append("                             √≈≠==≈                                  ")
        boltzmann_file_content_lines.append("   √≈≠==≠≈√   √≈≠==≠≈√         ÷++=                      ≠===≠       ")
        boltzmann_file_content_lines.append("     ÷++÷       ÷++÷           =++=                     ÷×××××=      ")
        boltzmann_file_content_lines.append("     =++=       =++=     ≠===≠ ÷++=      ≠====≠         ÷-÷ ÷-÷      ")
        boltzmann_file_content_lines.append("     =++=       =++=    =××÷=≠=÷++=    ≠÷÷÷==÷÷÷≈      ≠××≠ =××=     ")
        boltzmann_file_content_lines.append("     =++=       =++=   ≠××=    ÷++=   ≠×+×    ×+÷      ÷+×   ×+××    ")
        boltzmann_file_content_lines.append("     =++=       =++=   =+÷     =++=   =+-×÷==÷×-×≠    =×+×÷=÷×+-÷    ")
        boltzmann_file_content_lines.append("     ≠×+÷       ÷+×≠   =+÷     =++=   =+---×××××÷×   ≠××÷==×==÷××≠   ")
        boltzmann_file_content_lines.append("      =××÷     =××=    ≠××=    ÷++÷   ≠×-×           ÷+×       ×+÷   ")
        boltzmann_file_content_lines.append("       ≠=========≠      ≠÷÷÷=≠≠=×+×÷-  ≠======≠≈√  -÷×+×≠     ≠×+×÷- ")
        boltzmann_file_content_lines.append("          ≠===≠           ≠==≠  ≠===≠     ≠===≠    ≈====≈     ≈====≈ ")
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append(center_text_boltzmann("Universidad de Antioquia - Medellín - Colombia"))
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append(center_text_boltzmann("Boltzmann Population Distribution Analysis"))
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append(center_text_boltzmann(version))
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append(center_text_boltzmann("Química Física Teórica - QFT"))
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append("=" * 75 + "\n")
        
        # Reference configuration info
        boltzmann_file_content_lines.append("Reference Configuration:")
        boltzmann_file_content_lines.append(f"  Structure: {os.path.splitext(final_global_min_filename)[0]} from cluster_{final_global_min_cluster_id}")
        boltzmann_file_content_lines.append(f"  Reference Energy (Emin): {final_global_min_energy:.6f} Hartree ({hartree_to_kcal_mol(final_global_min_energy):.2f} kcal/mol, {hartree_to_ev(final_global_min_energy):.2f} eV)")
        boltzmann_file_content_lines.append(f"  Temperature (T): {temperature_k:.2f} K")
        boltzmann_file_content_lines.append("")
        
        # Population by Energy Minimum (gi = 1)
        boltzmann_file_content_lines.append("=" * 60)
        boltzmann_file_content_lines.append("Population by Energy Minimum")
        boltzmann_file_content_lines.append("(assuming non-degeneracy, gi = 1)")
        boltzmann_file_content_lines.append("=" * 60)
        boltzmann_file_content_lines.append("")
        
        # Sort by population percentage descending for better readability
        # Number motifs by population rank: highest population = motif_01/umotif_01
        sorted_final_data = sorted(boltzmann_final_data.items(), key=lambda item: item[1]['population'], reverse=True)
        
        # Add section header for motif assignment
        display_name = "Unique Motif (umotif)" if output_prefix == 'umotif' else "Motif"
        boltzmann_file_content_lines.append(f"{display_name} Assignment Summary")
        boltzmann_file_content_lines.append("(sorted by Boltzmann population)\n")
        
        for motif_rank, (cluster_id, data) in enumerate(sorted_final_data, 1):
            # Motif number is based on population rank (most populated = 01)
            cluster_line = f"cluster_{cluster_id} ({output_prefix}_{motif_rank:02d})"
                
            boltzmann_file_content_lines.append(cluster_line)
            boltzmann_file_content_lines.append(f"  From structure: {os.path.splitext(data['filename'])[0]}")
            boltzmann_file_content_lines.append(f"  Gibbs Energy: {data['energy']:.6f} Hartree ({hartree_to_kcal_mol(data['energy']):.2f} kcal/mol, {hartree_to_ev(data['energy']):.2f} eV)")
            boltzmann_file_content_lines.append(f"  Population: {data['population']:.2f} %")
            boltzmann_file_content_lines.append("")

        boltzmann_file_content_lines.append("=" * 75)
    
    # Write separate files
    summary_file = os.path.join(output_base_dir, "clustering_summary.txt")
    with open(summary_file, "w", newline='\n') as f:
        f.write("\n".join(summary_file_content_lines))

    # Write Boltzmann distribution file only if we have the analysis data
    if boltzmann_file_content_lines:
        boltzmann_file = os.path.join(output_base_dir, "boltzmann_distribution.txt")
        with open(boltzmann_file, "w", newline='\n') as f:
            f.write("\n".join(boltzmann_file_content_lines))
        print_step(f"Boltzmann distribution saved to '{os.path.basename(boltzmann_file)}'")

    print()
    print_step(f"Clustering summary saved to '{os.path.basename(summary_file)}'")
    vprint(f"   Full path: {output_base_dir}")

    # Emit marker for workflow detection: opt-only mode means no true minima.
    if not _dataset_has_freq:
        print("COSMIC_OPT_ONLY_MODE")

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COSMIC (COnfigurational Similarity via Motif Identification Code) - Hierarchical clustering for quantum chemistry structures\nPhysicochemical feature-based discrimination of conformational families",
        usage="cosmic [OPTIONS] [FOLDER]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""DESCRIPTION:
  COSMIC performs topological clustering of quantum chemistry outputs
  using a multi-dimensional physicochemical feature vector (energy, HOMO-LUMO
  gap, dipole moment, rotational constants, vibrational frequencies, H-bond geometry).
  Hierarchical clustering with optional RMSD refinement identifies unique
  conformational families and filters redundant structures.

METHODOLOGY:
  1. Feature Extraction: Parse QM outputs (.log/.out) for scalar descriptors
  2. Z-score Standardization: Normalize features across different units
  3. Weighted Euclidean Distance: Calculate cosmic matrix
  4. Hierarchical Clustering: UPGMA linkage with 2-sigma threshold on Z-standardized features
     (Calinski-Harabasz + Silhouette score optimization)
  5. RMSD Refinement (optional): Distinguish geometric stereoisomers
  6. Quality Control: Flag imaginary frequencies and convergence failures

OPTIONS:
  Manual Override (deprecated):
    --threshold=FLOAT, --th=FLOAT Manual distance threshold (overrides statistical
                                  consensus method). Consider removing this flag.

  Geometric Validation:
    --rmsd=FLOAT                  Enable RMSD validation in Ångström
                                  If no value given, defaults to 1.0 Å
                                  Recommended: 0.5-1.0 for tight geometric control

  Processing Control:
    --cores=INT, -j=INT           Number of CPU cores (default: auto-detect)
    --reprocess-files             Ignore cache and force feature re-extraction
    --output-dir=PATH             Output directory (default: current directory)

  Advanced Features:
    --weights=STRING              Custom feature weights in format:
                                  '(energy=0.1)(gap=0.2)(dipole=0.15)'
    --compare FILE [FILE ...]     Direct comparison mode (minimum 2 files)
    -T=FLOAT, --temperature=FLOAT Temperature for Boltzmann analysis in K
                                  (default: 298.15)
    --group-hb                    Group structures by H-bond count before
                                  clustering (separate dendrograms per family)
  
  Output:
    -v, --verbose                 Enable detailed progress output
    -V, --version                 Display version information and exit

INPUT:
  FOLDER                          Directory containing QM output files
                                  (.log for Gaussian, .out for ORCA)
                                  If omitted, interactive folder selection

OUTPUT FILES:
  clustering_summary.txt          Comprehensive clustering report with statistics
  data_cache.pkl                  Cache file for output data
  dendrogram_images/              Hierarchical clustering dendrograms
    └── dendrogram.png            Single dendrogram (or dendrogram_H{N}.png with --group-hb)
  extracted_data/                 Raw data files (.dat) for each cluster
    └── cluster_*.dat
  extracted_clusters/             Individual cluster directories
    ├── cluster_1/                Single-member cluster (no combined file)
    │   ├── structure.xyz         Individual structure file
    │   └── structure.mol         MOL format (if OpenBabel available)
    └── cluster_2_5/              Multi-member cluster (5 members)
        ├── structure1.xyz        Individual XYZ files for each member
        ├── structure2.xyz
        ├── cluster_2_5.xyz       Combined multi-frame XYZ file
        └── cluster_2_5.mol       Combined MOL file
  skipped_structures/             Structures with imaginary frequencies (if any)
    ├── skipped_summary.txt       Details of skipped structures
    ├── clustered_with_normal/    Imaginary freq. clustered with valid structures
    └── need_recalculation/       Isolated imaginary freq. structures

EXAMPLES:
  Basic clustering (2-sigma threshold - recommended):
    cosmic                          Default threshold=2.0 (moderate)
    cosmic --rmsd=1                 Add RMSD validation at 1.0 Å
    cosmic calculation/             Process specific folder

  Manual threshold (deprecated):
    cosmic --th=2                   Manual clustering at threshold 2.0
    cosmic --th=1 --rmsd=0.5        Tight clustering with RMSD control
  
  Performance optimization:
    cosmic --th=2 -j=8              Use 8 CPU cores for parallel processing
    cosmic --reprocess-files        Force cache refresh after updates
  
  Direct comparison:
    cosmic --compare s1.log s2.log s3.log  Compare specific structures
  
  Custom analysis:
    cosmic --th=2 --weights='(energy=0.3)(gap=0.2)'  Weighted features
    cosmic --th=2 -T=350.0          Boltzmann analysis at 350 K

WORKFLOW INTEGRATION:
  COSMIC is typically used after ASCEC sampling and QM optimization:
    1. ascec input.in r5        → 5 replicated annealing runs
    2. ascec calc template.inp  → generate QM inputs
    3. [Run ORCA/Gaussian calculations]
    4. ascec sort               → organize results
    5. cosmic --th=2        → identify unique conformers

RECOMMENDATIONS:
  - Start with --th=2 for initial exploration
  - Use --rmsd=1 for systems with subtle geometric differences
  - Adjust --th value based on desired clustering granularity
  - Use --reprocess-files after modifying QM outputs or settings
  - Check dendrogram.png to validate threshold selection

SUPPORTED FORMATS:
  - Gaussian: .log files (via cclib parser)
  - ORCA 5.0.x: .out files (via cclib parser)
  - ORCA 6.1+: .out files (via OPI parser)
  Note: ORCA 6.0 is not supported; use 5.0.x or upgrade to 6.1+

CITATION:
  If you use COSMIC in your research, please cite:
  Manuel, G.; Sara, G.; Albeiro, R. Universidad de Antioquia (2026)

MORE INFORMATION:
  Repository:     https://github.com/manuel2gl/qft-ascec-cosmic
  Documentation:  See user manual for theoretical background
  Support:        Química Física Teórica - Universidad de Antioquia
""")
    # Clustering threshold (default: 2.0 = 2-sigma rule on Z-standardized UPGMA distances)
    parser.add_argument("--threshold", "--th", type=float, default=2.0, metavar="FLOAT",
                        help="UPGMA distance threshold for dendrogram cut (default: 2.0, the 2-sigma rule on Z-standardized features). Use 0.5 for tight, 2.0 for moderate, 3.0-4.0 for loose clustering.")

    # Geometric validation
    parser.add_argument("--rmsd", type=float, nargs='?', const=1.0, default=None, metavar="FLOAT",
                        help="RMSD geometric validation in Ångström (default: 1.0)")

    # Processing control
    parser.add_argument("--cores", "-j", type=int, default=None, metavar="INT",
                        help="number of CPU cores (default: auto-detect)")
    parser.add_argument("--reprocess-files", "-r", action="store_true",
                        help="ignore cache and force re-extraction")
    parser.add_argument("--output-dir", type=str, default=None, metavar="PATH",
                        help="output directory (default: current directory)")
    parser.add_argument("--weights", type=str, default="", metavar="STRING",
                        help="custom feature weights: '(energy=0.1)(gap=0.2)'")
    parser.add_argument("--compare", nargs='+', metavar="FILE",
                        help="direct comparison mode (minimum 2 files)")
    parser.add_argument("-T", "--temperature", type=float, default=298.15, metavar="FLOAT",
                        help="temperature for Boltzmann analysis in K (default: 298.15)")
    
    # Output control
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable detailed progress output")
    parser.add_argument("-V", "--version", action="store_true",
                        help="display version and exit")
    
    parser.add_argument("--group-hb", action="store_true",
                        help="group structures by H-bond count before clustering (separate dendrograms per HB family)")

    # Hidden/advanced options
    parser.add_argument("--min-std-threshold", type=float, default=1e-6,
                        help=argparse.SUPPRESS)
    parser.add_argument("--abs-tolerance", type=str, default="",
                        help=argparse.SUPPRESS)
    parser.add_argument("--update-cache", type=str, default=None,
                        help=argparse.SUPPRESS)
    
    # Positional argument
    parser.add_argument('input_source', nargs='?', default=None, metavar="FOLDER",
                        help='directory containing QM output files')


    # Preprocess arguments to handle -j8 format
    processed_args = preprocess_j_argument(sys.argv[1:])
    args = parser.parse_args(processed_args)
    
    # Check if version is requested
    if args.version:
        print_version_banner()
        sys.exit(0)
    
    clustering_threshold = args.threshold
    rmsd_validation_threshold = args.rmsd
    output_directory = args.output_dir
    force_reprocess_cache = args.reprocess_files
    weights_dict = parse_weights_argument(args.weights)
    min_std_threshold_val = args.min_std_threshold
    abs_tolerances_dict = parse_abs_tolerance_argument(args.abs_tolerance)
    num_cores = args.cores if args.cores is not None else get_cpu_count_fast()
    temperature_k = args.temperature
    
    # Update the global verbose flag
    VERBOSE = args.verbose

    # Set default absolute tolerances if not provided via command line
    if not abs_tolerances_dict:
        abs_tolerances_dict = {
            "electronic_energy": 5e-6,  # Tighter, was 1e-6
            "gibbs_free_energy": 5e-6,  # Tighter, was 1e-6
            "homo_energy": 3e-4,        # Increased from 1e-4
            "lumo_energy": 2e-4,        # Keep as is
            "homo_lumo_gap": 3e-4,      # Increased from 1e-4
            "dipole_moment": 1.5e-3,    # Keep as is
            "radius_of_gyration": 1.5e-4, # Keep as is
            "rotational_constants_A": 7e-5, # Keep as is
            "rotational_constants_B": 3.5e-4, # Keep as is
            "rotational_constants_C": 3e-4, # Keep as is
            "first_vib_freq": 1e-2,     # Keep as is
            "last_vib_freq": 0.3,      # Keep as is
            "average_hbond_distance": 1e-3, # Keep as is
            "average_hbond_angle": 0.1    # Increased from 1e-2 to avoid boundary cases
        }

    current_dir = os.getcwd()

    
    if args.compare:
        if len(args.compare) < 2:
            print("Error: --compare requires at least 2 files.")
            exit(1)
        
        compare_files = args.compare
        
        # Check that all files exist
        for file_path in compare_files:
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                exit(1)
        
        # Determine file extensions and check compatibility
        extensions = [os.path.splitext(f)[1].lower() for f in compare_files]
        unique_extensions = set(extensions)
        
        if len(unique_extensions) > 1:
            print(f"Warning: Comparing files with different extensions ({', '.join(unique_extensions)}). Proceeding, but ensure they are compatible.")
        
        # Use the extension of the first file for pattern
        file_extension_pattern_for_compare = extensions[0] if extensions[0] in ['.log', '.out'] else None
        if not file_extension_pattern_for_compare:
            print("Error: Provided files do not have .log or .out extensions.")
            exit(1)

        file_names = [os.path.basename(f) for f in compare_files]
        print(f"\n--- Comparing {len(compare_files)} files: {', '.join(file_names)} ---\n")
        perform_clustering_and_analysis(
            input_source=compare_files,
            threshold=clustering_threshold,
            file_extension_pattern=file_extension_pattern_for_compare, # Pass for consistency, though not used for glob
            rmsd_threshold=rmsd_validation_threshold,
            output_base_dir=output_directory,
            force_reprocess_cache=True, # Always reprocess for comparison
            weights=weights_dict,
            is_compare_mode=True,
            min_std_threshold=min_std_threshold_val,
            abs_tolerances=abs_tolerances_dict,
            num_cores=num_cores,
            temperature_k=temperature_k,
            group_hb=args.group_hb
        )
        print(f"\n--- Finished comparing {len(compare_files)} files: {', '.join(file_names)} ---\n")

    else: # Normal mode (folder processing)
        if args.input_source:
            # Non-interactive mode
            if not os.path.isdir(args.input_source):
                print(f"Error: Input source '{args.input_source}' is not a directory.")
                exit(1)
            
            selected_folders = [args.input_source]
            
            # Auto-detect file extension
            has_log = bool(glob.glob(os.path.join(args.input_source, "*.log")))
            has_out = bool(glob.glob(os.path.join(args.input_source, "*.out")))
            
            if has_out:
                file_extension_pattern = "*.out"
            elif has_log:
                file_extension_pattern = "*.log"
            else:
                print(f"Error: No .log or .out files found in '{args.input_source}'.")
                exit(1)
                
        else:
            # Interactive mode
            all_potential_folders = [current_dir] + [d for d in glob.glob(os.path.join(current_dir, '*')) if os.path.isdir(d)]
            
            folders_with_log_files = []
            folders_with_out_files = []

            for folder in all_potential_folders:
                has_log = bool(glob.glob(os.path.join(folder, "*.log")))
                has_out = bool(glob.glob(os.path.join(folder, "*.out")))
                
                if has_log:
                    folders_with_log_files.append(folder)
                if has_out:
                    folders_with_out_files.append(folder)

            all_valid_folders_to_display = sorted(list(set(folders_with_log_files + folders_with_out_files)))

            if not all_valid_folders_to_display:
                print("No subdirectories containing .log or .out files found, or files are organized directly in the current directory.")
                exit(0)

            print("\nFound the following folder(s) containing quantum chemistry log/out files:\n")
            for i, folder in enumerate(all_valid_folders_to_display):
                display_name = os.path.basename(folder)
                if folder == current_dir:
                    display_name = "./"
                
                folder_types_present = []
                if folder in folders_with_log_files: folder_types_present.append(".log")
                if folder in folders_with_out_files: folder_types_present.append(".out")
                
                print(f"  [{i+1}] {display_name} (Contains: {', '.join(folder_types_present)})")

            selected_folders = []
            while True:
                choice = input("\nEnter the number of the folder to process, or type 'a' to process all: ").strip().lower()
                
                if choice == 'a':
                    selected_folders = all_valid_folders_to_display
                    break
                try:
                    folder_index = int(choice) - 1
                    if 0 <= folder_index < len(all_valid_folders_to_display):
                        selected_folders = [all_valid_folders_to_display[folder_index]]
                        break
                    else:
                        print("\nInvalid number. Please enter a valid number from the list.")
                except ValueError:
                    print("\nInvalid input. Please enter a number or 'a'.")

            selected_set_has_log = False
            selected_set_has_out = False
            for folder_path in selected_folders:
                if folder_path in folders_with_log_files:
                    selected_set_has_log = True
                if folder_path in folders_with_out_files:
                    selected_set_has_out = True
                if selected_set_has_log and selected_set_has_out:
                    break

            file_extension_pattern = None 
            if selected_set_has_log and selected_set_has_out:
                while file_extension_pattern is None:
                    type_choice = input("\nBoth .log and .out files are present in the selected folder(s).\nWhich file type would you like to process?\n  [1] .log files\n  [2] .out files\n  Enter your choice (1 or 2): ").strip()
                    if type_choice == '1':
                        file_extension_pattern = "*.log"
                    elif type_choice == '2':
                        file_extension_pattern = "*.out"
                    else:
                        print("Invalid choice. Please enter '1' or '2'.")
            elif selected_set_has_log:
                file_extension_pattern = "*.log" 
                print("\nOnly .log files found in the selected folder(s). Processing .log files.")
            elif selected_set_has_out:
                file_extension_pattern = "*.out" 
                print("\nOnly .out files found in the selected folder(s). Processing .out files.")
            else:
                print("\nNo .log or .out files found in the selected folder(s) that match available types. Exiting.")
                exit(0)

        print(f"\nProcessing {len(selected_folders)} folder(s) for files matching '{file_extension_pattern}'...")
        for folder_path in selected_folders:
            display_name = os.path.basename(folder_path)
            if folder_path == current_dir:
                display_name = "./"
            print(f"\nProcessing folder: {display_name}\n")

            perform_clustering_and_analysis(folder_path, clustering_threshold, file_extension_pattern, rmsd_validation_threshold, output_directory, force_reprocess_cache, weights_dict, is_compare_mode=False, min_std_threshold=min_std_threshold_val, abs_tolerances=abs_tolerances_dict, num_cores=num_cores, temperature_k=temperature_k, group_hb=args.group_hb)

            print(f"\nFinished processing folder: {display_name}\n")

    print()
    print_step("All selected molecular analyses complete!")
