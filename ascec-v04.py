#!/usr/bin/env python

import argparse
import dataclasses
import glob
import json
import math
import multiprocessing
import os
import pickle
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-Party Imports
import numpy as np

# Optional / Conditional Imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Set backend before importing pyplot
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore

# OPI (ORCA Python Interface) for ORCA 6.1+ support
try:
    from opi.output.core import Output as OPIOutput  # type: ignore
    OPI_AVAILABLE = True
except ImportError:
    OPI_AVAILABLE = False
    OPIOutput = None  # type: ignore

# cclib for ORCA 5.x and Gaussian support (used for output parsing)
try:
    from cclib.io import ccread
    CCLIB_AVAILABLE = True
except ImportError:
    CCLIB_AVAILABLE = False
    ccread = None  # type: ignore

# Set multiprocessing start method for better compatibility
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # If already set, continue
        pass

# Global Constants
ASCEC_VERSION = "* ASCEC-v04: Feb 2026 *"  # ASCEC version string for display

max_mole = 100  # Increase this if you have more than 100 molecules
B2 = 3.166811563e-6   # Boltzmann constant in Hartree/K (approx. 3.166811563 × 10^-6 Hartree/K)

# Constants for overlap prevention during initial configuration generation (in config_molecules)
overlap_scale_factor = 0.7 # Factor to make overlap check slightly more lenient (e.g., allow partial overlap)


max_overlap_placement_attempts = 100000 # Max attempts to place a single molecule without significant overlap

# Set this to True to create a SEPARATE COPY of the XYZ file
# (e.g., mtobox_seed.xyz) which will include 8 dummy 'X' atoms for
# visualizing the box in programs like Avogadro2 or GaussView.
# If False, only the original XYZ file (mto_seed.xyz) will be generated.
# This can be overridden by the --nobox command line flag.
create_box_xyz_copy = True 

def parse_verbosity_level(argv: List[str]) -> int:
    """Parse verbosity level from command line arguments.
    Returns: 0 (silent), 1 (-v), 2 (-v2), 3 (-v3)
    """
    for arg in argv:
        if arg == '-v':
            return 1
        elif arg == '-v2':
            return 2
        elif arg == '-v3':
            return 3
    return 0


# Supports placeholder marker (.asc,) and explicit input-file markers
# that end with .asc (e.g. formic_annealing.asc).
PROTOCOL_MARKER_RE = re.compile(
    r'^\s*(?:\.asc|[^,\s#]+\.asc)\s*,',
    re.IGNORECASE,
)


def is_protocol_marker_line(raw_line: str) -> bool:
    """Return True when a line starts an embedded protocol block."""
    if not raw_line:
        return False
    stripped = raw_line.strip().lstrip('\ufeff')
    if not stripped:
        return False
    # Ignore inline comments while preserving the leading command token.
    if '#' in stripped:
        stripped = stripped.split('#', 1)[0].strip()
    if not stripped:
        return False
    return bool(PROTOCOL_MARKER_RE.match(stripped))

def print_version_banner(script_name="ASCEC"):
    """Print the ASCII art banner with UdeA logo and version information."""
    banner = """
===========================================================================

                           *********************                           
                           *     A S C E C     *                           
                           *********************                           

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


                  Annealing Simulado Con Energía Cuántica                  

                           {version}                           

                        Química Física Teórica - QFT                       


===========================================================================
""".format(version=ASCEC_VERSION)
    print(banner)

# Symbol for dummy atoms used to mark box corners.

# WARNING: 'X' is not a standard element.
dummy_atom_symbol = "X"

# Atomic radii - Used in some initial configurations for overlap prevention
# These are typical Covalent Radii (single bond, in Angstroms).
# please ensure consistency.
r_atom = {
    # Covalent radii for elements in Angstroms (for volume calculations and steric interactions)
    # Based on Cordero et al. (2008) "Covalent radii revisited" Dalton Trans. 2832-2838
    # and optimized values from previous ASCEC usage
    
    # Period 1
    1: 0.31,   # H (Hydrogen)
    2: 0.28,   # He (Helium) - Van der Waals approximation
    
    # Period 2
    3: 1.28,   # Li (Lithium)
    4: 0.96,   # Be (Beryllium)
    5: 0.84,   # B (Boron)
    6: 0.73,   # C (Carbon) - Standard covalent radius
    7: 0.71,   # N (Nitrogen)
    8: 0.66,   # O (Oxygen)
    9: 0.57,   # F (Fluorine)
    10: 0.58,  # Ne (Neon) - Van der Waals approximation
    
    # Period 3
    11: 1.66,  # Na (Sodium)
    12: 1.41,  # Mg (Magnesium)
    13: 1.21,  # Al (Aluminum)
    14: 1.11,  # Si (Silicon)
    15: 1.07,  # P (Phosphorus)
    16: 1.05,  # S (Sulfur)
    17: 1.02,  # Cl (Chlorine)
    18: 1.06,  # Ar (Argon) - Van der Waals approximation
    
    # Period 4
    19: 2.03,  # K (Potassium)
    20: 1.76,  # Ca (Calcium)
    21: 1.70,  # Sc (Scandium)
    22: 1.60,  # Ti (Titanium)
    23: 1.53,  # V (Vanadium)
    24: 1.39,  # Cr (Chromium)
    25: 1.39,  # Mn (Manganese) - low spin
    26: 1.32,  # Fe (Iron) - low spin
    27: 1.26,  # Co (Cobalt) - low spin
    28: 1.24,  # Ni (Nickel)
    29: 1.32,  # Cu (Copper)
    30: 1.22,  # Zn (Zinc)
    31: 1.22,  # Ga (Gallium)
    32: 1.20,  # Ge (Germanium)
    33: 1.19,  # As (Arsenic)
    34: 1.20,  # Se (Selenium)
    35: 1.20,  # Br (Bromine)
    36: 1.16,  # Kr (Krypton) - Van der Waals approximation
    
    # Period 5
    37: 2.20,  # Rb (Rubidium)
    38: 1.95,  # Sr (Strontium)
    39: 1.90,  # Y (Yttrium)
    40: 1.75,  # Zr (Zirconium)
    41: 1.64,  # Nb (Niobium)
    42: 1.54,  # Mo (Molybdenum)
    43: 1.47,  # Tc (Technetium)
    44: 1.46,  # Ru (Ruthenium)
    45: 1.42,  # Rh (Rhodium)
    46: 1.39,  # Pd (Palladium)
    47: 1.45,  # Ag (Silver)
    48: 1.44,  # Cd (Cadmium)
    49: 1.42,  # In (Indium)
    50: 1.39,  # Sn (Tin)
    51: 1.39,  # Sb (Antimony)
    52: 1.38,  # Te (Tellurium)
    53: 1.39,  # I (Iodine)
    54: 1.40,  # Xe (Xenon) - Van der Waals approximation
    
    # Period 6
    55: 2.44,  # Cs (Cesium)
    56: 2.15,  # Ba (Barium)
    57: 2.07,  # La (Lanthanum)
    
    # Period 6 - Lanthanides (f-block, elements 58-71)
    58: 2.04,  # Ce (Cerium)
    59: 2.03,  # Pr (Praseodymium)
    60: 2.01,  # Nd (Neodymium)
    61: 1.99,  # Pm (Promethium)
    62: 1.98,  # Sm (Samarium)
    63: 1.98,  # Eu (Europium)
    64: 1.96,  # Gd (Gadolinium)
    65: 1.94,  # Tb (Terbium)
    66: 1.92,  # Dy (Dysprosium)
    67: 1.92,  # Ho (Holmium)
    68: 1.89,  # Er (Erbium)
    69: 1.90,  # Tm (Thulium)
    70: 1.87,  # Yb (Ytterbium)
    71: 1.87,  # Lu (Lutetium)
    
    # Period 6 - Transition metals (d-block, elements 72-80)
    72: 1.75,  # Hf (Hafnium)
    73: 1.70,  # Ta (Tantalum)
    74: 1.62,  # W (Tungsten)
    75: 1.51,  # Re (Rhenium)
    76: 1.44,  # Os (Osmium)
    77: 1.41,  # Ir (Iridium)
    78: 1.36,  # Pt (Platinum)
    79: 1.36,  # Au (Gold)
    80: 1.32,  # Hg (Mercury)
    
    # Period 6 - Main group (p-block, elements 81-86)
    81: 1.45,  # Tl (Thallium)
    82: 1.46,  # Pb (Lead)
    83: 1.48,  # Bi (Bismuth)
    84: 1.40,  # Po (Polonium)
    85: 1.50,  # At (Astatine)
    86: 1.50,  # Rn (Radon)
}

# Element Symbol to Atomic Number Mapping
# This dictionary will be used to convert element symbols from the input to atomic numbers.
element_symbols = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110,
    "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
}

# Atomic Number to Element Symbol Mapping (reverse of element_symbols)
# This dictionary will be used to convert atomic numbers to element symbols for output.
atomic_number_to_symbol = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
    41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
    61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
    71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
    81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
    91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm",
    101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds",
    111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"
}

# Atomic Weights for elements (Global Constant) - used for center of mass calculations
# T. Prohaska et al.: Pure Appl. Chem. 94, 573 (2022)
# Based on IUPAC Standard Atomic Weights (2021/2022) rounded to 4 significant figures
atomic_weights = {
    1: 1.008,    2: 4.003,    3: 6.940,    4: 9.012,    5: 10.81,
    6: 12.01,    7: 14.01,    8: 16.00,    9: 19.00,   10: 20.18,
    11: 22.99,   12: 24.31,   13: 26.98,   14: 28.09,   15: 30.97,
    16: 32.06,   17: 35.45,   18: 39.95,   19: 39.10,   20: 40.08,
    21: 44.96,   22: 47.87,   23: 50.94,   24: 52.00,   25: 54.94,
    26: 55.85,   27: 58.93,   28: 58.69,   29: 63.55,   30: 65.38,
    31: 69.72,   32: 72.63,   33: 74.92,   34: 78.97,   35: 79.90,
    36: 83.80,   37: 85.47,   38: 87.62,   39: 88.91,   40: 91.22,
    41: 92.91,   42: 95.95,   43: 97.00,   44: 101.1,   45: 102.9,
    46: 106.4,   47: 107.9,   48: 112.4,   49: 114.8,   50: 118.7,
    51: 121.8,   52: 127.6,   53: 126.9,   54: 131.3,   55: 132.9,
    56: 137.3,   57: 138.9,   58: 140.1,   59: 140.9,   60: 144.2,
    61: 145.0,   62: 150.4,   63: 152.0,   64: 157.2,   65: 158.9,
    66: 162.5,   67: 164.9,   68: 167.3,   69: 168.9,   70: 173.0,
    71: 175.0,   72: 178.5,   73: 180.9,   74: 183.8,   75: 186.2,
    76: 190.2,   77: 192.2,   78: 195.1,   79: 197.0,   80: 200.6,
    81: 204.4,   82: 207.2,   83: 209.0,   84: 209.0,   85: 210.0,
    86: 222.0,   87: 223.0,   88: 226.0,   89: 227.0,   90: 232.0,
    91: 231.0,   92: 238.0,   93: 237.0,   94: 244.0,   95: 243.0,
    96: 247.0,   97: 247.0,   98: 251.0,   99: 252.0,  100: 257.0,
    101: 258.0,  102: 259.0,  103: 262.0,  104: 267.0,  105: 270.0,
    106: 269.0,  107: 270.0,  108: 270.0,  109: 278.0,  110: 281.0,
    111: 281.0,  112: 285.0,  113: 286.0,  114: 289.0,  115: 289.0,
    116: 293.0,  117: 293.0,  118: 294.0
}

# Electronegativity for elements (used for sorting molecular formula strings)
electronegativity_values = {
    'H': 2.20, 'He': 0.0, 
    'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'Ne': 0.0,
    'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
    'Ar': 0.0,
    'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55,
    'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01,
    'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 0.0,
    'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.6, 'Mo': 2.16, 'Tc': 1.9,
    'Ru': 2.2, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69, 'In': 1.78, 'Sn': 1.96,
    'Sb': 2.05, 'Te': 2.1, 'I': 2.66, 'Xe': 0.0,
    'Cs': 0.79, 'Ba': 0.89, 'La': 1.1, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Pm': 1.13,
    'Sm': 1.17, 'Eu': 1.17, 'Gd': 1.2, 'Tb': 1.1, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24,
    'Tm': 1.25, 'Yb': 1.1, 'Lu': 1.27, 'Hf': 1.3, 'Ta': 1.5, 'W': 2.36, 'Re': 1.9,
    'Os': 2.2, 'Ir': 2.2, 'Pt': 2.28, 'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 1.87,
    'Bi': 2.02, 'Po': 2.0, 'At': 2.2, 'Rn': 0.0,
    'Fr': 0.7, 'Ra': 0.9, 'Ac': 1.1, 'Th': 1.3, 'Pa': 1.5, 'U': 1.38, 'Np': 1.36,
    'Pu': 1.28, 'Am': 1.13, 'Cm': 1.28, 'Bk': 1.3, 'Cf': 1.3, 'Es': 1.3, 'Fm': 1.3,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107,
    'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114,
    'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

# 2. Define SystemState Class
class SystemState:
    def __init__(self):
        # Configuration parameters read from input file
        self.random_generate_config: int = 0      # 0: Annealing; 1: Random configurations
        self.num_random_configs: int = 0          # Used if random_generate_config is 1
        self.cube_length: float = 0.0             # Simulation Cube Length (Angstroms)
        self.quenching_routine: int = 0           # 1: Linear, 2: Geometrical
        self.linear_temp_init: float = 0.0
        self.linear_temp_decrement: float = 0.0
        self.linear_num_steps: int = 0
        self.geom_temp_init: float = 0.0
        self.geom_temp_factor: float = 0.0
        self.geom_num_steps: int = 0
        self.max_cycle: int = 0                   # Maximum Monte Carlo Cycles per Temperature (initial value from input)
        self.max_cycle_floor: int = 10            # Floor value for maxstep reduction (default: 10)
        self.max_displacement_a: float = 0.0      # Maximum Displacement of each mass center (ds)
        self.max_rotation_angle_rad: float = 0.0  # Maximum Rotation angle in radians (dphi)
        self.conformational_move_prob: float = 0.0  # Probability of conformational move vs rigid-body move (read from input)
        self.max_dihedral_angle_rad: float = 0.0  # Maximum dihedral rotation angle in radians (read from input)
        self.ia: int = 0                          # QM program type (1: Gaussian, 2: ORCA, etc.)
        self.alias: str = ""                      # Program alias/executable name (e.g., "g09")
        self.qm_method: Optional[str] = None      # (e.g., "pm3", "hf") - Renamed from hamiltonian for clarity
        self.qm_basis_set: Optional[str] = None   # (e.g., "6-31G*", "STO-3G") - Renamed from basis_set for clarity
        self.charge: int = 0                      # (iQ)
        self.multiplicity: int = 0                # (iS2) - Renamed from spin_multiplicity for clarity
        self.qm_memory: Optional[str] = None      # memory - No default, will be None if not in input
        self.qm_nproc: Optional[int] = None       # nprocs
        self.qm_additional_keywords: str = ""     # if necessary
        self._orca_exe_checked: bool = False      # Cache ORCA executable check per run
        
        # Parallel processing settings for ASCEC
        self.ascec_parallel_cores: int = 1       # Number of cores for ASCEC operations (file I/O, coordinate transformations)
        self.use_ascec_parallel: bool = False    # Enable parallel processing within ASCEC operations
        
        self.num_molecules: int = 0               # (nmo) from input file
        self.output_dir: str = "."                # Directory for output files
        self.ivalE: int = 0                       # Evaluate Energy flag (0: No energy evaluation, just mto movements)
        self.mto: int = 0                         # Number of random configurations if ivalE=0 (Fortran's nrandom)
        self.qm_program: Optional[str] = None     # Will store "gaussian", "orca", etc., derived from alias or ia
        self.molecules_to_add = []                # Initialize as an empty list
        self.all_molecule_definitions = []        # Also good to initialize this if not already
        self.openbabel_alias = "obabel"           # Default Open Babel executable name
        
        # Add these from global constants
        self.r_atom = r_atom.copy() 
        self.overlap_scale_factor = overlap_scale_factor
        self.max_overlap_placement_attempts = 100000 # Max attempts to place a single molecule without significant overlap

        # Random number generator state (internal to ran0_method)
        # This MUST be an integer. It will be updated by ran0_method.
        self.random_seed: int = -1
        self.IY: int = 0                          # Internal state for ran0_method
        self.IVEC: List[int] = [0] * 32           # Internal state for ran0_method (NTAB=32 from ran0_method)

        # Simulation State Variables (updated during simulation)
        self.nstep: int = 0                        # Total energy evaluation steps (Fortran's cycle)
        self.icount: int = 0                       # Counter for steps at current temperature (Fortran's icount)
        self.iboltz: int = 0                       # Count of configurations accepted by Boltzmann (Fortran's iboltz)
        self.current_energy: float = 0.0          # Ep (current energy of the system)
        self.lowest_energy: float = float('inf')  # oldE (stores the lowest energy found, initialize to large value)
        self.lowest_energy_rp: Optional[np.ndarray] = None # Coordinates of the lowest energy config
        self.lowest_energy_iznu: Optional[List[int]] = None # Atomic numbers of the lowest energy config
        self.lowest_energy_config_idx: int = 0    # Index of the lowest energy config
        self.current_temp: float = 0.0            # Ti (current annealing temperature)
        self.maxstep: int = 0                     # Current maximum steps at a temperature (derived from max_cycle initially, then reduced)
        self.natom: int = 0                       # Total number of atoms, calculated from molecule definitions
        self.nmo: int = 0                         # Number of molecules (often kept in sync with num_molecules)
        self.jdum: str = ""                       # Timestamp/label for output files (can be populated from alias or datetime)
        self.xbox: float = 0.0                    # Length of the simulation cube (synced with cube_length)
        self.ds: float = 0.0                      # Max displacement (synced with max_displacement_a)
        self.dphi: float = 0.0                    # Max rotation angle (synced with max_rotation_angle_rad)
        self.last_accepted_qm_call_count: int = 0 # Stores qm_call_count at last accepted configuration for history
        self.total_accepted_configs: int = 0      # Counter for total accepted configurations written to XYZ
        self.lower_energy_configs: int = 0        # Counter for configurations with lower energy than previous lowest
        self.consecutive_na_steps: int = 0        # Counter for consecutive N/A steps in annealing
        
        # New variable to track the global QM call count at the last history line written
        self.last_history_qm_call_count: int = 0

        # Molecular and Atomic Data
        self.rp: np.ndarray = np.empty((0, 3), dtype=np.float64) # Current atomic coordinates
        self.rf: np.ndarray = np.empty((0, 3), dtype=np.float64) # Proposed atomic coordinates (not directly used as state, but as return value)
        self.rcm: np.ndarray = np.empty((max_mole, 3), dtype=np.float64) # Center of mass for each molecule
        self.imolec: List[int] = []               # Indices defining molecules (which atoms belong to which molecule)
        self.iznu: List[int] = []                 # Atomic numbers for each atom in the system
        self.all_molecule_definitions: List['MoleculeData'] = [] # Stores parsed molecule data from input
        self.coords_per_molecule: np.ndarray = np.empty((0,3), dtype=np.float64) # Template coordinates for one molecule
        self.atomic_numbers_per_molecule: List[int] = [] # Template atomic numbers for one molecule
        self.natom_per_molecule: int = 0          # Number of atoms in the template molecule
        self.molecule_label: str = ""             # Label of the template molecule

        # Element Maps
        self.atomic_number_to_symbol: Dict[int, str] = {}
        self.atomic_number_to_weight: Dict[int, float] = {} # Populated from atomic_weights
        self.atomic_number_to_mass: Dict[int, float] = {}   # Alias for atomic_number_to_weight, used for clarity in COM calcs

        # These Fortran-style arrays might be redundant if the dicts above are the primary source.
        self.sym: List[str] = [''] * 120          # From Fortran's symelem (symbol for atomic number index)
        self.wt: List[float] = [0.0] * 120        # From Fortran's wtelem (weight for atomic number index)
        
        # Additional QM related attributes
        self.qm_call_count: int = 0 # Counter for QM calculation calls (cumulative global count)
        self.initial_qm_retries: int = 100 # Number of retries for initial QM calculation, default to 100
        self.verbosity_level: int = 0 # 0: default, 1: --v (every 10 steps), 2: --va (all steps)
        # QM attempt debug preservation mode:
        #   'none'          -> do not copy per-attempt files
        #   'first_success' -> keep per-attempt files until first successful QM, then stop (recommended)
        #   'all'           -> keep per-attempt files for all attempts (slowest)
        self.qm_attempt_debug_mode: str = "first_success"
        self._first_success_qm_preserved: bool = False

        self.use_standard_metropolis: bool = False # Flag for Metropolis criterion
        
        # Additional attributes for proper type checking
        self.input_file_path: str = ""            # Path to the input file
        self.max_molecular_extent: float = 0.0    # Maximum molecular extent
        self.num_elements_defined: int = 0        # Number of unique elements defined
        self.element_types: List[Tuple[int, int]] = []  # List of (atomic_number, count) tuples
        self.iz_types: List[int] = []             # List of atomic numbers
        self.nnu_types: List[int] = []            # List of atom counts per element type
        self.volume_based_recommendations: Dict = {}  # Volume-based box recommendations

    def ran0_method(self) -> float:
        """
        A Python implementation of Numerical Recipes' ran0 random number generator.
        This method is kept for historical context/compatibility with the original Fortran.
        For new Python code, `random.random()` or `np.random.rand()` are generally preferred.
        """
        IA = 16807
        IM = 2147483647 # 2^31 - 1
        AM = 1.0 / IM
        IQ = 127773
        IR = 2836
        NTAB = 32
        NDIV = 1 + (IM - 1) // NTAB
        EPS = 1.2e-7
        RNMX = 1.0 - EPS

        if self.random_seed == 0:
            self.random_seed = 1

        if self.random_seed < 0:
            self.random_seed = -self.random_seed
            for j in range(NTAB + 7):
                k = self.random_seed // IQ
                self.random_seed = IA * (self.random_seed - k * IQ) - IR * k
                if self.random_seed < 0:
                    self.random_seed += IM
                if j < NTAB:
                    self.IVEC[j] = self.random_seed
            self.IY = self.IVEC[0]

        k = self.random_seed // IQ
        self.random_seed = IA * (self.random_seed - k * IQ) - IR * k
        if self.random_seed < 0:
            self.random_seed += IM
        j = self.random_seed // NDIV

        self.IVEC[j] = self.random_seed # This line was incorrect in previous version (used self.IVEC[j] = self.IY)
        self.IY = self.IVEC[j] # Correct usage: self.IY becomes the value from IVEC[j]

        if self.IY > RNMX * IM:
            return RNMX
        else:
            return float(self.IY) * AM

    def get_molecule_label_from_atom_index(self, atom_abs_idx: int) -> str:
        """
        Determines the label of the molecule to which a given atom belongs.
        Assumes `imolec` and `all_molecule_definitions` are correctly populated.
        """
        if not self.imolec or not self.all_molecule_definitions:
            return "Unknown"

        # Find which molecule this atom belongs to based on imolec
        # imolec stores the starting atom index for each molecule definition,
        # plus the total number of atoms at the end.
        for i in range(self.num_molecules):
            start_idx = self.imolec[i]
            end_idx = self.imolec[i+1] # This is the start of the *next* molecule, or total atoms

            if start_idx <= atom_abs_idx < end_idx:
                # The molecule_definitions list is indexed from 0 to (num_molecules - 1)
                # This needs to correctly map to the *instance* of the molecule.
                # Since molecules_to_add now stores the correct definition index for each instance, use that.
                mol_def_idx = self.molecules_to_add[i]
                return self.all_molecule_definitions[mol_def_idx].label
        return "Unknown"

# Helper for verbose printing
def _print_verbose(message: str, level: int, state: Optional[SystemState], file=sys.stderr):
    """
    Prints a message to stderr if the state's verbosity level meets or exceeds the required level.
    level 0: always print (critical errors, final summary, accepted configs, and key annealing steps)
    level 1: --v (every 10 steps), plus level 0)
    level 2: --va (all steps, plus level 0 and 1)
    """
    if state is None or state.verbosity_level >= level:
        print(message, file=file)
        file.flush()

def natural_sort_key(s):
    """
    Generate a sort key for natural (numerical) sorting.
    Converts 'opt_conf_10.inp' to ['opt_conf_', 10, '.inp'] for proper numerical sorting.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def get_optimal_workers(task_type: str, num_items: int) -> int:
    """
    Calculate optimal number of workers for different task types.
    
    Args:
        task_type: Type of task ('cpu_intensive', 'io_intensive', 'mixed')
        num_items: Number of items to process
    
    Returns:
        Optimal number of workers
    """
    cpu_count = multiprocessing.cpu_count()
    
    if task_type == 'cpu_intensive':
        # For CPU-intensive tasks like parsing files
        return min(cpu_count, num_items, 8)
    elif task_type == 'io_intensive':
        # For I/O-intensive tasks like file copying
        return min(cpu_count * 2, num_items, 6)  
    elif task_type == 'mixed':
        # For mixed tasks like file creation
        return min(cpu_count, num_items, 4)
    else:
        return min(cpu_count, num_items, 4)  # Default


def _process_xyz_file_for_calc(xyz_file_data):
    """
    Process a single XYZ file for calculation system creation.
    This is a module-level function to support multiprocessing.
    Used by both simple and standard calculation workflows.
    """
    xyz_file, template_content, optimization_dir_path, qm_program, input_ext, total_xyz_files, use_combined_naming = xyz_file_data
    
    # Extract configurations
    configurations = extract_configurations_from_xyz(xyz_file)
    if not configurations:
        return [], f"Warning: No configurations found in {xyz_file}"
    
    # Determine the run number from the directory name
    dir_name = os.path.dirname(xyz_file)
    if dir_name == ".":
        run_num = 1
    else:
        # Extract number from directory name (e.g., w6_annealing4_1 -> 1)
        parts = os.path.basename(dir_name).split('_')
        run_num = 1
        for part in reversed(parts):
            try:
                run_num = int(part)
                break
            except ValueError:
                continue
    
    file_input_files = []
    # Default fallback: use XYZ filename as source if extraction fails
    xyz_file_fallback = os.path.basename(xyz_file).replace('.xyz', '')
    
    # Create input files for each configuration
    for config in configurations:
        # Extract source name from the original comment line
        # Comment format: "Configuration: 221 | E = -114.08249081 a.u. | result_639730"
        # We want to preserve "result_639730" instead of using the XYZ filename
        original_comment = config['comment']
        energy_match = re.search(r'E = ([-\d.]+) a\.u\.', original_comment)
        config_match = re.search(r'Configuration: (\d+)', original_comment)
        
        # Try to extract source name from the last part of the comment (after the last |)
        source_name = xyz_file_fallback  # Default fallback
        if '|' in original_comment:
            parts = original_comment.split('|')
            # The last part should contain the source name
            last_part = parts[-1].strip()
            # Remove any trailing temperature info if present
            # e.g., "result_393519 | T = 10.0 K" -> "result_393519"
            if '|' in last_part:
                source_name = last_part.split('|')[0].strip()
            else:
                source_name = last_part
        
        energy = energy_match.group(1) if energy_match else "unknown"
        config_num = config_match.group(1) if config_match else config['config_num']
        
        # Create new comment without temperature, with source info
        if energy == "unknown":
            config['comment'] = f"Configuration: {config_num} | {source_name}"
        else:
            config['comment'] = f"Configuration: {config_num} | E = {energy} a.u. | {source_name}"
        
        # Use simpler naming when using -c flag OR only one XYZ file
        if use_combined_naming or total_xyz_files == 1:
            input_name = f"opt_conf_{config['config_num']}{input_ext}"
        else:
            input_name = f"opt{run_num}_conf_{config['config_num']}{input_ext}"
        input_path = os.path.join(optimization_dir_path, input_name)
        
        if create_qm_input_file(config, template_content, input_path, qm_program):
            file_input_files.append(input_name)
        
    # Include run number in message only if there are multiple files
    if total_xyz_files == 1:
        return file_input_files, f"Processed {xyz_file} with {len(configurations)} configurations"
    else:
        return file_input_files, f"Processed {xyz_file} (run {run_num}) with {len(configurations)} configurations"

def _process_xyz_file_for_opt(xyz_file_data):
    """
    Process a single XYZ file for optimization system creation.
    This is a module-level function to support multiprocessing.
    """
    xyz_file, template_content, opt_dir, qm_program, input_ext = xyz_file_data
    
    # Extract configurations
    configurations = extract_configurations_from_xyz(xyz_file)
    if not configurations:
        return [], f"Warning: No configurations found in {xyz_file}"
    
    file_input_files = []
    base_name = os.path.basename(xyz_file).replace('.xyz', '')
    
    # Create input files for each configuration
    for config in configurations:
        # Extract motif name from comment if available, otherwise use base filename
        import re
        comment = config['comment']
        
        # Check for umotif first (must come before motif check since umotif contains 'motif')
        umotif_match = re.search(r'(umotif_\d+)', comment, re.IGNORECASE)
        motif_match = re.search(r'(?<!u)(motif_\d+)', comment, re.IGNORECASE)
        
        # If not found in comment, try to extract from filename
        if not umotif_match and not motif_match:
            umotif_match = re.search(r'(umotif_\d+)', base_name, re.IGNORECASE)
            motif_match = re.search(r'(?<!u)(motif_\d+)', base_name, re.IGNORECASE)
        
        if umotif_match:
            # Keep umotif prefix, just add _opt suffix
            source_name = umotif_match.group(1).lower()
            input_name = f"{source_name}_opt{input_ext}"
        elif motif_match:
            # Keep motif prefix, just add _opt suffix
            # Similarity will later promote motif_##_opt → umotif_##
            source_name = motif_match.group(1).lower()
            input_name = f"{source_name}_opt{input_ext}"
        else:
            # For non-motif files, use simple opt_conf_X naming
            input_name = f"opt_conf_{config['config_num']}{input_ext}"
            source_name = base_name
        
        input_path = os.path.join(opt_dir, input_name)
        
        # Update config comment with source file info
        config['comment'] = f"Configuration: {config['config_num']} | Source: {source_name}"
        
        # Create input file
        if create_qm_input_file(config, template_content, input_path, qm_program):
            file_input_files.append(input_name)
    
    return file_input_files, f"Processed {xyz_file} with {len(configurations)} configurations"

def get_molecular_formula_string(atom_symbols_list: List[str]) -> str:
    """
    Generates a simple molecular formula string (e.g., H2O, CH4)
    from a list of atom symbols. Elements are sorted by a common chemical convention
    (C, then H, then others by increasing electronegativity).
    """
    if not atom_symbols_list:
        return ""

    counts = Counter(atom_symbols_list)

    def sort_key_for_formula(symbol):
        # Prioritize Carbon and Hydrogen based on common chemical formula conventions
        if symbol == 'C':
            return (-2, 0) # Carbon gets the highest priority (lowest tuple value)
        if symbol == 'H':
            return (-1, 0) # Hydrogen gets the second highest priority

        # For all other elements, sort by electronegativity (ascending)
        # If an element is not in our dictionary, assign a very high value (float('inf'))
        # so it sorts to the end of the list.
        electronegativity = electronegativity_values.get(symbol, float('inf'))
        return (0, electronegativity) # Others get lower priority, then sorted by EN

    sorted_elements = sorted(counts.keys(), key=sort_key_for_formula)

    formula_parts = []
    for symbol in sorted_elements:
        count = counts[symbol]
        formula_parts.append(symbol)
        if count > 1:
            formula_parts.append(str(count))
    return "".join(formula_parts)

def _post_process_mol_file(mol_filepath: str, state: SystemState):
    """
    Post-processes a .mol file to replace Open Babel's '*' dummy atom symbol with 'X'
    if the dummy_atom_symbol is 'X'.
    """
    if dummy_atom_symbol != "X": # Only needed if we actually use 'X' and obabel uses '*'
        return # Skip post-processing if not using 'X' as dummy or if dummy is default *

    temp_mol_path = mol_filepath + ".tmp"
    try:
        with open(mol_filepath, 'r') as infile, open(temp_mol_path, 'w') as outfile:
            for line in infile:
                # MOL format atom block: ATOM_NUMBER X Y Z ELEMENT_SYMBOL ...
                # Open Babel sometimes uses '*' as a dummy atom. The symbol is typically at column 32 (index 31)
                # Example: "  1 C       0.0000    0.0000    0.0000  0  0  0  0  0  0  0  0"
                # If there's a '*' at position 31 and a space after it, it's likely a dummy atom placeholder from OB.
                if len(line) >= 34 and line[31] == '*' and line[32] == ' ':
                            modified_line = line[:31] + dummy_atom_symbol + line[32:] 
                            outfile.write(modified_line)
                else:
                    outfile.write(line)
        shutil.move(temp_mol_path, mol_filepath)
        _print_verbose(f"    Post-processed '{os.path.basename(mol_filepath)}' for dummy atoms.", 1, state)
    except Exception as e:
        _print_verbose(f"    Warning: Could not post-process .mol file '{os.path.basename(mol_filepath)}': {e}", 0, state)


def convert_xyz_to_mol(xyz_filepath: str, openbabel_alias: str = "obabel", state: Optional[SystemState] = None) -> bool:
    """
    Converts a single .xyz file to a .mol file using Open Babel.
    Prints verbose messages indicating success or failure to sys.stderr.
    Returns True on successful conversion, False otherwise.
    """
    mol_filepath = os.path.splitext(xyz_filepath)[0] + ".mol"

    openbabel_full_path = shutil.which(openbabel_alias)
    if not openbabel_full_path:
        _print_verbose(f"\n  Open Babel ({openbabel_alias}) command not found or not executable. Skipping .mol conversion for '{os.path.basename(xyz_filepath)}'.", 0, state)
        _print_verbose("  Please ensure Open Babel is installed and added to your system's PATH, or provide the correct alias.", 0, state)
        return False

    try:
        conversion_command = [openbabel_full_path, "-i", "xyz", xyz_filepath, "-o", "mol", "-O", mol_filepath]
        _print_verbose(f"  Converting '{os.path.basename(xyz_filepath)}' to MOL...", 1, state)
        
        process = subprocess.run(conversion_command, check=False, capture_output=True, text=True)
        
        if process.returncode != 0:
            _print_verbose(f"  Open Babel conversion failed for '{os.path.basename(xyz_filepath)}'.", 1, state)
            # Only print detailed stdout/stderr if verbosity is high
            if state and state.verbosity_level >= 2:
                _print_verbose(f"  STDOUT (first 5 lines):\n{_format_stream_output(process.stdout, max_lines=5, prefix='    ')}", 2, state)
                _print_verbose(f"  STDERR (first 5 lines):\n{_format_stream_output(process.stderr, max_lines=5, prefix='    ')}", 2, state)
            return False

        if os.path.exists(mol_filepath):
            return True
        else:
            _print_verbose(f"  Open Babel conversion failed to create '{os.path.basename(mol_filepath)}'. Output file not found after command.", 1, state)
            return False

    except FileNotFoundError:
        _print_verbose(f"  Error: Open Babel executable '{openbabel_alias}' not found. Please ensure it's installed and in your PATH.", 0, state)
        return False
    except Exception as e:
        _print_verbose(f"  An unexpected error occurred during Open Babel conversion for '{os.path.basename(xyz_filepath)}': {e}", 0, state)
        return False

# 2. Output write
def write_simulation_summary(state: SystemState, output_file_handle, xyz_output_file_display_name: str, rless_filename: str, tvse_filename: str):
    """
    Writes a summary of the simulation parameters to the main out file.
    Includes dynamic filenames and lowest energy info.
    """
    # Construct the elemental composition string
    if hasattr(state, 'element_types') and state.element_types:
        # Determine max length of element symbol for alignment
        max_symbol_len = max(len(state.atomic_number_to_symbol.get(z, f"Unk({z})")) for z, _ in state.element_types)
        element_composition_lines = [
            f"   {state.atomic_number_to_symbol.get(atomic_num, f'Unk({atomic_num})'):<{max_symbol_len}} {count:>3}"
            for atomic_num, count in state.element_types
        ]
    else:
        element_composition_lines = ["  (Elemental composition not available)"]

    # Construct the molecular composition string
    if hasattr(state, 'all_molecule_definitions') and state.all_molecule_definitions and \
       hasattr(state, 'molecules_to_add') and state.molecules_to_add:
        # Determine max length of molecule instance name for alignment
        max_mol_name_len = max(
            len(state.all_molecule_definitions[mol_def_idx].label)
            for mol_def_idx in state.molecules_to_add
            if mol_def_idx < len(state.all_molecule_definitions)
        )
        molecular_composition_lines = []
        for i, mol_def_idx in enumerate(state.molecules_to_add):
            if mol_def_idx < len(state.all_molecule_definitions):
                mol_def = state.all_molecule_definitions[mol_def_idx]
                
                atom_symbols_list = [
                    state.atomic_number_to_symbol.get(atom_data[0], '')
                    for atom_data in mol_def.atoms_coords
                ]
                molecular_formula = get_molecular_formula_string(atom_symbols_list)
                
                molecule_instance_name = mol_def.label # This is the label from the input, e.g., "water1", "water2"
                
                molecular_composition_lines.append(f"  {molecule_instance_name:<{max_mol_name_len}} {molecular_formula}")
            else:
                molecular_composition_lines.append(f"  mol{i+1:<5} (Definition not found for index {mol_def_idx})")
    else:
        molecular_composition_lines = ["  (No molecules specified for addition or 'molecules_to_add' is empty/missing)"]

    # Determine the message for energy evaluation
    energy_eval_message = ""
    output_config_message = ""
    
    if state.random_generate_config == 0: # Random configuration generation mode
        energy_eval_message = "** Energy will not be evaluated **"
        output_config_message = f"Will produce {state.num_random_configs:>2} random configurations"
    elif state.random_generate_config == 1: # Annealing mode
        energy_eval_message = "** Energy will be evaluated **"
        if state.quenching_routine == 1:
            output_config_message = f"Linear quenching route.\n  To = {state.linear_temp_init:.1f} K    dT = {state.linear_temp_decrement:.1f}     nT = {state.linear_num_steps} steps"
        elif state.quenching_routine == 2:
            # Adjusted spacing and added newline
            output_config_message = f"Geometrical quenching route.\n  To = {state.geom_temp_init:.1f} K  %dism = {(1.0 - state.geom_temp_factor) * 100:.1f} %  nT = {state.geom_num_steps} steps"

    # Print the new ASCII art header
    
    # Helper function to center text within 75 characters
    def center_text(text, width=75):
        return text.center(width)
    
    # Write to the file handle
    print("===========================================================================", file=output_file_handle)
    print("", file=output_file_handle)
    print(center_text("*********************"), file=output_file_handle)
    print(center_text("*     A S C E C     *"), file=output_file_handle)
    print(center_text("*********************"), file=output_file_handle)
    print("", file=output_file_handle)
    print("                             √≈≠==≈                                  ", file=output_file_handle)
    print("   √≈≠==≠≈√   √≈≠==≠≈√         ÷++=                      ≠===≠       ", file=output_file_handle)
    print("     ÷++÷       ÷++÷           =++=                     ÷×××××=      ", file=output_file_handle)
    print("     =++=       =++=     ≠===≠ ÷++=      ≠====≠         ÷-÷ ÷-÷      ", file=output_file_handle)
    print("     =++=       =++=    =××÷=≠=÷++=    ≠÷÷÷==÷÷÷≈      ≠××≠ =××=     ", file=output_file_handle)
    print("     =++=       =++=   ≠××=    ÷++=   ≠×+×    ×+÷      ÷+×   ×+××    ", file=output_file_handle)
    print("     =++=       =++=   =+÷     =++=   =+-×÷==÷×-×≠    =×+×÷=÷×+-÷    ", file=output_file_handle)
    print("     ≠×+÷       ÷+×≠   =+÷     =++=   =+---×××××÷×   ≠××÷==×==÷××≠   ", file=output_file_handle)
    print("      =××÷     =××=    ≠××=    ÷++÷   ≠×-×           ÷+×       ×+÷   ", file=output_file_handle)
    print("       ≠=========≠      ≠÷÷÷=≠≠=×+×÷-  ≠======≠≈√  -÷×+×≠     ≠×+×÷- ", file=output_file_handle)
    print("          ≠===≠           ≠==≠  ≠===≠     ≠===≠    ≈====≈     ≈====≈ ", file=output_file_handle)
    print("", file=output_file_handle)
    print("", file=output_file_handle)
    print(center_text("Universidad de Antioquia - Medellín - Colombia"), file=output_file_handle)
    print("", file=output_file_handle)
    print("", file=output_file_handle)
    print(center_text("Annealing Simulado Con Energía Cuántica"), file=output_file_handle)
    print("", file=output_file_handle)
    print(center_text(ASCEC_VERSION), file=output_file_handle)
    print("", file=output_file_handle)
    print(center_text("Química Física Teórica - QFT"), file=output_file_handle)
    print("", file=output_file_handle)
    print("", file=output_file_handle)
    print("===========================================================================", file=output_file_handle)
    print("", file=output_file_handle)
    print("Elemental composition of the system:", file=output_file_handle)
    for line in element_composition_lines:
        print(line, file=output_file_handle)
    print(f"There are a total of {state.natom:>2} nuclei", file=output_file_handle)
    # Changed spacing to one space
    print(f"\nCube's length = {state.cube_length:.2f} A", file=output_file_handle) 
    
    # Write hydrogen bond-aware box analysis to output file
    write_box_analysis_to_file(state, output_file_handle)
    
    if hasattr(state, 'max_molecular_extent') and state.max_molecular_extent > 0:
        print(f"Largest molecular extent: {state.max_molecular_extent:.2f} A", file=output_file_handle) 
    
    print("\nNumber of molecules:", state.num_molecules, file=output_file_handle)
    print("\nMolecular composition", file=output_file_handle)
    for line in molecular_composition_lines:
        print(line, file=output_file_handle)

    # Changed spacing to one space
    print(f"\nMaximum displacement of each mass center = {state.max_displacement_a:.2f} A", file=output_file_handle) 
    print(f"Maximum rotation angle = {state.max_rotation_angle_rad:.2f} radians\n", file=output_file_handle) 
    
    # QM program details - formatted as requested
    qm_program_name = state.qm_program.capitalize() if state.qm_program else "Unknown"
    print(f"Energy calculated with {qm_program_name}", file=output_file_handle)
    print(f" Hamiltonian: {state.qm_method or 'Not specified'}", file=output_file_handle)
    print(f" Basis set: {state.qm_basis_set or 'Not specified'}", file=output_file_handle)
    print(f" Charge = {state.charge}   Multiplicity = {state.multiplicity}", file=output_file_handle)
    
    print(f"\nSeed = {state.random_seed:>6}\n", file=output_file_handle)
    
    print(f"{energy_eval_message}\n", file=output_file_handle)
    print(output_config_message, file=output_file_handle)

    if state.random_generate_config == 0: # Only print this line for random config mode
         print(f"\nCoordinates stored in {xyz_output_file_display_name}", file=output_file_handle)

    # Conditionally print the History header based on simulation mode
    if state.random_generate_config == 1: # Only for annealing mode
        print("\n" + "=" * 60 + "\n", file=output_file_handle) # This separator is only for annealing history
        print("History: [T(K), E(u.a.), n-eval, Criterion]", file=output_file_handle)
        print("", file=output_file_handle) # Blank line after history header
    
    output_file_handle.flush()


# 3. Translated symelem.f (initialize_element_symbols)
def initialize_element_symbols(state: SystemState):
    """
    Initializes a dictionary mapping atomic numbers to element symbols
    and populates the Fortran-style list state.sym.
    """
    state.atomic_number_to_symbol = atomic_number_to_symbol.copy()

    for z, symbol in atomic_number_to_symbol.items():
        if z < len(state.sym):
            state.sym[z] = symbol.strip()

# Helper function to get element symbol (used in rless.out and history output)
def get_element_symbol(atomic_number: int) -> str:
    """Retrieves the element symbol for a given atomic number."""
    return atomic_number_to_symbol.get(atomic_number, 'X') # Default to 'X' for unknown/dummy


def get_molecular_formula(mol_def) -> str:
    """Generate molecular formula from molecule definition."""
    if not mol_def.atoms_coords:
        return "Unknown"
    
    # Count atoms by element
    element_counts = {}
    for atom_data in mol_def.atoms_coords:
        # Handle both 4-field and 5+ field formats (atomic_num, x, y, z, [extra...])
        # mol_def.atoms_coords is already parsed, so we just need the atomic_num
        if len(atom_data) >= 1:
            atomic_num = atom_data[0]
        else:
            continue  # Skip malformed entries
        element = get_element_symbol(atomic_num)
        element_counts[element] = element_counts.get(element, 0) + 1
    
    # Sort elements: C, H, then by electronegativity
    sorted_elements = []
    if 'C' in element_counts:
        sorted_elements.append('C')
    if 'H' in element_counts:
        sorted_elements.append('H')
    
    # Get remaining elements
    remaining_elements = [e for e in element_counts.keys() if e not in ['C', 'H']]
    
    # Sort remaining elements by electronegativity (ascending)
    # Elements not in the dictionary get a high value to be at the end
    remaining_elements.sort(key=lambda e: electronegativity_values.get(e, 1000.0))
    
    sorted_elements.extend(remaining_elements)
    
    # Build formula string
    formula = ""
    for element in sorted_elements:
        count = element_counts[element]
        if count == 1:
            formula += element
        else:
            formula += f"{element}{count}"
    
    return formula if formula else "Unknown"

# 4. Initialize element weights
def initialize_element_weights(state: SystemState):
    """
    Initializes a dictionary mapping atomic numbers to their atomic weights
    and populates the list state.wt. Also populates atomic_number_to_mass.
    """
    state.atomic_number_to_weight = atomic_weights.copy()
    state.atomic_number_to_mass = atomic_weights.copy() # Populate this for COM calculations

    for atomic_num, weight in atomic_weights.items():
        if 0 < atomic_num < len(state.wt):
            state.wt[atomic_num] = weight

# 6. Calculate_mass_center
def calculate_mass_center(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Calculates the center of mass for a set of atoms."""
    total_mass = np.sum(masses)
    if total_mass == 0:
        return np.zeros(3)
    
    return np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass

# 7. Translated draw_molekel.f (draw_molekel) - KEPT FOR REFERENCE, NOT USED DIRECTLY IN CURRENT FLOW
def draw_molekel(natom: int, r_coords: np.ndarray, cycle: int, anneal_flag: int, energy: float, state: SystemState):
    """
    Fortran subroutine draw_molekel: Writes output files for visualization.
    This is a conceptual translation. Actual implementation needs to match Fortran's output format.
    NOTE: This function is currently not directly called in the main loop, as `write_single_xyz_configuration`
    is used instead for more flexible XYZ writing. Kept for reference.
    """
    output_filename = ""
    if anneal_flag == 0: # Random generation mode (mto_ files)
        output_filename = f"mto_{state.jdum}_{cycle:04d}.xyz"
    else: # Annealing mode (result_ files)
        output_filename = f"result_{state.jdum}_{cycle:04d}.xyz"

    full_output_path = os.path.join(state.output_dir, output_filename)

    with open(full_output_path, 'w') as f:
        f.write(f"{natom}\n")
        f.write(f"Energy: {energy:.6f}\n") 
        for i in range(natom):
            atomic_num = state.iznu[i] if hasattr(state, 'iznu') and len(state.iznu) > i else 0
            symbol = state.sym[atomic_num] if atomic_num > 0 and atomic_num < len(state.sym) else "X"
            f.write(f"{symbol} {r_coords[i,0]:10.6f} {r_coords[i,1]:10.6f} {r_coords[i,2]:10.6f}\n")


# 8. Config_molecules
def config_molecules(natom: int, nmo: int, r_coords: np.ndarray, state: SystemState):
    """
    Fortran subroutine config: Configures initial molecular positions and orientations.
    Modifies r_coords in place. Applies PBC.
    Now includes overlap prevention using r_atom during initial random configuration.
    Also dynamically adjusts placement ranges if a molecule is difficult to place,
    mimicking the Fortran's aggressive self-correction for steric clashes.
    """
    current_atom_idx = 0
    final_rp_coords = np.zeros_like(r_coords) 

    if not state.all_molecule_definitions:
        _print_verbose("Error: No molecule definitions found for configuration generation. Cannot proceed.", 0, state)
        return

    placed_atoms_data = [] # Stores (atomic_num, x, y, z) of placed atoms

    # Initialize local scaling factors for translation and rotation ranges.
    # These will increase if a molecule is hard to place without overlaps.
    local_translation_range_factor = 1.0
    local_rotation_range_factor = 1.0
    
    # Define the step for increasing the range (e.g., every 5000 attempts)
    RANGE_INCREASE_STEP = 5000 
    # Initialize the next threshold at which the range should increase
    next_increase_threshold_for_this_molecule = RANGE_INCREASE_STEP

    # Factor by which to increase the range (e.g., 10% increase)
    RANGE_INCREASE_FACTOR = 1.1 
    # Maximum factor for the range (e.g., up to 2 times the original range)
    MAX_RANGE_FACTOR = 2.0 

    for i, mol_def_idx in enumerate(state.molecules_to_add): # Iterate through the actual molecules to add
        mol_def = state.all_molecule_definitions[mol_def_idx] # Get the definition from the list

        overlap_found = True
        attempts = 0
        proposed_mol_atoms = []  # Initialize to prevent unbound variable warning
        
        # Reset local scales and the next increase threshold for each new molecule
        local_translation_range_factor = 1.0
        local_rotation_range_factor = 1.0
        next_increase_threshold_for_this_molecule = RANGE_INCREASE_STEP

        while overlap_found and attempts < state.max_overlap_placement_attempts:
            attempts += 1
            
            # Show progress for difficult placements
            if attempts % 10000 == 0:
                _print_verbose(f"    Molecule {mol_def.label} (instance {i+1}): {attempts} placement attempts...", 1, state)
            
            # Dynamically adjust translation and rotation ranges if placement is difficult.
            # This mimics Fortran's 'ds' adjustment to help break out of steric traps.
            if attempts >= next_increase_threshold_for_this_molecule:
                local_translation_range_factor = min(MAX_RANGE_FACTOR, local_translation_range_factor * RANGE_INCREASE_FACTOR)
                local_rotation_range_factor = min(MAX_RANGE_FACTOR, local_rotation_range_factor * RANGE_INCREASE_FACTOR)
                _print_verbose(f"    Increasing placement ranges for molecule {mol_def.label} (instance {i+1}). Current scales: Trans={local_translation_range_factor:.2f}x, Rot={local_rotation_range_factor:.2f}x", 2, state)
                # Move to the next threshold (e.g., from 5000 to 10000, then to 15000, etc.)
                next_increase_threshold_for_this_molecule += RANGE_INCREASE_STEP 

            # Generate a random translation for the molecule's center, scaled by the dynamic factor
            translation = np.random.uniform(-state.xbox/2 * local_translation_range_factor, state.xbox/2 * local_translation_range_factor, size=3)

            # Generate three random Euler angles (yaw, pitch, roll) in radians, scaled by the dynamic factor
            alpha = np.random.uniform(0, 2 * np.pi * local_rotation_range_factor) 
            beta = np.random.uniform(0, 2 * np.pi * local_rotation_range_factor)  
            gamma = np.random.uniform(0, 2 * np.pi * local_rotation_range_factor) 

            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(gamma), -np.sin(gamma)],
                [0, np.sin(gamma), np.cos(gamma)]
            ])

            Ry = np.array([
                [np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]
            ])

            Rz = np.array([
                [np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1]
            ])

            rotation_matrix = Rz @ Ry @ Rx

            proposed_mol_atoms = [] # (atomic_num, abs_coords_array)
            for atom_data_in_mol in mol_def.atoms_coords: # Corrected variable name here
                atomic_num, x_rel, y_rel, z_rel = atom_data_in_mol # Corrected variable name here
                relative_coords_vector = np.array([x_rel, y_rel, z_rel])
                rotated_coords = np.dot(rotation_matrix, relative_coords_vector)
                abs_coords = rotated_coords + translation
                proposed_mol_atoms.append((atomic_num, abs_coords))

            overlap_found = False
            for prop_atom_num, prop_coords in proposed_mol_atoms:
                for placed_atom_data in placed_atoms_data:
                    placed_atom_num = placed_atom_data[0] 
                    placed_coords = np.array(placed_atom_data[1:]) 

                    distance = np.linalg.norm(prop_coords - placed_coords)

                    radius1 = state.r_atom.get(prop_atom_num, 0.5)
                    radius2 = state.r_atom.get(placed_atom_num, 0.5)

                    min_distance_allowed = (radius1 + radius2) * state.overlap_scale_factor

                    if distance < min_distance_allowed and distance > 1e-4:
                        overlap_found = True
                        # If overlap is found, this molecule's placement attempt fails.
                        # The while loop will continue to retry placing *this same molecule*
                        # with potentially increased ranges until a non-overlapping position is found
                        # or max_overlap_placement_attempts is reached.
                        break 
                if overlap_found:
                    break 
            
        if overlap_found: 
            _print_verbose(f"Warning: Could not find non-overlapping placement for molecule {mol_def.label} (instance {i+1}) after {state.max_overlap_placement_attempts} attempts. Placing it anyway, may cause QM errors.", 1, state)

        for atom_data in proposed_mol_atoms:
            atomic_num, abs_coords = atom_data
            final_rp_coords[current_atom_idx, :] = abs_coords
            placed_atoms_data.append((atomic_num, abs_coords[0], abs_coords[1], abs_coords[2]))
            state.iznu[current_atom_idx] = atomic_num 
            current_atom_idx += 1
    
    state.rp[:] = final_rp_coords
    
    # PERIODIC BOUNDARY CONDITIONS (PBC) - commented out as per Fortran's implied behavior
    # state.rp[:] = state.rp - state.xbox * np.floor(state.rp / state.xbox)

# 9. Define MoleculeData Class for input parsing
class MoleculeData:
    """
    A data structure to hold information for a single molecule parsed from input.
    """
    def __init__(self, label: str, num_atoms: int, atoms_coords: List[Tuple[int, float, float, float]]):
        self.label = label
        self.num_atoms = num_atoms
        self.atoms_coords = atoms_coords

# 10. Initial configuration
def initialize_molecular_coords_in_box(state: SystemState) -> Tuple[np.ndarray, List[int]]:
    """
    Initializes the coordinates (state.rp) for all molecules by placing them
    superimposed, ensuring they are fully contained and centered within the
    simulation box, and assigning their atomic numbers (state.iznu).
    This function creates the *initial* configuration from the template molecule.
    """
    if not hasattr(state, 'coords_per_molecule') or state.coords_per_molecule is None or \
       not hasattr(state, 'atomic_numbers_per_molecule') or not state.atomic_numbers_per_molecule or \
       state.natom_per_molecule == 0:
        _print_verbose("Error: Molecular definitions (coords_per_molecule, atomic_numbers_per_molecule) not loaded from input file.", 0, state)
        _print_verbose("Please ensure the 'Molecule Definition' section in your input file is correct and parsed by read_input_file.", 0, state)
        raise ValueError("Cannot initialize coordinates: Molecular definition is missing or invalid.")

    # Use the already calculated total_atoms from read_input_file which correctly handles different molecule sizes
    total_atoms = state.natom

    rp = np.zeros((total_atoms, 3), dtype=np.float64)
    iznu = [0] * total_atoms

    box_center = state.cube_length / 2.0 * np.ones(3) 

    # Initialize coordinates and atomic numbers for each molecule separately
    # since molecules can have different numbers of atoms
    current_atom_idx = 0
    for mol_idx in range(state.num_molecules):
        mol_def = state.all_molecule_definitions[mol_idx]
        
        # Extract coordinates and atomic numbers for this specific molecule
        coords_list = [[atom[1], atom[2], atom[3]] for atom in mol_def.atoms_coords]
        molecule_coords = np.array(coords_list, dtype=np.float64)
        atomic_numbers = [atom[0] for atom in mol_def.atoms_coords]
        
        # Calculate geometric center for this molecule
        mol_min_coords = np.min(molecule_coords, axis=0)
        mol_max_coords = np.max(molecule_coords, axis=0)
        mol_geometric_center = (mol_min_coords + mol_max_coords) / 2.0
        
        # Center this molecule in the box
        initial_centering_offset = box_center - mol_geometric_center
        centered_coords = molecule_coords + initial_centering_offset
        
        # Place atoms for this molecule
        num_atoms_this_mol = len(atomic_numbers)
        end_idx = current_atom_idx + num_atoms_this_mol
        
        rp[current_atom_idx:end_idx, :] = centered_coords
        iznu[current_atom_idx:end_idx] = atomic_numbers
        
        current_atom_idx = end_idx
    
    # Populate the initial iznu array for the state object
    state.iznu = iznu 
    state.rp = rp

    # Now, randomly translate and rotate molecules from this superimposed state
    # This calls the more general config_molecules that includes overlap checking
    _print_verbose("Configuring molecular positions (this may take a moment for large systems)...", 1, state)
    config_molecules(state.natom, state.num_molecules, state.rp, state)

    return state.rp, state.iznu

# 11. read_input_file
def read_input_file(state: SystemState, source) -> List[MoleculeData]:
    """
    Reads and parses the input. Can read from a file path or a file-like object (like sys.stdin).
    Populates SystemState and returns a list of MoleculeData objects.
    This version uses a two-phase parsing approach for robustness and correctly handles '*' separators.
    """
    molecule_definitions: List[MoleculeData] = []
    
    lines = []
    if isinstance(source, str):
        state.input_file_path = source
        with open(source, 'r') as f_obj:
            lines = f_obj.readlines()
    else:
        state.input_file_path = "stdin_input.in" # Placeholder name for stdin
        lines = source.readlines()

    def clean_line(line: str) -> str:
        """Removes comments and leading/trailing whitespace from a line, including non-breaking spaces."""
        if '#' in line:
            line = line.split('#')[0]
        if '!' in line:
            line = line.split('!')[0]
        line = line.replace('\xa0', ' ')
        return line.strip()

    lines_iterator = iter(lines) 

    # PHASE 1: Read Fixed Configuration Parameters (Lines 1-13)
    config_lines_parsed = 0 

    # We expect 13 lines for fixed configuration parameters (including conformational sampling)
    while config_lines_parsed < 13:
        try:
            raw_line = next(lines_iterator)
        except StopIteration:
            raise EOFError(f"Unexpected end of input while reading configuration parameters. Expected 13 lines, but found only {config_lines_parsed}.")
        
        line_num = lines.index(raw_line) + 1 
        line = clean_line(raw_line)

        if not line: 
            continue

        parts = line.split()
        if not parts: 
            continue

        # Parsing config lines based on count
        if config_lines_parsed == 0: # Line 1: Simulation Mode & Number of Config
            state.random_generate_config = int(parts[0])
            state.num_random_configs = int(parts[1])
            
            if state.random_generate_config == 0: 
                state.ivalE = 0 
                state.mto = state.num_random_configs 
            elif state.random_generate_config == 1: 
                state.ivalE = 1 
                state.mto = 0   

        elif config_lines_parsed == 1: # Line 2: Simulation Cube Length
            state.cube_length = float(parts[0])
            state.xbox = state.cube_length

        elif config_lines_parsed == 2: # Line 3: Annealing Quenching Routine
            state.quenching_routine = int(parts[0])

        elif config_lines_parsed == 3: # Line 4: Linear Quenching Parameters
            state.linear_temp_init = float(parts[0])
            state.linear_temp_decrement = float(parts[1])
            state.linear_num_steps = int(parts[2])

        elif config_lines_parsed == 4: # Line 5: Geometric Quenching Parameters
            state.geom_temp_init = float(parts[0])
            # Parse the geometric factor. If quenching_routine is 2, interpret as percentage decrease.
            if state.quenching_routine == 2:
                # User specified "decrease a 5%". So 5.0 in input means 0.95 factor.
                factor_percentage = float(parts[1])
                if not (0.0 < factor_percentage < 100.0):
                    raise ValueError(f"Error parsing geometric temperature factor on line {line_num}: Expected a percentage between 0 and 100, got '{parts[1]}'.")
                state.geom_temp_factor = 1.0 - (factor_percentage / 100.0)
            else:
                # If not geometric, just read it directly (though it shouldn't be used).
                state.geom_temp_factor = float(parts[1])
            state.geom_num_steps = int(parts[2])

        elif config_lines_parsed == 5: # Line 6: Maximum Monte Carlo Cycles per T and floor value (optional)
            state.max_cycle = int(parts[0])
            # Check if floor value is provided (optional second parameter)
            if len(parts) >= 2:
                state.max_cycle_floor = int(parts[1])
            # If not provided, default value (10) is already set in __init__
            state.maxstep = state.max_cycle # Initialize maxstep with the initial max_cycle from input

        elif config_lines_parsed == 6: # Line 7: Maximum Displacement & Rotation
            state.max_displacement_a = float(parts[0])
            state.max_rotation_angle_rad = float(parts[1])
            state.ds = state.max_displacement_a
            state.dphi = state.max_rotation_angle_rad

        elif config_lines_parsed == 7: # Line 8: Conformational sampling (%) & Maximum dihedral rotation (degrees)
            # Parse conformational move probability as percentage (0-100)
            conformational_percent = float(parts[0])
            if not (0.0 <= conformational_percent <= 100.0):
                raise ValueError(f"Error parsing conformational move percentage on line {line_num}: Expected a value between 0 and 100, got '{parts[0]}'.")
            state.conformational_move_prob = conformational_percent / 100.0  # Convert to 0.0-1.0 range
            
            # Parse maximum dihedral rotation angle in degrees
            max_dihedral_degrees = float(parts[1])
            if not (0.0 <= max_dihedral_degrees <= 180.0):
                raise ValueError(f"Error parsing maximum dihedral angle on line {line_num}: Expected a value between 0 and 180 degrees, got '{parts[1]}'.")
            state.max_dihedral_angle_rad = np.radians(max_dihedral_degrees)  # Convert to radians

        elif config_lines_parsed == 8: # Line 9: QM Program Index & Alias (e.g., "1 g09")
            state.ia = int(parts[0])      
            state.qm_program = parts[1]   
            state.alias = parts[1]        
            state.jdum = state.alias      

        elif config_lines_parsed == 9: # Line 10: Hamiltonian & Basis Set (e.g., "pm3 zdo")
            state.qm_method = parts[0]      
            # Handle case where only method is provided (basis set is optional)
            if len(parts) > 1:
                state.qm_basis_set = parts[1]   
            else:
                # If no basis set provided, leave it as None - respect user's input
                state.qm_basis_set = None   

        elif config_lines_parsed == 10:      # Line 11: nprocs (QM calculations and ASCEC evaluation)
            state.qm_nproc = int(parts[0])   
            
            # Check for ASCEC parallel cores (optional second parameter)
            if len(parts) > 1:
                # User explicitly specified ASCEC cores
                ascec_cores = int(parts[1])
                if ascec_cores > 1:
                    state.ascec_parallel_cores = ascec_cores
                    state.use_ascec_parallel = True
                    _print_verbose(f"ASCEC parallel processing: {state.ascec_parallel_cores} cores (user-specified)", 1, state)
                else:
                    state.ascec_parallel_cores = 1
                    state.use_ascec_parallel = False
                    _print_verbose(f"ASCEC parallel processing disabled (user-specified)", 1, state)
            else:
                # Only one value provided - let ASCEC auto-decide based on available resources
                cpu_count = multiprocessing.cpu_count()
                if state.qm_nproc and state.qm_nproc < cpu_count:
                    # Use remaining cores for ASCEC operations
                    remaining_cores = cpu_count - state.qm_nproc
                    if remaining_cores >= 2:
                        state.ascec_parallel_cores = min(4, remaining_cores)  # Cap at 4 cores for ASCEC
                        state.use_ascec_parallel = True
                        _print_verbose(f"ASCEC parallel processing: {state.ascec_parallel_cores} cores (auto-detected)", 1, state)
                        _print_verbose(f"  (System: {cpu_count} cores, QM: {state.qm_nproc}, ASCEC: {state.ascec_parallel_cores})", 1, state)
                    else:
                        state.ascec_parallel_cores = 1
                        state.use_ascec_parallel = False
                        _print_verbose(f"ASCEC parallel processing disabled (insufficient remaining cores)", 1, state)
                else:
                    # QM uses all or more cores than available - no parallel ASCEC
                    state.ascec_parallel_cores = 1
                    state.use_ascec_parallel = False

        elif config_lines_parsed == 11:     # Line 12: Charge & Spin Multiplicity
            state.charge = int(parts[0])
            state.multiplicity = int(parts[1]) 

        elif config_lines_parsed == 12:     # Line 13: Number of Molecules
            try:
                state.num_molecules = int(parts[0])
            except ValueError:
                raise ValueError(f"Error parsing 'Number of Molecules' on line {line_num}: Expected an integer, but found '{parts[0]}'. "
                                 "Please ensure line 13 of your input file contains the total number of molecules.")
            state.nmo = state.num_molecules
        else:
            _print_verbose(f"Warning: Unexpected configuration line at index {config_lines_parsed}. Line: {line}", 1, state)

        config_lines_parsed += 1 

    # PHASE 2: Read Molecule Definitions
    reading_molecule = False
    current_molecule_num_atoms_expected = 0
    current_molecule_label = ""
    current_molecule_atoms: List[Tuple[int, float, float, float]] = []
    atoms_read_in_current_molecule = 0

    for raw_line in lines_iterator: 
        line_num = lines.index(raw_line) + 1 
        
        # Stop parsing if we hit the Protocol section (old or new format)
        stripped = raw_line.strip()
        if '# Protocol' in raw_line or '# protocol' in raw_line.lower() or is_protocol_marker_line(raw_line):
            break
        
        line = clean_line(raw_line)

        if not line:
            continue

        parts = line.split()

        if parts[0] == "*":
            if reading_molecule: # Found '*' and was reading a molecule -> this '*' closes the previous molecule
                if current_molecule_num_atoms_expected == atoms_read_in_current_molecule:
                    molecule_definitions.append(
                        MoleculeData(current_molecule_label, current_molecule_num_atoms_expected, current_molecule_atoms)
                    )
                else:
                    raise ValueError(f"Error parsing molecule block near line {line_num}: Expected {current_molecule_num_atoms_expected} atoms but read {atoms_read_in_current_molecule} for molecule {current_molecule_label}.")
                
                current_molecule_num_atoms_expected = 0
                current_molecule_label = ""
                current_molecule_atoms = []
                atoms_read_in_current_molecule = 0
                
                reading_molecule = True 
                continue 

            else: # Found '*' and was NOT reading a molecule -> this must be the very first '*' opening the first molecule
                reading_molecule = True
                continue 

        elif reading_molecule: # We are inside a molecule block 
            if current_molecule_num_atoms_expected == 0: 
                try:
                    current_molecule_num_atoms_expected = int(parts[0])
                except ValueError:
                    raise ValueError(f"Error parsing molecule block near line {line_num}: Expected number of atoms, got '{parts[0]}'.")
            elif not current_molecule_label: 
                current_molecule_label = parts[0]
            else: # Expecting atom coordinates
                if atoms_read_in_current_molecule < current_molecule_num_atoms_expected:
                    if len(parts) < 4:
                        raise ValueError(f"Error parsing atom coordinates near line {line_num}: Expected element symbol and 3 coordinates, got '{line}'.")
                    
                    symbol = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    
                    atomic_num = element_symbols.get(symbol)
                    if atomic_num is None:
                        raise ValueError(f"Error: Unknown element symbol '{symbol}' near line {line_num}.")
                    
                    current_molecule_atoms.append((atomic_num, x, y, z))
                    atoms_read_in_current_molecule += 1
                else:
                    raise ValueError(f"Error parsing molecule block near line {line_num}: Read more atoms than expected for molecule {current_molecule_label}. Missing '*' delimiter?")
        else: 
            _print_verbose(f"Warning: Unexpected content '{raw_line.strip()}' found outside of defined blocks near line {line_num}.", 1, state)
            continue 
            
    if reading_molecule and current_molecule_num_atoms_expected > 0: 
        if current_molecule_num_atoms_expected == atoms_read_in_current_molecule:
            molecule_definitions.append(
                MoleculeData(current_molecule_label, current_molecule_num_atoms_expected, current_molecule_atoms)
            )
        else:
            raise ValueError(f"Error: Last molecule block not properly closed or incomplete for molecule {current_molecule_label}.")

    if len(molecule_definitions) != state.num_molecules:
        raise ValueError(f"Error: Number of molecules defined in input ({len(molecule_definitions)}) does not match expected ({state.num_molecules}).")

    state.all_molecule_definitions = molecule_definitions
    # Populate molecules_to_add with indices corresponding to the order they were defined
    # This ensures that each instance (water1, water2, etc.) refers to its specific definition.
    state.molecules_to_add = list(range(state.num_molecules))

    state.natom = sum(mol.num_atoms for mol in molecule_definitions)
    
    if molecule_definitions: 
        # The template molecule for initial placement is still the first one defined
        first_molecule_data = molecule_definitions[0]
        state.natom_per_molecule = first_molecule_data.num_atoms
        state.molecule_label = first_molecule_data.label 

        coords_list = [[atom[1], atom[2], atom[3]] for atom in first_molecule_data.atoms_coords]
        state.coords_per_molecule = np.array(coords_list, dtype=np.float64)

        state.atomic_numbers_per_molecule = [atom[0] for atom in first_molecule_data.atoms_coords]
        
        # Calculate max molecular extent for suggestion
        min_coords_all_mols = np.min(state.coords_per_molecule, axis=0)
        max_coords_all_mols = np.max(state.coords_per_molecule, axis=0)
        max_extent = np.max(max_coords_all_mols - min_coords_all_mols)
        state.max_molecular_extent = max_extent
    else:
        raise ValueError("No molecule definitions found in the input file.")

    state.iznu = [0] * state.natom 
    
    unique_elements = {}
    atom_idx = 0
    for mol_def_idx in state.molecules_to_add: # Iterate through the *indices* of molecules to add
        mol_data = molecule_definitions[mol_def_idx] # Get the definition based on its index
        for atomic_num, _, _, _ in mol_data.atoms_coords:
            state.iznu[atom_idx] = atomic_num
            unique_elements[atomic_num] = unique_elements.get(atomic_num, 0) + 1
            atom_idx += 1

    state.num_elements_defined = len(unique_elements) 
    state.element_types = [(z, unique_elements[z]) for z in sorted(list(unique_elements.keys()))]
    state.iz_types = [z for z, _ in state.element_types] 
    state.nnu_types = [count for _, count in state.element_types] 

    initialize_element_symbols(state)
    initialize_element_weights(state) # This populates atomic_number_to_weight and atomic_number_to_mass

    state.imolec = [0] * (state.num_molecules + 1)
    current_atom_for_imolec = 0
    for i, mol_def_idx in enumerate(state.molecules_to_add): # Iterate through instances to add
        mol_data = molecule_definitions[mol_def_idx] # Get the definition
        state.imolec[i] = current_atom_for_imolec
        current_atom_for_imolec += mol_data.num_atoms
    state.imolec[state.num_molecules] = current_atom_for_imolec

    return molecule_definitions


def extract_protocol_from_input(input_file: str) -> Optional[str]:
    """
    Extract protocol line(s) from input file if present.
    Looks for lines starting with '.asc,' marker, supports multi-line format.
    
    Returns:
        Protocol command string or None if not found
    """
    try:
        with open(input_file, 'r') as f:
            content = f.read()

        protocol_lines: List[str] = []
        in_protocol_section = False

        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if '#' in stripped:
                stripped = stripped.split('#')[0].strip()
                if not stripped:
                    continue
            if is_protocol_marker_line(stripped):
                in_protocol_section = True
                protocol_lines.append(stripped)
                continue
            if in_protocol_section:
                protocol_lines.append(stripped)
                if not stripped.endswith(',') and not stripped.endswith('.'):
                    break

        if protocol_lines:
            protocol = ' '.join(protocol_lines)
            protocol = ' '.join(protocol.split())
            return protocol

    except Exception:
        pass
    
    return None


def extract_embedded_qm_template(input_file: str, template_label: str) -> Optional[Tuple[str, str]]:
    """
    Extract an embedded QM template block by label from an input file.

    Supported headers:
        #orca <label>
        #gaussian <label>

    Returns:
        Tuple (template_content, extension) where extension is ".inp" for ORCA
        and ".com" for Gaussian, or None if no matching block is found.
    """
    if not input_file or not template_label:
        return None

    label_norm = template_label.strip().lower()
    if not label_norm:
        return None

    header_re = re.compile(r'^\s*#\s*(orca|gaussian)\s+(\S+)\s*$', re.IGNORECASE)

    try:
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception:
        return None

    collecting = False
    detected_program = None
    collected: List[str] = []

    for raw_line in lines:
        match = header_re.match(raw_line)
        if match:
            program = match.group(1).lower()
            label = match.group(2).strip().lower()

            if collecting:
                # Reached next template header, stop current block.
                break

            if label == label_norm:
                collecting = True
                detected_program = program
            continue

        if collecting:
            collected.append(raw_line)

    if not collecting or not collected or not detected_program:
        return None

    extension = '.inp' if detected_program == 'orca' else '.com'
    return ''.join(collected).strip() + '\n', extension


def resolve_template_reference(context: 'WorkflowContext', template_token: str) -> Optional[str]:
    """
    Resolve template reference for workflow stages.

    Resolution order:
    1) Existing file path (as provided).
    2) Embedded template label in context.input_file (#orca/#gaussian blocks).

    Returns absolute path to a resolved template file, or None if unresolved.
    """
    if not template_token:
        return None

    token = template_token.strip()
    if not token:
        return None

    # Direct file path resolution first.
    token_abs = os.path.abspath(token)
    if os.path.exists(token_abs):
        return token_abs

    input_file = getattr(context, 'input_file', '')
    extracted = extract_embedded_qm_template(input_file, token)
    if not extracted:
        return None

    template_content, extension = extracted
    safe_label = re.sub(r'[^A-Za-z0-9_.-]+', '_', token)
    out_name = f"{safe_label}{extension}"
    out_path = os.path.abspath(out_name)

    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        return out_path
    except Exception:
        return None


def strip_protocol_from_content(content: str) -> str:
    """Remove the embedded .asc protocol section and everything after it.

    Annealing replicas only need the raw ASCEC configuration parameters
    (box size, temperature, molecules …). The protocol block and any
    embedded QM templates that follow it are not needed and must not be
    copied – otherwise each replica would re-trigger the workflow.
    """
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        # Strip inline comments to get the bare content of this line
        bare = line.strip()
        if '#' in bare:
            bare = bare.split('#', 1)[0].strip()
        # Stop as soon as we hit the embedded protocol marker
        if is_protocol_marker_line(bare):
            # Drop trailing blank lines so the file ends cleanly
            trimmed = lines[:i]
            while trimmed and not trimmed[-1].strip():
                trimmed.pop()
            return ''.join(trimmed)
    return content


def parse_exclusion_pattern(pattern: str) -> List[int]:
    """
    Parse exclusion pattern into list of numbers.
    
    Supported formats:
        "03" -> [3]
        "03,04" -> [3, 4]
        "03-15" -> [3, 4, 5, ..., 15]
        "03,05-15" -> [3, 5, 6, 7, ..., 15]
        "01,03-05,10" -> [1, 3, 4, 5, 10]
    
    Args:
        pattern: Exclusion pattern string
        
    Returns:
        List of numbers to exclude
    """
    excluded = []
    parts = pattern.replace(' ', '').split(',')
    
    for part in parts:
        if '-' in part:
            # Range: 03-15
            start, end = part.split('-')
            start_num = int(start)
            end_num = int(end)
            excluded.extend(range(start_num, end_num + 1))
        else:
            # Single number: 03
            excluded.append(int(part))
    
    return sorted(list(set(excluded)))  # Remove duplicates and sort


def match_exclusion(filename: str, excluded_numbers: List[int]) -> bool:
    """
    Check if a filename matches any excluded number.
    
    Examples:
        "motif_01_opt.inp" matches [1]
        "opt_conf_3.inp" matches [3]
        "motif_15_opt.inp" matches [15]
    
    Args:
        filename: Input filename
        excluded_numbers: List of numbers to exclude
        
    Returns:
        True if filename should be excluded
    """
    # Extract all numbers from filename
    numbers = re.findall(r'\d+', filename)
    
    for num_str in numbers:
        num = int(num_str)
        if num in excluded_numbers:
            return True
    
    return False


def load_protocol_cache(cache_file: str = "protocol_cache.pkl") -> Dict[str, Any]:
    """
    Load protocol cache from pickle file.
    
    Returns:
        Dictionary with cache data or empty dict if file doesn't exist
    """
    if not os.path.exists(cache_file):
        return {}
    
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Failed to load cache file: {e}")
        return {}


def save_protocol_cache(cache_data: Dict[str, Any], cache_file: str = "protocol_cache.pkl"):
    """Save protocol cache to pickle file."""
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"Warning: Failed to save cache file: {e}")


def update_protocol_cache(stage_name: str, status: str, result: Optional[Dict[str, Any]] = None, 
                          cache_file: str = "protocol_cache.pkl"):
    """
    Update protocol cache with stage completion info.
    
    Args:
        stage_name: Name of the stage (e.g., 'r1', 'opt', 'similarity')
        status: Status ('in_progress', 'completed', 'failed')
        result: Optional result dictionary with stage-specific data
        cache_file: Path to cache file
    """
    cache = load_protocol_cache(cache_file)
    
    # Initialize structure if needed
    if 'stages' not in cache:
        cache['stages'] = {}
    if 'start_time' not in cache:
        cache['start_time'] = time.time()
        cache['start_time_str'] = time.strftime('%Y-%m-%d %H:%M:%S')
    if 'protocol_file' not in cache:
        cache['protocol_file'] = cache_file  # Store which cache file this is
    
    # For in_progress: record start time or update existing entry
    if status == 'in_progress':
        # Get existing stage data if it exists
        existing_stage = cache['stages'].get(stage_name, {})
        existing_result = existing_stage.get('result', {})
        
        # Merge new result with existing result
        merged_result = existing_result.copy()
        if result:
            merged_result.update(result)
        
        cache['stages'][stage_name] = {
            'status': status,
            'start_time': existing_stage.get('start_time', time.time()),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'result': merged_result
        }
    else:
        # For completed/failed: calculate wall time
        stage_data = cache['stages'].get(stage_name, {})
        start_time = stage_data.get('start_time', time.time())
        wall_time = time.time() - start_time
        
        cache['stages'][stage_name] = {
            'status': status,
            'start_time': start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'wall_time': wall_time,
            'result': result or {}
        }
    
    cache['last_update'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    save_protocol_cache(cache, cache_file)


def parse_exclude_pattern(pattern: str) -> List[int]:
    """
    Parse exclude pattern like "2,5-9" into list of integers [2, 5, 6, 7, 8, 9].
    
    Args:
        pattern: String like "2,5-9" or "1,3,5" or "10-15"
        
    Returns:
        List of integers to exclude
    """
    numbers = []
    parts = pattern.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range like "5-9"
            try:
                start, end = part.split('-')
                start_num = int(start.strip())
                end_num = int(end.strip())
                numbers.extend(range(start_num, end_num + 1))
            except ValueError:
                print(f"Warning: Invalid range format '{part}', skipping")
        else:
            # Single number
            try:
                numbers.append(int(part))
            except ValueError:
                print(f"Warning: Invalid number '{part}', skipping")
    
    return sorted(set(numbers))  # Remove duplicates and sort


def save_exclude_patterns(input_file: str, optimization_patterns: Dict[str, List[int]],
                         refinement_patterns: Dict[str, List[int]]):
    """
    Save exclude patterns to a file associated with the input file.
    
    Args:
        input_file: Path to the input file
        optimization_patterns: Dictionary of optimization stage patterns {'opt1': [2, 5, 6], 'opt2': [1, 3]}
        refinement_patterns: Dictionary of refinement stage patterns {'ref1': [2, 5, 6]}
    """
    exclude_file = f"{input_file}.exclude"
    
    data = {
        'optimization': optimization_patterns,
        'refinement': refinement_patterns
    }
    
    try:
        with open(exclude_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Warning: Failed to save exclude patterns: {e}")


def load_exclude_patterns(input_file: str) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Load exclude patterns from file associated with the input file.
    
    Args:
        input_file: Path to the input file
        
    Returns:
        Tuple of (optimization_patterns, refinement_patterns)
    """
    exclude_file = f"{input_file}.exclude"
    
    if not os.path.exists(exclude_file):
        return {}, {}
    
    try:
        with open(exclude_file, 'rb') as f:
            data = pickle.load(f)
            optimization_patterns = data.get('optimization', {})
            refinement_patterns = data.get('refinement', {})
            return optimization_patterns, refinement_patterns
    except Exception as e:
        print(f"Warning: Failed to load exclude patterns: {e}")
        return {}, {}


def display_exclude_patterns(optimization_patterns: Dict[str, List[int]], refinement_patterns: Dict[str, List[int]]):
    """Display current exclude patterns."""
    print("\n" + "=" * 60)
    print("Current Exclude Patterns")
    print("=" * 60)
    
    if optimization_patterns:
        print("\nOptimization stages:")
        for stage_name, numbers in sorted(optimization_patterns.items()):
            if numbers:
                print(f"  {stage_name}: {', '.join(map(str, numbers))}")
            else:
                print(f"  {stage_name}: (empty)")
    else:
        print("\nOptimization stages: (none)")
    
    if refinement_patterns:
        print("\nRefinement stages:")
        for stage_name, numbers in sorted(refinement_patterns.items()):
            if numbers:
                print(f"  {stage_name}: {', '.join(map(str, numbers))}")
            else:
                print(f"  {stage_name}: (empty)")
    else:
        print("\nRefinement stages: (none)")
    
    print("=" * 60)


def plot_annealing_diagrams(tvse_file: str, output_dir: str, scaled: bool = False):
    """
    Generate Energy vs Step and Energy vs Temperature diagrams from tvse_*.dat file.
    
    Args:
        tvse_file: Path to tvse_*.dat file
        output_dir: Directory to save the plots
        scaled: If True, apply intelligent y-axis scaling (remove initial high-energy points)
    """
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return False
    
    if not os.path.exists(tvse_file):
        return False
    
    try:
        # Read tvse data file
        # Format: Step Temperature Energy
        steps = []
        temperatures = []
        energies = []
        
        with open(tvse_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        step = int(parts[0])
                        temp = float(parts[1])
                        energy = float(parts[2])
                        steps.append(step)
                        temperatures.append(temp)
                        energies.append(energy)
                    except ValueError:
                        continue
        
        if not steps:
            return False
        
        # Calculate intelligent y-axis scaling if requested
        y_min = None
        y_max = None
        if scaled and energies:
            sorted_energies = sorted(energies)
            
            # Find the minimum (most negative) energy - this is critical!
            y_min = min(energies)
            
            # For y_max: exclude only the top 5% highest energies (initial hot configurations)
            percentile_95_idx = int(len(sorted_energies) * 0.95)
            y_max_candidate = sorted_energies[percentile_95_idx]
            
            # Add small margins for visualization
            energy_range = y_max_candidate - y_min
            y_min = y_min - 0.02 * abs(energy_range)  # 2% margin below minimum
            y_max = y_max_candidate + 0.05 * abs(energy_range)  # 5% margin above 95th percentile
        
        # Create figure with two subplots with more separation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11), gridspec_kw={'hspace': 0.35})
        
        # Plot 1: Energy vs Step Number
        ax1.plot(steps, energies, 'k-', linewidth=0.8, alpha=0.7)
        ax1.set_xlabel('Step Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Energy [Hartrees]', fontsize=12, fontweight='bold')
        ax1.set_title('Simulated Annealing: Energy Evolution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.2, linewidth=0.5)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        # Make secondary axes less prominent
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Apply scaled y-limits if requested
        if y_min is not None and y_max is not None:
            ax1.set_ylim(y_min, y_max)
        
        # Plot 2: Energy vs Temperature
        ax2.scatter(temperatures, energies, c='black', s=10, alpha=0.5, edgecolors='none')
        ax2.set_xlabel('Temperature [K]', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Energy [Hartrees]', fontsize=12, fontweight='bold')
        ax2.set_title('Energy Distribution vs Temperature', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.2, linewidth=0.5)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        # Make secondary axes less prominent
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Apply scaled y-limits to second plot if requested
        if y_min is not None and y_max is not None:
            ax2.set_ylim(y_min, y_max)
        
        # Suppress tight_layout warning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()
        
        # Save figure
        basename = os.path.splitext(os.path.basename(tvse_file))[0]
        output_file = os.path.join(output_dir, f"{basename}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        # Log error for debugging
        if os.environ.get('ASCEC_DEBUG'):
            import sys
            print(f"Warning: Failed to generate diagram for {os.path.basename(tvse_file)}: {e}", file=sys.stderr)
        return False


def plot_combined_replicas_diagram(tvse_files: List[str], output_file: str, num_replicas: int):
    """
    Generate a single combined Energy vs Step diagram for all replicas.
    
    Args:
        tvse_files: List of paths to tvse_*.dat files
        output_file: Output path for the combined diagram
        num_replicas: Number of replicas
    """
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return False
    
    if not tvse_files:
        return False
    
    try:
        # Create single plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Collect all energies first to determine optimal y-axis range
        all_energies = []
        replica_data = []
        
        # Plot each replica and extract replica number from filename
        for idx, tvse_file in enumerate(tvse_files, 1):
            if not os.path.exists(tvse_file):
                continue
            
            steps = []
            energies = []
            
            with open(tvse_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            step = int(parts[0])
                            energy = float(parts[2])
                            steps.append(step)
                            energies.append(energy)
                            all_energies.append(energy)
                        except ValueError:
                            continue
            
            if steps:
                # Extract replica number from path (e.g., "annealing_1" -> 1)
                replica_label = f"ann_{idx}"
                parent_dir = os.path.basename(os.path.dirname(tvse_file))
                # Try to extract number from directory name
                import re
                match = re.search(r'_(\d+)$', parent_dir)
                if match:
                    replica_label = f"ann_{match.group(1)}"
                
                replica_data.append((steps, energies, replica_label))
        
        # Calculate optimal y-axis range by removing only initial high-energy points
        # Keep ALL final energies (they show convergence to minima - very important!)
        y_min = None
        y_max = None
        
        if all_energies:
            sorted_energies = sorted(all_energies)
            
            # Find the minimum (most negative) energy - this is critical!
            y_min = min(all_energies)
            
            # For y_max: exclude only the top 5% highest energies (initial hot configurations)
            # This removes the initial spike without affecting the convergence region
            percentile_95_idx = int(len(sorted_energies) * 0.95)
            y_max_candidate = sorted_energies[percentile_95_idx]
            
            # Add small margins for visualization
            energy_range = y_max_candidate - y_min
            y_min = y_min - 0.02 * abs(energy_range)  # 2% margin below minimum
            y_max = y_max_candidate + 0.05 * abs(energy_range)  # 5% margin above 95th percentile
        
        # Now plot all replica data with labels
        for steps, energies, label in replica_data:
            ax.plot(steps, energies, linewidth=0.8, alpha=0.6, label=label)
        
        # Apply the calculated y-axis range
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel('Step Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Energy [Hartrees]', fontsize=12, fontweight='bold')
        ax.set_title(f'Simulated Annealing: Energy Evolution (r{num_replicas})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)
        # Make secondary axes less prominent
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend at top right without box
        if replica_data:
            ax.legend(loc='upper right', frameon=False, fontsize=10)
        
        # Suppress tight_layout warning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        # Log error for debugging
        if os.environ.get('ASCEC_DEBUG'):
            import sys
            print(f"Warning: Failed to generate combined diagram: {e}", file=sys.stderr)
        return False


def generate_protocol_summary(cache_file: str = "protocol_cache.pkl", 
                              output_file: str = "protocol_summary.txt"):
    """Generate comprehensive protocol summary from cache data with professional formatting."""
    cache = load_protocol_cache(cache_file)
    
    if not cache:
        print("Warning: No cache data found for summary generation")
        return
    
    def format_duration(seconds: float) -> str:
        """Format duration as human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {mins}m {secs}s"
        else:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{days}d {hours}h {mins}m"
    
    def format_wall_time_timing(seconds: float) -> str:
        """Format wall time as H:MM:SS.mmm for timing breakdown."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:06.3f}"
    
    def center_text(text: str, width: int = 75) -> str:
        """Center text within given width."""
        padding = (width - len(text)) // 2
        return " " * padding + text
    
    def _extract_time_from_orca_summary(summary_file: str) -> Optional[float]:
        """Extract total execution time from orca_summary.txt file."""
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
                match = re.search(r'Total execution time:\s+(\d+):(\d+):(\d+\.\d+)', content)
                if match:
                    hours = int(match.group(1))
                    minutes = int(match.group(2))
                    seconds = float(match.group(3))
                    return hours * 3600 + minutes * 60 + seconds
        except:
            pass
        return None

    def _extract_energy_evals_from_annealing(annealing_dir: str) -> Optional[int]:
        """Extract total energy evaluations from annealing.out file."""
        annealing_out = os.path.join(annealing_dir, 'annealing.out')
        if not os.path.exists(annealing_out):
            return None
        try:
            with open(annealing_out, 'r') as f:
                content = f.read()
                # Look for "Energy calculations: XXXX"
                match = re.search(r'Energy calculations:\s+(\d+)', content)
                if match:
                    return int(match.group(1))
        except:
            pass
        return None

    def _extract_final_clusters_from_summary(summary_file: str) -> Optional[int]:
        """Extract final cluster count from clustering_summary.txt."""
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
            match = re.search(r'Total number of final clusters:\s*(\d+)', content, re.IGNORECASE)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return None

    def _resolve_similarity_summary_file(result: Dict[str, Any]) -> Optional[str]:
        """Resolve clustering_summary.txt for a similarity result entry."""
        sim_path = result.get('working_dir') or result.get('similarity_folder')
        if not sim_path:
            return None

        sim_base = sim_path
        base_name = os.path.basename(os.path.normpath(sim_path))
        if base_name.startswith('orca_out_') or base_name.startswith('opt_out_'):
            sim_base = os.path.dirname(sim_path)

        summary_file = os.path.join(sim_base, 'clustering_summary.txt')
        return summary_file if os.path.exists(summary_file) else None
    
    try:
        with open(output_file, 'w') as f:
            # Header
            f.write("=" * 75 + "\n")
            f.write(center_text("A S C E C") + "\n")
            f.write(center_text("Annealing Simulado Con Energía Cuántica") + "\n")
            f.write(center_text("Universidad de Antioquia - QFT") + "\n")
            f.write("=" * 75 + "\n\n")
            
            f.write(center_text("Protocol Workflow Summary") + "\n")
            f.write(center_text("-" * 30) + "\n\n")
            
            # ══════════════════════════════════════════════════════════════════════
            # EXECUTION OVERVIEW
            # ══════════════════════════════════════════════════════════════════════
            f.write("┌" + "─" * 73 + "┐\n")
            f.write("│" + center_text("Execution Overview", 73) + "│\n")
            f.write("└" + "─" * 73 + "┘\n\n")
            
            # Timing info
            if 'start_time_str' in cache:
                f.write(f"  Started:    {cache['start_time_str']}\n")
            if 'last_update' in cache:
                f.write(f"  Completed:  {cache['last_update']}\n")
            
            total_wall_time = 0
            if 'start_time' in cache:
                total_wall_time = time.time() - cache['start_time']
                f.write(f"  Duration:   {format_duration(total_wall_time)}\n")
            
            f.write("\n")
            
            # Workflow diagram
            if 'stages' in cache:
                sorted_stages = sorted(cache['stages'].items(), 
                                     key=lambda x: int(x[0].split('_')[1]) if '_' in x[0] else 0)
                
                completed_stages = []
                for stage_key, stage_info in sorted_stages:
                    if stage_info.get('status') == 'completed':
                        stage_type = stage_key.split('_')[0].capitalize()
                        type_map = {'Replication': 'Annealing', 'Calculation': 'Optimization',
                                  'Similarity': 'Similarity', 'Optimization': 'Optimization',
                                  'Refinement': 'Refinement'}
                        stage_name = type_map.get(stage_type, stage_type)
                        completed_stages.append(stage_name)
                
                f.write(f"  Workflow:   {' → '.join(completed_stages)}\n\n")
            
            # ══════════════════════════════════════════════════════════════════════
            # TIMING BREAKDOWN
            # ══════════════════════════════════════════════════════════════════════
            if 'stages' in cache and total_wall_time > 0:
                f.write("┌" + "─" * 73 + "┐\n")
                f.write("│" + center_text("Timing Breakdown", 73) + "│\n")
                f.write("└" + "─" * 73 + "┘\n\n")
                
                f.write(f"  {'Stage':<15} {'Duration':>15} {'% Total':>10}\n")
                f.write(f"  {'-' * 15} {'-' * 15} {'-' * 10}\n")
                
                sorted_stages = sorted(cache['stages'].items(), 
                                     key=lambda x: int(x[0].split('_')[1]) if '_' in x[0] else 0)
                
                total_qm_percentage = 0.0
                total_tracked_time = 0.0  # Track sum of all stage times
                for stage_key, stage_info in sorted_stages:
                    if stage_info.get('status') == 'completed':
                        stage_type = stage_key.split('_')[0].capitalize()
                        
                        # Skip similarity stages in timing (no QM time)
                        if stage_type == 'Similarity':
                            continue
                        
                        wall_time = stage_info.get('wall_time')
                        if not wall_time:
                            if stage_type == 'Calculation' and os.path.exists('calculation/orca_summary.txt'):
                                wall_time = _extract_time_from_orca_summary('calculation/orca_summary.txt')
                            elif stage_type == 'Optimization' and os.path.exists('optimization/orca_summary.txt'):
                                wall_time = _extract_time_from_orca_summary('optimization/orca_summary.txt')
                        
                        if wall_time:
                            total_tracked_time += wall_time
                            percentage = (wall_time / total_wall_time) * 100
                            total_qm_percentage += percentage
                            type_map = {'Replication': 'Annealing', 'Calculation': 'Optimization',
                                      'Optimization': 'Optimization', 'Refinement': 'Refinement'}
                            stage_name = type_map.get(stage_type, stage_type)
                            f.write(f"  {stage_name:<15} {format_wall_time_timing(wall_time):>15} {percentage:>9.1f}%\n")
                
                # Calculate and show recovery time if there's a significant difference
                recovery_time = total_wall_time - total_tracked_time
                if recovery_time > 60:  # Only show if recovery time > 1 minute
                    recovery_percentage = (recovery_time / total_wall_time) * 100
                    f.write(f"  {'Recovery':<15} {format_wall_time_timing(recovery_time):>15} {recovery_percentage:>9.1f}%\n")
                
                f.write("\n")
            
            # ══════════════════════════════════════════════════════════════════════
            # PROTOCOL DEFINITION
            # ══════════════════════════════════════════════════════════════════════
            if 'protocol_text' in cache:
                f.write("┌" + "─" * 73 + "┐\n")
                f.write("│" + center_text("Protocol Definition", 73) + "│\n")
                f.write("└" + "─" * 73 + "┘\n\n")
                
                protocol_lines = cache['protocol_text'].split(',')
                for line in protocol_lines:
                    line = line.strip()
                    if line:
                        f.write(f"  {line}\n")
                f.write("\n")
            
            # ══════════════════════════════════════════════════════════════════════
            # STAGE DETAILS
            # ══════════════════════════════════════════════════════════════════════
            f.write("┌" + "─" * 73 + "┐\n")
            f.write("│" + center_text("Stage Details", 73) + "│\n")
            f.write("└" + "─" * 73 + "┘\n\n")
            
            if 'stages' in cache:
                sorted_stages = sorted(cache['stages'].items(), 
                                     key=lambda x: int(x[0].split('_')[1]) if '_' in x[0] else 0)
                
                step_num = 1
                total_stages = cache.get('total_stages', len([s for s in sorted_stages if s[1].get('status') == 'completed']))
                
                for stage_key, stage_info in sorted_stages:
                    status = stage_info.get('status', 'unknown')
                    
                    if status != 'completed':
                        continue
                    
                    stage_type = stage_key.split('_')[0].capitalize()
                    type_map = {'Replication': 'Annealing', 'Calculation': 'Optimization',
                              'Similarity': 'Similarity', 'Optimization': 'Optimization',
                              'Refinement': 'Refinement'}
                    stage_name = type_map.get(stage_type, stage_type)
                    
                    result = stage_info.get('result', {})
                    wall_time = stage_info.get('wall_time')
                    
                    # Stage header with status indicator
                    status_icon = "✓"
                    f.write(f"  [{step_num}/{total_stages}] {stage_name} {status_icon}\n")
                    f.write(f"  {'─' * 40}\n")
                    
                    # Stage-specific details
                    if stage_type == 'Replication':  # Annealing
                        if 'box_size' in result:
                            f.write(f"    Box size:         {float(result['box_size']):.1f} Å")
                            if 'packing' in result:
                                f.write(f" ({result['packing']}% packing)")
                            f.write("\n")
                        if 'num_replicas' in result:
                            num_replicas = result['num_replicas']
                            replica_desc = "Duplicated" if num_replicas == 2 else "Triplicated" if num_replicas == 3 else f"{num_replicas}x"
                            f.write(f"    Replicas:         {num_replicas} ({replica_desc})\n")
                        if 'energy_evals' in result:
                            f.write(f"    Evaluations:      {result['energy_evals']} times\n")
                        if 'total_accepted' in result:
                            f.write(f"    Accepted:         {result['total_accepted']} configurations\n")
                    
                    elif stage_type == 'Calculation':
                        if 'xyz_source' in result:
                            xyz_source = result['xyz_source'] if result['xyz_source'] else "Annealing"
                            f.write(f"    Inputs from:      {xyz_source}\n")
                        if 'completed' in result and 'total' in result:
                            f.write(f"    Completed:        {result['completed']}/{result['total']} calculations\n")
                        # Calculate and show mean execution time
                        if wall_time and 'completed' in result and result['completed'] > 0:
                            mean_time = wall_time / result['completed']
                            f.write(f"    Mean exec time:   {format_wall_time_timing(mean_time)}\n")
                        if 'similarity_folder' in result and result['similarity_folder']:
                            f.write(f"    Outputs to:       {result['similarity_folder']}\n")
                    
                    elif stage_type == 'Similarity':
                        live_critical_pct = None
                        live_skipped_pct = None
                        live_critical_count = None
                        live_skipped_count = None
                        live_clusters = None
                        sim_summary_file = _resolve_similarity_summary_file(result)
                        if sim_summary_file:
                            live_critical_pct, live_skipped_pct = parse_similarity_summary(sim_summary_file)
                            sim_base = os.path.dirname(sim_summary_file)
                            live_critical_count, live_skipped_count = parse_similarity_output(sim_base)
                            live_clusters = _extract_final_clusters_from_summary(sim_summary_file)

                        if 'similarity_folder' in result and result['similarity_folder']:
                            f.write(f"    Working dir:      {result['similarity_folder']}\n")
                        if 'threshold' in result:
                            f.write(f"    Threshold:        {result['threshold']}\n")
                        if 'rmsd_threshold' in result:
                            f.write(f"    RMSD:             {result['rmsd_threshold']}\n")
                        else:
                            f.write(f"    RMSD:             N/A\n")
                        f.write("\n")
                        
                        motifs_created = live_clusters if live_clusters is not None else result.get('motifs_created')
                        if motifs_created is not None:
                            motif_label = "Unique Motifs" if ('output_dir' in result and 'umotif' in str(result.get('output_dir', ''))) else "Motifs"
                            label_col = f"    {motif_label}:"
                            f.write(f"{label_col:<22}{motifs_created} representatives\n")
                        
                        # Get threshold info for validation output
                        threshold_met = result.get('threshold_met', True)
                        threshold_type = result.get('threshold_type', 'critical')
                        threshold_value = result.get('threshold_value')
                        attempts = result.get('attempts', 0)
                        
                        # Check if there were redo attempts (initial values differ from final)
                        has_redo = 'initial_critical' in result or 'initial_skipped' in result
                        
                        if has_redo and attempts > 0:
                            # Show Initial validation
                            f.write("\n    Initial validation\n")
                            if 'initial_critical' in result:
                                init_crit = result['initial_critical']
                                init_crit_count = result.get('initial_critical_count', 0)
                                f.write(f"    Critical:         {init_crit}% ({init_crit_count} structures)\n")
                            if 'initial_skipped' in result:
                                init_skip = result['initial_skipped']
                                init_skip_count = result.get('initial_skipped_count', 0)
                                f.write(f"    Skipped:          {init_skip}% ({init_skip_count} structures)\n")
                            
                            if threshold_value is not None:
                                actual_init = result.get(f'initial_{threshold_type}', 'N/A')
                                f.write(f"\n    Target: {threshold_type} ≤ {threshold_value}% | Actual: {actual_init}%\n")
                            
                            # Show Final validation
                            f.write(f"\n    Final validation ({attempts} Redo Attempts)\n")
                            crit_pct = live_critical_pct if live_critical_pct is not None else result.get('critical_pct')
                            crit_count = live_critical_count if live_critical_count is not None else result.get('critical_count', 0)
                            if crit_pct is not None:
                                f.write(f"    Critical:         {crit_pct}% ({crit_count} structures)\n")
                            skip_pct = live_skipped_pct if live_skipped_pct is not None else result.get('skipped_pct')
                            skip_count = live_skipped_count if live_skipped_count is not None else result.get('skipped_count', 0)
                            if skip_pct is not None:
                                f.write(f"    Skipped:          {skip_pct}% ({skip_count} structures)\n")
                            
                            if threshold_value is not None:
                                actual_final = result.get(f'{threshold_type}_pct', 'N/A')
                                f.write(f"\n    Target: {threshold_type} ≤ {threshold_value}% | Actual: {actual_final}%\n")
                            
                            if threshold_met:
                                f.write(f"\n    Validation: Step [{step_num-1}] passed ✓\n")
                            else:
                                f.write(f"\n    Max redo attempts ({attempts}) reached\n")
                        else:
                            # No redo - show single validation
                            f.write("\n")
                            crit_pct = live_critical_pct if live_critical_pct is not None else result.get('critical_pct')
                            crit_count = live_critical_count if live_critical_count is not None else result.get('critical_count', 0)
                            if crit_pct is not None:
                                f.write(f"    Critical:         {crit_pct}% ({crit_count} structures)\n")
                            skip_pct = live_skipped_pct if live_skipped_pct is not None else result.get('skipped_pct')
                            skip_count = live_skipped_count if live_skipped_count is not None else result.get('skipped_count', 0)
                            if skip_pct is not None:
                                f.write(f"    Skipped:          {skip_pct}% ({skip_count} structures)\n")
                            
                            # Threshold validation
                            if threshold_value is not None:
                                actual = result.get(f'{threshold_type}_pct', 'N/A')
                                if threshold_met:
                                    f.write(f"\n    Validation: Step [{step_num-1}] passed ✓\n")
                                    f.write(f"    {threshold_type.capitalize()} ≤ {threshold_value}%\n")
                                else:
                                    f.write(f"\n    Validation: Step [{step_num-1}] threshold exceeded!\n")
                                    f.write(f"    Target: {threshold_type} ≤ {threshold_value}% | Actual: {actual}%\n")
                    
                    elif stage_type == 'Optimization':
                        if 'xyz_source' in result and result['xyz_source']:
                            xyz_source = result['xyz_source'] if result['xyz_source'] else "Annealing"
                            f.write(f"    Inputs from:      {xyz_source}\n")
                        if 'completed' in result and 'total' in result:
                            f.write(f"    Completed:        {result['completed']}/{result['total']} optimizations\n")
                        # Calculate and show mean execution time
                        if wall_time and 'completed' in result and result['completed'] > 0:
                            mean_time = wall_time / result['completed']
                            f.write(f"    Mean exec time:   {format_wall_time_timing(mean_time)}\n")
                        if 'similarity_folder' in result and result['similarity_folder']:
                            f.write(f"    Outputs to:       {result['similarity_folder']}\n")

                    elif stage_type == 'Refinement':
                        if 'motifs_source' in result and result['motifs_source']:
                            f.write(f"    Inputs from:      {result['motifs_source']}\n")
                        if 'completed' in result and 'total' in result:
                            f.write(f"    Completed:        {result['completed']}/{result['total']} refinements\n")
                        # Calculate and show mean execution time
                        if wall_time and 'completed' in result and result['completed'] > 0:
                            mean_time = wall_time / result['completed']
                            f.write(f"    Mean exec time:   {format_wall_time_timing(mean_time)}\n")
                        if 'similarity_folder' in result and result['similarity_folder']:
                            f.write(f"    Outputs to:       {result['similarity_folder']}\n")

                    # Wall time for non-similarity stages
                    if wall_time and stage_type != 'Similarity':
                        f.write(f"    Wall time:        {format_wall_time_timing(wall_time)}\n")
                    
                    f.write("\n")
                    step_num += 1
            
            # ══════════════════════════════════════════════════════════════════════
            # FOOTER
            # ══════════════════════════════════════════════════════════════════════
            f.write("=" * 75 + "\n")
            f.write(center_text("Workflow completed successfully") + "\n")
            f.write("=" * 75 + "\n")
        
        print(f"✓ Protocol summary saved to {output_file}")
        
    except Exception as e:
        print(f"Warning: Failed to generate protocol summary: {e}")


# 12. QM Program Details Mapping
qm_program_details = {
    1: {
        "name": "gaussian",
        "default_exe": "g09", # Common executable name.
        "input_ext": ".gjf",
        "output_ext": ".log",
        "energy_regex": r"SCF Done:\s+E\(.+\)\s*=\s*([-\d.]+)\s+A\.U\.",
        "termination_string": "Normal termination",
    },
    2: {
        "name": "orca",
        "default_exe": "orca", # Common executable name.
        "input_ext": ".inp",
        "output_ext": ".out",
        "energy_regex": r"FINAL SINGLE POINT ENERGY\s*:?\s*([-+]?\d+\.\d+)",  # Works for ORCA 5 and 6 (with or without colon)
        "termination_string": "ORCA TERMINATED NORMALLY",
        "alternative_termination": ["****ORCA TERMINATED NORMALLY****", "OPTIMIZATION RUN DONE"],  # Additional termination patterns
    },
}

# Helper function to preserve QM files from last accepted configuration
def preserve_last_qm_files(state: SystemState, run_dir: str, call_id: Optional[int] = None, for_debug: bool = False):
    """
    Preserve the QM input and output files from the most recent QM calculation
    by copying them to special "last_accepted" filenames for debugging purposes.
    
    Args:
        state: SystemState object
        run_dir: Run directory
        call_id: Optional specific call ID to preserve (defaults to state.qm_call_count)
        for_debug: If True, adds "for debugging" to verbose messages
    """
    if call_id is None:
        call_id = state.qm_call_count
    
    # Source files (from the most recent calculation)
    source_input = os.path.join(run_dir, f"qm_input_{call_id}{qm_program_details[state.ia]['input_ext']}")
    source_output = os.path.join(run_dir, f"qm_output_{call_id}{qm_program_details[state.ia]['output_ext']}")
    source_chk = os.path.join(run_dir, f"qm_chk_{call_id}.chk") if state.qm_program == "gaussian" else None
    
    # Destination files (preserved versions)
    dest_input = os.path.join(run_dir, f"anneal{qm_program_details[state.ia]['input_ext']}")
    dest_output = os.path.join(run_dir, f"anneal{qm_program_details[state.ia]['output_ext']}")
    dest_chk = os.path.join(run_dir, "anneal.chk") if state.qm_program == "gaussian" else None
    
    # Set message suffix based on debug flag
    msg_suffix = " for debugging" if for_debug else ""
    
    # Copy files if they exist
    try:
        if os.path.exists(source_input):
            shutil.copyfile(source_input, dest_input)
            _print_verbose(f"  Preserved QM input file{msg_suffix}: {os.path.basename(dest_input)}", 2, state)
        
        if os.path.exists(source_output):
            shutil.copyfile(source_output, dest_output)
            _print_verbose(f"  Preserved QM output file{msg_suffix}: {os.path.basename(dest_output)}", 2, state)
            
        if source_chk and dest_chk and os.path.exists(source_chk):
            shutil.copyfile(source_chk, dest_chk)
            _print_verbose(f"  Preserved QM checkpoint file{msg_suffix}: {os.path.basename(dest_chk)}", 2, state)
    except Exception as e:
        _print_verbose(f"  Warning: Could not preserve QM files{msg_suffix}: {e}", 1, state)

# Helper function to preserve QM files for debugging (last attempt, successful or failed)
def preserve_last_qm_files_debug(state: SystemState, run_dir: str, call_id: int, status: int):
    """
    Preserve the QM input and output files from the most recent calculation attempt
    for debugging purposes. This preserves files from both successful and failed calculations.
    
    Note: This is a wrapper for preserve_last_qm_files() with for_debug=True for backward compatibility.
    The status parameter is kept for API compatibility but not used.
    """
    mode = str(getattr(state, 'qm_attempt_debug_mode', 'first_success')).strip().lower()

    if mode == 'none':
        return

    if mode == 'first_success' and getattr(state, '_first_success_qm_preserved', False):
        return

    preserve_last_qm_files(state, run_dir, call_id=call_id, for_debug=True)

    # Mark completion once we have preserved the first successful attempt.
    if mode == 'first_success' and status == 1:
        state._first_success_qm_preserved = True

def preserve_failed_initial_qm_files(state: SystemState, run_dir: str, attempt_num: int):
    """
    Simply notify that the anneal.* files contain the last failed attempt.
    The files are already preserved by preserve_last_qm_files_debug.
    """
    _print_verbose(f"", 0, state)

    
    input_file = os.path.join(run_dir, "anneal.inp" if state.qm_program == "orca" else "anneal.com")
    output_file = os.path.join(run_dir, "anneal.out")
    
    if os.path.exists(input_file):
        _print_verbose(f"  ✓ {os.path.basename(input_file)} (QM input file - check this for problematic geometry)", 0, state)
    else:
        _print_verbose(f"  ✗ {os.path.basename(input_file)} (NOT FOUND)", 0, state)
        
    if os.path.exists(output_file):
        _print_verbose(f"  ✓ anneal.out (QM output file - check for error messages)", 0, state)
    else:
        _print_verbose(f"  ✗ anneal.out (NOT FOUND - QM program failed to start or crashed immediately)", 0, state)
        _print_verbose(f"      This usually indicates:", 0, state)
        _print_verbose(f"      - QM program not found in PATH", 0, state)
        _print_verbose(f"      - Severe geometry problems (atoms too close)", 0, state)
        _print_verbose(f"      - Invalid basis set for these atoms", 0, state)
        _print_verbose(f"      - Memory/resource issues", 0, state)

# 13. Calculate energy function
def calculate_energy(coords: np.ndarray, atomic_numbers: List[int], state: SystemState, run_dir: str) -> Tuple[float, int]:
    """
    Calculates the energy of the given configuration using the external QM program.
    Returns the energy and a status code (1 for success, 0 for failure).
    Cleans up QM input/output/checkpoint files immediately after execution.
    Now includes optimizations for parallel core usage.
    """
    # Optimize the execution environment for better core utilization
    optimize_qm_execution_environment(state)
    
    state.qm_call_count += 1 # Increment total QM calls
    call_id = state.qm_call_count
    
    qm_input_filename = f"qm_input_{call_id}{qm_program_details[state.ia]['input_ext']}"
    qm_output_filename = f"qm_output_{call_id}{qm_program_details[state.ia]['output_ext']}"
    
    # Checkpoint file naming convention, might vary per program (e.g., Gaussian uses .chk)
    qm_chk_filename = f"qm_chk_{call_id}.chk" if state.qm_program == "gaussian" else None

    qm_input_path = os.path.join(run_dir, qm_input_filename)
    qm_output_path = os.path.join(run_dir, qm_output_filename)
    qm_chk_path = os.path.join(run_dir, qm_chk_filename) if qm_chk_filename else None

    energy = 0.0
    status = 0 # 0 for failure, 1 for success
    
    temp_files_to_clean = [qm_input_path, qm_output_path]
    if qm_chk_path:
        temp_files_to_clean.append(qm_chk_path)

    try:
        # Generate QM input file
        with open(qm_input_path, 'w') as f:
            if state.qm_program == "gaussian":
                if qm_chk_path: f.write(f"%chk={os.path.basename(qm_chk_path)}\n") # Only filename for %chk
                # Only write %mem if qm_memory was explicitly provided in the input file
                if state.qm_memory: f.write(f"%mem={state.qm_memory}\n")
                if state.qm_nproc: f.write(f"%nproc={state.qm_nproc}\n")
                
                # Build the route line - only include basis set if provided
                if state.qm_basis_set:
                    f.write(f"# {state.qm_method} {state.qm_basis_set}\n\n")
                else:
                    f.write(f"# {state.qm_method}\n\n")
                    
                f.write("ASCEC QM Calculation\n\n")
                f.write(f"{state.charge} {state.multiplicity}\n")
                for i in range(state.natom):
                    symbol = state.atomic_number_to_symbol.get(atomic_numbers[i], "X")
                    f.write(f"{symbol} {coords[i, 0]:.6f} {coords[i, 1]:.6f} {coords[i, 2]:.6f}\n")
                f.write("\n") 
                if state.qm_additional_keywords: f.write(f"{state.qm_additional_keywords}\n")
                f.write("\n") 
            elif state.qm_program == "orca":
                # ORCA keyword line generation - respect user's exact input
                # Build keyword line with method and basis set if provided
                if state.qm_basis_set:
                    f.write(f"! {state.qm_method} {state.qm_basis_set}\n")
                else:
                    f.write(f"! {state.qm_method}\n")
                
                if state.qm_additional_keywords: f.write(f"! {state.qm_additional_keywords}\n")
                
                # Only write %maxcore if qm_memory was explicitly provided in the input file
                if state.qm_memory:
                    mem_val = state.qm_memory.replace('GB', '').replace('MB', '')
                    f.write(f"%maxcore {mem_val}\n") 
                
                # Parallel processing settings
                # Note: Some methods (like semi-empirical NDO methods) don't support parallel execution
                if state.qm_nproc:
                    f.write(f"%pal nprocs {state.qm_nproc} end\n")
                
                f.write(f"* xyz {state.charge} {state.multiplicity}\n")
                for i in range(state.natom):
                    symbol = state.atomic_number_to_symbol.get(atomic_numbers[i], "X")
                    f.write(f"{symbol} {coords[i, 0]:.6f} {coords[i, 1]:.6f} {coords[i, 2]:.6f}\n")
                f.write("*\n")
            else:
                raise ValueError(f"QM input generation not implemented for program '{state.qm_program}'")
    except IOError as e:
        _print_verbose(f"Error writing QM input file {qm_input_path}: {e}", 0, state)
        return 0.0, 0 
    except ValueError as e:
        _print_verbose(f"Error in QM input generation: {e}", 0, state)
        return 0.0, 0

    # Determine QM command - use alias from input file if provided, otherwise use default
    qm_exe = state.alias if state.alias else qm_program_details[state.ia]["default_exe"]
    if state.qm_program == "gaussian":
        qm_command = f"{qm_exe} < {qm_input_filename} > {qm_output_filename}"
    elif state.qm_program == "orca":
        # For ORCA, we'll use subprocess to capture output properly
        qm_command = [qm_exe, qm_input_filename]
    else:
        _print_verbose(f"Error: Unsupported QM program '{state.qm_program}' for command execution.", 0, state)
        return 0.0, 0

    try:
        # Check ORCA executable only once per run to avoid per-step overhead.
        if state.qm_program == "orca" and not getattr(state, "_orca_exe_checked", False):
            try:
                subprocess.run([qm_exe, "--version"], capture_output=True, text=True)
                _print_verbose(f"ORCA executable check passed: {qm_exe}", 2, state)
            except FileNotFoundError:
                _print_verbose(f"Error: ORCA executable '{qm_exe}' not found in PATH", 0, state)
                return 0.0, 0
            except Exception as e:
                _print_verbose(f"Warning: ORCA executable test failed: {e} (proceeding anyway)", 1, state)
            finally:
                state._orca_exe_checked = True
        
        if state.qm_program == "orca":
            # Special handling for ORCA to capture output properly
            _print_verbose(f"Executing ORCA command: {' '.join(qm_command)} in directory: {run_dir}", 2, state)
            with open(qm_output_path, 'w') as output_file:
                process = subprocess.run(
                    qm_command,
                    cwd=run_dir,
                    stdout=output_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False
                )
            _print_verbose(f"ORCA process completed with return code: {process.returncode}", 2, state)
        else:
            # For Gaussian and other programs
            process = subprocess.run(qm_command, shell=True, capture_output=True, text=True, cwd=run_dir, check=False)
        
        # Check for non-zero exit code first
        # NOTE: ORCA 6 (and some ORCA 5 configurations) may return non-zero exit codes
        # even on successful completion. We need to check the output file content regardless.
        non_zero_exit = process.returncode != 0
        should_check_output = False  # Flag to determine if we should parse the output file
        
        if non_zero_exit:
            _print_verbose(f"'{state.qm_program}' exited with non-zero status: {process.returncode}.", 2, state)
            if state.qm_program == "orca":
                _print_verbose(f"  Command executed: {' '.join(qm_command)}", 2, state)
                _print_verbose(f"  Working directory: {run_dir}", 2, state)
                # ORCA can return non-zero even on success, continue to check output file
                should_check_output = True
            else:
                # For non-ORCA programs, non-zero exit is a failure
                _print_verbose(f"  Command executed: {qm_command}", 0, state)
                _print_verbose(f"  STDOUT (first 10 lines):\n{_format_stream_output(process.stdout)}", 0, state)
                _print_verbose(f"  STDERR (first 10 lines):\n{_format_stream_output(process.stderr)}", 0, state)
                status = 0
        else:
            # Exit code was 0, we should check the output file
            should_check_output = True
        
        # Parse output file if appropriate
        if should_check_output:
            if not os.path.exists(qm_output_path):
                _print_verbose(f"QM output file '{qm_output_path}' was not generated.", 1, state)
                status = 0 
            else:
                with open(qm_output_path, 'r') as f:
                    output_content = f.read()
                
                # Check for normal termination string
                termination_found = qm_program_details[state.ia]["termination_string"] in output_content
                
                # For programs with alternative termination patterns, check those too
                if not termination_found and "alternative_termination" in qm_program_details[state.ia]:
                    for alt_term in qm_program_details[state.ia]["alternative_termination"]:
                        if alt_term in output_content:
                            termination_found = True
                            break
                
                if not termination_found:
                    _print_verbose(f"QM program '{state.qm_program}' did not terminate normally for config {call_id}.", 1, state)
                    status = 0
                else:
                    match = re.search(qm_program_details[state.ia]["energy_regex"], output_content)
                    if match:
                        energy = float(match.group(1))
                        status = 1
                    else:
                        # For ORCA, try alternative energy patterns as fallback
                        if state.qm_program == "orca":
                            # Try alternative patterns commonly found in ORCA output (5.x and 6.x)
                            fallback_patterns = [
                                r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+)",  # ORCA 5/6 format (no colon)
                                r"Total Energy\s*:\s*([-+]?\d+\.\d+)\s*Eh",
                                r"E\(SCF\)\s*=\s*([-+]?\d+\.\d+)\s*Eh",
                                r"TOTAL SCF ENERGY\s*=\s*([-+]?\d+\.\d+)\s*Eh?"
                            ]
                            for pattern in fallback_patterns:
                                match = re.search(pattern, output_content)
                                if match:
                                    energy = float(match.group(1))
                                    status = 1
                                    _print_verbose(f"Found energy using fallback pattern: {pattern}", 2, state)
                                    break
                        
                        if status == 0:
                            _print_verbose(f"Could not find energy in {state.qm_program} output file: {qm_output_path}", 1, state)
    
    except Exception as e:
        _print_verbose(f"An error occurred during QM calculation or parsing: {e}", 0, state)
        status = 0
    finally:
        # Always preserve the last QM files for debugging (both successful and failed attempts)
        preserve_last_qm_files_debug(state, run_dir, call_id, status)
        
        # For ORCA, discover and add all auxiliary files to cleanup list AFTER calculation completes
        if state.qm_program == "orca":
            input_basename = f"qm_input_{call_id}"
            orca_pattern = os.path.join(run_dir, f"{input_basename}*")
            for orca_file in glob.glob(orca_pattern):
                # Skip the main input and output files (already in temp_files_to_clean)
                if orca_file not in temp_files_to_clean and os.path.isfile(orca_file):
                    temp_files_to_clean.append(orca_file)
        
        # Clean up numbered QM files but keep the "anneal.*" versions for debugging
        for fpath in temp_files_to_clean:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError as e:
                    _print_verbose(f"  Error cleaning up {os.path.basename(fpath)}: {e}", 0, state)
    return energy, status

# 13a. Enhanced QM execution with parallel core optimization
def optimize_qm_execution_environment(state: SystemState):
    """
    Optimize the execution environment for QM calculations to make better use of available cores.
    This function sets environment variables that help QM programs utilize cores more efficiently.
    """
    if not state.use_ascec_parallel:
        return
    
    # Set environment variables for better parallel performance
    os.environ['OMP_NUM_THREADS'] = str(state.qm_nproc or 1)
    os.environ['MKL_NUM_THREADS'] = str(state.qm_nproc or 1)
    
    # For ORCA specifically
    if state.qm_program == "orca":
        os.environ['RSH_COMMAND'] = 'ssh -x'
        # Optimize ORCA's parallel execution
        if state.ascec_parallel_cores > 1:
            _print_verbose(f"Optimized environment for ORCA with {state.qm_nproc} QM cores and {state.ascec_parallel_cores} system cores", 2, state)
    
    # For Gaussian specifically  
    elif state.qm_program == "gaussian":
        if state.ascec_parallel_cores > 1:
            _print_verbose(f"Optimized environment for Gaussian with {state.qm_nproc} QM cores and {state.ascec_parallel_cores} system cores", 2, state)

def parallel_coordinate_operations(coords: np.ndarray, state: SystemState) -> np.ndarray:
    """
    Perform coordinate transformations using parallel processing when beneficial.
    This can speed up large system coordinate manipulations.
    """
    if not state.use_ascec_parallel or coords.shape[0] < 100:
        # For small systems, parallel overhead isn't worth it
        return coords
    
    # For large systems, we could implement parallel coordinate transformations
    # This is most beneficial for systems with hundreds of atoms
    _print_verbose(f"Using parallel coordinate operations for {coords.shape[0]} atoms", 2, state)
    return coords

def parallel_file_operations(file_path: str, content: str, state: SystemState) -> bool:
    """
    Handle file I/O operations more efficiently using parallel processing capabilities.
    This mainly helps with large file writes and reads.
    """
    if not state.use_ascec_parallel:
        # Fall back to standard file operations
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        except IOError:
            return False
    
    # For parallel-enabled systems, we can use more efficient I/O
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    except IOError:
        return False

# Helper function to format and limit stream output (stdout/stderr)
def _format_stream_output(stream_content, max_lines=10, prefix="  "):
    if not stream_content:
        return "[No output]"
    
    lines = stream_content.splitlines()
    output_str = ""
    for i, line in enumerate(lines):
        if i < max_lines:
            output_str += f"{prefix}{line}\n"
        else:
            output_str += f"{prefix}... ({len(lines) - max_lines} more lines)\n"
            break
    return output_str

# 14. xyz confifuration 
def write_single_xyz_configuration(file_handle, natom, rp, iznu, energy, config_idx, 
                                   atomic_number_to_symbol, random_generate_config_mode, 
                                   remark="", include_dummy_atoms=False, state=None): 
    """
    Writes a single XYZ configuration to a file handle.
    Optionally includes dummy 'X' atoms for the box corners.
    The 'remark' argument provides an optional string for the comment line.
    The output format of the second line changes based on 'random_generate_config_mode'.
    
    Args:
        file_handle (io.TextIOWrapper): The open file handle to write to.
        natom (int): Total number of actual atoms in the configuration.
        rp (np.ndarray): (N, 3) array of atomic coordinates.
        iznu (List[int]): List of atomic numbers.
        energy (float): Energy of the configuration.
        config_idx (int): The index/number of the current configuration.
        atomic_number_to_symbol (dict): Mapping from atomic number to symbol.
        random_generate_config_mode (int): The simulation mode (0 for random, 1 for annealing).
        remark (str, optional): An additional remark for the comment line.
        include_dummy_atoms (bool): If True, adds 8 dummy 'X' atoms for box visualization.
        state (SystemState, optional): The SystemState object, REQUIRED if include_dummy_atoms is True
                                       to get the 'xbox' (box dimensions).
    """
    if include_dummy_atoms and (state is None or not hasattr(state, 'xbox')):
        raise ValueError("SystemState object with 'xbox' attribute must be provided to "
                         "write_single_xyz_configuration when include_dummy_atoms is True "
                         "(to get box dimensions).")
    
    L = state.xbox if include_dummy_atoms and state else 0.0 
    half_L = L / 2.0  # Box is centered at origin

    # Box corners centered at origin (matches web generator)
    box_corners = np.array([
        [-half_L, -half_L, -half_L], [ half_L, -half_L, -half_L],
        [-half_L,  half_L, -half_L], [ half_L,  half_L, -half_L],
        [-half_L, -half_L,  half_L], [ half_L, -half_L,  half_L],
        [-half_L,  half_L,  half_L], [ half_L,  half_L,  half_L]
    ])
    dummy_atom_count = len(box_corners) 

    total_atoms_in_frame = natom
    if include_dummy_atoms:
        total_atoms_in_frame += dummy_atom_count

    file_handle.write(f"{total_atoms_in_frame}\n")
    
    # Updated comment line format with | separators and fixed BoxL format
    base_comment_line = f"Configuration: {config_idx} | E = {energy:.8f} a.u."
    if include_dummy_atoms:
        if L > 0: 
            base_comment_line += f" | BoxL={L:.1f} A ({dummy_atom_symbol} box markers)" 
    elif random_generate_config_mode == 1 and state and hasattr(state, 'current_temp'): # Only add T for annealing mode, not for random mode
        base_comment_line += f" | T = {state.current_temp:.1f} K" 

    file_handle.write(f"{base_comment_line}\n")
    
    for i in range(natom):
        symbol = atomic_number_to_symbol.get(iznu[i], 'X') 
        file_handle.write(f"{symbol: <3} {rp[i, 0]: 12.6f} {rp[i, 1]: 12.6f} {rp[i, 2]: 12.6f}\n")

    if include_dummy_atoms:
        for coords in box_corners:
            file_handle.write(f"{dummy_atom_symbol: <3} {coords[0]: 12.6f} {coords[1]: 12.6f} {coords[2]: 12.6f}\n")

    # The caller opens files in append mode per write; explicit flush here adds avoidable I/O overhead.

# New wrapper function for writing accepted XYZ configurations
def write_accepted_xyz(prefix: str, config_number: int, energy: float, temp: float, state: SystemState, is_initial: bool = False):
    """
    Writes accepted XYZ configurations (both normal and with box) to files.
    This wraps write_single_xyz_configuration.
    
    Args:
        prefix (str): Base filename prefix (e.g., "result_123456").
        config_number (int): The sequential number for this accepted configuration.
        energy (float): The energy of the configuration.
        temp (float): The temperature at which the configuration was accepted.
        state (SystemState): The current SystemState object.
        is_initial (bool): True if this is the very first accepted configuration.
    """
    xyz_path = os.path.join(state.output_dir, f"{prefix}.xyz")
    box_xyz_path = os.path.join(state.output_dir, f"{prefix.replace('mto_', 'mtobox_').replace('result_', 'resultbox_')}.xyz") # Corrected replacement for box name

    remark = "" # Removed specific remarks, now handled by write_single_xyz_configuration's internal logic

    # Write to the ORIGINAL XYZ file handle (no dummy atoms)
    with open(xyz_path, 'a') as f_xyz:
        write_single_xyz_configuration(
            f_xyz,
            state.natom,
            state.rp,  # Coords already centered at origin, no shift needed
            state.iznu,
            energy, 
            config_number, # Use the sequential config_number here
            state.atomic_number_to_symbol,
            state.random_generate_config, 
            remark=remark, # Empty remark as formatting is handled internally
            include_dummy_atoms=False, 
            state=state 
        )                
    
    # Conditionally write to the BOX XYZ COPY file handle (with dummy atoms)
    if create_box_xyz_copy:
        with open(box_xyz_path, 'a') as f_box_xyz:
            write_single_xyz_configuration(
                f_box_xyz,
                state.natom,
                state.rp,  # Coords already centered at origin, no shift needed
                state.iznu,
                energy, # Energy is 0.0 as it's not evaluated
                config_number, # Use the sequential config_number here
                state.atomic_number_to_symbol,
                state.random_generate_config, 
                remark=remark, # Empty remark as formatting is handled internally
                include_dummy_atoms=True, 
                state=state 
            )

# 14.5. Dihedral Rotation Functions for Conformational Sampling
def rotate_around_bond(coords: np.ndarray, atom1_idx: int, atom2_idx: int, 
                      moving_atoms: List[int], angle_rad: float) -> np.ndarray:
    """
    Rotates a set of atoms around a bond defined by atom1_idx and atom2_idx.
    
    Args:
        coords: Atomic coordinates array (N x 3)
        atom1_idx: Index of first atom defining the rotation axis
        atom2_idx: Index of second atom defining the rotation axis  
        moving_atoms: List of atom indices that will be rotated
        angle_rad: Rotation angle in radians
    
    Returns:
        Modified coordinates array
    """
    new_coords = np.copy(coords)
    
    # Define rotation axis vector
    axis_vector = coords[atom2_idx] - coords[atom1_idx]
    axis_length = np.linalg.norm(axis_vector)
    
    if axis_length < 1e-6:
        # Atoms are too close, skip rotation
        return new_coords
    
    # Normalize axis vector
    axis_unit = axis_vector / axis_length
    
    # Rotation point (we'll use atom1 as the rotation center)
    rotation_center = coords[atom1_idx]
    
    # Rodrigues' rotation formula
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    for atom_idx in moving_atoms:
        if atom_idx == atom1_idx or atom_idx == atom2_idx:
            continue  # Don't rotate the atoms defining the axis
            
        # Vector from rotation center to atom
        point_vector = coords[atom_idx] - rotation_center
        
        # Apply Rodrigues' rotation formula
        rotated_vector = (point_vector * cos_angle + 
                         np.cross(axis_unit, point_vector) * sin_angle +
                         axis_unit * np.dot(axis_unit, point_vector) * (1 - cos_angle))
        
        new_coords[atom_idx] = rotation_center + rotated_vector
    
    return new_coords

def find_rotatable_bonds(mol_coords: np.ndarray, mol_atomic_numbers: List[int], 
                        state: SystemState) -> List[Tuple[int, int, List[int]]]:
    """
    Identifies rotatable bonds in a molecule and determines which atoms move with each rotation.
    
    Returns:
        List of tuples: (atom1_idx, atom2_idx, [moving_atom_indices])
    """
    rotatable_bonds = []
    n_atoms = len(mol_atomic_numbers)
    
    # Simple bond detection based on distance (could be improved with connectivity)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = np.linalg.norm(mol_coords[j] - mol_coords[i])
            
            # Get atomic radii for bond length estimation
            radius_i = state.r_atom.get(mol_atomic_numbers[i], 1.5)
            radius_j = state.r_atom.get(mol_atomic_numbers[j], 1.5)
            max_bond_length = (radius_i + radius_j) * 1.3  # 30% tolerance
            
            if distance < max_bond_length:
                # This looks like a bond
                # For single bonds (excluding terminal bonds), consider as rotatable
                if (mol_atomic_numbers[i] != 1 and mol_atomic_numbers[j] != 1):  # Not H-X bonds
                    # Find which atoms would move if we rotate around this bond
                    # Simple approach: atoms connected to atom j (and beyond) move
                    moving_atoms = find_connected_atoms(j, i, mol_coords, mol_atomic_numbers, state)
                    
                    if len(moving_atoms) > 0 and len(moving_atoms) < n_atoms - 2:
                        # Valid rotatable bond (not terminal, not all atoms)
                        rotatable_bonds.append((i, j, moving_atoms))
    
    return rotatable_bonds

def find_connected_atoms(start_atom: int, exclude_atom: int, mol_coords: np.ndarray, 
                        mol_atomic_numbers: List[int], state: SystemState) -> List[int]:
    """
    Find all atoms connected to start_atom, excluding the path through exclude_atom.
    Uses depth-first search to find the molecular fragment.
    """
    n_atoms = len(mol_atomic_numbers)
    visited = set([exclude_atom])  # Don't cross back through the bond
    to_visit = [start_atom]
    connected = []
    
    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue
            
        visited.add(current)
        connected.append(current)
        
        # Find neighbors of current atom
        for neighbor in range(n_atoms):
            if neighbor in visited:
                continue
                
            distance = np.linalg.norm(mol_coords[neighbor] - mol_coords[current])
            radius_curr = state.r_atom.get(mol_atomic_numbers[current], 1.5)
            radius_neigh = state.r_atom.get(mol_atomic_numbers[neighbor], 1.5)
            max_bond_length = (radius_curr + radius_neigh) * 1.3
            
            if distance < max_bond_length:
                to_visit.append(neighbor)
    
    # Remove the start_atom itself from the moving atoms list
    if start_atom in connected:
        connected.remove(start_atom)
    
    return connected

def check_intramolecular_overlap(mol_coords: np.ndarray, mol_atomic_numbers: List[int], 
                                  state: SystemState) -> bool:
    """
    Checks if there are any severe overlaps (atom clashes) within a single molecule.
    This is used after conformational moves to ensure the rotation didn't cause 
    atoms to overlap unrealistically.
    
    Args:
        mol_coords: Coordinates of atoms in the molecule (N x 3 array)
        mol_atomic_numbers: List of atomic numbers for the molecule
        state: SystemState object containing overlap parameters
        
    Returns:
        True if severe overlap is detected, False otherwise
    """
    n_atoms = len(mol_atomic_numbers)
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = np.linalg.norm(mol_coords[i] - mol_coords[j])
            
            radius_i = state.r_atom.get(mol_atomic_numbers[i], 0.5)
            radius_j = state.r_atom.get(mol_atomic_numbers[j], 0.5)
            
            # Use a stricter overlap criterion for intramolecular checks
            # (0.5 is more strict than 0.7 used for intermolecular placement)
            min_distance_allowed = (radius_i + radius_j) * 0.5
            
            # Ignore very small distances that might be numerical errors
            if distance < min_distance_allowed and distance > 1e-4:
                return True  # Overlap detected
    
    return False  # No overlap

def propose_conformational_move(state: SystemState, current_rp: np.ndarray, 
                               current_imolec: List[int]) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
    DEPRECATED: This function is no longer used in the main annealing loop.
    Use propose_unified_move() instead, which properly handles all molecules.
    
    Kept for backward compatibility only.
    """
    # Redirect to propose_unified_move for correct behavior
    return propose_unified_move(state, current_rp, current_imolec)

def propose_unified_move(state: SystemState, current_rp: np.ndarray, 
                        current_imolec: List[int]) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
    Proposes moves for ALL molecules simultaneously with each molecule independently deciding:
    1. Attempt a conformational change (dihedral rotation) based on conformational_move_prob, OR
    2. Attempt rigid-body moves (translation and/or rotation - independent and random magnitude)
    
    All molecules are moved in each annealing attempt. For rigid-body moves:
    - Translation and rotation are INDEPENDENT (can have both, just one, or neither)
    - Each has random magnitude up to max limits (max_displacement_a ≤ 1 Å, max_rotation_angle_rad ≤ 1 rad)
    - Can result in: both translation+rotation, only translation, only rotation, or minimal movement
    
    The probability of attempting a conformational move per molecule is controlled by
    state.conformational_move_prob (0.0-1.0, can be 0% if disabled by user).
    """
    # Start with a copy of current coordinates
    proposed_rp_full_system = np.copy(current_rp)
    last_moved_mol_idx = -1
    move_type_used = "translate_rotate"
    
    # Iterate through ALL molecules
    for molecule_idx in range(state.num_molecules):
        start_atom_idx = current_imolec[molecule_idx] 
        end_atom_idx = current_imolec[molecule_idx + 1]
        
        # Each molecule independently decides whether to attempt conformational or rigid-body move
        attempt_conformational = (state.conformational_move_prob > 0 and 
                                 np.random.rand() < state.conformational_move_prob)
        
        conformational_success = False
        if attempt_conformational:
            # Try conformational move for this molecule
            mol_coords = proposed_rp_full_system[start_atom_idx:end_atom_idx, :]
            mol_atomic_numbers = [state.iznu[i] for i in range(start_atom_idx, end_atom_idx)]
            
            # Find rotatable bonds in this molecule
            rotatable_bonds = find_rotatable_bonds(mol_coords, mol_atomic_numbers, state)
            
            if rotatable_bonds:
                # Randomly select a rotatable bond
                bond_atom1, bond_atom2, moving_atoms = rotatable_bonds[np.random.randint(len(rotatable_bonds))]
                
                # Generate random rotation angle (can be up to max_dihedral_angle_rad)
                max_rotation = state.max_dihedral_angle_rad if state.max_dihedral_angle_rad > 0 else np.pi / 3
                rotation_angle = (np.random.rand() - 0.5) * 2.0 * max_rotation
                
                # Apply rotation to molecule coordinates
                new_mol_coords = rotate_around_bond(mol_coords, bond_atom1, bond_atom2, 
                                                   moving_atoms, rotation_angle)
                
                # Check for intramolecular overlaps
                if not check_intramolecular_overlap(new_mol_coords, mol_atomic_numbers, state):
                    proposed_rp_full_system[start_atom_idx:end_atom_idx, :] = new_mol_coords
                    last_moved_mol_idx = molecule_idx
                    move_type_used = "conformational"
                    conformational_success = True
        
        # If conformational move succeeded, this molecule is done (skip rigid-body moves)
        if conformational_success:
            continue
        
        # Apply rigid-body moves (translation and/or rotation - both are independent and random)
        mol_coords_current = proposed_rp_full_system[start_atom_idx:end_atom_idx, :]
        mol_atomic_numbers = [state.iznu[i] for i in range(start_atom_idx, end_atom_idx)]
        mol_masses = np.array([state.atomic_number_to_mass.get(anum, 1.0) 
                               for anum in mol_atomic_numbers])
        
        current_rcm = calculate_mass_center(mol_coords_current, mol_masses)

        # Translation Part - random displacement with random magnitude up to max_displacement_a
        # This generates a random vector where each component is in range [-max_displacement_a, +max_displacement_a]
        # The actual magnitude will be random (can be small or up to the max)
        random_displacement_vector = (np.random.rand(3) - 0.5) * 2.0 * state.max_displacement_a
        half_xbox = state.cube_length / 2.0 
        
        new_rcm_after_translation = np.copy(current_rcm)
        actual_atom_displacement = np.copy(random_displacement_vector)

        for dim in range(3):
            new_rcm_after_translation[dim] += random_displacement_vector[dim]
            if np.abs(new_rcm_after_translation[dim]) > half_xbox:
                new_rcm_after_translation[dim] -= 2.0 * random_displacement_vector[dim] 
                actual_atom_displacement[dim] = -random_displacement_vector[dim] 
        
        proposed_rp_full_system[start_atom_idx:end_atom_idx, :] += actual_atom_displacement

        # Rotation Part - random Euler angles with random magnitude up to max_rotation_angle_rad
        # Each angle is independently random (can be small or up to the max)
        alpha_rot = (np.random.rand() - 0.5) * 2.0 * state.max_rotation_angle_rad 
        beta_rot = (np.random.rand() - 0.5) * 2.0 * state.max_rotation_angle_rad  
        gamma_rot = (np.random.rand() - 0.5) * 2.0 * state.max_rotation_angle_rad 

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(gamma_rot), -np.sin(gamma_rot)],
            [0, np.sin(gamma_rot), np.cos(gamma_rot)]
        ], dtype=np.float64)

        Ry = np.array([
            [np.cos(beta_rot), 0, np.sin(beta_rot)],
            [0, 1, 0],
            [-np.sin(beta_rot), 0, np.cos(beta_rot)]
        ], dtype=np.float64)

        Rz = np.array([
            [np.cos(alpha_rot), -np.sin(alpha_rot), 0],
            [np.sin(alpha_rot), np.cos(alpha_rot), 0],
            [0, 0, 1]
        ], dtype=np.float64)

        rotation_matrix = Rz @ Ry @ Rx

        mol_coords_after_translation = proposed_rp_full_system[start_atom_idx:end_atom_idx, :]
        mol_coords_relative_to_cm = mol_coords_after_translation - new_rcm_after_translation
        
        rotated_relative_coords = (rotation_matrix @ mol_coords_relative_to_cm.T).T
        proposed_rp_full_system[start_atom_idx:end_atom_idx, :] = rotated_relative_coords + new_rcm_after_translation
        
        last_moved_mol_idx = molecule_idx
    
    # Return results
    return proposed_rp_full_system, proposed_rp_full_system[current_imolec[last_moved_mol_idx]:current_imolec[last_moved_mol_idx + 1], :] if last_moved_mol_idx >= 0 else np.array([]), last_moved_mol_idx, move_type_used

# 15. Propose Move Function (No longer used for full system randomization in annealing)
# Keeping this function definition for reference if a future iterative single-molecule
# movement strategy is desired.
def propose_move(state: SystemState, current_rp: np.ndarray, current_imolec: List[int]) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
    Proposes a random translation and rotation for ALL molecules,
    applying Fortran's 'trans' (CM bounce) and 'rotac' (Euler angle rotation) logic.
    Each molecule independently has a probability to be moved/rotated.
    Returns the new full system coordinates, the coordinates of the last moved molecule,
    the index of the last moved molecule, and the move type.
    """
    # Start with a copy of current coordinates
    proposed_rp_full_system = np.copy(current_rp)
    last_moved_mol_idx = -1
    
    # Iterate through ALL molecules
    for molecule_idx in range(state.num_molecules):
        start_atom_idx = current_imolec[molecule_idx] 
        end_atom_idx = current_imolec[molecule_idx + 1] 
        
        # Get the current coordinates of the selected molecule (from proposed_rp_full_system to accumulate changes)
        mol_coords_current = proposed_rp_full_system[start_atom_idx:end_atom_idx, :]
        
        # Calculate current center of mass for the selected molecule
        # Ensure state.iznu and state.atomic_number_to_mass are correctly populated
        mol_atomic_numbers = [state.iznu[i] for i in range(start_atom_idx, end_atom_idx)]
        mol_masses = np.array([state.atomic_number_to_mass.get(anum, 1.0) 
                               for anum in mol_atomic_numbers])
        
        current_rcm = calculate_mass_center(mol_coords_current, mol_masses)

        # Translation Part (matching Fortran's 'trans' subroutine)
        # Generate random displacement vector for CM, range [-max_displacement_a, +max_displacement_a]
        random_displacement_vector = (np.random.rand(3) - 0.5) * 2.0 * state.max_displacement_a

        half_xbox = state.cube_length / 2.0 
        
        new_rcm_after_translation = np.copy(current_rcm)
        actual_atom_displacement = np.copy(random_displacement_vector)

        for dim in range(3):
            new_rcm_after_translation[dim] += random_displacement_vector[dim]

            if np.abs(new_rcm_after_translation[dim]) > half_xbox:
                new_rcm_after_translation[dim] -= 2.0 * random_displacement_vector[dim] 
                actual_atom_displacement[dim] = -random_displacement_vector[dim] 
        
        proposed_rp_full_system[start_atom_idx:end_atom_idx, :] += actual_atom_displacement

        # Rotation Part (matching Fortran's 'rotac' subroutine with Euler angles)
        # Generate random Euler angles (alpha, beta, gamma) for Z, Y, X rotations respectively
        alpha_rot = (np.random.rand() - 0.5) * 2.0 * state.max_rotation_angle_rad 
        beta_rot = (np.random.rand() - 0.5) * 2.0 * state.max_rotation_angle_rad  
        gamma_rot = (np.random.rand() - 0.5) * 2.0 * state.max_rotation_angle_rad 

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(gamma_rot), -np.sin(gamma_rot)],
            [0, np.sin(gamma_rot), np.cos(gamma_rot)]
        ], dtype=np.float64)

        Ry = np.array([
            [np.cos(beta_rot), 0, np.sin(beta_rot)],
            [0, 1, 0],
            [-np.sin(beta_rot), 0, np.cos(beta_rot)]
        ], dtype=np.float64)

        Rz = np.array([
            [np.cos(alpha_rot), -np.sin(alpha_rot), 0],
            [np.sin(alpha_rot), np.cos(alpha_rot), 0],
            [0, 0, 1]
        ], dtype=np.float64)

        rotation_matrix = Rz @ Ry @ Rx

        mol_coords_after_translation = proposed_rp_full_system[start_atom_idx:end_atom_idx, :]
        mol_coords_relative_to_cm = mol_coords_after_translation - new_rcm_after_translation
        
        rotated_relative_coords = (rotation_matrix @ mol_coords_relative_to_cm.T).T

        proposed_rp_full_system[start_atom_idx:end_atom_idx, :] = rotated_relative_coords + new_rcm_after_translation
        
        last_moved_mol_idx = molecule_idx

    # Return the full system coordinates, the coordinates of the last moved molecule (for potential debugging),
    # the index of the last moved molecule, and a string indicating the move type.
    return proposed_rp_full_system, proposed_rp_full_system[current_imolec[last_moved_mol_idx]:current_imolec[last_moved_mol_idx + 1], :] if last_moved_mol_idx >= 0 else np.array([]), last_moved_mol_idx, "translate_rotate"

def propose_conformational_or_rigid_move(state: SystemState) -> bool:
    """
    Proposes either a conformational move or rigid-body move with proper fallback.
    This is applied DURING annealing, after initial placement.
    
    Returns:
        bool: True if a conformational move was successfully applied, False if rigid-body was used
    """
    conformational_applied = False
    
    # Always default to rigid-body move for now - conformational moves are too risky
    # This ensures annealing always progresses
    # TODO: Add more sophisticated conformational move validation before re-enabling
    
    # Select random molecule for rigid-body move
    molecule_idx = np.random.randint(0, state.num_molecules)
    
    start_atom_idx = state.imolec[molecule_idx]
    end_atom_idx = state.imolec[molecule_idx + 1]
    
    mol_atomic_numbers = [state.iznu[i] for i in range(start_atom_idx, end_atom_idx)]
    mol_masses = np.array([state.atomic_number_to_mass.get(anum, 1.0) 
                          for anum in mol_atomic_numbers])
    
    # Calculate current center of mass
    current_mol_rcm = calculate_mass_center(state.rp[start_atom_idx:end_atom_idx, :], mol_masses)
    
    # Create temporary relative coordinates
    mol_rel_coords_temp = state.rp[start_atom_idx:end_atom_idx, :] - current_mol_rcm
    
    # Apply translation and rotation
    trans(molecule_idx, state.rp, mol_rel_coords_temp, state.rcm, state.ds, state)
    rotac(molecule_idx, state.rp, mol_rel_coords_temp, state.rcm, state.dphi, state)
    
    return conformational_applied


# Subroutine: trans (Translates molecules randomly)
def trans(imo: int, r_coords: np.ndarray, rel_coords: np.ndarray, rcm_coords: np.ndarray, ds_val: float, state: SystemState):
    """
    Translates a molecule (imo) randomly.
    r_coords: Current atom coordinates (r in Fortran) - modified in place
    rel_coords: Relative atom coordinates (rel in Fortran) - modified in place
    rcm_coords: Center of mass coordinates for all molecules (rcm in Fortran) - modified in place
    ds_val: Maximum displacement parameter
    state: SystemState object for accessing imolec and xbox
    """
    for i in range(3): # Loop over x, y, z coordinates
        z = state.ran0_method() # Use state's ran0_method
        is_val = -1 if z < 0.5 else 1
        s = is_val * state.ran0_method() * ds_val

        # Store original rcm for potential rollback if out of bounds
        # original_rcm_i = rcm_coords[imo-1, i] # Not needed with current config_move logic

        # Move CM
        rcm_coords[imo, i] += s # imo is 0-indexed here

        # Apply periodic boundary conditions for CM
        if np.abs(rcm_coords[imo, i]) > state.xbox / 2.0: # Use state's xbox
            rcm_coords[imo, i] -= 2.0 * s # Effectively wraps around
            s = -s # Invert displacement for atom translation

        # Translate individual atoms within the molecule
        for inu in range(state.imolec[imo], state.imolec[imo+1]): # imo is 0-indexed
            r_coords[inu, i] += s
            rel_coords[inu - state.imolec[imo], i] = r_coords[inu, i] - rcm_coords[imo, i] # Update relative position

# Subroutine: rotac (Rotates molecules randomly)
def rotac(imo: int, r_coords: np.ndarray, rel_coords: np.ndarray, rcm_coords: np.ndarray, dphi_val: float, state: SystemState):
    """
    Rotates a molecule (imo) randomly around its center of mass.
    r_coords: Current atom coordinates (r in Fortran) - modified in place
    rel_coords: Relative atom coordinates (rel in Fortran) - modified in place
    rcm_coords: Center of mass coordinates for all molecules (rcm in Fortran) - used for reference
    dphi_val: Maximum rotation parameter
    state: SystemState object for accessing imolec and ran0_method
    """
    ix = -1 if state.ran0_method() < 0.5 else 1
    iy = -1 if state.ran0_method() < 0.5 else 1
    iz = -1 if state.ran0_method() < 0.5 else 1

    xa = ix * state.ran0_method() * dphi_val # Rotation angle around X
    ya = iy * state.ran0_method() * dphi_val # Rotation angle around Y
    za = iz * state.ran0_method() * dphi_val # Rotation angle around Z

    cos_xa = np.cos(xa)
    sin_xa = np.sin(xa)
    cos_ya = np.cos(ya)
    sin_ya = np.sin(ya)
    cos_za = np.cos(za)
    sin_za = np.sin(za)

    # Combined rotation matrix for ZYX intrinsic rotations (as implied by Fortran code)
    # The Fortran rotation matrix seems to be slightly unusual, let's replicate it directly.
    # It appears to be a specific sequence of rotations around X, Y, Z.
    # The provided Fortran code's rotation matrix corresponds to a rotation by xa around x,
    # then ya around y, then za around z (intrinsic rotations).
    # For accuracy, we'll use the exact matrix as written in Fortran.

    # Atom indices for the current molecule
    mol_start_idx = state.imolec[imo]
    mol_end_idx = state.imolec[imo+1]

    for i_rel, inu in enumerate(range(mol_start_idx, mol_end_idx)):
        X_orig = rel_coords[i_rel, 0]
        Y_orig = rel_coords[i_rel, 1]
        Z_orig = rel_coords[i_rel, 2]

        # Apply Fortran's specific rotation matrix elements
        rel_coords[i_rel, 0] = (
            X_orig * (cos_za * cos_ya) +
            Y_orig * (-cos_za * sin_ya * sin_xa + sin_za * cos_xa) +
            Z_orig * (cos_za * sin_ya * cos_xa + sin_za * sin_xa)
        )
        rel_coords[i_rel, 1] = (
            X_orig * (-sin_za * cos_ya) +
            Y_orig * (sin_za * sin_ya * sin_xa + cos_za * cos_xa) +
            Z_orig * (-sin_za * sin_ya * cos_xa + cos_za * sin_xa)
        )
        rel_coords[i_rel, 2] = (
            X_orig * (-sin_ya) +
            Y_orig * (-cos_ya * sin_xa) +
            Z_orig * (cos_ya * cos_xa)
        )

        # Update absolute coordinates based on new relative coordinates and CM
        for k in range(3):
            r_coords[inu, k] = rel_coords[i_rel, k] + rcm_coords[imo, k] # imo is 0-indexed

# Main Subroutine: config_move
def config_move(state: SystemState):
    """
    Generates a new configuration by randomly translating and rotating ALL molecules.
    This is used for INITIAL configuration generation (exiting overlap positions).
    For annealing moves, use propose_conformational_or_rigid_move() instead.
    Modifies state.rp and state.rcm in place.
    """
    # This function is now simplified for initial configuration generation only
    # It applies rigid-body moves to all molecules without extensive collision checking
    
    for imo in range(state.num_molecules):
        mol_start_idx = state.imolec[imo]
        mol_end_idx = state.imolec[imo+1]
        
        # Calculate current center of mass for the molecule
        mol_atomic_numbers = [state.iznu[i] for i in range(mol_start_idx, mol_end_idx)]
        mol_masses = np.array([state.atomic_number_to_mass.get(anum, 1.0) 
                               for anum in mol_atomic_numbers])
        
        # Calculate CM based on current coordinates
        current_mol_rcm = calculate_mass_center(state.rp[mol_start_idx:mol_end_idx, :], mol_masses)
        
        # Create temporary array for relative coordinates
        mol_rel_coords_temp = state.rp[mol_start_idx:mol_end_idx, :] - current_mol_rcm
        
        # Apply rigid-body move (translation and rotation)
        trans(imo, state.rp, mol_rel_coords_temp, state.rcm, state.ds, state) 
        rotac(imo, state.rp, mol_rel_coords_temp, state.rcm, state.dphi, state)

    # Recalculate all molecular centers of mass after the moves
    for i in range(state.num_molecules):
        mol_start_idx = state.imolec[i]
        mol_end_idx = state.imolec[i+1]
        mol_atomic_numbers = [state.iznu[j] for j in range(mol_start_idx, mol_end_idx)]
        mol_masses = np.array([state.atomic_number_to_mass.get(anum, 1.0) 
                               for anum in mol_atomic_numbers])
        state.rcm[i, :] = calculate_mass_center(state.rp[mol_start_idx:mol_end_idx, :], mol_masses)


# Function to clean up generated QM input/output files (now mostly handled by calculate_energy's finally block)
def cleanup_qm_files(files_to_clean: List[str], state: SystemState):
    """
    Cleans up any QM-related files explicitly added to the list.
    This function now primarily serves as a safeguard for files that might not have been
    cleaned by individual calculate_energy calls due to unexpected crashes or other issues.
    """
    # This list is mostly unused now as calculate_energy handles its own cleanup.
    # It remains here as a safety net if a file was created outside that scope and added.
    cleaned_count = 0
    files_to_remove = list(files_to_clean) 
    for fpath in files_to_remove:
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                cleaned_count += 1
            except OSError as e:
                _print_verbose(f"Error removing file {fpath} during final cleanup: {e}", 0, state)
    if cleaned_count > 0:
        _print_verbose(f"Cleaned up {cleaned_count} leftover QM files during final cleanup.", 1, state)

# Add this function somewhere in your script, e.g., near helper functions.
def calculate_molecular_volume(mol_def, method='covalent_spheres') -> float:
    """
    Calculates the approximate volume of a molecule using different methods.
    
    Args:
        mol_def: MoleculeData object containing atomic coordinates and numbers
        method (str): Calculation method - 'covalent_spheres', 'convex_hull', or 'grid_based'
    
    Returns:
        float: Estimated molecular volume in Angstroms^3
    """
    if not mol_def.atoms_coords:
        return 0.0
    
    if method == 'covalent_spheres':
        # Sum of individual atomic volumes using covalent radii
        # This is an upper bound estimate but computationally simple
        total_volume = 0.0
        for atomic_num, x, y, z in mol_def.atoms_coords:
            radius = r_atom.get(atomic_num, 1.5)  # Default to 1.5 Å for unknown atoms
            atomic_volume = (4.0/3.0) * np.pi * (radius ** 3)
            total_volume += atomic_volume
        
        # Apply an overlap correction factor (molecules are not just isolated spheres)
        # Typical values: 0.6-0.8 for organic molecules, 0.7-0.9 for inorganic
        overlap_factor = 0.74  # Based on typical molecular packing
        return total_volume * overlap_factor
    
    elif method == 'convex_hull':
        # Calculate volume using the convex hull of atomic spheres
        # More accurate for elongated or complex-shaped molecules
        try:
            from scipy.spatial import ConvexHull
            
            # Create points on the surface of each atomic sphere
            all_surface_points = []
            for atomic_num, x, y, z in mol_def.atoms_coords:
                radius = r_atom.get(atomic_num, 1.5)
                center = np.array([x, y, z])
                
                # Generate points on sphere surface (using spherical coordinates)
                n_points = 20  # Number of points per atom
                phi = np.random.uniform(0, 2*np.pi, n_points)
                costheta = np.random.uniform(-1, 1, n_points)
                theta = np.arccos(costheta)
                
                sphere_points = center + radius * np.column_stack([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
                all_surface_points.extend(sphere_points)
            
            if len(all_surface_points) < 4:
                # Fall back to covalent_spheres method
                return calculate_molecular_volume(mol_def, 'covalent_spheres')
            
            hull = ConvexHull(all_surface_points)
            return hull.volume
            
        except ImportError:
            # scipy not available, fall back to covalent_spheres
            return calculate_molecular_volume(mol_def, 'covalent_spheres')
        except Exception:
            # Any other error, fall back to covalent_spheres
            return calculate_molecular_volume(mol_def, 'covalent_spheres')
    
    else:
        # Default to covalent_spheres method
        return calculate_molecular_volume(mol_def, 'covalent_spheres')


def calculate_hydrogen_bond_potential(mol_def) -> Dict:
    """
    Calculate the potential hydrogen bonding volume based on donor/acceptor counts.
    
    Args:
        mol_def: MoleculeData object containing atomic coordinates and numbers
    
    Returns:
        Dictionary with hydrogen bond analysis and volume estimation
    """
    # Average hydrogen bond length in organic clusters
    avg_hb_length = 2.5  # Angstroms
    
    # Volume estimation: cylindrical volume around H-bond
    # Using radius of ~1.2 Å (approximate H-bond interaction zone)
    hb_interaction_radius = 1.2
    hb_volume_per_bond = math.pi * (hb_interaction_radius ** 2) * avg_hb_length
    
    # Count potential donors and acceptors
    donors = 0
    acceptors = 0
    
    for atomic_num, x, y, z in mol_def.atoms_coords:
        element = get_element_symbol(atomic_num)
        
        # Hydrogen bond donors (H attached to N, O, F)
        if element == 'H':
            donors += 1
        
        # Hydrogen bond acceptors (N, O, F with lone pairs)
        elif element in ['N', 'O', 'F']:
            acceptors += 1
        
        # Special cases for other elements that can participate
        elif element == 'S':
            acceptors += 0.5  # Weaker acceptor
        elif element == 'Cl':
            acceptors += 0.3  # Weak acceptor
    
    # Estimate potential hydrogen bonds (limited by the smaller of donors/acceptors)
    potential_hb_bonds = min(donors, acceptors)
    
    # Total volume needed for hydrogen bonding network
    hb_network_volume = potential_hb_bonds * hb_volume_per_bond
    
    return {
        'donors': int(donors),
        'acceptors': int(acceptors),
        'potential_bonds': potential_hb_bonds,
        'avg_bond_length': avg_hb_length,
        'hb_volume_per_bond': hb_volume_per_bond,
        'total_hb_volume': hb_network_volume
    }


def calculate_optimal_box_length(state: SystemState, target_packing_fractions: Optional[List[float]] = None) -> Dict:
    """
    Calculates optimal box lengths based on molecular volumes and target packing densities.
    
    Args:
        state: SystemState object containing molecule definitions
        target_packing_fractions: List of target packing fractions to calculate box sizes for
    
    Returns:
        Dict: Results containing volumes, box lengths for different packing fractions, and recommendations
    """
    if target_packing_fractions is None:
        # Use more conservative packing fractions for hydrogen-bonded systems
        target_packing_fractions = [0.05, 0.10, 0.15, 0.20, 0.25]  # 5% to 25% packing
    
    if not state.all_molecule_definitions:
        return {'error': 'No molecule definitions found'}
    
    results = {
        'individual_molecular_volumes': [],
        'total_molecular_volume': 0.0,
        'num_molecules': state.num_molecules,
        'box_length_recommendations': {},
        'packing_analysis': {}
    }
    
    # Calculate volume and hydrogen bond potential for each unique molecule definition
    unique_molecular_volumes = []
    unique_hb_analyses = []
    total_hb_volume = 0.0
    
    for mol_def in state.all_molecule_definitions:
        volume = calculate_molecular_volume(mol_def, method='covalent_spheres')
        hb_analysis = calculate_hydrogen_bond_potential(mol_def)
        
        unique_molecular_volumes.append(volume)
        unique_hb_analyses.append(hb_analysis)
        
        results['individual_molecular_volumes'].append({
            'molecule_label': mol_def.label,
            'num_atoms': mol_def.num_atoms,
            'volume_A3': volume,
            'hb_donors': hb_analysis['donors'],
            'hb_acceptors': hb_analysis['acceptors'],
            'potential_hb_bonds': hb_analysis['potential_bonds'],
            'hb_network_volume_A3': hb_analysis['total_hb_volume']
        })
    
    # Calculate total volume of all molecules that will be placed
    total_molecular_volume = 0.0
    total_hb_network_volume = 0.0
    
    for i, mol_def_idx in enumerate(state.molecules_to_add):
        if mol_def_idx < len(unique_molecular_volumes):
            total_molecular_volume += unique_molecular_volumes[mol_def_idx]
            total_hb_network_volume += unique_hb_analyses[mol_def_idx]['total_hb_volume']
    
    # Total effective volume includes both molecular and hydrogen bonding network volumes
    total_effective_volume = total_molecular_volume + total_hb_network_volume
    
    results['total_molecular_volume'] = total_molecular_volume
    results['total_hb_network_volume'] = total_hb_network_volume
    results['total_effective_volume'] = total_effective_volume
    
    if total_effective_volume <= 0:
        return {'error': 'Total effective volume is zero or negative'}
    
    # Calculate box lengths for different packing fractions
    # For hydrogen-bonded systems, use more conservative packing fractions
    for packing_fraction in target_packing_fractions:
        # Box volume = total_effective_volume / packing_fraction
        required_box_volume = total_effective_volume / packing_fraction
        
        # For a cubic box: L^3 = required_box_volume
        box_length = required_box_volume ** (1.0/3.0)
        
        results['box_length_recommendations'][f'{packing_fraction:.1%}'] = {
            'packing_fraction': packing_fraction,
            'box_length_A': box_length,
            'box_volume_A3': required_box_volume,
            'free_volume_A3': required_box_volume - total_effective_volume,
            'free_volume_fraction': 1.0 - packing_fraction,
            'molecular_volume_fraction': total_molecular_volume / required_box_volume,
            'hb_network_volume_fraction': total_hb_network_volume / required_box_volume
        }
    
    # Add analysis for current box length if available
    if hasattr(state, 'cube_length') and state.cube_length > 0:
        current_box_volume = state.cube_length ** 3
        current_packing_fraction = total_effective_volume / current_box_volume
        
        results['current_box_analysis'] = {
            'current_box_length_A': state.cube_length,
            'current_box_volume_A3': current_box_volume,
            'current_packing_fraction': current_packing_fraction,
            'current_free_volume_A3': current_box_volume - total_effective_volume,
            'current_free_volume_fraction': 1.0 - current_packing_fraction,
            'molecular_packing_fraction': total_molecular_volume / current_box_volume,
            'hb_network_fraction': total_hb_network_volume / current_box_volume
        }
    
    # Calculate largest molecular dimension for comparison with old method
    max_molecular_extent = 0.0
    for mol_def in state.all_molecule_definitions:
        if not mol_def.atoms_coords:
            continue
        coords_array = np.array([atom[1:] for atom in mol_def.atoms_coords])
        min_coords = np.min(coords_array, axis=0)
        max_coords = np.max(coords_array, axis=0)
        extents = max_coords - min_coords
        current_max_extent = np.max(extents)
        if current_max_extent > max_molecular_extent:
            max_molecular_extent = current_max_extent
    
    results['max_molecular_extent_A'] = max_molecular_extent
    results['old_method_recommendation_A'] = max_molecular_extent + 16.0  # 8 Å on each side
    
    return results


def write_box_analysis_to_file(state: SystemState, output_file_handle):
    """
    Writes box length analysis results to the output file.
    """
    if not state.all_molecule_definitions:
        return
    
    # Calculate optimal box lengths using volume-based approach
    results = calculate_optimal_box_length(state)
    
    if 'error' in results:
        return
    
    # Get recommendations for output file
    recommendations = results['box_length_recommendations']
    rec_5 = recommendations.get('5.0%', {}).get('box_length_A', 0)
    rec_10 = recommendations.get('10.0%', {}).get('box_length_A', 0)
    rec_15 = recommendations.get('15.0%', {}).get('box_length_A', 0)
    
    # Write to output file
    if rec_5 > 0 and rec_10 > 0 and rec_15 > 0:
        print(f"H-bond aware suggestions:", file=output_file_handle)
        print(f"  • Isolated clusters: {rec_5:.1f} A (5% effective packing)", file=output_file_handle)
        print(f"  • Cluster formation: {rec_10:.1f} A (10% effective packing)", file=output_file_handle)
        print(f"  • Network studies: {rec_15:.1f} A (15% effective packing)", file=output_file_handle)
    
    # Store results in state for potential use elsewhere
    state.max_molecular_extent = results['max_molecular_extent_A']
    state.volume_based_recommendations = recommendations

def provide_box_length_advice(state: SystemState):
    """
    Provides comprehensive advice on appropriate box lengths based on molecular volumes
    and target packing densities. This is a much more rigorous approach than the simple
    8 Angstrom rule of thumb.
    """
    if not state.all_molecule_definitions:
        _print_verbose("Cannot provide box length advice: No molecule definitions found.", 0, state)
        return

    _print_verbose("\n" + "="*78, 1, state)
    _print_verbose("Box length analysis", 1, state)
    _print_verbose("="*78, 1, state)
    _print_verbose(f"Successfully parsed {state.natom} atoms", 1, state)
    _print_verbose("", 1, state)
    
    # Calculate optimal box lengths using volume-based approach
    results = calculate_optimal_box_length(state)
    
    if 'error' in results:
        _print_verbose(f"Error in volume analysis: {results['error']}", 0, state)
        return
    
    # Display molecular volume analysis
    _print_verbose("1. Molecular volume and hydrogen bonding analysis:", 1, state)
    _print_verbose("-" * 50, 1, state)
    
    total_molecular_volume = results['total_molecular_volume']
    total_hb_volume = results['total_hb_network_volume']
    total_effective_volume = results['total_effective_volume']
    
    _print_verbose(f"Number of molecules to place: {results['num_molecules']}", 1, state)
    _print_verbose(f"  Total molecular volume: {total_molecular_volume:.2f} Å³", 1, state)
    _print_verbose(f"  Total H-bond network volume: {total_hb_volume:.2f} Å³", 1, state)
    _print_verbose(f"  Total effective volume: {total_effective_volume:.2f} Å³", 1, state)
    
    _print_verbose("\nIndividual molecule analysis:", 1, state)
    for i, mol_info in enumerate(results['individual_molecular_volumes']):
        # Get the molecular formula from the corresponding molecule definition
        if i < len(state.all_molecule_definitions):
            mol_def = state.all_molecule_definitions[i]
            molecular_formula = get_molecular_formula(mol_def)
            _print_verbose(f"  • {mol_info['molecule_label']}: {molecular_formula} {mol_info['volume_A3']:.2f} Å³", 1, state)
        else:
            _print_verbose(f"  • {mol_info['molecule_label']}: {mol_info['volume_A3']:.2f} Å³", 1, state)
    
    # Display box length recommendations
    _print_verbose("\n2. Box length recommendations (H-Bond Network Aware):", 1, state)
    _print_verbose("-" * 70, 1, state)
    
    recommendations = results['box_length_recommendations']
    _print_verbose("Packing (%)    Box Length (Å)     Box Volume (Å³)       Free (%)", 1, state)
    _print_verbose("-" * 70, 1, state)

    for key, rec in recommendations.items():
        pf = rec['packing_fraction']
        bl = rec['box_length_A']
        bv = rec['box_volume_A3']
        free_pct = rec['free_volume_fraction'] * 100
        _print_verbose(f"    {pf*100:4.1f}          {bl:6.2f}             {bv:6.0f}               {free_pct:4.1f}", 1, state)
    
    # Current box analysis - show current cube length and largest molecular extent
    if 'current_box_analysis' in results:
        _print_verbose("\n3. Current box analysis:", 1, state)
        _print_verbose("-" * 26, 1, state)
        current = results['current_box_analysis']
        _print_verbose(f"Cube's length = {current['current_box_length_A']:.2f} Å", 1, state)
        _print_verbose(f"  Current effective packing: {current['current_packing_fraction']:.1%}", 1, state)
        _print_verbose(f"    └ Molecular: {current['molecular_packing_fraction']:.1%}, H-bond network: {current['hb_network_fraction']:.1%}", 1, state)
        _print_verbose(f"  Current free volume: {current['current_free_volume_A3']:.0f} Å³ "
                      f"({current['current_free_volume_fraction']:.1%})", 1, state)
        _print_verbose(f"  Largest molecular extent: {results['max_molecular_extent_A']:.2f} Å", 1, state)
        
        # Provide assessment of current box size for H-bonded systems
        pf = current['current_packing_fraction']
        if pf < 0.05:
            assessment = "Very dilute - good for isolated cluster studies"
        elif pf < 0.15:
            assessment = "Dilute - appropriate for H-bonded cluster formation"
        elif pf < 0.25:
            assessment = "Moderate - suitable for network formation studies"
        elif pf < 0.35:
            assessment = "Dense - good for condensed phase simulations"
        elif pf < 0.45:
            assessment = "Very dense - may constrain H-bond network flexibility"
        else:
            assessment = "Extremely dense - may prevent proper H-bond formation"
        
        _print_verbose(f"  {assessment}", 1, state)
    
    # Store results in state for potential use elsewhere
    max_extent = results['max_molecular_extent_A']
    state.max_molecular_extent = max_extent
    state.volume_based_recommendations = recommendations
    
    # Final recommendations
    _print_verbose("\n4. Recommendations for H-bonded systems:", 1, state)
    _print_verbose("-" * 40, 1, state)
    
    # Get recommendations as reasonable defaults for H-bonded systems
    rec_5 = recommendations.get('5.0%', {}).get('box_length_A', 0)
    rec_10 = recommendations.get('10.0%', {}).get('box_length_A', 0)
    rec_15 = recommendations.get('15.0%', {}).get('box_length_A', 0)
    
    if rec_5 > 0 and rec_10 > 0 and rec_15 > 0:
        _print_verbose(f"• For isolated clusters: {rec_5:.1f} Å (5% effective packing)", 1, state)
        _print_verbose(f"• For cluster formation: {rec_10:.1f} Å (10% effective packing)", 1, state)
        _print_verbose(f"• For network studies: {rec_15:.1f} Å (15% effective packing)", 1, state)
        _print_verbose(f"• Includes space for H-bond network (avg. bond length: 2.5 Å)", 1, state)
    
    _print_verbose("\n" + "="*78, 1, state)
    _print_verbose("Note: This analysis accounts for hydrogen bonding networks in molecular clusters.", 1, state)
    _print_verbose("H-bond volume estimated using 2.5 Å average bond length and 1.2 Å interaction radius.", 1, state)
    _print_verbose("Run the full simulation to validate these recommendations.", 1, state)
    _print_verbose("="*78, 1, state)

def format_time_difference(seconds: float) -> str:
    """Formats a time difference in seconds into days, hours, minutes, seconds, milliseconds."""
    days = int(seconds // (24 * 3600))
    seconds %= (24 * 3600)
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    
    return f"{days} days {hours} h {minutes} min {seconds} s {milliseconds} ms"

def write_lowest_energy_config_file(state: SystemState, rless_filepath: str):
    """
    Writes the lowest energy configuration to a .out file,
    separated by molecule.
    """
    if state.lowest_energy_rp is None or state.lowest_energy_iznu is None:
        _print_verbose(f"Warning: No lowest energy configuration found to write to '{rless_filepath}'.", 0, state)
        return

    try:
        with open(rless_filepath, 'w') as f:
            # Modified format as requested (reduced spacing)
            f.write(f"# Configuration: {state.lowest_energy_config_idx} | Energy = {state.lowest_energy:.8f} u.a.\n") # Added |
            
            current_atom_idx = 0
            for i, mol_def_idx in enumerate(state.molecules_to_add):
                mol_def = state.all_molecule_definitions[mol_def_idx]
                
                f.write(f"{mol_def.num_atoms}\n")
                f.write(f"{mol_def.label}\n") # Changed to use full label from mol_def.label

                for _ in range(mol_def.num_atoms):
                    atomic_num = state.lowest_energy_iznu[current_atom_idx]
                    symbol = state.atomic_number_to_symbol.get(atomic_num, 'X')
                    coords = state.lowest_energy_rp[current_atom_idx]
                    f.write(f"{symbol: <3} {coords[0]: 12.6f} {coords[1]: 12.6f} {coords[2]: 12.6f}\n")
                    current_atom_idx += 1
        _print_verbose(f"Lowest energy configuration saved to: {rless_filepath}", 1, state)
    except IOError as e:
        _print_verbose(f"Error writing lowest energy configuration to '{rless_filepath}': {e}", 0, state)

def write_tvse_file(tvse_filepath: str, entry: Dict, state: SystemState):
    """
    Appends a single accepted configuration entry to the .dat file.
    """
    try:
        # Check if file exists and is empty to write header
        file_exists = os.path.exists(tvse_filepath)
        file_is_empty = not file_exists or os.stat(tvse_filepath).st_size == 0

        with open(tvse_filepath, 'a') as f: # Open in append mode
            if file_is_empty:
                # Adjusted header and spacing for alignment
                f.write(f"{'# n-eval (Cuml)':>16} {'T(K)':>10} {'E(u.a.)':>15}\n")
                f.write("\n") # Blank line after header
                f.write("#----------------------------------\n") # Separator line
            # Adjusted spacing for data columns to match header
            f.write(f"  {entry['n_eval']:>14} {entry['T']:>10.2f} {entry['E']:>15.6f}\n") 
        # _print_verbose(f"Energy evolution history appended to: {tvse_filepath}", 2, state) # Too verbose
    except IOError as e:
        _print_verbose(f"Error writing energy evolution history to '{tvse_filepath}': {e}", 0, state)


def create_launcher_script(replicated_files: List[str], input_dir: str, script_name: str = "launcher_ascec.sh") -> str:
    """
    Creates a bash launcher script for sequential execution of replicated runs.
    
    Args:
        replicated_files (List[str]): List of paths to the replicated input files
        input_dir (str): Directory where the launcher script should be created
        script_name (str): Name of the launcher script
    
    Returns:
        str: Path to the created launcher script
    """
    launcher_path = os.path.join(input_dir, script_name)
    
    # Get the directory where ascec-v04.py is located
    ascec_script_path = os.path.abspath(__file__)
    ascec_directory = os.path.dirname(ascec_script_path)
    
    try:
        with open(launcher_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            
            # Configuration for ASCEC v04
            f.write("# Configuration for ASCEC v04\n")
            f.write("# Set ASCEC_ROOT to the directory containing ascec-v04.py\n")
            f.write(f'export ASCEC_ROOT="{ascec_directory}"\n\n')
            
            f.write("# Save original environment paths\n")
            f.write('_SYSTEM_PATH="$PATH"\n\n')
            
            f.write("# Add the ASCEC directory to the system PATH for direct execution\n")
            f.write('export PATH="$ASCEC_ROOT:$_SYSTEM_PATH"\n\n')
            
            f.write('echo "ASCEC v04 environment is now active via direct script setup."\n')
            f.write('echo "ASCEC_ROOT set to: $ASCEC_ROOT"\n\n')
            
            f.write("# Run ASCEC using the full path\n")
            
            commands = []
            for i, replicated_file in enumerate(replicated_files):
                rel_path = os.path.relpath(replicated_file, input_dir)
                output_name = os.path.splitext(rel_path)[0] + ".out"
                
                # Add separator before each calculation (except the first one)
                if i > 0:
                    commands.append('echo "=================================================================="')
                
                commands.append(f"python {ascec_script_path} {rel_path} > {output_name}")
            
            # Join commands with " ; \\\n" for sequential execution
            f.write(" ; \\\n".join(commands))
            f.write("\n")
        
        # Make the script executable
        os.chmod(launcher_path, 0o755)
        
        print(f"Created launcher script: {script_name}")
        return launcher_path
        
    except IOError as e:
        print(f"Error creating launcher script '{launcher_path}': {e}")
        return ""


def merge_launcher_scripts(working_dir: str = ".") -> str:
    """
    Finds all launcher_ascec.sh scripts in the working directory and subfolders,
    and merges them into a single launcher script.
    
    Args:
        working_dir (str): Working directory to search for launcher scripts
    
    Returns:
        str: Path to the merged launcher script
    """
    working_dir_full = os.path.abspath(working_dir)
    merged_launcher_path = os.path.join(working_dir_full, "launcher_ascec.sh")
    
    # Find all launcher scripts
    launcher_scripts = []
    for root, dirs, files in os.walk(working_dir_full):
        for file in files:
            if file == "launcher_ascec.sh":
                launcher_scripts.append(os.path.join(root, file))
    
    if not launcher_scripts:
        print("No launcher_ascec.sh scripts found in the working directory or subfolders.")
        return ""
    
    print(f"Found {len(launcher_scripts)} launcher scripts:")
    for script in launcher_scripts:
        rel_path = os.path.relpath(script, working_dir_full)
        print(f"  {rel_path}")
    
    # Merge all launcher scripts
    all_commands = []
    
    try:
        for script_path in launcher_scripts:
            with open(script_path, 'r') as f:
                lines = f.readlines()
                
            # Extract commands (skip shebang and comments)
            commands = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and 'python ascec-v04.py' in line:
                    # Remove trailing " ; \\" if present
                    line = line.rstrip(' \\;')
                    commands.append(line)
            
            if commands:
                # Add commands from this script
                all_commands.extend(commands)
                # Add separator comment between different script groups
                if script_path != launcher_scripts[-1]:  # Don't add separator after last script
                    all_commands.append("###")
        
        # Write merged launcher script
        with open(merged_launcher_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            
            # Configuration for ASCEC v04
            # Get the directory where ascec-v04.py is located
            ascec_script_path = os.path.abspath(__file__)
            ascec_directory = os.path.dirname(ascec_script_path)
            
            f.write("# Configuration for ASCEC v04\n")
            f.write("# Set ASCEC_ROOT to the directory containing ascec-v04.py\n")
            f.write(f'export ASCEC_ROOT="{ascec_directory}"\n\n')
            
            f.write("# Save original environment paths\n")
            f.write('_SYSTEM_PATH="$PATH"\n\n')
            
            f.write("# Add the ASCEC directory to the system PATH for direct execution\n")
            f.write('export PATH="$ASCEC_ROOT:$_SYSTEM_PATH"\n\n')
            
            f.write('echo "ASCEC v04 environment is now active via direct script setup."\n')
            f.write('echo "ASCEC_ROOT set to: $ASCEC_ROOT"\n\n')
            
            f.write("# Run ASCEC using the full path\n")
            
            # Process commands with proper formatting
            for i, cmd in enumerate(all_commands):
                if cmd == "###":
                    # Write separator on its own line
                    f.write(" ; \\\n###\n")
                else:
                    f.write(cmd)
                    # Add " ; \" only if this is not the last command and the next command is not "###"
                    if i < len(all_commands) - 1 and all_commands[i + 1] != "###":
                        f.write(" ; \\\n")
                    elif i == len(all_commands) - 1:
                        f.write("\n")  # Last command, just add newline
        
        # Make the script executable
        os.chmod(merged_launcher_path, 0o755)
        
        print(f"\nCreated merged launcher script: launcher_ascec.sh")
        print(f"Total commands: {len([cmd for cmd in all_commands if cmd != '###'])}")
        return merged_launcher_path
        
    except IOError as e:
        print(f"Error creating merged launcher script: {e}")
        return ""


def extract_configurations_from_xyz(xyz_file_path: str) -> List[Dict]:
    """
    Extracts all configurations from an XYZ file.
    
    Args:
        xyz_file_path (str): Path to the XYZ file
    
    Returns:
        List[Dict]: List of configuration dictionaries with atoms, energy, and config number
    """
    configurations = []
    
    try:
        with open(xyz_file_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            # Skip empty lines
            if not lines[i].strip():
                i += 1
                continue
            
            # Read number of atoms
            try:
                num_atoms = int(lines[i].strip())
            except (ValueError, IndexError):
                i += 1
                continue
            
            # Read comment line with configuration info
            if i + 1 >= len(lines):
                break
            
            comment_line = lines[i + 1].strip()
            
            # Extract configuration number and energy from comment
            config_num = 1
            energy = 0.0
            
            if "Configuration:" in comment_line:
                parts = comment_line.split("|")
                for part in parts:
                    part = part.strip()
                    if part.startswith("Configuration:"):
                        try:
                            config_num = int(part.split(":")[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif "E =" in part:
                        try:
                            energy_str = part.split("=")[1].strip().split()[0]
                            energy = float(energy_str)
                        except (ValueError, IndexError):
                            pass
            else:
                # Handle other formats like "Motif_01_opt_conf_4 (G = -55.537389 hartree)"
                import re
                # Try to extract configuration number from patterns like "conf_4" or "Motif_01"
                conf_match = re.search(r'conf_(\d+)', comment_line)
                if conf_match:
                    try:
                        config_num = int(conf_match.group(1))
                    except ValueError:
                        pass
                else:
                    # Try to extract from Motif_XX or motif_XX pattern
                    motif_match = re.search(r'[Mm]otif_(\d+)', comment_line)
                    if motif_match:
                        try:
                            config_num = int(motif_match.group(1))
                        except ValueError:
                            pass
                
                # Try to extract energy from patterns like "G = -55.537389" or "E = -55.537389"
                energy_match = re.search(r'[GE]\s*=\s*([-\d.]+)', comment_line)
                if energy_match:
                    try:
                        energy = float(energy_match.group(1))
                    except ValueError:
                        pass
            
            # Read atom coordinates (preserve original string format for precision)
            atoms = []
            for j in range(num_atoms):
                if i + 2 + j >= len(lines):
                    break
                
                atom_line = lines[i + 2 + j].strip()
                if atom_line:
                    parts = atom_line.split()
                    if len(parts) >= 4:
                        symbol = parts[0]
                        x_str = parts[1]  # Keep original string format
                        y_str = parts[2]  # Keep original string format
                        z_str = parts[3]  # Keep original string format
                        # Also store float values for numerical operations if needed
                        try:
                            x_float = float(x_str)
                            y_float = float(y_str)
                            z_float = float(z_str)
                            atoms.append((symbol, x_str, y_str, z_str, x_float, y_float, z_float))
                        except ValueError:
                            # Skip malformed coordinates
                            continue
            
            if len(atoms) == num_atoms:
                # Convert Motif_ to motif_ in comment for consistency
                processed_comment = comment_line.replace('Motif_', 'motif_')
                configurations.append({
                    'config_num': config_num,
                    'energy': energy,
                    'atoms': atoms,
                    'comment': processed_comment
                })
            
            i += num_atoms + 2
    
    except IOError as e:
        print(f"Error reading XYZ file '{xyz_file_path}': {e}")
        return []
    
    return configurations


def create_qm_input_file(config_data: Dict, template_content: str, output_path: str, qm_program: str) -> bool:
    """
    Creates a QM input file from configuration data and template.
    
    Args:
        config_data (Dict): Configuration data with atoms, energy, etc.
        template_content (str): Template file content
        output_path (str): Path where to save the input file
        qm_program (str): QM program type ('orca' or 'gaussian')
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create coordinates section (preserve original precision with proper alignment)
        coords_section = ""
        for atom in config_data['atoms']:
            if len(atom) == 7:  # New format with string coordinates
                symbol, x_str, y_str, z_str, x_float, y_float, z_float = atom
                # Right-align coordinates to maintain column alignment while preserving original precision
                coords_section += f"{symbol: <3} {x_str: >12}  {y_str: >12}  {z_str: >12}\n"
            else:  # Old format compatibility
                symbol, x, y, z = atom
                coords_section += f"{symbol: <3} {x: 12.6f}  {y: 12.6f}  {z: 12.6f}\n"
        
        # Replace name placeholders with the configuration comment (only if placeholders exist)
        content = template_content
        if qm_program == 'orca':
            # ORCA uses # for placeholders
            if "#name" in content:
                content = content.replace("#name", f"# {config_data['comment']}")
        elif qm_program == 'gaussian':
            # Gaussian uses ! for placeholders
            if "!name" in content:
                content = content.replace("!name", f"! {config_data['comment']}")
        
        if qm_program == 'orca':
            # For ORCA, replace the coordinate section between * xyz and *
            # Look for * xyz pattern (case-insensitive) with charge and multiplicity
            import re
            
            lines = content.split('\n')
            new_lines = []
            in_coords = False
            xyz_pattern = re.compile(r'^\s*\*\s+xyz\s+[-\d]+\s+\d+\s*$', re.IGNORECASE)
            
            for line in lines:
                if xyz_pattern.match(line.strip()):
                    new_lines.append(line)
                    in_coords = True
                elif in_coords and line.strip() == "*":
                    new_lines.append(line)
                    in_coords = False
                elif in_coords and line.strip() == "#":
                    # Replace the # placeholder with coordinates
                    new_lines.append(coords_section.rstrip())
                elif in_coords:
                    # Skip other coordinate lines (they will be replaced by coords_section when we hit #)
                    continue
                elif not in_coords:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
        
        elif qm_program == 'gaussian':
            # For Gaussian, replace ! placeholder with coordinates
            lines = content.split('\n')
            new_lines = []
            
            for line in lines:
                if line.strip() == "!":
                    # Replace the ! placeholder with coordinates
                    new_lines.append(coords_section.rstrip())
                else:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Write the input file
        with open(output_path, 'w') as f:
            f.write(content)
        
        return True
        
    except IOError as e:
        print(f"Error creating QM input file '{output_path}': {e}")
        return False


def interactive_directory_selection_with_pattern(input_ext: str, pattern: str = "") -> List[str]:
    """
    Provides interactive directory selection for updating input files with optional pattern filtering.
    Shows only directories that contain matching files and lets user choose.
    
    Args:
        input_ext (str): File extension to search for
        pattern (str): Pattern to filter files (empty string means no filtering)
    
    Returns:
        List[str]: List of selected input file paths
    """
    verbose = True  # Interactive mode always shows detailed output
    
    print("\n" + "=" * 60)
    print("Directory selection".center(60))
    if pattern and pattern.strip():
        print(f"Filtering files containing: '{pattern}'".center(60))
    print("=" * 60)
    
    def filter_files(files, pattern):
        """Filter files by pattern if pattern is provided"""
        if not pattern or not pattern.strip():
            return files
        return [f for f in files if pattern in os.path.basename(f)]
    
    # Scan for directories with matching files, but group similar paths
    directories_with_files = {}
    all_files = []
    
    # Find all matching files recursively
    for root, dirs, files in os.walk("."):
        matching_files_in_dir = []
        for file in files:
            if file.endswith(input_ext):
                full_path = os.path.join(root, file)
                if not pattern or not pattern.strip() or pattern in file:
                    matching_files_in_dir.append(full_path)
                    all_files.append(full_path)
        
        # Only include directories that have matching files
        if matching_files_in_dir:
            # Normalize and group directory paths
            if root == ".":
                dir_display = "Current working directory"
            else:
                dir_display = root
                if dir_display.startswith("./"):
                    dir_display = dir_display[2:]
                dir_display += "/"
            
            # Group similar directories (e.g., merge individual semiempiric/calculation/opt1_conf_XX/ into one)
            if "semiempiric/calculation/" in dir_display and dir_display.count("/") > 2:
                # Group all semiempiric individual calculation directories
                parent_dir = "semiempiric/calculation/ (individual directories)"
                if parent_dir not in directories_with_files:
                    directories_with_files[parent_dir] = []
                directories_with_files[parent_dir].extend(matching_files_in_dir)
            else:
                directories_with_files[dir_display] = matching_files_in_dir
    
    if not directories_with_files:
        if pattern and pattern.strip():
            print(f"No files found containing pattern '{pattern}'.")
        else:
            print("No input files found in any directory.")
        return []
    
    # Display options - only directories with files
    print("\nDirectories with matching files:")
    print("-" * 40 + "\n")
    
    # Create numbered options
    options = {}
    option_num = 1
    
    # Sort directories by name for consistent ordering
    sorted_dirs = sorted(directories_with_files.items(), key=lambda x: (len(x[1]), x[0]))
    
    for dir_name, files in sorted_dirs:
        options[str(option_num)] = (dir_name, files)
        print(f"{option_num}. {dir_name}: {len(files)} files")
        
        # Show first few files as examples
        if len(files) <= 5:
            # Show all files if 5 or fewer
            for file_path in files:
                filename = os.path.basename(file_path)
                print(f"   - {filename}")
        else:
            # Show first 3 files if more than 5
            examples = files[:3]
            for example in examples:
                filename = os.path.basename(example)
                print(f"   - {filename}")
            print(f"   ... and {len(files) - 3} more")
        if verbose:
            print()
        option_num += 1
    
    # Add "all" option if there are multiple directories
    if len(directories_with_files) > 1:
        options["a"] = ("All directories", all_files)
        print(f"a. All directories: {len(all_files)} files total")
        if verbose:
            print()
    
    # Get user choice
    while True:
        try:
            valid_options = list(options.keys()) + ['q']
            if len(valid_options) > 6:  # If too many options, simplify the prompt
                choice = input(f"Select option (1-{len(valid_options)-2}, 'a' for all, or 'q' to quit): ").strip().lower()
            else:
                # Build a cleaner prompt with explicit 'a' for all description
                option_parts = []
                for opt in valid_options[:-1]:  # Exclude 'q'
                    if opt == 'a':
                        option_parts.append("'a' for all")
                    else:
                        option_parts.append(opt)
                choice = input(f"Select option ({', '.join(option_parts)}, or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("Update cancelled.")
                return []
            
            if choice in options:
                selected_files = options[choice][1]
                dir_name = options[choice][0]
                
                print(f"\nSelected: {dir_name}")
                print(f"Files to update: {len(selected_files)}")
                if pattern and pattern.strip():
                    print(f"Pattern filter: '{pattern}'")
                
                return selected_files
            else:
                if len(valid_options) > 6:
                    print(f"Invalid option. Please choose 1-{len(valid_options)-2}, 'a', or 'q' to quit.")
                else:
                    print(f"Invalid option. Please choose {', '.join(valid_options[:-1])}, or 'q' to quit.")
                
        except KeyboardInterrupt:
            print("\nUpdate cancelled by user.")
            return []
        except EOFError:
            print("\nUpdate cancelled.")
            return []


# REMOVED: Duplicate of interactive_directory_selection_with_pattern (use that with pattern="")
# This function had identical logic to the _with_pattern variant


def find_files_by_pattern(pattern: str, input_ext: str) -> List[str]:
    """
    Finds input files matching a specific pattern across all directories.
    
    Args:
        pattern (str): Pattern to match in filenames
        input_ext (str): File extension to search for
    
    Returns:
        List[str]: List of matching file paths
    """
    matching_files = []
    
    # Search recursively in all directories
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(input_ext) and pattern in file:
                matching_files.append(os.path.join(root, file))
    
    print(f"\nFound {len(matching_files)} files matching pattern '{pattern}':")
    for file in matching_files:
        print(f"  - {file}")
    
    return matching_files


def update_existing_input_files(template_file: str, target_pattern: str = "") -> str:
    """
    Updates existing QM input files with a new template, preserving the coordinates 
    and configuration information from the original files.
    
    Args:
        template_file (str): New template file (e.g., new_template.inp)
        target_pattern (str): Search pattern - empty string for all locations, or specific pattern for filtered search
    
    Returns:
        str: Status message
    """
    # Determine QM program from template file extension
    if template_file.endswith('.inp'):
        qm_program = 'orca'
        input_ext = '.inp'
    elif template_file.endswith('.com'):
        qm_program = 'gaussian'
        input_ext = '.com'
    elif template_file.endswith('.gjf'):
        qm_program = 'gaussian'
        input_ext = '.gjf'
    else:
        return f"Error: Unsupported template file extension. Use .inp for ORCA or .com/.gjf for Gaussian."
    
    # Check if template file exists
    if not os.path.exists(template_file):
        return f"Error: Template file '{template_file}' not found."
    
    # Read template content
    try:
        with open(template_file, 'r') as f:
            template_content = f.read()
    except IOError as e:
        return f"Error reading template file '{template_file}': {e}"
    
    # Always show interactive directory selection, but filter by pattern if provided
    input_files = interactive_directory_selection_with_pattern(input_ext, target_pattern)
    if not input_files:
        return "No files selected for update."
    
    # Exclude the template file from the update list
    template_file_abs = os.path.abspath(template_file)
    input_files = [f for f in input_files if os.path.abspath(f) != template_file_abs]
    
    if not input_files:
        return "No input files found for update (excluding template file)."
    
    updated_count = 0
    skipped_count = 0
    backup_files = []  # Track backup files for potential revert
    
    print(f"\nStarting update of {len(input_files)} files...")
    
    for input_file in input_files:
        try:
            # Read existing input file
            with open(input_file, 'r') as f:
                existing_content = f.read()
            
            # Create backup file
            backup_file = input_file + '.backup_temp'
            with open(backup_file, 'w') as f:
                f.write(existing_content)
            backup_files.append((input_file, backup_file))
            
            # Extract configuration information and coordinates from existing file
            config_info = extract_config_from_input_file(existing_content, qm_program)
            if not config_info:
                print(f"Warning: Could not extract configuration from {os.path.basename(input_file)}, skipping.")
                skipped_count += 1
                continue
            
            # Create updated content using the new template
            if create_qm_input_file(config_info, template_content, input_file, qm_program):
                print(f"  Updated: {os.path.basename(input_file)}")
                updated_count += 1
            else:
                print(f"  Failed to update: {os.path.basename(input_file)}")
                skipped_count += 1
                
        except IOError as e:
            print(f"Error processing {os.path.basename(input_file)}: {e}")
            skipped_count += 1
    
    # Show final result and offer revert option
    print(f"\nUpdate completed: {updated_count} files updated, {skipped_count} files skipped.")
    
    if updated_count > 0:
        print("\nPress ENTER to finish and keep changes, or type 'r' and ENTER to revert all changes:")
        try:
            user_choice = input().strip().lower()
            if user_choice == 'r':
                # Revert all changes
                reverted_count = 0
                for original_file, backup_file in backup_files:
                    try:
                        if os.path.exists(backup_file):
                            # Restore original content
                            with open(backup_file, 'r') as f:
                                original_content = f.read()
                            with open(original_file, 'w') as f:
                                f.write(original_content)
                            reverted_count += 1
                    except IOError as e:
                        print(f"Error reverting {os.path.basename(original_file)}: {e}")
                
                # Clean up backup files
                for _, backup_file in backup_files:
                    try:
                        if os.path.exists(backup_file):
                            os.remove(backup_file)
                    except OSError:
                        pass
                
                return f"Reverted {reverted_count} files to original state."
            else:
                # Keep changes, clean up backup files
                for _, backup_file in backup_files:
                    try:
                        if os.path.exists(backup_file):
                            os.remove(backup_file)
                    except OSError:
                        pass
                return f"Changes kept: {updated_count} files updated, {skipped_count} files skipped."
        except KeyboardInterrupt:
            print("\nKeeping changes (Ctrl+C pressed).")
            # Clean up backup files
            for _, backup_file in backup_files:
                try:
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                except OSError:
                    pass
            return f"Changes kept: {updated_count} files updated, {skipped_count} files skipped."
        except EOFError:
            print("\nKeeping changes.")
            # Clean up backup files
            for _, backup_file in backup_files:
                try:
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                except OSError:
                    pass
            return f"Changes kept: {updated_count} files updated, {skipped_count} files skipped."
    else:
        # No files were updated, just clean up any backup files
        for _, backup_file in backup_files:
            try:
                if os.path.exists(backup_file):
                    os.remove(backup_file)
            except OSError:
                pass
        return f"Update completed: {updated_count} files updated, {skipped_count} files skipped."


def extract_config_from_input_file(content: str, qm_program: str) -> Optional[Dict]:
    """
    Extracts configuration information from an existing QM input file.
    
    Args:
        content (str): Content of the input file
        qm_program (str): QM program type ('orca' or 'gaussian')
    
    Returns:
        Dict: Configuration data or None if extraction fails
    """
    try:
        lines = content.split('\n')
        
        # Extract comment line (configuration info)
        comment = ""
        for line in lines:
            if line.strip().startswith('#') and ('Configuration:' in line or 'E =' in line):
                comment = line.strip()[1:].strip()  # Remove # and extra spaces
                break
        
        # Extract coordinates
        atoms = []
        
        if qm_program == 'orca':
            # For ORCA, find coordinates between "* xyz 0 1" and "*"
            in_coords = False
            for line in lines:
                if line.strip() == "* xyz 0 1":
                    in_coords = True
                    continue
                elif line.strip() == "*" and in_coords:
                    break
                elif in_coords and line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        symbol = parts[0]
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        atoms.append((symbol, x, y, z))
        
        elif qm_program == 'gaussian':
            # For Gaussian, find coordinates after charge/multiplicity line
            found_charge_mult = False
            for line in lines:
                if not found_charge_mult and line.strip() and len(line.strip().split()) == 2:
                    try:
                        int(line.strip().split()[0])  # charge
                        int(line.strip().split()[1])  # multiplicity
                        found_charge_mult = True
                        continue
                    except ValueError:
                        pass
                elif found_charge_mult and line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        symbol = parts[0]
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        atoms.append((symbol, x, y, z))
                elif found_charge_mult and not line.strip():
                    # Empty line might indicate end of coordinates
                    if atoms:  # Only break if we have found some atoms
                        break
        
        if not atoms:
            return None
        
        return {
            'comment': comment,
            'atoms': atoms
        }
        
    except Exception as e:
        print(f"Error extracting configuration: {e}")
        return None


def interactive_xyz_file_selection(xyz_files: List[str], optimization_dir_path: str = ".", auto_select: Optional[str] = None,
                                   quiet: bool = False) -> List[str]:
    """
    Provides interactive selection for XYZ files to process.
    
    Args:
        xyz_files (List[str]): List of available XYZ file paths
        optimization_dir_path (str): Directory where combined files should be created
        auto_select (Optional[str]): Auto-selection mode:
            - 'all': Process all result_*.xyz files (excludes combined files)
            - 'combined': Combine all result_*.xyz files into combined_r{N}.xyz, then process that
            - None: Interactive prompt for user selection
    
    Returns:
        List[str]: List of selected XYZ file paths
    """
    if not xyz_files:
        print("No XYZ files found.")
        return []
    
    # Separate result_*.xyz files from combined_results.xyz and combined_r*.xyz
    result_files = [f for f in xyz_files if not (os.path.basename(f).startswith("combined_results") or os.path.basename(f).startswith("combined_r"))]
    combined_files = [f for f in xyz_files if (os.path.basename(f).startswith("combined_results") or os.path.basename(f).startswith("combined_r"))]
    
    # Handle auto-selection
    if auto_select == 'all':
        if not quiet:
            print(f"\nAuto-selected: All result_*.xyz files ({len(result_files)} files)")
        return result_files
    elif auto_select == 'combined':
        if len(result_files) == 1:
            # Single file: just use it directly (treat as "combined")
            if not quiet:
                print(f"\nAuto-selected: Single result file (treating as combined)")
                print(f"Using: {os.path.basename(result_files[0])}")
            return result_files
        elif len(result_files) > 1:
            # Multiple files: combine them
            if not quiet:
                print(f"\nAuto-combining {len(result_files)} result_*.xyz files...")
            combined_filename = os.path.join(optimization_dir_path, f"combined_r{len(result_files)}.xyz")
            success = merge_xyz_files(result_files, combined_filename, quiet=quiet)
            if success:
                if not quiet:
                    print(f"Successfully created {os.path.basename(combined_filename)}")
                return [combined_filename]
            else:
                if not quiet:
                    print("Failed to combine files. Using individual files instead.")
                return result_files
        else:
            # No files
            if not quiet:
                print("\nNo result files found")
            return []
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("XYZ file selection".center(60))
    print("=" * 60)
    
    # Display options
    print("\nAvailable XYZ files:")
    print("-" * 40)
    
    # Create numbered options
    options = {}
    option_num = 1
    
    # First show result_*.xyz files
    if result_files:
        print("\nResult files:")
        for xyz_file in result_files:
            options[str(option_num)] = xyz_file
            print(f"{option_num}. {xyz_file}")
            option_num += 1
    
    # Then show combined files
    if combined_files:
        print("\nCombined files:")
        for xyz_file in combined_files:
            options[str(option_num)] = xyz_file
            print(f"{option_num}. {xyz_file}")
            option_num += 1
    
    # Add special options
    print("\nSpecial options:")
    if len(result_files) > 1:
        options["a"] = "All result files"
        print(f"a. Process all result_*.xyz files ({len(result_files)} files, excluding combined)")
        options["c"] = "Combine result files"
        print(f"c. Combine all result_*.xyz files first, then process the combined file (combined_r{len(result_files)}.xyz)")
    
    print("q. Quit")
    
    # Get user choice
    while True:
        try:
            choice = input("\nSelect files (enter numbers separated by spaces, 'a' for all, or 'c' to combine): ").strip()
            
            if choice.lower() == 'q':
                print("Operation cancelled.")
                return []
            
            if choice.lower() == 'a' and len(result_files) > 1:
                print(f"Selected: All result_*.xyz files ({len(result_files)} files)")
                return result_files
            
            if choice.lower() == 'c' and len(result_files) > 1:
                print(f"Combining {len(result_files)} result_*.xyz files...")
                # Use merge_xyz_files to combine result files only
                combined_filename = os.path.join(optimization_dir_path, f"combined_r{len(result_files)}.xyz")
                success = merge_xyz_files(result_files, combined_filename)
                if success:
                    print(f"Successfully created {os.path.basename(combined_filename)}")
                    return [combined_filename]
                else:
                    print("Failed to combine files. Using individual files instead.")
                    return result_files
            
            # Handle numbered selections (single or multiple)
            try:
                numbers = [int(x.strip()) for x in choice.split() if x.strip().isdigit()]
                selected_files = []
                
                for num in numbers:
                    if str(num) in options:
                        selected_files.append(options[str(num)])
                    else:
                        print(f"Invalid number: {num}")
                        raise ValueError
                
                if selected_files:
                    print(f"Selected {len(selected_files)} file(s):")
                    for f in selected_files:
                        print(f"  - {f}")
                    return selected_files
                else:
                    print("No valid files selected.")
                    
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces, or 'a' for all result files.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return []
        except EOFError:
            print("\nOperation cancelled.")
            return []


def create_combined_xyz_from_list(xyz_files: List[str]) -> bool:
    """
    Create a combined XYZ file from a list of XYZ files.
    
    Args:
        xyz_files (List[str]): List of XYZ file paths to combine
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not xyz_files:
        print("No XYZ files to combine.")
        return False
    
    combined_content = []
    
    for xyz_file in xyz_files:
        try:
            with open(xyz_file, 'r') as f:
                content = f.read().strip()
                if content:
                    combined_content.append(f"# {xyz_file}")
                    combined_content.append(content)
                    combined_content.append("")  # Empty line between structures
        except Exception as e:
            print(f"Error reading {xyz_file}: {e}")
    
    if combined_content:
        try:
            with open("combined_results.xyz", 'w') as f:
                f.write("\n".join(combined_content))
            print(f"Created combined_results.xyz with {len(xyz_files)} structures.")
            return True
        except Exception as e:
            print(f"Error creating combined_results.xyz: {e}")
            return False
    else:
        print("No content to combine.")
        return False
    
    # Add special options
    if len(xyz_files) > 1:
        options["a"] = "All files"
        print(f"a. All files: {len(xyz_files)} files total")
    
    options["c"] = "Combined results"
    num_result_files = len([f for f in xyz_files if 'result_' in os.path.basename(f)])
    print(f"c. Combine all result_*.xyz files first, then process the combined file (combined_r{num_result_files}.xyz)")
    print()
    
    # Get user choice
    while True:
        try:
            valid_options = list(options.keys()) + ['q']
            if len(valid_options) > 6:
                choice = input(f"Select option (1-{len(valid_options)-3}, 'a' for all, 'c' for combined, or 'q' to quit): ").strip().lower()
            else:
                # Build a cleaner prompt
                option_parts = []
                for opt in valid_options[:-1]:  # Exclude 'q'
                    if opt == 'a':
                        option_parts.append("'a' for all")
                    elif opt == 'c':
                        option_parts.append("'c' for combined")
                    else:
                        option_parts.append(opt)
                choice = input(f"Select option ({', '.join(option_parts)}, or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("Calculation system creation cancelled.")
                return []
            
            if choice == 'c':
                # Create combined_r{N}.xyz first where N is number of result files
                num_result_files = len([f for f in xyz_files if 'result_' in os.path.basename(f)])
                combined_filename = f"combined_r{num_result_files}.xyz"
                print(f"\nCreating {combined_filename}...")
                if combine_xyz_files(output_filename=combined_filename):
                    if os.path.exists(combined_filename):
                        print(f"Successfully created {combined_filename}")
                        return [combined_filename]
                    else:
                        print(f"Error: {combined_filename} was not created.")
                        return []
                else:
                    print(f"Failed to create {combined_filename}")
                    return []
            
            if choice == 'a':
                if len(xyz_files) > 1:
                    print(f"\nSelected: All files ({len(xyz_files)} files)")
                    return xyz_files
                else:
                    print("Only one file available, selecting it.")
                    return xyz_files
            
            if choice in options and choice not in ['a', 'c']:
                selected_file = options[choice]
                print(f"\nSelected: {selected_file}")
                return [selected_file]
            else:
                if len(valid_options) > 6:
                    print(f"Invalid option. Please choose 1-{len(valid_options)-3}, 'a', 'c', or 'q' to quit.")
                else:
                    print(f"Invalid option. Please choose {', '.join(valid_options[:-1])}, or 'q' to quit.")
                
        except KeyboardInterrupt:
            print("\nCalculation system creation cancelled by user.")
            return []
        except EOFError:
            print("\nCalculation system creation cancelled.")
            return []


def extract_qm_executable_from_launcher(launcher_content: str, qm_program_idx: int) -> str:
    """
    Extract the QM executable path from the launcher template by looking for
    any exported variable ending in _ROOT and constructing the executable path,
    or by finding existing run commands in the template.
    
    Args:
        launcher_content (str): Content of the launcher template
        qm_program_idx (int): QM program index (1: Gaussian, 2: ORCA)
    
    Returns:
        str: The executable path or command to use
    """
    import re
    
    qm_name = qm_program_details.get(qm_program_idx, {}).get('name', 'unknown')
    
    # Get default executable based on program type
    if qm_name == 'orca':
        default_exe = qm_program_details.get(qm_program_idx, {}).get('default_exe', 'orca')
    elif qm_name == 'gaussian':
        default_exe = qm_program_details.get(qm_program_idx, {}).get('default_exe', 'g16')
    else:
        default_exe = qm_program_details.get(qm_program_idx, {}).get('default_exe', 'unknown')
    
    if not launcher_content:
        return default_exe
    
    lines = launcher_content.split('\n')
    
    # Strategy 1: Look for existing run commands after the ### separator
    # This extracts the actual command the user intends to use
    after_separator = False
    for line in lines:
        if line.strip() == '###':
            after_separator = True
            continue
        
        if after_separator:
            # Look for command patterns like: $VAR/exe file > output or exe file > output
            # Match patterns: <executable> <input_file> [>] [output_file] [&&]
            match = re.search(r'^\s*(\S+)\s+\S+\.(inp|gjf|com)\s*(?:>|\s)', line)
            if match:
                return match.group(1)  # Return the executable part
    
    # Strategy 2: Look for any *_ROOT variable exports and construct executable path
    # This is generic and works with any naming convention (ORCA5_ROOT, GAUSSIAN_ROOT, QM_ROOT, etc.)
    root_pattern = r'export\s+([A-Z_0-9]+ROOT)\s*=\s*(.+)'
    root_var = None
    root_value = None
    
    for line in lines:
        match = re.search(root_pattern, line)
        if match:
            var_name = match.group(1)
            var_value = match.group(2).strip().strip('"\'')
            
            # For ORCA, any variable containing "ORCA" in the name
            if qm_name == 'orca' and 'ORCA' in var_name.upper():
                root_var = var_name
                root_value = var_value
                break
            # For Gaussian, look for variables containing G09, G16, or GAUSS
            elif qm_name == 'gaussian' and any(x in var_name.upper() for x in ['G09', 'G16', 'GAUSS']):
                root_var = var_name
                root_value = var_value
                # Try to detect version from variable name
                if 'G09' in var_name.upper():
                    default_exe = 'g09'
                elif 'G16' in var_name.upper():
                    default_exe = 'g16'
                break
    
    # If we found a ROOT variable, construct the executable path
    if root_var and root_value:
        if '$' in root_value or not root_value.startswith('/'):
            # Variable expansion in path, use the variable
            return f"${root_var}/{default_exe}"
        else:
            # Absolute path
            return f"{root_value}/{default_exe}"
    
    # Strategy 3: Check if executable appears to be in PATH
    # Look for any sourcing of profiles or PATH modifications
    for line in lines:
        # If we see the QM program mentioned in PATH or source commands
        if 'PATH=' in line or 'source' in line.lower() or '. ' in line:
            if qm_name == 'orca' and 'orca' in line.lower():
                return default_exe
            elif qm_name == 'gaussian':
                if 'g09' in line.lower():
                    return 'g09'
                elif 'g16' in line.lower():
                    return 'g16'
                elif 'gaussian' in line.lower():
                    return default_exe
    
    # Default fallback
    return default_exe


def calculate_input_files(template_file: str, launcher_template: Optional[str] = None, 
                          auto_select: str = 'interactive', stage_type: str = "optimization", 
                          workflow_mode: bool = False, qm_alias: str = "orca") -> str:
    """
    Unified function to create QM input files and launcher scripts for both
    optimization and refinement stages.
    
    Args:
        template_file (str): Path to the QM input template file.
        launcher_template (Optional[str]): Path to the launcher script template.
        auto_select (str): 'interactive', 'all', or 'combined' for file selection.
        stage_type (str): "optimization" or "refinement".
        workflow_mode (bool): True if called from a workflow, suppresses some print statements.
        
    Returns:
        str: Status message indicating success or failure.
    """
    # Determine QM program and file extension from template
    if template_file.lower().endswith('.inp'):
        qm_program = 'orca'
        input_ext = '.inp'
        output_ext = '.out'
        qm_program_idx = 2 # For extract_qm_executable_from_launcher (matches qm_program_details key)
    elif template_file.lower().endswith('.com'):
        qm_program = 'gaussian'
        input_ext = '.com'
        output_ext = '.log'
        qm_program_idx = 1 # For extract_qm_executable_from_launcher (matches qm_program_details key)
    elif template_file.lower().endswith('.gjf'):
        qm_program = 'gaussian'
        input_ext = '.gjf'
        output_ext = '.log'
        qm_program_idx = 1 # For extract_qm_executable_from_launcher (matches qm_program_details key)
    else:
        return f"Error: Unsupported template file extension. Use .inp for ORCA or .com/.gjf for Gaussian."
    
    # Check if template files exist
    if not os.path.exists(template_file):
        return f"Error: Template file '{template_file}' not found."
    
    launcher_content = None
    if launcher_template:
        if not os.path.exists(launcher_template):
            return f"Error: Launcher template '{launcher_template}' not found."
        try:
            with open(launcher_template, 'r') as f:
                launcher_content = f.read()
        except IOError as e:
            return f"Error reading launcher template '{launcher_template}': {e}"
    
    # Read template content
    try:
        with open(template_file, 'r') as f:
            template_content = f.read()
    except IOError as e:
        return f"Error reading template file '{template_file}': {e}"
    
    # Normalize accepted stage aliases.
    stage_type_aliases = {
        'opt': 'optimization',
        'optimization': 'optimization',
        'ref': 'refinement',
        'refinement': 'refinement',
    }
    stage_type = stage_type_aliases.get(stage_type.lower(), stage_type.lower())

    # Determine output directory name
    def get_next_dir(base):
        """Find the next available directory (base, base_2, etc.)"""
        if not os.path.exists(base):
            return base
        counter = 2
        while True:
            dir_name = f"{base}_{counter}"
            if not os.path.exists(dir_name):
                return dir_name
            counter += 1
    
    output_dir = get_next_dir(stage_type)
    
    # File Search Logic
    xyz_files = []
    if stage_type == "optimization":
        # FIRST: Check for retry_input folder (structures from need_recalculation)
        if os.path.exists("retry_input"):
            if not workflow_mode:
                print("Found retry_input folder, using structures from previous similarity analysis")
            for file in os.listdir("retry_input"):
                if file.endswith(".xyz") and not file.startswith("combined"):
                    xyz_files.append(os.path.join("retry_input", file))
        else:
            # Normal flow: Check for combined files in current directory
            for file in os.listdir("."):
                if (file.startswith("combined_results") or file.startswith("combined_r")) and file.endswith(".xyz"):
                    xyz_files.append(file)
            
            # Look for result_*.xyz files recursively in subdirectories
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.startswith("result_") and file.endswith(".xyz") and not file.startswith("resultbox_"):
                        xyz_files.append(os.path.join(root, file))
                        
        if not xyz_files:
            return "No result_*.xyz, combined_results.xyz, or combined_r*.xyz files found."
            
        # Sort XYZ files by annealing number
        def get_annealing_number(file_path):
            import re
            directory = os.path.dirname(file_path)
            match = re.search(r'_(\d+)$', directory)
            if match: return int(match.group(1))
            match = re.search(r'result_(\d+)', os.path.basename(file_path))
            if match: return int(match.group(1))
            return float('inf')
        xyz_files.sort(key=get_annealing_number)
        
    elif stage_type == "refinement":
        # Check for combined files
        for file in os.listdir("."):
            if file.endswith(".xyz") and "combined" in file.lower():
                xyz_files.append(file)
        
        # Look for motif_*.xyz files
        for root, dirs, files in os.walk("."):
            for file in files:
                if (file.startswith("motif_") and file.endswith(".xyz")) or \
                   (file.endswith(".xyz") and "combined" in file.lower() and root != "."):
                    xyz_files.append(os.path.join(root, file))
                    
        if not xyz_files:
            return "No files with 'combined' in name or motif_*.xyz files found."
            
    # File Selection Logic
    if stage_type == "optimization":
        if not workflow_mode:
            print(f"Found {len(xyz_files)} XYZ file(s) to process:")
            for xyz_file in xyz_files:
                print(f"  - {xyz_file}")
            
        # Store first XYZ source for protocol summary
        if hasattr(sys, '_current_workflow_context') and sys._current_workflow_context is not None:  # type: ignore[attr-defined]
            if xyz_files:
                first_file = xyz_files[0]
                if 'result' in first_file or 'annealing' in first_file.lower():
                    sys._current_workflow_context.optimization_xyz_source = "Annealing"  # type: ignore[attr-defined]
                else:
                    sys._current_workflow_context.optimization_xyz_source = first_file  # type: ignore[attr-defined]
            else:
                sys._current_workflow_context.optimization_xyz_source = "Annealing"  # type: ignore[attr-defined]
                
        if auto_select == 'combined':
            os.makedirs(output_dir, exist_ok=True)
            
        selected_xyz_files = interactive_xyz_file_selection(
            xyz_files,
            output_dir,
            auto_select=auto_select,
            quiet=workflow_mode,
        )
        
    elif stage_type == "refinement":
        selected_xyz_files = interactive_optimization_file_selection(xyz_files, output_dir)
    else:
        selected_xyz_files = []
        
    if not selected_xyz_files:
        return "No XYZ files selected for processing."
    xyz_files = selected_xyz_files
    
    # Create directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if not workflow_mode:
            print(f"Created {stage_type} directory: {output_dir}")
        
    # Process files in parallel
    all_input_files = []
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    total_xyz_files = len(xyz_files)
    use_combined_naming = (auto_select == 'combined')
    
    # Prepare arguments
    if stage_type == "optimization":
        xyz_file_args = [(xyz_file, template_content, output_dir, qm_program, input_ext, total_xyz_files, use_combined_naming) 
                         for xyz_file in xyz_files]
        process_func = _process_xyz_file_for_calc
    else: # refinement
        xyz_file_args = [(xyz_file, template_content, output_dir, qm_program, input_ext) 
                         for xyz_file in xyz_files]
        process_func = _process_xyz_file_for_opt
        
    max_workers = get_optimal_workers('mixed', len(xyz_files))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_func, args): args[0] for args in xyz_file_args}
        
        for future in as_completed(future_to_file):
            xyz_file = future_to_file[future]
            try:
                file_input_files, status_msg = future.result()
                all_input_files.extend(file_input_files)
                if not workflow_mode:
                    print(f"  {status_msg}")
                    for input_file in file_input_files:
                        print(f"    Created: {input_file}")
            except Exception as e:
                if not workflow_mode:
                    print(f"  Error processing {xyz_file}: {e}")
                    
    if not workflow_mode:
        print(f"\nCompleted processing. Created {len(all_input_files)} input files total.")
    
    if not all_input_files:
        return "No input files were created successfully."
        
    # Deduplicate input files: keep only flat filenames (basenames)
    # This prevents duplicate commands like motif_29/motif_29_opt.inp AND motif_29_opt.inp
    seen_basenames = set()
    launcher_input_files = []
    for input_file in all_input_files:
        basename = os.path.basename(input_file)
        if basename not in seen_basenames:
            seen_basenames.add(basename)
            launcher_input_files.append(basename)

    # Sort input files in natural numeric order where possible
    def sort_key(f):
        import re
        nums = re.findall(r'\d+', f)
        return [int(n) for n in nums] if nums else [f]

    launcher_input_files.sort(key=sort_key)

    # Decide launcher source: user template or auto-generated ORCA environment
    launcher_base_content = None
    launcher_generated_automatically = False

    if launcher_template and launcher_content:
        launcher_base_content = launcher_content
    elif qm_program == 'orca':
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            auto_launcher_tmp = create_auto_launcher(tmpdir, qm_program, qm_alias, quiet=workflow_mode)
            if auto_launcher_tmp and os.path.exists(auto_launcher_tmp):
                with open(auto_launcher_tmp, 'r') as f:
                    launcher_base_content = f.read()
                launcher_generated_automatically = True

    if launcher_base_content:
        launcher_path = os.path.join(output_dir, f"launcher_{qm_program}.sh")
        try:
            qm_executable = extract_qm_executable_from_launcher(launcher_base_content, qm_program_idx)
            if not workflow_mode:
                print(f"Using QM executable: {qm_executable}")

            with open(launcher_path, 'w') as f:
                # Header - copy everything up to ### separator, skip any existing run commands
                launcher_lines = launcher_base_content.rstrip().split('\n')
                separator_found = False
                import re

                for line in launcher_lines:
                    if line.strip() == '###':
                        separator_found = True
                        f.write(line + "\n\n# Run QM using the full path\n")
                        break
                    else:
                        # Skip lines that look like run commands (generic pattern)
                        if re.search(r'\S+\s+\S+\.(inp|gjf|com)\s*(?:>|;|&&|\s*$)', line):
                            continue
                        # Skip standalone continuations
                        if line.strip() in ('&&', '&& \\', ';', '; \\', '\\'):
                            continue
                        f.write(line + "\n")

                if not separator_found:
                    f.write("\n###\n\n# Run QM using the full path\n")

                for i, input_file in enumerate(launcher_input_files):
                    output_file = input_file.replace(input_ext, output_ext)
                    if qm_program == 'gaussian':
                        cmd = f"{qm_executable} < {input_file} > {output_file}"
                    else:  # orca
                        cmd = f"{qm_executable} {input_file} > {output_file}"

                    if i < len(launcher_input_files) - 1:
                        f.write(f"{cmd} ; \\\n")
                    else:
                        f.write(f"{cmd}\n")

            os.chmod(launcher_path, 0o755)

            msg = f"\nCreated {stage_type} system in '{output_dir}' directory:\n"
            msg += f"  Input files: {len(all_input_files)}\n"
            if launcher_generated_automatically:
                msg += f"  Launcher script: launcher_{qm_program}.sh (auto-generated)"
            else:
                msg += f"  Launcher script: launcher_{qm_program}.sh"
            if not workflow_mode:
                msg += f"\n\nTo run all calculations, use:\n"
                msg += f"  cd {output_dir}\n"
                msg += f"  ./launcher_{qm_program}.sh"
            return msg

        except IOError as e:
            return f"Error creating launcher script: {e}"

    return f"Created {stage_type} system in '{output_dir}' with {len(all_input_files)} input files (no launcher)."


def create_optimization_system(template_file: str, launcher_template: str) -> str:
    """
    Creates an optimization system by extracting configurations from XYZ files
    and generating QM input files with launcher scripts.
    
    Args:
        template_file (str): Template input file (e.g., example_input.inp)
        launcher_template (str): Template launcher file (e.g., launcher_orca.sh)
    
    Returns:
        str: Status message
    """
    return calculate_input_files(template_file, launcher_template, stage_type="optimization")


def create_simple_optimization_system(template_file: str, launcher_template: Optional[str] = None) -> str:
    """
    Creates an optimization system.
    
    Args:
        template_file (str): Path to the QM input template file
        launcher_template (str): Path to the launcher script template
    
    Returns:
        str: Status message indicating success or failure
    """
    return calculate_input_files(template_file, launcher_template, stage_type="optimization")


def create_refinement_system(template_file: str, launcher_template: Optional[str] = None) -> str:
    """
    Creates a refinement system by looking for files with 'combined' in their name 
    (like all_motifs_combined.xyz) and motif_*.xyz files.
    
    Args:
        template_file (str): Path to the QM input template file
        launcher_template (str): Path to the launcher script template
    
    Returns:
        str: Status message indicating success or failure
    """
    return calculate_input_files(template_file, launcher_template, stage_type="refinement")


def interactive_optimization_file_selection(xyz_files: List[str], opt_dir: str = ".") -> List[str]:
    """
    Provides interactive selection for optimization XYZ files.
    
    Args:
        xyz_files (List[str]): List of available XYZ file paths
        opt_dir (str): Directory where files will be processed
    
    Returns:
        List[str]: List of selected XYZ file paths
    """
    print("\n" + "=" * 60)
    print("Optimization XYZ file selection".center(60))
    print("=" * 60)
    
    if not xyz_files:
        print("No XYZ files found.")
        return []
    
    # Separate combined files from motif/umotif files
    combined_files = [f for f in xyz_files if "combined" in os.path.basename(f).lower()]
    # Match both motif_ and umotif_ files
    motif_files = [f for f in xyz_files if ("motif_" in os.path.basename(f) or "umotif_" in os.path.basename(f)) and "combined" not in os.path.basename(f).lower()]
    
    # Sort motif/umotif files by number
    def extract_motif_number(filepath):
        import re
        filename = os.path.basename(filepath)
        # Match both motif_XX and umotif_XX patterns
        match = re.search(r'u?motif_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    motif_files.sort(key=extract_motif_number)
    
    # Display options
    print("\nAvailable XYZ files:")
    print("-" * 40)
    
    # Create numbered options
    options = {}
    option_num = 0
    
    # First show combined files
    if combined_files:
        print("\nCombined files:")
        for xyz_file in combined_files:
            options[str(option_num)] = xyz_file
            print(f"{option_num}) {xyz_file}")
            option_num += 1
    
    # Then show motif files
    if motif_files:
        print("\nMotif files:")
        for xyz_file in motif_files:
            options[str(option_num)] = xyz_file
            print(f"{option_num}) {xyz_file}")
            option_num += 1
    
    # Add special options
    print("\nSpecial options:")
    if len(xyz_files) > 1:
        options["a"] = "All files"
        print(f"a. Process all files ({len(xyz_files)} files total)")
    
    print("q. Quit")
    
    # Get user choice
    while True:
        try:
            choice = input("\nSelect files (enter numbers separated by spaces, 'a' for all, or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                print("Optimization system creation cancelled.")
                return []
            
            if choice.lower() == 'a':
                print(f"Selected: All files ({len(xyz_files)} files)")
                return xyz_files
            
            # Handle numbered selections (single or multiple)
            try:
                choices = choice.split()
                selected_files = []
                
                for ch in choices:
                    if ch in options and ch not in ['a', 'q']:
                        selected_files.append(options[ch])
                    else:
                        print(f"Invalid choice: {ch}")
                        break
                else:
                    if selected_files:
                        print("Selected %s file(s):" % len(selected_files))
                        for f in selected_files:
                            print(f"  - {f}")
                        return selected_files
                    else:
                        print("No valid files selected.")
                        
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces, or 'a' for all files.")
                
        except KeyboardInterrupt:
            print("\nOptimization system creation cancelled by user.")
            return []
        except EOFError:
            print("\nOptimization system creation cancelled.")
            return []


def execute_merge_command(result_files_only=False):
    """
    Execute the merge command functionality.
    Shows directories with XYZ files and allows user to merge them.
    
    Args:
        result_files_only (bool): If True, only show result_*.xyz files. 
                                   If False, show all XYZ files except _trj, combined_results, and combined_r
    """
    verbose = True  # Interactive mode always shows detailed output
    
    # Set title and file filter based on mode
    if result_files_only:
        title = "Result XYZ Files Merge System"
        file_descriptor = "result XYZ"
    else:
        title = "XYZ Files Merge System"
        file_descriptor = "XYZ"
    
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)
    
    # Scan for directories with XYZ files based on filter
    directories_with_xyz = {}
    
    # Check current directory
    current_xyz = []
    for file in os.listdir("."):
        if result_files_only:
            if file.startswith("result_") and file.endswith(".xyz"):
                current_xyz.append(file)
        else:
            if (file.endswith(".xyz") and not file.endswith("_trj.xyz") and 
                not file.startswith("combined_results") and not file.startswith("combined_r")):
                current_xyz.append(file)
    
    if current_xyz:
        directories_with_xyz["."] = current_xyz
    
    # Check subdirectories
    for root, dirs, files in os.walk("."):
        if root == ".":
            continue
            
        xyz_files = []
        for file in files:
            if result_files_only:
                if file.startswith("result_") and file.endswith(".xyz"):
                    xyz_files.append(os.path.join(root, file))
            else:
                if (file.endswith(".xyz") and not file.endswith("_trj.xyz") and 
                    not file.startswith("combined_results") and not file.startswith("combined_r")):
                    xyz_files.append(os.path.join(root, file))
        
        if xyz_files:
            directories_with_xyz[root] = xyz_files
    
    if not directories_with_xyz:
        if result_files_only:
            print("No result_*.xyz files found in current directory or subdirectories.")
        else:
            print("No .xyz files found in current directory or subdirectories (excluding _trj.xyz, combined_results, and combined_r files).")
        return

    # Display options
    print(f"\nDirectories with {file_descriptor} files:")
    print("-" * 40)
    
    options = {}
    option_num = 1
    
    total_files = 0
    for dir_path, xyz_files in directories_with_xyz.items():
        dir_display = "Current directory" if dir_path == "." else dir_path
        options[str(option_num)] = (dir_path, xyz_files)
        file_count = len(xyz_files)
        total_files += file_count
        
        print(f"{option_num}. {dir_display}: {file_count} files")
        
        # Show first few files as examples
        examples = xyz_files[:3]
        for example in examples:
            filename = os.path.basename(example)
            print(f"   - {filename}")
        if len(xyz_files) > 3:
            print(f"   ... and {len(xyz_files) - 3} more")
        if verbose:
            print()
        option_num += 1
    
    # Add "all" option if there are multiple directories
    if len(directories_with_xyz) > 1:
        all_files = []
        for files in directories_with_xyz.values():
            all_files.extend(files)
        options["a"] = ("All directories", all_files)
        print(f"a. All directories: {total_files} files total")
        if verbose:
            print()
    
    # Get user choice
    while True:
        try:
            valid_options = list(options.keys()) + ['q']
            choice = input(f"Select option (1-{len(directories_with_xyz)}, 'a' for all, or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("Merge operation cancelled.")
                return
            
            if choice in options:
                dir_path, files_to_merge = options[choice]
                
                if choice == "a":
                    output_file = "combined_results.xyz"
                    merge_msg = f"result files" if result_files_only else "files"
                    print(f"\nMerging {len(files_to_merge)} {merge_msg} from all directories...")
                else:
                    if dir_path == ".":
                        output_file = "combined_results.xyz"
                        merge_msg = f"result files" if result_files_only else "files"
                        print(f"\nMerging {len(files_to_merge)} {merge_msg} from current directory...")
                    else:
                        dir_name = os.path.basename(dir_path)
                        output_file = f"combined_results_{dir_name}.xyz"
                        merge_msg = f"result files" if result_files_only else "files"
                        print(f"\nMerging {len(files_to_merge)} {merge_msg} from {dir_path}...")
                
                # Perform the merge
                success = merge_xyz_files(files_to_merge, output_file)
                
                if success:
                    print(f"✓ Successfully created {output_file}")
                    file_type = "result XYZ files" if result_files_only else "XYZ files"
                    print(f"  Combined {len(files_to_merge)} {file_type}")
                    
                    # Check if .mol file was also created
                    mol_file = output_file.replace('.xyz', '.mol')
                    if os.path.exists(mol_file):
                        print(f"  Also created {mol_file}")
                else:
                    print(f"✗ Failed to create {output_file}")
                
                return
            else:
                print(f"Invalid option. Please select 1-{len(directories_with_xyz)}, 'a', or 'q'.")
                
        except KeyboardInterrupt:
            print("\nMerge operation cancelled by user.")
            return
        except EOFError:
            print("\nMerge operation cancelled.")
            return


def execute_merge_result_command():
    """
    Execute the merge result command (wrapper for backward compatibility).
    Shows directories with result_*.xyz files and allows user to merge them.
    """
    execute_merge_command(result_files_only=True)


def merge_xyz_files(xyz_files: List[str], output_filename: str, quiet: bool = False) -> bool:
    """
    Merge a list of XYZ files into a single output file.
    Renumbers configurations sequentially and adds source file information.
    
    Args:
        xyz_files (List[str]): List of XYZ file paths to merge
        output_filename (str): Name of the output file
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not xyz_files:
        return False
    
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # Sort files by the number in the filename for consistent ordering
        def get_sort_key(filepath):
            filename = os.path.basename(filepath)
            # Try multiple patterns to extract configuration numbers
            patterns = [
                r'result_(\d+)\.xyz',           # result_123.xyz
                r'conf_(\d+)\.xyz',             # conf_20.xyz
                r'opt\d+_conf_(\d+)\.xyz',      # opt1_conf_20.xyz
                r'_(\d+)\.xyz',                 # any_123.xyz (general pattern)
                r'(\d+)\.xyz'                   # 123.xyz (number only)
            ]
            
            for pattern in patterns:
                match = re.search(pattern, filename)
                if match:
                    return int(match.group(1))
            
            # If no pattern matches, sort alphabetically by filename
            return (float('inf'), filename)
        
        sorted_files = sorted(xyz_files, key=get_sort_key)
        
        total_configs = 0
        
        with open(output_filename, 'w') as outfile:
            for file_idx, xyz_file in enumerate(sorted_files):
                try:
                    # Extract configurations from this file
                    configurations = extract_configurations_from_xyz(xyz_file)
                    
                    if not configurations:
                        print(f"Warning: No configurations found in {xyz_file}")
                        continue
                    
                    source_name = os.path.basename(xyz_file).replace('.xyz', '')
                    
                    for config in configurations:
                        total_configs += 1
                        
                        # Write atom count
                        outfile.write(f"{len(config['atoms'])}\n")
                        
                        # Parse original comment to extract energy
                        original_comment = config['comment']
                        energy_match = re.search(r'E = ([-\d.]+) a\.u\.', original_comment)
                        
                        energy = energy_match.group(1) if energy_match else "unknown"
                        
                        # Create new comment with sequential numbering and source info (no temperature, no "Source:" label)
                        if energy == "unknown":
                            new_comment = f"Configuration: {total_configs} | {source_name}"
                        else:
                            new_comment = f"Configuration: {total_configs} | E = {energy} a.u. | {source_name}"
                        
                        outfile.write(f"{new_comment}\n")
                        
                        # Write atoms (preserve original coordinate precision with proper column alignment)
                        for atom in config['atoms']:
                            if len(atom) == 7:  # New format with string coordinates
                                symbol, x_str, y_str, z_str, x_float, y_float, z_float = atom
                                # Right-align coordinates to maintain column alignment while preserving original precision
                                outfile.write(f"{symbol: <3} {x_str: >12}  {y_str: >12}  {z_str: >12}\n")
                            else:  # Old format compatibility (fallback to 6 decimal places)
                                symbol, x, y, z = atom
                                outfile.write(f"{symbol: <3} {x: 12.6f}  {y: 12.6f}  {z: 12.6f}\n")
                        
                except IOError as e:
                    print(f"Warning: Could not read {xyz_file}: {e}")
                    continue
        
        if not quiet:
            print(f"Merged {len(sorted_files)} files into {output_filename} with {total_configs} configurations")
        
        # Create .mol file if obabel is available
        if shutil.which("obabel"):
            success, error_msg = convert_xyz_to_mol_simple(output_filename, output_filename.replace('.xyz', '.mol'))
            if success:
                if not quiet:
                    print(f"Also created {output_filename.replace('.xyz', '.mol')}")
            else:
                if not quiet:
                    print(f"Warning: Could not create .mol file: {error_msg}")
        
        return True
        
    except IOError as e:
        print(f"Error writing to {output_filename}: {e}")
        return False


def get_box_size_recommendation(input_file_path: str, packing_percent: float = 10.0) -> Optional[float]:
    """
    Runs box analysis and extracts the recommended box size for specified packing percentage.
    
    Args:
        input_file_path (str): Path to the input file
        packing_percent (float): Desired packing percentage (default: 10.0)
    
    Returns:
        Optional[float]: Recommended box size in Angstroms, or None if analysis failed
    """
    # Create a temporary state object for box analysis
    state = SystemState()
    state.verbosity_level = 0  # Suppress verbose output
    
    try:
        # Read input file
        input_file_path_full = os.path.abspath(input_file_path)
        
        # Parse the input file using existing function
        read_input_file(state, input_file_path_full)
        
        # Perform box analysis (this populates state.volume_based_recommendations)
        provide_box_length_advice(state)
        
        # Extract recommendation for specified packing percentage
        if hasattr(state, 'volume_based_recommendations'):
            key = f"{packing_percent:.1f}%"
            if key in state.volume_based_recommendations:
                return state.volume_based_recommendations[key].get('box_length_A')
        
        return None
        
    except Exception as e:
        print(f"Warning: Could not determine box size recommendation: {e}")
        return None

def update_box_size_in_input(input_file_path: str, new_box_size: float) -> str:
    """
    Updates the box size (Line 2) in an ASCEC input file.
    
    Args:
        input_file_path (str): Path to the input file
        new_box_size (float): New box size in Angstroms
    
    Returns:
        str: Modified content with updated box size
    """
    with open(input_file_path, 'r') as f:
        lines = f.readlines()
    
    # Find and update Line 2 (Simulation Cube Length)
    # Skip comment lines and empty lines
    line_count = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip empty lines and pure comment lines
        if not stripped or stripped.startswith('#'):
            continue
        line_count += 1
        
        # Line 2 is the box size
        if line_count == 2:
            # Preserve any inline comment
            if '#' in line:
                comment_part = '#' + line.split('#', 1)[1]
                lines[i] = f"{new_box_size:.1f}           {comment_part}"
            else:
                lines[i] = f"{new_box_size:.1f}           # Line 2: Simulation Cube Length (Angstroms)\n"
            break
    
    return ''.join(lines)

def create_replicated_runs(input_file_path: str, num_replicas: int, create_launcher: bool = True, box_size: Optional[float] = None, verbose: bool = True) -> List[str]:
    """
    Creates replicated folders and input files for multiple annealing runs.
    
    Args:
        input_file_path (str): Path to the original input file
        num_replicas (int): Number of replicas to create
        create_launcher (bool): Whether to create launcher script (default: True)
        box_size (Optional[float]): If provided, updates the box size in replicated files
    
    Returns:
        List[str]: List of paths to the replicated input files
    """
    input_file_path_full = os.path.abspath(input_file_path)
    input_dir = os.path.dirname(input_file_path_full)
    input_basename = os.path.basename(input_file_path_full)
    input_name, input_ext = os.path.splitext(input_basename)
    
    replicated_files = []
    
    # Create parent directory: annealing
    parent_folder_name = "annealing"
    parent_folder_path = os.path.join(input_dir, parent_folder_name)
    os.makedirs(parent_folder_path, exist_ok=True)
    if verbose:
        print(f"Creating {num_replicas} replicated runs in '{parent_folder_name}/'...")
    
    for i in range(1, num_replicas + 1):
        # Create folder name: e.g., example_1, example_2, example_3
        folder_name = f"{input_name}_{i}"
        folder_path = os.path.join(parent_folder_path, folder_name)
        
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Create the replicated input file name: e.g., example_1.in, example_2.in, example_3.in
        replicated_input_name = f"{input_name}_{i}{input_ext}"
        replicated_input_path = os.path.join(folder_path, replicated_input_name)
        
        # Copy the original input file to the new location
        try:
            with open(input_file_path_full, 'r') as src:
                content = src.read()
            
            # Update box size if specified
            if box_size is not None:
                content = update_box_size_in_input(input_file_path_full, box_size)

            content = strip_protocol_from_content(content)
            
            with open(replicated_input_path, 'w') as dst:
                dst.write(content)
            
            replicated_files.append(replicated_input_path)
            if verbose:
                print(f"  Created: {folder_name}/{replicated_input_name}")
            
        except IOError as e:
            print(f"Error creating replicated file '{replicated_input_path}': {e}")
            continue
    
    # Create launcher script only if requested
    if create_launcher and replicated_files:
        create_launcher_script(replicated_files, input_dir)
        if verbose:
            print("\nTo run all simulations sequentially, use:")
            print("  ./launcher_ascec.sh")
    
    return replicated_files


# Sort functionality - integrated from sort_files.py
def extract_base(filename):
    """Extract base name from filename by removing extension and known suffixes."""
    # Define optional suffixes that can appear before extensions
    # Note: _rescue must come before _inp/_out to handle _rescue.inp properly
    KNOWN_SUFFIXES = ['_trj', '_opt', '_property', '_gu', '_xtbrestart', '_engrad', '_xyz', '_out', '_inp', '_tmp', '_rescue']
    
    # Remove extension
    name, *_ = filename.split('.', 1)
    # Remove all known suffixes (not just the last one) by repeatedly checking
    changed = True
    while changed:
        changed = False
        for suffix in KNOWN_SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
                break  # Start over to handle multiple suffixes
    return name

def group_files_by_base(directory='.'):
    """Group files by base name and move them to folders using parallel processing."""
    import multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Filter out ORCA intermediate files right from the start
    excluded_patterns = ['.scfgrad.', '.scfp.', '.tmp.', '.densities.', '.scfhess.']
    files = [f for f in os.listdir(directory) 
             if os.path.isfile(os.path.join(directory, f))
             and not any(pattern in f for pattern in excluded_patterns)]
    base_map = defaultdict(list)

    print(f"Analyzing {len(files)} files for grouping...")
    
    # Define function to extract base for a batch of files
    def extract_bases_batch(file_batch):
        batch_map = defaultdict(list)
        for file in file_batch:
            base = extract_base(file)
            if base:
                batch_map[base].append(file)
        return batch_map

    # Use parallel processing for base extraction if we have many files
    if len(files) > 100:
        # Split files into batches for parallel processing
        batch_size = max(10, len(files) // mp.cpu_count())
        file_batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
        
        max_workers = get_optimal_workers('cpu_intensive', len(file_batches))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch processing tasks
            future_to_batch = {executor.submit(extract_bases_batch, batch): batch 
                              for batch in file_batches}
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                try:
                    batch_map = future.result()
                    for base, file_list in batch_map.items():
                        base_map[base].extend(file_list)
                except Exception as e:
                    print(f"Error processing batch: {e}")
    else:
        # For smaller file sets, process directly
        for file in files:
            base = extract_base(file)
            if base:
                base_map[base].append(file)

    # Move files using parallel processing
    def move_file_group(group_data):
        base, grouped_files, directory = group_data
        moved_files = []
        
        if len(grouped_files) > 1:
            folder_path = os.path.join(directory, base)
            os.makedirs(folder_path, exist_ok=True)
            
            for file in grouped_files:
                try:
                    src = os.path.join(directory, file)
                    dest = os.path.join(folder_path, file)
                    shutil.move(src, dest)
                    moved_files.append(file)
                except Exception as e:
                    print(f"Error moving {file}: {e}")
            
            if moved_files:
                return f"Moved {len(moved_files)} files to folder: {base}", len(moved_files)
        
        return None, 0

    # Prepare groups for parallel moving
    groups_to_move = [(base, grouped_files, directory) 
                     for base, grouped_files in base_map.items() 
                     if len(grouped_files) > 1]
    
    moved_count = 0
    
    if groups_to_move:
        print(f"Moving files in {len(groups_to_move)} groups using parallel processing...")
        
        max_workers = get_optimal_workers('io_intensive', len(groups_to_move))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all moving tasks
            future_to_group = {executor.submit(move_file_group, group): group[0] 
                              for group in groups_to_move}
            
            # Collect results as they complete
            for future in as_completed(future_to_group):
                try:
                    message, count = future.result()
                    if message:
                        print(message)
                        moved_count += count
                except Exception as e:
                    print(f"Error in file moving: {e}")
    
    if moved_count == 0:
        print("No files needed to be grouped.")
    else:
        print(f"Total files moved: {moved_count}")

# Merge XYZ functionality - integrated from mergexyz_files.py
def get_sort_key(filename):
    """Extract the configuration number from filename for sorting.
    
    Handles various patterns:
    - umotif_28_opt.xyz -> 28
    - motif_28_opt.xyz -> 28
    - opt_conf_3.xyz -> 3
    - result_123.xyz -> 123
    - any_456.xyz -> 456
    """
    import re
    # Try multiple patterns in order of specificity
    patterns = [
        r'u?motif_(\d+)_opt\.xyz',   # umotif_28_opt.xyz or motif_28_opt.xyz
        r'u?motif_(\d+)\.xyz',        # umotif_28.xyz or motif_28.xyz
        r'opt_conf_(\d+)\.xyz',      # opt_conf_3.xyz
        r'result_(\d+)\.xyz',        # result_123.xyz
        r'conf_(\d+)\.xyz',          # conf_20.xyz
        r'_(\d+)\.xyz',              # any_123.xyz (general pattern)
        r'(\d+)\.xyz'                # 123.xyz (number only)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    
    return float('inf')




def combine_xyz_files(output_filename="combined_results.xyz", exclude_pattern="_trj.xyz"):
    """Combine all relevant .xyz files into a single .xyz file."""
    
    all_xyz_files = []
    for root, _, files in os.walk("."):
        for file in files:
            filepath = os.path.join(root, file)
            if not filepath.endswith(".xyz"):
                continue
            # Never merge existing aggregate files into combined_results.
            if file == output_filename or file.startswith("combined_r") or file.startswith("combined_results"):
                continue
            if exclude_pattern in file:
                continue
            all_xyz_files.append(filepath)

    if not all_xyz_files:
        print(f"No relevant .xyz files found (excluding '{exclude_pattern}' and '{output_filename}').")
        return False

    # Sort the files based on the first configuration number found
    sorted_xyz_files = sorted(all_xyz_files, key=lambda x: get_sort_key(os.path.basename(x)))

    with open(output_filename, "w") as outfile:
        for xyz_file in sorted_xyz_files:
            print(f"Processing: {xyz_file}")
            with open(xyz_file, "r") as infile:
                lines = infile.readlines()
                outfile.writelines(lines)

    print(f"\nSuccessfully combined {len(sorted_xyz_files)}.xyz files into: {output_filename}")
    
    # Create .mol file if obabel is available
    if shutil.which("obabel"):
        success, error_msg = convert_xyz_to_mol_simple(output_filename, output_filename.replace('.xyz', '.mol'))
        if success:
            print(f"Also created {output_filename.replace('.xyz', '.mol')}")
        else:
            print(f"Warning: Could not create .mol file: {error_msg}")
    
    return True

# MOL conversion functionality - integrated from mol_files.py
def convert_xyz_to_mol_simple(input_xyz, output_mol):
    """Convert an XYZ file to a MOL file using Open Babel."""
    try:
        # Make paths absolute
        input_xyz = os.path.abspath(input_xyz)
        output_mol = os.path.abspath(output_mol)

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_mol)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        command = ["obabel", "-i", "xyz", input_xyz, "-o", "mol", "-O", output_mol]
        subprocess.run(command, check=True, capture_output=True)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, f"Error converting {input_xyz}: {e.stderr.decode()}"
    except FileNotFoundError:
        return False, "Error: Open Babel ('obabel') command not found. Make sure it's installed and in your system's PATH."
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"

def create_combined_mol():
    """Create MOL file from combined_results.xyz."""
    if os.path.exists("combined_results.xyz"):
        success, error_message = convert_xyz_to_mol_simple("combined_results.xyz", "combined_results.mol")
        if success:
            print("Successfully created combined_results.mol")
            return True
        else:
            print(f"Failed to create MOL file: {error_message}")
            return False
    else:
        print("No combined_results.xyz file found to convert.")
        return False


# ============================================================================
# OPI (ORCA Python Interface) Helper Functions for ORCA 6.1+ Support
# ============================================================================

def detect_orca_version(logfile_path: str) -> Optional[Tuple[int, int]]:
    """
    Detect ORCA version from output file.
    
    Args:
        logfile_path: Path to ORCA output file (.out)
        
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


def check_orca_terminated_normally_opi(output_path: str) -> bool:
    """
    Check if ORCA calculation terminated normally using OPI for ORCA 6.1+.
    Falls back to text search for ORCA 5.x or when OPI is not available.
    
    Args:
        output_path: Path to ORCA output file (.out)
        
    Returns:
        True if terminated normally, False otherwise
    """
    if not os.path.exists(output_path):
        return False
    
    # Detect ORCA version
    version = detect_orca_version(output_path)
    
    # For ORCA 6.1+, try to use OPI
    if version and version >= (6, 1) and OPI_AVAILABLE and OPIOutput is not None:
        try:
            file_path = Path(output_path)
            basename = file_path.stem
            working_dir = file_path.parent
            
            opi_output = OPIOutput(
                basename=basename,
                working_dir=working_dir,
                version_check=False
            )
            
            return opi_output.terminated_normally()
        except Exception:
            # Fall back to text search if OPI fails
            pass
    
    # Fall back to text search (works for all ORCA versions)
    try:
        with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return '****ORCA TERMINATED NORMALLY****' in content
    except Exception:
        return False


def detect_convergence_status(logfile_path: str) -> dict:
    """
    Detect the convergence status of an ORCA optimization.
    
    Args:
        logfile_path: Path to ORCA output file
        
    Returns:
        dict with keys:
            'status': 'converged', 'not_converged', 'error', or 'unknown'
            'terminated_normally': bool
            'max_iterations_reached': bool
            'error_message': str or None
    """
    result = {
        'status': 'unknown',
        'terminated_normally': False,
        'max_iterations_reached': False,
        'error_message': None
    }
    
    try:
        with open(logfile_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check for normal termination
        if 'ORCA TERMINATED NORMALLY' in content:
            result['terminated_normally'] = True
            result['status'] = 'converged'
        
        # Check for optimization not converged error (max iterations reached)
        not_converged_patterns = [
            'The optimization did not converge but reached the maximum',
            'maximum number of optimization cycles',
            'ORCA will abort at this point',
            'Please restart the calculation with the lowest energy geometry',
            'larger maxiter for the geometry optimization'
        ]
        
        for pattern in not_converged_patterns:
            if pattern in content:
                result['max_iterations_reached'] = True
                result['status'] = 'not_converged'
                result['error_message'] = 'Optimization reached maximum cycles without converging'
                break
        
        # Also check for explicit convergence success
        if '*** THE OPTIMIZATION HAS CONVERGED ***' in content:
            result['status'] = 'converged'
            result['max_iterations_reached'] = False
        
        # If terminated normally but no explicit convergence indicator, check for geometry opt
        if result['terminated_normally'] and '*** OPTIMIZATION RUN DONE ***' in content:
            if result['status'] == 'unknown':
                result['status'] = 'converged'
        
        # Check for SCF convergence issues
        if 'SCF NOT CONVERGED' in content or 'The wavefunction could not be converged' in content:
            result['status'] = 'error'
            result['error_message'] = 'SCF did not converge'
        
    except Exception as e:
        result['error_message'] = str(e)
    
    return result


def detect_orca_version_from_executable(orca_path: Optional[str] = None) -> Optional[Tuple[int, int]]:
    """
    Detect ORCA version from the executable by running 'orca --version'.
    
    Args:
        orca_path: Path to ORCA executable. If None, uses 'orca' from PATH.
        
    Returns:
        Tuple of (major, minor) version numbers, or None if not detected
    """
    if orca_path is None:
        # Try to find ORCA in PATH
        orca_path = shutil.which("orca")
        if not orca_path:
            return None
    
    try:
        result = subprocess.run(
            [orca_path, "--version"],
            capture_output=True,
            text=True
        )
        # Parse version from output (works for both stdout and stderr)
        output = result.stdout + result.stderr
        # Look for patterns like "Program Version 6.1.0" or just "6.1.0"
        match = re.search(r'(?:Program Version\s+)?(\d+)\.(\d+)', output, re.IGNORECASE)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            return (major, minor)
    except (FileNotFoundError, Exception):
        pass
    return None


def detect_orca_version_from_launcher(launcher_content: str) -> Optional[Tuple[int, int]]:
    """
    Detect ORCA version from launcher variable names like ORCA611_ROOT.
    
    This is a fallback when the ORCA executable is not yet in PATH.
    Parses variable names following the pattern ORCA{major}{minor}[patch]_ROOT.
    
    Examples:
        ORCA611_ROOT -> (6, 1)  # ORCA 6.1.1
        ORCA61_ROOT  -> (6, 1)  # ORCA 6.1
        ORCA6_ROOT   -> (6, 0)  # ORCA 6.x (minor assumed 0)
        ORCA5_ROOT   -> (5, 0)  # ORCA 5.x
        ORCA_ROOT    -> None    # No version info
        
    Args:
        launcher_content: Content of the launcher script
        
    Returns:
        Tuple of (major, minor) version numbers, or None if not detected
    """
    if not launcher_content:
        return None
    
    # Look for ORCA version variable patterns: ORCA611_ROOT, ORCA61_ROOT, ORCA6_ROOT
    # Pattern: ORCA followed by digits, then _ROOT
    pattern = r'ORCA(\d)(\d*)_ROOT'
    
    for line in launcher_content.split('\n'):
        match = re.search(pattern, line)
        if match:
            major = int(match.group(1))
            minor_str = match.group(2)
            if minor_str:
                # First digit after major is the minor version
                # e.g., ORCA611_ROOT: major=6, minor_str="11" -> minor=1
                # e.g., ORCA61_ROOT: major=6, minor_str="1" -> minor=1
                minor = int(minor_str[0])
            else:
                minor = 0
            return (major, minor)
    
    return None


def detect_orca_version_combined(orca_path: Optional[str] = None, 
                                  launcher_content: Optional[str] = None) -> Optional[Tuple[int, int]]:
    """
    Detect ORCA version using multiple methods:
    1. From executable (orca --version)
    2. From launcher variable names (ORCA611_ROOT)
    
    Args:
        orca_path: Path to ORCA executable (optional)
        launcher_content: Content of the launcher script (optional)
        
    Returns:
        Tuple of (major, minor) version numbers, or None if not detected
    """
    # Try executable first (most reliable)
    version = detect_orca_version_from_executable(orca_path)
    if version:
        return version
    
    # Fall back to launcher variable parsing
    if launcher_content:
        version = detect_orca_version_from_launcher(launcher_content)
        if version:
            return version
    
    return None


# xTB method synonyms - maps all synonyms to canonical form
XTB_SYNONYMS = {
    # Native xTB (ORCA 6.1+)
    'NATIVE-GFN-XTB': 'NATIVE-GFN-XTB',
    'NATIVE-GFN1-XTB': 'NATIVE-GFN-XTB',
    'NATIVE-XTB1': 'NATIVE-GFN-XTB',
    'NATIVE-GFN2-XTB': 'NATIVE-GFN2-XTB',
    'NATIVE-XTB2': 'NATIVE-GFN2-XTB',
    # Non-native xTB (ORCA 5.x, requires external xtb)
    'GFN0-XTB': 'GFN0-XTB',
    'XTB0': 'GFN0-XTB',
    'GFN-XTB': 'GFN-XTB',
    'XTB1': 'GFN-XTB',
    'GFN2-XTB': 'GFN2-XTB',
    'XTB2': 'GFN2-XTB',
    'GFN-FF': 'GFN-FF',
    'XTBFF': 'GFN-FF',
}

# Mapping from non-native to native xTB methods (for ORCA 6.1+ upgrade)
XTB_NATIVE_MAP = {
    'GFN-XTB': 'Native-GFN-xTB',
    'XTB1': 'Native-GFN-xTB',
    'GFN2-XTB': 'Native-GFN2-xTB',
    'XTB2': 'Native-GFN2-xTB',
}


def is_xtb_method(method: str) -> bool:
    """
    Check if a method string is an xTB method.
    
    Args:
        method: Method string (e.g., "Native-GFN2-xTB", "GFN2-xTB", "XTB2")
        
    Returns:
        True if it's an xTB method, False otherwise
    """
    method_upper = method.upper().replace('_', '-')
    return method_upper in XTB_SYNONYMS or any(kw in method_upper for kw in ['GFN', 'XTB'])


def convert_xtb_for_orca_version(method: str, orca_path: Optional[str] = None, 
                                   launcher_content: Optional[str] = None) -> str:
    """
    Convert xTB method to the appropriate version for the detected ORCA version.
    
    - ORCA 6.1+: Convert to native xTB methods (Native-GFN-xTB, Native-GFN2-xTB)
    - ORCA 5.x: Keep non-native xTB methods (GFN-xTB, GFN2-xTB)
    
    Args:
        method: xTB method string
        orca_path: Path to ORCA executable (for version detection)
        launcher_content: Launcher script content (for version detection from ORCA611_ROOT etc.)
        
    Returns:
        Appropriate xTB method for the ORCA version
    """
    method_upper = method.upper().replace('_', '-')
    
    # Detect ORCA version (tries executable first, then launcher variable names)
    version = detect_orca_version_combined(orca_path, launcher_content)
    is_orca_61_plus = version and version[0] >= 6 and version[1] >= 1
    
    # If already native and ORCA 6.1+, keep it
    if 'NATIVE' in method_upper and is_orca_61_plus:
        return method
    
    # If native but ORCA < 6.1, convert to non-native
    if 'NATIVE' in method_upper and not is_orca_61_plus:
        if 'GFN2' in method_upper or 'XTB2' in method_upper:
            return 'GFN2-xTB'
        elif 'GFN' in method_upper or 'XTB1' in method_upper:
            return 'GFN-xTB'
        return 'GFN2-xTB'  # Default to GFN2
    
    # If non-native and ORCA 6.1+, convert to native
    if method_upper in XTB_NATIVE_MAP and is_orca_61_plus:
        return XTB_NATIVE_MAP[method_upper]
    
    # Return as-is
    return method


def detect_xtb_in_template(template_content: str) -> Optional[str]:
    """
    Detect if the template uses an xTB method in the ! line.
    
    Args:
        template_content: Content of the ORCA input template
        
    Returns:
        The xTB method found (canonical form), or None if not xTB
    """
    # Look for the ! line(s) in the template
    for line in template_content.split('\n'):
        line_stripped = line.strip()
        if line_stripped.startswith('!'):
            # Extract all tokens from the ! line
            tokens = line_stripped[1:].split()
            for token in tokens:
                token_upper = token.upper().replace('_', '-')
                # Check if this token is an xTB method
                if token_upper in XTB_SYNONYMS:
                    return token  # Return original case
                # Also check for partial matches
                if any(kw in token_upper for kw in ['GFN', 'NATIVE-GFN', 'NATIVE-XTB']):
                    return token
    return None


def detect_orca_executable(alias: str = "orca") -> Optional[str]:
    """
    Detect ORCA executable path using 'which <alias>' or shutil.which.
    
    Args:
        alias: The command alias to search for (default: "orca")
               This can come from line 9 of input.in (e.g., "orca", "orca6", etc.)
    
    Returns:
        Full path to ORCA executable, or None if not found
    """
    # Try shutil.which first (works cross-platform)
    orca_path = shutil.which(alias)
    if orca_path:
        return orca_path
    
    # Fallback: try running 'which <alias>' in shell
    try:
        result = subprocess.run(
            ["which", alias],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            path = result.stdout.strip()
            if os.path.exists(path):
                return path
    except (FileNotFoundError, Exception):
        pass
    
    return None


def create_auto_launcher(output_dir: str, qm_program: str = "orca", orca_alias: str = "orca",
                         quiet: bool = False) -> Optional[str]:
    """
    Create an auto-generated launcher script when no launcher file is provided.
    
    This function detects the ORCA installation path using the alias from input.in
    and creates a proper launcher script with OpenMPI environment setup.
    
    Args:
        output_dir: Directory to create the launcher script in
        qm_program: QM program type ('orca' or 'gaussian')
        orca_alias: The ORCA executable alias from input.in line 9 (default: "orca")
        
    Returns:
        Path to the created launcher script, or None if ORCA not found
    """
    if qm_program != 'orca':
        if not quiet:
            print(f"  Warning: Auto-launcher only supported for ORCA, not {qm_program}")
        return None
    
    # Detect ORCA executable using the alias
    orca_path = detect_orca_executable(orca_alias)
    if not orca_path:
        if not quiet:
            print(f"  Warning: ORCA executable '{orca_alias}' not found in PATH. Please provide a launcher script.")
        return None
    
    if not quiet:
        print(f"  Auto-detected ORCA at: {orca_path}")
    
    # Create launcher script
    launcher_path = os.path.join(output_dir, "launcher_orca.sh")
    
    # Get ORCA directory (for parallel execution, ORCA needs full path)
    orca_dir = os.path.dirname(orca_path)
    
    # Try to detect OpenMPI directory (commonly in same parent as ORCA or in software folder)
    orca_base = os.path.dirname(orca_dir)  # e.g., /home/user/software
    
    # Look for OpenMPI installation in common locations
    openmpi_dir = None
    openmpi_search_patterns = [
        os.path.join(orca_base, "openmpi*"),
        os.path.join(orca_base, "OpenMPI*"),
        "/usr/lib/openmpi*",
        "/usr/local/lib/openmpi*",
    ]
    for pattern in openmpi_search_patterns:
        matches = glob.glob(pattern)
        if matches:
            # Sort and take the latest version
            matches.sort(reverse=True)
            for match in matches:
                if os.path.isdir(match):
                    openmpi_dir = match
                    break
            if openmpi_dir:
                break
    
    # Try to detect ORCA version from directory name (e.g., orca_6_1_1 -> 611)
    orca_version_str = ""
    orca_dir_name = os.path.basename(orca_dir)
    version_match = re.search(r'orca[_-]?(\d+)[_.]?(\d+)?[_.]?(\d+)?', orca_dir_name, re.IGNORECASE)
    if version_match:
        major = version_match.group(1) or ""
        minor = version_match.group(2) or ""
        patch = version_match.group(3) or ""
        orca_version_str = f"{major}{minor}{patch}"  # e.g., "611"
    
    # Build the launcher content using the default template structure
    if openmpi_dir:
        openmpi_version = os.path.basename(openmpi_dir)
        launcher_content = f'''#!/bin/bash
# Auto-generated launcher script for ORCA
# Created by ASCEC

# Define the paths to your ORCA installation
export ORCA_BASE="{orca_base}"
export ORCA{orca_version_str}_ROOT="{orca_dir}"
export OPENMPI_ROOT="{openmpi_dir}"

# Save system paths to prevent infinite nesting if sourced multiple times
_SYSTEM_PATH="$PATH"
_SYSTEM_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

# Prepend ORCA and OpenMPI to the system paths
export PATH="$ORCA{orca_version_str}_ROOT:$OPENMPI_ROOT/bin:$_SYSTEM_PATH"
export LD_LIBRARY_PATH="$ORCA{orca_version_str}_ROOT:$OPENMPI_ROOT/lib:$_SYSTEM_LD_LIBRARY_PATH"

###
'''
    else:
        # No OpenMPI found - create simpler launcher
        launcher_content = f'''#!/bin/bash
# Auto-generated launcher script for ORCA
# Created by ASCEC

# Define the paths to your ORCA installation
export ORCA_BASE="{orca_base}"
export ORCA{orca_version_str}_ROOT="{orca_dir}"

# Save system paths to prevent infinite nesting if sourced multiple times
_SYSTEM_PATH="$PATH"
_SYSTEM_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

# Prepend ORCA to the system paths
export PATH="$ORCA{orca_version_str}_ROOT:$_SYSTEM_PATH"
export LD_LIBRARY_PATH="$ORCA{orca_version_str}_ROOT:$_SYSTEM_LD_LIBRARY_PATH"

###
'''
    
    try:
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        os.chmod(launcher_path, 0o755)
        return launcher_path
    except Exception as e:
        if not quiet:
            print(f"  Warning: Could not create launcher script: {e}")
        return None


def parse_rescue_method(input_content: str, orca_path: Optional[str] = None,
                         launcher_content: Optional[str] = None) -> Tuple[str, bool]:
    """
    Parse the #rescue(Method) directive from an ORCA input file, or detect
    the rescue method from the template's method line.
    
    Logic:
    1. If #rescue(method) is specified, use that method
    2. If #rescue(method,num) is specified, use that method with NumFreq
    3. If template uses an xTB method, reuse that method with NumFreq
       (converting to native for ORCA 6.1+ or non-native for 5.x)
    4. Default: HF-3c with Freq
    
    Example input:
        #rescue(Native-GFN2-xTB)       -> Uses Native-GFN2-xTB with NumFreq (xTB)
        #rescue(b97-c)                 -> Uses b97-c with Freq
        #rescue(b97-c,num)             -> Uses b97-c with NumFreq
        Template: ! Native-GFN2-xTB Opt -> Uses Native-GFN2-xTB with NumFreq
        Template: ! B98-3c Opt         -> Uses HF-3c with Freq (default)
    
    Args:
        input_content: Content of the ORCA input file
        orca_path: Optional path to ORCA executable (for version detection)
        launcher_content: Optional launcher script content (for version detection from ORCA611_ROOT etc.)
        
    Returns:
        Tuple of (rescue_method, use_numfreq) where:
        - rescue_method: Method string (e.g., "Native-GFN2-xTB", "HF-3c")
        - use_numfreq: True if NumFreq should be used, False for Freq
    """
    def _parse_rescue_spec(spec: str) -> Tuple[str, Optional[bool]]:
        """Parse rescue specifiers like method,num or method/num into method + freq mode."""
        raw = spec.strip()
        if not raw:
            return ('', None)

        # Accept both comma and slash qualifiers, e.g.:
        #   Native-GFN2-xTB/num, HF-3c/freq, b97-c,num
        method = raw
        freq_mode: Optional[bool] = None

        if '/' in raw:
            base, suffix = raw.split('/', 1)
            method = base.strip()
            suffix_token = suffix.strip().lower()
            if suffix_token in ('num', 'numfreq'):
                freq_mode = True
            elif suffix_token in ('freq',):
                freq_mode = False
        elif ',' in raw:
            parts = [p.strip() for p in raw.split(',') if p.strip()]
            method = parts[0]
            for token in parts[1:]:
                token_l = token.lower()
                if token_l in ('num', 'numfreq'):
                    freq_mode = True
                elif token_l in ('freq',):
                    freq_mode = False

        return (method, freq_mode)

    # Look for #rescue(method) / #rescue(method,num) / #rescue(method/num) pattern
    match = re.search(r'#rescue\(([^)]+)\)', input_content, re.IGNORECASE)
    if match:
        method, explicit_numfreq = _parse_rescue_spec(match.group(1))
        if not method:
            return ('HF-3c', False)

        # Convert xTB method for ORCA version and default to NumFreq unless explicitly overridden.
        if is_xtb_method(method):
            method = convert_xtb_for_orca_version(method, orca_path, launcher_content)
            if explicit_numfreq is not None:
                return (method, explicit_numfreq)
            return (method, True)

        # Non-xTB methods default to Freq unless explicitly overridden.
        if explicit_numfreq is not None:
            return (method, explicit_numfreq)
        return (method, False)
    
    # No #rescue directive - check if template uses xTB method
    xtb_method = detect_xtb_in_template(input_content)
    if xtb_method:
        # Reuse the template's xTB method, converted for ORCA version
        converted_method = convert_xtb_for_orca_version(xtb_method, orca_path, launcher_content)
        return (converted_method, True)  # xTB methods use NumFreq
    
    # Default: HF-3c with Freq
    return ('HF-3c', False)


def generate_rescue_hessian_input(template_content: str, rescue_method: str, xyz_coords: str, 
                                   charge: int = 0, multiplicity: int = 1, 
                                   nprocs: int = 8, use_numfreq: Optional[bool] = None) -> str:
    """
    Generate an ORCA input file for calculating the Hessian with the rescue method.
    
    Args:
        template_content: Original input file content (for extracting settings)
        rescue_method: Method to use for Hessian calculation (e.g., "Native-GFN2-xTB")
        xyz_coords: XYZ coordinates block (without first two lines)
        charge: Molecular charge
        multiplicity: Spin multiplicity
        nprocs: Number of processors to use
        use_numfreq: If True, use NumFreq; if False, use Freq; if None, auto-detect
        
    Returns:
        ORCA input file content for Hessian calculation
    """
    # Check if it's a semiempirical/xTB method (no basis set needed)
    rescue_upper = rescue_method.upper()
    is_xtb = any(kw in rescue_upper for kw in ['GFN', 'XTB'])
    is_semiempirical = any(kw in rescue_upper for kw in ['PM3', 'PM6', 'PM7', 'AM1', 'MNDO'])
    
    # Determine whether to use NumFreq or Freq
    # If use_numfreq is explicitly set, use that; otherwise auto-detect
    if use_numfreq is None:
        use_numfreq = is_xtb  # xTB methods default to NumFreq
    
    lines = []
    lines.append(f"# Rescue Hessian calculation with {rescue_method}")
    
    # Build the method line
    freq_keyword = "NumFreq" if use_numfreq else "Freq"
    if is_xtb or is_semiempirical:
        lines.append(f"! {rescue_method} {freq_keyword}")
    else:
        lines.append(f"! {rescue_method} {freq_keyword} TightSCF")
    
    lines.append("")
    
    # %pal block - use the same nprocs as the template
    lines.append(f"%pal")
    lines.append(f"  nprocs {nprocs}")
    lines.append(f"end")
    
    lines.append("")
    lines.append(f"* xyz {charge} {multiplicity}")
    lines.append(xyz_coords.strip())
    lines.append("*")
    lines.append("")
    
    return "\n".join(lines)


def enable_hessian_restart(input_path: str, hessian_path: str) -> bool:
    """
    Modify an ORCA input file to use a pre-calculated Hessian for the optimization.
    
    This function automatically adds inhess read and inhessname directives to the
    %geom block. If no %geom block exists, one is created. If the user included
    #inhess read and #inhessname placeholders, they are uncommented.
    
    Args:
        input_path: Path to the ORCA input file to modify
        hessian_path: Path to the .hess file to read
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Get basename of hess file
        hess_basename = os.path.basename(hessian_path)
        
        # First, check if placeholders exist and uncomment them
        has_inhess_placeholder = re.search(r'#\s*inhess\s+read', content, re.IGNORECASE)
        has_inhessname_placeholder = re.search(r'#\s*inhessname\s+["\'][^"\']*["\']', content, re.IGNORECASE)
        
        if has_inhess_placeholder or has_inhessname_placeholder:
            # User provided placeholders - uncomment them
            content = re.sub(r'#\s*inhess\s+read', 'inhess read', content, flags=re.IGNORECASE)
            content = re.sub(
                r'#\s*inhessname\s+["\'][^"\']*["\']',
                f'inhessname "{hess_basename}"',
                content,
                flags=re.IGNORECASE
            )
            # If inhessname wasn't in the file, add it after inhess read
            if 'inhess read' in content.lower() and 'inhessname' not in content.lower():
                content = re.sub(
                    r'(inhess\s+read)',
                    f'\\1\n  inhessname "{hess_basename}"',
                    content,
                    flags=re.IGNORECASE
                )
        else:
            # No placeholders - auto-add to %geom block
            # Check if %geom block already exists
            geom_match = re.search(r'(%geom\b.*?)(end)', content, re.IGNORECASE | re.DOTALL)
            
            if geom_match:
                # %geom block exists - add inhess directives at the end
                geom_block = geom_match.group(1)
                # Check if inhess already exists (shouldn't, but be safe)
                if 'inhess' not in geom_block.lower():
                    new_geom_block = geom_block.rstrip() + f'\n  inhess read\n  inhessname "{hess_basename}"\n'
                    content = content[:geom_match.start()] + new_geom_block + 'end' + content[geom_match.end():]
            else:
                # No %geom block - create one
                # Insert before the coordinate block (* xyz or * xyzfile)
                coord_match = re.search(r'^\s*\*\s*(xyz|xyzfile)', content, re.IGNORECASE | re.MULTILINE)
                if coord_match:
                    geom_block = f'\n%geom\n  inhess read\n  inhessname "{hess_basename}"\nend\n\n'
                    content = content[:coord_match.start()] + geom_block + content[coord_match.start():]
                else:
                    # Fallback: append to end (before last * if it exists)
                    # This is a less common case
                    geom_block = f'\n%geom\n  inhess read\n  inhessname "{hess_basename}"\nend\n'
                    # Insert before the last section
                    content += geom_block
        
        with open(input_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"  Warning: Could not enable Hessian restart: {e}")
        return False


def run_rescue_hessian_calculation(xyz_file: str, rescue_method: str, launcher_path: str,
                                     charge: int = 0, multiplicity: int = 1,
                                     nprocs: int = 8, verbose: bool = False,
                                     output_basename: Optional[str] = None,
                                     use_numfreq: Optional[bool] = None) -> Optional[str]:
    """
    Run a rescue Hessian calculation for a structure that failed to converge.
    
    Args:
        xyz_file: Path to XYZ file with the geometry
        rescue_method: Method to use (e.g., "Native-GFN2-xTB")
        launcher_path: Path to launcher script
        charge: Molecular charge
        multiplicity: Spin multiplicity
        nprocs: Number of processors
        verbose: If True, print detailed error information
        output_basename: Base name for output files (e.g., "opt_conf_8" -> "opt_conf_8_rescue.hess")
                        If None, derives from xyz_file name
        use_numfreq: If True, use NumFreq; if False, use Freq; if None, auto-detect
        
    Returns:
        Path to the generated .hess file, or None if failed
    """
    workflow_ctx = getattr(sys, '_current_workflow_context', None)
    workflow_concise = bool(
        workflow_ctx
        and getattr(workflow_ctx, 'is_workflow', False)
        and getattr(workflow_ctx, 'workflow_verbose_level', 0) < 1
        and not verbose
    )

    def _rescue_log(message: str) -> None:
        if not workflow_concise:
            print(message)

    # Get working directory and basename
    working_dir = os.path.dirname(xyz_file)
    if not working_dir:
        working_dir = "."
    # Make working_dir absolute for reliable path handling
    working_dir = os.path.abspath(working_dir)
    
    # Use provided basename or derive from xyz_file
    if output_basename:
        basename = output_basename
    else:
        basename = os.path.splitext(os.path.basename(xyz_file))[0]
    
    # Read XYZ coordinates
    try:
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
        # Skip first two lines (atom count and comment)
        xyz_coords = "".join(lines[2:])
    except Exception as e:
        _rescue_log(f"  Error reading XYZ file: {e}")
        return None
    
    # Generate rescue input (pass use_numfreq for explicit control)
    rescue_input = generate_rescue_hessian_input(
        "", rescue_method, xyz_coords, charge, multiplicity, nprocs, use_numfreq
    )
    
    # Write rescue input file
    rescue_inp_path = os.path.join(working_dir, f"{basename}_rescue.inp")
    rescue_out_path = os.path.join(working_dir, f"{basename}_rescue.out")
    rescue_hess_path = os.path.join(working_dir, f"{basename}_rescue.hess")
    
    with open(rescue_inp_path, 'w') as f:
        f.write(rescue_input)
    
    # Read launcher for environment setup
    try:
        with open(launcher_path, 'r') as f:
            launcher_content = f.read()
    except Exception as e:
        _rescue_log(f"  Error reading launcher: {e}")
        return None
    
    # Extract environment setup from launcher (before ### separator if present)
    env_setup = launcher_content.split('###')[0]
    
    # Ensure shebang is present
    if not env_setup.strip().startswith('#!'):
        env_setup = "#!/bin/bash\nset -e\n\n" + env_setup
    
    # Find the ORCA root variable name from the launcher (e.g., ORCA611_ROOT, ORCA_ROOT, etc.)
    # Look for patterns like: export ORCA*_ROOT=... or ORCA*_ROOT=...
    orca_root_var = None
    for line in env_setup.split('\n'):
        # Match patterns like ORCA611_ROOT, ORCA_ROOT, ORCA5_ROOT, etc.
        match = re.search(r'\b(ORCA\w*_ROOT)\s*=', line)
        if match:
            orca_root_var = match.group(1)
            # Prefer more specific names (e.g., ORCA611_ROOT over ORCA_ROOT)
            if 'ORCA_ROOT' not in orca_root_var or orca_root_var != 'ORCA_ROOT':
                break  # Use the specific one
    
    # Create run script with full paths
    temp_script = os.path.join(working_dir, f'_run_rescue_{basename}.sh')
    with open(temp_script, 'w') as f:
        f.write(env_setup)
        f.write("\n\n")
        f.write(f"cd \"{working_dir}\"\n")
        # Use the ORCA root variable from the launcher if found, otherwise fall back to PATH
        if orca_root_var:
            f.write(f"\"${orca_root_var}/orca\" \"{os.path.basename(rescue_inp_path)}\" > \"{os.path.basename(rescue_out_path)}\" 2>&1\n")
        else:
            # Fallback: use orca from PATH
            f.write(f"orca \"{os.path.basename(rescue_inp_path)}\" > \"{os.path.basename(rescue_out_path)}\" 2>&1\n")
    
    os.chmod(temp_script, 0o755)
    
    # Run calculation with timeout
    try:
        result = subprocess.run(
            ['bash', temp_script],
            cwd=working_dir,
            capture_output=True,
            text=True
        )
    except Exception as e:
        _rescue_log(f"  ✗ Rescue Hessian calculation error: {e}")
        if os.path.exists(temp_script):
            os.remove(temp_script)
        return None
    
    # Cleanup script
    if os.path.exists(temp_script):
        os.remove(temp_script)
    
    # Check if Hessian was generated
    if os.path.exists(rescue_hess_path):
        # Success - return path silently (calling code will print status)
        return rescue_hess_path
    else:
        # Detailed error reporting
        error_msg = "  ✗ Rescue Hessian calculation failed"
        
        # Check if output file exists and look for errors
        if os.path.exists(rescue_out_path):
            try:
                with open(rescue_out_path, 'r', encoding='utf-8', errors='replace') as f:
                    out_content = f.read()
                
                # Check for common ORCA errors
                if "ORCA TERMINATED NORMALLY" in out_content:
                    # Calculation finished but no .hess file - check if freq was actually requested
                    if ".hess" not in out_content and "VIBRATIONAL FREQUENCIES" not in out_content:
                        error_msg += " (Freq calculation may not be supported for this method)"
                elif "ERROR" in out_content.upper():
                    # Find the error message
                    error_lines = [line for line in out_content.split('\n') 
                                   if 'ERROR' in line.upper() or 'ABORTING' in line.upper()]
                    if error_lines:
                        error_msg += f"\n    Error: {error_lines[0][:100]}"
                elif "ORCA finished by error termination" in out_content:
                    error_msg += " (ORCA error termination)"
                elif len(out_content) < 100:
                    error_msg += " (Output file is nearly empty - ORCA may not have started)"
                    if verbose:
                        error_msg += f"\n    Launcher check: {result.stderr[:200] if result.stderr else 'No stderr'}"
            except Exception:
                pass
        else:
            error_msg += " (No output file generated - ORCA may not have been found)"
            if verbose and result.stderr:
                error_msg += f"\n    stderr: {result.stderr[:200]}"
        
        _rescue_log(error_msg)
        return None


def extract_orca_energy_opi(output_path: str) -> Optional[float]:
    """
    Extract final energy from ORCA output using OPI for ORCA 6.1+.
    Falls back to regex for ORCA 5.x or when OPI is not available.
    
    Args:
        output_path: Path to ORCA output file (.out)
        
    Returns:
        Final single point energy in Hartree, or None if not found
    """
    if not os.path.exists(output_path):
        return None
    
    # Detect ORCA version
    version = detect_orca_version(output_path)
    
    # For ORCA 6.1+, try to use OPI
    if version and version >= (6, 1) and OPI_AVAILABLE and OPIOutput is not None:
        try:
            file_path = Path(output_path)
            basename = file_path.stem
            working_dir = file_path.parent
            
            opi_output = OPIOutput(
                basename=basename,
                working_dir=working_dir,
                version_check=False
            )
            
            # Check if terminated normally
            if not opi_output.terminated_normally():
                return None
            
            # Try to parse energy from OPI (if JSON exists)
            prop_json = working_dir / f"{basename}.property.json"
            if prop_json.exists():
                try:
                    opi_output.parse()
                    if opi_output.results_properties and hasattr(opi_output.results_properties, 'energies'):
                        energies = opi_output.results_properties.energies
                        if energies:
                            return energies[-1]  # Last energy (final SCF)
                except Exception:
                    pass
        except Exception:
            pass
    
    # Fall back to regex (works for all ORCA versions)
    try:
        with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Pattern matches ORCA 5 (with colon) and ORCA 6 (without colon)
        matches = re.findall(r"FINAL SINGLE POINT ENERGY:?\s*([-+]?\d+\.\d+)", content)
        if matches:
            return float(matches[-1])
    except Exception:
        pass
    
    return None


def extract_orca_geometry_opi(output_path: str) -> Optional[List[str]]:
    """
    Extract final geometry from ORCA output using OPI for ORCA 6.1+.
    Falls back to text parsing for ORCA 5.x or when OPI is not available.
    
    Args:
        output_path: Path to ORCA output file (.out)
        
    Returns:
        List of XYZ lines (including atom count and comment), or None if not found
    """
    if not os.path.exists(output_path):
        return None
    
    # Detect ORCA version
    version = detect_orca_version(output_path)
    
    # For ORCA 6.1+, try to use OPI
    if version and version >= (6, 1) and OPI_AVAILABLE and OPIOutput is not None:
        try:
            file_path = Path(output_path)
            basename = file_path.stem
            working_dir = file_path.parent
            
            opi_output = OPIOutput(
                basename=basename,
                working_dir=working_dir,
                version_check=False
            )
            
            # Try to parse geometry from OPI (if JSON exists)
            prop_json = working_dir / f"{basename}.property.json"
            if prop_json.exists():
                try:
                    opi_output.parse()
                    if opi_output.results_properties and hasattr(opi_output.results_properties, 'geometries'):
                        geom = opi_output.results_properties.geometries[-1]
                        if hasattr(geom.geometry, 'coordinates') and geom.geometry.coordinates:
                            coords_data = geom.geometry.coordinates.cartesians
                            
                            # Check if conversion from Bohr is needed
                            coord_units = getattr(geom.geometry.coordinates, 'units', None)
                            bohr_to_angstrom = 0.529177210903
                            need_conversion = True
                            if coord_units and 'angst' in str(coord_units).lower():
                                need_conversion = False
                            
                            xyz_lines = []
                            xyz_lines.append(f"{len(coords_data)}\n")
                            xyz_lines.append(f"{basename} - extracted by ASCEC OPI\n")
                            
                            for atom_data in coords_data:
                                element = atom_data[0]
                                if need_conversion:
                                    x = atom_data[1] * bohr_to_angstrom
                                    y = atom_data[2] * bohr_to_angstrom
                                    z = atom_data[3] * bohr_to_angstrom
                                else:
                                    x, y, z = atom_data[1], atom_data[2], atom_data[3]
                                xyz_lines.append(f"{element:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")
                            
                            return xyz_lines
                except Exception:
                    pass
        except Exception:
            pass
    
    # Fall back to text parsing (extract_final_geometry function handles this)
    return None


def extract_orca_root_from_launcher(launcher_content: str) -> Optional[str]:
    """
    Extract the ORCA root environment variable name from launcher content.
    
    ORCA 6.1+ requires full pathname for parallel runs. This function finds
    the ORCA root variable (e.g., ORCA611_ROOT, ORCA_ROOT) defined in the launcher.
    
    Args:
        launcher_content: Content of the launcher script
        
    Returns:
        The variable name (e.g., "ORCA611_ROOT") if found, None otherwise
    """
    # Look for patterns like: ORCA611_ROOT=, ORCA_ROOT=, ORCA6_ROOT=, etc.
    # Must be an actual variable assignment, not just PATH modification
    pattern = r'^(ORCA[A-Z0-9_]*ROOT)\s*='
    for line in launcher_content.split('\n'):
        line = line.strip()
        match = re.match(pattern, line)
        if match:
            return match.group(1)
    
    # Alternative: check for ORCA*_ROOT in export statements
    export_pattern = r'^export\s+(ORCA[A-Z0-9_]*ROOT)\s*='
    for line in launcher_content.split('\n'):
        line = line.strip()
        match = re.match(export_pattern, line)
        if match:
            return match.group(1)
    
    return None


# Summary functionality - integrated from summary_files.py
def parse_orca_output(filepath):
    """Parse an ORCA output file to extract key information."""
    
    results = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except (FileNotFoundError, IOError, UnicodeDecodeError) as e:
        print(f"Error reading file {filepath}: {e}")
        return None

    # Check for ORCA signature (compatible with ORCA 5 and 6)
    orca_signatures = [
        "O   R   C   A",  # ASCII art header (ORCA 5 and 6)
        "ORCA - Electronic Structure Program",  # Alternative text header
        "Program Version 5.",  # ORCA 5 version
        "Program Version 6."   # ORCA 6 version
    ]
    is_orca_file = any(sig in content for sig in orca_signatures)
    if not is_orca_file:
        print(f"File {filepath} is not an ORCA output file.")
        return None

    results['input_file'] = os.path.splitext(os.path.basename(filepath))[0]

    # Extract final single point energy (handles both ORCA 5 and 6 formats)
    # Pattern matches with or without colon after ENERGY
    energy_matches = re.findall(r"FINAL SINGLE POINT ENERGY:?\s*([-+]?\d+\.\d+)", content)
    if energy_matches:
        optimization_done_index = content.find("*** OPTIMIZATION RUN DONE ***")
        if optimization_done_index != -1:
            last_valid_energy_index = -1
            for i, match in enumerate(energy_matches):
                match_index = content.find(r"FINAL SINGLE POINT ENERGY\s*(-?\d+\.\d+)", 0)
                if match_index < optimization_done_index:
                    last_valid_energy_index = i
            if last_valid_energy_index != -1:
                results['energy'] = float(energy_matches[last_valid_energy_index])
            else:
                results['energy'] = None
        else:
            results['energy'] = float(energy_matches[-1])
    else:
        results['energy'] = None

    # Extract optimization cycles (AFTER XXX CYCLES)
    cycles_match = re.search(r"\(AFTER\s+(\d+)\s+CYCLES\)", content)
    if cycles_match:
        results['cycles'] = int(cycles_match.group(1))
    else:
        results['cycles'] = None

    # Check if optimization converged
    if "THE OPTIMIZATION HAS CONVERGED" in content:
        results['converged'] = True
    else:
        results['converged'] = False

    # Extract total run time
    time_match = re.search(r"TOTAL RUN TIME:\s*(\d+)\s*days\s*(\d+)\s*hours\s*(\d+)\s*minutes\s*(\d+)\s*seconds\s*(\d+)\s*msec", content)
    if time_match:
        days = int(time_match.group(1))
        hours = int(time_match.group(2))
        minutes = int(time_match.group(3))
        seconds = int(time_match.group(4))
        milliseconds = int(time_match.group(5))
        results['time'] = (days * 24 * 3600) + (hours * 3600) + (minutes * 60) + seconds + (milliseconds / 1000.0)
    else:
        results['time'] = None
    return results

def format_time_summary(seconds, include_days=False):
    """Format time for summary output."""
    if include_days:
        days, rem = divmod(seconds, 24 * 3600)
        hours, rem = divmod(rem, 3600)
        minutes, sec = divmod(rem, 60)
        return f"{int(days)} days, {int(hours)}:{int(minutes)}:{sec:.3f}"
    else:
        # Convert all time to hours (even if >24), minutes, seconds
        total_hours, rem = divmod(seconds, 3600)
        minutes, sec = divmod(rem, 60)
        return f"{int(total_hours)}:{int(minutes)}:{sec:.3f}"

def format_total_time(seconds):
    """Format the total execution time in H:M:S format."""
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{int(hours)}:{int(minutes)}:{sec:.3f}"

def format_mean_time(seconds):
    """Format mean execution time in H:M:S format."""
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{int(hours)}:{int(minutes)}:{sec:.3f}"

def format_wall_time(seconds):
    """Format the wall time showing only the two most significant time units."""
    days, rem = divmod(seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, sec = divmod(rem, 60)
    
    # Build time string showing only the two most significant units
    time_parts = []
    
    # Check for weeks and days
    if days >= 7:
        weeks, remaining_days = divmod(days, 7)
        time_parts.append(f"{int(weeks)} week{'s' if weeks != 1 else ''}")
        if remaining_days > 0:
            time_parts.append(f"{int(remaining_days)} day{'s' if remaining_days != 1 else ''}")
        elif hours > 0:
            # If no remaining days but we have hours, show hours
            time_parts.append(f"{int(hours)} hour{'s' if hours != 1 else ''}")
        elif minutes > 0:
            # If no days or hours but we have minutes, show minutes
            time_parts.append(f"{int(minutes)} minute{'s' if minutes != 1 else ''}")
    elif days > 0:
        # Days and hours/minutes
        time_parts.append(f"{int(days)} day{'s' if days != 1 else ''}")
        if hours > 0:
            time_parts.append(f"{int(hours)} hour{'s' if hours != 1 else ''}")
        elif minutes > 0:
            # If no hours but we have minutes, show minutes
            time_parts.append(f"{int(minutes)} minute{'s' if minutes != 1 else ''}")
    elif hours > 0:
        # Hours and minutes
        time_parts.append(f"{int(hours)} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            time_parts.append(f"{int(minutes)} minute{'s' if minutes != 1 else ''}")
        elif sec > 0:
            # If no minutes but we have seconds, show seconds
            time_parts.append(f"{int(sec)} second{'s' if sec != 1 else ''}")
    elif minutes > 0:
        # Minutes and seconds
        time_parts.append(f"{int(minutes)} minute{'s' if minutes != 1 else ''}")
        if sec > 0:
            time_parts.append(f"{int(sec)} second{'s' if sec != 1 else ''}")
        # Note: For sub-second precision, we could add milliseconds here if needed
    else:
        # Just seconds (including fractional seconds)
        if sec >= 1:
            time_parts.append(f"{sec:.1f} second{'s' if sec != 1 else ''}")
        else:
            # Show milliseconds for very short times
            milliseconds = sec * 1000
            time_parts.append(f"{milliseconds:.0f} millisecond{'s' if milliseconds != 1 else ''}")
    
    # Return only the first two parts (most significant)
    return ", ".join(time_parts[:2])

def summarize_calculations(directory=".", file_types=None):
    """Create summary of calculations for ORCA (.out) and/or Gaussian (.log) files."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    if file_types is None:
        file_types = ['orca', 'gaussian']  # Default: process both types
    
    results_by_type = {}
    
    # Process each requested file type
    for file_type in file_types:
        if file_type == 'orca':
            summary_file = "orca_summary.txt"
            file_extension = ".out"
            parse_function = parse_orca_output
        elif file_type == 'gaussian':
            summary_file = "gaussian_summary.txt" 
            file_extension = ".log"
            parse_function = parse_gaussian_output
        else:
            continue
        
        job_summaries = []
        all_results = {
            'job_count': 0,
            'total_time': 0,
            'min_time': None,
            'max_time': None,
            'total_cycles': 0,
            'min_cycles': None,
            'max_cycles': None,
            'cycles_count': 0,  # Count of jobs with cycle data
            'non_converged': 0,  # Count of non-converged optimizations
        }

        # Find files of this type (exclude rescue files like _rescue.out, _rescue.log)
        found_files = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith(file_extension):
                    # Skip rescue files - they are intermediate files from redo workflow
                    if '_rescue' in filename:
                        continue
                    found_files.append(os.path.join(root, filename))
        
        if not found_files:
            continue  # Skip this type if no files found
            
        print(f"Processing {len(found_files)} {file_type.upper()} files in parallel...")
        
        # Use parallel processing to parse files
        max_workers = get_optimal_workers('cpu_intensive', len(found_files))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all parsing tasks - use keyword argument to avoid type checker issues
            future_to_file = {executor.submit(parse_function, filepath=filepath): filepath for filepath in found_files}
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                try:
                    results = future.result()
                    if results:
                        job_summaries.append(results)
                        all_results['job_count'] += 1
                        if results.get('time') is not None:
                            all_results['total_time'] += results['time']
                            if all_results['min_time'] is None or results['time'] < all_results['min_time']:
                                all_results['min_time'] = results['time']
                            if all_results['max_time'] is None or results['time'] > all_results['max_time']:
                                all_results['max_time'] = results['time']
                        if results.get('cycles') is not None:
                            all_results['total_cycles'] += results['cycles']
                            all_results['cycles_count'] += 1
                            if all_results['min_cycles'] is None or results['cycles'] < all_results['min_cycles']:
                                all_results['min_cycles'] = results['cycles']
                            if all_results['max_cycles'] is None or results['cycles'] > all_results['max_cycles']:
                                all_results['max_cycles'] = results['cycles']
                        if results.get('converged') is False:
                            all_results['non_converged'] += 1
                except Exception as e:
                    print(f"Error processing file: {e}")
        
        print(f"Completed processing {len(job_summaries)} {file_type.upper()} files successfully.")
        
        # Write summary file with collected results
        with open(summary_file, 'w', encoding='utf-8') as outfile:

            # Sort the job summaries by time, then by cycles
            job_summaries.sort(key=lambda x: (x.get('time') or float('inf'), x.get('cycles') or float('inf')))

            # Write summary
            outfile.write("=" * 40 + "\n")
            outfile.write("Summary of all calculations:\n")
            outfile.write(f"  Number of jobs: {all_results['job_count']}\n")
            outfile.write("-" * 40 + "\n")
            if all_results['total_time']:
                outfile.write(f"  Total execution time: {format_total_time(all_results['total_time'])}\n")
                outfile.write(f"  Mean execution time: {format_mean_time(all_results['total_time'] / all_results['job_count'])}\n")
                outfile.write(f"  Shortest execution time: {format_time_summary(all_results['min_time'], include_days=False)}\n")
                outfile.write(f"  Longest execution time: {format_time_summary(all_results['max_time'], include_days=False)}\n")
                outfile.write(f"  Total wall time: {format_wall_time(all_results['total_time'])}\n")
            outfile.write("-" * 40 + "\n")
            if all_results['cycles_count'] > 0:
                outfile.write(f"  Mean cycles: {all_results['total_cycles'] // all_results['cycles_count']}\n")
                outfile.write(f"  Min cycles: {all_results['min_cycles']}\n")
                outfile.write(f"  Max cycles: {all_results['max_cycles']}\n")
            else:
                # Keep cycle metrics explicit even when all jobs are non-converged.
                outfile.write("  Mean cycles: N/A\n")
                outfile.write("  Min cycles: N/A\n")
                outfile.write("  Max cycles: N/A\n")
            outfile.write(f"  Non-converged: {all_results['non_converged']}\n")

            outfile.write("=" * 40 + "\n\n")

            # Write individual job details
            job_index = 1
            for result in job_summaries:
                if file_type == 'orca':
                    outfile.write(f"=> {job_index}. {result['input_file']}.out\n")
                else:  # gaussian
                    outfile.write(f"=> {job_index}. {result['input_file']}.log\n")
                job_index += 1
                # Write in specific order: energy, cycles, time
                if 'energy' in result and result['energy'] is not None:
                    outfile.write(f"  energy = {result['energy']}\n")
                if 'cycles' in result and result['cycles'] is not None:
                    outfile.write(f"  cycles = {result['cycles']}\n")
                else:
                    outfile.write("  cycles = N/A\n")
                if 'time' in result and result['time'] is not None:
                    outfile.write(f"  time = {format_time_summary(result['time'], include_days=False)}\n")
                outfile.write("\n")

        print(f"Summary written to {summary_file}")
        results_by_type[file_type] = len(job_summaries)

    # Return total number of summaries created
    return sum(results_by_type.values())

def find_out_files(root_dir, include_orca: bool = True, include_gaussian: bool = True):
    """Find calculation output files in the directory tree using parallel processing."""
    import multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os
    
    out_files = []
    directories_to_search = []
    
    # First, collect all directories to search
    for root, dirs, files in os.walk(root_dir):
        directories_to_search.append((root, files))
    
    # Define function to search a single directory
    def search_directory(dir_data):
        root, files = dir_data
        local_out_files = []
        for file in files:
            if (include_orca and file.endswith('.out')) or (include_gaussian and file.endswith('.log')):
                local_out_files.append(os.path.join(root, file))
        return local_out_files
    
    # Use parallel processing to search directories
    if len(directories_to_search) > 1:
        max_workers = get_optimal_workers('io_intensive', len(directories_to_search))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all directory search tasks
            future_to_dir = {executor.submit(search_directory, dir_data): dir_data[0] 
                            for dir_data in directories_to_search}
            
            # Collect results as they complete
            for future in as_completed(future_to_dir):
                try:
                    local_files = future.result()
                    out_files.extend(local_files)
                except Exception as e:
                    print(f"Error searching directory: {e}")
    else:
        # For single directory, just do it directly
        out_files = search_directory(directories_to_search[0]) if directories_to_search else []
    
    return out_files


def get_unique_folder_name(base_name, current_dir):
    """Generate a unique folder name if base_name already exists."""
    folder_path = os.path.join(current_dir, base_name)
    counter = 1
    
    while os.path.exists(folder_path):
        new_name = f"{base_name}_{counter}"
        folder_path = os.path.join(current_dir, new_name)
        counter += 1
    
    return os.path.basename(folder_path)


def collect_out_files():
    """Collect all .out (ORCA) and .log (Gaussian) files into a single folder inside a similarity directory."""
    current_directory = os.getcwd()
    print(f"Searching for output files in: {current_directory} and its subfolders...")

    all_out_files = find_out_files(current_directory)

    if not all_out_files:
        print("No output files found in the current directory or its subfolders.")
        return False

    num_files = len(all_out_files)
    
    # Detect file types to name folder appropriately
    has_orca = any(f.endswith('.out') for f in all_out_files)
    has_gaussian = any(f.endswith('.log') for f in all_out_files)
    
    if has_orca and has_gaussian:
        base_destination_folder_name = f"calc_out_{num_files}"
        file_type_desc = "output files (ORCA and Gaussian)"
    elif has_gaussian:
        base_destination_folder_name = f"gaussian_out_{num_files}"
        file_type_desc = "Gaussian files"
    else:
        base_destination_folder_name = f"orca_out_{num_files}"
        file_type_desc = "ORCA files"
    
    # Create similarity directory at parent level
    parent_directory = os.path.dirname(current_directory)
    similarity_path = os.path.join(parent_directory, "similarity")
    os.makedirs(similarity_path, exist_ok=True)
    
    destination_folder_name = get_unique_folder_name(base_destination_folder_name, similarity_path)
    destination_path = os.path.join(similarity_path, destination_folder_name)

    os.makedirs(destination_path)
    print(f"\nCreated destination folder: similarity/{destination_folder_name}")

    # Define function to copy a single file
    def copy_file(file_data):
        file_path, destination_path = file_data
        try:
            shutil.copy2(file_path, destination_path)
            return f"Copied: {os.path.basename(file_path)}"
        except Exception as e:
            return f"Error copying {os.path.basename(file_path)}: {e}"

    print(f"Copying {num_files} {file_type_desc} to 'similarity/{destination_folder_name}' in parallel...")
    
    # Use parallel processing for file copying
    import multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    max_workers = get_optimal_workers('io_intensive', num_files)
    
    copy_tasks = [(file_path, destination_path) for file_path in all_out_files]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all copy tasks
        future_to_file = {executor.submit(copy_file, task): task[0] 
                         for task in copy_tasks}
        
        # Collect results as they complete
        successful_copies = 0
        for future in as_completed(future_to_file):
            try:
                result = future.result()
                if "Copied:" in result:
                    successful_copies += 1
                if successful_copies % 10 == 0 or "Error" in result:  # Show progress every 10 files or errors
                    print(f"  {result}")
            except Exception as e:
                print(f"  Unexpected error: {e}")

    print(f"\nCopied {successful_copies}/{num_files} {file_type_desc} to similarity/{destination_folder_name}")
    print("Process complete. Original files remain untouched.")
    return True

def capture_current_state(directory):
    """Capture the current state of files and folders for potential revert."""
    state = {
        'files': {},  # filename -> full_path
        'folders': set()
    }
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            state['files'][item] = item_path
        elif os.path.isdir(item_path):
            state['folders'].add(item_path)
    
    return state


def unsort_directory(directory='.'):
    """
    Reverse the sort operation: move all files from subdirectories back to root.
    This is the opposite of group_files_by_base_with_tracking.
    """
    if not os.path.exists(directory):
        return 0
    
    moved_count = 0
    subdirs = [d for d in os.listdir(directory) 
              if os.path.isdir(os.path.join(directory, d)) and not d.startswith('.')]
    
    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        try:
            # Move all files from subdir to parent directory
            for item in os.listdir(subdir_path):
                src = os.path.join(subdir_path, item)
                if os.path.isfile(src):
                    dst = os.path.join(directory, item)
                    # If destination exists, remove it first (overwrite)
                    if os.path.exists(dst):
                        os.remove(dst)
                    shutil.move(src, dst)
                    moved_count += 1
            
            # Remove empty subdirectory
            if not os.listdir(subdir_path):
                os.rmdir(subdir_path)
        except Exception as e:
            print(f"    Warning: Could not unsort {subdir}: {e}")
    
    return moved_count


def group_files_by_base_with_tracking(directory='.'):
    """Group files by base name and track what was moved."""
    # Filter out ORCA intermediate files right from the start
    excluded_patterns = ['.scfgrad.', '.scfp.', '.tmp.', '.densities.', '.scfhess.']
    files = [f for f in os.listdir(directory) 
             if os.path.isfile(os.path.join(directory, f)) 
             and not any(pattern in f for pattern in excluded_patterns)]
    base_map = defaultdict(list)
    
    for file in files:
        # Skip combined results to avoid moving them to subfolders
        if file.startswith("combined_results"):
             continue
        base = extract_base(file)
        if base:
            base_map[base].append(file)
    
    # Track moved files and created folders
    tracking = {'folders': [], 'moved_files': {}}
    moved_count = 0
    
    for base, grouped_files in base_map.items():
        if len(grouped_files) > 1:
            folder_path = os.path.join(directory, base)
            os.makedirs(folder_path, exist_ok=True)
            tracking['folders'].append(folder_path)
            
            for file in grouped_files:
                src = os.path.join(directory, file)
                dest = os.path.join(folder_path, file)
                tracking['moved_files'][dest] = src  # destination -> original
                shutil.move(src, dest)
            
            print(f"Moved {len(grouped_files)} files to folder: {base}")
            moved_count += len(grouped_files)
    
    if moved_count == 0:
        print("No files needed to be grouped.")
    else:
        print(f"Total files moved: {moved_count}")
    
    # Clean up orphaned .inp/.com/.gjf files at root if their subfolder exists
    # This handles case where files were already organized but input files were left behind
    # CRITICAL: Filter out ORCA intermediate files (.scfgrad.inp, .scfp.inp, etc.)
    excluded_patterns = ['.scfgrad.', '.scfp.', '.tmp.', '.densities.', '.scfhess.']
    for file in os.listdir(directory):
        if file.endswith(('.inp', '.com', '.gjf')) and os.path.isfile(os.path.join(directory, file)):
            # Skip ORCA intermediate files
            if any(pattern in file for pattern in excluded_patterns):
                continue
            base = extract_base(file)
            folder_path = os.path.join(directory, base)
            if base and os.path.isdir(folder_path):
                # Subfolder exists, move the orphaned input file there
                src = os.path.join(directory, file)
                dest = os.path.join(folder_path, file)
                if not os.path.exists(dest):  # Don't overwrite if already exists
                    shutil.move(src, dest)
    
    return tracking


def create_summary_with_tracking(directory, file_types_override: Optional[List[str]] = None):
    """Create summaries and return list of created files."""
    created_files = []
    
    # Check for ORCA files (.out) and Gaussian files (.log)
    orca_files = []
    gaussian_files = []
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".out"):
                orca_files.append(os.path.join(root, filename))
            elif filename.endswith(".log"):
                gaussian_files.append(os.path.join(root, filename))
    
    try:
        # Determine which file types to process
        if file_types_override is not None:
            file_types_to_process = [ft for ft in file_types_override if ft in ('orca', 'gaussian')]
        else:
            file_types_to_process = []
            if orca_files:
                file_types_to_process.append('orca')
            if gaussian_files:
                file_types_to_process.append('gaussian')
        
        # Create summaries for found file types
        if file_types_to_process:
            num_summaries = summarize_calculations(directory, file_types_to_process)
            
            # Check which summary files were created
            if 'orca' in file_types_to_process and os.path.exists("orca_summary.txt"):
                created_files.append("orca_summary.txt")
            if 'gaussian' in file_types_to_process and os.path.exists("gaussian_summary.txt"):
                created_files.append("gaussian_summary.txt")
    except Exception:
        pass
    
    return created_files


def collect_out_files_with_tracking(reuse_existing=False, target_sim_folder=None, include_gaussian: bool = True):
    """Collect output files and return the created similarity folder path."""
    try:
        current_directory = os.getcwd()
        all_out_files = find_out_files(current_directory, include_orca=True, include_gaussian=include_gaussian)
        
        # Filter out backup files, ORCA intermediate files, and non-calculation output files
        all_out_files = [f for f in all_out_files if not (
            '.backup' in f or 
            f.endswith('.out.backup') or
            f.endswith('.log.backup') or
            'orca_summary' in os.path.basename(f).lower() or
            'gaussian_summary' in os.path.basename(f).lower() or
            '.scfhess.' in os.path.basename(f) or
            '.scfgrad.' in os.path.basename(f) or
            '.scfp.' in os.path.basename(f) or
            '.tmp.' in os.path.basename(f)
        )]
        
        if not all_out_files:
            return None
        
        num_files = len(all_out_files)
        
        # Detect file types to name folder appropriately
        has_orca = any(f.endswith('.out') for f in all_out_files)
        has_gaussian = any(f.endswith('.log') for f in all_out_files)
        
        if has_orca and has_gaussian:
            base_destination_folder_name = f"calc_out_{num_files}"
        elif has_gaussian:
            base_destination_folder_name = f"gaussian_out_{num_files}"
        else:
            base_destination_folder_name = f"orca_out_{num_files}"
        
        # Create similarity folder with incremental numbering at parent level
        parent_directory = os.path.dirname(current_directory)
        
        def get_next_similarity_dir():
            """Find the next available similarity directory (similarity, similarity_2, etc.)"""
            # If target folder is explicitly provided, use it
            if target_sim_folder:
                # Handle both full path and relative path
                if os.path.isabs(target_sim_folder):
                    return target_sim_folder
                else:
                    return os.path.join(parent_directory, target_sim_folder)

            base_name = "similarity"
            similarity_path = os.path.join(parent_directory, base_name)
            
            # If reuse_existing is True, return the base path if it exists
            if reuse_existing and os.path.exists(similarity_path):
                return similarity_path
            
            if not os.path.exists(similarity_path):
                return similarity_path
            
            counter = 2
            while True:
                similarity_dir_name = f"{base_name}_{counter}"
                similarity_path = os.path.join(parent_directory, similarity_dir_name)
                if not os.path.exists(similarity_path):
                    return similarity_path
                counter += 1
        
        similarity_dir = get_next_similarity_dir()
        os.makedirs(similarity_dir, exist_ok=True)
        
        # Reuse existing destination folder when possible, otherwise create one.
        # This prevents redo/resume flows from producing orca_out_N_1/_2/_3.
        existing_exact_dest = os.path.join(similarity_dir, base_destination_folder_name)

        def _clear_destination_outputs(dest_dir: str) -> None:
            """Remove previous copied outputs so destination reflects current state."""
            try:
                for item in os.listdir(dest_dir):
                    item_path = os.path.join(dest_dir, item)
                    if os.path.isfile(item_path) and item.endswith(('.out', '.log')):
                        os.remove(item_path)
            except Exception:
                pass

        # Create the calc_out/orca_out/gaussian_out subfolder inside similarity folder
        if reuse_existing:
            # In redo mode: always reuse the same output folder (cleanup if exists)
            # Look for any of the possible folder patterns
            existing_orca_dirs = glob.glob(os.path.join(similarity_dir, f"orca_out_{num_files}*"))
            existing_gaussian_dirs = glob.glob(os.path.join(similarity_dir, f"gaussian_out_{num_files}*"))
            existing_calc_dirs = glob.glob(os.path.join(similarity_dir, f"calc_out_{num_files}*"))
            existing_orca_dirs.extend(existing_gaussian_dirs)
            existing_orca_dirs.extend(existing_calc_dirs)

            # Prefer exact folder name (e.g., orca_out_180) over suffixed variants.
            if os.path.isdir(existing_exact_dest):
                destination_path = existing_exact_dest
                destination_folder_name = os.path.basename(destination_path)
                _clear_destination_outputs(destination_path)
            elif existing_orca_dirs:
                existing_orca_dirs = sorted(existing_orca_dirs, key=lambda p: (0 if os.path.basename(p) == base_destination_folder_name else 1, p))
                destination_path = existing_orca_dirs[0]
                destination_folder_name = os.path.basename(destination_path)
                _clear_destination_outputs(destination_path)
            else:
                # First time - create the folder
                destination_folder_name = base_destination_folder_name
                destination_path = os.path.join(similarity_dir, destination_folder_name)
                os.makedirs(destination_path, exist_ok=True)

            # Cleanup old artifacts before reusing similarity folder
            print(f"Cleaning up previous similarity results in {os.path.basename(similarity_dir)}...")
            items_to_remove = [
                'dendrogram_images', 'extracted_clusters', 'extracted_data',
                'skipped_structures', 'clustering_summary.txt', 'boltzmann_distribution.txt'
            ]

            # Also remove motifs folders
            for item in os.listdir(similarity_dir):
                if item.startswith('motifs_') or item.startswith('umotifs_'):
                    items_to_remove.append(item)

            for item in items_to_remove:
                item_path = os.path.join(similarity_dir, item)
                if os.path.exists(item_path):
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                    except Exception as e:
                        print(f"Warning: Could not remove {item}: {e}")
        else:
            # Default mode: if the exact destination exists, update it in-place.
            # This keeps workflow outputs stable instead of creating *_1 variants.
            if os.path.isdir(existing_exact_dest):
                destination_folder_name = base_destination_folder_name
                destination_path = existing_exact_dest
                _clear_destination_outputs(destination_path)
            else:
                destination_folder_name = get_unique_folder_name(base_destination_folder_name, similarity_dir)
                destination_path = os.path.join(similarity_dir, destination_folder_name)
                os.makedirs(destination_path, exist_ok=True)
        
        # Copy files to the output subfolder
        for file_path in all_out_files:
            shutil.copy2(file_path, destination_path)
        
        # Get just the folder name for display (without full path)
        similarity_folder_name = os.path.basename(similarity_dir)
        file_type_desc = "output files"
        if has_orca and has_gaussian:
            file_type_desc = "ORCA and Gaussian files"
        elif has_gaussian:
            file_type_desc = "Gaussian files"
        else:
            file_type_desc = "ORCA files"
        print(f"Copied {num_files} {file_type_desc} to {similarity_folder_name}/{destination_folder_name}")
        return destination_path
    except Exception:
        return None

def parse_gaussian_output(filepath):
    """Parse Gaussian .log output file for energy and time."""
    results = {'input_file': os.path.splitext(os.path.basename(filepath))[0]}
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Extract SCF Done energy (final energy)
    energy_matches = re.findall(r"SCF Done:\s+E\([^)]+\)\s*=\s*(-?\d+\.\d+)", content)
    if energy_matches:
        results['energy'] = float(energy_matches[-1])  # Take the last SCF Done energy
    else:
        results['energy'] = None

    # Extract number of processors
    nproc = 1  # Default to 1 if not specified
    nproc_match = re.search(r"Will use up to\s+(\d+)\s+processors", content)
    if nproc_match:
        nproc = int(nproc_match.group(1))
    else:
        # Try alternative format %NProcShared
        nproc_match = re.search(r"%NProcShared\s*=\s*(\d+)", content, re.IGNORECASE)
        if nproc_match:
            nproc = int(nproc_match.group(1))

    # Extract optimization cycles (last "Optimization completed" or SCF cycles count)
    # Gaussian uses "-- Stationary point found" after geometry optimization
    cycles_match = re.search(r"Step number\s+(\d+)\s+out of", content)
    if cycles_match:
        results['cycles'] = int(cycles_match.group(1))
    else:
        # Alternative: count number of optimization steps
        opt_steps = re.findall(r"Step number\s+(\d+)", content)
        if opt_steps:
            results['cycles'] = int(opt_steps[-1])  # Take the last step number
        else:
            results['cycles'] = None

    # Check if optimization converged
    if "Stationary point found" in content or "Optimization completed" in content:
        results['converged'] = True
    else:
        results['converged'] = False

    # Extract all job cpu times and sum them (for multi-step calculations)
    time_matches = re.findall(r"Job cpu time:\s*(\d+)\s*days\s*(\d+)\s*hours\s*(\d+)\s*minutes\s*(\d+\.\d+)\s*seconds", content)
    if time_matches:
        total_cpu_time = 0
        for time_match in time_matches:
            days = int(time_match[0])
            hours = int(time_match[1])
            minutes = int(time_match[2])
            seconds = float(time_match[3])
            total_cpu_time += (days * 24 * 3600) + (hours * 3600) + (minutes * 60) + seconds
        
        # Convert CPU time to wall time by dividing by number of processors
        results['time'] = total_cpu_time / nproc
    else:
        results['time'] = None
    
    return results

def execute_summary_only():
    """Execute only summary creation without sorting files."""
    print("=" * 50)
    print("ASCEC Summary Creation")
    print("=" * 50)
    
    # Check for ORCA files (.out) and Gaussian files (.log)
    orca_files = []
    gaussian_files = []
    
    for root, _, files in os.walk("."):
        for filename in files:
            if filename.endswith(".out"):
                orca_files.append(os.path.join(root, filename))
            elif filename.endswith(".log"):
                gaussian_files.append(os.path.join(root, filename))
    
    created_summaries = []
    file_types_to_process = []
    
    if orca_files:
        print(f"\nFound {len(orca_files)} ORCA output files.")
        file_types_to_process.append('orca')
        
    if gaussian_files:
        print(f"\nFound {len(gaussian_files)} Gaussian output files.")
        file_types_to_process.append('gaussian')
    
    if not orca_files and not gaussian_files:
        print("\nNo ORCA (.out) or Gaussian (.log) output files found in the current directory or its subfolders.")
        return
        
    # Create summaries for found file types
    if file_types_to_process:
        print("Creating summaries...")
        num_summaries = summarize_calculations(".", file_types_to_process)
        
        # Check which summary files were created
        if 'orca' in file_types_to_process and os.path.exists("orca_summary.txt"):
            created_summaries.append("orca_summary.txt")
        if 'gaussian' in file_types_to_process and os.path.exists("gaussian_summary.txt"):
            created_summaries.append("gaussian_summary.txt")
    
    print("\n" + "=" * 50)
    print("Summary Creation Completed")
    print("=" * 50)
    
    if created_summaries:
        print(f"\nCreated summary files: {', '.join(created_summaries)}")
    else:
        print("\nNo summary files were created (no valid calculation data found).")


def execute_sort_command(include_summary=True, target_sim_folder=None, reuse_existing=False):
    """Execute the complete sort process with option to revert."""
    print("=" * 50)
    print("ASCEC Sort Process Started")
    print("=" * 50)
    
    # Store original state for potential revert
    original_state = capture_current_state(".")
    created_files = []
    created_folders = []
    
    try:
        # Step 1: Sort files by base names
        print("\n1. Sorting files by base names...")
        moved_files = group_files_by_base_with_tracking(".")
        created_folders.extend(moved_files.get('folders', []))
        
        # Step 2: Merge XYZ files
        print("\n2. Merging XYZ files...")
        combined_file = "combined_results.xyz"
        if combine_xyz_files():
            created_files.append(combined_file)
            # Step 3: Create MOL file
            print("\n3. Creating MOL file...")
            mol_file = "combined_results.mol"
            if create_combined_mol():
                created_files.append(mol_file)
        
        # Step 4: Create summary (if requested)
        if include_summary:
            print("\n4. Creating calculation summary...")
            summary_files = create_summary_with_tracking(".")
            created_files.extend(summary_files)
        else:
            print("\n4. Skipping summary creation (--nosum flag used)")
        
        # Step 5: Collect .out files
        print("\n5. Collecting .out files...")
        similarity_folder = collect_out_files_with_tracking(reuse_existing=reuse_existing, target_sim_folder=target_sim_folder)
        if similarity_folder:
            created_folders.append(similarity_folder)
        
        print("\n" + "=" * 50)
        print("ASCEC Sort Process Completed")
        print("=" * 50)
        
        # Suggest similarity analysis if .out files were collected
        # Check if any similarity folder exists at the parent level
        parent_dir = os.path.dirname(os.getcwd())
        similarity_dirs = []
        for item in os.listdir(parent_dir):
            if item == "similarity" or item.startswith("similarity_"):
                similarity_path = os.path.join(parent_dir, item)
                if os.path.isdir(similarity_path):
                    similarity_dirs.append(item)
            
        if similarity_dirs:
            print("\nSuggested next step:")
            print("  python similarity --threshold 0.9")
            print("  Run similarity analysis on collected output files")

    except Exception as e:
        print(f"\nError during sort process: {e}")

def execute_similarity_analysis(*args):
    """Execute similarity analysis by calling the similarity script."""
    import subprocess
    import sys
    
    # Get the directory where ascec-v04.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    similarity_script = os.path.join(script_dir, "similarity-v01.py")
    
    if not os.path.exists(similarity_script):
        print(f"Error: similarity_v01.py not found in {script_dir}")
        print("Make sure similarity_v01.py is in the same directory as ascec-v04.py")
        return
    
    # Build command
    cmd = ["python", similarity_script] + list(args)
    has_threshold_arg = any(str(arg).startswith("--th=") or str(arg).startswith("--threshold=") for arg in args)
    if not has_threshold_arg:
        cmd.extend(["--th", "2.0"])
    
    print("=" * 50)
    print("ASCEC Similarity Analysis")
    print("=" * 50)
    print(f"Executing: {' '.join(cmd[1:])}")  # Don't show python path
    print()
    
    try:
        # Execute the similarity script with all arguments
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("Similarity analysis completed successfully.")
        print("=" * 50)
    except subprocess.CalledProcessError as e:
        print(f"\nError executing similarity analysis: {e}")
        print("Check the similarity script arguments and try again.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


def execute_diagram_generation(scaled: bool = False):
    """
    Generate or regenerate all annealing diagrams from tvse_*.dat files.
    Searches for tvse_*.dat files in current directory and subdirectories,
    creates individual diagrams, and generates combined replica diagrams.
    
    Args:
        scaled: If True, apply intelligent y-axis scaling to remove initial high-energy clutter
    """
    if not MATPLOTLIB_AVAILABLE or plt is None:
        print("Error: matplotlib is not available")
        print("Install with: pip install matplotlib")
        return
    
    print("=" * 60)
    print("ASCEC Diagram Generation")
    if scaled:
        print("(Scaled mode: intelligent y-axis scaling enabled)")
    print("=" * 60)
    
    # Find all tvse_*.dat files
    tvse_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.startswith('tvse_') and file.endswith('.dat'):
                tvse_files.append(os.path.join(root, file))
    
    if not tvse_files:
        print("No tvse_*.dat files found in current directory or subdirectories")
        return
    
    print(f"Found {len(tvse_files)} tvse file(s)")
    print()
    
    # Generate individual diagrams
    diagrams_generated = 0
    for tvse_file in tvse_files:
        output_dir = os.path.dirname(tvse_file)
        filename = os.path.basename(tvse_file)
        seed = filename.replace('tvse_', '').replace('.dat', '')
        
        print(f"  {filename}...", end=" ", flush=True)
        
        if plot_annealing_diagrams(tvse_file, output_dir, scaled=scaled):
            print(f"✓")
            diagrams_generated += 1
        else:
            print(f"✗ Failed")
    
    print(f"\nGenerated {diagrams_generated} individual diagram(s)")
    
    # Check for replica groups (multiple tvse files in same parent directory)
    # Group by parent directory
    parent_dirs = {}
    for tvse_file in tvse_files:
        parent = os.path.dirname(os.path.dirname(tvse_file))
        if parent not in parent_dirs:
            parent_dirs[parent] = []
        parent_dirs[parent].append(tvse_file)
    
    # Generate combined diagrams for parent directories with multiple replicas
    combined_generated = 0
    for parent_dir, files in parent_dirs.items():
        if len(files) > 1:
            # This looks like a replica group
            num_replicas = len(files)
            output_file = os.path.join(parent_dir, f"tvse_r{num_replicas}.png")
            
            print(f"\nGenerating combined diagram for {num_replicas} replica(s)...")
            
            if plot_combined_replicas_diagram(files, output_file, num_replicas):
                print(f"  ✓ Created: {os.path.basename(output_file)}")
                combined_generated += 1
            else:
                print(f"  ✗ Failed to create combined diagram")
    
    if combined_generated > 0:
        print(f"\nGenerated {combined_generated} combined diagram(s)")
    
    print("\n" + "=" * 60)
    print("Diagram generation completed")
    print("=" * 60)


def execute_box_analysis(input_file: str):
    """
    Analyze an input file and provide box length recommendations without running the simulation.
    This reads the input file, parses the molecular structure, and outputs the volume-based 
    box length analysis to help users choose appropriate box sizes.
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            sys.exit(1)
        
        # Create a minimal SystemState for parsing
        state = SystemState()
        state.verbosity_level = 1  # Enable verbose output for box analysis
        
        # Parse the input file
        read_input_file(state, input_file)
        
        # Provide box length analysis using the existing function
        provide_box_length_advice(state)
        
        # Exit successfully after box analysis
        return
        
    except Exception as e:
        print(f"Error during box analysis: {e}")
        sys.exit(1)

# 16. main ascec integrated function
def print_all_commands():
    """Print command line usage information (now uses argparse help)."""
    # This function is kept for backward compatibility but now just points to argparse help
    # The actual help text is defined in the ArgumentParser epilog
    print("Use 'ascec -h' or 'ascec --help' for detailed command reference.")


# ============================================================================
# Workflow automation functions
# ============================================================================

@dataclasses.dataclass
class WorkflowContext:
    """Context object to pass information between workflow stages."""
    input_file: str = ""
    num_replicas: int = 0
    annealing_dirs: List[str] = dataclasses.field(default_factory=list)
    optimization_stage_dir: str = ""
    similarity_dir: str = ""
    refinement_stage_dir: str = ""
    critical_count: int = 0
    skipped_count: int = 0
    total_structures: int = 0
    current_try: int = 1
    max_tries: int = 3
    optimization_stage_number: int = 0  # Track which optimization/refinement cycle stage (1 or 2)
    cache_file: str = ""  # Protocol-specific cache filename
    current_stage_key: str = ""  # Current stage key (e.g., "optimization_2")
    similarity_args: List[str] = dataclasses.field(default_factory=list)  # Store all similarity args
    is_workflow: bool = False  # True when running in workflow mode (with , or then separators)
    workflow_verbose: bool = False  # True only when workflow should print detailed stage logs
    workflow_verbose_level: int = 0  # 0: silent, 1: -v, 2: -v2, 3: -v3
    max_launch_retries: int = 10  # Hardcoded retry attempts for launch failures only (instant crashes)
    ascec_parallel_cores: int = 0  # Number of cores for parallel processing (0 = auto-detect, capped at 12)
    # Exclude patterns for optimization/refinement stages
    optimization_exclude_patterns: Dict[str, List[int]] = dataclasses.field(default_factory=dict)  # e.g., {'opt1': [2, 5, 6, 7, 8, 9]}
    refinement_exclude_patterns: Dict[str, List[int]] = dataclasses.field(default_factory=dict)  # e.g., {'ref1': [2, 5, 6, 7, 8, 9]}
    # QM program alias from input.in line 9 (e.g., "orca", "g16")
    qm_alias: str = "orca"  # Default to "orca" for ORCA installations
    # Data capture attributes for protocol summary
    annealing_box_size: Optional[float] = None
    annealing_packing: Optional[float] = None
    optimization_xyz_source: Optional[str] = None
    optimization_completed: Optional[int] = None
    optimization_total: Optional[int] = None
    optimization_sim_folder: Optional[str] = None
    sim_folder: Optional[str] = None
    sim_motifs_created: Optional[int] = None
    last_similarity_input_count: Optional[int] = None
    last_similarity_motif_count: Optional[int] = None
    last_similarity_umotif_count: Optional[int] = None
    similarity_stage_counts: Dict[int, int] = dataclasses.field(default_factory=dict)  # stage index -> representative count
    similarity_stage_input_counts: Dict[int, int] = dataclasses.field(default_factory=dict)  # stage index -> input structure count
    refinement_motifs_source: Optional[str] = None
    refinement_completed: Optional[int] = None
    refinement_total: Optional[int] = None
    refinement_sim_folder: Optional[str] = None
    recalculated_files: Optional[List[str]] = None  # List of basenames for files being recalculated in redo
    pending_similarity_folder: Optional[str] = None  # Folder set by optimization/refinement for the next similarity stage
    use_skipped_threshold: bool = False  # True if --skipped flag is used, False if --critical (default)
    current_stage: Optional[Dict[str, Any]] = None  # Active workflow stage (for stage-aware helpers)
    update_progress: Optional[Callable[[str], None]] = None  # Compact workflow progress callback
    completed_stage_count: int = 0  # Number of finished workflow stages for progress rendering
    
    def get_previous_stage_output_dir(self, stage_type: str) -> Optional[str]:
        """
        Get output directory from the most recent completed stage of given type.
        
        Args:
            stage_type: Stage type prefix (e.g., 'r', 'optimization', 'similarity', 'refinement')
        
        Returns:
            Output directory path or None if not found
            
        Example:
            context.get_previous_stage_output_dir('similarity')  # Returns 'similarity/motifs'
            context.get_previous_stage_output_dir('optimization')  # Returns 'calculation'
        """
        if not self.cache_file:
            return None
        
        cache = load_protocol_cache(self.cache_file)
        stages = cache.get('stages', {})
        
        # Search backwards through stages for matching type
        for stage_key in reversed(list(stages.keys())):
            if stage_key.startswith(stage_type):
                stage_data = stages[stage_key]
                if stage_data.get('status') == 'completed':
                    result = stage_data.get('result', {})
                    return result.get('output_dir')
        
        return None
    
    def get_stage_working_dir(self, stage_key: str) -> Optional[str]:
        """
        Get working directory for a specific stage by its key.
        
        Args:
            stage_key: Exact stage key (e.g., 'optimization_1', 'similarity_2')
        
        Returns:
            Working directory path or None if not found
            
        Example:
            context.get_stage_working_dir('optimization_1')  # Returns 'calculation'
        """
        if not self.cache_file:
            return None
        
        cache = load_protocol_cache(self.cache_file)
        stages = cache.get('stages', {})
        
        if stage_key in stages:
            result = stages[stage_key].get('result', {})
            return result.get('working_dir')
        
        return None
    
    def get_previous_stage_input_dir(self, stage_type: str) -> Optional[str]:
        """
        Get input directory from the most recent completed stage of given type.
        
        Args:
            stage_type: Stage type prefix (e.g., 'r', 'optimization', 'similarity', 'refinement')
        
        Returns:
            Input directory path or None if not found
        """
        if not self.cache_file:
            return None
        
        cache = load_protocol_cache(self.cache_file)
        stages = cache.get('stages', {})
        
        # Search backwards through stages for matching type
        for stage_key in reversed(list(stages.keys())):
            if stage_key.startswith(stage_type):
                stage_data = stages[stage_key]
                if stage_data.get('status') == 'completed':
                    result = stage_data.get('result', {})
                    return result.get('input_dir')
        
        return None


def contains_workflow_separator(args: List[str]) -> bool:
    """Check if command line arguments contain ',' or 'then' separator (workflow mode)."""
    for arg in args:
        if not arg:
            continue
        token = arg.strip().lower()
        if token == 'then' or token == ',':
            return True
        # Support comma-attached forms such as "r3," or "input.asc,"
        if ',' in token:
            return True
    return False

def parse_workflow_stages(args: List[str]) -> List[Dict[str, Any]]:
    """
    Parse compound workflow commands with ',' or 'then' separator.
    Supports both space-separated and comma-attached formats.
    
    Examples:
        ascec04 at_annealing.in , r3 , opt --redo=3 preopt.inp launcher.sh , similarity --th=2
        ascec04 at_annealing.asc, r3, opt --redo=3 preopt.inp launcher.sh, similarity --th=2
        ascec04 .asc, r3, opt -c --redo=3 preopt.inp launcher.sh, similarity --th=2
        
        With pause after stage (using dot separator):
        ascec04 .asc, r3, opt --redo=3 preopt.inp launcher.sh. similarity --th=2
            (dot after launcher.sh means pause after optimization stage for manual review)
        
        With auto-selection flags:
        ascec04 at_annealing.asc , r3 , opt -a --redo=3 preopt.inp launcher.sh
            (-a: Process all result_*.xyz files separately)
        ascec04 at_annealing.asc , r3 , opt -c --redo=3 preopt.inp launcher.sh
            (-c: Combine all result_*.xyz into combined_r{N}.xyz first, then process)
        
        Flag meanings:
            --redo=N: Redo entire stage (optimization+similarity or refinement+similarity) up to N times
            
        Note: Launch failures (instant crashes) are automatically retried up to 10 times.
        Once an optimization run starts running normally, it will not be retried regardless of exit code.
    
    Returns:
        List of stage dictionaries with 'type', 'args', and optional 'pause_after' keys
    """
    stages = []
    current_stage = []
    pause_after_current = False
    
    for arg in args:
        # Skip blank/empty arguments (allows blank lines in protocol)
        if not arg or not arg.strip():
            continue
            
        # Handle comma or dot attached to argument (e.g., "r3," or "launcher.sh.")
        has_comma = ',' in arg and arg != ','
        has_dot = '.' in arg and not arg.startswith('.') and not arg.endswith(('.inp', '.sh', '.xyz', '.in'))
        
        if has_comma or (has_dot and len(arg) > 1 and arg[-1] == '.'):
            # Split by separator (comma or dot)
            if has_dot and arg[-1] == '.':
                # Dot at end means pause after this stage
                part = arg[:-1].strip()  # Remove the dot
                if part:
                    current_stage.append(part)
                # Finalize stage with pause marker
                if current_stage:
                    stage = finalize_stage(current_stage, pause_after=True)
                    if stage:
                        stages.append(stage)
                    current_stage = []
                    pause_after_current = False
            else:
                # Split by comma
                parts = arg.split(',')
                for i, part in enumerate(parts):
                    part = part.strip()
                    if part:
                        current_stage.append(part)
                    # After each part except the last, finalize stage
                    if i < len(parts) - 1:
                        if current_stage:
                            stage = finalize_stage(current_stage, pause_after=pause_after_current)
                            if stage:
                                stages.append(stage)
                            current_stage = []
                            pause_after_current = False
        elif arg in [',', 'then']:
            # Separator found - finalize current stage if it has content
            if current_stage:
                stage = finalize_stage(current_stage, pause_after=pause_after_current)
                if stage:
                    stages.append(stage)
                current_stage = []
                pause_after_current = False
        elif arg == '.':
            # Standalone dot - mark current stage for pause
            pause_after_current = True
        else:
            # Regular argument - add to current stage
            current_stage.append(arg)
    
    # Don't forget the last stage
    if current_stage:
        stage = finalize_stage(current_stage, pause_after=pause_after_current)
        if stage:
            stages.append(stage)
    
    return stages

def finalize_stage(stage_args: List[str], pause_after: bool = False) -> Optional[Dict[str, Any]]:
    """
    Convert raw stage arguments into structured stage dictionary.
    
    Args:
        stage_args: List of arguments between ',' or 'then' separators
        pause_after: If True, workflow should pause after this stage completes
        
    Returns:
        Dictionary with 'type', 'args', and optional 'pause_after' keys, or None if invalid
    """
    if not stage_args:
        return None
    
    first_arg = stage_args[0].lower()
    
    # Replication stage: r3, r5, etc.
    if first_arg.startswith('r') and first_arg[1:].isdigit():
        stage_dict = {
            'type': 'replication',
            'num_replicas': int(first_arg[1:]),
            'args': stage_args[1:]
        }
        if pause_after:
            stage_dict['pause_after'] = True
        return stage_dict
    
    # Optimization stage: opt/optimization ...
    # New naming: "opt" means Optimization stage (internal type: optimization)
    elif first_arg in ['opt', 'optimization']:
        stage_dict = {
            'type': 'optimization',
            'args': stage_args[1:]
        }
        if pause_after:
            stage_dict['pause_after'] = True
        return stage_dict
    
    # Similarity stage: similarity ... or simil ...
    elif first_arg in ['similarity', 'simil']:
        stage_dict = {
            'type': 'similarity',
            'args': stage_args[1:]
        }
        if pause_after:
            stage_dict['pause_after'] = True
        return stage_dict
    
    # Refinement stage: ref/refinement ... (internal type: refinement)
    elif first_arg in ['ref', 'refinement']:
        # Parse opt stage: opt [flags] template_file launcher_file
        # Find template and launcher (non-flag arguments)
        remaining_args = stage_args[1:]
        flags = []
        files = []
        
        for arg in remaining_args:
            if arg.startswith('--') or arg.startswith('-'):
                flags.append(arg)
            else:
                files.append(arg)
        
        # Expect at least 2 files: template and launcher
        template_inp = files[0] if len(files) >= 1 else None
        launcher_sh = files[1] if len(files) >= 2 else None
        
        stage_dict = {
            'type': 'refinement',
            'args': flags,  # Only flags, not file paths
            'template_inp': template_inp,
            'launcher_sh': launcher_sh
        }
        if pause_after:
            stage_dict['pause_after'] = True
        return stage_dict
    
    else:
        print(f"Warning: Unknown stage type '{first_arg}', skipping.")
        return None

def find_similarity_script() -> Optional[str]:
    """
    Locate similarity-v01.py script.
    Searches in: current dir, same dir as ascec script, scripts/ dir, parent dir, and PATH.
    
    Returns:
        Path to similarity script, or None if not found
    """
    # Get directory where this script (ascec-v04.py) is located
    ascec_dir = os.path.dirname(os.path.abspath(__file__))
    
    search_locations = [
        'similarity-v01.py',  # Current working directory
        'similarity.py',
        os.path.join(ascec_dir, 'similarity-v01.py'),  # Same dir as ascec script
        os.path.join(ascec_dir, 'similarity.py'),
        'scripts/similarity-v01.py',
        '../scripts/similarity-v01.py',
        os.path.expanduser('~/scripts/similarity-v01.py'),
    ]
    
    # Check each location
    for location in search_locations:
        if os.path.exists(location):
            return os.path.abspath(location)
    
    # Check if in PATH
    try:
        result = subprocess.run(['which', 'similarity-v01.py'], 
                              capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    return None

def parse_similarity_percentages(stdout_text: str) -> Tuple[float, float]:
    """
    Parse similarity stdout to extract critical and skipped percentages.
    
    Example output:
        Total files skipped: 47 (11.2%)
        Critical skipped files: 4 (1.0%)
    
    Returns:
        (critical_percentage, skipped_percentage)
    """
    import re
    
    critical_pct = 0.0
    skipped_pct = 0.0
    
    try:
        # Look for critical percentages from both legacy and new similarity outputs.
        critical_matches = re.findall(
            r'(?:Critical skipped files|Critical reduced-vector unmatched):.*?\(([0-9.]+)%\)',
            stdout_text,
            re.IGNORECASE,
        )
        if critical_matches:
            critical_pct = max(float(val) for val in critical_matches)
        
        # Look for "Total files skipped: 47 (11.2%)"
        skipped_match = re.search(r'Total files skipped:.*?\(([0-9.]+)%\)', stdout_text, re.IGNORECASE)
        if skipped_match:
            skipped_pct = float(skipped_match.group(1))
    except Exception as e:
        print(f"Warning: Could not parse similarity percentages: {e}")
    
    return (critical_pct, skipped_pct)


def parse_similarity_summary(summary_file: str) -> Tuple[float, float]:
    """
    Parse clustering_summary.txt file to extract critical and skipped percentages.
    
    Example content:
        Total files skipped: 1 (50.0%)
        Critical skipped files: 1 (50.0%)
    
    Returns:
        (critical_percentage, skipped_percentage)
    """
    import re
    
    critical_pct = 0.0
    skipped_pct = 0.0
    
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
            
        # Capture critical percentages from both formats and use the strictest value.
        critical_matches = re.findall(
            r'(?:Critical skipped files|Critical reduced-vector unmatched):.*?\(([0-9.]+)%\)',
            content,
            re.IGNORECASE,
        )
        if critical_matches:
            critical_pct = max(float(val) for val in critical_matches)
        
        # Look for "Total files skipped: 1 (50.0%)"
        skipped_match = re.search(r'Total files skipped:.*?\(([0-9.]+)%\)', content, re.IGNORECASE)
        if skipped_match:
            skipped_pct = float(skipped_match.group(1))
    except Exception as e:
        print(f"Warning: Could not parse similarity summary: {e}")
    
    return (critical_pct, skipped_pct)

def parse_similarity_output(similarity_dir: str) -> Tuple[int, int]:
    """
    Parse similarity output to extract critical and skipped file counts.
    
    Returns:
        (critical_count, skipped_count)
    """
    summary_file = os.path.join(similarity_dir, 'clustering_summary.txt')
    
    if not os.path.exists(summary_file):
        print(f"Warning: clustering_summary.txt not found in {similarity_dir}")
        return (0, 0)
    
    critical_count = 0
    skipped_count = 0
    
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
            
            # Critical count can come from skipped critical files or reduced-vector unmatched.
            import re
            critical_matches = re.findall(
                r'(?:Critical skipped files|Critical reduced-vector unmatched):\s*(\d+)',
                content,
                re.IGNORECASE,
            )
            if critical_matches:
                critical_count = max(int(val) for val in critical_matches)
            
            # Look for total skipped count like "Total files skipped: 97 (19.9%)"
            skipped_match = re.search(r'Total files skipped:\s*(\d+)', content, re.IGNORECASE)
            if skipped_match:
                skipped_count = int(skipped_match.group(1))
                
    except Exception as e:
        print(f"Warning: Error parsing similarity output: {e}")
    
    return (critical_count, skipped_count)

def get_critical_count(optimization_dir_path: str) -> int:
    """
    Count files with imaginary frequencies in calculation directory.
    
    Returns:
        Number of critical files found
    """
    critical_count = 0
    
    # Look for ORCA (.out) and Gaussian (.log) output files
    for root, dirs, files in os.walk(optimization_dir_path):
        for file in files:
            if file.endswith('.out') or file.endswith('.log'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        # Check for imaginary frequencies (ORCA and Gaussian)
                        if 'imaginary mode' in content.lower() or '***imaginary' in content.lower():
                            critical_count += 1
                except:
                    pass
    
    return critical_count

def get_critical_files_list(optimization_dir_path: str) -> List[str]:
    """
    Get list of files with imaginary frequencies.
    
    Returns:
        List of filepaths for files with imaginary frequencies
    """
    critical_files = []
    
    for root, dirs, files in os.walk(optimization_dir_path):
        for file in files:
            if file.endswith('.out') or file.endswith('.log'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        if 'imaginary mode' in content.lower() or '***imaginary' in content.lower():
                            critical_files.append(filepath)
                except:
                    pass
    
    return critical_files

def handle_imaginary_frequencies(critical_files: List[str], optimization_dir_path: str) -> int:
    """
    Process structures with imaginary frequencies:
    - For 1 imaginary frequency: Use orca_pltvib to displace along the mode
    - For 2+ imaginary frequencies: Re-optimize from final geometry
    
    Args:
        critical_files: List of output files with imaginary frequencies
        optimization_dir_path: Stage working directory
        
    Returns:
        Number of structures successfully processed
    """
    processed = 0
    
    for out_file in critical_files:
        try:
            # Count imaginary frequencies in this file
            imag_count = count_imaginary_frequencies(out_file)
            
            print(f"    {os.path.basename(out_file)}: {imag_count} imaginary frequency(ies)")
            
            if imag_count == 1:
                # Single imaginary frequency: displace along the mode
                if displace_along_imaginary_mode(out_file, optimization_dir_path):
                    processed += 1
                    print(f"      Displaced structure created")
                else:
                    print(f"      Could not create displaced structure")
            elif imag_count >= 2:
                # Multiple imaginary frequencies: displace along highest (most negative) mode
                if displace_along_imaginary_mode(out_file, optimization_dir_path, use_highest_mode=True):
                    processed += 1
                    print(f"      Displaced structure created (highest imaginary mode)")
                else:
                    print(f"      Could not create displaced structure")
        except Exception as e:
            print(f"      Error processing {os.path.basename(out_file)}: {e}")
    
    return processed

def count_imaginary_frequencies(out_file: str) -> int:
    """Count number of imaginary frequencies in ORCA output file by checking for negative values."""
    try:
        with open(out_file, 'r') as f:
            content = f.read()
            
        # Regex to match vibrational frequencies in ORCA output
        # Format:    6:      -123.45 cm-1
        import re
        # Look for the vibrational frequencies section
        if "VIBRATIONAL FREQUENCIES" in content:
            # Extract frequencies - handle both cm-1 and cm**-1
            # Format:    6:      -123.45 cm-1  OR  6:      -123.45 cm**-1
            freq_matches = re.findall(r"\s+\d+:\s+(-?\d+\.\d+)\s+cm(?:\*\*|)-1", content)
            if freq_matches:
                # Count negative frequencies
                imag_count = sum(1 for freq in freq_matches if float(freq) < 0)
                return imag_count
        
        # Fallback to old method if section not found (e.g. different format)
        count = content.lower().count('***imaginary mode***')
        if count == 0:
            matches = re.findall(r'imaginary mode', content, re.IGNORECASE)
            count = len(matches)
        return count
    except:
        return 0

def displace_along_imaginary_mode(out_file: str, optimization_dir_path: str, use_highest_mode: bool = False):
    """
    Create displaced structure along imaginary mode.
    
    MULTI-SOFTWARE SUPPORT (ORCA & Gaussian):
    ==========================================
    For ORCA: Directly extracts normal mode from output and applies displacement
    For Gaussian: Manually extracts and applies normal mode displacement vectors
    
    Args:
        out_file: Path to output file (.out for ORCA, .log for Gaussian)
        optimization_dir_path: Stage working directory (unused, retained for stable function signatures)
        use_highest_mode: If True, displace along the highest (most negative) imaginary mode.
                         If False (default), displace along the first imaginary mode.
        
    Returns:
        List of XYZ lines if successful, None otherwise
    """
    try:
        # Check if this is ORCA or Gaussian
        with open(out_file, 'r') as f:
            first_lines = ''.join(f.readlines()[:50])
            is_orca = 'O   R   C   A' in first_lines
            is_gaussian = 'Gaussian' in first_lines
        
        if is_orca:
            # ORCA: Parse normal modes directly from output file
            return displace_orca_imaginary_mode(out_file, use_highest_mode=use_highest_mode)
        
        elif is_gaussian:
            # Gaussian: Manually displace along imaginary mode
            return displace_gaussian_imaginary_mode(out_file, use_highest_mode=use_highest_mode)
        
        return None
        
    except Exception as e:
        return None

def displace_orca_imaginary_mode(out_file: str, displacement_factor: float = 0.5, use_highest_mode: bool = False):
    """
    Manually displace geometry along imaginary mode for ORCA output.
    Extracts geometry and normal mode directly from output, then creates displaced structure.
    
    This function parses the ORCA output to find:
    1. The final optimized geometry from "CARTESIAN COORDINATES (ANGSTROEM)" section
    2. The imaginary frequency mode number and vectors from "NORMAL MODES" section
    3. Applies displacement along the imaginary mode
    
    Args:
        out_file: ORCA output file (.out)
        displacement_factor: Scaling factor for displacement (default: 0.5 Angstrom)
                           This is larger than Gaussian default because ORCA modes
                           are mass-weighted and normalized
        use_highest_mode: If True, use the mode with the most negative frequency.
                         If False (default), use the first imaginary mode found.
    
    Returns:
        List of lines in XYZ format (including header) if successful, None otherwise
    """
    try:
        with open(out_file, 'r') as f:
            lines = f.readlines()
        
        # Step 1: Extract final geometry from last "CARTESIAN COORDINATES (ANGSTROEM)" section
        geometry = []
        coord_start = None
        for i in range(len(lines)-1, -1, -1):
            if 'CARTESIAN COORDINATES (ANGSTROEM)' in lines[i]:
                coord_start = i + 2  # Skip header lines
                break
        
        if coord_start is None:
            return False
        
        for i in range(coord_start, len(lines)):
            line = lines[i].strip()
            if not line or line.startswith('---') or line.startswith('='):
                break
            parts = line.split()
            if len(parts) >= 4:
                symbol = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                geometry.append((symbol, x, y, z))
        
        if not geometry:
            return False
        
        # Step 2: Find ALL imaginary frequency modes
        # Look in VIBRATIONAL FREQUENCIES section for negative frequencies
        imaginary_modes = []  # List of (mode_idx, freq_val)
        for i, line in enumerate(lines):
            if 'VIBRATIONAL FREQUENCIES' in line:
                # Search next 200 lines for imaginary modes
                for j in range(i, min(i+200, len(lines))):
                    # Check for frequency line (handle both cm-1 and cm**-1)
                    if 'cm-1' in lines[j] or 'cm**-1' in lines[j]:
                        # Format: "   6:       -16.68 cm**-1 ***imaginary mode***"
                        parts = lines[j].split(':')
                        if len(parts) >= 2:
                            try:
                                # Parse frequency value
                                freq_str = parts[1].strip().split()[0]
                                freq_val = float(freq_str)
                                
                                # Check if negative
                                if freq_val < 0:
                                    mode_idx = int(parts[0].strip())
                                    imaginary_modes.append((mode_idx, freq_val))
                            except (ValueError, IndexError):
                                continue
                break
        
        if not imaginary_modes:
            return False
        
        # Select which imaginary mode to use
        if use_highest_mode:
            # Use the mode with the most negative frequency (lowest value)
            imaginary_mode_idx, imaginary_freq = min(imaginary_modes, key=lambda x: x[1])
        else:
            # Use the first imaginary mode found
            imaginary_mode_idx, imaginary_freq = imaginary_modes[0]
        
        # Step 3: Find NORMAL MODES section and extract displacement vectors for imaginary mode
        mode_section_start = None
        for i, line in enumerate(lines):
            if 'NORMAL MODES' in line and '---' in lines[i+1] if i+1 < len(lines) else False:
                mode_section_start = i
                break
        
        if mode_section_start is None:
            return False
        
        # ORCA prints normal modes in blocks of 6 columns at a time
        # Format:
        #                   0          1          2          3          4          5    
        #       0       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
        #       1       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
        #       ...
        #                   6          7          8          9         10         11    
        #       0       0.009579   0.124156  -0.111529   0.004374   0.071995  -0.017842
        #       1      -0.002790   0.086907   0.114308   0.093436  -0.112979   0.018752
        
        displacements = []
        in_mode_block = False
        mode_column_idx = None
        
        for i in range(mode_section_start + 5, len(lines)):  # Skip header lines
            line = lines[i]
            
            # Check if this is a mode header line (contains only mode numbers)
            stripped = line.strip()
            if not stripped:
                continue
            
            # Mode header: line with only integers separated by spaces
            parts = stripped.split()
            if parts and all(p.isdigit() for p in parts):
                mode_nums = [int(p) for p in parts]
                if imaginary_mode_idx in mode_nums:
                    # Found the block containing our imaginary mode
                    mode_column_idx = mode_nums.index(imaginary_mode_idx)
                    in_mode_block = True
                    displacements = []
                    continue
                else:
                    in_mode_block = False
            
            # Read displacement vectors if we're in the right block
            if in_mode_block and mode_column_idx is not None:
                # Data line format: "     0       0.009579   0.124156  -0.111529 ..."
                # First column is row index, rest are displacement values
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        row_idx = int(parts[0])
                        # Get the value for our mode (column index + 1 to skip row number)
                        if len(parts) > mode_column_idx + 1:
                            value = float(parts[mode_column_idx + 1])
                            displacements.append(value)
                    except (ValueError, IndexError):
                        continue
                
                # Check if we have all displacements (3 * num_atoms)
                if len(displacements) == len(geometry) * 3:
                    break
        
        # Verify we got the right number of displacements
        if len(displacements) != len(geometry) * 3:
            return False
        
        # Step 4: Apply displacement to geometry
        displaced_geometry = []
        for atom_idx, (symbol, x, y, z) in enumerate(geometry):
            dx = displacements[atom_idx * 3]
            dy = displacements[atom_idx * 3 + 1]
            dz = displacements[atom_idx * 3 + 2]
            
            new_x = x + dx * displacement_factor
            new_y = y + dy * displacement_factor
            new_z = z + dz * displacement_factor
            displaced_geometry.append((symbol, new_x, new_y, new_z))
        
        # Step 5: Return displaced structure as XYZ lines (no file creation)
        xyz_lines = []
        xyz_lines.append(f"{len(displaced_geometry)}\n")
        # Include frequency in header for verification
        if imaginary_freq is not None:
            xyz_lines.append(f"Displaced along imaginary mode {imaginary_mode_idx} ({imaginary_freq:.2f} cm**-1, factor={displacement_factor:.2f} A)\n")
        else:
            xyz_lines.append(f"Displaced along imaginary mode {imaginary_mode_idx} (factor={displacement_factor:.2f} A)\n")
        for symbol, x, y, z in displaced_geometry:
            xyz_lines.append(f"{symbol:2s}  {x:15.8f}  {y:15.8f}  {z:15.8f}\n")
        
        return xyz_lines
        
    except Exception as e:
        return None

def displace_gaussian_imaginary_mode(out_file: str, displacement_factor: float = 0.3, use_highest_mode: bool = False):
    """
    Manually displace geometry along imaginary mode for Gaussian output.
    Extracts geometry and normal mode, then creates displaced structure.
    
    Args:
        out_file: Gaussian output file (.log)
        displacement_factor: Scaling factor for displacement (default: 0.3)
        use_highest_mode: If True, use the mode with the most negative frequency.
                         If False (default), use the first imaginary mode found.
    
    Returns:
        List of lines in XYZ format (including header) if successful, None otherwise
    """
    try:
        with open(out_file, 'r') as f:
            lines = f.readlines()
        
        # Extract final geometry (Standard orientation)
        geometry = []
        coord_start = None
        for i in range(len(lines)-1, -1, -1):
            if 'Standard orientation:' in lines[i]:
                coord_start = i + 5
                break
        
        if coord_start is None:
            return False
        
        for i in range(coord_start, len(lines)):
            line = lines[i].strip()
            if line.startswith('---'):
                break
            parts = line.split()
            if len(parts) >= 6:
                atomic_num = int(parts[1])
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                # Convert atomic number to symbol
                symbol = 'X'
                for sym, num in element_symbols.items():
                    if num == atomic_num:
                        symbol = sym
                        break
                geometry.append((symbol, x, y, z))
        
        if not geometry:
            return False
        
        # Find ALL imaginary frequency modes
        # Store as list of (line_index, column_index, frequency_value)
        imaginary_modes = []
        for i, line in enumerate(lines):
            if 'Frequencies --' in line:
                freqs = [float(x) for x in line.split()[2:]]
                for col_idx, freq_val in enumerate(freqs):
                    if freq_val < 0:
                        imaginary_modes.append((i, col_idx, freq_val))
        
        if not imaginary_modes:
            return False
        
        # Select which mode to use
        if use_highest_mode:
            # Use the mode with the most negative frequency
            mode_start, mode_column, selected_freq = min(imaginary_modes, key=lambda x: x[2])
        else:
            # Use the first imaginary mode found
            mode_start, mode_column, selected_freq = imaginary_modes[0]
        
        # Extract displacement vectors for this mode
        displacements = []
        reading_coords = False
        for i in range(mode_start + 1, min(mode_start + 200, len(lines))):
            line = lines[i]
            if 'Atom  AN' in line:
                reading_coords = True
                continue
            if reading_coords:
                if line.strip() == '' or 'Frequencies --' in line:
                    break
                parts = line.split()
                if len(parts) >= 5:
                    # Format: Atom AN X Y Z (for each mode column)
                    # Mode columns are in groups of 3: X, Y, Z
                    base_idx = 2 + mode_column * 3
                    if len(parts) > base_idx + 2:
                        dx = float(parts[base_idx])
                        dy = float(parts[base_idx + 1])
                        dz = float(parts[base_idx + 2])
                        displacements.append((dx, dy, dz))
        
        if len(displacements) != len(geometry):
            return False
        
        # Create displaced geometry
        displaced_geometry = []
        for (symbol, x, y, z), (dx, dy, dz) in zip(geometry, displacements):
            new_x = x + dx * displacement_factor
            new_y = y + dy * displacement_factor
            new_z = z + dz * displacement_factor
            displaced_geometry.append((symbol, new_x, new_y, new_z))
        
        # Return displaced structure as XYZ lines (no file creation)
        xyz_lines = []
        xyz_lines.append(f"{len(displaced_geometry)}\n")
        xyz_lines.append(f"Displaced along imaginary mode ({selected_freq:.2f} cm-1, factor={displacement_factor})\n")
        for symbol, x, y, z in displaced_geometry:
            xyz_lines.append(f"{symbol:2s}  {x:15.8f}  {y:15.8f}  {z:15.8f}\n")
        
        return xyz_lines
        
    except Exception as e:
        return None

def extract_displaced_frame(traj_file: str, frame: int = 10) -> Optional[str]:
    """Extract a specific frame from an XYZ trajectory file."""
    try:
        with open(traj_file, 'r') as f:
            lines = f.readlines()
        
        # Parse XYZ format
        current_frame = 0
        i = 0
        while i < len(lines):
            try:
                natoms = int(lines[i].strip())
                if current_frame == frame:
                    # Extract this frame
                    frame_lines = lines[i:i+natoms+2]
                    return ''.join(frame_lines)
                i += natoms + 2
                current_frame += 1
            except (ValueError, IndexError):
                i += 1
        
        return None
    except:
        return None

def format_ordinal(n):
    """
    Format a number with its ordinal suffix.
    Examples: 1→1st, 2→2nd, 3→3rd, 11→11th, 21→21st, 22→22nd
    """
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def extract_final_geometry(out_file: str, optimization_dir_path: str):
    """
    Extract final optimized geometry from output file.
    
    MULTI-SOFTWARE SUPPORT (ORCA & Gaussian):
    ==========================================
    For ORCA: Extracts from "CARTESIAN COORDINATES (ANGSTROEM)" section
    For Gaussian: Extracts from "Standard orientation" section
    
    Args:
        out_file: Path to output file (.out for ORCA, .log for Gaussian)
        optimization_dir_path: Stage working directory (unused, retained for stable function signatures)
        
    Returns:
        List of XYZ lines if successful, None otherwise
    """
    try:
        with open(out_file, 'r') as f:
            lines = f.readlines()
        
        # Determine if ORCA or Gaussian
        is_orca = any('O   R   C   A' in line for line in lines[:50])
        is_gaussian = any('Gaussian' in line for line in lines[:50])
        
        coords = []
        
        if is_orca:
            # Find the last "CARTESIAN COORDINATES (ANGSTROEM)" section
            coord_start = None
            for i in range(len(lines)-1, -1, -1):
                if 'CARTESIAN COORDINATES (ANGSTROEM)' in lines[i]:
                    coord_start = i + 2  # Skip header lines
                    break
            
            if coord_start is None:
                return None
            
            # Extract coordinates until blank line
            for i in range(coord_start, len(lines)):
                line = lines[i].strip()
                if not line or line.startswith('---'):
                    break
                coords.append(line)
        
        elif is_gaussian:
            # Find the last "Standard orientation" section
            coord_start = None
            for i in range(len(lines)-1, -1, -1):
                if 'Standard orientation:' in lines[i]:
                    coord_start = i + 5  # Skip header lines
                    break
            
            if coord_start is None:
                return None
            
            # Extract coordinates until separator line
            for i in range(coord_start, len(lines)):
                line = lines[i].strip()
                if line.startswith('---'):
                    break
                parts = line.split()
                if len(parts) >= 6:
                    # Gaussian format: Center Number, Atomic Number, Atomic Type, X, Y, Z
                    atomic_num = int(parts[1])
                    x, y, z = parts[3], parts[4], parts[5]
                    # Convert atomic number to symbol (reverse lookup from element_symbols)
                    symbol = 'X'
                    for sym, num in element_symbols.items():
                        if num == atomic_num:
                            symbol = sym
                            break
                    coords.append(f"{symbol}  {x}  {y}  {z}")
        
        if not coords:
            return None
        
        # Return as XYZ lines (no file creation)
        xyz_lines = []
        xyz_lines.append(f"{len(coords)}\n")
        xyz_lines.append("Final geometry for re-optimization\n")
        for line in coords:
            parts = line.split()
            if len(parts) >= 4:
                xyz_lines.append(f"{parts[0]:2s}  {parts[1]:>12s}  {parts[2]:>12s}  {parts[3]:>12s}\n")
        
        return xyz_lines
        
    except Exception as e:
        return None

def check_workflow_pause(stage: Dict[str, Any], stage_num: int, total_stages: int, 
                        cache_file: str, use_cache: bool) -> bool:
    """
    Check if workflow should pause after this stage.
    
    Args:
        stage: Stage dictionary that was just completed
        stage_num: Current stage number (1-indexed)
        total_stages: Total number of stages in workflow
        cache_file: Path to protocol cache file
        use_cache: Whether protocol caching is enabled
        
    Returns:
        True if workflow should continue, False if paused (will exit)
    """
    if not stage.get('pause_after', False):
        return True  # No pause requested, continue
    
    # Save pause state in cache
    if use_cache:
        cache = load_protocol_cache(cache_file)
        if not cache:
            cache = {}
        cache['paused_at_stage'] = stage_num
        cache['pause_timestamp'] = datetime.now().isoformat()
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
    
    # Display pause message
    print(f"\n{'='*70}")
    print(f"⏸  Workflow paused for manual review")
    print(f"{'='*70}")
    print(f"\nCompleted stage {stage_num}/{total_stages}: {stage['type'].capitalize()}")
    print(f"\nPlease review the results before continuing.")
    print(f"\nTo resume the workflow, run the same command again:")
    print(f"  The workflow will automatically continue from stage {stage_num + 1}")
    if use_cache:
        print(f"\nCache file: {cache_file}")
    print(f"\n{'='*70}\n")
    
    return False  # Signal to stop execution

def validate_cached_optimization_similarity(cache: dict, stage: Dict[str, Any], stage_num: int,
                                            stages: List[Dict[str, Any]], stage_idx: int,
                                            cache_file: str) -> Tuple[bool, int]:
    """
    Validate cached optimization+similarity results against thresholds.
    
    Returns:
        Tuple of (should_skip, new_stage_idx)
        - should_skip: True if cache is valid and should be skipped
        - new_stage_idx: Updated stage index if validation changes flow
    """
    stage_type = stage['type']
    stage_key = f"{stage_type}_{stage_num}"
    
    # Check if next stage is similarity AND it's also cached
    next_is_similarity = (stage_idx < len(stages) and stages[stage_idx]['type'] == 'similarity')
    if not next_is_similarity:
        return True, stage_idx
    
    next_stage_num = stage_idx + 1
    next_stage_key = f"similarity_{next_stage_num}"
    if next_stage_key not in cache.get('stages', {}) or \
       cache['stages'][next_stage_key].get('status') != 'completed':
        return True, stage_idx
    
    # Check if optimization stage has redo parameters
    optimization_args = stage['args']
    max_critical = None
    max_skipped = None
    
    for arg in optimization_args:
        if arg.startswith('--critical='):
            max_critical = float(arg.split('=')[1])
        elif arg.startswith('--skipped='):
            max_skipped = float(arg.split('=')[1])
    
    # If no thresholds set, cache is valid
    if max_critical is None and max_skipped is None:
        # Print skipped similarity stage
        sim_stage_cache = cache['stages'][next_stage_key]
        print(f"\n{'-' * 60}")
        print(f"[{next_stage_num}/{len(stages)}] Similarity (cached)")
        print(f"  ✓ Skipped (completed at {sim_stage_cache.get('timestamp', 'unknown')})")
        print('-' * 60)
        return True, stage_idx + 1
    
    # Get similarity directory from cache and validate
    sim_cache = cache['stages'][next_stage_key]
    sim_dir = sim_cache.get('result', {}).get('working_dir', 'similarity')
    summary_file = os.path.join(sim_dir, "clustering_summary.txt")
    
    if not os.path.exists(summary_file):
        # Can't validate, accept cache
        sim_stage_cache = cache['stages'][next_stage_key]
        print(f"\n{'-' * 60}")
        print(f"[{next_stage_num}/{len(stages)}] Similarity (cached)")
        print(f"  ✓ Skipped (completed at {sim_stage_cache.get('timestamp', 'unknown')})")
        print('-' * 60)
        return True, stage_idx + 1
    
    critical_pct, skipped_pct = parse_similarity_summary(summary_file)
    
    threshold_met = True
    if max_critical is not None:
        threshold_met = critical_pct <= max_critical
        if not threshold_met:
            print(f"\n⚠ Cached result invalid: critical {critical_pct:.1f}% > {max_critical}%")
            print(f"  Invalidating cache and re-running optimization with redo logic...")
    elif max_skipped is not None:
        threshold_met = skipped_pct <= max_skipped
        if not threshold_met:
            print(f"\n⚠ Cached result invalid: skipped {skipped_pct:.1f}% > {max_skipped}%")
            print(f"  Invalidating cache and re-running optimization with redo logic...")
    
    if not threshold_met:
        # Remove both optimization and similarity from cache
        if stage_key in cache.get('stages', {}):
            del cache['stages'][stage_key]
        if next_stage_key in cache.get('stages', {}):
            del cache['stages'][next_stage_key]
        # Save updated cache
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        # Go back to re-execute this stage
        return False, stage_idx - 1
    
    # Threshold met - skip both stages
    sim_stage_cache = cache['stages'][next_stage_key]
    print(f"\n{'-' * 60}")
    print(f"[{next_stage_num}/{len(stages)}] Similarity (cached)")
    print(f"  ✓ Skipped (completed at {sim_stage_cache.get('timestamp', 'unknown')})")
    print('-' * 60)
    return True, stage_idx + 1

def validate_cached_refinement_similarity(cache: dict, stage: Dict[str, Any], stage_num: int, 
                                   stages: List[Dict[str, Any]], stage_idx: int,
                                   cache_file: str) -> Tuple[bool, int]:
    """
    Validate cached optimization+similarity results against thresholds.
    
    Returns:
        Tuple of (should_skip, new_stage_idx)
        - should_skip: True if cache is valid and should be skipped
        - new_stage_idx: Updated stage index if validation changes flow
    """
    stage_type = stage['type']
    stage_key = f"{stage_type}_{stage_num}"
    
    # Check if next stage is similarity AND it's also cached
    next_is_similarity = (stage_idx < len(stages) and stages[stage_idx]['type'] == 'similarity')
    if not next_is_similarity:
        return True, stage_idx
    
    next_stage_num = stage_idx + 1
    next_stage_key = f"similarity_{next_stage_num}"
    if next_stage_key not in cache.get('stages', {}) or \
       cache['stages'][next_stage_key].get('status') != 'completed':
        return True, stage_idx
    
    # Check if optimization stage has redo parameters
    opt_args = stage['args']
    max_critical = None
    max_skipped = None
    
    for arg in opt_args:
        if arg.startswith('--critical='):
            max_critical = float(arg.split('=')[1])
        elif arg.startswith('--skipped='):
            max_skipped = float(arg.split('=')[1])
    
    # If no thresholds set, cache is valid
    if max_critical is None and max_skipped is None:
        # Print skipped similarity stage
        sim_stage_cache = cache['stages'][next_stage_key]
        print(f"\n{'-' * 60}")
        print(f"[{next_stage_num}/{len(stages)}] Similarity (cached)")
        print(f"  ✓ Skipped (completed at {sim_stage_cache.get('timestamp', 'unknown')})")
        print('-' * 60)
        return True, stage_idx + 1
    
    # Get similarity directory from cache and validate
    sim_cache = cache['stages'][next_stage_key]
    sim_dir = sim_cache.get('result', {}).get('working_dir', 'similarity')
    summary_file = os.path.join(sim_dir, "clustering_summary.txt")
    
    if not os.path.exists(summary_file):
        # Can't validate, accept cache
        sim_stage_cache = cache['stages'][next_stage_key]
        print(f"\n{'-' * 60}")
        print(f"[{next_stage_num}/{len(stages)}] Similarity (cached)")
        print(f"  ✓ Skipped (completed at {sim_stage_cache.get('timestamp', 'unknown')})")
        print('-' * 60)
        return True, stage_idx + 1
    
    critical_pct, skipped_pct = parse_similarity_summary(summary_file)
    
    threshold_met = True
    if max_critical is not None:
        threshold_met = critical_pct <= max_critical
        if not threshold_met:
            print(f"\n⚠ Cached result invalid: critical {critical_pct:.1f}% > {max_critical}%")
            print(f"  Invalidating cache and re-running optimization with redo logic...")
    elif max_skipped is not None:
        threshold_met = skipped_pct <= max_skipped
        if not threshold_met:
            print(f"\n⚠ Cached result invalid: skipped {skipped_pct:.1f}% > {max_skipped}%")
            print(f"  Invalidating cache and re-running optimization with redo logic...")
    
    if not threshold_met:
        # Remove both opt and similarity from cache
        if stage_key in cache.get('stages', {}):
            del cache['stages'][stage_key]
        if next_stage_key in cache.get('stages', {}):
            del cache['stages'][next_stage_key]
        # Save updated cache
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        # Go back to re-execute this stage
        return False, stage_idx - 1
    
    # Threshold met - skip both stages
    sim_stage_cache = cache['stages'][next_stage_key]
    print(f"\n{'-' * 60}")
    print(f"[{next_stage_num}/{len(stages)}] Similarity (cached)")
    print(f"  ✓ Skipped (completed at {sim_stage_cache.get('timestamp', 'unknown')})")
    print('-' * 60)
    return True, stage_idx + 1

# pyright: reportGeneralTypeIssues=false
def execute_workflow_stages(input_file: str, stages: List[Dict[str, Any]], 
                           use_cache: bool = False, protocol_text: str = "") -> int:
    """
    Execute all workflow stages in sequence with context passing.
    
    Args:
        input_file: Path to initial input file
        stages: List of stage dictionaries from parse_workflow_stages()
        use_cache: If True, use protocol_cache for resumability
        protocol_text: Original protocol text from .in file for summary
        
    Returns:
        0 on success, non-zero on failure
    """
    workflow_start_dt = datetime.now()
    cache_file: Optional[str] = None

    def format_compact_wall_time(seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60

        if days > 0:
            return f"{days}d {hours}h"
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def render_final_workflow_summary() -> None:
        nonlocal progress_lines, last_progress_render

        total = len(stages)
        bar = render_progress_bar(total, total, width=30)

        # Prefer cache-backed stage data so redo mode reports FINAL similarity counts.
        final_cache: Dict[str, Any] = {}
        if use_cache and isinstance(cache_file, str) and cache_file and os.path.exists(cache_file):
            final_cache = load_protocol_cache(cache_file) or {}
        cache_stages = final_cache.get('stages', {}) if isinstance(final_cache, dict) else {}

        similarity_counter = 0
        summary_lines: List[str] = []

        for idx, stage_def in enumerate(stages, start=1):
            stage_type = stage_def.get('type', '')
            stage_name = stage_display_map.get(stage_type, str(stage_type).capitalize())
            stage_line = f"[{idx}/{total}] {stage_name} ✓"

            if stage_type == 'similarity':
                similarity_counter += 1
                stage_name = "Similarity" if similarity_counter == 1 else f"Similarity_{similarity_counter}"
                stage_key = f"similarity_{idx}"
                stage_data = cache_stages.get(stage_key, {}) if isinstance(cache_stages, dict) else {}
                stage_result = stage_data.get('result', {}) if isinstance(stage_data, dict) else {}

                motifs_created = None
                inputs_count = None
                if isinstance(stage_result, dict):
                    motifs_created = stage_result.get('motifs_created')
                    inputs_count = stage_result.get('input_count')

                # Fallback to last known count if cache is unavailable.
                if motifs_created is None:
                    motifs_created = context.sim_motifs_created
                if inputs_count is None:
                    stage_input_counts = getattr(context, 'similarity_stage_input_counts', {})
                    inputs_count = stage_input_counts.get(idx)

                if motifs_created is not None:
                    if inputs_count is not None and inputs_count > 0:
                        stage_line = f"[{idx}/{total}] {stage_name} ({inputs_count}/{motifs_created}) ✓"
                    else:
                        stage_line = f"[{idx}/{total}] {stage_name} ({motifs_created}) ✓"
                else:
                    stage_line = f"[{idx}/{total}] {stage_name} ✓"

            summary_lines.append(stage_line)

        workflow_end_dt = datetime.now()
        wall_time_str = format_compact_wall_time((workflow_end_dt - workflow_start_dt).total_seconds())

        # Build the final panel as a single block so it repaints the live progress
        # in-place instead of printing a duplicate header below it.
        lines = [
            "",
            "=== ASCEC & Similarity ===",
            "-" * 60,
            f"Progress [{bar}] 100.0%",
            "-" * 60,
        ] + summary_lines + [
            "",
            "Workflow finished",
            f"Start: {workflow_start_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"End:   {workflow_end_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total wall time: {wall_time_str}",
            "-" * 60,
        ]

        if supports_ansi_repaint and progress_lines > 0:
            sys.stdout.write(f"\033[{progress_lines}A")
            for _ in range(progress_lines):
                sys.stdout.write("\033[2K\033[1B")
            sys.stdout.write(f"\033[{progress_lines}A")
            sys.stdout.flush()

        for line in lines:
            print(line)
        progress_lines = len(lines)
        last_progress_render = tuple(lines)

    def copy_final_ensemble_to_root() -> None:
        """Copy final ensemble files from the last similarity output to input root."""
        input_root = os.path.dirname(os.path.abspath(input_file))

        # Resolve the last similarity directory from context first, then cache fallback.
        similarity_candidates: List[str] = []
        if getattr(context, 'similarity_dir', None):
            similarity_candidates.append(str(context.similarity_dir))

        if use_cache and isinstance(cache_file, str) and cache_file and os.path.exists(cache_file):
            cache_data = load_protocol_cache(cache_file) or {}
            cache_stages = cache_data.get('stages', {}) if isinstance(cache_data, dict) else {}
            for idx in range(len(stages), 0, -1):
                if stages[idx - 1].get('type') != 'similarity':
                    continue
                stage_key = f"similarity_{idx}"
                stage_data = cache_stages.get(stage_key, {}) if isinstance(cache_stages, dict) else {}
                if not isinstance(stage_data, dict):
                    continue
                stage_result = stage_data.get('result', {})
                if not isinstance(stage_result, dict):
                    continue
                working_dir = stage_result.get('working_dir')
                if isinstance(working_dir, str) and working_dir:
                    similarity_candidates.append(working_dir)
                break

        resolved_similarity_dir = None
        for candidate in similarity_candidates:
            candidate_dir = candidate
            base_name = os.path.basename(candidate_dir)
            if base_name.startswith('orca_out_') or base_name.startswith('gaussian_out_') or base_name.startswith('opt_out_'):
                candidate_dir = os.path.dirname(candidate_dir)

            abs_candidate = os.path.abspath(candidate_dir)
            if os.path.isdir(abs_candidate):
                resolved_similarity_dir = abs_candidate
                break

        if not resolved_similarity_dir:
            print("Warning: Could not resolve final similarity directory for final ensemble copy.")
            return

        umotif_dirs = sorted(glob.glob(os.path.join(resolved_similarity_dir, 'umotifs_*')))
        motif_dirs = sorted(glob.glob(os.path.join(resolved_similarity_dir, 'motifs_*')))

        # Prefer the most refined/final ensemble when present.
        source_dir = umotif_dirs[-1] if umotif_dirs else (motif_dirs[-1] if motif_dirs else None)
        if not source_dir:
            print(f"Warning: No motifs_/umotifs_ folder found in {resolved_similarity_dir} for final ensemble copy.")
            return

        source_xyz = None
        source_mol = None

        preferred_xyz = [
            os.path.join(source_dir, 'all_umotifs_combined.xyz'),
            os.path.join(source_dir, 'all_motifs_combined.xyz'),
        ]
        preferred_mol = [
            os.path.join(source_dir, 'all_umotifs_combined.mol'),
            os.path.join(source_dir, 'all_motifs_combined.mol'),
        ]

        for path in preferred_xyz:
            if os.path.exists(path):
                source_xyz = path
                break
        for path in preferred_mol:
            if os.path.exists(path):
                source_mol = path
                break

        if source_xyz is None:
            xyz_candidates = sorted(glob.glob(os.path.join(source_dir, 'all_*_combined.xyz')))
            if xyz_candidates:
                source_xyz = xyz_candidates[-1]
        if source_mol is None:
            mol_candidates = sorted(glob.glob(os.path.join(source_dir, 'all_*_combined.mol')))
            if mol_candidates:
                source_mol = mol_candidates[-1]

        copied_any = False
        copied_xyz = False
        copied_mol = False
        if source_xyz and os.path.exists(source_xyz):
            shutil.copy2(source_xyz, os.path.join(input_root, 'final_ensemble.xyz'))
            copied_any = True
            copied_xyz = True
        if source_mol and os.path.exists(source_mol):
            shutil.copy2(source_mol, os.path.join(input_root, 'final_ensemble.mol'))
            copied_any = True
            copied_mol = True

        if copied_any:
            if context.workflow_verbose_level >= 1:
                if copied_xyz:
                    print(f"Final ensemble copied to: {os.path.join(input_root, 'final_ensemble.xyz')}")
                if copied_mol:
                    print(f"Final ensemble copied to: {os.path.join(input_root, 'final_ensemble.mol')}")
        else:
            print(f"Warning: No final combined ensemble files found in {source_dir}.")

    context = WorkflowContext(input_file=input_file)
    context.is_workflow = True  # We're in workflow mode
    context.workflow_verbose_level = parse_verbosity_level(sys.argv)
    
    # Read configuration from input file
    # - Line 9: QM program index and alias (e.g., "2 orca")
    # - Line 11: nprocs (QM calculations and ASCEC evaluation)
    # Default to reasonable values if not specified or if input file doesn't exist
    try:
        if os.path.exists(input_file):
            with open(input_file, 'r') as f:
                lines = f.readlines()
                config_line_count = 0
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    config_line_count += 1
                    
                    if config_line_count == 9:  # Line 9: QM Program Index & Alias
                        parts = line.split('#')[0].strip().split()
                        if len(parts) > 1:
                            # Second value is the QM program alias (e.g., "orca", "g16")
                            context.qm_alias = parts[1]
                        elif len(parts) == 1:
                            # Only index provided, use default based on index
                            qm_idx = int(parts[0])
                            context.qm_alias = "orca" if qm_idx == 2 else "g16" if qm_idx == 1 else "orca"
                    
                    if config_line_count == 11:  # Line 11: nprocs
                        parts = line.split('#')[0].strip().split()
                        if len(parts) > 1:
                            # User explicitly specified ASCEC cores
                            context.ascec_parallel_cores = int(parts[1])
                        else:
                            # Only QM procs specified, use auto-detect (0)
                            context.ascec_parallel_cores = 0
                        break
    except (ValueError, IndexError, IOError):
        context.ascec_parallel_cores = 0  # Default to auto-detect if parsing fails
        context.qm_alias = "orca"  # Default alias
    
    # Use protocol-specific cache filename with random seed to support parallel protocols
    # First, check if there's an existing protocol cache file for THIS input file
    import glob
    existing_caches = sorted(glob.glob("protocol_*.pkl"))
    cache_file = None
    
    if existing_caches and use_cache:
        # Look for cache that matches this input file
        for cache_path in existing_caches:
            test_cache = load_protocol_cache(cache_path)
            if test_cache and test_cache.get('input_file') == input_file:
                cache_file = cache_path
                if context.workflow_verbose_level >= 1:
                    print(f"Found existing protocol cache: {cache_file} (for {input_file})")
                break
    
    if cache_file is None:
        # Generate new random 6-digit seed (similar to annealing seed)
        import random
        protocol_seed = random.randint(100000, 999999)
        cache_file = f"protocol_{protocol_seed}.pkl"
        if context.workflow_verbose_level >= 1:
            print(f"Creating new protocol cache: {cache_file} (for {input_file})")
    
    context.cache_file = cache_file  # Store in context for use by stages
    
    # Load cache if in protocol mode
    cache = {}
    if use_cache:
        cache = load_protocol_cache(cache_file)
        # Always ensure input_file is stored in cache
        if not cache:
            cache = {}
        if 'input_file' not in cache:
            cache['input_file'] = input_file
            # Save immediately so the association is recorded
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
        if cache:
            start_time = cache.get('start_time', None)
            completed_stages = cache.get('stages', {})
            
            # Check if protocol is completed
            is_completed = cache.get('completed', False)
            
            if is_completed and completed_stages:
                # Protocol was completed - offer resume from specific stage
                print(f"\n{'='*70}")
                print(f"Protocol was previously completed!")
                if start_time:
                    dt = datetime.fromtimestamp(start_time)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Original run started: {time_str}")
                
                # Show completed stages
                print(f"\nCompleted stages:")
                stage_list = []
                for idx, (stage_key, stage_data) in enumerate(sorted(completed_stages.items()), 1):
                    stage_type = stage_data.get('type', 'Unknown')
                    stage_list.append((idx, stage_key, stage_type))
                    print(f"  [{idx}] {stage_type}")
                
                print(f"\nOptions:")
                print(f"  [0] Start from beginning (will use existing directories)")
                for idx, _, stage_type in stage_list:
                    print(f"  [{idx}] Resume from {stage_type}")
                print(f"  [q] Quit")
                
                choice = input(f"\nSelect starting point [0-{len(stage_list)}, q]: ").strip().lower()
                
                if choice == 'q':
                    print("Exiting...")
                    sys.exit(0)
                
                try:
                    choice_idx = int(choice)
                    if choice_idx == 0:
                        # Start from beginning - clear cache but keep directories
                        print(f"Starting from beginning (existing directories will be reused)")
                        cache = {}
                        if protocol_text:
                            cache['protocol_text'] = protocol_text
                        cache['total_stages'] = len(stages)
                        save_protocol_cache(cache, cache_file)
                    elif 1 <= choice_idx <= len(stage_list):
                        # Resume from specific stage - mark previous stages as completed
                        resume_stage_key = stage_list[choice_idx - 1][1]
                        print(f"Resuming from stage: {stage_list[choice_idx - 1][2]}")
                        print(f"Previous stages marked as completed")
                        # Keep cache but mark as not completed (we're resuming)
                        cache['completed'] = False
                    else:
                        print(f"Invalid choice. Starting from beginning.")
                        cache = {}
                        if protocol_text:
                            cache['protocol_text'] = protocol_text
                        cache['total_stages'] = len(stages)
                        save_protocol_cache(cache, cache_file)
                except ValueError:
                    print(f"Invalid input. Starting from beginning.")
                    cache = {}
                    if protocol_text:
                        cache['protocol_text'] = protocol_text
                    cache['total_stages'] = len(stages)
                    save_protocol_cache(cache, cache_file)
                
                print(f"{'='*70}\n")
            elif start_time:
                # Protocol in progress - just resume
                dt = datetime.fromtimestamp(start_time)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                if context.workflow_verbose_level >= 1:
                    print(f"Resuming workflow (started: {time_str})")
            else:
                if context.workflow_verbose_level >= 1:
                    print(f"Resuming workflow")
            
            # Backfill total_stages for old cache files (compatibility)
            if 'total_stages' not in cache:
                cache['total_stages'] = len(stages)
                save_protocol_cache(cache, cache_file)
        else:
            # Store protocol text and total stages when first starting
            if protocol_text:
                cache['protocol_text'] = protocol_text
            cache['total_stages'] = len(stages)
            save_protocol_cache(cache, cache_file)
    
    # Pre-scan stages to extract similarity args for use in optimization stage
    for stage in stages:
        if stage['type'] == 'similarity':
            context.similarity_args = stage.get('args', [])
            break
    
    # Count optimization stages for proper numbering (opt, opt2)
    optimization_stage_counter = 0
    
    # Map stage types to display names
    stage_display_map: Dict[str, str] = {
        'replication': 'Annealing',
        'optimization': 'Optimization',
        'similarity': 'Similarity',
        'refinement': 'Refinement'
    }

    progress_lines = 0
    last_progress_render: Optional[Tuple[str, ...]] = None
    supports_ansi_repaint = bool(sys.stdout.isatty() or (os.environ.get("TERM") not in (None, "", "dumb")))

    def render_progress_bar(current: float, total: float, width: int = 30) -> str:
        if total <= 0:
            return "░" * width
        ratio = min(current / total, 1.0)
        filled = int(ratio * width)
        return "█" * filled + "░" * (width - filled)

    def render_workflow_progress(completed_stages: int, current_stage_num: int, sub_progress: str = "") -> None:
        """Render compact workflow progress with in-place updates (always visible)."""
        nonlocal progress_lines, last_progress_render

        total = len(stages)
        completed_stages = max(0, min(completed_stages, total))
        current_stage_num = max(1, min(current_stage_num, total))

        # Ensure the active stage is always ahead of completed stages to avoid blank panels.
        if completed_stages < total and current_stage_num <= completed_stages:
            current_stage_num = completed_stages + 1

        stage_lines = []
        for i, st in enumerate(stages, start=1):
            name = stage_display_map.get(st['type'], st['type'].capitalize())
            if i <= completed_stages:
                line = f"[{i}/{total}] {name} ✓"
                if st.get('type') == 'similarity':
                    stage_counts = getattr(context, 'similarity_stage_counts', {})
                    stage_input_counts = getattr(context, 'similarity_stage_input_counts', {})
                    stage_total = stage_counts.get(i)
                    stage_inputs = stage_input_counts.get(i)
                    if stage_total is None:
                        motif_count = getattr(context, 'last_similarity_motif_count', None)
                        umotif_count = getattr(context, 'last_similarity_umotif_count', None)
                        if motif_count is not None or umotif_count is not None:
                            m_val = motif_count if motif_count is not None else 0
                            u_val = umotif_count if umotif_count is not None else 0
                            stage_total = m_val + u_val
                    if stage_total is not None and stage_total > 0:
                        if stage_inputs is not None and stage_inputs > 0:
                            line += f" ({stage_inputs}/{stage_total})"
                        else:
                            line += f" ({stage_total})"
                stage_lines.append(line)
            elif i == current_stage_num and completed_stages < total:
                suffix = f" {sub_progress}" if sub_progress else " ..."
                stage_lines.append(f"[{i}/{total}] {name}{suffix}")
                break

        # Smooth progress within a stage using n/N updates (annealing/optimization/refinement only).
        stage_fraction = 0.0
        if completed_stages < total and 1 <= current_stage_num <= total:
            active_stage_type = stages[current_stage_num - 1].get('type')
            if active_stage_type in ('replication', 'optimization', 'refinement') and sub_progress:
                m = re.search(r'(\d+)\s*/\s*(\d+)', sub_progress)
                if m:
                    done = int(m.group(1))
                    stage_total = int(m.group(2))
                    if stage_total > 0:
                        stage_fraction = min(max(done / stage_total, 0.0), 1.0)

        progress_units = min(completed_stages + stage_fraction, float(total))
        pct = ((progress_units / total) * 100.0) if total > 0 else 0.0
        bar = render_progress_bar(progress_units, total, width=30)

        lines = [
            "",
            "=== ASCEC & Similarity ===",
            "-" * 60,
            f"Progress [{bar}] {pct:.1f}%",
            "-" * 60,
        ] + stage_lines + [""]

        # Skip duplicate redraw if the rendered panel text is identical.
        render_snapshot = tuple(lines)
        if render_snapshot == last_progress_render:
            return

        if supports_ansi_repaint and progress_lines > 0:
            sys.stdout.write(f"\033[{progress_lines}A")
            for _ in range(progress_lines):
                sys.stdout.write("\033[2K\033[1B")
            sys.stdout.write(f"\033[{progress_lines}A")
            sys.stdout.flush()

        for line in lines:
            print(line)
        progress_lines = len(lines)
        last_progress_render = render_snapshot

    context.completed_stage_count = 0
    last_progress_call: Optional[Tuple[int, int, str]] = None

    def _update_workflow_progress(sub_progress: str = "") -> None:
        nonlocal last_progress_call
        completed = context.completed_stage_count
        current = min(completed + 1, len(stages))
        call_sig = (completed, current, sub_progress)
        if call_sig == last_progress_call:
            return
        last_progress_call = call_sig
        render_workflow_progress(completed, current, sub_progress)

    context.update_progress = _update_workflow_progress
    
    # Cleaner output for protocol mode
    if use_cache:
        stage_display_names = []
        for s in stages:
            display_name = stage_display_map.get(s['type'], s['type'].capitalize()) or s['type'].capitalize()
            if s.get('pause_after', False):
                display_name += ' ⏸'
            stage_display_names.append(display_name)
        stage_names = ' → '.join(stage_display_names)
        if context.workflow_verbose_level >= 1:
            print(f"\nWorkflow: {stage_names}\n")
    else:
        if context.workflow_verbose_level >= 1:
            print("-" * 60)
            print(f"Workflow: {input_file}")
            stage_display_parts = []
            for s in stages:
                part = s['type']
                if s.get('pause_after', False):
                    part += ' ⏸'
                stage_display_parts.append(part)
            stage_names = ' → '.join(stage_display_parts)
            print(f"Pipeline: {stage_names}")
            print("-" * 60)

    completed_stage_count = 0
    
    # Execute each stage in sequence with optimization+similarity retry logic
    stage_idx = 0
    while stage_idx < len(stages):
        stage_num = stage_idx + 1
        stage = stages[stage_idx]
        stage_type = stage['type']
        # Expose active stage to helper functions that inspect threshold flags.
        context.current_stage = stage  # type: ignore[attr-defined]
        
        # Check if stage already completed (from cache)
        if use_cache and 'stages' in cache:
            stage_key = f"{stage_type}_{stage_num}"
            if stage_key in cache['stages']:
                stage_cache = cache['stages'][stage_key]
                if stage_cache.get('status') == 'completed':
                    # Display name (use new naming convention)
                    display_name = stage_display_map.get(stage_type, stage_type.capitalize())
                    if context.workflow_verbose_level >= 1:
                        print(f"\n{'-' * 60}")
                        print(f"[{stage_num}/{len(stages)}] {display_name} (cached)")
                        print(f"  ✓ Skipped (completed at {stage_cache.get('timestamp', 'unknown')})")
                        print('-' * 60)

                    completed_stage_count = stage_num
                    context.completed_stage_count = completed_stage_count
                    stage_idx += 1
                    
                    # Validate cached optimization+similarity if applicable
                    if stage_type == 'optimization':
                        should_skip, new_idx = validate_cached_optimization_similarity(
                            cache, stage, stage_num, stages, stage_idx, cache_file
                        )
                        if not should_skip:
                            stage_idx = new_idx
                            continue
                        stage_idx = new_idx
                    # Validate cached refinement+similarity if applicable
                    elif stage_type == 'refinement':
                        should_skip, new_idx = validate_cached_refinement_similarity(
                            cache, stage, stage_num, stages, stage_idx, cache_file
                        )
                        if not should_skip:
                            stage_idx = new_idx
                            continue
                        stage_idx = new_idx
                    continue
        
        # Display name for stage (use new naming convention)
        display_name = stage_display_map.get(stage_type, stage_type.capitalize())
        if context.workflow_verbose_level >= 1:
            print(f"\n{'-' * 60}")
            print(f"[{stage_num}/{len(stages)}] {display_name}")
            print('-' * 60)
        else:
            context.update_progress("")
        
        # Update cache - mark stage as in progress
        if use_cache:
            stage_key = f"{stage_type}_{stage_num}"
            context.current_stage_key = stage_key  # Store for use in stage execution
            update_protocol_cache(stage_key, 'in_progress', cache_file=cache_file)
        
        try:
            if stage_type == 'replication':
                result = execute_replication_stage(context, stage)
                
                # Update cache on success with detailed results
                if use_cache and result == 0:
                    stage_key = f"{stage_type}_{stage_num}"
                    
                    # Collect detailed annealing results
                    result_data = {'num_replicas': stage['num_replicas']}
                    
                    # Store directories for next stages to use
                    result_data['working_dir'] = '.'  # Annealing works in current dir
                    result_data['output_dir'] = '.'   # XYZ files created in current dir
                    result_data['annealing_dirs'] = context.annealing_dirs  # List of w6_annealing4_N directories
                    
                    # Get box size from stage context if available
                    if hasattr(context, 'annealing_box_size'):
                        result_data['box_size'] = context.annealing_box_size
                    if hasattr(context, 'annealing_packing'):
                        result_data['packing'] = context.annealing_packing
                    
                    # Count total accepted configurations from result files
                    total_accepted = 0
                    if context.annealing_dirs:
                        for adir in context.annealing_dirs:
                            # Count structures in result files
                            result_files = glob.glob(os.path.join(adir, "result_*.xyz"))
                            if result_files:
                                # Read XYZ file to count structures
                                with open(result_files[0], 'r') as f:
                                    lines = f.readlines()
                                    i = 0
                                    while i < len(lines):
                                        if lines[i].strip().isdigit():
                                            total_accepted += 1
                                            natoms = int(lines[i].strip())
                                            i += natoms + 2  # Skip atoms + comment line
                                        else:
                                            i += 1
                    
                    if total_accepted > 0:
                        result_data['total_accepted'] = total_accepted
                    
                    # Extract energy evaluations from annealing.out files
                    total_energy_evals = 0
                    if context.annealing_dirs:
                        for adir in context.annealing_dirs:
                            annealing_out = os.path.join(adir, 'annealing.out')
                            if os.path.exists(annealing_out):
                                try:
                                    with open(annealing_out, 'r') as f:
                                        content = f.read()
                                        match = re.search(r'Energy calculations:\s+(\d+)', content)
                                        if match:
                                            total_energy_evals += int(match.group(1))
                                except:
                                    pass
                    
                    if total_energy_evals > 0:
                        result_data['energy_evals'] = total_energy_evals
                    
                    update_protocol_cache(stage_key, 'completed', 
                                        result=result_data, 
                                        cache_file=cache_file)
                
                # Check if workflow should pause after this stage
                if result == 0:
                    if not check_workflow_pause(stage, stage_num, len(stages), cache_file, use_cache):
                        return 0  # Paused successfully
                
                stage_idx += 1
                completed_stage_count = stage_num
                context.completed_stage_count = completed_stage_count
                
            elif stage_type == 'optimization':
                # Increment optimization stage counter
                optimization_stage_counter += 1
                context.optimization_stage_number = optimization_stage_counter
                
                # Check if next stage is similarity - if so, handle optimization+similarity with retry
                next_is_similarity = (stage_idx + 1 < len(stages) and 
                                     stages[stage_idx + 1]['type'] == 'similarity')
                
                if next_is_similarity:
                    # Extract redo parameters from optimization stage
                    optimization_args = stage['args']
                    max_redos = 3  # Default: 3 redo attempts for critical structures
                    max_critical = 0  # Default: 0% critical structures allowed (retry all)
                    max_skipped = None   # Not set by default
                    
                    for arg in optimization_args:
                        if arg.startswith('--redo='):
                            max_redos = int(arg.split('=')[1])
                        elif arg.startswith('--critical='):
                            max_critical = float(arg.split('=')[1])
                        elif arg.startswith('--skipped='):
                            max_skipped = float(arg.split('=')[1])
                    
                    # Determine which threshold is active
                    if max_critical is not None and max_skipped is not None:
                        print("✗ Error: Cannot use both --critical and --skipped flags")
                        return 1
                    
                    # Show redo configuration
                    if max_redos > 1:
                        if context.workflow_verbose_level >= 1:
                            if max_critical is not None:
                                print(f"  Stage redo enabled: max {max_redos} redo attempts, target critical ≤ {max_critical}%")
                            elif max_skipped is not None:
                                print(f"  Stage redo enabled: max {max_redos} redo attempts, target skipped ≤ {max_skipped}%")
                    
                    # Redo loop for optimization+similarity
                    # attempt 0 = initial run, attempts 1..max_redos = redo attempts
                    final_attempt = 0
                    initial_critical = None  # Track initial critical % from first attempt
                    initial_skipped = None   # Track initial skipped % from first attempt
                    initial_critical_count = None
                    initial_skipped_count = None
                    for attempt in range(max_redos + 1):
                        final_attempt = attempt
                        if attempt > 0 and context.workflow_verbose_level >= 1:
                            print(f"\n{'-' * 60}")
                            print(f"Redo {attempt}/{max_redos}")
                        
                        # Don't delete similarity folder - we'll update it with corrected calculations
                        
                        # Run calculation (which includes sort step that creates similarity/)
                        result = execute_optimization_stage(context, stage)
                        if result != 0:
                            print(f"\nError: Optimization failed with code {result}")
                            return result
                        
                        # If this is a redo attempt and we have recalculated files, copy them to similarity folder
                        if attempt > 0 and hasattr(context, 'recalculated_files') and context.recalculated_files:
                            # Get calculation and similarity directories
                            optimization_dir_path = getattr(context, 'optimization_stage_dir', 'optimization')
                            # Get similarity orca output directory
                            sim_dir = context.similarity_dir if hasattr(context, 'similarity_dir') else "similarity"
                            
                            # Find the orca_out directory in similarity
                            orca_dirs = glob.glob(os.path.join(sim_dir, "orca_out*"))
                            if orca_dirs:
                                sim_orca_dir = orca_dirs[0]  # Use first match (e.g., orca_out_581)
                                
                                # Clean similarity directory BEFORE copying files (keep only orca_out and cache)
                                if sim_dir and os.path.exists(sim_dir):
                                    items_to_remove = [
                                        'dendrogram_images', 'extracted_clusters', 'extracted_data',
                                        'skipped_structures', 'clustering_summary.txt', 'boltzmann_distribution.txt'
                                    ]
                                    # Also remove motifs and umotifs folders
                                    for item in os.listdir(sim_dir):
                                        if item.startswith('motifs_') or item.startswith('umotifs_'):
                                            items_to_remove.append(item)
                                    
                                    for item in items_to_remove:
                                        item_path = os.path.join(sim_dir, item)
                                        if os.path.exists(item_path):
                                            try:
                                                if os.path.isdir(item_path):
                                                    shutil.rmtree(item_path)
                                                else:
                                                    os.remove(item_path)
                                            except Exception as e:
                                                print(f"    Error removing {item}: {e}")
                                
                                # Copy updated .out files from calculation subdirectories to similarity
                                for basename in context.recalculated_files:
                                    # Find the .out file in calculation subdirectories
                                    calc_subdir = os.path.join(optimization_dir_path, basename)
                                    calc_out_file = os.path.join(calc_subdir, f"{basename}.out")
                                    
                                    if os.path.exists(calc_out_file):
                                        # Copy to similarity orca directory
                                        sim_out_file = os.path.join(sim_orca_dir, f"{basename}.out")
                                        shutil.copy2(calc_out_file, sim_out_file)
                            else:
                                print(f"\n  Warning: No orca_out directory found in {sim_dir}/")
                        
                        # Run similarity; stage header is only shown in verbose mode.
                        if attempt == 0 and context.workflow_verbose_level >= 1:
                            print(f"\n{'-' * 60}")
                            print(f"[{stage_idx + 2}/{len(stages)}] Similarity")
                            print('-' * 60)
                        
                        similarity_stage = stages[stage_idx + 1]
                        result = execute_similarity_stage(context, similarity_stage)
                        if result != 0:
                            print(f"\nError: Similarity failed with code {result}")
                            return result
                        
                        # Parse similarity results from clustering_summary.txt
                        summary_file = os.path.join(context.similarity_dir, "clustering_summary.txt")
                        if os.path.exists(summary_file):
                            critical_pct, skipped_pct = parse_similarity_summary(summary_file)
                            
                            # Capture initial values on first attempt
                            if attempt == 0:
                                initial_critical = critical_pct
                                initial_skipped = skipped_pct
                                # Get counts for initial attempt
                                sim_dir = context.similarity_dir if context.similarity_dir else "similarity"
                                init_crit_count, init_skip_count = parse_similarity_output(sim_dir)
                                initial_critical_count = init_crit_count
                                initial_skipped_count = init_skip_count
                            
                            if context.workflow_verbose_level >= 1:
                                print(f"\nResults: {critical_pct:.1f}% critical, {skipped_pct:.1f}% skipped")
                            
                            # Check thresholds based on which was set
                            threshold_met = True
                            
                            if max_critical is not None:
                                threshold_met = critical_pct <= max_critical
                                
                                if threshold_met:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold met (critical ≤ {max_critical}%)")
                                    break
                                else:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold exceeded (critical {critical_pct:.1f}% > {max_critical}%)")
                                    
                            elif max_skipped is not None:
                                threshold_met = skipped_pct <= max_skipped
                                
                                if threshold_met:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold met (skipped ≤ {max_skipped}%)")
                                    break
                                else:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold exceeded (skipped {skipped_pct:.1f}% > {max_skipped}%)")
                            else:
                                # No thresholds set - accept results
                                break
                            
                            # If threshold not met and attempts remain, continue to redo logic below
                            if not threshold_met:
                                if attempt < max_redos:
                                    # Loop will continue to next iteration
                                    pass
                                else:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"Max attempts reached")
                        else:
                            if context.workflow_verbose_level >= 1:
                                print("⚠ Warning: Could not find clustering_summary.txt")
                            break
                    
                    # Get final similarity results for cache
                    final_critical = None
                    final_skipped = None
                    summary_file = os.path.join(context.similarity_dir, "clustering_summary.txt")
                    if os.path.exists(summary_file):
                        final_critical, final_skipped = parse_similarity_summary(summary_file)
                    
                    # Update cache for both optimization and similarity stages
                    if use_cache:
                        calc_key = f"optimization_{stage_num}"
                        sim_key = f"similarity_{stage_num + 1}"
                        
                        calc_result: Dict[str, Any] = {
                            'attempts': final_attempt,
                            'max_redos': max_redos,
                        }
                        
                        # Store directories for stage memory
                        # Get input dir from previous stage (annealing or current dir)
                        input_dir = context.get_previous_stage_output_dir('replication')
                        if input_dir is None:
                            input_dir = '.'  # Default if no previous stage
                        
                        optimization_dir_path = context.optimization_stage_dir if context.optimization_stage_dir else "calculation"
                        calc_result['input_dir'] = input_dir
                        calc_result['working_dir'] = optimization_dir_path
                        calc_result['output_dir'] = optimization_dir_path  # .out files are in optimization_dir_path
                        
                        if max_critical is not None:
                            calc_result['critical_threshold'] = max_critical
                        if max_skipped is not None:
                            calc_result['skipped_threshold'] = max_skipped
                        
                        # Add XYZ source info if available
                        if hasattr(context, 'optimization_xyz_source'):
                            calc_result['xyz_source'] = context.optimization_xyz_source
                        
                        # Add completion counts if available
                        if hasattr(context, 'optimization_completed'):
                            calc_result['completed'] = context.optimization_completed
                        if hasattr(context, 'optimization_total'):
                            calc_result['total'] = context.optimization_total
                        
                        # Add similarity folder info if available
                        if hasattr(context, 'optimization_sim_folder'):
                            calc_result['similarity_folder'] = context.optimization_sim_folder
                        
                        update_protocol_cache(calc_key, 'completed', 
                                            result=calc_result, cache_file=cache_file)
                        
                        sim_result = {}
                        
                        # Store directories for stage memory
                        sim_dir = context.similarity_dir if context.similarity_dir else "similarity"
                        optimization_dir_path = context.optimization_stage_dir if context.optimization_stage_dir else "calculation"
                        sim_result['input_dir'] = optimization_dir_path  # Read from calculation
                        sim_result['working_dir'] = sim_dir
                        # After calculation: use "motifs" prefix (first level clustering)
                        sim_result['output_dir'] = os.path.join(sim_dir, "motifs")  # Motifs for opt stage
                        
                        if final_critical is not None:
                            sim_result['critical_pct'] = final_critical
                        if final_skipped is not None:
                            sim_result['skipped_pct'] = final_skipped
                        
                        # Extract critical and skipped counts from similarity output
                        critical_count, skipped_count = parse_similarity_output(sim_dir)
                        sim_result['critical_count'] = critical_count
                        sim_result['skipped_count'] = skipped_count
                        
                        # Extract threshold value from similarity command args
                        sim_stage = stages[stage_idx + 1] if stage_idx + 1 < len(stages) else {}
                        sim_args = sim_stage.get('args', [])
                        for arg in sim_args:
                            if arg.startswith('--th=') or arg.startswith('--threshold='):
                                threshold_val = float(arg.split('=')[1])
                                sim_result['threshold'] = threshold_val
                            elif arg.startswith('--rmsd='):
                                rmsd_val = float(arg.split('=')[1])
                                sim_result['rmsd_threshold'] = rmsd_val
                        
                        # Add similarity folder and motifs info if available
                        if hasattr(context, 'sim_folder'):
                            sim_result['similarity_folder'] = context.sim_folder
                        if hasattr(context, 'sim_motifs_created'):
                            sim_result['motifs_created'] = context.sim_motifs_created
                        
                        # Add initial validation values (from first attempt)
                        if initial_critical is not None:
                            sim_result['initial_critical'] = initial_critical
                        if initial_skipped is not None:
                            sim_result['initial_skipped'] = initial_skipped
                        if initial_critical_count is not None:
                            sim_result['initial_critical_count'] = initial_critical_count
                        if initial_skipped_count is not None:
                            sim_result['initial_skipped_count'] = initial_skipped_count
                        
                        # Add threshold info and attempts
                        sim_result['attempts'] = final_attempt
                        if max_critical is not None:
                            sim_result['threshold_type'] = 'critical'
                            sim_result['threshold_value'] = max_critical
                            sim_result['threshold_met'] = (final_critical is not None and final_critical <= max_critical)
                        elif max_skipped is not None:
                            sim_result['threshold_type'] = 'skipped'
                            sim_result['threshold_value'] = max_skipped
                            sim_result['threshold_met'] = (final_skipped is not None and final_skipped <= max_skipped)
                        
                        update_protocol_cache(sim_key, 'completed', 
                                            result=sim_result, cache_file=cache_file)
                    
                    # Check if workflow should pause after optimization stage
                    if not check_workflow_pause(stage, stage_num, len(stages), cache_file, use_cache):
                        return 0  # Paused after optimization
                    
                    # Check if workflow should pause after similarity stage (check next stage for pause marker)
                    if stage_idx + 1 < len(stages):
                        sim_stage = stages[stage_idx + 1]
                        if not check_workflow_pause(sim_stage, stage_num + 1, len(stages), cache_file, use_cache):
                            return 0  # Paused after similarity
                    
                    # Skip both optimization and similarity stages since we handled them
                    stage_idx += 2
                    completed_stage_count = stage_num + 1
                    context.completed_stage_count = completed_stage_count
                else:
                    # Standalone optimization without similarity
                    result = execute_optimization_stage(context, stage)
                    
                    # Check if workflow should pause after this stage
                    if result == 0:
                        if not check_workflow_pause(stage, stage_num, len(stages), cache_file, use_cache):
                            return 0  # Paused successfully
                    
                    stage_idx += 1
                    completed_stage_count = stage_num
                    context.completed_stage_count = completed_stage_count
                    
            elif stage_type == 'similarity':
                # Standalone similarity (not after optimization/refinement in combined mode)
                result = execute_similarity_stage(context, stage)
                
                # Check if previous stage was optimization or refinement with threshold requirements
                if result == 0 and stage_idx > 0:
                    prev_stage = stages[stage_idx - 1]
                    prev_stage_type = prev_stage['type']
                    
                    if prev_stage_type in ['optimization', 'refinement']:
                        # Check if previous stage had redo parameters
                        prev_args = prev_stage['args']
                        max_critical = None
                        max_skipped = None
                        max_redos = 1
                        
                        for arg in prev_args:
                            if arg.startswith('--critical='):
                                max_critical = float(arg.split('=')[1])
                            elif arg.startswith('--skipped='):
                                max_skipped = float(arg.split('=')[1])
                            elif arg.startswith('--redo='):
                                max_redos = int(arg.split('=')[1])
                        
                        # If thresholds are set and redo is enabled, check results
                        if (max_critical is not None or max_skipped is not None) and max_redos > 1:
                            summary_file = os.path.join(context.similarity_dir, "clustering_summary.txt")
                            
                            if os.path.exists(summary_file):
                                critical_pct, skipped_pct = parse_similarity_summary(summary_file)
                                
                                if context.workflow_verbose_level >= 1:
                                    print(f"\nResults: {critical_pct:.1f}% critical, {skipped_pct:.1f}% skipped")
                                
                                threshold_met = True
                                if max_critical is not None:
                                    threshold_met = critical_pct <= max_critical
                                    if not threshold_met:
                                        if context.workflow_verbose_level >= 1:
                                            print(f"→ Threshold exceeded (critical {critical_pct:.1f}% > {max_critical}%)")
                                        print(f"  Invalidating cache and re-running {prev_stage_type} with redo logic...")
                                        
                                        # Invalidate both previous stage and similarity stages
                                        prev_key = f"{prev_stage_type}_{stage_idx}"  # Previous stage
                                        sim_key = f"similarity_{stage_num}"          # Current stage
                                        
                                        if use_cache and 'stages' in cache:
                                            if prev_key in cache['stages']:
                                                del cache['stages'][prev_key]
                                            if sim_key in cache['stages']:
                                                del cache['stages'][sim_key]
                                            
                                            # Save updated cache
                                            with open(cache_file, 'wb') as f:
                                                pickle.dump(cache, f)
                                        
                                        # Go back to re-execute previous stage
                                        stage_idx -= 1
                                        continue
                                    else:
                                        if context.workflow_verbose_level >= 1:
                                            print(f"→ Threshold met (critical ≤ {max_critical}%)")
                                        
                                elif max_skipped is not None:
                                    threshold_met = skipped_pct <= max_skipped
                                    if not threshold_met:
                                        if context.workflow_verbose_level >= 1:
                                            print(f"→ Threshold exceeded (skipped {skipped_pct:.1f}% > {max_skipped}%)")
                                        print(f"  Invalidating cache and re-running {prev_stage_type} with redo logic...")
                                        
                                        # Invalidate both previous stage and similarity stages
                                        prev_key = f"{prev_stage_type}_{stage_idx}"
                                        sim_key = f"similarity_{stage_num}"
                                        
                                        if use_cache and 'stages' in cache:
                                            if prev_key in cache['stages']:
                                                del cache['stages'][prev_key]
                                            if sim_key in cache['stages']:
                                                del cache['stages'][sim_key]
                                            
                                            # Save updated cache
                                            with open(cache_file, 'wb') as f:
                                                pickle.dump(cache, f)
                                        
                                        # Go back to re-execute previous stage
                                        stage_idx -= 1
                                        continue
                                    else:
                                        if context.workflow_verbose_level >= 1:
                                            print(f"→ Threshold met (skipped ≤ {max_skipped}%)")
                
                # Save similarity result to cache
                if result == 0 and use_cache:
                    from datetime import datetime as dt_sim
                    sim_key = f"similarity_{stage_num}"
                    sim_result: Dict[str, Any] = {
                        'status': 'completed',
                        'end_time': dt_sim.now().isoformat()
                    }
                    
                    # Store directories for stage memory
                    sim_dir = context.similarity_dir if context.similarity_dir else "similarity"
                    
                    # Determine input directory and output prefix based on previous stage
                    # Check if previous stage was refinement or optimization
                    prev_was_opt = (stage_idx > 0 and stages[stage_idx - 1]['type'] == 'refinement')
                    
                    if prev_was_opt:
                        # After refinement: input from refinement dir, output to umotifs
                        opt_dir = context.refinement_stage_dir if context.refinement_stage_dir else "refinement"
                        sim_result['input_dir'] = opt_dir
                        sim_result['output_dir'] = os.path.join(sim_dir, "umotifs")
                    else:
                        # After optimization: input from optimization dir, output to motifs
                        optimization_dir_path = context.optimization_stage_dir if context.optimization_stage_dir else "calculation"
                        sim_result['input_dir'] = optimization_dir_path
                        sim_result['output_dir'] = os.path.join(sim_dir, "motifs")
                    
                    sim_result['working_dir'] = sim_dir
                    
                    # Add similarity folder and motifs info if available
                    if hasattr(context, 'sim_folder') and context.sim_folder:
                        sim_result['similarity_folder'] = context.sim_folder
                    if hasattr(context, 'sim_motifs_created') and context.sim_motifs_created is not None:
                        sim_result['motifs_created'] = context.sim_motifs_created
                    sim_stage_num_s = stage_num
                    input_cnt_s = getattr(context, 'similarity_stage_input_counts', {}).get(sim_stage_num_s)
                    if input_cnt_s:
                        sim_result['input_count'] = input_cnt_s
                    
                    # Extract critical and skipped percentages and counts
                    summary_file = os.path.join(sim_dir, "clustering_summary.txt")
                    if os.path.exists(summary_file):
                        critical_pct, skipped_pct = parse_similarity_summary(summary_file)
                        if critical_pct is not None:
                            sim_result['critical_pct'] = critical_pct
                        if skipped_pct is not None:
                            sim_result['skipped_pct'] = skipped_pct
                        
                        critical_count, skipped_count = parse_similarity_output(sim_dir)
                        sim_result['critical_count'] = critical_count
                        sim_result['skipped_count'] = skipped_count
                    
                    # Extract threshold value from command args
                    sim_args = stage.get('args', [])
                    for arg in sim_args:
                        if arg.startswith('--th=') or arg.startswith('--threshold='):
                            threshold_val = float(arg.split('=')[1])
                            sim_result['threshold'] = threshold_val
                            break
                    
                    update_protocol_cache(sim_key, 'completed', 
                                        result=sim_result, cache_file=cache_file)
                
                # Check if workflow should pause after this stage
                if result == 0:
                    if not check_workflow_pause(stage, stage_num, len(stages), cache_file, use_cache):
                        return 0  # Paused successfully
                
                stage_idx += 1
                completed_stage_count = stage_num
                context.completed_stage_count = completed_stage_count
                
            elif stage_type == 'refinement':
                # Check if next stage is similarity - if so, handle opt+similarity with retry
                next_is_similarity = (stage_idx + 1 < len(stages) and 
                                     stages[stage_idx + 1]['type'] == 'similarity')
                
                if next_is_similarity:
                    # Extract redo parameters from opt stage
                    opt_args = stage['args']
                    max_redos = 3  # Default: 3 redo attempts for critical structures
                    max_critical = 0  # Default: 0% critical structures allowed (retry all)
                    max_skipped = None
                    
                    for arg in opt_args:
                        if arg.startswith('--redo='):
                            max_redos = int(arg.split('=')[1])
                        elif arg.startswith('--critical='):
                            max_critical = float(arg.split('=')[1])
                        elif arg.startswith('--skipped='):
                            max_skipped = float(arg.split('=')[1])
                    
                    # Determine which threshold is active
                    if max_critical is not None and max_skipped is not None:
                        print("✗ Error: Cannot use both --critical and --skipped flags")
                        return 1
                    
                    # Show redo configuration
                    if max_redos > 1:
                        if context.workflow_verbose_level >= 1:
                            if max_critical is not None:
                                print(f"  Stage redo enabled: max {max_redos} redo attempts, target critical ≤ {max_critical}%")
                            elif max_skipped is not None:
                                print(f"  Stage redo enabled: max {max_redos} redo attempts, target skipped ≤ {max_skipped}%")
                    
                    # Redo loop for opt+similarity
                    # attempt 0 = initial run, attempts 1..max_redos = redo attempts
                    final_attempt = 0
                    initial_critical = None  # Track initial critical % from first attempt
                    initial_skipped = None   # Track initial skipped % from first attempt
                    initial_critical_count = None
                    initial_skipped_count = None
                    for attempt in range(max_redos + 1):
                        final_attempt = attempt
                        if attempt > 0 and context.workflow_verbose_level >= 1:
                            print(f"\n{'-' * 60}")
                            print(f"Redo {attempt}/{max_redos}")
                        
                        # Run refinement (includes organizing/copying files to similarity)
                        result = execute_refinement_stage(context, stage)
                        if result != 0:
                            print(f"\nError: Refinement failed with code {result}")
                            # Invalidate cache so we can resume from this stage
                            opt_key = f"refinement_{stage_num}"
                            if use_cache and 'stages' in cache and opt_key in cache['stages']:
                                del cache['stages'][opt_key]
                                with open(cache_file, 'wb') as f:
                                    pickle.dump(cache, f)
                                print(f"  Cache invalidated for stage {stage_num}")
                            return result
                        
                        # If this is a redo attempt and we have recalculated files, copy them to similarity folder
                        if attempt > 0 and hasattr(context, 'recalculated_files') and context.recalculated_files:
                            # Get optimization and similarity directories
                            opt_dir = getattr(context, 'refinement_stage_dir', 'optimization') or 'optimization'
                            # Get similarity orca output directory
                            sim_dir = context.refinement_sim_folder if hasattr(context, 'refinement_sim_folder') else context.similarity_dir
                            
                            # Strip orca_out suffix if present - we want the BASE similarity directory
                            base_sim_dir = sim_dir
                            if base_sim_dir and 'orca_out' in base_sim_dir:
                                base_sim_dir = os.path.dirname(base_sim_dir)
                            
                            # Find the orca_out directory in similarity
                            # Check if sim_dir already includes orca_out folder
                            if sim_dir and ('orca_out' in sim_dir or 'gaussian_out' in sim_dir):
                                # sim_dir already points to orca_out folder (e.g., "similarity_2/orca_out_5")
                                sim_orca_dir = sim_dir
                            elif sim_dir:
                                # sim_dir is base folder, search for orca_out subdirectory
                                orca_dirs = glob.glob(os.path.join(sim_dir, "orca_out*"))
                                if orca_dirs:
                                    sim_orca_dir = orca_dirs[0]
                                else:
                                    sim_orca_dir = None
                            else:
                                sim_orca_dir = None
                            
                            if sim_orca_dir:
                                # Clean similarity directory BEFORE copying files (keep only orca_out and cache)
                                if base_sim_dir and os.path.exists(base_sim_dir):
                                    items_to_remove = [
                                        'dendrogram_images', 'extracted_clusters', 'extracted_data',
                                        'skipped_structures', 'clustering_summary.txt', 'boltzmann_distribution.txt'
                                    ]
                                    # Also remove motifs and umotifs folders
                                    for item in os.listdir(base_sim_dir):
                                        if item.startswith('motifs_') or item.startswith('umotifs_'):
                                            items_to_remove.append(item)
                                    
                                    for item in items_to_remove:
                                        item_path = os.path.join(base_sim_dir, item)
                                        if os.path.exists(item_path):
                                            try:
                                                if os.path.isdir(item_path):
                                                    shutil.rmtree(item_path)
                                                else:
                                                    os.remove(item_path)
                                            except Exception:
                                                pass
                                
                                # Copy updated .out files from optimization subdirectories to similarity
                                for basename in context.recalculated_files:
                                    # For optimization, files are grouped by shortened base name (motif_01_opt -> motif_01/)
                                    short_name = basename.replace('_opt', '').replace('_calc', '')
                                    opt_subdir = os.path.join(opt_dir, short_name)
                                    opt_out_file = os.path.join(opt_subdir, f"{basename}.out")
                                    
                                    if os.path.exists(opt_out_file):
                                        # Copy to similarity orca directory
                                        sim_out_file = os.path.join(sim_orca_dir, f"{basename}.out")
                                        shutil.copy2(opt_out_file, sim_out_file)
                            else:
                                print(f"\n  Warning: No orca_out directory found (sim_dir={sim_dir})")
                        
                        # Run similarity; stage header is only shown in verbose mode.
                        if attempt == 0 and context.workflow_verbose_level >= 1:
                            print(f"\n{'-' * 60}")
                            print(f"[{stage_idx + 2}/{len(stages)}] Similarity")
                            print('-' * 60)
                        
                        similarity_stage = stages[stage_idx + 1]
                        result = execute_similarity_stage(context, similarity_stage)
                        if result != 0:
                            print(f"\nError: Similarity failed with code {result}")
                            return result
                        
                        # Parse similarity results from clustering_summary.txt
                        summary_file = os.path.join(context.similarity_dir, "clustering_summary.txt")
                        if os.path.exists(summary_file):
                            critical_pct, skipped_pct = parse_similarity_summary(summary_file)
                            
                            # Capture initial values on first attempt
                            if attempt == 0:
                                initial_critical = critical_pct
                                initial_skipped = skipped_pct
                                # Get counts for initial attempt
                                sim_dir = context.similarity_dir if context.similarity_dir else "similarity_2"
                                init_crit_count, init_skip_count = parse_similarity_output(sim_dir)
                                initial_critical_count = init_crit_count
                                initial_skipped_count = init_skip_count
                            
                            if context.workflow_verbose_level >= 1:
                                print(f"\nResults: {critical_pct:.1f}% critical, {skipped_pct:.1f}% skipped")
                            
                            # Check thresholds
                            threshold_met = True
                            
                            if max_critical is not None:
                                threshold_met = critical_pct <= max_critical
                                if threshold_met:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold met (critical ≤ {max_critical}%)")
                                    break
                                else:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold exceeded (critical {critical_pct:.1f}% > {max_critical}%)")
                            elif max_skipped is not None:
                                threshold_met = skipped_pct <= max_skipped
                                if threshold_met:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold met (skipped ≤ {max_skipped}%)")
                                    break
                                else:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold exceeded (skipped {skipped_pct:.1f}% > {max_skipped}%)")
                            else:
                                break
                            
                            # If threshold not met and attempts remain, prepare for redo
                            if not threshold_met:
                                if attempt < max_redos:
                                    # Loop will continue to next iteration
                                    pass
                                else:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"Max attempts reached")
                        else:
                            if context.workflow_verbose_level >= 1:
                                print("⚠ Warning: Could not find clustering_summary.txt")
                            break
                    
                    # Get final similarity results for cache
                    final_critical = None
                    final_skipped = None
                    summary_file = os.path.join(context.similarity_dir, "clustering_summary.txt")
                    if os.path.exists(summary_file):
                        final_critical, final_skipped = parse_similarity_summary(summary_file)
                    
                    # Update cache for both refinement and similarity stages
                    if use_cache:
                        opt_key = f"refinement_{stage_num}"
                        sim_key = f"similarity_{stage_num + 1}"
                        
                        # Build result data
                        opt_result: Dict[str, Any] = {
                            'attempts': final_attempt,
                            'max_redos': max_redos,
                        }
                        
                        # Store directories for stage memory
                        # Get input dir from previous similarity stage
                        motifs_dir = context.get_previous_stage_output_dir('similarity')
                        if not motifs_dir:
                            motifs_dir = "similarity/motifs"  # Fallback
                        
                        opt_dir = context.refinement_stage_dir if context.refinement_stage_dir else "refinement"
                        opt_result['input_dir'] = motifs_dir
                        opt_result['working_dir'] = opt_dir
                        opt_result['output_dir'] = opt_dir
                        
                        if max_critical is not None:
                            opt_result['critical_threshold'] = max_critical
                        if max_skipped is not None:
                            opt_result['skipped_threshold'] = max_skipped
                        
                        # Add motifs source info if available
                        if hasattr(context, 'refinement_motifs_source') and context.refinement_motifs_source:
                            opt_result['motifs_source'] = context.refinement_motifs_source
                        
                        # Add completion counts if available
                        if hasattr(context, 'refinement_completed') and context.refinement_completed is not None:
                            opt_result['completed'] = context.refinement_completed
                        if hasattr(context, 'refinement_total') and context.refinement_total is not None:
                            opt_result['total'] = context.refinement_total
                        
                        # Add similarity folder info if available
                        if hasattr(context, 'refinement_sim_folder') and context.refinement_sim_folder:
                            opt_result['similarity_folder'] = context.refinement_sim_folder
                        
                        update_protocol_cache(opt_key, 'completed', result=opt_result, cache_file=cache_file)
                        
                        sim_result = {}
                        
                        # Store directories for stage memory
                        sim_dir = context.similarity_dir if context.similarity_dir else "similarity_2"
                        opt_dir = context.refinement_stage_dir if context.refinement_stage_dir else "refinement"
                        sim_result['input_dir'] = opt_dir  # Read from refinement
                        sim_result['working_dir'] = sim_dir
                        # After optimization: use "umotifs" prefix (unique motifs, second level)
                        sim_result['output_dir'] = os.path.join(sim_dir, "umotifs")  # Unique motifs
                        
                        if final_critical is not None:
                            sim_result['critical_pct'] = final_critical
                        if final_skipped is not None:
                            sim_result['skipped_pct'] = final_skipped
                        
                        # Extract critical and skipped counts from similarity output
                        critical_count, skipped_count = parse_similarity_output(sim_dir)
                        sim_result['critical_count'] = critical_count
                        sim_result['skipped_count'] = skipped_count
                        
                        # Extract threshold value from similarity command args
                        sim_stage = stages[stage_idx + 1] if stage_idx + 1 < len(stages) else {}
                        sim_args = sim_stage.get('args', [])
                        for arg in sim_args:
                            if arg.startswith('--th=') or arg.startswith('--threshold='):
                                threshold_val = float(arg.split('=')[1])
                                sim_result['threshold'] = threshold_val
                            elif arg.startswith('--rmsd='):
                                rmsd_val = float(arg.split('=')[1])
                                sim_result['rmsd_threshold'] = rmsd_val
                        
                        # Add similarity folder and motifs info if available
                        if hasattr(context, 'sim_folder'):
                            sim_result['similarity_folder'] = context.sim_folder
                        if hasattr(context, 'sim_motifs_created'):
                            sim_result['motifs_created'] = context.sim_motifs_created
                        ref_sim_stage_num = stage_num + 1
                        input_cnt_r = getattr(context, 'similarity_stage_input_counts', {}).get(ref_sim_stage_num)
                        if input_cnt_r:
                            sim_result['input_count'] = input_cnt_r

                        # Add initial validation values (from first attempt)
                        if initial_critical is not None:
                            sim_result['initial_critical'] = initial_critical
                        if initial_skipped is not None:
                            sim_result['initial_skipped'] = initial_skipped
                        if initial_critical_count is not None:
                            sim_result['initial_critical_count'] = initial_critical_count
                        if initial_skipped_count is not None:
                            sim_result['initial_skipped_count'] = initial_skipped_count
                        
                        # Add threshold info and attempts
                        sim_result['attempts'] = final_attempt
                        if max_critical is not None:
                            sim_result['threshold_type'] = 'critical'
                            sim_result['threshold_value'] = max_critical
                            sim_result['threshold_met'] = (final_critical is not None and final_critical <= max_critical)
                        elif max_skipped is not None:
                            sim_result['threshold_type'] = 'skipped'
                            sim_result['threshold_value'] = max_skipped
                            sim_result['threshold_met'] = (final_skipped is not None and final_skipped <= max_skipped)
                        
                        update_protocol_cache(sim_key, 'completed', result=sim_result, cache_file=cache_file)
                    
                    # Check if workflow should pause after refinement stage
                    if not check_workflow_pause(stage, stage_num, len(stages), cache_file, use_cache):
                        return 0  # Paused after refinement
                    
                    # Check if workflow should pause after similarity stage (check next stage for pause marker)
                    if stage_idx + 1 < len(stages):
                        sim_stage = stages[stage_idx + 1]
                        if not check_workflow_pause(sim_stage, stage_num + 1, len(stages), cache_file, use_cache):
                            return 0  # Paused after similarity
                    
                    # Skip the similarity stage (already executed)
                    stage_idx += 2
                    completed_stage_count = stage_num + 1
                    context.completed_stage_count = completed_stage_count
                    continue
                else:
                    # No similarity after refinement - just run it once
                    result = execute_refinement_stage(context, stage)
                
                # Build and save refinement result if successful (for standalone refinement)
                if result == 0 and use_cache and not next_is_similarity:
                    opt_key = f"refinement_{stage_num}"
                    
                    # Extract optimization parameters from stage
                    max_skipped = None
                    for arg in stage.get('args', []):
                        if arg.startswith('--skipped='):
                            max_skipped = float(arg.split('=')[1])
                    
                    from datetime import datetime as dt_now  # Local import to avoid scope issues
                    opt_result = {
                        'status': 'completed',
                        'end_time': dt_now.now().isoformat()
                    }
                    
                    # Add threshold info
                    if max_skipped is not None:
                        opt_result['skipped_threshold'] = max_skipped
                    
                    # Add detailed execution data from context
                    if hasattr(context, 'refinement_motifs_source') and context.refinement_motifs_source:
                        opt_result['motifs_source'] = context.refinement_motifs_source
                    
                    if hasattr(context, 'refinement_completed') and context.refinement_completed is not None:
                        opt_result['completed'] = context.refinement_completed
                    if hasattr(context, 'refinement_total') and context.refinement_total is not None:
                        opt_result['total'] = context.refinement_total
                    
                    if hasattr(context, 'refinement_sim_folder') and context.refinement_sim_folder:
                        opt_result['similarity_folder'] = context.refinement_sim_folder
                    
                    # Save to cache
                    update_protocol_cache(opt_key, 'completed', 
                                        result=opt_result, cache_file=cache_file)
                
                # Check if workflow should pause after this stage
                if result == 0:
                    if not check_workflow_pause(stage, stage_num, len(stages), cache_file, use_cache):
                        return 0  # Paused successfully
                
                stage_idx += 1
                completed_stage_count = stage_num
                context.completed_stage_count = completed_stage_count
                
                if result != 0:
                    print(f"\nError: Stage {stage_num} ({stage_type}) failed with code {result}")
                    return result
            else:
                print(f"Error: Unknown stage type '{stage_type}'")
                return 1
                
        except Exception as e:
            print(f"\nError executing stage {stage_num} ({stage_type}): {e}")
            traceback.print_exc()
            return 1
    
    if context.workflow_verbose_level >= 1:
        print(f"\n{'-' * 60}")
        print("✓ Workflow completed")
        print(f"{'-' * 60}")
    else:
        render_final_workflow_summary()

    # Export final ensemble at root directory from the last similarity stage.
    copy_final_ensemble_to_root()
    
    # Clean up temporary folders from retry attempts (final safety cleanup)
    temp_calc_folders = glob.glob("calculation_tmp_*")
    temp_sim_folders = glob.glob("similarity_tmp_*")
    retry_input = ["retry_input"] if os.path.exists("retry_input") else []
    all_temp = temp_calc_folders + temp_sim_folders + retry_input
    
    if all_temp:
        if context.workflow_verbose_level >= 1:
            print("\nCleaning up temporary folders...")
        for folder in all_temp:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                if context.workflow_verbose_level >= 1:
                    print(f"  Removed: {folder}")
        if context.workflow_verbose_level >= 1:
            print(f"  Cleaned {len(all_temp)} temporary folder(s)")
    
    # If using cache (protocol mode), generate summary
    # NOTE: Cache is NOT deleted to allow protocol resume
    if use_cache:
        generate_protocol_summary(cache_file=cache_file)
        if context.workflow_verbose_level >= 1:
            print(f"\n→ Protocol cache saved: {cache_file}")
    
    return 0

def execute_replication_stage(context: WorkflowContext, stage: Dict[str, Any]) -> int:
    """Execute replication stage (r3, r5, etc.) and run annealing simulations."""
    num_replicas = stage['num_replicas']
    context.num_replicas = num_replicas
    verbose = getattr(context, 'workflow_verbose_level', 0) >= 1
    
    # Parse --box flags from stage args
    # Note: --retry is removed; launch failures are now auto-retried 10 times
    box_size_override = None
    args = stage.get('args', [])
    for arg in args:
        if arg.startswith('--box'):
            # Extract packing percentage from flag (e.g., --box10 -> 10%)
            try:
                packing_str = arg.replace('--box', '')
                if packing_str:
                    packing_percent = float(packing_str)
                    # Get box size recommendation for this packing percentage
                    recommended_box = get_box_size_recommendation(context.input_file, packing_percent)
                    if recommended_box is not None:
                        box_size_override = recommended_box
                        # Store for protocol summary
                        context.annealing_box_size = box_size_override
                        context.annealing_packing = packing_percent
                        if verbose:
                            print(f"Using recommended box size: {box_size_override:.1f} Å ({packing_percent}% effective packing)")
            except ValueError:
                pass
    
    # Clean up old annealing folder from previous runs to avoid duplicate replicas
    # (Only in workflow mode - standalone mode keeps it for reference)
    annealing_folder = os.path.join(os.path.dirname(context.input_file), "annealing")
    if os.path.exists(annealing_folder):
        try:
            shutil.rmtree(annealing_folder)
            if verbose:
                print(f"Cleaned up previous annealing folder")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not clean old annealing folder: {e}")
    
    # Create replicated runs (without launcher script in workflow mode)
    replicated_files = create_replicated_runs(
        context.input_file,
        num_replicas,
        create_launcher=False,
        box_size=box_size_override,
        verbose=verbose
    )
    
    if not replicated_files:
        print("✗ Failed to create replicated runs")
        return 1
    
    context.annealing_dirs = [os.path.dirname(f) for f in replicated_files]
    
    # Actually run the annealing simulations
    if verbose:
        print(f"Running {num_replicas} annealing simulation(s)")
    
    failed_runs = []
    progress_cb = getattr(context, 'update_progress', None)
    if not verbose and callable(progress_cb):
        progress_cb(f"0/{num_replicas} ...")
    completed_replicas = 0

    def _replica_already_completed(replica_dir: str, replica_input_name: str) -> bool:
        """Return True when a replica has a completed annealing result set."""
        output_file = os.path.join(replica_dir, os.path.splitext(replica_input_name)[0] + '.out')
        has_result = bool(glob.glob(os.path.join(replica_dir, 'result_*.xyz')))
        has_tvse = bool(glob.glob(os.path.join(replica_dir, 'tvse_*.dat')))
        if not (has_result and has_tvse):
            return False
        if not os.path.exists(output_file):
            return False
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            return ('Normal annealing termination' in content) or ('Annealing simulation finished' in content)
        except OSError:
            return False

    for i, input_file in enumerate(replicated_files, 1):
        run_dir = os.path.dirname(input_file)
        run_name = os.path.basename(run_dir)
        input_basename = os.path.basename(input_file)

        # Keep workflow idempotent: if a replica already completed, do not re-run it.
        if _replica_already_completed(run_dir, input_basename):
            completed_replicas += 1
            if verbose:
                print(f"\n  {run_name}... ✓ (already completed)")
            elif callable(progress_cb):
                progress_cb(f"{completed_replicas}/{num_replicas} ...")
            continue
        
        if verbose:
            print(f"\n  {run_name}...", end=" ", flush=True)
        
        success = False
        last_error = None
        # Single execution per replica (no retries): each replica should generate one seed/run.
        try:
            # Run as subprocess in the run directory using the current interpreter.
            cmd = [sys.executable, os.path.abspath(sys.argv[0]), input_basename]

            # Discard child stdio to avoid extra per-replica log files in annealing folders.
            result = subprocess.run(
                cmd,
                cwd=run_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env={**os.environ, "ASCEC_DISABLE_EMBEDDED_PROTOCOL": "1"}
            )

            output_file = os.path.join(run_dir, os.path.splitext(input_basename)[0] + '.out')
            artifacts_exist = bool(glob.glob(os.path.join(run_dir, 'result_*.xyz'))) and bool(
                glob.glob(os.path.join(run_dir, 'tvse_*.dat'))
            )

            # Success criteria: normal termination marker OR expected annealing artifacts.
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    content = f.read()
                    if 'Normal annealing termination' in content or 'Annealing simulation finished' in content:
                        success = True

            if not success and artifacts_exist:
                success = True

            if not success and result.returncode != 0:
                last_error = f"Exit code {result.returncode}"
            elif not success and not os.path.exists(output_file):
                last_error = "Output file not created"
            elif not success:
                last_error = "Annealing finished without normal termination marker"

        except Exception as e:
            last_error = str(e)
        
        if success:
            completed_replicas += 1
            if verbose:
                print("✓")
            elif callable(progress_cb):
                progress_cb(f"{completed_replicas}/{num_replicas} ...")
        else:
            # Check if result files were created even without proper termination message
            result_files = []
            for pattern in ['result_*.xyz', 'result_*.mol']:
                result_files.extend(glob.glob(os.path.join(run_dir, pattern)))
            
            if result_files:
                # Files were created, consider it a success
                if verbose:
                    print(f"✓ (output files created)")
                success = True
                completed_replicas += 1
                if not verbose and callable(progress_cb):
                    progress_cb(f"{completed_replicas}/{num_replicas} ...")
            else:
                if verbose:
                    print(f"✗ (no output files)")
                # Check output file for errors
                output_file = os.path.join(run_dir, input_basename.replace('.in', '.out'))
                if os.path.exists(output_file):
                    try:
                        with open(output_file, 'r') as f:
                            content = f.read()
                            # Look for traceback or error messages
                            if 'Traceback' in content:
                                lines = content.split('\n')
                                # Find traceback and show it
                                for i, line in enumerate(lines):
                                    if 'Traceback' in line:
                                        # Show traceback and a few lines after
                                        error_lines = lines[i:min(i+10, len(lines))]
                                        if verbose:
                                            print(f"    Traceback found in {run_name}/{input_basename.replace('.in', '.out')}:")
                                            for eline in error_lines:
                                                if eline.strip():
                                                    print(f"      {eline}")
                                        break
                            elif last_error:
                                if verbose:
                                    print(f"    {last_error}")
                    except:
                        if last_error:
                            if verbose:
                                print(f"    {last_error}")
                if last_error:
                    if verbose:
                        print(f"    {last_error}")
                failed_runs.append(run_name)
    
    if failed_runs:
        print(f"\n✗ {len(failed_runs)} simulation(s) failed")
        return 1  # Fail the stage if any runs didn't complete
    
    # Generate annealing diagrams for each replica
    if MATPLOTLIB_AVAILABLE:
        # Only show messages if not in workflow mode
        is_workflow = getattr(context, 'is_workflow', False)
        
        if not is_workflow:
            print(f"\nGenerating annealing diagrams...")
        
        diagrams_generated = 0
        all_tvse_files = []
        
        for annealing_dir in context.annealing_dirs:
            # Find tvse_*.dat file in this directory
            tvse_files = glob.glob(os.path.join(annealing_dir, 'tvse_*.dat'))
            if tvse_files:
                for tvse_file in tvse_files:
                    # Always add to list for combined diagram, regardless of individual diagram success
                    all_tvse_files.append(tvse_file)
                    if plot_annealing_diagrams(tvse_file, annealing_dir):
                        diagrams_generated += 1
        
        if diagrams_generated > 0 and not is_workflow:
            print(f"  Generated {diagrams_generated} diagram(s)")
            
        # Generate combined replica diagram in parent annealing directory
        if len(all_tvse_files) > 1:
            # Get parent directory (annealing/)
            annealing_parent = os.path.dirname(context.annealing_dirs[0])
            if not annealing_parent:
                annealing_parent = "annealing"
            
            combined_diagram = os.path.join(annealing_parent, f"tvse_r{num_replicas}.png")
            if plot_combined_replicas_diagram(all_tvse_files, combined_diagram, num_replicas):
                if not is_workflow:
                    print(f"  Generated combined: {os.path.basename(combined_diagram)}")
    
    return 0

def find_out_file_in_subdirs(base_dir: str, basename: str):
    """Find .out file in subdirectories, checking shortened basename variants."""
    # Check root first
    root_file = os.path.join(base_dir, f"{basename}.out")
    if os.path.exists(root_file):
        return root_file
    
    # Build list of subdirectories to check, prioritizing exact and shortened names
    subdirs_to_check = [basename]
    
    # Add shortened basename variants
    if '_opt' in basename:
        subdirs_to_check.insert(0, basename.replace('_opt', ''))
    elif '_calc' in basename:
        subdirs_to_check.insert(0, basename.replace('_calc', ''))
    
    # Add common subdirectory names
    subdirs_to_check.extend(['orca_out', 'gaussian_out', 'completed', 'failed', 'skipped'])
    
    # Check subdirectories
    for subdir in subdirs_to_check:
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.exists(subdir_path):
            out_file = os.path.join(subdir_path, f"{basename}.out")
            if os.path.exists(out_file):
                return out_file
    
    # Last resort: search all subdirectories recursively
    for root, dirs, files in os.walk(base_dir):
        # Skip deep nesting (max 2 levels)
        if root.count(os.sep) - base_dir.count(os.sep) > 2:
            continue
        if f"{basename}.out" in files:
            return os.path.join(root, f"{basename}.out")
    
    return None

def process_redo_structures(context: WorkflowContext, stage_dir: str, template_file: str) -> bool:
    """
    Process structures that need recalculation (from similarity stage).
    Regenerates input files using new geometries and deletes old output files.
    
    Args:
        context: Workflow context
        stage_dir: Directory of the current stage (calculation or optimization)
        template_file: Path to the template input file
        
    Returns:
        bool: True if any structures were processed, False otherwise
    """
    # Determine similarity directory (check context or default)
    sim_dir = getattr(context, 'similarity_dir', None)
    workflow_concise = getattr(context, 'is_workflow', False) and getattr(context, 'workflow_verbose_level', 0) < 1

    def _redo_log(*args, **kwargs):
        if not workflow_concise:
            print(*args, **kwargs)

    if not sim_dir:
        # For optimization stages, check for similarity_2, similarity_3, etc.
        # For optimization stages, use similarity
        if 'optimization' in stage_dir.lower():
            # Check for similarity_2 first (most common for first optimization)
            if os.path.exists('similarity_2'):
                sim_dir = 'similarity_2'
            elif os.path.exists('similarity_3'):
                sim_dir = 'similarity_3'
            elif os.path.exists('similarity_4'):
                sim_dir = 'similarity_4'
            else:
                # Fallback to similarity if none found
                sim_dir = 'similarity'
        else:
            # Calculation stage - use similarity
            sim_dir = 'similarity'
    
    if not os.path.exists(sim_dir):
        return False
        
    need_recalc_dir = os.path.join(sim_dir, "skipped_structures", "need_recalculation")
    critical_non_conv_dir = os.path.join(sim_dir, "skipped_structures", "critical_non_converged")
    
    # Check if either directory exists
    has_need_recalc = os.path.exists(need_recalc_dir)
    has_critical_nc = os.path.exists(critical_non_conv_dir)
    
    if not has_need_recalc and not has_critical_nc:
        return False
        
    # Collect XYZ files from both directories
    xyz_files = []
    if has_need_recalc:
        xyz_files.extend(glob.glob(os.path.join(need_recalc_dir, "*.xyz")))
    if has_critical_nc:
        xyz_files.extend(glob.glob(os.path.join(critical_non_conv_dir, "*.xyz")))
    
    if not xyz_files:
        return False
    
    # Get basenames and filter out combined files BEFORE printing count
    need_recalc_basenames = [os.path.splitext(os.path.basename(f))[0] for f in xyz_files]
    filtered_basenames = []
    for b in need_recalc_basenames:
        low = b.lower()
        if low.startswith('combined') or 'combined_' in low or low == 'combined':
            continue
        filtered_basenames.append(b)
    need_recalc_basenames = filtered_basenames
    
    if not need_recalc_basenames:
        return False
    
    # Determine display directory
    display_dir = os.path.join(sim_dir, "skipped_structures")
    _redo_log(f"\nProcessing redo structures from: {display_dir}")
    _redo_log(f"\nFound {len(need_recalc_basenames)} structure(s) to retry")

    processed_count = 0
    processed_basenames = []
    
    # Get template content
    if template_file and os.path.exists(template_file):
        with open(template_file, 'r') as f:
            template_content = f.read()
    else:
        _redo_log(f"  Warning: Template file {template_file} not found. Cannot regenerate inputs.")
        return False

    # Find launcher path for ORCA calculations (load early for version detection)
    launcher_path = None
    launcher_content = None
    possible_launchers = [
        os.path.join(stage_dir, 'launcher_orca.sh'),
        os.path.join(stage_dir, 'launcher_ascec.sh'),
        'launcher_orca.sh',
        'launcher_ascec.sh',
    ]
    for lp in possible_launchers:
        if os.path.exists(lp):
            launcher_path = lp
            try:
                with open(lp, 'r') as f:
                    launcher_content = f.read()
            except Exception:
                pass
            break

    # Parse rescue method from template (for structures with 2+ imaginary frequencies)
    rescue_method, rescue_use_numfreq = parse_rescue_method(template_content, launcher_content=launcher_content)
    
    # Extract charge and multiplicity from template
    charge_val = 0
    mult_val = 1
    xyz_match = re.search(r'\*\s*(?:xyz|xyzfile)\s+(-?\d+)\s+(\d+)', template_content, re.IGNORECASE)
    if xyz_match:
        charge_val = int(xyz_match.group(1))
        mult_val = int(xyz_match.group(2))
    
    # Extract nprocs from template (ORCA or Gaussian format)
    nprocs_val = 1  # default to 1 if not found
    # ORCA format: nprocs N
    nprocs_match = re.search(r'nprocs\s+(\d+)', template_content, re.IGNORECASE)
    if nprocs_match:
        nprocs_val = int(nprocs_match.group(1))
    else:
        # Gaussian format: %NProcShared=N
        nprocs_match = re.search(r'%NProcShared\s*=\s*(\d+)', template_content, re.IGNORECASE)
        if nprocs_match:
            nprocs_val = int(nprocs_match.group(1))
    
    # Track structures needing rescue Hessian (2+ imaginary freqs)
    rescue_hessian_tasks = []  # List of (basename, xyz_file, imag_count)

    for basename in need_recalc_basenames:
        # Find xyz_file in either need_recalculation or critical_non_converged
        xyz_file: str = ""  # Will be set below
        if has_need_recalc and os.path.exists(os.path.join(need_recalc_dir, f"{basename}.xyz")):
            xyz_file = os.path.join(need_recalc_dir, f"{basename}.xyz")
        elif has_critical_nc and os.path.exists(os.path.join(critical_non_conv_dir, f"{basename}.xyz")):
            xyz_file = os.path.join(critical_non_conv_dir, f"{basename}.xyz")
        
        # Skip if no xyz_file found (shouldn't happen since basenames come from xyz files)
        if not xyz_file:
            _redo_log(f"    {basename}: XYZ file not found, skipping")
            continue
        
        # Check if this structure came from critical_non_converged (non-converged optimization)
        is_critical_non_converged = critical_non_conv_dir in xyz_file

        
        # Possible subdirectory names where .out files may be located
        # Include both standard directories AND the basename itself (where ORCA/Gaussian creates output)
        subdirs_to_check = [basename, 'orca_out', 'gaussian_out', 'completed', 'failed', 'skipped']
        
        # Also add shortened versions of basename (e.g., "motif_01" for "motif_01_opt")
        # ORCA often creates subdirectories without the suffix
        if '_opt' in basename:
            subdirs_to_check.insert(0, basename.replace('_opt', ''))
        elif '_calc' in basename:
            subdirs_to_check.insert(0, basename.replace('_calc', ''))
        
        # Find .out file for this structure
        out_file = None
        
        # Search in critical_non_converged first (if applicable)
        if has_critical_nc and os.path.exists(os.path.join(critical_non_conv_dir, f"{basename}.out")):
            out_file = os.path.join(critical_non_conv_dir, f"{basename}.out")
        # Search in need_recalc_dir
        elif has_need_recalc and os.path.exists(os.path.join(need_recalc_dir, f"{basename}.out")):
            out_file = os.path.join(need_recalc_dir, f"{basename}.out")
        # Search in root
        elif os.path.exists(os.path.join(stage_dir, f"{basename}.out")):
            out_file = os.path.join(stage_dir, f"{basename}.out")
        elif os.path.exists(os.path.join(stage_dir, f"{basename}.out.backup")):
            out_file = os.path.join(stage_dir, f"{basename}.out.backup")
        else:
            # Search in subdirectories
            for subdir in subdirs_to_check:
                subdir_path = os.path.join(stage_dir, subdir)
                if os.path.exists(os.path.join(subdir_path, f"{basename}.out")):
                    out_file = os.path.join(subdir_path, f"{basename}.out")
                    break
                elif os.path.exists(os.path.join(subdir_path, f"{basename}.out.backup")):
                    out_file = os.path.join(subdir_path, f"{basename}.out.backup")
                    break
            
            # If still not found, check similarity directory (where files are moved after stage completion)
            if not out_file and sim_dir and os.path.exists(sim_dir):
                # Check for orca_out_*/gaussian_out_*/calc_out_* directories in similarity folder
                for item in os.listdir(sim_dir):
                    if item.startswith('orca_out_') or item.startswith('gaussian_out_') or item.startswith('calc_out_'):
                        out_dir = os.path.join(sim_dir, item)
                        if os.path.isdir(out_dir):
                            if os.path.exists(os.path.join(out_dir, f"{basename}.out")):
                                out_file = os.path.join(out_dir, f"{basename}.out")
                                break

        
        # Determine new geometry based on imaginary frequencies
        new_geometry = None
        if out_file and os.path.exists(out_file):
            imag_count = count_imaginary_frequencies(out_file)
            
            if imag_count == 1:
                # Single imaginary: displace along mode
                xyz_lines = displace_along_imaginary_mode(out_file, os.path.dirname(out_file))
                if xyz_lines:
                    try:
                        natoms = int(xyz_lines[0].strip())
                        if len(xyz_lines) >= natoms + 2:
                            new_geometry = xyz_lines[2:]  # Skip first 2 lines
                            _redo_log(f"    {basename}: 1 imaginary freq, displacing along mode ✓")
                        else:
                            _redo_log(f"    {basename}: 1 imaginary freq, displacing along mode ✗ (malformed XYZ)")
                    except (ValueError, IndexError):
                        _redo_log(f"    {basename}: 1 imaginary freq, displacing along mode ✗ (invalid XYZ format)")
                else:
                    _redo_log(f"    {basename}: 1 imaginary freq, displacing along mode ✗ (displacement failed)")
            
            elif imag_count >= 2:
                # Multiple imaginary: displace along the highest (most negative) imaginary mode
                xyz_lines = displace_along_imaginary_mode(out_file, os.path.dirname(out_file), use_highest_mode=True)
                if xyz_lines:
                    try:
                        natoms = int(xyz_lines[0].strip())
                        if len(xyz_lines) >= natoms + 2:
                            new_geometry = xyz_lines[2:]  # Skip first 2 lines
                            _redo_log(f"    {basename}: {imag_count} imaginary freqs, displacing along highest mode ✓")
                        else:
                            _redo_log(f"    {basename}: {imag_count} imaginary freqs, displacing along highest mode ✗ (malformed XYZ)")
                    except (ValueError, IndexError):
                        _redo_log(f"    {basename}: {imag_count} imaginary freqs, displacing along highest mode ✗ (invalid XYZ format)")
                else:
                    _redo_log(f"    {basename}: {imag_count} imaginary freqs, displacing along highest mode ✗ (displacement failed)")
            else:
                # No imaginary frequencies - check if non-converged (max iterations reached)
                conv_status = detect_convergence_status(out_file)
                
                if conv_status['status'] == 'not_converged' and rescue_method and launcher_path:
                    # Non-converged structure - use final geometry with rescue Hessian
                    xyz_lines = extract_final_geometry(out_file, os.path.dirname(out_file))
                    if xyz_lines:
                        new_geometry = xyz_lines[2:]  # Skip first 2 lines
                        
                        # Save geometry to temporary XYZ for rescue calculation
                        rescue_xyz_path = os.path.join(stage_dir, f"{basename}_rescue_geom.xyz")
                        try:
                            with open(rescue_xyz_path, 'w') as f:
                                natoms = len([line for line in new_geometry if line.strip()])
                                f.write(f"{natoms}\n")
                                f.write(f"{basename} geometry for rescue Hessian\n")
                                for line in new_geometry:
                                    if line.strip():
                                        f.write(line if line.endswith('\n') else line + '\n')
                            
                            # Run rescue Hessian calculation
                            hess_file = run_rescue_hessian_calculation(
                                rescue_xyz_path, rescue_method, launcher_path,
                                charge=charge_val, multiplicity=mult_val, nprocs=nprocs_val,
                                output_basename=basename,  # Use original basename
                                use_numfreq=rescue_use_numfreq
                            )
                            
                            if hess_file and os.path.exists(hess_file):
                                rescue_hessian_tasks.append((basename, hess_file))
                                _redo_log(f"    {basename}: non-converged (max iter), rescue Hessian computed ✓")
                            else:
                                _redo_log(f"    {basename}: non-converged (max iter), using final geometry (rescue failed)")
                            
                            # Cleanup temporary XYZ
                            if os.path.exists(rescue_xyz_path):
                                os.remove(rescue_xyz_path)
                        except Exception as e:
                            _redo_log(f"    {basename}: non-converged (max iter), using final geometry (error: {e})")
                    else:
                        # Fallback to similarity XYZ
                        _redo_log(f"    {basename}: non-converged (max iter), using similarity XYZ", end='')
                        if os.path.exists(xyz_file):
                            with open(xyz_file, 'r') as f:
                                new_geometry = f.readlines()[2:]
                            _redo_log(f" ✓")
                        else:
                            _redo_log(f" ✗ (extraction failed)")
                elif conv_status['status'] == 'not_converged':
                    # Non-converged but no rescue method - use final geometry
                    xyz_lines = extract_final_geometry(out_file, os.path.dirname(out_file))
                    if xyz_lines:
                        new_geometry = xyz_lines[2:]
                        _redo_log(f"    {basename}: non-converged (max iter), using final geometry ✓")
                    else:
                        _redo_log(f"    {basename}: non-converged (max iter), using similarity XYZ", end='')
                        if os.path.exists(xyz_file):
                            with open(xyz_file, 'r') as f:
                                new_geometry = f.readlines()[2:]
                            _redo_log(f" ✓")
                        else:
                            _redo_log(f" ✗")
                else:
                    # Converged with no imaginary freqs - use similarity XYZ
                    _redo_log(f"    {basename}: No imaginary freqs, using similarity XYZ", end='')
                    if os.path.exists(xyz_file):
                        with open(xyz_file, 'r') as f:
                            new_geometry = f.readlines()[2:]
                        _redo_log(f" ✓")
                    else:
                        _redo_log(f" ✗ (similarity XYZ not found)")
        else:
            # No .out file found - check if from critical_non_converged (needs rescue hessian)
            if is_critical_non_converged and xyz_file and os.path.exists(xyz_file):
                # Structure from critical_non_converged - use XYZ with rescue Hessian
                with open(xyz_file, 'r') as f:
                    new_geometry = f.readlines()[2:]
                
                if rescue_method and launcher_path:
                    # Run rescue Hessian calculation
                    rescue_xyz_path = os.path.join(stage_dir, f"{basename}_rescue_geom.xyz")
                    try:
                        with open(rescue_xyz_path, 'w') as f:
                            natoms = len([line for line in new_geometry if line.strip()])
                            f.write(f"{natoms}\n")
                            f.write(f"{basename} geometry for rescue Hessian\n")
                            for line in new_geometry:
                                if line.strip():
                                    f.write(line if line.endswith('\n') else line + '\n')
                        
                        hess_file = run_rescue_hessian_calculation(
                            rescue_xyz_path, rescue_method, launcher_path,
                            charge=charge_val, multiplicity=mult_val, nprocs=nprocs_val,
                            output_basename=basename,
                            use_numfreq=rescue_use_numfreq
                        )
                        
                        if hess_file and os.path.exists(hess_file):
                            rescue_hessian_tasks.append((basename, hess_file))
                            _redo_log(f"    {basename}: critical non-converged, rescue Hessian computed ✓")
                        else:
                            _redo_log(f"    {basename}: critical non-converged, using XYZ geometry (rescue failed)")
                        
                        if os.path.exists(rescue_xyz_path):
                            os.remove(rescue_xyz_path)
                    except Exception as e:
                        _redo_log(f"    {basename}: critical non-converged, using XYZ geometry (error: {e})")
                else:
                    _redo_log(f"    {basename}: critical non-converged, using XYZ geometry (no rescue method)")
            elif xyz_file and os.path.exists(xyz_file):
                # Regular case - just use XYZ from similarity
                _redo_log(f"    {basename}: No .out file, using similarity XYZ", end='')
                with open(xyz_file, 'r') as f:
                    new_geometry = f.readlines()[2:]
                _redo_log(f" ✓")
            else:
                _redo_log(f"    {basename}: No .out file, similarity XYZ not found ✗")
        
        # Regenerate input file with new geometry
        if new_geometry:
            # Determine QM program and input extension
            template_lower = template_file.lower().strip()
            if template_lower.endswith('.inp'):
                qm_program = 'orca'
                input_ext = '.inp'
            else:
                qm_program = 'gaussian'
                input_ext = '.com' if template_lower.endswith('.com') else '.gjf'
            
            # Define input path
            input_path = os.path.join(stage_dir, f"{basename}{input_ext}")
            
            # Check if original input file exists - if so, update it; otherwise create new
            if os.path.exists(input_path):
                # Original file exists - just update coordinates
                try:
                    with open(input_path, 'r') as f:
                        original_lines = f.readlines()
                    
                    # Find and replace the coordinate block
                    updated_lines = []
                    in_coords = False
                    coords_written = False
                    
                    for line in original_lines:
                        # Detect start of coordinate block (after "* xyz" or similar)
                        if not in_coords and ('* xyz' in line.lower() or '* xyzfile' in line.lower()):
                            updated_lines.append(line)
                            in_coords = True
                            continue
                        
                        # Write new coordinates once we're in the block
                        if in_coords and not coords_written:
                            # Write all new geometry lines
                            for geom_line in new_geometry:
                                updated_lines.append(geom_line)
                            coords_written = True
                            # Skip old coordinates until we hit the end marker
                            if line.strip() == '*':
                                updated_lines.append(line)
                                in_coords = False
                            continue
                        
                        # Skip old coordinate lines
                        if in_coords:
                            if line.strip() == '*':
                                updated_lines.append(line)
                                in_coords = False
                            continue
                        
                        # Keep all other lines as-is
                        updated_lines.append(line)
                    
                    # Write updated file
                    with open(input_path, 'w') as f:
                        f.writelines(updated_lines)
                    
                    processed_count += 1
                    processed_basenames.append(basename)
                except Exception as e:
                    _redo_log(f"\n    Warning: Could not update {basename}: {e}")
            else:
                # Original file doesn't exist - create new from template
                # Parse XYZ lines into atom fields (symbol, x, y, z)
                atoms_list = []
                for line in new_geometry:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        # Convert coordinates to float
                        symbol = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        atoms_list.append([symbol, x, y, z])
                
                config_data = {
                    'atoms': atoms_list,
                    'comment': f"{basename}"
                }
                
                if create_qm_input_file(config_data, template_content, input_path, qm_program):
                    processed_count += 1
                    processed_basenames.append(basename)
    
    if processed_count > 0:
        _redo_log(f"\n  Regenerated {processed_count} input file(s) with new geometries")
        
        # Enable Hessian restart for structures with rescue Hessians computed
        if rescue_hessian_tasks:
            _redo_log(f"  Enabling Hessian restart for {len(rescue_hessian_tasks)} structure(s)")
            for task_basename, hess_file in rescue_hessian_tasks:
                # Find the input file path
                template_lower = template_file.lower().strip()
                if template_lower.endswith('.inp'):
                    input_ext = '.inp'
                else:
                    input_ext = '.com' if template_lower.endswith('.com') else '.gjf'
                
                # Determine where input file will run from (subdirectory takes precedence)
                subdir_path = os.path.join(stage_dir, task_basename)
                if os.path.isdir(subdir_path):
                    # Calculation runs from subdirectory
                    input_path = os.path.join(subdir_path, f"{task_basename}{input_ext}")
                    hess_dest = os.path.join(subdir_path, os.path.basename(hess_file))
                else:
                    # Calculation runs from stage_dir root
                    input_path = os.path.join(stage_dir, f"{task_basename}{input_ext}")
                    hess_dest = os.path.join(stage_dir, os.path.basename(hess_file))
                
                if os.path.exists(input_path):
                    # Copy Hessian file to same directory as input file
                    same_file = os.path.exists(hess_dest) and os.path.samefile(hess_file, hess_dest)
                    if not same_file and os.path.exists(hess_file):
                        shutil.copy2(hess_file, hess_dest)
                    
                    if enable_hessian_restart(input_path, hess_dest):
                        _redo_log(f"    {task_basename}: Hessian restart enabled ✓")
                    else:
                        _redo_log(f"    {task_basename}: Hessian restart failed ✗")
        
        # Store recalculated basenames in context for similarity cache update
        context.recalculated_files = processed_basenames
        
        # RENAME (not delete) output files for structures needing recalculation
        # This preserves the old .out file in case the new calculation fails
        # We can then extract geometry from the backup on next redo attempt
        _redo_log(f"  Backing up old output files for structures needing recalculation")
        for basename in processed_basenames:
            # Build list of subdirectories to search, including shortened basename variants
            subdirs_to_search = ['orca_out', 'gaussian_out', 'completed', basename]
            
            # Add shortened versions for common patterns (e.g., "motif_01" for "motif_01_opt")
            if '_opt' in basename:
                subdirs_to_search.insert(0, basename.replace('_opt', ''))
            elif '_calc' in basename:
                subdirs_to_search.insert(0, basename.replace('_calc', ''))
            
            # Check in subdirectories
            search_dirs_for_out = [stage_dir] + [os.path.join(stage_dir, sd) for sd in subdirs_to_search]
            
            out_file_found = False
            for search_dir in search_dirs_for_out:
                if not os.path.exists(search_dir):
                    continue
                
                out_file = os.path.join(search_dir, f"{basename}.out")
                if os.path.exists(out_file):
                    # Only create backup if one doesn't already exist (keep first backup only)
                    backup_file = out_file + ".backup"
                    if not os.path.exists(backup_file):
                        shutil.move(out_file, backup_file)
                    else:
                        # Backup exists, just remove the current .out file
                        os.remove(out_file)
                    out_file_found = True
                    break # Only rename the first .out file found
            
            # Also remove auxiliary files from all possible locations
            # Use the broader search_dirs list for auxiliary files
            all_possible_search_dirs = [stage_dir] + [os.path.join(stage_dir, sd) for sd in subdirs_to_search + ['failed', 'skipped']]
            for search_dir in all_possible_search_dirs:
                if not os.path.exists(search_dir):
                    continue
                for ext in ['.gbw', '.prop', '.densities', '.tmp', '_property.txt', '.engrad']:
                    aux_file = os.path.join(search_dir, f"{basename}{ext}")
                    if os.path.exists(aux_file):
                        try:
                            os.remove(aux_file)
                        except Exception:
                            pass
    
    return processed_count > 0
def process_optimization_redo(context: WorkflowContext, stage_dir: str, template_file: str) -> bool:
    """
    Specialized redo processing for optimization stage.
    Handles the specific directory structure of optimization/similarity interactions.
    """
    # 1. Determine Similarity Directory
    # For optimization, we look for the similarity folder that THIS optimization feeds into.
    # Usually similarity_2, similarity_3, etc.
    # Use refinement_sim_folder (the dedicated variable for optimization outputs)
    sim_dir = getattr(context, 'refinement_sim_folder', None)
    workflow_concise = getattr(context, 'is_workflow', False) and getattr(context, 'workflow_verbose_level', 0) < 1

    def _redo_log(*args, **kwargs):
        if not workflow_concise:
            print(*args, **kwargs)
    
    if not sim_dir:
        # Try to guess based on existence
        if os.path.exists('similarity_2'):
            sim_dir = 'similarity_2'
        elif os.path.exists('similarity_3'):
            sim_dir = 'similarity_3'
        elif os.path.exists('similarity_4'):
            sim_dir = 'similarity_4'
        else:
            return False
    
    # CRITICAL: If sim_dir includes orca_out_X subdirectory, strip it
    # organize step sets context.refinement_sim_folder to "similarity_2/orca_out_5_1"
    # but skipped_structures is at "similarity_2/skipped_structures/"
    if '/' in sim_dir and ('orca_out_' in sim_dir or 'gaussian_out_' in sim_dir):
        sim_dir = sim_dir.split('/')[0]
    
    # 2. Locate need_recalculation directory and optionally clustered_with_minima
    need_recalc_dir = os.path.join(sim_dir, "skipped_structures", "need_recalculation")
    clustered_dir = os.path.join(sim_dir, "skipped_structures", "clustered_with_minima")
    critical_non_conv_dir = os.path.join(sim_dir, "skipped_structures", "critical_non_converged")
    
    # Check threshold mode from context
    # --critical: only use need_recalculation (structures with imaginary freqs)
    # --skipped: use both need_recalculation AND clustered_with_minima
    use_skipped_threshold = getattr(context, 'use_skipped_threshold', False)
    
    # 3. Find XYZ files to process
    xyz_files = []
    
    # Always include critical structures (need_recalculation) - these have imaginary frequencies
    if os.path.exists(need_recalc_dir):
        need_recalc_files = glob.glob(os.path.join(need_recalc_dir, "*.xyz"))
        # Filter out combined files
        need_recalc_files = [f for f in need_recalc_files if "combined" not in os.path.basename(f).lower()]
        xyz_files.extend(need_recalc_files)
    
    # Always include critical non-converged structures (these need rescue hessian)
    if os.path.exists(critical_non_conv_dir):
        critical_nc_files = glob.glob(os.path.join(critical_non_conv_dir, "*.xyz"))
        critical_nc_files = [f for f in critical_nc_files if "combined" not in os.path.basename(f).lower()]
        xyz_files.extend(critical_nc_files)
    
    # Only include clustered_with_minima if --skipped threshold is used
    # These are structures that clustered with existing minima (skipped but not critical)
    if use_skipped_threshold and os.path.exists(clustered_dir):
        clustered_files = glob.glob(os.path.join(clustered_dir, "*.xyz"))
        clustered_files = [f for f in clustered_files if "combined" not in os.path.basename(f).lower()]
        xyz_files.extend(clustered_files)
    
    if not xyz_files:
        return False
    
    # Sort files naturally (motif_01, motif_02, ...)
    xyz_files = sorted(xyz_files, key=lambda x: natural_sort_key(os.path.basename(x)))
        
    # Determine which directory to display based on what we found
    display_dir = need_recalc_dir if os.path.exists(need_recalc_dir) else clustered_dir
    if os.path.exists(need_recalc_dir) and os.path.exists(clustered_dir):
        display_dir = os.path.dirname(need_recalc_dir)  # Show parent "skipped_structures"
    if os.path.exists(critical_non_conv_dir):
        display_dir = os.path.dirname(need_recalc_dir)  # Show parent "skipped_structures"
    
    _redo_log(f"\nProcessing redo structures from: {display_dir}")
    _redo_log(f"Found {len(xyz_files)} structure(s) to retry")
    
    # 4. Process each file
    processed_count = 0
    processed_basenames = []
    
    # Read template
    if not os.path.exists(template_file):
        print(f"Error: Template file {template_file} not found")
        return False
        
    with open(template_file, 'r') as f:
        template_content = f.read()
    
    # Find launcher path for ORCA calculations (load early for version detection)
    launcher_path = None
    launcher_content = None
    possible_launchers = [
        os.path.join(stage_dir, 'launcher_orca.sh'),
        os.path.join(stage_dir, 'launcher_ascec.sh'),
        'launcher_orca.sh',
        'launcher_ascec.sh',
    ]
    for lp in possible_launchers:
        if os.path.exists(lp):
            launcher_path = lp
            try:
                with open(lp, 'r') as f:
                    launcher_content = f.read()
            except Exception:
                pass
            break

    # Parse rescue method from template (for structures with 2+ imaginary frequencies)
    rescue_method, rescue_use_numfreq = parse_rescue_method(template_content, launcher_content=launcher_content)
    
    # Extract charge and multiplicity from template
    charge_val = 0
    mult_val = 1
    xyz_match = re.search(r'\*\s*(?:xyz|xyzfile)\s+(-?\d+)\s+(\d+)', template_content, re.IGNORECASE)
    if xyz_match:
        charge_val = int(xyz_match.group(1))
        mult_val = int(xyz_match.group(2))
    
    # Extract nprocs from template (ORCA or Gaussian format)
    nprocs_val = 1  # default to 1 if not found
    # ORCA format: nprocs N
    nprocs_match = re.search(r'nprocs\s+(\d+)', template_content, re.IGNORECASE)
    if nprocs_match:
        nprocs_val = int(nprocs_match.group(1))
    else:
        # Gaussian format: %NProcShared=N
        nprocs_match = re.search(r'%NProcShared\s*=\s*(\d+)', template_content, re.IGNORECASE)
        if nprocs_match:
            nprocs_val = int(nprocs_match.group(1))
    
    # Track structures needing rescue Hessian (2+ imaginary freqs)
    rescue_hessian_tasks = []  # List of (basename, hess_file)
        
    for xyz_file in xyz_files:
        basename = os.path.splitext(os.path.basename(xyz_file))[0]
        
        # Check if this structure came from critical_non_converged (non-converged optimization)
        is_critical_non_converged = os.path.exists(critical_non_conv_dir) and critical_non_conv_dir in xyz_file
        
        # 4a. Find the previous output file (to check imaginary freqs and extract geometry)
        # Optimization outputs are tricky. They could be in:
        # - optimization/umotif_XX_opt.out (root)
        # - optimization/umotif_XX_opt/umotif_XX_opt.out (subdir)
        # - similarity_2/orca_out_X/umotif_XX_opt.out (moved there)
        # - need_recalc_dir/umotif_XX_opt.out (copied there by similarity script)
        
        out_file = None
        
        # Priority 1: Check optimization directory (subdirs) - THIS IS THE REAL FILE WE NEED TO BACKUP
        # Check exact basename subdir
        check_path = os.path.join(stage_dir, basename, f"{basename}.out")
        if os.path.exists(check_path):
            out_file = check_path
        else:
            # Check subdir (motif_01 for motif_01_opt)
            short_name = basename.replace('_opt', '').replace('_calc', '')
            check_path = os.path.join(stage_dir, short_name, f"{basename}.out")
            if os.path.exists(check_path):
                out_file = check_path
                    
        # Priority 2: Check optimization root
        if not out_file:
            check_path = os.path.join(stage_dir, f"{basename}.out")
            if os.path.exists(check_path):
                out_file = check_path
        
        # Priority 3: Check similarity output folders (if not found in optimization yet)
        if not out_file and os.path.exists(sim_dir):
            for item in os.listdir(sim_dir):
                if item.startswith('orca_out_') or item.startswith('gaussian_out_') or item.startswith('calc_out_'):
                    check_path = os.path.join(sim_dir, item, f"{basename}.out")
                    if os.path.exists(check_path):
                        out_file = check_path
                        break
        
        # Priority 4: Check need_recalc_dir or critical_non_converged (similarity script copy - ONLY for reading geometry)
        if not out_file:
            check_path = os.path.join(need_recalc_dir, f"{basename}.out")
            if os.path.exists(check_path):
                out_file = check_path
            elif os.path.exists(critical_non_conv_dir):
                check_path = os.path.join(critical_non_conv_dir, f"{basename}.out")
                if os.path.exists(check_path):
                    out_file = check_path
            
        # 4b. Determine new geometry
        new_geometry = None
        
        if out_file:
            imag_count = count_imaginary_frequencies(out_file)
            
            if imag_count == 1:
                # Displace
                xyz_lines = displace_along_imaginary_mode(out_file, os.path.dirname(out_file))
                if xyz_lines and len(xyz_lines) > 2:
                    new_geometry = xyz_lines[2:]
                    _redo_log(f"    {basename}: 1 imaginary freq, displacing along mode \u2713")
            elif imag_count >= 2:
                # Multiple imaginary: displace along the highest (most negative) imaginary mode
                xyz_lines = displace_along_imaginary_mode(out_file, os.path.dirname(out_file), use_highest_mode=True)
                if xyz_lines and len(xyz_lines) > 2:
                    new_geometry = xyz_lines[2:]
                    _redo_log(f"    {basename}: {imag_count} imaginary freqs, displacing along highest mode ✓")
                else:
                    _redo_log(f"    {basename}: {imag_count} imaginary freqs, displacing along highest mode ✗")
            else:
                # No imaginary frequencies - check if non-converged (max iterations reached)
                conv_status = detect_convergence_status(out_file)
                
                if conv_status['status'] == 'not_converged' and rescue_method and launcher_path:
                    # Non-converged structure - use final geometry with rescue Hessian
                    xyz_lines = extract_final_geometry(out_file, os.path.dirname(out_file))
                    if xyz_lines and len(xyz_lines) > 2:
                        new_geometry = xyz_lines[2:]
                        
                        # Save geometry to temporary XYZ for rescue calculation
                        rescue_xyz_path = os.path.join(stage_dir, f"{basename}_rescue_geom.xyz")
                        try:
                            with open(rescue_xyz_path, 'w') as f:
                                natoms = len([line for line in new_geometry if line.strip()])
                                f.write(f"{natoms}\n")
                                f.write(f"{basename} geometry for rescue Hessian\n")
                                for line in new_geometry:
                                    if line.strip():
                                        f.write(line if line.endswith('\n') else line + '\n')
                            
                            # Run rescue Hessian calculation
                            hess_file = run_rescue_hessian_calculation(
                                rescue_xyz_path, rescue_method, launcher_path,
                                charge=charge_val, multiplicity=mult_val, nprocs=nprocs_val,
                                output_basename=basename,  # Use original basename
                                use_numfreq=rescue_use_numfreq
                            )
                            
                            if hess_file and os.path.exists(hess_file):
                                rescue_hessian_tasks.append((basename, hess_file))
                                _redo_log(f"    {basename}: non-converged (max iter), rescue Hessian computed ✓")
                            else:
                                _redo_log(f"    {basename}: non-converged (max iter), using final geometry (rescue failed)")
                            
                            # Cleanup temporary XYZ
                            if os.path.exists(rescue_xyz_path):
                                os.remove(rescue_xyz_path)
                        except Exception as e:
                            _redo_log(f"    {basename}: non-converged (max iter), using final geometry (error: {e})")
                    else:
                        _redo_log(f"    {basename}: non-converged (max iter), using similarity XYZ", end='')
                elif conv_status['status'] == 'not_converged':
                    # Non-converged but no rescue method - use final geometry
                    xyz_lines = extract_final_geometry(out_file, os.path.dirname(out_file))
                    if xyz_lines and len(xyz_lines) > 2:
                        new_geometry = xyz_lines[2:]
                        _redo_log(f"    {basename}: non-converged (max iter), using final geometry ✓")
                    else:
                        _redo_log(f"    {basename}: non-converged (max iter), using similarity XYZ", end='')
                else:
                    # Converged with no imaginary freqs - use similarity XYZ
                    _redo_log(f"    {basename}: No imaginary freqs, using similarity XYZ", end='')
        else:
            # No .out file found - check if from critical_non_converged (needs rescue hessian)
            if is_critical_non_converged and os.path.exists(xyz_file):
                # Structure from critical_non_converged - use XYZ with rescue Hessian
                with open(xyz_file, 'r') as f:
                    new_geometry = f.readlines()[2:]
                
                if rescue_method and launcher_path:
                    # Run rescue Hessian calculation
                    rescue_xyz_path = os.path.join(stage_dir, f"{basename}_rescue_geom.xyz")
                    try:
                        with open(rescue_xyz_path, 'w') as f:
                            natoms = len([line for line in new_geometry if line.strip()])
                            f.write(f"{natoms}\n")
                            f.write(f"{basename} geometry for rescue Hessian\n")
                            for line in new_geometry:
                                if line.strip():
                                    f.write(line if line.endswith('\n') else line + '\n')
                        
                        hess_file = run_rescue_hessian_calculation(
                            rescue_xyz_path, rescue_method, launcher_path,
                            charge=charge_val, multiplicity=mult_val, nprocs=nprocs_val,
                            output_basename=basename,
                            use_numfreq=rescue_use_numfreq
                        )
                        
                        if hess_file and os.path.exists(hess_file):
                            rescue_hessian_tasks.append((basename, hess_file))
                            _redo_log(f"    {basename}: critical non-converged, rescue Hessian computed ✓")
                        else:
                            _redo_log(f"    {basename}: critical non-converged, using XYZ geometry (rescue failed)")
                        
                        if os.path.exists(rescue_xyz_path):
                            os.remove(rescue_xyz_path)
                    except Exception as e:
                        _redo_log(f"    {basename}: critical non-converged, using XYZ geometry (error: {e})")
                else:
                    _redo_log(f"    {basename}: critical non-converged, using XYZ geometry (no rescue method)")
            else:
                # Regular case - just use XYZ from similarity
                _redo_log(f"    {basename}: No .out file, using similarity XYZ", end='')
        
        # Fallback to XYZ file
        if not new_geometry:
            with open(xyz_file, 'r') as f:
                new_geometry = f.readlines()[2:]
            imag_count = 0  # Initialize
            if out_file is None or imag_count == 0:
                _redo_log(f" \u2713")
                
        # 4c. Generate Input File
        if new_geometry:
            # Determine extension
            template_lower = template_file.lower().strip()
            if template_lower.endswith('.inp'):
                qm_program = 'orca'
                input_ext = '.inp'
            else:
                qm_program = 'gaussian'
                input_ext = '.com'
            
            # Always create redo input files at root level
            # The running loop expects flat files and will move them after sort
            input_path = os.path.join(stage_dir, f"{basename}{input_ext}")
            
            # Parse atoms
            atoms_list = []
            for line in new_geometry:
                parts = line.strip().split()
                if len(parts) >= 4:
                    symbol = parts[0]
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        atoms_list.append([symbol, x, y, z])
                    except ValueError:
                        pass
            
            config_data = {
                'atoms': atoms_list,
                'comment': f"{basename} generated by ASCEC redo"
            }
            
            if create_qm_input_file(config_data, template_content, input_path, qm_program):
                processed_count += 1
                processed_basenames.append(basename)
                
                # 4d. Backup old output file (CRITICAL for optimization to re-run)
                # We must move the old .out file so the workflow sees it as "not done"
                # Only keep first backup - don't create incrementing backups
                if out_file and os.path.exists(out_file):
                    backup_file = out_file + ".backup"
                    if not os.path.exists(backup_file):
                        shutil.move(out_file, backup_file)
                    else:
                        # Backup exists, just remove the current .out file
                        os.remove(out_file)
                    
    if processed_count > 0:
        _redo_log(f"\n  Regenerated {processed_count} input file(s) with new geometries")
        
        # Enable Hessian restart for structures with rescue Hessians computed
        if rescue_hessian_tasks:
            _redo_log(f"  Enabling Hessian restart for {len(rescue_hessian_tasks)} structure(s)")
            for task_basename, hess_file in rescue_hessian_tasks:
                # Find the input file path
                template_lower = template_file.lower().strip()
                if template_lower.endswith('.inp'):
                    input_ext = '.inp'
                else:
                    input_ext = '.com' if template_lower.endswith('.com') else '.gjf'
                
                # Determine where input file will run from (subdirectory takes precedence)
                subdir_path = os.path.join(stage_dir, task_basename)
                if os.path.isdir(subdir_path):
                    # Calculation runs from subdirectory
                    input_path = os.path.join(subdir_path, f"{task_basename}{input_ext}")
                    hess_dest = os.path.join(subdir_path, os.path.basename(hess_file))
                else:
                    # Calculation runs from stage_dir root
                    input_path = os.path.join(stage_dir, f"{task_basename}{input_ext}")
                    hess_dest = os.path.join(stage_dir, os.path.basename(hess_file))
                
                if os.path.exists(input_path):
                    # Copy Hessian file to same directory as input file
                    same_file = os.path.exists(hess_dest) and os.path.samefile(hess_file, hess_dest)
                    if not same_file and os.path.exists(hess_file):
                        shutil.copy2(hess_file, hess_dest)
                    
                    if enable_hessian_restart(input_path, hess_dest):
                        _redo_log(f"    {task_basename}: Hessian restart enabled ✓")
                    else:
                        _redo_log(f"    {task_basename}: Hessian restart failed ✗")
        
        # Sort for natural ordering (motif_01, motif_02, etc.)
        context.recalculated_files = sorted(processed_basenames, key=natural_sort_key)
    
    return processed_count > 0

def execute_optimization_stage(context: WorkflowContext, stage: Dict[str, Any]) -> int:
    """Execute optimization stage with automatic retry logic."""
    # Store context globally for access in helper functions
    sys._current_workflow_context = context  # type: ignore[attr-defined]
    
    args = stage['args']
    workflow_concise = getattr(context, 'is_workflow', False) and getattr(context, 'workflow_verbose_level', 0) < 1
    
    # Parse flags - defaults for when flags are not explicitly provided
    max_critical = 0      # default: 0% critical structures allowed (strict)
    max_skipped = None    # Not set by default (only --critical is used unless --skipped specified)
    max_stage_redos = 3   # default: 3 stage redos (--redo: redo entire optimization+similarity)
    # Note: --retry removed; launch failures auto-retry up to 10 times (hardcoded)
    auto_select = 'combined'  # Workflow mode defaults to combining files (like -c flag)
    template_file = None
    launcher_file = None
    unknown_template_token = None
    
    i = 0
    while i < len(args):
        arg = args[i]
        
        if arg.startswith('--critical='):
            max_critical = float(arg.split('=')[1])
        elif arg.startswith('--skipped='):
            max_skipped = float(arg.split('=')[1])
        elif arg.startswith('--redo='):
            max_stage_redos = int(arg.split('=')[1])
        elif arg.startswith('--auto-select='):
            auto_select = arg.split('=')[1]
        elif arg == '-a':
            # Auto-select all result_*.xyz files (excludes combined files)
            # Example: If you have result_443189.xyz, result_130389.xyz, result_536046.xyz
            # All 3 files will be processed separately
            auto_select = 'all'
        elif arg == '-c':
            # Auto-combine all result_*.xyz files first, then process the combined file
            # Example: If you have 3 result files, they'll be combined into combined_r3.xyz
            # Then only combined_r3.xyz will be processed
            auto_select = 'combined'
        elif arg.endswith(('.inp', '.com', '.gjf')) and template_file is None:
            template_file = arg
        elif arg.endswith('.sh') and launcher_file is None:
            launcher_file = arg
        elif not arg.startswith('-') and unknown_template_token is None:
            # Keep first unrecognized positional token for clearer diagnostics.
            unknown_template_token = arg
        
        i += 1
    
    # Allow template labels from embedded blocks (e.g., "opt input1").
    if not template_file and unknown_template_token:
        resolved_template = resolve_template_reference(context, unknown_template_token)
        if resolved_template:
            template_file = resolved_template

    if not template_file:
        if unknown_template_token:
            print(
                f"Error: No template file specified for optimization stage. "
                f"Got '{unknown_template_token}', expected .inp/.com/.gjf or an embedded template label"
            )
        else:
            print("Error: No template file specified for optimization stage (.inp/.com/.gjf or embedded label)")
        return 1
    
    context.max_tries = max_stage_redos  # For compatibility with existing code

    # Pin optimization output to a deterministic similarity base folder for this cycle.
    # This prevents redo/resume runs from drifting to similarity_3, similarity_4, etc.
    optimization_cycle = getattr(context, 'optimization_stage_number', 1)
    if optimization_cycle <= 1:
        fixed_opt_sim_base = "similarity"
    else:
        fixed_opt_sim_base = f"similarity_{optimization_cycle}"

    context.similarity_dir = fixed_opt_sim_base
    context.pending_similarity_folder = fixed_opt_sim_base
    
    # Process redo structures at the START of the stage (if need_recalculation exists)
    # This ensures that when the workflow restarts this stage after a similarity failure,
    # we immediately regenerate inputs and delete old outputs before checking completion
    optimization_dir_path = getattr(context, 'optimization_stage_dir', 'optimization')
    if not optimization_dir_path:  # Handle empty string
        optimization_dir_path = 'optimization'
    if os.path.exists(optimization_dir_path):
        redo_result = process_redo_structures(context, optimization_dir_path, template_file)
    
    
    # Check if we're resuming and optimization directory already exists
    cache_file = getattr(context, 'cache_file', 'protocol_cache.pkl')
    cache = load_protocol_cache(cache_file) if os.path.exists(cache_file) else {}
    
    # Get the current stage key from context (e.g., "optimization_2" for first opt at position 2)
    stage_key = getattr(context, 'current_stage_key', '')
    
    # Get completed calculations from cache
    completed_calcs = cache.get('stages', {}).get(stage_key, {}).get('result', {}).get('completed_files', [])
    opt_dir_exists = os.path.exists("optimization")
    
    # If resuming (cache exists + optimization stage was started before), reuse existing directory  
    # This works even if no runs completed yet (e.g., interrupted during first opt)
    stage_was_started = stage_key in cache.get('stages', {})
    
    # Determine if this is a fresh start or resume
    optimization_dir_path = getattr(context, 'optimization_stage_dir', None)
    if opt_dir_exists and stage_was_started:
        # Use absolute path from context if available, otherwise use relative path
        if not optimization_dir_path:
            optimization_dir_path = "optimization"
        
        # Check if redo structures exist (files scheduled for recalculation)
        redo_files = set()
        if hasattr(context, 'recalculated_files') and context.recalculated_files:
            redo_files = set(context.recalculated_files)
        
        # Scan ALL subdirectories for completed calculations (check for .out files, NOT .out.backup)
        # Files with only .out.backup are being redone and should NOT be counted as completed
        actual_completed = []
        
        # 1. Check optimization directory subfolders
        if os.path.exists(optimization_dir_path):
            for item in os.listdir(optimization_dir_path):
                item_path = os.path.join(optimization_dir_path, item)
                if os.path.isdir(item_path) and not item.startswith('_'):  # Skip _run_ directories
                    # Skip if this file is marked for redo
                    # Check both "item" AND "item_opt"/"item_calc" against redo_files
                    redo_variants = [item, f"{item}_opt", f"{item}_calc"]
                    if any(variant in redo_files for variant in redo_variants):
                        continue
                        
                    # Check if this subdirectory has a completed calculation (.out file, not just .backup)
                    out_file = os.path.join(item_path, f"{item}.out")
                    if os.path.exists(out_file):
                        # Verify completion (OPI-aware for ORCA 6.1+)
                        if check_orca_terminated_normally_opi(out_file):
                            actual_completed.append(item)
            
            # 1b. Check for flat files in optimization directory (e.g. opt_conf_1.out)
            # This handles the case where calculations are not in subdirectories
            for item in os.listdir(optimization_dir_path):
                if item.endswith('.out') or item.endswith('.log'):
                    basename = os.path.splitext(item)[0]
                    # Skip if already found in subdir or marked for redo
                    if basename not in actual_completed and basename not in redo_files:
                        out_file = os.path.join(optimization_dir_path, item)
                        # Verify completion (OPI-aware for ORCA 6.1+)
                        if item.endswith('.out'):
                            if check_orca_terminated_normally_opi(out_file):
                                actual_completed.append(basename)
                        else:  # Gaussian .log files
                            try:
                                with open(out_file, 'r', encoding='utf-8', errors='replace') as f:
                                    content = f.read()
                                    if 'Normal termination' in content:
                                        actual_completed.append(basename)
                            except Exception:
                                pass
        
        # 2. Check similarity/orca_out_* folders (files moved there after sorting)
        # This is CRITICAL for showing correct counts (e.g. 11/11) when resuming
        sim_dir = getattr(context, 'similarity_dir', 'similarity')
        if os.path.exists(sim_dir):
            for item in os.listdir(sim_dir):
                if item.startswith('orca_out_') or item.startswith('gaussian_out_') or item.startswith('calc_out_'):
                    out_dir = os.path.join(sim_dir, item)
                    if os.path.isdir(out_dir):
                        files_in_subdir = os.listdir(out_dir)
                        for f in files_in_subdir:
                            if f.endswith('.out') or f.endswith('.log'):
                                basename = os.path.splitext(f)[0]
                                if basename not in actual_completed and basename not in redo_files:
                                    # Verify completion (OPI-aware for ORCA 6.1+)
                                    out_file = os.path.join(out_dir, f)
                                    if f.endswith('.out'):
                                        if check_orca_terminated_normally_opi(out_file):
                                            actual_completed.append(basename)
                                    else:  # Gaussian .log files
                                        try:
                                            with open(out_file, 'r', encoding='utf-8', errors='replace') as f_obj:
                                                content = f_obj.read()
                                                if 'Normal termination' in content:
                                                    actual_completed.append(basename)
                                        except:
                                            pass
        
        # Also check for similarity_2, similarity_3, etc.
        # This is needed if we have multiple similarity stages
        parent_dir = os.getcwd()
        for item in os.listdir(parent_dir):
            if item.startswith('similarity_') and os.path.isdir(item):
                sim_dir_n = item
                for subitem in os.listdir(sim_dir_n):
                    if subitem.startswith('orca_out_') or subitem.startswith('gaussian_out_') or subitem.startswith('calc_out_'):
                        out_dir = os.path.join(sim_dir_n, subitem)
                        if os.path.isdir(out_dir):
                            for f in os.listdir(out_dir):
                                if f.endswith('.out') or f.endswith('.log'):
                                    basename = os.path.splitext(f)[0]
                                    if basename not in actual_completed and basename not in redo_files:
                                        actual_completed.append(basename)
        
        # Update completed_calcs to match reality
        completed_calcs = actual_completed
        
        # CRITICAL: If we found NO completed files but input_files is very small,
        # this might be a redo scenario where all other files are already done
        # Count ALL subdirectories in calculation/ as potential completed files
        if len(completed_calcs) == 0 and opt_dir_exists:
            for item in os.listdir(optimization_dir_path):
                item_path = os.path.join(optimization_dir_path, item)
                if os.path.isdir(item_path) and not item.startswith('_'):
                    # Even if no .out file, if the subdirectory exists, a run was attempted
                    # Check for .out.backup (means it's being redone)
                    # Check both "item" AND "item_opt"/"item_calc" against redo_files
                    redo_variants = [item, f"{item}_opt", f"{item}_calc"]
                    is_redo = any(variant in redo_files for variant in redo_variants)
                    if not is_redo and item not in actual_completed:
                        # Count any subdirectory as a potential completed run for total count
                        backup_file = os.path.join(item_path, f"{item}.out.backup")
                        # Only count if it has either .out or .out.backup
                        out_file = os.path.join(item_path, f"{item}.out")
                        if os.path.exists(backup_file) or os.path.exists(out_file):
                            actual_completed.append(item)
            completed_calcs = actual_completed

        
        if not workflow_concise:
            if completed_calcs:
                print(f"\nResuming: Using existing optimization directory ({len(completed_calcs)} files already completed)\n")
            else:
                print(f"\nResuming: Using existing optimization directory\n")
        
        # Clean up ORCA intermediate files at root level (they should not be there)
        excluded_patterns = ['.scfgrad.', '.scfp.', '.tmp.', '.densities.', '.scfhess.']
        for item in os.listdir(optimization_dir_path):
            item_path = os.path.join(optimization_dir_path, item)
            if os.path.isfile(item_path) and any(pattern in item for pattern in excluded_patterns):
                try:
                    os.remove(item_path)
                except:
                    pass
        
        # Clean up any old input files at the root level (they're now in subdirectories after calculations run)
        for old_inp in glob.glob(os.path.join(optimization_dir_path, "*.inp")) + glob.glob(os.path.join(optimization_dir_path, "*.com")):
            # Skip ORCA intermediate files (already handled above)
            if any(pattern in old_inp for pattern in excluded_patterns):
                continue
                
            # Only remove if a corresponding subdirectory exists (calculation was run)
            basename = os.path.basename(old_inp).replace('.inp', '').replace('.com', '')
            
            # CRITICAL: Do NOT remove if this file was just regenerated by redo logic!
            if basename in redo_files:
                continue
                
            subdir = os.path.join(optimization_dir_path, basename)
            if os.path.isdir(subdir):
                os.remove(old_inp)
    else:
        # Handle launcher: if not provided, auto-detect ORCA and create launcher
        if launcher_file:
            if not os.path.exists(launcher_file):
                print(f"Warning: Launcher script not found: {launcher_file}")
                print("  Attempting to auto-detect ORCA installation...")
                launcher_file = None
        
        # Track existing optimization directories before running calculate_input_files
        existing_opt_dirs = set(d for d in os.listdir('.') if d.startswith('optimization') and os.path.isdir(d))
        
        # Get QM alias from context (read from input.in line 9)
        qm_alias = getattr(context, 'qm_alias', 'orca')
        
        # Run calculation system creation with auto_select (in workflow mode)
        status = calculate_input_files(
            template_file,
            launcher_file,
            auto_select=auto_select,
            stage_type="optimization",
            workflow_mode=True,
            qm_alias=qm_alias,
        )
        # Check if calculate_input_files succeeded (returns string message)
        # Successfully created files return: "Created optimization system in '...' with X input files..."
        # Errors return: "Error: ..."
        if isinstance(status, str) and status.startswith("Error"):
            print(f"\n{status}") # Print the error message directly
            return 1  # Return error code
        # Don't print result message in workflow mode - output is already shown
        
        # Find the optimization directory that was just created (may be optimization, optimization_2, etc.)
        current_opt_dirs = set(d for d in os.listdir('.') if d.startswith('optimization') and os.path.isdir(d))
        new_opt_dirs = current_opt_dirs - existing_opt_dirs
        if new_opt_dirs:
            # Use the newly created directory
            optimization_dir_path = sorted(new_opt_dirs)[-1]  # Get the highest numbered one
        elif "optimization" in current_opt_dirs:
            optimization_dir_path = "optimization"
        else:
            optimization_dir_path = None
        
    
    if optimization_dir_path and os.path.exists(optimization_dir_path):
        context.optimization_stage_dir = optimization_dir_path
        
        # Find and execute the launcher script
        launcher_script = os.path.join(optimization_dir_path, "launcher_orca.sh")
        if not os.path.exists(launcher_script):
            launcher_script = os.path.join(optimization_dir_path, "launcher_gaussian.sh")
        if not os.path.exists(launcher_script):
            launcher_script = os.path.join(optimization_dir_path, "launcher.sh")
        
        if os.path.exists(launcher_script):
            if not workflow_concise:
                print(f"\nExecuting calculations...\n")
            
            # Helper function to filter out ORCA intermediate files and rescue inputs
            def is_valid_input_file(filename):
                """Check if file is a valid input file (not an ORCA intermediate or rescue file)"""
                if not filename.endswith(('.inp', '.com', '.gjf')):
                    return False
                # Exclude ORCA intermediate files and rescue inputs (already processed)
                excluded_patterns = ['.scfgrad.', '.scfp.', '.tmp.', '.densities.', '.scfhess.', '_rescue.']
                return not any(pattern in filename for pattern in excluded_patterns)
            
            # Get list of input files to process
            # Check if files are at root level or in subdirectories (after sort command)
            input_files = sorted([f for f in os.listdir(optimization_dir_path) if is_valid_input_file(f)], key=natural_sort_key)
            
            if not input_files:
                # No input files at root - check if they're in subdirectories (sorted)
                # Look for subdirectories with input files
                for item in os.listdir(optimization_dir_path):
                    item_path = os.path.join(optimization_dir_path, item)
                    if os.path.isdir(item_path):
                        subdir_files = [f for f in os.listdir(item_path) if is_valid_input_file(f)]
                        if subdir_files:
                            # Found input files in subdirectory - add them with relative path
                            for f in subdir_files:
                                input_files.append(os.path.join(item, f))
                input_files = sorted(input_files, key=natural_sort_key)
            
            # Determine QM program from input files (check first file)
            if input_files:
                first_file = os.path.basename(input_files[0])
                qm_program = 'orca' if first_file.endswith('.inp') else 'gaussian'
            else:
                qm_program = 'orca'  # Default
            
            # Get exclusions from cache (already loaded earlier)
            # Use appropriate key based on which optimization stage this is
            optimization_stage_num = getattr(context, 'optimization_stage_number', 1)
            if optimization_stage_num == 1:
                excluded_numbers = cache.get('excluded_optimizations', [])
            else:
                excluded_numbers = cache.get('excluded_optimizations_2', [])
            
            # Apply exclusion filtering to completed_calcs and input_files
            completed_calcs = [f for f in completed_calcs if not match_exclusion(f, excluded_numbers)]
            
            # Count total inputs (including those already completed and cached)
            # This should be the TOTAL expected calculations, not just new ones
            all_input_basenames = set()
            
            # Add basenames from input files (pending calculations)
            for f in input_files:
                if not match_exclusion(f, excluded_numbers):
                    # Handle both simple filenames and paths like "opt_conf_1/opt_conf_1.inp"
                    basename = os.path.splitext(os.path.basename(f))[0]
                    all_input_basenames.add(basename)
            
            # Add basenames from already completed files (CRITICAL for correct count)
            for f in completed_calcs:
                # completed_calcs contains basenames, not full paths
                all_input_basenames.add(f)
            
            num_inputs = len(all_input_basenames)
            
            # Store initial completed count (before loop may modify completed_calcs)
            initial_completed_count = len(completed_calcs)

            progress_cb = context.update_progress
            if workflow_concise and callable(progress_cb):
                progress_cb(f"{initial_completed_count}/{num_inputs} ...")
            
            if completed_calcs and not workflow_concise:
                print(f"Resuming: {len(completed_calcs)}/{num_inputs} calculations already completed")
            
            if excluded_numbers and not workflow_concise:
                print(f"Exclusions active: {excluded_numbers}")
                print()
            
            # Read launcher script to get environment setup
            with open(launcher_script, 'r') as f:
                launcher_content = f.read()
            
            try:
                # Make script executable (for manual use later)
                os.chmod(launcher_script, 0o755)
                
                # Track completions: resumed count + newly completed
                num_completed = 0  # Track only newly completed in this run
                num_failed = 0
                failed_calculations = []  # Track failed input files
                # Hardcoded: retry launch failures up to 10 times
                max_launch_retries = 10
                # Time threshold for detecting launch failure (seconds)
                # If calculation exits within this time, it's considered a launch failure
                launch_failure_threshold = 5.0
                
                for idx, input_file in enumerate(input_files):
                    # Skip excluded calculations
                    if match_exclusion(input_file, excluded_numbers):
                        if not workflow_concise:
                            print(f"  Skipping: {input_file} (excluded)")
                        continue
                    
                    # Handle both root-level and subdirectory files
                    basename = os.path.splitext(input_file)[0]
                    output_file = basename + ('.out' if qm_program == 'orca' else '.log')
                    input_path = os.path.join(optimization_dir_path, input_file)
                    output_path = os.path.join(optimization_dir_path, output_file)
                    
                    # Skip if output already exists AND is successfully completed
                    # This handles redo scenarios where failed .out files were deleted
                    if os.path.exists(output_path):
                        try:
                            if qm_program == 'orca':
                                # Use OPI-aware check for ORCA 6.1+ support
                                is_complete = check_orca_terminated_normally_opi(output_path)
                            else:
                                with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
                                    output_content = f.read()
                                is_complete = 'Normal termination of Gaussian' in output_content
                            
                            if is_complete:
                                continue
                        except Exception:
                            # File is corrupted or unreadable, treat as incomplete
                            pass
                    
                    # For display, use just the filename
                    display_name = os.path.basename(input_file)
                    
                    progress_cb = context.update_progress
                    if workflow_concise and callable(progress_cb):
                        progress_cb(f"{initial_completed_count + num_completed}/{num_inputs} ...")
                    elif not workflow_concise:
                        print(f"  Running: {display_name}...", end='', flush=True)
                    
                    # ═══════════════════════════════════════════════════════════
                    # LAUNCH FAILURE RETRY STRATEGY
                    # ═══════════════════════════════════════════════════════════
                    # Only retry if the calculation fails to launch (instant crash).
                    # If the calculation starts normally (runs for a reasonable time),
                    # we don't retry regardless of exit code - let redo mode handle it.
                    # ═══════════════════════════════════════════════════════════
                    
                    success = False
                    launch_attempt = 0
                    calculation_started = False  # True once calculation runs past threshold
                    
                    while launch_attempt < max_launch_retries and not calculation_started:
                        launch_attempt += 1
                        
                        # Update display for retries
                        if launch_attempt > 1 and not workflow_concise:
                            print(f"\r  Running: {display_name}... ↻ (launch attempt {launch_attempt})\033[K", end='', flush=True)
                        
                        # ═══ STEP 1: CLEAN OUTPUT FILES ═══
                        
                        # Determine working directory
                        if '/' in input_file or '\\' in input_file:
                            # File is in subdirectory (e.g., "opt_conf_1/opt_conf_1.inp")
                            calc_subdir = os.path.dirname(input_file)
                            calc_working_dir = os.path.join(optimization_dir_path, calc_subdir)
                            input_file_relative = os.path.basename(input_file)
                            output_file_relative = os.path.basename(output_file)
                            script_basename = os.path.splitext(os.path.basename(input_file))[0]
                        else:
                            calc_working_dir = optimization_dir_path
                            input_file_relative = input_file
                            output_file_relative = output_file
                            script_basename = basename
                        
                        # Remove ALL auxiliary/output files for clean run (except input files)
                        for item in os.listdir(calc_working_dir):
                            if item.startswith(script_basename) and not item.endswith(('.inp', '.com', '.gjf')):
                                item_path = os.path.join(calc_working_dir, item)
                                if os.path.isfile(item_path):
                                    try:
                                        os.remove(item_path)
                                    except:
                                        pass
                        
                        # ═══ STEP 2: RUN CALCULATION ═══
                        
                        temp_script = os.path.join(calc_working_dir, f'_run_{script_basename}.sh')
                        with open(temp_script, 'w') as f:
                            f.write(launcher_content.split('###')[0])  # Environment setup
                            f.write("\n\n")
                            if qm_program == 'orca':
                                f.write(f"# Set unique scratch directory for this ORCA process\n")
                                f.write(f"export TMPDIR=\"$(pwd)/.orca_tmp_{script_basename}_$$\"\n")
                                f.write(f"mkdir -p \"$TMPDIR\"\n")
                                f.write(f"trap 'rm -rf \"$TMPDIR\"' EXIT\n\n")
                                orca_root_var = extract_orca_root_from_launcher(launcher_content)
                                if orca_root_var:
                                    f.write(f"${{{orca_root_var}}}/orca {input_file_relative} > {output_file_relative}\n")
                                else:
                                    f.write(f"orca {input_file_relative} > {output_file_relative}\n")
                            elif qm_program == 'gaussian':
                                f.write(f"$GAUSS_ROOT/g16 {input_file_relative}\n")
                        
                        os.chmod(temp_script, 0o755)
                        
                        script_name = os.path.basename(temp_script)
                        
                        # Measure execution time to detect launch failures
                        start_time = time.time()
                        result = subprocess.run(
                            ['bash', script_name],
                            cwd=calc_working_dir,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        elapsed_time = time.time() - start_time
                        
                        # If calculation ran past the threshold, it started normally
                        # Don't retry even if it failed - redo mode will handle it
                        if elapsed_time > launch_failure_threshold:
                            calculation_started = True
                        
                        # Cleanup temp script
                        if os.path.exists(temp_script):
                            os.remove(temp_script)
                        if qm_program == 'orca':
                            for tmp_dir in glob.glob(os.path.join(calc_working_dir, f'.orca_tmp_{script_basename}_*')):
                                try:
                                    shutil.rmtree(tmp_dir)
                                except:
                                    pass
                        
                        # ═══ STEP 3: CHECK SUCCESS ═══
                        
                        if os.path.exists(output_path):
                            try:
                                if qm_program == 'orca':
                                    normal_term = check_orca_terminated_normally_opi(output_path)
                                else:
                                    with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
                                        output_content = f.read()
                                    normal_term = 'Normal termination of Gaussian' in output_content
                                
                                if normal_term:
                                    # SUCCESS!
                                    num_completed += 1
                                    progress_cb = context.update_progress
                                    if workflow_concise and callable(progress_cb):
                                        progress_cb(f"{initial_completed_count + num_completed}/{num_inputs} ...")
                                    else:
                                        if launch_attempt > 1:
                                            print(f"\r  Running: {display_name}... ✓ (launch attempt {launch_attempt})\033[K")
                                        else:
                                            print(f"\r  Running: {display_name}... ✓\033[K")
                                    success = True
                                    
                                    # Update cache
                                    completed_calcs.append(input_file)
                                    basename_only = os.path.splitext(os.path.basename(input_file))[0]
                                    all_input_basenames.add(basename_only)
                                    stage_key = getattr(context, 'current_stage_key', 'calculation')
                                    update_protocol_cache(stage_key, 'in_progress',
                                                        result={'completed_files': completed_calcs,
                                                               'total_files': num_inputs,
                                                               'num_completed': num_completed},
                                                        cache_file=cache_file)
                                    break  # EXIT RETRY LOOP
                            except:
                                pass
                        
                        # If calculation started but failed, don't retry - exit loop
                        # Only continue loop if it was a launch failure
                        if calculation_started:
                            break
                    
                    # If loop exits without success
                    if not success:
                        if not workflow_concise:
                            if calculation_started:
                                # Calculation started but didn't complete normally - don't retry
                                # This will be handled by redo mode if needed
                                print(f"\r  Running: {display_name}... ✗ (no normal termination)\033[K")
                            elif launch_attempt >= max_launch_retries:
                                # Launch failed after all retries
                                print(f"\r  Running: {display_name}... ✗ (launch failed after {launch_attempt} attempts)\033[K")
                            else:
                                print(f"\r  Running: {display_name}... ✗\033[K")
                        num_failed += 1
                        failed_calculations.append(input_file)
                
                # Print status (not "results" - that's redundant)
                # Recalculate num_inputs from the updated set to reflect newly completed calculations
                num_inputs = len(all_input_basenames)
                # Total completed = initial completed + newly completed in this run
                total_completed = initial_completed_count + num_completed
                # Ensure we don't show more completed than total inputs (can happen if files are removed)
                if total_completed > num_inputs:
                    total_completed = num_inputs
                if not workflow_concise:
                    print(f"\nStatus: {total_completed}/{num_inputs} calculations completed")
                
                # Store for protocol summary (use total)
                context.optimization_completed = total_completed
                context.optimization_total = num_inputs
                
                # Handle failed calculations
                if failed_calculations:
                    if not workflow_concise:
                        print(f"Failed calculations: {len(failed_calculations)}/{num_inputs}")
                
                # Use total_completed for status checks (includes resumed + new)
                if total_completed == 0:
                    print("Error: No calculations completed successfully")
                    return 1
                # All calculations completed
                if total_completed == num_inputs:
                    if not workflow_concise:
                        print(f"All calculations completed successfully")
                elif total_completed < num_inputs:
                    # Continue - don't stop workflow, similarity will handle quality control
                    pass # Status line already printed above
                
                # Sort output files by energy (only in workflow mode)
                # Run organize step if any calculations completed (new or already done)
                if total_completed > 0 and context.is_workflow:
                    saved_cwd = os.getcwd()
                    try:
                        # Pin this stage to a deterministic similarity base folder so redo attempts
                        # and resumes do not create extra folders (similarity_3, similarity_4, ...).
                        parent_dir = os.path.dirname(os.getcwd())

                        target_sim_base = fixed_opt_sim_base

                        already_organized = os.path.exists(os.path.join(parent_dir, target_sim_base))
                        
                        if already_organized:
                            if not workflow_concise:
                                print("\n✓ Files already organized (resuming from cache)")
                            # Reuse this optimization cycle's fixed similarity folder.
                            context.optimization_sim_folder = target_sim_base
                            context.pending_similarity_folder = target_sim_base
                        else:
                            # Merge good structures back if they exist (from retry)
                            if os.path.exists("good_structures"):
                                for folder in os.listdir("good_structures"):
                                    src_folder = os.path.join("good_structures", folder)
                                    if os.path.isdir(src_folder):
                                        dest_folder = os.path.join(optimization_dir_path, folder)
                                        if not os.path.exists(dest_folder):
                                            shutil.copytree(src_folder, dest_folder)
                                # Clean up good_structures folder
                                shutil.rmtree("good_structures")
                            
                            os.chdir(optimization_dir_path)
                            
                            # Group files by base names into subfolders (silent in workflow)
                            import io
                            import contextlib
                            f = io.StringIO()
                            with contextlib.redirect_stdout(f):
                                group_files_by_base_with_tracking(".")
                                combine_xyz_files()
                                create_combined_mol()
                                create_summary_with_tracking(".")
                                
                                # Reuse stage folder for redo/resume and write into fixed target folder.
                                is_redo = hasattr(context, 'recalculated_files') and bool(context.recalculated_files)
                                reuse_folder = is_redo or os.path.exists(os.path.join(parent_dir, target_sim_base))
                                similarity_folder = collect_out_files_with_tracking(
                                    reuse_existing=reuse_folder,
                                    target_sim_folder=target_sim_base
                                )
                            
                            # Extract key info from output
                            output = f.getvalue()
                            if context.workflow_verbose_level >= 1:
                                if 'Summary written to' in output:
                                    print("\nSummary file(s) generated")
                            # Look for similarity folder reference
                            if 'Copied' in output and 'similarity' in output:
                                for line in output.split('\n'):
                                    if 'Copied' in line and '.out files to' in line:
                                        if context.workflow_verbose_level >= 1:
                                            print(line)
                                        # Extract similarity folder name (e.g., "similarity/orca_out_3")
                                        import re
                                        match = re.search(r'to\s+(similarity[^\s]*)', line)
                                        if match:
                                            sim_folder = match.group(1)
                                            context.optimization_sim_folder = sim_folder
                                            # Also set as pending for next similarity stage
                                            sim_base = sim_folder.split('/')[0] if '/' in sim_folder else sim_folder
                                            context.pending_similarity_folder = sim_base
                                        break
                            
                            if context.workflow_verbose_level >= 1:
                                print(f"\n✓ Files organized and sorted")
                            
                    except Exception as e:
                        print(f"⚠ Warning: Could not organize files: {e}")
                    finally:
                        os.chdir(saved_cwd)
            except Exception as e:
                print(f"✗ Error running calculations: {e}")
                return 1
        else:
            print("✗ Error: No launcher script found, cannot execute calculations")
            return 1
            
    else:
        print(f"Warning: Optimization directory not found")
        return 1
    
    # Clean up retry_input folder if it exists
    if os.path.exists("retry_input"):
        shutil.rmtree("retry_input")
    
    return 0

def execute_similarity_stage(context: WorkflowContext, stage: Dict[str, Any]) -> int:
    """
    Execute similarity analysis stage.
    Runs from the similarity/ folder which contains orca_out_N/ subfolders.
    Supports both similarity/ (first run) and similarity_2/ (after optimization).
    """
    def _count_latest_similarity_representatives(sim_dir: str) -> Tuple[Optional[int], Optional[int]]:
        """Return counts for latest motifs_* and umotifs_* representative xyz files."""
        try:
            motif_dirs = sorted(glob.glob(os.path.join(sim_dir, "motifs_*")))
            umotif_dirs = sorted(glob.glob(os.path.join(sim_dir, "umotifs_*")))

            motif_count: Optional[int] = None
            umotif_count: Optional[int] = None

            if motif_dirs:
                latest_motif_dir = motif_dirs[-1]
                motif_count = len(glob.glob(os.path.join(latest_motif_dir, "motif_*.xyz")))

            if umotif_dirs:
                latest_umotif_dir = umotif_dirs[-1]
                umotif_count = len(glob.glob(os.path.join(latest_umotif_dir, "umotif_*.xyz")))

            return motif_count, umotif_count
        except Exception:
            return None, None

    update_list_file: Optional[str] = None

    # Determine which similarity folder to use dynamically based on the most recent stage
    # Priority: use the most recently set similarity folder (from optimization/refinement that just ran)
    similarity_base = None
    
    # Check if there's a pending_similarity_folder (set by optimization/refinement organize step)
    if hasattr(context, 'pending_similarity_folder') and context.pending_similarity_folder:
        similarity_base = context.pending_similarity_folder
        # Clear it after use so it doesn't affect next stage
        context.pending_similarity_folder = None
    
    # Fallback: check what optimization_sim_folder or refinement_sim_folder were set
    if not similarity_base:
        # If refinement_sim_folder is more recent (set after optimization_sim_folder), prefer it
        if hasattr(context, 'refinement_sim_folder') and context.refinement_sim_folder:
            opt_base = context.refinement_sim_folder.split('/')[0] if '/' in context.refinement_sim_folder else context.refinement_sim_folder
            if os.path.exists(opt_base):
                similarity_base = opt_base
        
        # Otherwise use optimization_sim_folder
        if not similarity_base and hasattr(context, 'optimization_sim_folder') and context.optimization_sim_folder:
            calc_base = context.optimization_sim_folder.split('/')[0] if '/' in context.optimization_sim_folder else context.optimization_sim_folder
            if os.path.exists(calc_base):
                similarity_base = calc_base
    
    # Final fallback: pick the latest similarity folder if present.
    if not similarity_base:
        similarity_candidates = [
            d for d in os.listdir('.')
            if d.startswith('similarity') and os.path.isdir(d)
        ]
        if similarity_candidates:
            def _sim_sort_key(name: str) -> int:
                if name == 'similarity':
                    return 1
                match = re.search(r'^similarity_(\d+)$', name)
                return int(match.group(1)) if match else 0

            similarity_base = sorted(similarity_candidates, key=_sim_sort_key)[-1]
        else:
            print("Warning: similarity folder not found")
            if getattr(context, 'is_workflow', False):
                return 1
            return 0
    
    # Store similarity folder for protocol summary
    context.sim_folder = similarity_base
    # CRITICAL: Also update similarity_dir so redo logic uses correct folder
    context.similarity_dir = similarity_base
    
    # Clean old similarity results before re-running (but keep orca_out and cache)
    # This is CRITICAL to prevent reusing stale skipped_structures from previous runs
    # Check if this is a re-run by looking for existing clustering results
    has_old_results = (
        os.path.exists(os.path.join(similarity_base, "clustering_summary.txt")) or
        os.path.exists(os.path.join(similarity_base, "skipped_structures"))
    )
    
    if has_old_results:
        # Clean everything except orca_out_* folders and cache
        items_to_remove = [
            'dendrogram_images', 'extracted_clusters', 'extracted_data',
            'skipped_structures', 'clustering_summary.txt', 'boltzmann_distribution.txt'
        ]
        # Also remove motifs and umotifs folders
        for item in os.listdir(similarity_base):
            if item.startswith('motifs_') or item.startswith('umotifs_'):
                items_to_remove.append(item)
        
        for item in items_to_remove:
            item_path = os.path.join(similarity_base, item)
            if os.path.exists(item_path):
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception:
                    pass
    
    # Verify output subfolder exists and count input structures.
    out_candidates = []
    for item in sorted(os.listdir(similarity_base)):
        if item.startswith("orca_out_") or item.startswith("opt_out_") or item.startswith("gaussian_out_"):
            out_candidates.append(item)

    out_folder_found = len(out_candidates) > 0
    sim_input_count = 0
    if out_folder_found:
        if getattr(context, 'is_workflow', False) and len(out_candidates) > 1:
            print(
                f"Error: Multiple output folders found in {similarity_base}/: {', '.join(out_candidates)}. "
                "Workflow mode requires a single deterministic folder."
            )
            return 1

        selected_out = out_candidates[0]
        out_dir = os.path.join(similarity_base, selected_out)
        sim_input_count = (
            len(glob.glob(os.path.join(out_dir, "*.out")))
            + len(glob.glob(os.path.join(out_dir, "*.log")))
        )
    
    if not out_folder_found:
        print(f"Warning: No orca_out_*, gaussian_out_*, or opt_out_* folder found in {similarity_base}/")
        if getattr(context, 'is_workflow', False):
            return 1
        return 0
    
    args = stage['args']
    
    # Find similarity script
    similarity_script = None
    other_args = []
    
    for arg in args:
        if arg.endswith('.py') and 'similarity' in arg.lower():
            similarity_script = arg
        else:
            other_args.append(arg)
    
    if not similarity_script:
        similarity_script = find_similarity_script()
        if not similarity_script:
            print("Warning: similarity-v01.py script not found")
            if getattr(context, 'is_workflow', False):
                return 1
            return 0  # Not an error
    
    # Default: concise output (no extra printing)
    # In workflow mode, always concise. Could add --verbose flag to stage args later if needed.
    verbose = False
    
    if verbose:
        print(f"\nRunning similarity analysis...")
        print(f"Using similarity script: {os.path.basename(similarity_script)}")
    
    # Build command - pass '1' via stdin to auto-select the first folder
    cmd = ['python', similarity_script] + other_args
    
    # Add --cores if not already specified and user explicitly set ascec_parallel_cores in input file
    # (ascec_parallel_cores > 0 means it was explicitly set)
    has_cores_arg = any(arg.startswith('--cores') or arg.startswith('-j') for arg in other_args)
    if not has_cores_arg and hasattr(context, 'ascec_parallel_cores') and context.ascec_parallel_cores > 0:
        cmd.extend(['--cores', str(context.ascec_parallel_cores)])

    # Default similarity threshold when user doesn't provide one.
    has_threshold_arg = any(arg.startswith('--th=') or arg.startswith('--threshold=') for arg in other_args)
    if not has_threshold_arg:
        cmd.extend(['--th', '2.0'])
    
    # No need to specify motif prefix - similarity script auto-detects from filenames:
    # conf_* files → creates motifs_*/ folder (after calculation)
    # motif_* files → creates motifs_*/ folder (first similarity)
    # umotif_* files → creates umotifs_*/ folder (after optimization)
    
    # If this is a redo and we have a list of recalculated files, pass them for incremental update
    if hasattr(context, 'recalculated_files') and context.recalculated_files:
        # Create temp file with list of files to update
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            update_list_file = f.name
            for basename in context.recalculated_files:
                f.write(f"{basename}\n")
        cmd.extend(['--update-cache', update_list_file])
    
    if verbose:
        print(f"{' '.join(cmd)}\n")
        print(f"Working directory: {similarity_base}\n")
    try:
        # Auto-select folder 1 by providing '1\n' as stdin
        # Stream output to avoid pipe buffer deadlock on large outputs
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT, text=True, bufsize=1, 
                               cwd=similarity_base, universal_newlines=True)
        
        # Send input and close stdin immediately
        if proc.stdin:
            proc.stdin.write('1\n')
            proc.stdin.close()
        
        # Filter patterns for output
        skip_lines = [
            'Found the following folder(s) containing quantum chemistry',
            'Enter the number of the folder to process',
            'Only .log files found',
            'Only .out files found',
            'Processing ',
            'folder(s) for files matching',
            'Created motifs dendrogram'
        ]
        
        # Lines that should have blank line BEFORE them
        add_blank_before = [
            'H-bond group',
            'Processed ',
            'Creating ',
            ',   Critical:'  # Summary line "Redundant: X,   Critical: Y"
        ]
        
        # Lines that should have blank line AFTER them
        add_blank_after = [
            'Data extraction complete. Proceeding to clustering.',
            'Motifs created:',
            ',   Critical:',  # Summary line "Redundant: X,   Critical: Y"
            'Clustering summary saved to'
        ]
        
        # Stream output line by line in real-time and collect for post-processing
        # Only print in verbose mode
        if verbose:
            print()
        stdout_lines = []
        prev_line = ""
        
        if proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                line = line.rstrip('\n')
                stdout_lines.append(line)
                
                # Capture processing folder line
                if 'Processing folder:' in line:
                    match = re.search(r'Processing folder:\s+(\S+)', line)
                    if match:
                        folder_name = match.group(1)
                        context.sim_folder = f"{similarity_base}/{folder_name}"
                
                
                # Skip printing in non-verbose mode (but still collect lines)
                if not verbose:
                    continue
                # Skip interactive prompts and folder listing
                if any(skip in line for skip in skip_lines):
                    continue
                
                # Skip folder listing lines
                if re.match(r'\s*\[\d+\]\s+\S+', line):
                    continue
                
                # Skip blank lines
                if not line.strip():
                    continue
                
                # Add blank line before certain sections
                if any(phrase in line for phrase in add_blank_before):
                    if prev_line.strip():  # Only if previous wasn't blank
                        print()
                
                print(line)
                
                # Add blank line after certain sections
                if any(phrase in line for phrase in add_blank_after):
                    print()
                
                prev_line = line
            
            proc.stdout.close()
        
        proc.wait()
        
        stdout = '\n'.join(stdout_lines)
        
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, stdout, '')
        
        # Parse collected output for motifs count (already printed in real-time above)
        for line in stdout_lines:
            # Capture motifs created count for protocol summary
            if 'Motifs created:' in line and 'representatives' in line:
                match = re.search(r'(\d+)\s+representatives', line)
                if match:
                    context.sim_motifs_created = int(match.group(1))

        # Persist latest motif/umotif counts for compact workflow progress display.
        motif_count, umotif_count = _count_latest_similarity_representatives(similarity_base)
        context.last_similarity_motif_count = motif_count
        context.last_similarity_umotif_count = umotif_count
        context.last_similarity_input_count = sim_input_count if sim_input_count > 0 else None
        stage_total = (motif_count or 0) + (umotif_count or 0)
        if stage_total > 0:
            # Keep summary count stage-local to avoid carrying over previous similarity values.
            context.sim_motifs_created = stage_total
        stage_key = getattr(context, 'current_stage_key', '')
        match = re.search(r'^similarity_(\d+)$', stage_key)
        if match:
            stage_num = int(match.group(1))
            context.similarity_stage_counts[stage_num] = stage_total
            if sim_input_count > 0:
                context.similarity_stage_input_counts[stage_num] = sim_input_count
        else:
            # Combined mode runs similarity while current_stage_key is still optimization/refinement.
            prev_match = re.search(r'^(optimization|refinement)_(\d+)$', stage_key)
            if prev_match:
                stage_num = int(prev_match.group(2)) + 1
                context.similarity_stage_counts[stage_num] = stage_total
                if sim_input_count > 0:
                    context.similarity_stage_input_counts[stage_num] = sim_input_count
        
        if verbose:
            print("\n✓ Similarity analysis completed")
        
        # Check if files were saved to need_recalculation directory
        # This happens when structures with imaginary frequencies need to be recalculated
        need_recalc_dir = os.path.join(similarity_base, "skipped_structures", "need_recalculation")
        clustered_with_minima_dir = os.path.join(similarity_base, "skipped_structures", "clustered_with_minima")
        critical_non_conv_dir = os.path.join(similarity_base, "skipped_structures", "critical_non_converged")
        
        recalc_basenames = []
        
        # Always include critical structures (need_recalculation - imaginary frequencies)
        if os.path.exists(need_recalc_dir):
            xyz_files = glob.glob(os.path.join(need_recalc_dir, "*.xyz"))
            if xyz_files:
                # Extract basenames (without .xyz extension)
                recalc_basenames.extend([os.path.splitext(os.path.basename(f))[0] for f in xyz_files])
        
        # Always include critical non-converged structures (need rescue hessian)
        if os.path.exists(critical_non_conv_dir):
            xyz_files = glob.glob(os.path.join(critical_non_conv_dir, "*.xyz"))
            if xyz_files:
                recalc_basenames.extend([os.path.splitext(os.path.basename(f))[0] for f in xyz_files])
        
        # If user wants all structures to be true minima (--skipped=0), also include clustered_with_minima
        # Check if this is optimization stage with --skipped=0 threshold
        if os.path.exists(clustered_with_minima_dir):
            # Check if we're in an optimization context with skipped threshold = 0
            # This would be stored in context from the stage arguments
            include_clustered = False
            
            # For optimization stage, check if skipped threshold is 0
            current_stage = getattr(context, 'current_stage', None)
            if current_stage:
                if current_stage.get('type') == 'optimization':  # type: ignore[union-attr]
                    args = current_stage.get('args', [])  # type: ignore[union-attr]
                    for arg in args:
                        if arg.startswith('--skipped='):
                            skipped_val = float(arg.split('=')[1])
                            # User wants 0% skipped (all structures to be true minima)
                            if skipped_val <= 0.0:
                                include_clustered = True
                                break
            
            if include_clustered:
                xyz_files = glob.glob(os.path.join(clustered_with_minima_dir, "*.xyz"))
                if xyz_files:
                    recalc_basenames.extend([os.path.splitext(os.path.basename(f))[0] for f in xyz_files])
        
        if recalc_basenames:
            # Store in context for the optimization stage to use
            context.recalculated_files = recalc_basenames
        
        # Store similarity directory for context
        context.similarity_dir = similarity_base
        
        # Cleanup temp file if created (but DON'T clear recalculated_files - it's needed for the redo loop!)
        if update_list_file and os.path.exists(update_list_file):
            try:
                os.remove(update_list_file)
                # NOTE: Do NOT clear context.recalculated_files here!
                # The optimization stage redo needs this list to know which files to regenerate.
            except Exception:
                pass
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Similarity analysis failed with code {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        # Cleanup temp file if it exists
        if update_list_file and os.path.exists(update_list_file):
            try:
                os.remove(update_list_file)
            except Exception:
                pass
        return e.returncode
    except Exception as e:
        if update_list_file and os.path.exists(update_list_file):
            try:
                os.remove(update_list_file)
            except Exception:
                pass
        print(f"Error running similarity analysis: {e}")
        return 1

def execute_refinement_stage(context: WorkflowContext, stage: Dict[str, Any]) -> int:
    """Execute refinement stage (for motifs from similarity clustering)."""
    # Very similar to optimization stage, but uses motifs from similarity analysis
    
    # Store context globally for helper functions
    sys._current_workflow_context = context  # type: ignore[attr-defined]
    
    # Parse arguments - defaults for when flags are not explicitly provided
    max_stage_redos = 3   # --redo: redo entire opt+similarity
    max_critical = 0      # default: 0% critical structures allowed (strict)
    max_skipped = None
    # Note: --retry removed; launch failures auto-retry up to 10 times (hardcoded)
    
    args = stage.get('args', [])
    workflow_concise = getattr(context, 'is_workflow', False) and getattr(context, 'workflow_verbose_level', 0) < 1
    template_inp = stage.get('template_inp')
    launcher_sh = stage.get('launcher_sh')
    
    for arg in args:
        if arg.startswith('--redo='):
            max_stage_redos = int(arg.split('=')[1])
        elif arg.startswith('--critical='):
            max_critical = float(arg.split('=')[1])
        elif arg.startswith('--skipped='):
            max_skipped = float(arg.split('=')[1])
    
    # Store threshold mode in context for redo logic
    # --critical: only retry structures with imaginary freqs (need_recalculation)
    # --skipped: retry all skipped structures (need_recalculation + clustered_with_minima)
    context.use_skipped_threshold = (max_skipped is not None)
    
    if not template_inp:
        print("Error: Refinement requires template input file or embedded template label")
        return 1

    # Resolve path or embedded label from the workflow input file.
    resolved_template = resolve_template_reference(context, template_inp)
    if not resolved_template:
        print(f"Error: Template file/label not found: {template_inp}")
        return 1
    template_inp = resolved_template
    
    # Handle launcher: if not provided, auto-detect ORCA and create launcher
    if launcher_sh:
        launcher_sh = os.path.abspath(launcher_sh)
        if not os.path.exists(launcher_sh):
            print(f"Warning: Launcher script not found: {launcher_sh}")
            if not workflow_concise:
                print("  Attempting to auto-detect ORCA installation...")
            launcher_sh = None
    
    # Get QM alias from context (read from input.in line 9)
    qm_alias = getattr(context, 'qm_alias', 'orca')
    
    if not launcher_sh:
        # Try to auto-create launcher
        opt_dir = "optimization"
        if not os.path.exists(opt_dir):
            os.makedirs(opt_dir, exist_ok=True)
        auto_launcher = create_auto_launcher(opt_dir, "orca", qm_alias, quiet=workflow_concise)
        if auto_launcher:
            launcher_sh = auto_launcher
            if not workflow_concise:
                print(f"  Created auto-generated launcher: {os.path.basename(auto_launcher)}")
        else:
            print("Error: No launcher script provided and could not auto-detect ORCA")
            return 1
    
    # CRITICAL: Optimization needs to READ motifs from calculation's similarity folder
    # The optimization stage created motifs in similarity/motifs_XX/ (or umotifs in similarity_N/)
    # Optimization will READ from similarity/ and WRITE outputs to similarity_2/ (or next numbered)
    # The subsequent similarity will create umotifs_YY/ (since input came from optimization)
    
    # Step 1: Find where the MOTIFS are (from calculation's similarity stage)
    motifs_source_folder = None
    
    # Check if optimization stage set a similarity folder
    if hasattr(context, 'optimization_sim_folder') and context.optimization_sim_folder:
        # Extract base folder (e.g., "similarity" from "similarity/orca_out_10")
        calc_base = context.optimization_sim_folder.split('/')[0] if '/' in context.optimization_sim_folder else context.optimization_sim_folder
        motifs_source_folder = calc_base
    else:
        # Fallback: search for existing similarity folders with motifs
        parent_dir = os.getcwd()
        existing_sims = []
        for item in os.listdir(parent_dir):
            if item.startswith("similarity") and os.path.isdir(item):
                existing_sims.append(item)
        
        if existing_sims:
            # Sort numerically
            existing_sims.sort(key=lambda x: (int(m.group(1)) if (m := re.search(r'_(\d+)', x)) else 0))
            # Find the first one with motifs or umotifs
            for sim_folder in existing_sims:
                if glob.glob(os.path.join(sim_folder, "motifs_*/")) or glob.glob(os.path.join(sim_folder, "umotifs_*/")):
                    motifs_source_folder = sim_folder
                    break
    
    if not motifs_source_folder:
        motifs_source_folder = "similarity"  # Default
    
    # Step 2: Find motifs or umotifs in the source folder (prefer umotifs if both exist)
    umotif_dirs = glob.glob(os.path.join(motifs_source_folder, "umotifs_*/"))
    motif_dirs = glob.glob(os.path.join(motifs_source_folder, "motifs_*/"))
    
    # Prefer umotifs over motifs (more refined clustering)
    if umotif_dirs:
        motif_dirs = umotif_dirs
    elif not motif_dirs:
        print(f"Warning: No motif/umotif directories found in {motifs_source_folder}/")
        print("  Skipping optimization stage")
        return 0
    
    # Use the most recent motifs directory
    motif_dirs.sort()
    motif_dir = motif_dirs[-1]
    
    # Step 3: Determine where optimization OUTPUTS will go
    # This is typically the next similarity folder (similarity_2, similarity_3, etc.)
    # Always prefer the expected folder from motifs_source_folder and only reuse cached
    # refinement_sim_folder if it matches; this prevents stale folder drift.
    if motifs_source_folder == "similarity":
        expected_sim_folder: str = "similarity_2"
    else:
        # Extract number and increment
        match = re.search(r'similarity_(\d+)', motifs_source_folder)
        if match:
            next_num = int(match.group(1)) + 1
            expected_sim_folder = f"similarity_{next_num}"
        else:
            expected_sim_folder = "similarity_2"

    existing_ref_sim_raw = getattr(context, 'refinement_sim_folder', None)
    existing_ref_sim: Optional[str] = existing_ref_sim_raw if isinstance(existing_ref_sim_raw, str) else None
    existing_ref_base: Optional[str] = existing_ref_sim.split('/')[0] if existing_ref_sim and '/' in existing_ref_sim else existing_ref_sim
    if isinstance(existing_ref_base, str) and existing_ref_base == expected_sim_folder:
        used_sim_folder: str = str(existing_ref_base)
    else:
        used_sim_folder = str(expected_sim_folder)
    
    # Store in refinement_sim_folder (optimization's dedicated variable)
    context.refinement_sim_folder = used_sim_folder
    # Also update similarity_dir so the similarity stage knows where to look
    context.similarity_dir = used_sim_folder
    context.pending_similarity_folder = used_sim_folder
    
    # Store motifs source in context
    context.refinement_motifs_source = motif_dir
    
    # CRITICAL: When resuming optimization stage (with -i flag), clean the OUTPUT similarity folder
    # This must happen BEFORE process_optimization_redo() checks for skipped_structures
    # This ensures we don't reuse stale skipped_structures from previous similarity runs
    # Only delete the OUTPUT folder (used_sim_folder), NOT the INPUT folder (motifs_source_folder)
    cache_file = getattr(context, 'cache_file', 'protocol_cache.pkl')
    cache = load_protocol_cache(cache_file) if os.path.exists(cache_file) else {}
    stage_key = getattr(context, 'current_stage_key', '')
    stage_was_started = stage_key in cache.get('stages', {})
    
    if stage_was_started:
        output_sim_folder: str = str(used_sim_folder)
        if os.path.exists(output_sim_folder):
            # Verify this is NOT the motifs source folder (don't delete our input!)
            if output_sim_folder != motifs_source_folder:
                # CRITICAL: Check if this similarity folder has skipped_structures for redo
                # If skipped_structures exists, we're in a redo scenario - do NOT delete!
                skipped_dir = os.path.join(output_sim_folder, "skipped_structures")
                if os.path.exists(skipped_dir):
                    # Redo mode - preserve the similarity folder with skipped structures
                    pass
                else:
                    # No skipped structures - safe to delete and rebuild
                    try:
                        shutil.rmtree(output_sim_folder)
                    except Exception:
                        pass
    
    # Process redo structures at the START of the stage (if need_recalculation exists)
    # This ensures that when the workflow restarts this stage after a similarity failure,
    # we immediately regenerate inputs and delete old outputs before checking completion
    opt_dir = getattr(context, 'refinement_stage_dir', 'optimization')
    if not opt_dir:  # Handle empty string
        opt_dir = 'optimization'
    
    # Only process redo if optimization directory exists
    # process_optimization_redo will check for skipped_structures internally and return False if none
    if os.path.exists(opt_dir):
        redo_result = process_optimization_redo(context, opt_dir, template_inp)
        
        # If no redo structures found, clear recalculated_files
        if not redo_result and hasattr(context, 'recalculated_files'):
            context.recalculated_files = None
        
        # Clean up any old input files at the root level (they're now in subdirectories after calculations run)
        # This matches execute_optimization_stage logic to prevent duplicates
        # Note: Do NOT clean redo files here - they need to be run first!
        # Cleanup happens AFTER sorting when files are moved to subdirectories
        if not redo_result:
            # Not a redo - safe to clean orphaned root files
            for old_inp in glob.glob(os.path.join(opt_dir, "*.inp")) + glob.glob(os.path.join(opt_dir, "*.com")):
                basename = os.path.basename(old_inp).replace('.inp', '').replace('.com', '')
                short_name = basename.replace('_opt', '').replace('_calc', '')
                subdir = os.path.join(opt_dir, short_name)
                if os.path.isdir(subdir):
                    try:
                        os.remove(old_inp)
                    except:
                        pass
    
    # Only print motifs message if this is NOT a redo attempt
    # (redo attempts already printed "Processing redo structures..." message)
    if not (hasattr(context, 'recalculated_files') and context.recalculated_files):
        if not workflow_concise:
            print(f"Using motifs from: {motif_dir}")
    
    # Get motif/umotif XYZ files from the motifs directory
    # Look for both motif_*.xyz and umotif_*.xyz patterns
    motif_files = glob.glob(os.path.join(motif_dir, "motif_*.xyz"))
    umotif_files = glob.glob(os.path.join(motif_dir, "umotif_*.xyz"))
    motif_files.extend(umotif_files)  # Combine both patterns
    combined_file = glob.glob(os.path.join(motif_dir, "*combined*.xyz"))
    
    if not motif_files and not combined_file:
        print("Warning: No motif/umotif files found in motifs directory")
        if getattr(context, 'is_workflow', False):
            return 1
        return 0
    
    # Create refinement directory (or reuse if resuming)
    opt_dir = "refinement"
    
    # Check if we're resuming - if so, reuse existing directory
    cache_file = getattr(context, 'cache_file', 'protocol_cache.pkl')
    cache = load_protocol_cache(cache_file) if os.path.exists(cache_file) else {}
    stage_key = getattr(context, 'current_stage_key', '')
    stage_was_started = stage_key in cache.get('stages', {})
    
    # Check if this is a redo scenario (process_redo_structures was called and found files)
    is_redo = hasattr(context, 'recalculated_files') and bool(context.recalculated_files)
    
    # Check if directory has meaningful content (input or output files)
    has_content = False
    if os.path.exists(opt_dir):
        for item in os.listdir(opt_dir):
            if item.endswith(('.inp', '.com', '.gjf', '.out', '.log')):
                has_content = True
                break
            # Also check subdirectories
            item_path = os.path.join(opt_dir, item)
            if os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    if subitem.endswith(('.inp', '.com', '.gjf', '.out', '.log')):
                        has_content = True
                        break
                if has_content:
                    break
    
    if os.path.exists(opt_dir) and has_content and (stage_was_started or is_redo):
        # Resuming or redo - reuse existing directory
        if is_redo:
            if not workflow_concise:
                print("Resuming: Using existing refinement directory (Redo Mode)\n")
        else:
            if not workflow_concise:
                print("Resuming: Using existing refinement directory\n")
    else:
        # Not resuming or empty directory - create fresh directory
        if os.path.exists(opt_dir):
            # Remove old directory if it exists
            shutil.rmtree(opt_dir)
        os.makedirs(opt_dir)
        # CRITICAL: If we recreated the directory, this is NOT a redo scenario
        is_redo = False

    # Persist the refinement working directory for downstream stages.
    context.refinement_stage_dir = opt_dir
    
    # Choose files to process:
    # - If only 1 motif file and a combined file exists: use only the single motif file
    # - If multiple motif files: use the combined file (if it exists), otherwise use individual files
    if len(motif_files) == 1 and combined_file:
        # Single motif - use the individual file, ignore combined
        all_xyz_files = motif_files
    elif combined_file:
        # Multiple motifs - prefer combined file
        all_xyz_files = combined_file
    else:
        # No combined file - use individual motifs
        all_xyz_files = motif_files
    
    # Determine QM program and input extension from template
    qm_program = "orca" if template_inp.endswith(".inp") else "gaussian"
    input_ext = ".inp" if qm_program == "orca" else ".com"
    
    # Read template content
    with open(template_inp, 'r') as f:
        template_content = f.read()
    
    # Determine if we should skip input file creation:
    # - is_redo: redo mode, use existing files
    # - stage_was_started AND has_content: resume mode, use existing files
    should_skip_input_creation = is_redo or (stage_was_started and has_content)
    
    # Only generate new input files if NOT in redo/resume mode
    if not should_skip_input_creation:
        # Process each XYZ file and create input files
        all_input_files = []
        for xyz_file in all_xyz_files:
            # Call _process_xyz_file_for_opt with the correct parameters
            xyz_file_data = (xyz_file, template_content, opt_dir, qm_program, input_ext)
            result_files, message = _process_xyz_file_for_opt(xyz_file_data)
            all_input_files.extend(result_files)
        
        if not all_input_files:
            print("Error: No input files created")
            return 1
        
        if not workflow_concise:
            print(f"Created {len(all_input_files)} input file(s)")
    else:
        # Redo/Resume mode: reuse existing input files
        # process_redo_structures handles updating failed ones in redo mode
        # Get list of input files from optimization directory
        all_input_files = []
        
        # Prefer subdirectory files over root files (they may have been organized)
        subdirs = [d for d in os.listdir(opt_dir) if os.path.isdir(os.path.join(opt_dir, d))]
        for subdir in sorted(subdirs, key=natural_sort_key):
            subdir_path = os.path.join(opt_dir, subdir)
            for f in os.listdir(subdir_path):
                if f.endswith(('.inp', '.com', '.gjf')):
                    all_input_files.append(os.path.join(subdir, f))
        
        # Only check root if no subdirectory files found
        if not all_input_files:
            root_inputs = [f for f in os.listdir(opt_dir) if f.endswith(('.inp', '.com', '.gjf'))]
            all_input_files.extend(root_inputs)
        
        if not all_input_files:
            print("Error: No existing input files found in optimization directory")
            return 1
    
    # Create launcher script if provided
    if launcher_sh:
        launcher_path = os.path.join(opt_dir, f"launcher_{qm_program}.sh")
        
        # Read launcher template to get environment setup
        with open(launcher_sh, 'r') as f:
            launcher_template = f.read()
        
        # Extract the environment setup part (everything before ###)
        env_setup = ""
        if "###" in launcher_template:
            env_setup = launcher_template.split("###")[0].strip()
        else:
            # If no ### marker, use the whole template as environment setup
            env_setup = launcher_template.strip()
        
        # For launcher, only use base filenames (not subdirectory paths)
        # The launcher is for user convenience with flat structure
        launcher_input_files = []
        for inp_file in all_input_files:
            # Extract just the filename, not subdirectory paths
            basename_with_ext = os.path.basename(inp_file)
            if basename_with_ext not in launcher_input_files:
                launcher_input_files.append(basename_with_ext)
        
        # Create launcher with environment setup and execution commands
        with open(launcher_path, 'w') as f:
            # Write environment setup from template
            f.write(env_setup)
            f.write("\n\n###\n\n")
            
            # Write execution commands for each input file (flat structure)
            for i, inp_file in enumerate(sorted(launcher_input_files, key=natural_sort_key)):
                basename = os.path.splitext(inp_file)[0]
                if qm_program == "orca":
                    # Execute ORCA (launcher sets up PATH with ORCA directory)
                    f.write(f"orca {basename}.inp > {basename}.out")
                else:
                    # For Gaussian, use g16 or g09
                    f.write(f"g16 {basename}.com")
                
                # Add continuation for all but last command
                if i < len(launcher_input_files) - 1:
                    f.write(" ; \\\n")
                else:
                    f.write("\n")
        
        os.chmod(launcher_path, 0o755)
        if not workflow_concise:
            print(f"Created launcher script: {launcher_path}")
    
    # Execute the refinement calculations
    if os.path.exists(os.path.join(opt_dir, "launcher_orca.sh")) or os.path.exists(os.path.join(opt_dir, "launcher_gaussian.sh")):
        if not workflow_concise:
            print(f"\nExecuting refinement calculations...")
        
        # Determine launcher name and QM program
        if os.path.exists(os.path.join(opt_dir, "launcher_orca.sh")):
            launcher_path = os.path.join(opt_dir, "launcher_orca.sh")
            qm_program = 'orca'
        else:
            launcher_path = os.path.join(opt_dir, "launcher_gaussian.sh")
            qm_program = 'gaussian'
        
        # Get list of input files to process
        # First check at root level
        # Filter out ORCA intermediate files (.scfgrad.inp, .scfp.inp, etc.) and rescue inputs
        def is_valid_input_file(filename):
            """Check if file is a valid input file (not an ORCA intermediate or rescue file)"""
            if not filename.endswith(('.inp', '.com', '.gjf')):
                return False
            # Exclude ORCA intermediate files and rescue inputs (already processed)
            excluded_patterns = ['.scfgrad.', '.scfp.', '.tmp.', '.densities.', '.scfhess.', '_rescue.']
            return not any(pattern in filename for pattern in excluded_patterns)
        
        input_files = sorted([f for f in os.listdir(opt_dir) if is_valid_input_file(f)], key=natural_sort_key)
        
        # If no files at root and sort command was used, check subdirectories
        if not input_files:
            subdirs = [d for d in os.listdir(opt_dir) if os.path.isdir(os.path.join(opt_dir, d))]
            for subdir in sorted(subdirs, key=natural_sort_key):
                subdir_path = os.path.join(opt_dir, subdir)
                subdir_files = [os.path.join(subdir, f) for f in os.listdir(subdir_path) if is_valid_input_file(f)]
                input_files.extend(sorted(subdir_files, key=natural_sort_key))
        
        # Load cache and exclusions BEFORE using them
        cache_file = getattr(context, 'cache_file', 'protocol_cache.pkl')
        cache = load_protocol_cache(cache_file) if os.path.exists(cache_file) else {}
        
        # Get the current stage key from context (e.g., "optimization_4")
        stage_key = getattr(context, 'current_stage_key', '')
        
        # Check if redo structures exist (files scheduled for recalculation)
        redo_files = set()
        if hasattr(context, 'recalculated_files') and context.recalculated_files:
            redo_files = set(context.recalculated_files)
        
        # Scan ALL subdirectories for completed calculations (check for .out files)
        # Files with only .out.backup are being redone and should NOT be counted as completed
        actual_completed = []
        opt_dir = context.refinement_stage_dir if context.refinement_stage_dir else "refinement"
        
        # 1. Check optimization directory subfolders
        if os.path.exists(opt_dir):
            for item in os.listdir(opt_dir):
                item_path = os.path.join(opt_dir, item)
                if os.path.isdir(item_path):
                    # Skip if this file is marked for redo
                    # CRITICAL: redo_files contains basenames with _opt suffix (e.g., "umotif_01_opt")
                    # but item is the directory name (e.g., "umotif_01")
                    # So we need to check both "item" AND "item_opt"/"item_calc" against redo_files
                    redo_variants = [item, f"{item}_opt", f"{item}_calc"]
                    if any(variant in redo_files for variant in redo_variants):
                        continue
                    
                    # Check if this subdirectory has a completed calculation
                    # Try different naming patterns: umotif_01/umotif_01.out or umotif_01/umotif_01_opt.out
                    possible_names = [f"{item}.out", f"{item}_opt.out", f"{item}_calc.out"]
                    out_file = None
                    for name in possible_names:
                        test_path = os.path.join(item_path, name)
                        if os.path.exists(test_path):
                            out_file = test_path
                            break
                    
                    if out_file:
                        # Verify completion (OPI-aware for ORCA 6.1+)
                        if check_orca_terminated_normally_opi(out_file):
                            # Store with _opt suffix for consistency
                            if '_opt' not in item and '_calc' not in item:
                                actual_completed.append(f"{item}_opt")
                            else:
                                actual_completed.append(item)
            
            # 1b. Check for flat files in optimization directory
            for item in os.listdir(opt_dir):
                if item.endswith('.out') or item.endswith('.log'):
                    basename = os.path.splitext(item)[0]
                    # Skip if already found in subdir or marked for redo
                    if basename not in actual_completed and basename not in redo_files:
                        out_file = os.path.join(opt_dir, item)
                        # Verify completion (OPI-aware for ORCA 6.1+)
                        if item.endswith('.out'):
                            if check_orca_terminated_normally_opi(out_file):
                                actual_completed.append(basename)
                        else:  # Gaussian .log files
                            try:
                                with open(out_file, 'r', encoding='utf-8', errors='replace') as f:
                                    content = f.read()
                                    if 'Normal termination' in content:
                                        actual_completed.append(basename)
                            except Exception:
                                pass
        
        # 2. Check similarity_X/orca_out_* folders (files moved there after sorting)
        # This is CRITICAL for showing correct counts when resuming
        # IMPORTANT: Only check the optimization's OWN similarity folder, not all similarity folders
        # (to avoid counting calculation results from similarity/ as optimization results)
        parent_dir = os.getcwd()
        
        # Determine which similarity folder belongs to THIS optimization stage
        refinement_sim_folder = getattr(context, 'refinement_sim_folder', None)
        
        # CRITICAL: Strip orca_out_X suffix if present
        # organize step may set refinement_sim_folder to "similarity_2/orca_out_5"
        # but we need just "similarity_2" to scan for orca_out subdirectories
        if refinement_sim_folder and '/' in refinement_sim_folder:
            refinement_sim_folder = refinement_sim_folder.split('/')[0]
        
        if not refinement_sim_folder:
            # Calculate the expected folder based on motifs source
            if hasattr(context, 'refinement_motifs_source'):
                motifs_source = context.refinement_motifs_source
                # Extract base folder (e.g., "similarity" from "similarity/motifs_03/")
                if '/' in motifs_source:
                    calc_base = motifs_source.split('/')[0]
                else:
                    calc_base = motifs_source
                
                # Optimization outputs go to the NEXT similarity folder
                if calc_base == "similarity":
                    refinement_sim_folder = "similarity_2"
                else:
                    # Extract number and increment
                    match = re.search(r'similarity_(\d+)', calc_base)
                    if match:
                        next_num = int(match.group(1)) + 1
                        refinement_sim_folder = f"similarity_{next_num}"
                    else:
                        refinement_sim_folder = "similarity_2"
        
        # Only check the optimization's designated similarity folder
        if refinement_sim_folder and os.path.isdir(refinement_sim_folder):
            for subitem in os.listdir(refinement_sim_folder):
                if subitem.startswith('orca_out_') or subitem.startswith('gaussian_out_') or subitem.startswith('calc_out_'):
                    out_dir = os.path.join(refinement_sim_folder, subitem)
                    if os.path.isdir(out_dir):
                        for f in os.listdir(out_dir):
                            if f.endswith('.out') or f.endswith('.log'):
                                basename = os.path.splitext(f)[0]
                                if basename not in actual_completed and basename not in redo_files:
                                    # Verify completion (OPI-aware for ORCA 6.1+)
                                    out_file = os.path.join(out_dir, f)
                                    if f.endswith('.out'):
                                        if check_orca_terminated_normally_opi(out_file):
                                            actual_completed.append(basename)
                                    else:  # Gaussian .log files
                                        try:
                                            with open(out_file, 'r', encoding='utf-8', errors='replace') as f_obj:
                                                content = f_obj.read()
                                                if 'Normal termination' in content:
                                                    actual_completed.append(basename)
                                        except:
                                            pass
        
        # Update completed_opts to match reality
        completed_opts = actual_completed
        
        excluded_numbers = cache.get('excluded_refinements', [])
        
        # Apply exclusion filtering to completed_opts
        completed_opts = [f for f in completed_opts if not match_exclusion(f, excluded_numbers)]
        
        # Count total inputs (including those already completed and cached)
        # This should be the TOTAL expected optimizations, not just new ones
        all_input_basenames = set()
        
        # Add basenames from input files (pending optimizations)
        for f in input_files:
            if not match_exclusion(f, excluded_numbers):
                # Handle both simple filenames and paths like "motif_01/motif_01_opt.inp"
                basename = os.path.splitext(os.path.basename(f))[0]
                all_input_basenames.add(basename)
        
        # Add basenames from already completed files (CRITICAL for correct count)
        for f in completed_opts:
            # completed_opts contains basenames, not full paths
            all_input_basenames.add(f)
        
        num_inputs = len(all_input_basenames)
        
        # Store initial completed count (before loop may modify completed_opts)
        initial_completed_count = len(completed_opts)

        progress_cb = context.update_progress
        if workflow_concise and callable(progress_cb):
            progress_cb(f"{initial_completed_count}/{num_inputs} ...")
        
        if completed_opts and not workflow_concise:
            print(f"Resuming: {len(completed_opts)}/{num_inputs} optimizations already completed")
        
        if excluded_numbers and not workflow_concise:
            print(f"Exclusions active: {excluded_numbers}")
            print()
        
        # Read launcher script to get environment setup
        with open(launcher_path, 'r') as f:
            launcher_content = f.read()
        
        # Run calculations one by one, checking for normal termination with retry
        num_completed = 0  # Track only newly completed in this run
        num_failed = 0
        failed_optimizations = []  # Track failed input files
        # Hardcoded: retry launch failures up to 10 times
        max_launch_retries = 10
        # Time threshold for detecting launch failure (seconds)
        launch_failure_threshold = 5.0
        
        for input_file in input_files:
            basename = os.path.splitext(input_file)[0]
            
            # Skip excluded optimizations UNLESS this file is being redone
            # In redo mode, recalculated_files should be processed even if previously excluded
            if match_exclusion(input_file, excluded_numbers):
                # Check if this file is in the redo list
                is_redo_file = (is_redo and hasattr(context, 'recalculated_files') and
                                context.recalculated_files is not None and
                                basename in context.recalculated_files)
                if not is_redo_file:
                    if not is_redo:
                        if not workflow_concise:
                            print(f"  Skipping: {input_file} (excluded)")
                    continue
            
            output_file = basename + ('.out' if qm_program == 'orca' else '.log')
            input_path = os.path.join(opt_dir, input_file)
            output_path = os.path.join(opt_dir, output_file)
            
            # Skip if output already exists AND is successfully completed
            # This handles redo scenarios where failed .out files were deleted
            output_exists = False
            
            # Check root directory first
            if os.path.exists(output_path):
                output_exists = True
            else:
                # Check subdirectory (if files were sorted)
                # E.g. refinement/motif_01_opt/motif_01_opt.out
                subdir_path = os.path.join(opt_dir, basename, output_file)
                if os.path.exists(subdir_path):
                    output_path = subdir_path
                    output_exists = True
                else:
                    # Also check shortened basename subdirectory
                    # E.g. refinement/motif_01/motif_01_opt.out
                    short_basename = basename
                    if '_opt' in basename:
                        short_basename = basename.replace('_opt', '')
                    elif '_calc' in basename:
                        short_basename = basename.replace('_calc', '')
                    
                    subdir_path = os.path.join(opt_dir, short_basename, output_file)
                    if os.path.exists(subdir_path):
                        output_path = subdir_path
                        output_exists = True

            if output_exists:
                if qm_program == 'orca':
                    # Use OPI-aware check for ORCA 6.1+ support
                    is_complete = check_orca_terminated_normally_opi(output_path)
                else:
                    with open(output_path, 'r') as f:
                        output_content = f.read()
                    is_complete = 'Normal termination of Gaussian' in output_content
                
                if is_complete:
                    continue
            
            # Extract display name (basename only) for cleaner output
            display_name = os.path.basename(input_file)
            
            progress_cb = context.update_progress
            if workflow_concise and callable(progress_cb):
                progress_cb(f"{initial_completed_count + num_completed}/{num_inputs} ...")
            elif not workflow_concise:
                print(f"  Running: {display_name}...", end='', flush=True)
            
            # ═══════════════════════════════════════════════════════════
            # LAUNCH FAILURE RETRY STRATEGY
            # ═══════════════════════════════════════════════════════════
            # Only retry if the calculation fails to launch (instant crash).
            # If the calculation starts normally (runs for a reasonable time),
            # we don't retry regardless of exit code - let redo mode handle it.
            # ═══════════════════════════════════════════════════════════
            
            success = False
            launch_attempt = 0
            calculation_started = False  # True once calculation runs past threshold
            
            while launch_attempt < max_launch_retries and not calculation_started:
                launch_attempt += 1
                
                # Update display for retries
                if launch_attempt > 1 and not workflow_concise:
                    print(f"\r  Running: {display_name}... ↻ (launch attempt {launch_attempt})\033[K", end='', flush=True)
                
                # ═══ STEP 1: CLEAN OUTPUT FILES ═══
                
                # Determine working directory
                if '/' in input_file or '\\' in input_file:
                    opt_subdir = os.path.dirname(input_file)
                    opt_working_dir = os.path.join(opt_dir, opt_subdir)
                    input_file_relative = os.path.basename(input_file)
                    output_file_relative = os.path.basename(output_file)
                else:
                    opt_working_dir = opt_dir
                    input_file_relative = input_file
                    output_file_relative = output_file
                
                input_filename = os.path.basename(input_file)
                basename_only = os.path.splitext(input_filename)[0]
                
                # Remove ALL auxiliary/output files for clean run (except input files)
                for item in os.listdir(opt_working_dir):
                    if item.startswith(basename_only) and not item.endswith(('.inp', '.com', '.gjf')):
                        item_path = os.path.join(opt_working_dir, item)
                        if os.path.isfile(item_path):
                            try:
                                os.remove(item_path)
                            except:
                                pass
                
                # ═══ STEP 2: RUN CALCULATION ═══
                
                temp_script = os.path.join(opt_working_dir, f'_run_{basename_only}.sh')
                with open(temp_script, 'w') as f:
                    f.write(launcher_content.split('###')[0])  # Environment setup
                    f.write("\n\n")
                    if qm_program == 'orca':
                        f.write(f"# Set unique scratch directory for this ORCA process\n")
                        f.write(f"export TMPDIR=\"$(pwd)/.orca_tmp_{basename_only}_$$\"\n")
                        f.write(f"mkdir -p \"$TMPDIR\"\n")
                        f.write(f"trap 'rm -rf \"$TMPDIR\"' EXIT\n\n")
                        orca_root_var = extract_orca_root_from_launcher(launcher_content)
                        if orca_root_var:
                            f.write(f"${{{orca_root_var}}}/orca {input_file_relative} > {output_file_relative}\n")
                        else:
                            f.write(f"orca {input_file_relative} > {output_file_relative}\n")
                    elif qm_program == 'gaussian':
                        f.write(f"$GAUSS_ROOT/g16 {input_file_relative}\n")
                
                os.chmod(temp_script, 0o755)
                
                script_name = os.path.basename(temp_script)
                
                # Measure execution time to detect launch failures
                start_time = time.time()
                result = subprocess.run(
                    ['bash', script_name],
                    cwd=opt_working_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                elapsed_time = time.time() - start_time
                
                # If calculation ran past the threshold, it started normally
                # Don't retry even if it failed - redo mode will handle it
                if elapsed_time > launch_failure_threshold:
                    calculation_started = True
                
                # Cleanup temp script
                if os.path.exists(temp_script):
                    os.remove(temp_script)
                if qm_program == 'orca':
                    for tmp_dir in glob.glob(os.path.join(opt_working_dir, f'.orca_tmp_{basename_only}_*')):
                        try:
                            shutil.rmtree(tmp_dir)
                        except:
                            pass
                
                # ═══ STEP 3: CHECK SUCCESS ═══
                
                if os.path.exists(output_path):
                    try:
                        if qm_program == 'orca':
                            normal_term = check_orca_terminated_normally_opi(output_path)
                        else:
                            with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
                                output_content = f.read()
                            normal_term = 'Normal termination of Gaussian' in output_content
                        
                        if normal_term:
                            # SUCCESS!
                            num_completed += 1
                            progress_cb = context.update_progress
                            if workflow_concise and callable(progress_cb):
                                progress_cb(f"{initial_completed_count + num_completed}/{num_inputs} ...")
                            else:
                                if launch_attempt > 1:
                                    print(f"\r  Running: {display_name}... ✓ (launch attempt {launch_attempt})\033[K")
                                else:
                                    print(f"\r  Running: {display_name}... ✓\033[K")
                            success = True
                            
                            # Update cache - avoid duplicates during redo
                            if input_file not in completed_opts:
                                completed_opts.append(input_file)
                            basename_only = os.path.splitext(os.path.basename(input_file))[0]
                            all_input_basenames.add(basename_only)
                            stage_key = getattr(context, 'current_stage_key', 'optimization')
                            update_protocol_cache(stage_key, 'in_progress',
                                                result={'completed_files': completed_opts,
                                                       'total_files': num_inputs,
                                                       'num_completed': num_completed},
                                                cache_file=cache_file)
                            break  # EXIT RETRY LOOP
                    except:
                        pass
                
                # If calculation started but failed, don't retry - exit loop
                # Only continue loop if it was a launch failure
                if calculation_started:
                    break
            
            # If loop exits without success
            if not success:
                if not workflow_concise:
                    if calculation_started:
                        # Calculation started but didn't complete normally - don't retry
                        # This will be handled by redo mode if needed
                        print(f"\r  Running: {display_name}... ✗ (no normal termination)\033[K")
                    elif launch_attempt >= max_launch_retries:
                        # Launch failed after all retries
                        print(f"\r  Running: {display_name}... ✗ (launch failed after {launch_attempt} attempts)\033[K")
                    else:
                        print(f"\r  Running: {display_name}... ✗\033[K")
                num_failed += 1
                failed_optimizations.append(input_file)
        
        # Print status
        # Recalculate num_inputs from the updated set to reflect newly completed optimizations
        num_inputs = len(all_input_basenames)
        # Total completed = initial completed + newly completed in this run
        total_completed = initial_completed_count + num_completed
        # Ensure we don't show more completed than total inputs (can happen if files are removed)
        if total_completed > num_inputs:
            total_completed = num_inputs
        if not workflow_concise:
            print(f"\nStatus: {total_completed}/{num_inputs} optimizations completed")
        
        # Store for protocol summary (use total)
        context.refinement_completed = total_completed
        context.refinement_total = num_inputs
        
        # Clean up old failed files if they exist and all succeeded
        if not failed_optimizations:
            for old_file in [os.path.join(opt_dir, "failed_opt.txt"), os.path.join(opt_dir, "launcher_failed.sh")]:
                if os.path.exists(old_file):
                    try:
                        os.remove(old_file)
                    except Exception:
                        pass
        
        # Handle failed optimizations
        if failed_optimizations:
            if not workflow_concise:
                print(f"Failed optimizations: {len(failed_optimizations)}/{num_inputs}")
            
            # Write failed_opt.txt
            failed_list_file = os.path.join(opt_dir, "failed_opt.txt")
            with open(failed_list_file, 'w') as f:
                f.write(f"# Failed optimizations: {len(failed_optimizations)}/{num_inputs}\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for failed_file in sorted(failed_optimizations, key=natural_sort_key):
                    f.write(f"{failed_file}\n")
            
            if not workflow_concise:
                print(f"Failed optimizations list written to: {failed_list_file}")
        
        # Organize files if any optimizations are completed (new or already done)
        if total_completed > 0:
            if num_completed > 0 and not workflow_concise:
                print("\n\nRefinement calculations completed")
            
            # Check if this is a redo (recalculated files exist)
            if not workflow_concise:
                print(f"All refinements completed successfully")
            
            # Organize results - run full sort/organize for both redo and normal mode
            saved_cwd = os.getcwd()
            try:
                # Determine similarity folder - use refinement_sim_folder (set earlier in this function)
                if hasattr(context, 'refinement_sim_folder') and context.refinement_sim_folder:
                    # Use the folder determined at the start of execute_optimization_stage
                    sim_base = context.refinement_sim_folder.split('/')[0] if '/' in context.refinement_sim_folder else context.refinement_sim_folder
                else:
                    # Calculate next similarity folder
                    root_dir = os.getcwd()
                    base_name = "similarity"
                    counter = 2
                    sim_base = base_name
                    
                    if os.path.exists(os.path.join(root_dir, sim_base)):
                        while True:
                            sim_base = f"{base_name}_{counter}"
                            if not os.path.exists(os.path.join(root_dir, sim_base)):
                                break
                            counter += 1
                    
                    context.similarity_dir = sim_base
                
                os.chdir(opt_dir)
                
                # Get exclusions from cache to filter output files
                cache = load_protocol_cache(cache_file) if os.path.exists(cache_file) else {}
                excluded_numbers = cache.get('excluded_refinements', [])
                
                # Group files by base names into subfolders (silent in workflow)
                import io
                import contextlib
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    group_files_by_base_with_tracking(".")
                    combine_xyz_files()
                    create_combined_mol()
                    create_summary_with_tracking(".")
                    
                    # Collect output files - reuse existing similarity folder in redo mode
                    # But first, we need to temporarily filter excluded files
                    # Save original find_out_files function
                    original_find_out_files = find_out_files
                    
                    def filtered_find_out_files(root_dir):
                        """Find .out files but exclude those matching exclusion patterns."""
                        all_files = original_find_out_files(root_dir)
                        filtered_files = []
                        for file_path in all_files:
                            basename = os.path.splitext(os.path.basename(file_path))[0]
                            # Check if this file matches any exclusion
                            if not match_exclusion(basename, excluded_numbers):
                                filtered_files.append(file_path)
                        return filtered_files
                    
                    # Temporarily replace function in globals
                    globals()['find_out_files'] = filtered_find_out_files
                    
                    try:
                        # Check if we should reuse existing folder:
                        # - True if redo mode (recalculated_files exist)
                        # - Also True if resuming and similarity folder already has orca_out folder
                        reuse_folder = is_redo
                        if not reuse_folder and os.path.exists(os.path.join(os.path.dirname(os.getcwd()), sim_base)):
                            # Check if orca_out folder already exists in similarity
                            sim_full_path = os.path.join(os.path.dirname(os.getcwd()), sim_base)
                            existing_orca = glob.glob(os.path.join(sim_full_path, "orca_out_*"))
                            if existing_orca:
                                reuse_folder = True
                        
                        similarity_folder = collect_out_files_with_tracking(
                            reuse_existing=reuse_folder,
                            target_sim_folder=sim_base
                        )
                    finally:
                        # Restore original function
                        globals()['find_out_files'] = original_find_out_files
                
                # Extract key info from output
                output = f.getvalue()
                if context.workflow_verbose_level >= 1:
                    if 'Summary written to' in output:
                        print("\nSummary file(s) generated")
                if 'Copied' in output and 'similarity' in output:
                    # Extract the copy message and similarity folder
                    for line in output.split('\n'):
                        if 'Copied' in line and '.out files to' in line:
                            if context.workflow_verbose_level >= 1:
                                print(line)
                            # Extract similarity folder name
                            match = re.search(r'to\s+(similarity[^\s]*)', line)
                            if match:
                                sim_folder = match.group(1)
                                context.refinement_sim_folder = sim_folder
                                # Also set as pending for next similarity stage
                                sim_base = sim_folder.split('/')[0] if '/' in sim_folder else sim_folder
                                context.pending_similarity_folder = sim_base
                            break
                
                # Note: File update messages are now handled within collect_out_files_with_tracking
                # No need to print them again here (unlike optimization stage where we copy manually)
                
                # Clean up orphaned root input files after sorting
                # Files should now be in subdirectories, so root copies are duplicates
                for old_inp in glob.glob(os.path.join(".", "*.inp")) + glob.glob(os.path.join(".", "*.com")):
                    inp_basename = os.path.basename(old_inp).replace('.inp', '').replace('.com', '')
                    short_name = inp_basename.replace('_opt', '').replace('_calc', '')
                    subdir = os.path.join(".", short_name)
                    if os.path.isdir(subdir):
                        # Subfolder exists - safe to remove root file
                        try:
                            os.remove(old_inp)
                        except:
                            pass
            
                if context.workflow_verbose_level >= 1:
                    print(f"\n✓ Files organized and sorted")
            except Exception as e:
                print(f"⚠ Warning: Could not organize files: {e}")
            finally:
                    os.chdir(saved_cwd)
        else:
            print("✗ No output files found")
            return 1
    else:
        print(f"Warning: No launcher script found in {opt_dir}/")
        return 1
    
    return 0

def main_ascec_integrated():
    # Ensure glob is accessible throughout this function (imported at module level)
    import glob as _glob_module
    import re as _re_module
    glob = _glob_module
    re = _re_module
    
    # CHECK FOR VERSION COMMAND (early check before other processing)
    if len(sys.argv) >= 2 and sys.argv[1] in ["--version", "-V", "version"]:
        print_version_banner("ASCEC")
        return
    
    # CHECK FOR EXCLUDE COMMAND (pause protocol and add exclusions)
    if len(sys.argv) >= 3 and sys.argv[2].lower() == "exclude":
        # Syntax: ascec04 <protocol.in> exclude [stage] [pattern]
        protocol_file = sys.argv[1]
        
        # Find existing protocol cache file for this input file
        existing_caches = sorted(glob.glob("protocol_*.pkl"))
        
        if not existing_caches:
            print(f"Error: No active protocol found")
            print(f"No protocol_*.pkl cache files found in current directory")
            print("This command is used to exclude stage inputs from a paused protocol.")
            sys.exit(1)
        
        # Find cache that matches this protocol file
        cache_file = None
        for cache_path in existing_caches:
            test_cache = load_protocol_cache(cache_path)
            if test_cache and test_cache.get('input_file') == protocol_file:
                cache_file = cache_path
                break
        
        if cache_file is None:
            print(f"Error: No protocol cache found for {protocol_file}")
            print(f"\nAvailable protocol caches:")
            for cache_path in existing_caches:
                test_cache = load_protocol_cache(cache_path)
                associated_input = test_cache.get('input_file', 'unknown')
                print(f"  {cache_path} -> {associated_input}")
            sys.exit(1)
        
        print(f"Using cache file: {cache_file} (for {protocol_file})\n")
        
        # Load cache
        cache = load_protocol_cache(cache_file)
        
        if len(sys.argv) == 3:
            # Show current exclusions for this protocol
            print("\n" + "=" * 60)
            print(f"Protocol Exclusion Manager - {protocol_file}")
            print("=" * 60)
            
            # Show excluded files for optimization
            optimization_excluded = cache.get('excluded_optimizations', [])
            optimization2_excluded = cache.get('excluded_optimizations_2', [])
            if optimization_excluded:
                print(f"\nExcluded optimizations (1st stage): {optimization_excluded}")
            else:
                print("\nNo optimizations (1st stage) excluded")
            
            if optimization2_excluded:
                print(f"Excluded optimizations (2nd stage): {optimization2_excluded}")
            else:
                print("No optimizations (2nd stage) excluded")
            
            # Show excluded files for optimization
            opt_excluded = cache.get('excluded_refinements', [])
            if opt_excluded:
                print(f"Excluded optimizations: {opt_excluded}")
            else:
                print("No optimizations excluded")
            
            print("\nUsage:")
            print(f"  ascec04 {protocol_file} exclude opt <pattern>     - Exclude 1st optimization files")
            print(f"  ascec04 {protocol_file} exclude opt2 <pattern>    - Exclude 2nd optimization files")
            print(f"  ascec04 {protocol_file} exclude ref <pattern>     - Exclude refinement files")
            print(f"  ascec04 {protocol_file} exclude clear             - Clear all exclusions")
            print(f"  ascec04 {protocol_file} exclude opt clear         - Clear 1st optimization exclusions")
            print(f"  ascec04 {protocol_file} exclude opt2 clear        - Clear 2nd optimization exclusions")
            print(f"  ascec04 {protocol_file} exclude ref clear         - Clear refinement exclusions")
            print("\nPattern examples:")
            print("  2               -> Exclude conf_2 (for opt) or motif_02 (for ref)")
            print("  2,5-9           -> Exclude 2, 5, 6, 7, 8, 9")
            print("  3-15            -> Exclude 3 through 15")
            print("  1,3,5-10        -> Exclude 1, 3, and 5 through 10")
            print(f"\nAfter adding exclusions, resume with: ascec04 {protocol_file} protocol")
            sys.exit(0)
        
        if len(sys.argv) < 4:
            print("Error: Missing stage type (opt, opt2, or ref)")
            sys.exit(1)
        
        stage_type = sys.argv[3].lower()

        if stage_type == "clear":
            # Clear all exclusions
            cache['excluded_optimizations'] = []
            cache['excluded_optimizations_2'] = []
            cache['excluded_refinements'] = []
            save_protocol_cache(cache, cache_file)
            print("✓ All exclusions cleared")
            sys.exit(0)
        
        if stage_type not in ['opt', 'opt2', 'ref']:
            print(f"Error: Invalid stage type '{stage_type}'. Use 'opt', 'opt2', or 'ref'")
            sys.exit(1)
        
        if len(sys.argv) < 5:
            print("Error: Missing exclusion pattern or 'clear' command")
            print("Examples:")
            print(f"  ascec04 {protocol_file} exclude opt 3")
            print(f"  ascec04 {protocol_file} exclude opt2 4-5")
            print(f"  ascec04 {protocol_file} exclude ref 03-15")
            print(f"  ascec04 {protocol_file} exclude opt clear   # Clear opt exclusions")
            print(f"  ascec04 {protocol_file} exclude ref clear   # Clear ref exclusions")
            sys.exit(1)
        
        pattern = sys.argv[4]
        
        # Check for stage-specific clear
        if pattern.lower() == 'clear':
            if stage_type == 'opt':
                cache['excluded_optimizations'] = []
                save_protocol_cache(cache, cache_file)
                print("✓ Cleared exclusions for opt (1st stage)")
            elif stage_type == 'opt2':
                cache['excluded_optimizations_2'] = []
                save_protocol_cache(cache, cache_file)
                print("✓ Cleared exclusions for opt2 (2nd stage)")
            else:  # ref
                cache['excluded_refinements'] = []
                save_protocol_cache(cache, cache_file)
                print("✓ Cleared exclusions for ref")
            sys.exit(0)
        
        try:
            excluded_numbers = parse_exclusion_pattern(pattern)
            
            if stage_type == 'opt':
                key = 'excluded_optimizations'
                existing = cache.get(key, [])
                existing.extend(excluded_numbers)
                cache[key] = sorted(list(set(existing)))
                print(f"✓ Excluded optimizations (1st stage) matching: {excluded_numbers}")
                print(f"  Total excluded: {cache[key]}")
            elif stage_type == 'opt2':
                key = 'excluded_optimizations_2'
                existing = cache.get(key, [])
                existing.extend(excluded_numbers)
                cache[key] = sorted(list(set(existing)))
                print(f"✓ Excluded optimizations (2nd stage) matching: {excluded_numbers}")
                print(f"  Total excluded: {cache[key]}")
            else:  # ref
                key = 'excluded_refinements'
                existing = cache.get(key, [])
                existing.extend(excluded_numbers)
                cache[key] = sorted(list(set(existing)))
                print(f"✓ Excluded refinements matching: {excluded_numbers}")
                print(f"  Total excluded: {cache[key]}")
            
            save_protocol_cache(cache, cache_file)
            print(f"\nResume protocol with: ascec04 {protocol_file} protocol")
            
        except ValueError as e:
            print(f"Error parsing exclusion pattern: {e}")
            sys.exit(1)
        
        sys.exit(0)
    
    # CHECK FOR PROTOCOL MODE (workflow embedded in input file)
    # Check for protocol mode with optional stage restart
    # Syntax: ascec04 input.asc protocol [stage]
    # Examples: 
    #   ascec04 input.asc protocol          # Run full protocol
    #   ascec04 input.asc protocol opt      # Restart optimization stage
    #   ascec04 input.asc protocol opt1     # Restart first optimization stage
    #   ascec04 input.asc protocol opt2     # Restart second optimization stage
    #   ascec04 input.asc protocol ref      # Restart refinement stage
    if len(sys.argv) >= 3 and sys.argv[2].lower() == "protocol":
        input_file = sys.argv[1]
        restart_stage = None
        incomplete_mode = False  # New flag for -i option
        
        # Check for stage restart argument and -i flag
        if len(sys.argv) >= 4:
            restart_stage = sys.argv[3].lower()
            
            # Check if -i flag is present
            if len(sys.argv) >= 5 and sys.argv[4] == "-i":
                incomplete_mode = True
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found")
            sys.exit(1)
        
        # Extract protocol from input file
        protocol = extract_protocol_from_input(input_file)
        
        if protocol is None:
            print("Error: No protocol found in input file")
            print("\nExpected format in input file:")
            print(".asc,")
            print("r1 --box10,")
            print("opt -c --redo=3 ../preopt_input.inp ../launcher_orca.sh,")
            print("similarity --th=2")
            print("\nFlag meanings:")
            print("  --redo=N: Redo entire stage (opt/ref + similarity) up to N times")
            print("\nNote: Launch failures are automatically retried up to 10 times.")
            print("\nStage restart:")
            print("  ascec04 input.asc protocol opt    - Restart optimization stage (deletes files)")
            print("  ascec04 input.asc protocol opt1   - Restart first optimization stage")
            print("  ascec04 input.asc protocol opt2   - Restart second optimization stage")
            print("  ascec04 input.asc protocol ref    - Restart refinement stage")
            print("  ascec04 input.asc protocol 2 -i   - Mark stage 2 as incomplete (keeps files)")
            print("\nThe -i flag marks a stage as incomplete without deleting files,")
            print("allowing the workflow to continue from where it left off.")
            sys.exit(1)
        
        # Print ASCII logo
        print("\n===========================================================================")
        
        # Helper function to center text
        def center_text(text, width=75):
            return text.center(width)
        
        print(center_text("*********************"))
        print(center_text("*     A S C E C     *"))
        print(center_text("*********************"))
        print("")
        print("                             √≈≠==≈                                  ")
        print("   √≈≠==≠≈√   √≈≠==≠≈√         ÷++=                      ≠===≠       ")
        print("     ÷++÷       ÷++÷           =++=                     ÷×××××=      ")
        print("     =++=       =++=     ≠===≠ ÷++=      ≠====≠         ÷-÷ ÷-÷      ")
        print("     =++=       =++=    =××÷=≠=÷++=    ≠÷÷÷==÷÷÷≈      ≠××≠ =××=     ")
        print("     =++=       =++=   ≠××=    ÷++=   ≠×+×    ×+÷      ÷+×   ×+××    ")
        print("     =++=       =++=   =+÷     =++=   =+-×÷==÷×-×≠    =×+×÷=÷×+-÷    ")
        print("     ≠×+÷       ÷+×≠   =+÷     =++=   =+---×××××÷×   ≠××÷==×==÷××≠   ")
        print("      =××÷     =××=    ≠××=    ÷++÷   ≠×-×           ÷+×       ×+÷   ")
        print("       ≠=========≠      ≠÷÷÷=≠≠=×+×÷-  ≠======≠≈√  -÷×+×≠     ≠×+×÷- ")
        print("          ≠===≠           ≠==≠  ≠===≠     ≠===≠    ≈====≈     ≈====≈ ")
        print("")
        print("")
        
        print(center_text("Universidad de Antioquia - Medellín - Colombia"))
        print("")
        print("")
        print(center_text("Annealing Simulado Con Energía Cuántica"))
        print("")
        print(center_text(ASCEC_VERSION))
        print("")
        print(center_text("Química Física Teórica - QFT"))
        print("")
        print("===========================================================================")
        print("\nProtocol mode activated")
        
        # Store original protocol text for summary
        protocol_text = protocol.strip()
        
        # Replace .asc or .asc, placeholder with actual input filename
        # Handle both ".asc" and ".asc," patterns
        import re
        protocol = re.sub(r'\.asc,?', input_file + ',', protocol, count=1)
        # Clean up any double commas that might result
        protocol = protocol.replace(',,', ',')
        
        # Parse protocol into arguments (split by spaces but respect quoted strings)
        protocol_args = shlex.split(protocol)
        
        # Parse workflow stages (skip input file, start from first ',' or 'then')
        stages = parse_workflow_stages(protocol_args[1:])
        
        if not stages:
            print("Error: No valid workflow stages found in protocol")
            sys.exit(1)
        
        # Handle stage restart if specified
        if restart_stage:
            # Find and modify cache to mark stage as incomplete
            import glob
            existing_caches = sorted(glob.glob("protocol_*.pkl"))
            cache_file = None
            
            if existing_caches:
                for cache_path in existing_caches:
                    test_cache = load_protocol_cache(cache_path)
                    if test_cache and test_cache.get('input_file') == input_file:
                        cache_file = cache_path
                        break
            
            if cache_file:
                cache = load_protocol_cache(cache_file)
                
                # Get all stage keys and organize them
                all_stage_keys = sorted(cache.get('stages', {}).keys(), 
                                       key=lambda k: int(k.split('_')[1]))
                
                # Build a map of stage types for user guidance
                optimization_stages = [k for k in all_stage_keys if k.startswith('optimization_')]
                refinement_stages = [k for k in all_stage_keys if k.startswith('refinement_')]
                
                # Find matching stage(s) to restart by stage number
                stages_to_restart = []
                
                # Check if restart_stage is a number (e.g., "2", "3", "4")
                try:
                    stage_num = int(restart_stage)
                    # Find all stages with this number
                    for stage_key in cache.get('stages', {}).keys():
                        key_num = int(stage_key.split('_')[1])
                        if key_num == stage_num:
                            stages_to_restart.append(stage_key)
                except ValueError:
                    # Not a number - match by stage alias
                    # New naming:
                    #   opt -> optimization stage
                    #   ref -> refinement stage
                    if restart_stage in ['opt', 'opt1']:
                        # Find first optimization stage
                        if optimization_stages:
                            stages_to_restart.append(optimization_stages[0])
                        else:
                            print(f"\nError: No optimization stages found in cache")
                            sys.exit(1)
                    elif restart_stage == 'opt2':
                        # Find second optimization stage
                        if len(optimization_stages) >= 2:
                            stages_to_restart.append(optimization_stages[1])
                        elif len(optimization_stages) == 1:
                            print(f"\nError: Only one optimization stage exists: {optimization_stages[0]}")
                            print(f"There is no '{restart_stage}' stage in this protocol")
                            sys.exit(1)
                        else:
                            print(f"\nError: No optimization stages found in cache")
                            sys.exit(1)
                    elif restart_stage.startswith('opt') and len(restart_stage) > 3:
                        # Handle opt3, opt4, etc.
                        try:
                            opt_idx = int(restart_stage[3:]) - 1
                            if 0 <= opt_idx < len(optimization_stages):
                                stages_to_restart.append(optimization_stages[opt_idx])
                            else:
                                print(f"\nError: {restart_stage} not found")
                                print(f"There are {len(optimization_stages)} optimization stage(s): {', '.join(optimization_stages)}")
                                sys.exit(1)
                        except ValueError:
                            pass
                    elif restart_stage in ['ref', 'ref1']:
                        # Find first refinement stage
                        if refinement_stages:
                            stages_to_restart.append(refinement_stages[0])
                        else:
                            print(f"\nError: No refinement stages found in cache")
                            sys.exit(1)
                    elif restart_stage == 'ref2':
                        # Find second refinement stage
                        if len(refinement_stages) >= 2:
                            stages_to_restart.append(refinement_stages[1])
                        elif len(refinement_stages) == 1:
                            print(f"\nError: Only one refinement stage exists: {refinement_stages[0]}")
                            print(f"There is no 'ref2' stage in this protocol")
                            sys.exit(1)
                        else:
                            print(f"\nError: No refinement stages found in cache")
                            sys.exit(1)
                    elif restart_stage.startswith('ref') and len(restart_stage) > 3:
                        # Handle ref3, ref4, etc.
                        try:
                            ref_idx = int(restart_stage[3:]) - 1
                            if 0 <= ref_idx < len(refinement_stages):
                                stages_to_restart.append(refinement_stages[ref_idx])
                            else:
                                print(f"\nError: {restart_stage} not found")
                                print(f"There are {len(refinement_stages)} refinement stage(s): {', '.join(refinement_stages)}")
                                sys.exit(1)
                        except ValueError:
                            pass
                
                if stages_to_restart:
                    if incomplete_mode:
                        print(f"\nMarking stage(s) as incomplete: {', '.join(stages_to_restart)}")
                        print(f"  → Keeping existing directories (incomplete mode)")
                    else:
                        print(f"\nRestarting stage(s): {', '.join(stages_to_restart)}")
                    
                    # Find the minimum stage number to restart
                    min_restart_num = min([int(key.split('_')[1]) for key in stages_to_restart])
                    
                    # Only delete directories if NOT in incomplete mode
                    if not incomplete_mode:
                        print(f"  → Deleting all directories and cache entries from stage {min_restart_num} onwards\n")
                        
                        # Delete directories based on stage type
                        # Map of directories to delete based on restart stage
                        if 'calculation' in [k.split('_')[0] for k in stages_to_restart]:
                            # Restarting calculation: delete calculation/, similarity/, optimization/
                            for dir_name in ['calculation', 'similarity', 'optimization']:
                                if os.path.exists(dir_name):
                                    print(f"     Removing {dir_name}/")
                                    shutil.rmtree(dir_name)
                            # Also remove numbered variants
                            for pattern in ['calculation_*', 'similarity_*', 'optimization_*']:
                                for dir_path in glob.glob(pattern):
                                    if os.path.isdir(dir_path):
                                        print(f"     Removing {dir_path}/")
                                        shutil.rmtree(dir_path)
                        
                        elif 'optimization' in [k.split('_')[0] for k in stages_to_restart]:
                            # Restarting optimization: delete optimization/ and later similarity/
                            for dir_name in ['optimization']:
                                if os.path.exists(dir_name):
                                    print(f"     Removing {dir_name}/")
                                    shutil.rmtree(dir_name)
                            # Also remove numbered variants
                            for pattern in ['optimization_*']:
                                for dir_path in glob.glob(pattern):
                                    if os.path.isdir(dir_path):
                                        print(f"     Removing {dir_path}/")
                                        shutil.rmtree(dir_path)
                            # Remove similarity folders that come after optimization
                            similarity_dirs = sorted(glob.glob('similarity*'))
                            for sim_dir in similarity_dirs:
                                if os.path.isdir(sim_dir):
                                    # Check if it's similarity_N where N > optimization-stage similarity index
                                    if '_' in sim_dir:
                                        try:
                                            sim_num = int(sim_dir.split('_')[1])
                                            if sim_num > min_restart_num:
                                                print(f"     Removing {sim_dir}/")
                                                shutil.rmtree(sim_dir)
                                        except:
                                            pass
                    
                    # Clear cache entries from restart stage onwards
                    all_stage_keys = sorted(cache.get('stages', {}).keys(), 
                                           key=lambda k: int(k.split('_')[1]))
                    
                    stages_to_remove = [k for k in all_stage_keys 
                                       if int(k.split('_')[1]) >= min_restart_num]
                    
                    if stages_to_remove:
                        print(f"\n  → Clearing cache entries: {', '.join(stages_to_remove)}")
                        for stage_key in stages_to_remove:
                            del cache['stages'][stage_key]
                    
                    # Mark protocol as incomplete so it resumes automatically
                    cache['completed'] = False
                    
                    # Save modified cache
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cache, f)
                    print(f"Cache updated: {cache_file}\n")
                else:
                    # Stage not found in cache - provide helpful error message
                    print(f"\nError: Stage '{restart_stage}' not found in cache")
                    print(f"\nAvailable stages in cache:")
                    for stage_key in all_stage_keys:
                        stage_type = stage_key.split('_')[0]
                        stage_num = stage_key.split('_')[1]
                        print(f"  [{stage_num}] {stage_type}")
                    
                    # Show optimization-specific guidance
                    if optimization_stages:
                        print(f"\nOptimization stages ({len(optimization_stages)} total):")
                        for i, opt_stage in enumerate(optimization_stages, 1):
                            stage_num = opt_stage.split('_')[1]
                            if i == 1:
                                print(f"  opt or opt1 -> stage {stage_num}")
                            else:
                                print(f"  opt{i} -> stage {stage_num}")

                    # Show refinement-specific guidance
                    if refinement_stages:
                        print(f"\nRefinement stages ({len(refinement_stages)} total):")
                        for i, ref_stage in enumerate(refinement_stages, 1):
                            stage_num = ref_stage.split('_')[1]
                            if i == 1:
                                print(f"  ref or ref1 -> stage {stage_num}")
                            else:
                                print(f"  ref{i} -> stage {stage_num}")
                    
                    print(f"\nUsage:")
                    print(f"  ascec04 {input_file} protocol <stage_number>")
                    print(f"  ascec04 {input_file} protocol opt    (or opt1, opt2, etc.)")
                    print(f"  ascec04 {input_file} protocol ref    (or ref1, ref2, etc.)")
                    sys.exit(1)
            else:
                print(f"\nError: No protocol cache found for {input_file}")
                print("Run the protocol first before trying to restart a stage")
                sys.exit(1)
        
        # Execute workflow with caching enabled
        result = execute_workflow_stages(input_file, stages, use_cache=True, protocol_text=protocol_text)
        sys.exit(result)

    # AUTO-DETECT EMBEDDED PROTOCOL (single input-file invocation)
    # Example: ascec04 glaw.asc
    if len(sys.argv) == 2 and os.environ.get("ASCEC_DISABLE_EMBEDDED_PROTOCOL") != "1":
        input_file = sys.argv[1]
        if os.path.exists(input_file):
            protocol = extract_protocol_from_input(input_file)
            if protocol:
                protocol_text = protocol

                # Replace placeholder ".asc," with actual input filename
                import re
                protocol = re.sub(r'\.asc,?', input_file + ',', protocol, count=1)
                protocol = protocol.replace(',,', ',')

                # Parse protocol into workflow stages
                protocol_args = shlex.split(protocol)
                stages = parse_workflow_stages(protocol_args[1:])

                if not stages:
                    print("Error: No valid workflow stages found in embedded protocol")
                    sys.exit(1)

                result = execute_workflow_stages(input_file, stages, use_cache=True, protocol_text=protocol_text)
                sys.exit(result)
    
    # CHECK FOR WORKFLOW MODE (',' or 'then' separator-based commands)
    if len(sys.argv) >= 2 and contains_workflow_separator(sys.argv[1:]):
        # First argument should be input file
        if len(sys.argv) < 4:  # Need at least: script, input, ,, stage
            print("Error: Workflow mode requires input file and at least one stage")
            print("Usage: ascec04 input.asc , r3 , opt template.inp launcher.sh , similarity --th=2")
            print("   or: ascec04 input.asc then r3 then opt template.inp launcher.sh then similarity --th=2")
            sys.exit(1)
        
        input_file = sys.argv[1]
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found")
            sys.exit(1)
        
        # Parse workflow stages (skip input file, start from first ',' or 'then')
        stages = parse_workflow_stages(sys.argv[2:])
        
        if not stages:
            print("Error: No valid workflow stages found")
            print("Usage: ascec04 input.asc , r3 , opt template.inp launcher.sh , similarity --th=2")
            print("   or: ascec04 input.asc then r3 then opt template.inp launcher.sh then similarity --th=2")
            sys.exit(1)
        
        # Execute workflow
        result = execute_workflow_stages(input_file, stages)
        sys.exit(result)
    
    # STANDARD SINGLE-COMMAND MODE (backward compatibility)
    # Setup argument parser - use parse_known_args to handle shell expansion
    parser = argparse.ArgumentParser(
        description="ASCEC - Simulated Annealing with Quantum Energy\nConformational sampling via Monte Carlo with quantum mechanical evaluation",
        usage="ascec [OPTIONS] COMMAND [ARGUMENTS]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""DESCRIPTION:
  ASCEC (Annealing Simulado con Energía Cuántica) performs automated 
  configurational sampling of molecular clusters using simulated annealing
  with quantum mechanical energy evaluation via ORCA or Gaussian.

COMMANDS:
  Sampling:
    input.asc [options]         Single annealing simulation
    input.asc rN                N replicated runs (e.g., r3 = 3 replicas)
    input.asc rN --boxP         N replicas with P%% box packing density
    input.asc box               Analyze simulation box requirements
  
    Analysis:
        opt TEMPLATE LAUNCHER       Generate optimization inputs from results
        ref TEMPLATE LAUNCHER       Generate refinement inputs from motifs
    sort [--nosum|--justsum]    Organize outputs and generate summary reports
    simil [OPTIONS]             Clustering analysis (see: ascec simil --help)
  
  Utilities:
    diagram [--scaled]          Generate energy evolution plots
    merge [result]              Interactively combine XYZ structure files
    update TEMPLATE [PATTERN]   Update existing inputs with new template
    launcher                    Consolidate launcher scripts
    input.asc protocol [N]      Execute automated multi-stage workflow
    input.asc exclude [STAGE] [PATTERN]  Exclude structures from paused protocol

WORKFLOW:
  Typical manual workflow:
    1. ascec input.asc box      → validate simulation parameters
    2. ascec input.asc r3       → perform 3 replicated samplings
    3. ascec opt T.inp L.sh     → generate QM input files
    4. [Execute quantum evaluations externally]
    5. ascec sort               → organize results and generate summaries
    6. ascec simil --th=0.9     → cluster structures by similarity

  Automated workflow:
    ascec input.asc protocol    → executes all stages automatically

  Resuming the protocol:
    The system maintains a cache file to track progress. If interrupted,
    resume from the last successful stage:
    ascec input.asc protocol

    Resume from a specific stage (e.g., stage 2):
    ascec input.asc protocol 2       → resume from stage 2
    ascec input.asc protocol 2 -i    → resume interactively

    Pipeline stages:
    1. Annealing → 2. Optimization → 3. Similarity → 4. Refinement → 5. Similarity(2)

  Excluding problematic structures:
    For resumable protocols, exclude structures that cause errors:
            ascec input.asc exclude              → view current exclusions
            ascec input.asc exclude opt 5,10-15  → exclude 1st optimization stage
            ascec input.asc exclude ref 1,5,9    → exclude refinements
            ascec input.asc exclude clear         → clear all exclusions
            ascec input.asc exclude opt clear     → clear optimization exclusions only
        Then resume: ascec input.asc protocol

PROTOCOL FLAGS (optional):
  These flags modify stage behavior. Defaults shown in [brackets].

    Optimization/Refinement stages:
    --critical=N    Max %% of "critical" structures (imaginary freqs) [0]
    --redo=N        Max stage redos (re-run all failed structures) [3]
    --skipped=N     Max %% of skipped structures (use instead of --critical)

  Launch failure handling:
    Runs that fail to start (instant crashes) are automatically
    retried up to 10 times. Once a run starts running normally,
    it will not be retried regardless of exit code.

  Example protocol with custom flags:
    opt template.inp launcher.sh , similarity --th=0.9 , \\
    opt --critical=5 template.inp launcher.sh

  Launcher file is OPTIONAL:
    If omitted, ASCEC auto-detects ORCA from PATH and generates a launcher.
    Example: opt template.inp , similarity --th=0.9

TEMPLATE DIRECTIVES:
  Add these comments to your ORCA template (.inp) for rescue hessian behavior:

    #rescue(METHOD)       Specify method for rescue Hessian evaluation
  #rescue(METHOD,num)   Use NumFreq instead of Freq for Hessian

  Default rescue method: HF-3c with Freq
  
  If template uses xTB method (e.g., ! Native-GFN2-xTB Opt), the same
  method is reused for rescue with NumFreq automatically.

  Examples:
    #rescue(B97-3c)         Use B97-3c with Freq
    #rescue(GFN2-xTB,num)   Use GFN2-xTB with NumFreq
    #rescue(HF-3c)          Explicit HF-3c (same as default)

  xTB methods are auto-converted for ORCA version:
    ORCA 6.1+: Uses Native-GFN2-xTB, Native-GFN-xTB
    ORCA 5.x:  Uses GFN2-xTB, GFN-xTB (requires external xtb)

OPTIONS:
  -v              Verbose level 1: show intermediate progress
  -v2             Verbose level 2: extended progress tracking
  -v3             Verbose level 3: maximum detail output
  --standard       Use standard Metropolis acceptance criterion
  --nobox          Disable generation of box-visualization XYZ files
  --nosum          Skip summary file generation in sort command
  --justsum        Generate summary only without sorting structures
  -V, --version    Display version information and exit

EXAMPLES:
  ascec w6.in r5                      Perform 5 replicated water hexamer runs
  ascec diagram --scaled              Generate auto-scaled energy diagrams
    ascec opt opt.inp launcher.sh       Create inputs with launcher script
  ascec sort                          Organize results with summary
  ascec simil --th=0.95 --rmsd=0.5    Cluster with 0.95 th similarity + RMSD

REFERENCE:
  For comprehensive documentation including theoretical background,
  installation instructions, and advanced usage, consult the user manual.

CITATION:
  If you use ASCEC in your research, please cite:
  Manuel, G.; Sara, G.; Albeiro, R. Universidad de Antioquia (2026)

MORE INFORMATION:
  Repository: https://github.com/manuel2gl/qft-ascec-similarity
  Support:    Química Física Teórica - Universidad de Antioquia
""")
    parser.add_argument("command", metavar="COMMAND", 
                       help="Input file or command (opt, ref, sort, simil, diagram, etc.)")
    parser.add_argument("arg1", nargs='?', default=None, metavar="ARG1",
                       help="Command-specific argument (e.g., template file, mode)")
    parser.add_argument("arg2", nargs='?', default=None, metavar="ARG2",
                       help="Additional command-specific argument")
    parser.add_argument("-v", action="count", default=0,
                       help="Increase verbosity level (use -v, -v2, -v3, etc.)")
    parser.add_argument("--standard", action="store_true", 
                       help="Use standard Metropolis criterion instead of modified")
    
    # Sort command arguments (used when command='sort')
    parser.add_argument("--nosum", action="store_true", 
                       help="Skip summary file generation during sort")
    parser.add_argument("--justsum", action="store_true", 
                       help="Generate summary file only without sorting structures")
    # Internal arguments (hidden from help)
    parser.add_argument("--target-sim-folder", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reuse-existing", action="store_true", help=argparse.SUPPRESS)
    
    parser.add_argument("--nobox", action="store_true", 
                       help="Disable generation of box-visualization XYZ files")
    parser.add_argument("-V", "--version", action="store_true", 
                       help="Display version information and exit")
    
    # Use parse_known_args to handle shell expansion gracefully
    args, unknown_args = parser.parse_known_args()
    
    # Check if version is requested
    if getattr(args, 'version', False):
        print_version_banner("ASCEC")
        return
    
    # Check if help is requested
    if args.command.lower() in ["help", "commands"]:
        parser.print_help()
        return
    
    # Check if similarity analysis mode is requested
    if args.command.lower() == "simil":
        # Pass all remaining arguments to similarity script
        similarity_args = sys.argv[2:]  # Skip 'ascec-v04.py' and 'simil'
        execute_similarity_analysis(*similarity_args)
        return
    
    # Check if sort mode is requested
    if args.command.lower() == "sort":
        if args.justsum:
            execute_summary_only()
        else:
            # Check for target similarity folder argument
            target_sim = getattr(args, 'target_sim_folder', None)
            reuse = getattr(args, 'reuse_existing', False)
            execute_sort_command(include_summary=not args.nosum, target_sim_folder=target_sim, reuse_existing=reuse)
        return
    
    # Check if box analysis mode is requested
    if args.command.lower() == "box":
        if args.arg1:
            # Use specified input file
            input_file = args.arg1
        else:
            print("Error: box command requires an input file.")
            print("Usage: python ascec-v04.py box input_file.inp")
            print("   or: python ascec-v04.py box input_file.inp > box_info.txt")
            sys.exit(1)
        
        execute_box_analysis(input_file)
        return
    
    # Check if optimization-input generation mode is requested
    if args.command.lower() == "opt":
        if not args.arg1:
            print("Error: opt command requires a template file.")
            print("Usage: python ascec-v04.py opt template_file [launcher_template]")
            print("Example: python ascec-v04.py opt example_input.inp launcher_orca.sh")
            print("         python ascec-v04.py opt example_input.inp  # Creates inputs only")
            sys.exit(1)
        
        result = create_simple_optimization_system(args.arg1, args.arg2)
        if result:
            print(result)
        return

    # Check if refinement-input generation mode is requested
    # Refinement-input generation command
    if args.command.lower() in ["ref", "refinement"]:
        if not args.arg1:
            print("Error: ref command requires a template file.")
            print("Usage: python ascec-v04.py ref template_file [launcher_template]")
            print("Example: python ascec-v04.py ref example_input.inp launcher_orca.sh")
            print("         python ascec-v04.py ref example_input.inp  # Creates inputs only")
            sys.exit(1)
        
        result = create_refinement_system(args.arg1, args.arg2)
        if result:
            print(result)
        return
    
    # Check if merge mode is requested
    if args.command.lower() == "merge":
        if args.arg1 and args.arg1.lower() == "result":
            execute_merge_result_command()
        else:
            execute_merge_command()
        return
    
    # Check if update mode is requested
    if args.command.lower() == "update":
        if not args.arg1:
            print("Error: update command requires a template file.")
            print("Usage: python ascec-v04.py update new_template.inp")
            print("   or: python ascec-v04.py update new_template.inp pattern")
            print("This will search for files with the same extension as the template")
            sys.exit(1)
        
        # Handle shell expansion issue - if we have unknown arguments, it means shell expanded *
        if unknown_args:
            # If we have unknown arguments, it means shell expanded * 
            print(f"Detected shell expansion (found {len(unknown_args) + 2} total arguments). Switching to interactive mode...")
            target_pattern = ""
        elif args.arg2 is None:
            # If no second argument, default to interactive selection for same extension
            target_pattern = ""
        else:
            target_pattern = args.arg2
        
        result = update_existing_input_files(args.arg1, target_pattern)
        print(result)
        return
    
    # Check if cleanup mode is requested
    if args.command.lower() == "cleanup":
        print("Cleaning up temporary folders...")
        
        # Legacy _tmp_ folders (no longer created, but might exist from old runs)
        temp_calc_folders = glob.glob("calculation_tmp_*")
        temp_sim_folders = glob.glob("similarity_tmp_*")
        
        # Current temporary folders
        retry_input = ["retry_input"] if os.path.exists("retry_input") else []
        good_structures = ["good_structures"] if os.path.exists("good_structures") else []
        
        all_temp = temp_calc_folders + temp_sim_folders + retry_input + good_structures
        
        # Refactored optimization redo logic:
        # Instead of creating _tmp_ folders, we now unsort existing folders.
        # This means the "optimization redo block" is effectively removed,
        # and the cleanup now just handles the remaining temporary folders.
        # The "similarity spacing" fix is applied here as a general print spacing.
        
        if all_temp:
            if temp_calc_folders or temp_sim_folders:
                print("\n  Note: Found legacy _tmp_ folders from old runs")
                print("  (Current redo logic unsorts folders instead of creating _tmp_ copies)\n")
            
            for folder in all_temp:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    print(f"  Removed: {folder}")
            print(f"\n✓ Cleaned {len(all_temp)} temporary folder(s)")
        else:
            print("  No temporary folders found")
        return
    
    # Check if launcher merge mode is requested
    if args.command.lower() == "launcher":
        # Merge all launcher scripts in current directory and subfolders
        merge_launcher_scripts(".")
        return
    
    # Check if diagram generation mode is requested
    if args.command.lower() == "diagram":
        # Check for --scaled flag in arg1 or unknown_args
        scaled = False
        if args.arg1 and args.arg1.lower() == "--scaled":
            scaled = True
        elif unknown_args and "--scaled" in [arg.lower() for arg in unknown_args]:
            scaled = True
        
        execute_diagram_generation(scaled=scaled)
        return
    
    # For simulation mode, the command is the input file
    input_file = args.command
    replication = args.arg1
    
    # Check if box analysis is requested as second argument
    if replication is not None and replication.lower() == "box":
        execute_box_analysis(input_file)
        sys.exit(0)  # Explicitly exit after box analysis to prevent any further execution
        
    # Check if replication mode is requested
    if replication is not None:
        # Parse replication argument (e.g., "r3" -> 3 replicas)
        if replication.lower().startswith('r') and len(replication) > 1:
            try:
                num_replicas = int(replication[1:])
                if num_replicas <= 0:
                    raise ValueError("Number of replicas must be positive")
                
                # Check if there's a --box flag in remaining arguments (positional arg2, unknown_args, or raw argv)
                box_size_override = None
                candidates = []
                if args.arg2 is not None:
                    candidates.append(args.arg2)
                if unknown_args:
                    candidates.extend(unknown_args)
                # Also check raw argv as a last resort
                candidates.extend(sys.argv[1:])

                for candidate in candidates:
                    try:
                        if isinstance(candidate, str) and candidate.lower().startswith('--box'):
                            packing_str = candidate.lower().replace('--box', '')
                            if packing_str:
                                packing_percent = float(packing_str)
                                # Get box size recommendation for this packing percentage
                                recommended_box = get_box_size_recommendation(input_file, packing_percent)
                                if recommended_box is not None:
                                    box_size_override = recommended_box
                                    print(f"Using recommended box size: {box_size_override:.1f} Å ({packing_percent}% effective packing)")
                                else:
                                    print(f"Warning: Could not determine box size for {packing_percent}% packing. Using original box size.")
                                break
                            else:
                                print("Warning: Invalid --box flag format. Expected --box<number> (e.g., --box10)")
                    except ValueError:
                        print(f"Warning: Could not parse packing percentage from '{candidate}'. Using original box size.")
                    except Exception:
                        continue
                
                # Create replicated runs and exit
                create_replicated_runs(input_file, num_replicas, box_size=box_size_override)
                return
                
            except ValueError as e:
                print(f"Error: Invalid replication argument '{replication}'. {e}")
                print("Usage: python ascec-v04.py input_file.in r<number> [--box<percentage>]")
                print("Example: python ascec-v04.py example.in r3")
                print("Example: python ascec-v04.py example.in r3 --box10  # Use 10% packing")
                sys.exit(1)
        else:
            print(f"Error: Invalid replication format '{replication}'.")
            print("Usage: python ascec-v04.py input_file.in r<number> [--box<percentage>]")
            print("Example: python ascec-v04.py example.in r3")
            print("Example: python ascec-v04.py example.in r3 --box10  # Use 10% packing")
            sys.exit(1)

    # Initialize file handles and paths to None to prevent UnboundLocalError
    out_file_handle = None 
    failed_initial_configs_xyz_handle = None 
    failed_configs_path = None 
    
    rless_file_path = None 
    tvse_file_path = None  
    xyz_filename_base = "" # Initialize for error path
    rless_filename = ""    # Initialize for error path
    tvse_filename = ""     # Initialize for error path
    initial_failed_config_idx = 0  # Initialize for failed config tracking
    
    # Determine the directory where the input file is located
    input_file_path_full = os.path.abspath(input_file)
    run_dir = os.path.dirname(input_file_path_full)
    if not run_dir:
        run_dir = os.getcwd()

    qm_files_to_clean: List[str] = [] 
    initial_qm_successful = False 
    
    start_time = time.time() # Start wall time measurement

    state = SystemState()
    
    # Print startup message early to confirm script is running
    _print_verbose(f"{ASCEC_VERSION}", 0, state)
    _print_verbose("Starting ASCEC simulation...", 0, state)
    
    # Set verbosity level based on command line arguments
    # args.v is now an integer (0, 1, 2, 3...) from action="count" in argparse
    state.verbosity_level = args.v if hasattr(args, 'v') else 0
    
    state.use_standard_metropolis = args.standard # Set the flag in state
    
    # Handle --nobox flag to disable box XYZ file creation
    global create_box_xyz_copy
    if args.nobox:
        create_box_xyz_copy = False

    # Check for Open Babel executable early
    if not shutil.which("obabel"):
        _print_verbose("\nCRITICAL ERROR: Open Babel executable 'obabel' not found in your system's PATH.", 0, state)
        _print_verbose("Please ensure Open Babel is installed and its executable is accessible from your command line.", 0, state)
        _print_verbose("Cannot proceed with .mol file generation.", 0, state)
    
    try:
        # Call read_input_file as early as possible to populate state
        # Add file existence check to prevent blocking on missing files
        if not os.path.exists(input_file):
            _print_verbose(f"\nCRITICAL ERROR: Input file '{input_file}' not found.", 0, state)
            _print_verbose("Please check the file path and ensure the file exists.", 0, state)
            sys.exit(1)
        
        # Add file readability check
        try:
            with open(input_file, 'r') as test_file:
                test_file.readline()  # Try to read first line
        except (PermissionError, IOError) as e:
            _print_verbose(f"\nCRITICAL ERROR: Cannot read input file '{input_file}': {e}", 0, state)
            _print_verbose("Please check file permissions and ensure the file is not locked.", 0, state)
            sys.exit(1)
            
        read_input_file(state, input_file)

        # Set output directory to the directory containing the input file
        state.output_dir = run_dir

        # Call for Box Length Advice (only for simulation mode, not box analysis)
        provide_box_length_advice(state) # This will print to stderr

        # Set QM program name based on qm_program_details mapping
        state.qm_program = qm_program_details[state.ia]["name"]
        
        # Initialize parallel execution environment optimization
        optimize_qm_execution_environment(state)

        # If random seed is not explicitly set in input or invalid, generate one
        if state.random_seed == -1: 
            state.random_seed = random.randint(100000, 999999) # Generate a 6-digit integer
            _print_verbose(f"\nUsing random seed: {state.random_seed}\n", 0, state) # Modified print
        else:
             _print_verbose(f"\nUsing random seed: {state.random_seed}\n", 0, state) # Modified print
        
        # Initialize Python's random and NumPy's random with the seed
        random.seed(state.random_seed)
        np.random.seed(state.random_seed)

        input_base_name = os.path.splitext(os.path.basename(input_file))[0]

        # Define output filenames
        out_filename = f"{input_base_name}.out" 
        xyz_filename_base = f"result_{state.random_seed}" if state.random_generate_config == 1 else f"mto_{state.random_seed}"
        rless_filename = f"rless_{state.random_seed}.out"
        tvse_filename = f"tvse_{state.random_seed}.dat"

        out_file_path = os.path.join(run_dir, out_filename)
        rless_file_path = os.path.join(run_dir, rless_filename) # Full path defined here
        tvse_file_path = os.path.join(run_dir, tvse_filename)   # Full path defined here

        # Define and Open .out File
        try:
            out_file_handle = open(out_file_path, 'w')
            _print_verbose(f"Main output will be written to: {out_file_path}\n", 0, state)
        except IOError as e:
            _print_verbose(f"CRITICAL ERROR: Could not open out file '{out_file_path}': {e}", 0, state)
            sys.exit(1)

        # Write Simulation Summary to .out File (passing dynamic filenames)
        write_simulation_summary(state, out_file_handle, xyz_filename_base + ".xyz", rless_filename, tvse_filename) # Pass with .xyz for display
        out_file_handle.flush() 

        # For annealing mode, open a file for failed initial configurations
        if state.random_generate_config == 1: 
            failed_configs_filename = os.path.splitext(os.path.basename(input_file))[0] + "_failed_initial_configs.xyz"
            failed_configs_path = os.path.join(run_dir, failed_configs_filename)
            try:
                # Open in write mode if it needs to be created or overwritten
                failed_initial_configs_xyz_handle = open(failed_configs_path, 'w')
                _print_verbose(f"Failed initial configurations will be logged to: {failed_configs_path}\n", 1, state)
            except IOError as e:
                _print_verbose(f"Warning: Could not open file for failed initial configs '{failed_configs_path}': {e}. Continuing without logging these.", 0, state)
                failed_initial_configs_xyz_handle = None # Ensure it's None if opening failed

            _print_verbose("Attempting initial QM energy calculation...\n", 0, state) 
            initial_failed_config_idx = 0 

        # Define half_xbox once for coordinate system shift for visualization
        half_xbox = state.xbox / 2.0

        # Random Configuration Generation Mode (no QM, so files are opened directly)
        if state.random_generate_config == 0: 
            _print_verbose(f"Generating {state.num_random_configs} random configurations (no energy evaluation).\n", 0, state)
            
            # Use total_accepted_configs as the sequential counter for random configs
            for i in range(state.num_random_configs):
                state.total_accepted_configs += 1 # Increment for each random config
                # Re-initialize rp and iznu for each new random configuration
                # initialize_molecular_coords_in_box generates random configuration using config_molecules
                state.rp, state.iznu = initialize_molecular_coords_in_box(state) 
                
                # Write to the ORIGINAL XYZ file handle (no dummy atoms)
                write_accepted_xyz(
                    xyz_filename_base, 
                    state.total_accepted_configs, # Pass the incremented count
                    0.0, # Energy is 0.0 as it's not evaluated in this mode
                    state.current_temp, # Temp not relevant for MTO, but passed for consistency
                    state,
                    is_initial=False 
                )
            _print_verbose("Random configuration generation complete.\n", 0, state)
            
            # Add the final output for random mode here
            with open(out_file_path, 'a') as out_file_handle_local:
                out_file_handle_local.write("\n") # Explicit blank line before the final summary separator
                out_file_handle_local.write("=" * 60 + "\n") 
                out_file_handle_local.write("\n  ** Normal random generation termination **\n") # Corrected termination message
                end_time = time.time() # End wall time measurement
                total_wall_time_seconds = end_time - start_time
                time_delta = timedelta(seconds=int(total_wall_time_seconds))
                days = time_delta.days
                hours = time_delta.seconds // 3600
                minutes = (time_delta.seconds % 3600) // 60
                seconds = time_delta.seconds % 60
                milliseconds = int((total_wall_time_seconds - int(total_wall_time_seconds)) * 1000)
                out_file_handle_local.write(f"  Total Wall time: {days} days {hours} h {minutes} min {seconds} s {milliseconds} ms\n")
                out_file_handle_local.write("\n" + "=" * 60 + "\n")


        # Annealing Mode (state.random_generate_config = 1)
        elif state.random_generate_config == 1: 
            
            # Set initial temperature
            state.current_temp = state.linear_temp_init if state.quenching_routine == 1 else state.geom_temp_init

            # Initialize last_accepted_qm_call_count and last_history_qm_call_count before the first QM call
            state.last_accepted_qm_call_count = 0 
            state.last_history_qm_call_count = 0 

            # Initial QM Calculation Loop (with retries)
            initial_qm_calculation_succeeded = False 
            for attempt in range(state.initial_qm_retries):
                _print_verbose(f"  Attempt {attempt + 1}/{state.initial_qm_retries} for initial QM calculation.", 0, state)

                # Generate a new random initial configuration for each retry
                state.rp, state.iznu = initialize_molecular_coords_in_box(state)
                
                initial_failed_config_idx += 1 

                # Log initial configuration (shifted for visualization) to the failed configs file
                if failed_initial_configs_xyz_handle:
                    rp_for_viz_failed = state.rp + half_xbox 
                    # Re-using write_single_xyz_configuration for failed configs as it doesn't need full accepted_xyz logic
                    write_single_xyz_configuration(
                        failed_initial_configs_xyz_handle, 
                        state.natom, rp_for_viz_failed, state.iznu, 0.0, # Energy is 0.0 as it's not yet calculated or failed
                        initial_failed_config_idx, state.atomic_number_to_symbol,
                        state.random_generate_config, 
                        remark=f"Initial Setup Attempt {attempt + 1}", 
                        include_dummy_atoms=True, # Always include dummy atoms for failed initial configs
                        state=state 
                    )
                
                try:
                    initial_energy, jo_status = calculate_energy(state.rp, state.iznu, state, run_dir)
                    
                    if jo_status == 0:
                        raise RuntimeError("QM program returned non-zero status or could not calculate energy for initial configuration.") 

                    state.current_energy = initial_energy
                    state.lowest_energy = initial_energy 
                    state.lowest_energy_rp = np.copy(state.rp) # Store initial lowest energy coords
                    state.lowest_energy_iznu = list(state.iznu) # Store initial lowest energy atomic numbers
                    state.lowest_energy_config_idx = 1 # Initial config is always config 1
                    state.lower_energy_configs = 1 # Initial config counts as the first lower energy config
                    
                    # Preserve QM files from the initial accepted configuration for debugging
                    preserve_last_qm_files(state, run_dir)
                    
                    _print_verbose(f"  Calculation successful. Energy: {state.current_energy:.8f} a.u.\n", 0, state) # Modified print
                    
                    # Print conformational sampling information
                    if state.conformational_move_prob > 0.0:
                        _print_verbose(f"Conformational sampling enabled: {state.conformational_move_prob*100:.1f}% probability", 0, state)
                        _print_verbose(f"  Maximum dihedral rotation: ±{np.degrees(state.max_dihedral_angle_rad):.1f}°", 0, state)
                        _print_verbose(f"  Move types: {state.conformational_move_prob*100:.1f}% conformational (dihedral rotations), {(1-state.conformational_move_prob)*100:.1f}% rigid-body (translation+rotation)", 0, state)
                        _print_verbose(f"  Note: Conformational moves with atom overlaps automatically fall back to rigid-body moves", 0, state)
                    else:
                        _print_verbose("Using rigid-body moves only (translation + rotation)", 0, state)
                    
                    _print_verbose("\nStarting annealing simulation...\n", 0, state)
                    initial_qm_calculation_succeeded = True
                    initial_qm_successful = True  # Also set this flag for cleanup logic
                    state.total_accepted_configs += 1 # Increment for initial accepted config
                    break # Exit retry loop on success

                except RuntimeError as e:
                    _print_verbose(f"  Initial QM calculation attempt {attempt + 1} failed: {e}", 0, state)
                    
                    if attempt < state.initial_qm_retries - 1:
                        _print_verbose("  Generating new initial configuration and retrying...\n", 0, state)
                    else:
                        # Preserve failed QM files from the last attempt only
                        preserve_failed_initial_qm_files(state, run_dir, attempt + 1)
                        raise RuntimeError(
                            f"All {state.initial_qm_retries} attempts to perform the initial QM energy calculation failed. "
                            "Please verify your QM input parameters (method, basis set, memory, processors) "
                            "or inspect the run directory for specific QM program output errors. "
                            "Cannot proceed with annealing simulation."
                        )
                except Exception as e:
                    _print_verbose(f"  An unexpected error occurred during initial QM attempt {attempt + 1}: {e}", 0, state)
                    raise # Re-raise other unexpected errors

            if not initial_qm_calculation_succeeded:
                # If initial QM failed, proceed to generate MOL for failed configs
                if failed_initial_configs_xyz_handle:
                    try: failed_initial_configs_xyz_handle.close() # Close before attempting conversion
                    except Exception as e: _print_verbose(f"Error closing failed initial configs file during final cleanup: {e}", 0, state)
                    
                    if failed_configs_path and os.path.exists(failed_configs_path):
                        failed_mol_filename = os.path.splitext(failed_configs_path)[0] + ".mol"
                        try:
                            _print_verbose(f"Attempting to convert failed initial configs XYZ '{os.path.basename(failed_configs_path)}' to MOL...", 1, state)
                            if convert_xyz_to_mol(failed_configs_path, state.openbabel_alias, state):
                                _post_process_mol_file(failed_mol_filename, state)
                                _print_verbose(f"Successfully created and post-processed '{os.path.basename(failed_mol_filename)}'.", 1, state)
                            else:
                                _print_verbose(f"Failed to create '{os.path.basename(failed_mol_filename)}'. See previous warnings.", 1, state)
                        except Exception as e:
                            _print_verbose(f"Warning: Could not create or post-process .mol file for failed initial configs '{os.path.basename(failed_configs_path)}': {e}", 0, state)

                raise RuntimeError("Initial QM energy calculation did not succeed after all retries. Exiting.")

            # If initial QM succeeded, we can remove the failed initial configs file
            if failed_initial_configs_xyz_handle and failed_configs_path and os.path.exists(failed_configs_path):
                try:
                    failed_initial_configs_xyz_handle.close() # Close before removing
                    os.remove(failed_configs_path)
                    _print_verbose(f"Removed '{failed_configs_path}' as initial QM calculation succeeded.\n", 1, state)
                except OSError as e:
                    _print_verbose(f"Error removing '{failed_configs_path}': {e}", 0, state)
                finally:
                    failed_initial_configs_xyz_handle = None # Clear handle

            # Initial calculation of all molecular centers of mass
            # This is crucial for the first call to config_move
            for i in range(state.num_molecules):
                mol_start_idx = state.imolec[i]
                mol_end_idx = state.imolec[i+1]
                mol_atomic_numbers = [state.iznu[j] for j in range(mol_start_idx, mol_end_idx)]
                mol_masses = np.array([state.atomic_number_to_mass.get(anum, 1.0) for anum in mol_atomic_numbers])
                state.rcm[i, :] = calculate_mass_center(state.rp[mol_start_idx:mol_end_idx, :], mol_masses)

            # Determine total annealing steps
            if state.quenching_routine == 1: 
                total_annealing_steps = state.linear_num_steps
            elif state.quenching_routine == 2: 
                total_annealing_steps = state.geom_num_steps
            else:
                _print_verbose("Error: Invalid quenching_routine specified for annealing mode. Must be 1 (linear) or 2 (geometric).", 0, state)
                return 

            # Add the initial successful QM calculation to history and tvse
            if initial_qm_calculation_succeeded:
                # n-eval for initial entry is 0 as per user's desired output format
                history_n_eval = 0 
                accepted_history_entry = {
                    "T": state.current_temp, # Initial temp
                    "E": state.current_energy,
                    "n_eval": history_n_eval, 
                    "criterion": "Intv"
                }
                # Write to .out file history immediately (adjusted spacing)
                with open(out_file_path, 'a') as f_out: # Open in append mode
                    # Added two spaces to n-eval column
                    f_out.write(f"  {accepted_history_entry['T']:>8.2f} {accepted_history_entry['E']:>12.6f} {accepted_history_entry['n_eval']:>7} {accepted_history_entry['criterion']:>8}\n")
                
                # Update last_history_qm_call_count after writing the initial entry
                state.last_history_qm_call_count = state.qm_call_count

                # Write to .dat file (tvse) immediately
                write_tvse_file(tvse_file_path, {
                    "n_eval": state.qm_call_count, # Use total QM calls for TVSE
                    "T": state.current_temp,
                    "E": state.current_energy
                }, state)
                
                _print_verbose(f"  Initial accepted config added to history. Initial T: {state.current_temp:.2f} K, E: {state.current_energy:.8f} a.u., QM calls: {state.qm_call_count}", 1, state)
                
                # IMPORTANT: Update last_accepted_qm_call_count after the initial accepted config
                state.last_accepted_qm_call_count = state.qm_call_count 

            # Write the first *successful* QM configuration to the main XYZ file(s)
            write_accepted_xyz(
                xyz_filename_base, 
                state.total_accepted_configs, 
                state.current_energy, 
                state.current_temp, 
                state,
                is_initial=True 
            )

            # Main Annealing Loop
            for step_num in range(total_annealing_steps):
                # Apply temperature quenching BEFORE the MC cycles for this step to use the correct temperature
                if step_num > 0: # Don't quench for the very first step, as current_temp is already init_temp
                    if state.quenching_routine == 1: # Linear quenching
                        state.current_temp = max(state.current_temp - state.linear_temp_decrement, 0.001) 
                    elif state.quenching_routine == 2: # Geometric quenching
                        state.current_temp = max(state.current_temp * state.geom_temp_factor, 0.001)

                # Always print the start of a new annealing step if not very verbose (level 0 or 1)
                if state.verbosity_level <= 1:
                    print(f"\nAnnealing Step {step_num + 1}/{total_annealing_steps} at Temperature: {state.current_temp:.2f} K", file=sys.stderr) # Modified print
                    sys.stderr.flush()
                elif state.verbosity_level == 2: # For very verbose, print every step
                     _print_verbose(f"\nAnnealing Step {step_num + 1}/{total_annealing_steps} at Temperature: {state.current_temp:.2f} K", 2, state) # Modified print
                
                # Keep reference energy of the current accepted state for Metropolis criterion.
                original_energy_for_revert = state.current_energy

                qm_calls_made_this_temp_step = 0 # Counter for QM calls within this temperature step
                
                # Flag to control moving to the next temperature step
                # Set to True if LwE is accepted, or if maxstep is reached for this temperature
                should_move_to_next_temperature = False

                # Run Monte Carlo cycles at current temperature, up to maxstep evaluations
                while qm_calls_made_this_temp_step < state.maxstep:
                    qm_calls_made_this_temp_step += 1 # Increment QM calls for this temperature step
                    
                    # PROPOSE NEW CONFIGURATION
                    # The propose_unified_move function returns proposed coordinates
                    # without modifying the state - we test them first, then accept/reject
                    proposed_rp_full, _, _, _ = propose_unified_move(
                        state, state.rp, state.imolec
                    )

                    # Evaluate proposed geometry directly to avoid extra state copies on rejected moves.
                    # This will use parallel cores within the QM calculation itself
                    proposed_energy, jo_status = calculate_energy(proposed_rp_full, state.iznu, state, run_dir)
                    
                    # Verbose output for each cycle (using state.qm_call_count for global attempt number)
                    if state.verbosity_level == 2 or (state.verbosity_level == 1 and state.qm_call_count % 10 == 0):
                        _print_verbose(f"  Attempt {state.qm_call_count} (global), {qm_calls_made_this_temp_step} (step-local): T={state.current_temp:.2f} K", 1, state)

                    # Check if QM calculation failed
                    if jo_status == 0:
                        _print_verbose(f"  Warning: Proposed QM energy calculation failed for global attempt {state.qm_call_count}. Rejecting move.", 1, state)
                        continue # Continue to next MC cycle attempt

                    accept_move = False
                    # Compare proposed energy with the energy of the *original* (last accepted) configuration
                    delta_e = proposed_energy - original_energy_for_revert 
                    
                    criterion_str = "" # Reset criterion string for each attempt

                    if delta_e <= 0.0: # If proposed energy is lower or equal, always accept
                        accept_move = True
                        criterion_str = "LwE" 
                        state.lower_energy_configs += 1  # Increment counter for lower energy configs 
                    else: # If proposed energy is higher, apply Metropolis criterion
                        # Ensure temperature is not zero for division
                        if state.current_temp < 1e-6: 
                            pE = 0.0
                        else:
                            pE = math.exp(-delta_e / (B2 * state.current_temp))
                        
                        if state.use_standard_metropolis: # Standard Metropolis
                            if random.random() < pE:
                                accept_move = True
                                criterion_str = "EMpol"
                                state.iboltz += 1
                        else: # Modified Metropolis (default)
                            # Handle division by zero for crt (if proposed_energy is zero)
                            if abs(proposed_energy) < 1e-12: # Avoid division by near-zero energy
                                crt = float('inf') 
                            else:
                                crt = delta_e / abs(proposed_energy)

                            if crt < pE: # This is the modified criterion
                                accept_move = True
                                criterion_str = "Mpol" 
                                state.iboltz += 1

                    if accept_move:
                        # If accepted, update the system's current state to the *proposed* one
                        state.rp[:] = proposed_rp_full[:]
                        state.current_energy = proposed_energy 
                        
                        # Preserve QM files from this accepted configuration for debugging
                        preserve_last_qm_files(state, run_dir)

                        # Update reference energy to the newly accepted state
                        original_energy_for_revert = state.current_energy

                        _print_verbose(f"  Attempt {state.qm_call_count} (global): Move accepted ({criterion_str}). New energy: {state.current_energy:.8f} a.u.", 1, state)
                    
                        # Update lowest energy found overall
                        if state.current_energy < state.lowest_energy:
                            state.lowest_energy = state.current_energy
                            state.lowest_energy_rp = np.copy(state.rp) 
                            state.lowest_energy_iznu = list(state.iznu) 
                            state.lowest_energy_config_idx = state.total_accepted_configs + 1 # Will be the number of this accepted config
                            _print_verbose(f"  New lowest energy found: {state.lowest_energy:.8f} a.u. (Config {state.lowest_energy_config_idx})", 1, state)
                        
                        # Calculate n-eval for .out history: QM calls since last history entry
                        history_n_eval = state.qm_call_count - state.last_history_qm_call_count
                        
                        with open(out_file_path, 'a') as f_out:
                            f_out.write(f"  {state.current_temp:>8.2f} {state.current_energy:>12.6f} {history_n_eval:>7} {criterion_str:>8}\n")
                        
                        # Update last_history_qm_call_count after writing this entry
                        state.last_history_qm_call_count = state.qm_call_count

                        write_tvse_file(tvse_file_path, {
                            "n_eval": state.qm_call_count, # Use total QM calls for TVSE (cumulative)
                            "T": state.current_temp,
                            "E": state.current_energy
                        }, state)
                        
                        # Update last_accepted_qm_call_count to the current global QM call count
                        state.last_accepted_qm_call_count = state.qm_call_count

                        state.total_accepted_configs += 1 
                        write_accepted_xyz(
                            xyz_filename_base, 
                            state.total_accepted_configs, 
                            state.current_energy, 
                            state.current_temp, 
                            state,
                            is_initial=False 
                        )
                        
                        # This is the key change for loop control:
                        if criterion_str == "LwE":
                            _print_verbose(f"  Lower energy accepted. Moving to next temperature step.", 1, state)
                            should_move_to_next_temperature = True
                            break # Break out of the inner for loop (Monte Carlo cycles)
                        elif criterion_str in ["Mpol", "EMpol"]:
                            _print_verbose(f"  Metropolis accepted. Continuing Monte Carlo cycles at current temperature.", 1, state)
                            # Do NOT set should_move_to_next_temperature = True, continue the loop until maxstep is reached
                        
                    else: # Move rejected
                        _print_verbose(f"  Attempt {state.qm_call_count} (global): Move rejected. Energy: {proposed_energy:.8f} a.u. (Current: {original_energy_for_revert:.8f})", 1, state)
                        # No revert needed because state.rp was not mutated for rejected proposals.
                
                # After the inner loop (maxstep attempts or LwE break)
                # If we did NOT move to the next temperature due to LwE, it means maxstep was reached
                # or only Mpol/EMpol was accepted (and maxstep was reached).
                # In this case, if the last history entry was NOT the current QM call count,
                # it means there were unlogged QM calls since the last history entry.
                # This happens if Mpol/EMpol was accepted, or if no configuration was accepted.
                if not should_move_to_next_temperature and state.qm_call_count > state.last_history_qm_call_count:
                    # This means the inner loop completed because maxstep was reached,
                    # and either an Mpol/EMpol was accepted earlier in this loop (and a line was written),
                    # or no config was accepted at all.
                    # We need to write an N/A line to account for the remaining QM calls
                    # at the *current* temperature before it quenches.
                    history_n_eval = state.qm_call_count - state.last_history_qm_call_count
                    if history_n_eval > 0: # Only write if there were actual calls since last history entry
                        with open(out_file_path, 'a') as f_out:
                            f_out.write(f"  {state.current_temp:>8.2f} {original_energy_for_revert:>12.6f} {history_n_eval:>7} {'N/A':>8}\n") 
                        state.last_history_qm_call_count = state.qm_call_count
                    _print_verbose(f"  Max cycles reached for {state.current_temp:.2f} K. Moving to next temperature step.", 1, state)
                elif should_move_to_next_temperature:
                    # If LwE was accepted, we already broke and logged it.
                    pass # Nothing more to do here, the loop will naturally go to the next step_num

                # Dynamic max_cycle reduction
                state.maxstep = max(state.max_cycle_floor, int(state.maxstep * 0.90)) # Reduce by 10% with user-defined floor
                _print_verbose(f"  Max QM evaluations for next temperature step reduced to: {state.maxstep}", 1, state)

            _print_verbose("\nAnnealing simulation finished.\n", 0, state)
            _print_verbose(f"Final lowest energy found: {state.lowest_energy:.8f} a.u.", 0, state)
            _print_verbose(f"Total QM calculations performed: {state.qm_call_count}", 0, state)

            # Final summary to .out file for annealing mode
            with open(out_file_path, 'a') as out_file_handle_local: # Use a local alias for clarity
                out_file_handle_local.write("\n") # Explicit blank line before the final summary separator
                out_file_handle_local.write("=" * 60 + "\n") 
                out_file_handle_local.write("\n  ** Normal annealing termination **\n") # Added \n before
                end_time = time.time() # End wall time measurement
                total_wall_time_seconds = end_time - start_time
                time_delta = timedelta(seconds=int(total_wall_time_seconds))
                days = time_delta.days
                hours = time_delta.seconds // 3600
                minutes = (time_delta.seconds % 3600) // 60
                seconds = time_delta.seconds % 60
                milliseconds = int((total_wall_time_seconds - int(total_wall_time_seconds)) * 1000)
                out_file_handle_local.write(f"  Total Wall time: {days} days {hours} h {minutes} min {seconds} s {milliseconds} ms\n")
                out_file_handle_local.write(f"  Energy was evaluated {state.qm_call_count} times\n\n")
                out_file_handle_local.write(f"Energy evolution in {tvse_filename}\n")
                out_file_handle_local.write(f"Configurations accepted by Max.-Boltz. statistics = {state.iboltz}\n")
                out_file_handle_local.write(f"Accepted lower energy configurations = {state.lower_energy_configs}\n")
                # Updated to include total accepted configurations
                out_file_handle_local.write(f"Accepted configurations in {xyz_filename_base}.xyz = {state.total_accepted_configs}\n") 
                out_file_handle_local.write(f"Lowest energy configuration in {rless_filename}\n")
                
                # This will print the lowest energy found by the end of the simulation
                if state.lowest_energy_rp is not None:
                    out_file_handle_local.write(f"Lowest energy = {state.lowest_energy:.8f} u.a. (Config. {state.lowest_energy_config_idx})\n")
                else:
                    out_file_handle_local.write("Lowest energy = N/A (No configurations accepted)\n")
                
                out_file_handle_local.write("\n" + "=" * 60 + "\n")

    finally:
        # Ensure all output files are closed
        if out_file_handle:
            try: out_file_handle.close()
            except Exception as e: _print_verbose(f"Error closing main output file: {e}", 0, state)
        
        # Write lowest energy config file only if it was successfully found (already handled in annealing loop)
        if state.random_generate_config == 1 and state.lowest_energy_rp is not None and rless_file_path:
            write_lowest_energy_config_file(state, rless_file_path)
        elif state.random_generate_config == 1 and state.lowest_energy_rp is None:
            _print_verbose(f"No lowest energy configuration was successfully found and stored. Skipping rless file generation.", 0, state)

        # Handle MOL conversion for XYZ files (which were created by write_accepted_xyz)
        # We need to ensure paths are defined before attempting conversion
        _print_verbose(f"\nInitiating .mol file conversions", 1, state)

        # Only attempt mol conversion if obabel was found at startup
        if shutil.which("obabel"):
            main_xyz_path = os.path.join(run_dir, f"{xyz_filename_base}.xyz")
            box_xyz_path = os.path.join(run_dir, f"{xyz_filename_base.replace('mto_', 'mtobox_').replace('result_', 'resultbox_')}.xyz")

            if os.path.exists(main_xyz_path):
                _print_verbose(f"  Processing main XYZ file for .mol conversion: '{os.path.basename(main_xyz_path)}'", 1, state)
                mol_filename = os.path.splitext(main_xyz_path)[0] + ".mol"
                try:
                    if convert_xyz_to_mol(main_xyz_path, state.openbabel_alias, state):
                        _post_process_mol_file(mol_filename, state)
                        _print_verbose(f"  Successfully processed '{os.path.basename(mol_filename)}'.", 1, state)
                    else:
                        _print_verbose(f"  Failed to create '{os.path.basename(mol_filename)}'. Review Open Babel output above for details.", 1, state)
                except Exception as e:
                    _print_verbose(f"  Warning: Unexpected error during .mol creation for '{os.path.basename(main_xyz_path)}': {e}", 0, state)

            if create_box_xyz_copy and os.path.exists(box_xyz_path):
                _print_verbose(f"  Processing box XYZ file for .mol conversion: '{os.path.basename(box_xyz_path)}'", 1, state)
                mol_box_filename = os.path.splitext(box_xyz_path)[0] + ".mol"
                try:
                    if convert_xyz_to_mol(box_xyz_path, state.openbabel_alias, state):
                        _post_process_mol_file(mol_box_filename, state)
                        _print_verbose(f"  Successfully processed '{os.path.basename(mol_box_filename)}'.", 1, state)
                    else:
                        _print_verbose(f"  Failed to create '{mol_box_filename}'. Review Open Babel output above for details.", 1, state)
                except Exception as e:
                    _print_verbose(f"  Warning: Unexpected error during .mol creation for '{os.path.basename(box_xyz_path)}': {e}", 0, state)
            
            # This block now handles MOL conversion for failed initial configs *only if* the initial QM ultimately failed.
            # The file is not removed if initial_qm_successful is False, so it persists for MOL conversion.
            if state.random_generate_config == 1 and not initial_qm_successful and failed_configs_path and os.path.exists(failed_configs_path):
                if failed_initial_configs_xyz_handle:
                    try: failed_initial_configs_xyz_handle.close() # Ensure it's closed before MOL conversion
                    except Exception as e: _print_verbose(f"Error closing failed initial configs XYZ file: {e}", 0, state)
                
                _print_verbose(f"  Processing failed initial configs XYZ for .mol conversion: '{os.path.basename(failed_configs_path)}'", 1, state)
                failed_mol_filename = os.path.splitext(failed_configs_path)[0] + ".mol"
                try:
                    if convert_xyz_to_mol(failed_configs_path, state.openbabel_alias, state):
                        _post_process_mol_file(failed_mol_filename, state)
                        _print_verbose(f"  Successfully processed '{os.path.basename(failed_mol_filename)}'.", 1, state)
                    else:
                        _print_verbose(f"  Failed to create '{os.path.basename(failed_mol_filename)}'. Review Open Babel output above for details.", 1, state)
                except Exception as e:
                    _print_verbose(f"  Warning: Unexpected error during .mol creation for failed initial configs '{os.path.basename(failed_configs_path)}': {e}", 0, state)
        else:
            _print_verbose(f"  Skipping .mol file conversions because Open Babel was not found in your PATH.", 1, state)
        
        _print_verbose(f"\n.mol file conversions completed", 1, state)

        # Generate annealing diagrams if this was an annealing simulation
        if state.random_generate_config == 1 and tvse_file_path and os.path.exists(tvse_file_path):
            # Check if we're in workflow mode (don't print messages if so)
            in_workflow = hasattr(sys, '_current_workflow_context') and sys._current_workflow_context is not None  # type: ignore[attr-defined]
            
            if MATPLOTLIB_AVAILABLE:
                if not in_workflow:
                    _print_verbose(f"\nGenerating annealing diagrams...", 0, state)
                if plot_annealing_diagrams(tvse_file_path, run_dir):
                    if not in_workflow:
                        diagram_file = os.path.join(run_dir, f"tvse_{state.random_seed}.png")
                        _print_verbose(f"  ✓ Created: {os.path.basename(diagram_file)}", 0, state)
                else:
                    if not in_workflow:
                        _print_verbose(f"  ✗ Failed to generate diagrams", 1, state)
                
                # Check if this is part of a replicated run and generate combined diagram if all replicas are complete
                if not in_workflow:
                    parent_dir = os.path.dirname(run_dir)
                    parent_name = os.path.basename(parent_dir)
                    current_dir_name = os.path.basename(run_dir)
                    
                    # Check if we're in an "annealing" directory structure with replicated runs
                    # Pattern: annealing/basename_N/ where N is the replica number
                    if parent_name == "annealing" and '_' in current_dir_name:
                        # Find all sibling replica directories
                        try:
                            replica_dirs = []
                            replica_tvse_files = []
                            
                            # Extract base name (everything before the last underscore and number)
                            dir_match = re.match(r'(.+)_(\d+)$', current_dir_name)
                            if dir_match:
                                base_name = dir_match.group(1)
                                
                                # Find all directories matching the pattern
                                for entry in sorted(os.listdir(parent_dir)):
                                    entry_path = os.path.join(parent_dir, entry)
                                    if os.path.isdir(entry_path):
                                        # Check if directory matches pattern: basename_N
                                        if re.match(rf'{re.escape(base_name)}_\d+$', entry):
                                            replica_dirs.append(entry_path)
                                            # Look for tvse file in this directory
                                            tvse_found = glob.glob(os.path.join(entry_path, 'tvse_*.dat'))
                                            if tvse_found:
                                                replica_tvse_files.append(tvse_found[0])
                                
                                # If we found multiple replicas and all have tvse files, generate combined diagram
                                if len(replica_dirs) > 1 and len(replica_tvse_files) == len(replica_dirs):
                                    num_replicas = len(replica_dirs)
                                    combined_diagram = os.path.join(parent_dir, f"tvse_r{num_replicas}.png")
                                    
                                    _print_verbose(f"\nAll {num_replicas} replica(s) complete. Generating combined diagram...", 0, state)
                                    if plot_combined_replicas_diagram(replica_tvse_files, combined_diagram, num_replicas):
                                        _print_verbose(f"  ✓ Created: {os.path.basename(combined_diagram)}", 0, state)
                                    else:
                                        _print_verbose(f"  ✗ Failed to create combined diagram", 0, state)
                        except Exception as e:
                            _print_verbose(f"  Warning: Combined diagram check failed: {type(e).__name__}: {e}", 0, state)
            else:
                if not in_workflow:
                    _print_verbose(f"\nSkipping diagram generation (matplotlib not available)", 1, state)

        # Final cleanup of any lingering QM files (should be minimal with per-call cleanup)
        cleanup_qm_files(qm_files_to_clean, state) 


def analyze_box_length_from_xyz(xyz_file: str, num_molecules: int = 1) -> None:
    """
    Standalone function to analyze optimal box lengths from an existing XYZ file.
    Useful for testing the new volume-based approach.
    
    Args:
        xyz_file (str): Path to XYZ file containing molecular structure
        num_molecules (int): Number of copies of this molecule that will be simulated
    """
    import sys
    
    print(f"\nAnalyzing box length requirements for: {xyz_file}")
    print(f"Number of molecule copies: {num_molecules}")
    print("="*70)
    
    if not os.path.exists(xyz_file):
        print(f"Error: File '{xyz_file}' not found!")
        return
    
    try:
        # Read the XYZ file
        configurations = extract_configurations_from_xyz(xyz_file)
        
        if not configurations:
            print("Error: No valid configurations found in XYZ file!")
            return
        
        # Use the first configuration
        config = configurations[0]
        print(f"Using configuration {config['config_num']} with {len(config['atoms'])} atoms")
        
        # Convert to MoleculeData format
        atoms_coords = []
        for atom in config['atoms']:
            if len(atom) >= 7:  # New format with string and float coordinates
                symbol, x_str, y_str, z_str, x_float, y_float, z_float = atom
                # Look up atomic number from symbol
                atomic_num = element_symbols.get(symbol)
                
                if atomic_num is None:
                    print(f"Warning: Unknown element symbol '{symbol}', skipping atom")
                    continue
                
                atoms_coords.append((atomic_num, x_float, y_float, z_float))
            else:
                print(f"Warning: Unexpected atom format: {atom}")
                continue
        
        if not atoms_coords:
            print("Error: No valid atoms found!")
            return
        
        # Create a MoleculeData object
        mol_data = MoleculeData("molecule", len(atoms_coords), atoms_coords)
        
        # Create a minimal SystemState for the analysis
        state = SystemState()
        state.num_molecules = num_molecules
        state.all_molecule_definitions = [mol_data]
        state.molecules_to_add = [0] * num_molecules  # All refer to the first (and only) molecule definition
        state.verbosity_level = 1
        
        # Perform the analysis
        results = calculate_optimal_box_length(state)
        
        if 'error' in results:
            print(f"Error in analysis: {results['error']}")
            return
        
        # Display results
        total_volume = results['total_molecular_volume']
        print(f"\nMOLECULAR VOLUME ANALYSIS:")
        print(f"  Single molecule volume: {results['individual_molecular_volumes'][0]['volume_A3']:.2f} Å³")
        print(f"  Total volume ({num_molecules} molecules): {total_volume:.2f} Å³")
        
        print(f"\nBOX LENGTH RECOMMENDATIONS:")
        print("Packing    Box Length    Box Volume    Free Volume   Typical Use")
        print("Fraction   (Å)          (Å³)         (Å³)") 
        print("-" * 65)
        
        recommendations = results['box_length_recommendations']
        contexts = {
            '10.0%': 'Very dilute gas phase',
            '20.0%': 'Dilute gas/vapor phase', 
            '30.0%': 'Moderate density fluid',
            '40.0%': 'Dense fluid phase',
            '50.0%': 'Very dense/liquid-like'
        }
        
        for key, rec in recommendations.items():
            pf = rec['packing_fraction']
            bl = rec['box_length_A']
            bv = rec['box_volume_A3']
            fv = rec['free_volume_A3']
            context = contexts.get(key, 'Custom density')
            print(f"{pf:6.1%}     {bl:8.2f}      {bv:8.0f}       {fv:8.0f}     {context}")
        
        # Calculate old method for comparison
        coords_array = np.array([atom[1:4] for atom in atoms_coords])
        min_coords = np.min(coords_array, axis=0)
        max_coords = np.max(coords_array, axis=0)
        extents = max_coords - min_coords
        max_extent = np.max(extents)
        old_method_box = max_extent + 16.0  # 8 Å on each side
        
        print(f"\nCOMPARISON WITH SIMPLE METHOD:")
        print(f"  Largest molecular dimension: {max_extent:.2f} Å")
        print(f"  Old method (8 Å rule): {old_method_box:.2f} Å")
        
        # Compare with 30% recommendation
        rec_30 = recommendations.get('30.0%', {}).get('box_length_A', 0)
        if rec_30 > 0:
            ratio = old_method_box / rec_30
            print(f"  Ratio (old/volume-based): {ratio:.2f}")
            if ratio > 1.5:
                print(f"    Old method may be wastefully large")
            elif ratio < 0.7:
                print(f"    Old method may be too small")
            else:
                print(f"    Methods are reasonably consistent")
        
        print(f"\nRECOMMENDATIONS:")
        rec_20 = recommendations.get('20.0%', {}).get('box_length_A', 0)
        if rec_20 > 0 and rec_30 > 0:
            print(f"  • For most applications: {rec_30:.1f} Å (30% packing)")
            print(f"  • For gas phase studies: {rec_20:.1f} Å (20% packing)")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        traceback.print_exc()


# Example usage and testing function
def test_box_length_analysis():
    """
    Test function to demonstrate the new volume-based box length calculation.
    Creates a simple water molecule and analyzes it.
    """
    print("\n" + "="*70)
    print("TESTING VOLUME-BASED BOX LENGTH CALCULATION")
    print("="*70)
    
    # Create a simple water molecule for testing
    # H2O coordinates (in Angstroms, approximate)
    water_atoms = [
        (8, 0.0, 0.0, 0.0),      # O at origin
        (1, 0.757, 0.587, 0.0),  # H1
        (1, -0.757, 0.587, 0.0)  # H2
    ]
    
    water_mol = MoleculeData("H2O", 3, water_atoms)
    
    # Test with different numbers of molecules
    test_cases = [1, 10, 50, 100]
    
    for num_mols in test_cases:
        print(f"\nTesting {num_mols} water molecule(s):")
        print("-" * 40)
        
        # Create test state
        state = SystemState()
        state.num_molecules = num_mols
        state.all_molecule_definitions = [water_mol]
        state.molecules_to_add = [0] * num_mols
        state.verbosity_level = 0  # Suppress verbose output for testing
        
        results = calculate_optimal_box_length(state)
        
        if 'error' not in results:
            total_vol = results['total_molecular_volume']
            rec_20 = results['box_length_recommendations']['20.0%']['box_length_A']
            rec_30 = results['box_length_recommendations']['30.0%']['box_length_A']
            
            print(f"  Total molecular volume: {total_vol:.2f} Å³")
            print(f"  Recommended box (20% packing): {rec_20:.1f} Å")
            print(f"  Recommended box (30% packing): {rec_30:.1f} Å")
            
            # Calculate density at 30% packing
            box_vol_30 = rec_30**3
            density_30 = (num_mols * 18.015) / (box_vol_30 * 6.022e23 * 1e-24)  # g/cm³
            print(f"  Approximate density at 30%: {density_30:.3f} g/cm³")
        else:
            print(f"  Error: {results['error']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Allow command-line usage for box length analysis
        if sys.argv[1] == "analyze_box" and len(sys.argv) >= 3:
            xyz_file = sys.argv[2]
            num_molecules = int(sys.argv[3]) if len(sys.argv) > 3 else 1
            analyze_box_length_from_xyz(xyz_file, num_molecules)
        elif sys.argv[1] == "test_box":
            test_box_length_analysis()
        elif sys.argv[1] == "input":
            # Open the web-based input generator
            import http.server
            import socketserver
            import webbrowser
            from functools import partial
            
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
            script_dir = os.path.dirname(os.path.abspath(__file__))
            web_dir = script_dir
            input_page = 'index.html'
            
            if not os.path.exists(os.path.join(web_dir, input_page)):
                print("Error: Web input generator not found.")
                print(f"Expected at: {web_dir}/{input_page}")
                sys.exit(1)
            
            handler = partial(http.server.SimpleHTTPRequestHandler, directory=web_dir)
            
            try:
                with socketserver.TCPServer(("", port), handler) as httpd:
                    url = f"http://localhost:{port}/{input_page}"
                    print(f"\n  ASCEC Input Generator")
                    print(f"  ─────────────────────")
                    print(f"  Opening: {url}")
                    print(f"  Press Ctrl+C to stop\n")
                    webbrowser.open(url)
                    httpd.serve_forever()
            except OSError as e:
                if "Address already in use" in str(e):
                    print(f"Error: Port {port} is already in use.")
                    print(f"Try: python ascec-v04.py input {port + 1}")
                else:
                    raise
            except KeyboardInterrupt:
                print("\n  Server stopped.")
        else:
            main_ascec_integrated()
    else:
        # Run normal ASCEC if no special arguments
        main_ascec_integrated()

