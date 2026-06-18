"""Static element data ported byte-for-byte from v04.

Source: ``ascec-v04.py`` lines 163-475. Frozen module constants. Any future
update must be mirrored in v04 first or the parity tests break.

R8 re-home note (D-039 source-of-truth). The byte-exact verbatim ports of
v04's ``r_atom`` (line 163), ``r_vdw`` (274), ``atomic_weights`` (425),
``element_symbols`` (390) and ``atomic_number_to_symbol`` (407) live in
:mod:`cosmic_ascec.workflow.stages` (the R6d helper closure). The mappings
below are the v04-typed presentation of the same v03 data and are pinned to
the R6d ports by ``tests/parity/test_elements_data_parity.py``. To update
any element-table entry, edit the R6d port first; this module is a typed
re-export.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping


_R_ATOM_RAW: dict[int, float] = {
    # Covalent radii (Angstrom), used for polyatomic molecules in box-length
    # and overlap checks. Source:
    #   B. Cordero, V. Gomez, A. E. Platero-Prats, M. Reves, J. Echeverria,
    #   E. Cremades, F. Barragan, S. Alvarez, "Covalent radii revisited,"
    #   Dalton Trans. 2008, 2832-2838. doi:10.1039/B801115J
    # (Carbon uses Cordero's sp2 value, 0.73 A.)
    # Period 1
    1: 0.31,   2: 0.28,
    # Period 2
    3: 1.28,   4: 0.96,   5: 0.84,   6: 0.73,   7: 0.71,
    8: 0.66,   9: 0.57,  10: 0.58,
    # Period 3
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07,
    16: 1.05, 17: 1.02, 18: 1.06,
    # Period 4
    19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60, 23: 1.53,
    24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24,
    29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19,
    34: 1.20, 35: 1.20, 36: 1.16,
    # Period 5
    37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75, 41: 1.64,
    42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42, 46: 1.39,
    47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39, 51: 1.39,
    52: 1.38, 53: 1.39, 54: 1.40,
    # Period 6
    55: 2.44, 56: 2.15, 57: 2.07,
    58: 2.04, 59: 2.03, 60: 2.01, 61: 1.99, 62: 1.98,
    63: 1.98, 64: 1.96, 65: 1.94, 66: 1.92, 67: 1.92,
    68: 1.89, 69: 1.90, 70: 1.87, 71: 1.87,
    72: 1.75, 73: 1.70, 74: 1.62, 75: 1.51, 76: 1.44,
    77: 1.41, 78: 1.36, 79: 1.36, 80: 1.32,
    81: 1.45, 82: 1.46, 83: 1.48, 84: 1.40, 85: 1.50,
    86: 1.50,
    # Period 7 - Cordero et al. (2008) extends only to Cm (Z=96); Fr is the
    # paper's extrapolated value. Z>=97 are absent there and fall back to the
    # 1.50 A default in radii.get_radius.
    87: 2.60, 88: 2.21, 89: 2.15,
    90: 2.06, 91: 2.00, 92: 1.96, 93: 1.90, 94: 1.87,
    95: 1.80, 96: 1.69,
}


_R_VDW_RAW: dict[int, float] = {
    # Van der Waals radii (Angstrom), used for monatomic species (atoms/ions)
    # in box-length and overlap checks. Sources:
    #   A. Bondi, "van der Waals Volumes and Radii," J. Phys. Chem. 1964,
    #   68(3), 441-451. doi:10.1021/j100785a001  (base set: H, C, N, O, F,
    #   alkali/alkaline-earth, etc.)
    #   S. Alvarez, "A cartography of the van der Waals territories,"
    #   Dalton Trans. 2013, 42, 8617-8636. doi:10.1039/C3DT50599E
    #   (extension to heavier/transition elements; main-group extensions
    #   also coincide with M. Mantina et al., J. Phys. Chem. A 2009, 113,
    #   5806-5812, doi:10.1021/jp8111556).
    # NOTE: several transition-metal entries are a flat 2.00 A placeholder,
    # not a measured literature value.
    # Period 1
    1: 1.20,   2: 1.40,
    # Period 2
    3: 1.82,   4: 1.53,   5: 1.92,   6: 1.70,   7: 1.55,
    8: 1.52,   9: 1.47,  10: 1.54,
    # Period 3
    11: 2.27, 12: 1.73, 13: 1.84, 14: 2.10, 15: 1.80,
    16: 1.80, 17: 1.75, 18: 1.88,
    # Period 4
    19: 2.75, 20: 2.31, 21: 2.11, 22: 2.00, 23: 2.00,
    24: 2.00, 25: 2.00, 26: 2.00, 27: 2.00, 28: 1.63,
    29: 1.40, 30: 1.39, 31: 1.87, 32: 2.11, 33: 1.85,
    34: 1.90, 35: 1.85, 36: 2.02,
    # Period 5
    37: 3.03, 38: 2.49, 39: 2.31, 40: 2.23, 41: 2.18,
    42: 2.17, 43: 2.16, 44: 2.13, 45: 2.10, 46: 1.63,
    47: 1.72, 48: 1.58, 49: 1.93, 50: 2.17, 51: 2.06,
    52: 2.06, 53: 1.98, 54: 2.16,
    # Period 6
    55: 3.43, 56: 2.68, 57: 2.43, 58: 2.42, 59: 2.40,
    60: 2.39, 61: 2.38, 62: 2.36, 63: 2.35, 64: 2.34,
    65: 2.33, 66: 2.31, 67: 2.30, 68: 2.29, 69: 2.27,
    70: 2.26, 71: 2.24, 72: 2.23, 73: 2.22, 74: 2.18,
    75: 2.16, 76: 2.16, 77: 2.13, 78: 1.75, 79: 1.66,
    80: 1.55, 81: 1.96, 82: 2.02, 83: 2.07, 84: 1.97,
    85: 2.02, 86: 2.20,
}


_SYMBOL_TO_Z_RAW: dict[str, int] = {
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
    "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118,
}


_Z_TO_SYMBOL_RAW: dict[int, str] = {z: sym for sym, z in _SYMBOL_TO_Z_RAW.items()}


_ATOMIC_WEIGHTS_RAW: dict[int, float] = {
    # T. Prohaska et al., Pure Appl. Chem. 94, 573 (2022).
    # IUPAC Standard Atomic Weights (2021/2022), 4 significant figures.
    1: 1.008,    2: 4.003,    3: 6.940,    4: 9.012,    5: 10.81,
    6: 12.01,    7: 14.01,    8: 16.00,    9: 19.00,   10: 20.18,
    11: 22.99,  12: 24.31,   13: 26.98,   14: 28.09,   15: 30.97,
    16: 32.06,  17: 35.45,   18: 39.95,   19: 39.10,   20: 40.08,
    21: 44.96,  22: 47.87,   23: 50.94,   24: 52.00,   25: 54.94,
    26: 55.85,  27: 58.93,   28: 58.69,   29: 63.55,   30: 65.38,
    31: 69.72,  32: 72.63,   33: 74.92,   34: 78.97,   35: 79.90,
    36: 83.80,  37: 85.47,   38: 87.62,   39: 88.91,   40: 91.22,
    41: 92.91,  42: 95.95,   43: 97.00,   44: 101.1,   45: 102.9,
    46: 106.4,  47: 107.9,   48: 112.4,   49: 114.8,   50: 118.7,
    51: 121.8,  52: 127.6,   53: 126.9,   54: 131.3,   55: 132.9,
    56: 137.3,  57: 138.9,   58: 140.1,   59: 140.9,   60: 144.2,
    61: 145.0,  62: 150.4,   63: 152.0,   64: 157.2,   65: 158.9,
    66: 162.5,  67: 164.9,   68: 167.3,   69: 168.9,   70: 173.0,
    71: 175.0,  72: 178.5,   73: 180.9,   74: 183.8,   75: 186.2,
    76: 190.2,  77: 192.2,   78: 195.1,   79: 197.0,   80: 200.6,
    81: 204.4,  82: 207.2,   83: 209.0,   84: 209.0,   85: 210.0,
    86: 222.0,  87: 223.0,   88: 226.0,   89: 227.0,   90: 232.0,
    91: 231.0,  92: 238.0,   93: 237.0,   94: 244.0,   95: 243.0,
    96: 247.0,  97: 247.0,   98: 251.0,   99: 252.0,  100: 257.0,
    101: 258.0, 102: 259.0,  103: 262.0,  104: 267.0,  105: 270.0,
    106: 269.0, 107: 270.0,  108: 270.0,  109: 278.0,  110: 281.0,
    111: 281.0, 112: 285.0,  113: 286.0,  114: 289.0,  115: 289.0,
    116: 293.0, 117: 293.0,  118: 294.0,
}


_ELECTRONEGATIVITY_RAW: dict[str, float] = {
    "H": 2.20, "He": 0.0,
    "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "Ne": 0.0,
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
    "Ar": 0.0,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55,
    "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01,
    "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 0.0,
    "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.6, "Mo": 2.16, "Tc": 1.9,
    "Ru": 2.2, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96,
    "Sb": 2.05, "Te": 2.1, "I": 2.66, "Xe": 0.0,
    "Cs": 0.79, "Ba": 0.89, "La": 1.1, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Pm": 1.13,
    "Sm": 1.17, "Eu": 1.17, "Gd": 1.2, "Tb": 1.1, "Dy": 1.22, "Ho": 1.23, "Er": 1.24,
    "Tm": 1.25, "Yb": 1.1, "Lu": 1.27, "Hf": 1.3, "Ta": 1.5, "W": 2.36, "Re": 1.9,
    "Os": 2.2, "Ir": 2.2, "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 1.87,
    "Bi": 2.02, "Po": 2.0, "At": 2.2, "Rn": 0.0,
    "Fr": 0.7, "Ra": 0.9, "Ac": 1.1, "Th": 1.3, "Pa": 1.5, "U": 1.38, "Np": 1.36,
    "Pu": 1.28, "Am": 1.13, "Cm": 1.28, "Bk": 1.3, "Cf": 1.3, "Es": 1.3, "Fm": 1.3,
    # v04 carries these as integers (atomic numbers) rather than Pauling values; preserved verbatim.
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107,
    "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114,
    "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118,
}


COVALENT_RADII: Final[Mapping[int, float]] = MappingProxyType(_R_ATOM_RAW)
VDW_RADII: Final[Mapping[int, float]] = MappingProxyType(_R_VDW_RAW)
SYMBOL_TO_Z: Final[Mapping[str, int]] = MappingProxyType(_SYMBOL_TO_Z_RAW)
Z_TO_SYMBOL: Final[Mapping[int, str]] = MappingProxyType(_Z_TO_SYMBOL_RAW)
ATOMIC_WEIGHTS: Final[Mapping[int, float]] = MappingProxyType(_ATOMIC_WEIGHTS_RAW)
ELECTRONEGATIVITY: Final[Mapping[str, float]] = MappingProxyType(_ELECTRONEGATIVITY_RAW)

DUMMY_ATOM_SYMBOL: Final[str] = "X"


__all__ = [
    "ATOMIC_WEIGHTS",
    "COVALENT_RADII",
    "DUMMY_ATOM_SYMBOL",
    "ELECTRONEGATIVITY",
    "SYMBOL_TO_Z",
    "VDW_RADII",
    "Z_TO_SYMBOL",
]
