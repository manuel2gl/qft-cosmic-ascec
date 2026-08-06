"""Geometry-derived clustering features — verbatim port of cosmic-v01.

R5 (D-039 faithful decomposition) re-ports cosmic-v01's clustering feature
front-end. cosmic-v01 carries its **own** copies of the geometry maths —
``calculate_radius_of_gyration`` (cosmic-v01.py 425-455) and
``calculate_rotational_constants`` (457-531) — distinct from ascec-v04's
``geometry/inertia.py``. The pre-R5 v05 ``geometric.py`` delegated to
``geometry/inertia.py``; that was a divergence. This module ports cosmic-v01's
own functions verbatim so the clustering pipeline is self-contained, exactly as
cosmic-v01 is.

Ported verbatim from ``cosmic-v01.py``:

* ``element_masses`` — lines 236-255.
* ``atomic_number_to_symbol`` — lines 257-284.
* ``extract_xyz_from_output`` — lines 338-423.
* ``calculate_radius_of_gyration`` — lines 425-455.
* ``calculate_rotational_constants`` — lines 457-531.
* ``detect_hydrogen_bonds`` — lines 534-682.

The only adaptations are the mechanical ones D-039 permits: this is now a
module rather than a slice of one script. ``print`` statements that are part of
cosmic-v01's stdout contract are kept verbatim.
"""

from __future__ import annotations

import numpy as np

# Bohr↔Angstrom conversion. Mirrors the constant in parsers.py.
BOHR_TO_ANGSTROM = 0.529177210903

# Above this atom count the dense pair matrix in calculate_nuclear_repulsion
# stops fitting in memory (three N×N float64 arrays: ~1.9 GB at 9k atoms) and a
# blocked accumulation takes over. The cut is deliberately well above anything
# the dense path handles comfortably, so every system that works today keeps
# taking the original code path and returns the identical float.
_VNN_DENSE_MAX_ATOMS = 6000
_VNN_BLOCK_ROWS = 1024

# Element masses dictionary (in atomic mass units) — cosmic-v01.py lines 236-255.
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
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
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


def calculate_nuclear_repulsion(atomnos, atomcoords):
    """Calculate V_NN = Σ_{i<j} Z_i Z_j / r_ij in Hartree.

    Args:
        atomnos (array): Atomic numbers (Z) for each atom.
        atomcoords (array): Atomic coordinates (N, 3) in Angstroms.

    Returns:
        float: Nuclear repulsion energy in Hartree, or None on error.
    """
    try:
        Z = np.asarray(atomnos, dtype=float)
        coords = np.asarray(atomcoords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != Z.shape[0]:
            return None
        if Z.size < 2:
            return 0.0
        coords_bohr = coords / BOHR_TO_ANGSTROM

        if Z.size > _VNN_DENSE_MAX_ATOMS:
            # The dense form below materialises three N×N arrays — 5.8 GB at
            # 27k atoms, which simply fails. Row blocks give the same sum with
            # bounded memory. Kept off the small-system path because a blocked
            # summation reorders the additions, and the dense result is the one
            # every existing run was produced with.
            n = Z.size
            total = 0.0
            for start in range(0, n - 1, _VNN_BLOCK_ROWS):
                stop = min(start + _VNN_BLOCK_ROWS, n - 1)
                # Only columns to the right of the block's first row can be in
                # the upper triangle, so the block never spans the full matrix.
                cols = slice(start + 1, n)
                diff = coords_bohr[start:stop, None, :] - coords_bohr[None, cols, :]
                r_block = np.linalg.norm(diff, axis=-1)
                del diff
                # Zero out the sub-diagonal corner of the block by sending its
                # distances to infinity, which costs no branch in the division.
                keep = np.arange(start + 1, n)[None, :] > np.arange(start, stop)[:, None]
                np.copyto(r_block, np.inf, where=~keep)
                total += float(np.sum((Z[start:stop, None] * Z[None, cols]) / r_block))
            return total

        diff = coords_bohr[:, None, :] - coords_bohr[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        iu = np.triu_indices(Z.size, k=1)
        return float(np.sum(Z[iu[0]] * Z[iu[1]] / r[iu]))
    except Exception as e:
        print(f"Error in calculate_nuclear_repulsion: {e}")
        return None


def calculate_rotational_constants(atomnos, atomcoords):
    """
    Calculate rotational constants (A, B, C) from the moment of inertia tensor.

    Uses the principal moments of inertia Ia <= Ib <= Ic (in amu*Å²)
    and converts to rotational constants A >= B >= C (in cm^-1) via:

        X = h / (8 π² c Ix)

    where h is Planck's constant, c is the speed of light, and Ix is a principal
    moment.

    Args:
        atomnos (array): Atomic numbers.
        atomcoords (array): Atomic coordinates (N, 3) in Angstroms.

    Returns:
        numpy.ndarray of shape (3,) with [A, B, C] in cm^-1, or None on error.
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

        # h / (8 π²) in SI (J·s / rad²) → divide by I in kg·m² → Hz → cm^-1
        h = 6.62607015e-34  # J·s
        c_cm_per_s = 2.99792458e10  # speed of light in cm/s
        factor = h / (8.0 * np.pi**2)  # J·s / rad²

        rot_consts = []
        for I_val in eigenvalues:
            if I_val <= 0.0:
                rot_consts.append(0.0)
            else:
                I_si = I_val * amu_ang2_to_kg_m2
                freq_hz = factor / I_si
                rot_consts.append(freq_hz / c_cm_per_s)  # Hz → cm^-1

        # Convention: A >= B >= C
        rot_consts.sort(reverse=True)
        return np.array(rot_consts, dtype=float)
    except Exception:
        return None


# Hydrogen-bond detection criteria. These are the single source of truth: the
# detector below and the per-cluster .dat report (dat_writer) both read them, so
# the printed "Criterion:" line can never drift from the values actually used.
HB_MIN_DISTANCE = 1.4   # Minimum H...A distance (Å)
HB_MAX_DISTANCE = 3.2   # Maximum H...A distance (Å)
HB_MIN_ANGLE = 120.0    # Minimum D-H...A angle (degrees) for a bond to be counted
HB_MAX_ANGLE = 180.0    # Maximum D-H...A angle (degrees)
HB_COVALENT_DH_SEARCH_DISTANCE = 1.4  # D-H covalent bond search limit (Å)


# Below this atom count the original nested-loop detector is already fast
# (< 0.05 s) and stays in charge, so the smallest systems keep taking the exact
# code path every historical run used.
_HBOND_NEIGHBOUR_MIN_ATOMS = 300

# Slack added to a neighbour-search radius before the exact distance test is
# applied. The tree's own distances can differ from ``np.linalg.norm`` in the
# last bit, so candidate generation is made deliberately generous and the real
# cutoff is enforced afterwards on recomputed distances.
_HBOND_RADIUS_SLACK = 1e-6


def _hbond_statistics(hbonds):
    """Assemble the H-bond property dict from the accepted bonds.

    Shared by both detector implementations so the two can never disagree on
    how the count, means and spreads are derived.
    """
    extracted_props = {
        'num_hydrogen_bonds': len(hbonds),
        'hbond_details': hbonds,
        'average_hbond_distance': None,
        'min_hbond_distance': None,
        'max_hbond_distance': None,
        'std_hbond_distance': None,
        'average_hbond_angle': None,
        'min_hbond_angle': None,
        'max_hbond_angle': None,
        'std_hbond_angle': None
    }

    if hbonds:
        distances = [bond['H...A_distance'] for bond in hbonds]
        angles = [bond['D-H...A_angle'] for bond in hbonds]

        extracted_props['average_hbond_distance'] = np.mean(distances)
        extracted_props['min_hbond_distance'] = np.min(distances)
        extracted_props['max_hbond_distance'] = np.max(distances)
        extracted_props['std_hbond_distance'] = np.std(distances) if len(distances) > 1 else 0.0

        extracted_props['average_hbond_angle'] = np.mean(angles)
        extracted_props['min_hbond_angle'] = np.min(angles)
        extracted_props['max_hbond_angle'] = np.max(angles)
        extracted_props['std_hbond_angle'] = np.std(angles) if len(angles) > 1 else 0.0

    return extracted_props


def _detect_hydrogen_bonds_neighbour(atomnos, coords, atom_labels, donor_acceptor_z):
    """Neighbour-search H-bond detector — same result, near-linear cost.

    Returns the same dict as :func:`detect_hydrogen_bonds`, or ``None`` if
    SciPy is unavailable (the caller then falls back to the original loops).

    Exactness rests on three points, none of which are approximations:

    1. **Donor search.** The original scans every donor-capable atom for the
       globally nearest one, then keeps it only if it lies within
       ``HB_COVALENT_DH_SEARCH_DISTANCE``. The globally nearest donor is within
       that radius exactly when some donor is, and it is then also the nearest
       *within* the radius — so restricting the search to a ball of that radius
       cannot change the answer. Ties go to the lowest index, matching the
       strictly-less-than first-wins scan of the original.
    2. **Acceptor search.** Pairs are drawn from a ball of the maximum H···A
       distance, which is a superset of the accepted pairs by construction.
    3. **Arithmetic.** The tree only nominates candidates. Every distance and
       angle that reaches a threshold test or the output is recomputed with the
       same ``np.linalg.norm`` / ``np.dot`` calls the original uses, on the same
       two coordinate rows, so the reported values are bit-for-bit identical
       rather than merely close.

    The ``i_d == i_h`` / ``i_a == i_h`` guards in the original are vacuous —
    hydrogen is not in the donor/acceptor set — so they have no counterpart here.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None

    atomnos = np.asarray(atomnos)
    hydrogen_idx = np.flatnonzero(atomnos == 1)
    donor_idx = np.flatnonzero(np.isin(atomnos, sorted(donor_acceptor_z)))

    if hydrogen_idx.size == 0 or donor_idx.size == 0:
        return _hbond_statistics([])

    donor_tree = cKDTree(coords[donor_idx])

    # --- Pass 1: nearest covalently bonded donor for each hydrogen ----------
    covalent_candidates = donor_tree.query_ball_point(
        coords[hydrogen_idx], HB_COVALENT_DH_SEARCH_DISTANCE + _HBOND_RADIUS_SLACK
    )

    h_donor = {}  # {h_idx: (donor_idx, D-H distance)}
    for row, i_h in enumerate(hydrogen_idx):
        candidates = covalent_candidates[row]
        if not candidates:
            continue
        coord_h = coords[i_h]
        best_idx = -1
        best_dist = float('inf')
        # Ascending atom index reproduces the original's first-wins tie-break.
        for i_d in sorted(int(donor_idx[c]) for c in candidates):
            dist_dh = np.linalg.norm(coords[i_d] - coord_h)
            if dist_dh < best_dist:
                best_dist = dist_dh
                best_idx = i_d
        if best_idx != -1 and best_dist <= HB_COVALENT_DH_SEARCH_DISTANCE:
            h_donor[int(i_h)] = (best_idx, best_dist)

    if not h_donor:
        return _hbond_statistics([])

    # --- Pass 2: acceptors in range, with the D-H...A angle ----------------
    bonded_h = np.array(sorted(h_donor), dtype=int)
    acceptor_candidates = donor_tree.query_ball_point(
        coords[bonded_h], HB_MAX_DISTANCE + _HBOND_RADIUS_SLACK
    )

    all_potential_hbonds_details = []
    for row, i_h in enumerate(bonded_h):
        i_h = int(i_h)
        donor_i, dh_covalent = h_donor[i_h]
        coord_h = coords[i_h]
        coord_d = coords[donor_i]

        for i_a in sorted(int(donor_idx[c]) for c in acceptor_candidates[row]):
            if i_a == donor_i:
                continue

            coord_a = coords[i_a]
            dist_ha = np.linalg.norm(coord_h - coord_a)
            if not (HB_MIN_DISTANCE <= dist_ha <= HB_MAX_DISTANCE):
                continue

            vec_h_d = coord_d - coord_h
            vec_h_a = coord_a - coord_h
            norm_vec_h_d = np.linalg.norm(vec_h_d)
            norm_vec_h_a = np.linalg.norm(vec_h_a)

            if norm_vec_h_d == 0 or norm_vec_h_a == 0:
                angle_deg = 0.0
            else:
                cos_angle = np.dot(vec_h_d, vec_h_a) / (norm_vec_h_d * norm_vec_h_a)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))

            all_potential_hbonds_details.append({
                'donor_atom_label': atom_labels[donor_i],
                'hydrogen_atom_label': atom_labels[i_h],
                'acceptor_atom_label': atom_labels[i_a],
                'H...A_distance': dist_ha,
                'D-H...A_angle': angle_deg,
                'D-H_covalent_distance': dh_covalent
            })

    return _hbond_statistics([
        b for b in all_potential_hbonds_details
        if HB_MIN_ANGLE <= b['D-H...A_angle'] <= HB_MAX_ANGLE
    ])


def detect_hydrogen_bonds(atomnos, atomcoords):
    """
    Detect hydrogen bonds based on distance and angle criteria.

    Distance criterion: 1.4-3.2 Å between H and acceptor
    Angle criterion: D-H...A angle >= 120°

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
        HB_min_dist_actual = HB_MIN_DISTANCE  # Minimum H...A distance (Å)
        HB_max_dist_actual = HB_MAX_DISTANCE  # Maximum H...A distance (Å)
        COVALENT_DH_SEARCH_DIST = HB_COVALENT_DH_SEARCH_DISTANCE  # D-H covalent bond search limit (Å)
        HB_min_angle_actual = HB_MIN_ANGLE  # Minimum D-H...A angle (degrees)
        HB_max_angle_actual = HB_MAX_ANGLE  # Maximum D-H...A angle (degrees)

        symbols = [atomic_number_to_symbol(n) for n in atomnos]
        atom_labels = [f"{sym}{i+1}" for i, sym in enumerate(symbols)]
        all_potential_hbonds_details = []

        # The pass below is O(N_H · N) in interpreted Python — 6 s per structure
        # at 2700 atoms, hours at the sizes the XYZ front-end exists for. A
        # neighbour-search formulation gives the same answer in near-linear
        # time; see _detect_hydrogen_bonds_neighbour for why it is exact rather
        # than merely close.
        if coords.shape[0] >= _HBOND_NEIGHBOUR_MIN_ATOMS:
            neighbour_result = _detect_hydrogen_bonds_neighbour(
                atomnos, coords, atom_labels, potential_donor_acceptor_z
            )
            if neighbour_result is not None:
                return neighbour_result

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

        # Keep the contacts that meet the angle criterion — these are the
        # hydrogen bonds. Everything below the cutoff is discarded here, so the
        # count, the detail list and the geometry statistics all describe the
        # same set of bonds.
        hbonds = [
            b for b in all_potential_hbonds_details
            if HB_min_angle_actual <= b['D-H...A_angle'] <= HB_max_angle_actual
        ]
        return _hbond_statistics(hbonds)

    except Exception as e:
        return {
            'num_hydrogen_bonds': 0,
            'hbond_details': [],
            'average_hbond_distance': None, 'min_hbond_distance': None,
            'max_hbond_distance': None, 'std_hbond_distance': None,
            'average_hbond_angle': None, 'min_hbond_angle': None,
            'max_hbond_angle': None, 'std_hbond_angle': None
        }


__all__ = [
    "atomic_number_to_symbol",
    "calculate_nuclear_repulsion",
    "calculate_radius_of_gyration",
    "calculate_rotational_constants",
    "detect_hydrogen_bonds",
    "element_masses",
    "extract_xyz_from_output",
]
