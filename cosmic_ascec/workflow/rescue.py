"""Rescue helpers — imaginary-frequency detection and mode displacement.

When a workflow optimisation produces a structure with imaginary vibrational
frequencies, v04 *rescues* it: a single imaginary mode is displaced along its
eigenvector, multiple modes trigger displacement along the most-negative mode,
and the displaced geometry is fed back into a re-optimisation.

This module is a **verbatim port** of ``ascec-v04.py`` lines 10533-11171
(**D-039**: faithful decomposition). It covers the imaginary-frequency half of
the rescue protocol — counting imaginary modes across ORCA / Gaussian /
standalone-xTB output, parsing the normal-mode eigenvectors, applying the
displacement, and extracting a final optimised geometry for re-optimisation.

The Hessian-rescue half (``parse_rescue_method`` … ``run_rescue_hessian_calculation``,
v04 7939-8480) is re-ported with the optimisation stage in R6c, because that is
the only caller that drives it.

The atomic-number → symbol reverse lookup uses
:data:`cosmic_ascec.elements.SYMBOL_TO_Z` — the exact dict v04 names
``element_symbols`` — iterated with v04's own loop so the chosen symbol is
identical.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from cosmic_ascec.elements import SYMBOL_TO_Z as element_symbols

__all__ = [
    "handle_imaginary_frequencies",
    "count_imaginary_frequencies",
    "displace_along_imaginary_mode",
    "displace_orca_imaginary_mode",
    "displace_gaussian_imaginary_mode",
    "extract_displaced_frame",
    "format_ordinal",
    "extract_final_geometry",
]


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
    """Count number of imaginary frequencies in QM output file by checking for negative values.

    Supports ORCA, Gaussian, and standalone xTB output formats.
    For xTB, also checks the vibspectrum file in the same directory.
    """
    try:
        with open(out_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        content_lower = content.lower()

        # Detect standalone xTB output
        is_xtb = (
            'x]  t  b' in content_lower
            or 'xtb version' in content_lower
            or 'normal termination of xtb' in content_lower
        )

        if is_xtb:
            # Strategy 1: parse frequencies from xTB output (format: -123.45 cm^-1)
            freq_matches = re.findall(r'([-+]?\d+\.\d+)\s*cm\^-?1', content, re.IGNORECASE)
            if freq_matches:
                imag_count = sum(1 for freq in freq_matches if float(freq) < 0.0)
                return imag_count

            # Strategy 2: check vibspectrum file (written by --hess)
            out_dir = os.path.dirname(out_file) or '.'
            basename = os.path.splitext(os.path.basename(out_file))[0]
            # xTB with --namespace writes <namespace>.vibspectrum
            for vib_name in [f'{basename}.vibspectrum', 'vibspectrum']:
                vib_path = os.path.join(out_dir, vib_name)
                if os.path.exists(vib_path):
                    try:
                        with open(vib_path, 'r', encoding='utf-8', errors='replace') as vf:
                            vib_content = vf.read()
                        # vibspectrum format: freq(cm^-1)  IR-intensity  ...
                        # Lines after the $vibrational spectrum header, before $end
                        imag = 0
                        for line in vib_content.splitlines():
                            line = line.strip()
                            if not line or line.startswith('$') or line.startswith('#'):
                                continue
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    freq_val = float(parts[0])
                                    if freq_val < 0.0:
                                        imag += 1
                                except ValueError:
                                    continue
                        return imag
                    except Exception:
                        pass

            return 0

        # --- ORCA format ---
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
        # Check if this is ORCA, Gaussian, or standalone xTB
        with open(out_file, 'r', encoding='utf-8', errors='replace') as f:
            first_lines = ''.join(f.readlines()[:50])
            first_lower = first_lines.lower()
            is_orca = 'O   R   C   A' in first_lines
            is_gaussian = 'Gaussian' in first_lines
            is_xtb = (
                'x]  t  b' in first_lower
                or 'xtb version' in first_lower
                or 'normal termination of xtb' in first_lower
            )

        if is_orca:
            # ORCA: Parse normal modes directly from output file
            return displace_orca_imaginary_mode(out_file, use_highest_mode=use_highest_mode)

        elif is_gaussian:
            # Gaussian: Manually displace along imaginary mode
            return displace_gaussian_imaginary_mode(out_file, use_highest_mode=use_highest_mode)

        elif is_xtb:
            # xTB with --hess writes a g98.out file in Gaussian format
            # which contains normal modes. Use the Gaussian parser on that file.
            out_dir = os.path.dirname(out_file) or '.'
            basename = os.path.splitext(os.path.basename(out_file))[0]
            # Try namespace-prefixed g98.out first, then plain g98.out
            for g98_name in [f'{basename}.g98.out', 'g98.out']:
                g98_path = os.path.join(out_dir, g98_name)
                if os.path.exists(g98_path):
                    result = displace_gaussian_imaginary_mode(
                        g98_path, use_highest_mode=use_highest_mode
                    )
                    if result:
                        return result
            return None

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
        with open(out_file, 'r', encoding='utf-8', errors='replace') as f:
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
        with open(out_file, 'r', encoding='utf-8', errors='replace') as f:
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
        with open(traj_file, 'r', encoding='utf-8', errors='replace') as f:
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

    MULTI-SOFTWARE SUPPORT (ORCA, Gaussian & standalone xTB):
    =========================================================
    For ORCA: Extracts from "CARTESIAN COORDINATES (ANGSTROEM)" section
    For Gaussian: Extracts from "Standard orientation" section
    For xTB: Reads xtbopt.xyz (optimized geometry) or falls back to last
             geometry in the xTB trajectory output

    Args:
        out_file: Path to output file (.out for ORCA/xTB, .log for Gaussian)
        optimization_dir_path: Stage working directory (unused, retained for stable function signatures)

    Returns:
        List of XYZ lines if successful, None otherwise
    """
    try:
        with open(out_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # Determine program type
        content_snippet = ''.join(lines[:50]).lower()
        is_orca = any('O   R   C   A' in line for line in lines[:50])
        is_gaussian = any('Gaussian' in line for line in lines[:50])
        is_xtb = (
            'x]  t  b' in content_snippet
            or 'xtb version' in content_snippet
            or 'normal termination of xtb' in content_snippet
        )

        coords = []

        if is_xtb:
            # For standalone xTB, read the optimized XYZ file
            out_dir = os.path.dirname(out_file) or '.'
            basename = os.path.splitext(os.path.basename(out_file))[0]
            # xTB with --namespace writes <namespace>.xtbopt.xyz
            xyz_candidates = [
                os.path.join(out_dir, f'{basename}.xtbopt.xyz'),
                os.path.join(out_dir, 'xtbopt.xyz'),
                os.path.join(out_dir, f'{basename}.xyz'),
            ]
            for xyz_path in xyz_candidates:
                if os.path.exists(xyz_path):
                    try:
                        with open(xyz_path, 'r', encoding='utf-8', errors='replace') as xf:
                            xyz_lines_raw = xf.readlines()
                        if len(xyz_lines_raw) >= 3:
                            natoms = int(xyz_lines_raw[0].strip())
                            coord_lines = xyz_lines_raw[2:2 + natoms]
                            if len(coord_lines) == natoms:
                                result = []
                                result.append(f"{natoms}\n")
                                result.append("Final geometry for re-optimization\n")
                                for cl in coord_lines:
                                    parts = cl.strip().split()
                                    if len(parts) >= 4:
                                        result.append(f"{parts[0]:2s}  {parts[1]:>12s}  {parts[2]:>12s}  {parts[3]:>12s}\n")
                                if len(result) >= natoms + 2:
                                    return result
                    except Exception:
                        continue
            return None

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
