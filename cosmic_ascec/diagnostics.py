"""``analyze_box`` / ``test_box`` — verbatim port of v04's box-length tools.

v04's ``if __name__ == "__main__"`` block (ascec-v04.py 21172-21221) routes
``analyze_box <xyz> [N]`` to :func:`analyze_box_length_from_xyz` (v04 lines
20985-21116) and ``test_box`` to :func:`test_box_length_analysis` (lines
21120-21169). v04 keeps the same routing — the root shim ``ascec.py`` dispatches
``sys.argv[1] in {"analyze_box", "test_box"}`` here.

Both functions are byte-identical extracts of ``ascec-v04.py`` (D-039: faithful
decomposition). They consume ``SystemState`` / ``MoleculeData`` /
``calculate_optimal_box_length`` / ``extract_configurations_from_xyz`` /
``element_symbols`` from :mod:`cosmic_ascec.workflow.stages` (R6d ports); the
import shim below pulls those into the module namespace so the verbatim function
bodies resolve their bare-name calls. The block itself is produced by
``scripts/extract_r7_diagnostics.py`` — re-running the script reproduces the
in-tree section byte-for-byte.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

# Pull the names the verbatim bodies need into this module's namespace. v04
# resolves them as bare names in its single-file source; v04 ports them as
# ``cosmic_ascec.workflow.stages`` members (R6d), so the imports below are the
# only mechanical adaptation D-039 permits.
from cosmic_ascec.workflow.stages import (
    MoleculeData,
    SystemState,
    calculate_optimal_box_length,
    element_symbols,
    extract_configurations_from_xyz,
)


# =========================================================================== #
# R7 verbatim ports — analyze_box / test_box                                   #
#                                                                             #
# Byte-identical extracts of ``ascec-v04.py`` (D-039). Produced by             #
# ``scripts/extract_r7_diagnostics.py``; re-running the extractor reproduces  #
# this section byte-for-byte.                                                  #
# =========================================================================== #


# --- def analyze_box_length_from_xyz  (ascec-v04.py 20985-21116) ---
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


# --- def test_box_length_analysis  (ascec-v04.py 21120-21169) ---
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



__all__ = ["analyze_box_length_from_xyz", "test_box_length_analysis"]
