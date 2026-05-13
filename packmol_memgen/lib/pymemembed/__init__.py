"""
pymemembed - Python/Numba implementation of MEMEMBED

A knowledge-based membrane protein orientation tool using statistical potentials.
This is a pure Python implementation with Numba JIT compilation for performance
using Claude Code.

Usage:
    from pymemembed import memembed_align

    result = memembed_align('protein.pdb', 'output.pdb',
                           method='ga', threads=4)
    print(f"Energy: {result['energy']}")
"""

__version__ = '1.0.0'
__author__ = 'Stephan Schott-Verdugo'

import numpy as np
from .pdb import PDB
from .core import MEMPOT, BETAMEMPOT, orientate
from .optimizers import run_ga, run_grid, run_direct


def _get_nterm(backbone_x, backbone_y, backbone_z, xrt, yrt, z_trans):
    """
    Determine N-terminal location after transformation.

    Matches C++ get_nterm() in pdb.cpp:315-336

    Args:
        backbone_x/y/z: Backbone coordinate arrays
        xrt: X-axis rotation (radians)
        yrt: Y-axis rotation (radians)
        z_trans: Z-axis translation

    Returns:
        str: 'in' if N-terminus is inside (Z < 0), 'out' if outside (Z > 0)
    """
    # Get first residue coordinates (N-terminus)
    x = backbone_x[0]
    y = backbone_y[0]
    z = backbone_z[0]

    # Apply transformations (same as orientate function)
    # Pre-compute trig values
    cx = np.cos(xrt)
    sx = np.sin(xrt)
    cy = np.cos(yrt)
    sy = np.sin(yrt)

    # X-axis rotation
    yp = y * cx - z * sx
    zp = y * sx + z * cx
    y = yp
    z = zp

    # Y-axis rotation
    zp = z * cy - x * sy
    # Don't need x coordinate for Z check

    # Z-axis translation
    z = zp + z_trans

    # Check if inside (Z < 0) or outside (Z > 0)
    if z > 0:
        return 'out'
    else:
        return 'in'


def _optimize(pdb_file, method='ga', threads=4, max_calls=1000000,
              beta_barrel=False, force_span=False, chains=None,
              n_ter='in', verbose=True, protein=None):
    """
    Run membrane orientation optimization without writing output.

    Internal function used by memembed_align() and run_ga_multi() to
    separate the optimization step from PDB I/O, avoiding redundant file
    writes during multi-GA runs.

    Args:
        pdb_file (str): Input PDB file path (used if protein is None)
        method (str): Optimization method: 'ga', 'grid', or 'direct'
        threads (int): Number of parallel threads (GA only)
        max_calls (int): Maximum function evaluations
        beta_barrel (bool): Use beta-barrel potential
        force_span (bool): Enforce membrane-spanning constraint
        chains (list): Chain IDs to process (None = all)
        n_ter (str): N-terminal orientation constraint ('in', 'out', or '')
        verbose (bool): Print progress messages
        protein (PDB, optional): Pre-parsed PDB object. If provided, skips
            parsing pdb_file (enables reuse across multiple GA runs).

    Returns:
        dict with keys:
            'protein' (PDB): Parsed PDB object (for subsequent write)
            'genome' (ndarray): Best [xrt, yrt, z_trans]
            'energy' (float): Best energy found
            'n_calls' (int): Number of function evaluations
            'flip' (bool): Whether N-terminal flip is needed
            'method' (str): Optimization method used
    """
    valid_methods = ['ga', 'grid', 'direct']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Choose from: {valid_methods}")

    # Parse PDB file or reuse provided protein object
    if protein is None:
        if verbose:
            print("\nParsing PDB file...")
        protein = PDB(pdb_file, chains=chains)
        if verbose:
            backbone_x = protein.backbone['x']
            print(f"  Extracted {len(backbone_x)} CA/CB atoms")
            if chains:
                print(f"  Chains: {', '.join(chains)}")

    backbone_x, backbone_y, backbone_z, backbone_res = protein.get_backbone_arrays()
    max_c_dist = protein.max_c_dist

    # Select membrane potential
    mempot = BETAMEMPOT if beta_barrel else MEMPOT

    # Run optimization
    if method == 'ga':
        best_genome, best_energy, n_calls = run_ga(
            backbone_x, backbone_y, backbone_z, backbone_res, mempot,
            max_calls=max_calls, threads=threads, force_span=force_span,
            verbose=verbose, max_c_dist=max_c_dist
        )
    elif method == 'grid':
        best_genome, best_energy, n_calls = run_grid(
            backbone_x, backbone_y, backbone_z, backbone_res, mempot,
            force_span=force_span, verbose=verbose, max_c_dist=max_c_dist
        )
    elif method == 'direct':
        best_genome, best_energy, n_calls = run_direct(
            backbone_x, backbone_y, backbone_z, backbone_res, mempot,
            max_calls=max_calls, force_span=force_span, verbose=verbose,
            max_c_dist=max_c_dist
        )

    # Check N-terminal orientation and flip if needed (matches C++ main.cpp:313-316)
    xrt, yrt, z_trans = best_genome[0], best_genome[1], best_genome[2]
    flip = False
    if n_ter and n_ter != '':
        actual_nterm = _get_nterm(backbone_x, backbone_y, backbone_z, xrt, yrt, z_trans)
        if actual_nterm != n_ter:
            if verbose:
                print(f"\nN-terminus is '{actual_nterm}' but '{n_ter}' requested")
                print("Inverting to satisfy N-terminal constraint...")
            flip = True

    return {
        'protein': protein,
        'genome': best_genome,
        'energy': float(best_energy),
        'n_calls': int(n_calls),
        'flip': flip,
        'method': method
    }


def memembed_align(pdb_file, output_file=None, method='ga',
                   threads=4, max_calls=1000000, beta_barrel=False,
                   force_span=False, chains=None, n_ter='in', verbose=True,
                   polar_headgroups=False):
    """
    Align protein to membrane using knowledge-based potential.

    This function performs membrane protein orientation by optimizing
    the position and orientation of a protein structure relative to
    a lipid bilayer using statistical membrane potentials.

    Args:
        pdb_file (str): Input PDB file path
        output_file (str, optional): Output PDB path. If None, uses
            input filename with '_EMBED.pdb' suffix
        method (str): Optimization method:
            - 'ga': Genetic algorithm (default, recommended)
            - 'grid': Exhaustive grid search (slow, thorough)
            - 'direct': Hooke-Jeeves direct search (local optimization)
        threads (int): Number of parallel threads (GA only)
        max_calls (int): Maximum function evaluations
        beta_barrel (bool): Use beta-barrel potential instead of alpha-helical
        force_span (bool): Enforce membrane-spanning constraint (adds penalty
            if protein doesn't span from Z > +17.5 to Z < -17.5)
        chains (list): List of chain IDs to process (e.g., ['A', 'B']).
            If None, processes all chains
        n_ter (str): N-terminal orientation constraint:
            - 'in': N-terminus should be inside (cytoplasmic, Z < 0)
            - 'out': N-terminus should be outside (extracellular, Z > 0)
            - '': No constraint (default orientation from optimization)
        verbose (bool): Print progress messages
        polar_headgroups (bool): Draw lines representing polar head groups
            at ±24 Å in addition to hydrophobic core markers at ±15 Å

    Returns:
        dict: Results dictionary with keys:
            - 'energy' (float): Final membrane energy
            - 'x_rotation' (float): X-axis rotation in degrees
            - 'y_rotation' (float): Y-axis rotation in degrees
            - 'z_translation' (float): Z-axis translation in Angstroms
            - 'n_calls' (int): Number of function evaluations
            - 'output_file' (str): Path to output PDB file
            - 'method' (str): Optimization method used

    Raises:
        ValueError: If PDB file has no CA/CB atoms or invalid method
        FileNotFoundError: If input PDB file doesn't exist

    Example:
        >>> result = memembed_align('1BL8.pdb', method='ga', threads=8)
        >>> print(f"Membrane energy: {result['energy']:.2f}")
        >>> print(f"Orientation: {result['x_rotation']:.1f}° / {result['y_rotation']:.1f}°")

    Notes:
        - Uses CA atoms for glycine, CB atoms for all other residues
        - Membrane model: 48 Å thick (-24 to +24 Å), 34 slices of 1.5 Å
        - Output PDB includes dummy atoms (HETATM) marking membrane planes:
            * O atoms at Z = +15 Å (extracellular/hydrophobic boundary)
            * N atoms at Z = -15 Å (cytoplasmic/hydrophobic boundary)
        - Performance: GA typically completes in 1-5 minutes on modern hardware
    """
    # Set default output file
    if output_file is None:
        output_file = pdb_file.replace('.pdb', '_EMBED.pdb')
        if output_file == pdb_file:  # No .pdb extension
            output_file = pdb_file + '_EMBED.pdb'

    if verbose:
        print("=" * 70)
        print("PYMEMEMBED - Membrane Protein Orientation")
        print("=" * 70)
        print(f"Input:  {pdb_file}")
        print(f"Output: {output_file}")
        print(f"Method: {method.upper()}")
        if beta_barrel:
            print("Potential: Beta-barrel")
        else:
            print("Potential: Alpha-helical")
        print("=" * 70)

    opt_result = _optimize(
        pdb_file, method=method, threads=threads, max_calls=max_calls,
        beta_barrel=beta_barrel, force_span=force_span, chains=chains,
        n_ter=n_ter, verbose=verbose
    )

    protein = opt_result['protein']
    xrt, yrt, z_trans = opt_result['genome']

    # Write output PDB
    if verbose:
        print("\nWriting output PDB...")
    protein.write_oriented_pdb(
        output_file,
        xrt=xrt,
        yrt=yrt,
        z_trans=z_trans,
        energy=opt_result['energy'],
        polar_headgroups=polar_headgroups,
        extra_shift=0.0,
        cyto_shift=0.0,
        flip=opt_result['flip']
    )

    # Return results
    result = {
        'energy': opt_result['energy'],
        'x_rotation': float(np.degrees(xrt)),
        'y_rotation': float(np.degrees(yrt)),
        'z_translation': float(z_trans),
        'n_calls': opt_result['n_calls'],
        'output_file': output_file,
        'method': opt_result['method']
    }

    if verbose:
        print("=" * 70)
        print("MEMEMBED COMPLETE")
        print("=" * 70)

    return result


# Convenience function for programmatic use
def calculate_energy(pdb_file, xrt, yrt, z_trans, beta_barrel=False, chains=None):
    """
    Calculate membrane energy for given orientation parameters.

    Useful for evaluating a specific orientation without optimization.

    Args:
        pdb_file (str): Input PDB file path
        xrt (float): X-axis rotation in radians
        yrt (float): Y-axis rotation in radians
        z_trans (float): Z-axis translation in Angstroms
        beta_barrel (bool): Use beta-barrel potential
        chains (list): Chain IDs to process

    Returns:
        float: Membrane energy
    """
    # Parse PDB
    protein = PDB(pdb_file, chains=chains)
    backbone_x, backbone_y, backbone_z, backbone_res = protein.get_backbone_arrays()

    # Select potential
    mempot = BETAMEMPOT if beta_barrel else MEMPOT

    # Calculate energy
    energy = orientate(backbone_x, backbone_y, backbone_z, backbone_res,
                      xrt, yrt, z_trans, mempot, force_span=False)

    return energy


# Export public API
__all__ = [
    'memembed_align',
    'calculate_energy',
    'PDB',
    'MEMPOT',
    'BETAMEMPOT'
]
