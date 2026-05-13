"""
Wrapper functions for integrating pymemembed into packmol-memgen main.py

This module provides drop-in replacement functions for the C++ MEMEMBED calls
in PACKMOLMemgen class.
"""

import os
import sys
import numpy as np


def memembed_align_replacement(self, pdb, keepligs=False, double_span=False,
                               verbose=False, overwrite=False, barrel=False,
                               n_ter="in", opt="3", use_python=True):
    """
    Drop-in replacement for PACKMOLMemgen.memembed_align() using pymemembed.

    This function has the exact same interface as the original C++ version
    but uses the Python/Numba implementation instead.

    Args:
        self: PACKMOLMemgen instance (for logger, etc.)
        pdb (str): Input PDB file path
        keepligs (bool): Keep ligands by superimposing
        double_span (bool): Handle double-spanning membrane proteins
        verbose (bool): Verbose logging
        overwrite (bool): Overwrite existing output
        barrel (bool): Use beta-barrel potential
        n_ter (str): N-terminal orientation ("in" or "out")
        opt (str): Optimization option (matches C++ memembed -s flag):
            - "0": GA single run (C++ default)
            - "1": Grid search (exhaustive)
            - "2": Direct search (Hooke-Jeeves)
            - "3": GA x 5 runs, pick best (default)
        use_python (bool): Use Python implementation (for future C++/Python toggle)

    Returns:
        str or tuple: Output PDB path, or (output_path, z_dist) if double_span
    """
    output = pdb[:-4] + n_ter + "_EMBED.pdb"

    # Check if output exists and skip if not overwriting
    if os.path.exists(output) and not overwrite:
        if verbose:
            print(f"Output {output} exists, skipping orientation")
        if double_span:
            return handle_double_span(self, pdb, output, barrel, n_ter, opt, verbose)
        elif keepligs:
            return handle_keep_ligs(self, pdb, output, verbose)
        else:
            return output

    # Import here to avoid circular imports and to allow fallback if pymemembed not available
    try:
        # Try relative import first (when called from main.py)
        from . import memembed_align as run_pymemembed
    except ImportError:
        try:
            # Try absolute import (when installed as package)
            from packmol_memgen.lib.pymemembed import memembed_align as run_pymemembed
        except ImportError:
            print("ERROR: pymemembed module not found. Install numba: pip install numba")
            raise

    # Determine optimization method (matches C++ memembed -s flag)
    method_map = {
        "0": "ga",      # GA (single run) - C++ default
        "1": "grid",    # Grid search
        "2": "direct",  # Direct search
        "3": "ga_multi" # GA x 5 runs, pick best
    }
    method = method_map.get(opt, "ga_multi")

    # Determine number of threads
    threads = getattr(self, 'cpus', 1)

    # Run optimization
    if verbose:
        print(f"\nRunning pymemembed with method={method}, beta_barrel={barrel}")

    if method == "ga_multi":
        # Option 3: Run 5 GA optimizations, pick best
        result = run_ga_multi(
            pdb_file=pdb,
            output_file=output,
            beta_barrel=barrel,
            threads=threads,
            n_runs=5,
            max_calls_per_run=1000000,  # Max function calls per GA run
            n_ter=n_ter,
            verbose=verbose
        )
    else:
        # Single optimization run
        result = run_pymemembed(
            pdb_file=pdb,
            output_file=output,
            method=method,
            beta_barrel=barrel,
            threads=threads,
            max_calls=1000000,
            n_ter=n_ter,
            verbose=verbose
        )

    if verbose:
        print(f"pymemembed complete: Energy = {result['energy']:.4f}")

    # Write log file (compatible with C++ log format)
    log_file = output.replace("_EMBED.pdb", "_memembed.log")
    write_memembed_log(log_file, result, opt, barrel, n_ter)

    # Handle post-processing options
    if keepligs:
        return handle_keep_ligs(self, pdb, output, verbose)
    elif double_span:
        return handle_double_span(self, pdb, output, barrel, n_ter, opt, verbose)
    else:
        return output


def _run_single_ga(args):
    """Worker function for parallel GA runs (must be top-level for pickling)."""
    pdb_file, max_calls, beta_barrel, force_span, chains, n_ter, threads_per_run = args

    # Re-seed numpy RNG from OS entropy — forked processes inherit parent's
    # RNG state and would otherwise produce identical GA populations
    np.random.seed()

    try:
        from . import _optimize
    except ImportError:
        from packmol_memgen.lib.pymemembed import _optimize

    result = _optimize(
        pdb_file=pdb_file,
        method='ga',
        threads=threads_per_run,
        max_calls=max_calls,
        beta_barrel=beta_barrel,
        force_span=force_span,
        chains=chains,
        n_ter=n_ter,
        verbose=False,
        protein=None  # Each process parses PDB independently
    )
    # Return only picklable data (exclude PDB object)
    return {
        'genome': result['genome'].tolist(),
        'energy': result['energy'],
        'n_calls': result['n_calls'],
        'flip': result['flip'],
        'method': result['method']
    }


def run_ga_multi(pdb_file, output_file, beta_barrel=False, threads=4,
                n_runs=5, max_calls_per_run=1000000, n_ter='in',
                force_span=False, chains=None, verbose=True):
    """
    Run multiple GA optimizations and pick the best result.

    This implements MEMEMBED option 3: run N independent GA optimizations
    with different random seeds and select the one with lowest energy.

    When threads > 1, runs are executed in parallel using ProcessPoolExecutor
    (each GA run is independent). The output PDB is written once for the best
    result.

    Args:
        pdb_file (str): Input PDB file
        output_file (str): Output PDB file
        beta_barrel (bool): Use beta-barrel potential
        threads (int): Total number of CPU threads to use
        n_runs (int): Number of independent GA runs (default: 5)
        max_calls_per_run (int): Max function calls per run
        n_ter (str): N-terminal orientation ('in' or 'out')
        force_span (bool): Enforce membrane-spanning constraint
        chains (list): Chain IDs to process (None = all)
        verbose (bool): Print progress

    Returns:
        dict: Result from best GA run with keys: energy, x_rotation,
              y_rotation, z_translation, n_calls, output_file, method
    """
    # Import locally to avoid circular imports
    try:
        from . import _optimize
        from .pdb import PDB
    except ImportError:
        from packmol_memgen.lib.pymemembed import _optimize
        from packmol_memgen.lib.pymemembed.pdb import PDB

    if verbose:
        print(f"\nRunning {n_runs} independent GA optimizations...")
        print(f"  Max calls per run: {max_calls_per_run:,}")
        print(f"  Threads: {threads}")

    # Parse PDB for output writing (main process only)
    if verbose:
        print("\nParsing PDB file...")
    protein = PDB(pdb_file, chains=chains)
    if verbose:
        print(f"  Extracted {len(protein.backbone['x'])} CA/CB atoms")

    # Determine parallelism: use processes for GA runs, Numba threads within
    n_workers = min(n_runs, threads)
    threads_per_run = max(1, threads // n_workers)

    if n_workers > 1:
        # Parallel execution: each GA run in its own process
        # Use 'spawn' context to avoid OpenMP fork-safety issues in long-lived
        # processes (e.g. web server) where Numba/OpenMP may already be initialized
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        if verbose:
            print(f"  Parallel: {n_workers} workers x {threads_per_run} Numba threads")

        worker_args = [
            (pdb_file, max_calls_per_run, beta_barrel, force_span, chains, n_ter, threads_per_run)
            for _ in range(n_runs)
        ]

        results = []
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = list(executor.map(_run_single_ga, worker_args))
            results = list(futures)

        total_calls = sum(r['n_calls'] for r in results)
        best_opt = min(results, key=lambda r: r['energy'])
        best_run_idx = results.index(best_opt)

        if verbose:
            for i, r in enumerate(results):
                marker = " <-- best" if i == best_run_idx else ""
                print(f"  Run {i+1}/{n_runs}: Energy = {r['energy']:.6f}{marker}")
    else:
        # Sequential execution: reuse parsed PDB object
        best_opt = None
        best_energy = float('inf')
        best_run_idx = 0
        total_calls = 0

        for run_idx in range(n_runs):
            if verbose:
                print(f"\n{'='*70}")
                print(f"GA Run {run_idx + 1}/{n_runs}")
                print(f"{'='*70}")

            opt_result = _optimize(
                pdb_file=pdb_file,
                method='ga',
                threads=threads_per_run,
                max_calls=max_calls_per_run,
                beta_barrel=beta_barrel,
                force_span=force_span,
                chains=chains,
                n_ter=n_ter,
                verbose=verbose,
                protein=protein
            )

            total_calls += opt_result['n_calls']

            if verbose:
                print(f"Run {run_idx + 1} complete: Energy = {opt_result['energy']:.6f}")

            if opt_result['energy'] < best_energy:
                best_energy = opt_result['energy']
                best_opt = {
                    'genome': opt_result['genome'].tolist(),
                    'energy': opt_result['energy'],
                    'n_calls': opt_result['n_calls'],
                    'flip': opt_result['flip'],
                    'method': opt_result['method']
                }
                best_run_idx = run_idx

    best_energy = best_opt['energy']
    xrt, yrt, z_trans = best_opt['genome']

    if verbose:
        print(f"\n{'='*70}")
        print(f"Best result: Run {best_run_idx + 1} with energy = {best_energy:.6f}")
        print(f"{'='*70}")
        print("\nWriting output PDB...")

    protein.write_oriented_pdb(
        output_file,
        xrt=xrt,
        yrt=yrt,
        z_trans=z_trans,
        energy=best_energy,
        polar_headgroups=False,
        extra_shift=0.0,
        cyto_shift=0.0,
        flip=best_opt['flip']
    )

    return {
        'energy': best_energy,
        'x_rotation': float(np.degrees(xrt)),
        'y_rotation': float(np.degrees(yrt)),
        'z_translation': float(z_trans),
        'n_calls': total_calls,
        'output_file': output_file,
        'method': 'ga_multi'
    }


def handle_keep_ligs(self, pdb, output, verbose):
    """
    Handle keepligs option by superimposing oriented protein with original.

    Args:
        self: PACKMOLMemgen instance
        pdb: Original PDB file
        output: Oriented PDB file
        verbose: Verbose logging

    Returns:
        str: Path to output with ligands
    """
    if verbose:
        print("Superimposing to keep ligands")

    # Import from pdbremix
    try:
        from ..pdbremix.rmsd import rmsd_of_pdbs
    except ImportError:
        from packmol_memgen.lib.pdbremix.rmsd import rmsd_of_pdbs

    output_with_ligs = output.replace("_EMBED.pdb", "_EMBED_ligs.pdb")
    rmsd_of_pdbs(pdb, output, transform_pdb1=output_with_ligs, standard=True)

    return output_with_ligs


def handle_double_span(self, pdb, output, barrel, n_ter, opt, verbose):
    """
    Handle double-spanning membrane proteins.

    Re-orients the non-membrane-spanning portion of the protein.

    Args:
        self: PACKMOLMemgen instance
        pdb: Original PDB file
        output: Initial oriented PDB file
        barrel: Beta-barrel mode
        n_ter: N-terminal orientation
        opt: Optimization option
        verbose: Verbose logging

    Returns:
        tuple: (output_path, z_distance)
    """
    # Import utility functions
    try:
        from ..utils import pdb_parse, pdb_write, translate_pdb, superimpose_pdb
    except ImportError:
        from packmol_memgen.lib.utils import pdb_parse, pdb_write, translate_pdb, superimpose_pdb

    # Parse oriented PDB
    outpdb = pdb_parse(output, onlybb=False)

    # Find membrane boundaries (DUM atoms)
    low_bound = np.mean([outpdb[res][atom] for res in outpdb.keys()
                        if res[0] == "DUM" for atom in outpdb[res]
                        if atom[0] == "N"], axis=0)
    up_bound = np.mean([outpdb[res][atom] for res in outpdb.keys()
                       if res[0] == "DUM" for atom in outpdb[res]
                       if atom[0] == "O"], axis=0)

    offset = 0
    mem1_cen = np.mean([low_bound, up_bound], axis=0)

    # Extract residues outside membrane
    tmless = {res: outpdb[res] for res in outpdb.keys()
             if not any([outpdb[res][atom][2] >= low_bound[2] - offset
                        and outpdb[res][atom][2] <= up_bound[2] + offset
                        for atom in outpdb[res]])}
    tmless[('MEM', 1, 'X')] = {('MEM', 1): mem1_cen}

    # Write temporary PDB
    pdb_write(tmless, outfile="temp1.pdb")

    # Re-orient the second membrane domain
    try:
        from . import memembed_align as run_pymemembed
    except ImportError:
        from packmol_memgen.lib.pymemembed import memembed_align as run_pymemembed

    result2 = run_pymemembed(
        pdb_file="temp1.pdb",
        output_file="temp2.pdb",
        method="ga" if opt in ["0", "3"] else "grid" if opt == "1" else "direct",
        beta_barrel=barrel,
        threads=getattr(self, 'cpus', 1),
        max_calls=1000000,
        n_ter=n_ter,
        verbose=verbose
    )

    # Parse second orientation
    temppdb = pdb_parse("temp2.pdb", onlybb=False)
    low_bound = np.mean([temppdb[res][atom] for res in temppdb.keys()
                        if res[0] == "DUM" for atom in temppdb[res]
                        if atom[0] == "N"], axis=0)
    up_bound = np.mean([temppdb[res][atom] for res in temppdb.keys()
                       if res[0] == "DUM" for atom in temppdb[res]
                       if atom[0] == "O"], axis=0)

    offset = 5
    mem2_cen = np.mean([low_bound, up_bound], axis=0)

    # Remove dummy atoms and add membrane center markers
    dumless = {res: temppdb[res] for res in temppdb.keys() if res[0] != "DUM"}
    dumless[('MEM', 2, 'X')] = {('MEM', 2): mem2_cen}

    # Calculate Z distance and translate if needed
    z_dist = dumless[('MEM', 1, 'X')][('MEM', 1)][2] + dumless[('MEM', 2, 'X')][('MEM', 2)][2]
    if z_dist < 0:
        dumless = translate_pdb(dumless, vec=[0, 0, z_dist])
        z_dist *= -1

    pdb_write(dumless, outfile="presuper.pdb")

    # Superimpose with original orientation
    translated = superimpose_pdb(dumless, outpdb)
    final_output = output.replace("EMBED", "EMBED_double")
    pdb_write(translated, outfile=final_output)

    # Clean up
    for temp_file in ["temp1.pdb", "temp2.pdb", "presuper.pdb"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return (final_output, z_dist)


def write_memembed_log(log_file, result, opt, barrel, n_ter):
    """
    Write log file in MEMEMBED-compatible format.

    Args:
        log_file (str): Output log file path
        result (dict): Result dictionary from pymemembed
        opt (str): Optimization option
        barrel (bool): Beta-barrel mode
        n_ter (str): N-terminal orientation
    """
    with open(log_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PYMEMEMBED - Python/Numba MEMEMBED Implementation\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Optimization method: {opt}\n")
        if opt == "0":
            f.write("  (Genetic algorithm - single run)\n")
        elif opt == "1":
            f.write("  (Grid search - exhaustive)\n")
        elif opt == "2":
            f.write("  (Direct search - Hooke-Jeeves)\n")
        elif opt == "3":
            f.write("  (Genetic algorithm - 5 runs, best selected)\n")

        f.write(f"Potential: {'Beta-barrel' if barrel else 'Alpha-helical'}\n")
        f.write(f"N-terminal: {n_ter}\n\n")

        f.write("Results:\n")
        f.write(f"  Final energy: {result['energy']:.10f}\n")
        f.write(f"  X rotation: {result['x_rotation']:.6f} degrees\n")
        f.write(f"  Y rotation: {result['y_rotation']:.6f} degrees\n")
        f.write(f"  Z translation: {result['z_translation']:.6f} Angstroms\n")
        f.write(f"  Function evaluations: {result['n_calls']:,}\n")
        f.write(f"  Method: {result['method']}\n\n")

        f.write("=" * 70 + "\n")
        f.write("Orientation complete\n")
        f.write("=" * 70 + "\n")
