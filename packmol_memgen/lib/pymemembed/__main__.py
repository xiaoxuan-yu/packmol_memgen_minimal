"""
pymemembed CLI - Python/Numba implementation of MEMEMBED.

Usage:
    pymemembed [options] <input.pdb>

Replicates the C++ memembed command-line interface and can be used as
a drop-in replacement.
"""

import sys
import argparse
import os

try:
    from . import memembed_align, calculate_energy
    from .memembed_wrapper import run_ga_multi
except ModuleNotFoundError as exc:
    if exc.name == "numba":
        print(
            "Error: pymemembed requires numba. Install it with: "
            "pip install packmol-memgen-minimal",
            file=sys.stderr,
        )
        sys.exit(1)
    raise


def main():
    parser = argparse.ArgumentParser(
        prog='pymemembed',
        description='Membrane protein orientation using knowledge-based potentials.',
        usage='pymemembed [options] <input.pdb>'
    )
    parser.add_argument('input', help='Input PDB file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output PDB file (default: <input>_EMBED.pdb)')
    parser.add_argument('-s', '--search', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Search type: 0=GA (default), 1=Grid, 2=Direct, 3=GA x5')
    parser.add_argument('-n', '--nter', default='in', choices=['in', 'out'],
                        help='N-terminal location constraint (default: in)')
    parser.add_argument('-b', '--barrel', action='store_true',
                        help='Beta-barrel mode')
    parser.add_argument('-l', '--span', action='store_true',
                        help='Force membrane spanning')
    parser.add_argument('-p', '--polar', action='store_true',
                        help='Draw lines representing polar head groups (±24 Å markers)')
    parser.add_argument('-a', '--threads', type=int, default=4,
                        help='Number of threads (default: 4)')
    parser.add_argument('-c', '--chains', default=None,
                        help='Comma-separated chain list (e.g., A,B)')
    parser.add_argument('-e', '--energy-only', action='store_true',
                        help='Just compute energy at identity orientation (no optimization)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('--max-calls', type=int, default=1000000,
                        help='Maximum function evaluations (default: 1000000)')
    parser.add_argument('--n-runs', type=int, default=5,
                        help='Number of GA runs for -s 3 mode (default: 5)')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        return 1

    # Parse chains
    chains = args.chains.split(',') if args.chains else None

    # Energy-only mode
    if args.energy_only:
        energy = calculate_energy(args.input, xrt=0.0, yrt=0.0, z_trans=0.0,
                                  beta_barrel=args.barrel, chains=chains)
        print(f"Energy at identity orientation: {energy:.10f}")
        return 0

    # Set default output file
    output = args.output
    if output is None:
        base = args.input.rsplit('.pdb', 1)[0] if args.input.endswith('.pdb') else args.input
        output = base + '_EMBED.pdb'

    # Map search type to method (matches C++ memembed -s flag)
    search_map = {0: 'ga', 1: 'grid', 2: 'direct', 3: 'ga_multi'}
    method = search_map[args.search]

    verbose = not args.quiet

    if method == 'ga_multi':
        result = run_ga_multi(
            pdb_file=args.input,
            output_file=output,
            beta_barrel=args.barrel,
            threads=args.threads,
            n_runs=args.n_runs,
            max_calls_per_run=args.max_calls,
            n_ter=args.nter,
            force_span=args.span,
            chains=chains,
            verbose=verbose
        )
    else:
        result = memembed_align(
            pdb_file=args.input,
            output_file=output,
            method=method,
            threads=args.threads,
            max_calls=args.max_calls,
            beta_barrel=args.barrel,
            force_span=args.span,
            chains=chains,
            n_ter=args.nter,
            verbose=verbose,
            polar_headgroups=args.polar
        )

    if verbose:
        print(f"\nResults:")
        print(f"  Energy:        {result['energy']:.10f}")
        print(f"  X rotation:    {result['x_rotation']:.6f} deg")
        print(f"  Y rotation:    {result['y_rotation']:.6f} deg")
        print(f"  Z translation: {result['z_translation']:.6f} A")
        print(f"  Evaluations:   {result['n_calls']:,}")
        print(f"  Output:        {result['output_file']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
