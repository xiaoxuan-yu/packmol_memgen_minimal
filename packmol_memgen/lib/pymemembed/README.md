# SSchott pymemembed

Python/Numba implementation of MEMEMBED — a knowledge-based membrane protein
orientation tool using statistical potentials, developed with Claude Code.

## Features

- **Installable CLI** — `pymemembed` command available after `pip install .`
- **Drop-in replacement** for the C++ memembed binary (same `-s`, `-n`, `-b`, `-o`, `-l`, `-a`, `-c`, `-e`, `-p` flags)
- **70-85% of C++ performance** with Numba JIT compilation
- **Pure Python** — no compilation required
- **Module interface** — direct function calls, no subprocess overhead
- **Parallel execution** — multi-threaded GA optimizer
- **Reduced I/O for multi-GA** — PDB parsed once, written once for the best result
- **Minimal dependencies** — only NumPy and Numba

## Installation

Installed automatically with packmol-memgen:

```bash
pip install .
```

This registers the `pymemembed` console script alongside `packmol-memgen`.

## CLI Usage

```bash
# Basic GA optimization (default)
pymemembed protein.pdb

# Specify output file
pymemembed -o oriented.pdb protein.pdb

# Search types (match C++ memembed -s flag)
pymemembed -s 0 protein.pdb    # GA single run (default)
pymemembed -s 1 protein.pdb    # Grid search
pymemembed -s 2 protein.pdb    # Direct search
pymemembed -s 3 protein.pdb    # GA x 5, pick best

# N-terminal constraint
pymemembed -n in protein.pdb   # N-term cytoplasmic (default)
pymemembed -n out protein.pdb  # N-term extracellular

# Beta-barrel mode
pymemembed -b protein.pdb

# Multi-threaded
pymemembed -a 8 protein.pdb

# Specific chains
pymemembed -c A,B protein.pdb

# Energy only (no optimization)
pymemembed -e protein.pdb

# Force membrane spanning
pymemembed -l protein.pdb

# Polar headgroup markers in output (±24 Å)
pymemembed -p protein.pdb
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o` | Output PDB file | `<input>_EMBED.pdb` |
| `-s` | Search type: 0=GA, 1=Grid, 2=Direct, 3=GA×5 | 0 |
| `-n` | N-terminal: `in` or `out` | `in` |
| `-b` | Beta-barrel mode | off |
| `-l` | Force membrane spanning | off |
| `-p` | Polar head group markers at ±24 Å | off |
| `-a` | Number of threads | 4 |
| `-c` | Chain list (comma-separated) | all |
| `-e` | Energy only (no optimization) | off |
| `-q` | Quiet (suppress output) | off |
| `--max-calls` | Max function evaluations | 1000000 |
| `--n-runs` | GA runs for `-s 3` mode | 5 |

## Python API

### Basic Usage

```python
from packmol_memgen.lib.pymemembed import memembed_align

result = memembed_align('protein.pdb', 'output.pdb', method='ga', threads=4)

print(f"Energy: {result['energy']:.2f}")
print(f"Orientation: X={result['x_rotation']:.1f} Y={result['y_rotation']:.1f}")
print(f"Translation: Z={result['z_translation']:.2f} A")
```

### Advanced Usage

```python
# Beta-barrel potential
result = memembed_align('barrel.pdb', beta_barrel=True)

# Membrane spanning constraint
result = memembed_align('protein.pdb', force_span=True)

# Specific chains
result = memembed_align('complex.pdb', chains=['A', 'B'])

# N-terminal outside
result = memembed_align('protein.pdb', n_ter='out')

# Polar headgroup markers in output
result = memembed_align('protein.pdb', polar_headgroups=True)
```

### Calculate Energy

```python
from packmol_memgen.lib.pymemembed import calculate_energy
import numpy as np

energy = calculate_energy('protein.pdb',
                          xrt=np.radians(45.0),
                          yrt=np.radians(60.0),
                          z_trans=10.0)
```

### API Reference

**`memembed_align(pdb_file, output_file=None, method='ga', ...)`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_file` | str | required | Input PDB file |
| `output_file` | str | `<input>_EMBED.pdb` | Output PDB file |
| `method` | str | `'ga'` | `'ga'`, `'grid'`, or `'direct'` |
| `threads` | int | 4 | Parallel threads (GA only) |
| `max_calls` | int | 1000000 | Max function evaluations |
| `beta_barrel` | bool | False | Beta-barrel potential |
| `force_span` | bool | False | Membrane spanning constraint |
| `chains` | list | None | Chain IDs to process |
| `n_ter` | str | `'in'` | `'in'`, `'out'`, or `''` |
| `verbose` | bool | True | Print progress |
| `polar_headgroups` | bool | False | Add ±24 Å polar headgroup markers |

**Returns:** dict with keys `energy`, `x_rotation`, `y_rotation`,
`z_translation`, `n_calls`, `output_file`, `method`.

## Output Format

Output PDB includes:
1. HEADER lines with optimization results
2. Transformed protein coordinates
3. HETATM dummy atoms marking membrane planes:
   - O atoms at Z = +15 Å (extracellular hydrophobic boundary)
   - N atoms at Z = -15 Å (cytoplasmic hydrophobic boundary)
   - O/N atoms at ±24 Å (polar headgroup boundaries, if `-p` used)

```
HEADER MEMBRANE_ENERGY 1234.567890
HEADER X_ROTATION      45.234567
HEADER Y_ROTATION      67.891234
HEADER Z_TRANSLATION   12.345678
```

## Multi-GA I/O Optimization

When using `-s 3` (GA×5) mode, the PDB file is parsed **once** and reused
across all runs. No intermediate PDB files are written; only the best result
is written at the end. This reduces file I/O significantly for large proteins.

## Testing

```bash
# Unit tests only (no PDB needed)
python test_pymemembed.py --unit

# Full suite with PDB
python test_pymemembed.py ../../example/1BL8.pdb

# Quick subset
python test_pymemembed.py ../../example/1BL8.pdb --quick
```

## File Structure

```
pymemembed/
  __init__.py           # Public API: memembed_align, calculate_energy, _optimize
  __main__.py           # CLI entry point (pymemembed command)
  core.py               # Numba JIT functions (orientate, potentials)
  pdb.py                # PDB I/O and coordinate transforms
  optimizers.py         # GA, grid, direct search algorithms
  memembed_wrapper.py   # Integration with packmol-memgen main.py (run_ga_multi)
  test_pymemembed.py    # Test suite
  README.md             # This file
  DEVELOPMENT.md        # Development history, fixes, performance analysis
```

## See Also

- C++ MEMEMBED source in `../memembed/`

## Citation

Schott-Verdugo & Gohlke (2019). PACKMOL-Memgen: A Simple-To-Use, Generalized
Workflow for Membrane-Protein-Lipid-Bilayer System Building. *J. Chem. Inf.
Model.*, 59(6), 2522-2528.
