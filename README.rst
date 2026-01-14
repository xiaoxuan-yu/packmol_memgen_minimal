#############################################################
                 PACKMOL MEMGEN MINIMAL  2026.1.8
#############################################################
Xiaoxuan Yu
2026-01-08

################
THIS FORK NOTES
################

This repository is a fork of packmol-memgen. The original README content is
preserved below; this section summarizes the changes and current behavior.

- Packaging: this fork is published on PyPI (sdist/wheel). The Python wheel aims to
  avoid system-level dependencies (e.g., no AmberTools requirement). Some features 
  include SIRAH coarse-graining support, minimization, parametrization and protonation
  using reduce are not available in this fork.

- Packmol: packing is performed by the external PACKMOL command-line tool. This
  package does not bundle PACKMOL; it is expected to be available on PATH. For
  convenience, it can be installed via an optional dependency (extra).

- Membrane orientation: membrane alignment/orientation is performed via MemPrO
  (Python + JAX) instead of the upstream memembed/ppm3 workflow. MemPrO is invoked
  via subprocess as an external command-line tool. MemPrO is not bundled into this
  package but can be installed via an optional dependency for convenience.

- Protonation: protonation uses pdb2pqr when needed.

###############################
Martini Auto-Mapping (MemPrO/Insane4MemPrO)
###############################

When ``--martini`` is enabled, the script can auto-build ``--insane_args`` for
MemPrO/Insane4MemPrO with the following behavior:

- Lipids: ``--lipids`` + ``--ratio`` map to Insane ``-l/-u``. ``//`` syntax is
  supported for lower/upper leaflets. Two ``--lipids`` entries are supported and
  map to Insane ``-l/-u`` (inner membrane) and ``-lo/-uo`` (outer membrane).
- Box size: ``--dims`` is converted from Angstrom to nm and passed as ``-x/-y/-z``.
- Curvature: ``--curvature`` or ``--curv_radius`` maps to ``-curv c0,0,dir``
  where ``c0 = abs(curvature)``, ``dir = sign(curvature)``.
- Solvents: ``--solvents`` + ``--solvent_ratio`` map to ``-sol`` entries. The
  only automatic name mapping is ``WAT -> W``, which emits a warning once. All
  other solvent names must match Insane solventParticles (see
  ``packmol_memgen/data/insane_solvents.txt``).
- Salts: ``--salt``, ``--salt_c``, ``--salt_a`` map to ``-posi_c0/1/2`` and
  ``-negi_c0/1/2`` using ion tokens stripped to letters (e.g. ``Na+`` -> ``NA``).
  ``--saltcon`` maps to ``-ion_conc`` with the value replicated for all three
  compartments.

The following options are *not* supported in Martini auto-mapping and will emit
warnings: ``--distxy_fix``, ``--dist``, ``--dist_wat``, ``--xygauss``,
``--apl_offset``, ``--lip_offset``, ``--pbc``, ``--nocounter``,
``--charge_imbalance``, ``--imbalance_ion``. You can still supply custom
``--insane_args`` to override or extend the auto-generated arguments.

############
Installation
############
Minimal installation (this fork without PACKMOL or MemPrO):

``pip install packmol-memgen-minimal``

With optional packmol as a dependency:

``pip install packmol-memgen-minimal[packmol]``

With optional mempro as a dependency:

``pip install packmol-memgen-minimal[mempro]``

With both optional dependencies:

``pip install packmol-memgen-minimal[full]``

You may prefer mordern Python packaging tools such as `uv` to install 

``uv tool install packmol-memgen-minimal``

extras are supported in `uv` as well. Other tools may work similarly.


############
LICENSE NOTE
############

This fork is distributed under GPL-2.0-only, inherited from the upstream project.

MemPrO is an optional external command-line tool distributed under the GNU GPL v3.
This project interacts with MemPrO via subprocess calls; MemPrO is not bundled into
this package and its license applies to MemPrO itself.

The martini force field parameter file `packmol_memgen/data/martini_v3.0.0.itp` is 
sourced from https://github.com/Martini-Force-Field-Initiative/martini-forcefields 
and is distributed under the Apache License 2.0. The full Apache 2.0 license text 
is provided at `packmol_memgen/data/LICENSE.Apache-2.0`.

##########
CREDITS
##########

This project depends on and/or interoperates with the following open-source
software. Please refer to each project for license terms.

Core dependencies (installed via PyPI):

- NumPy (https://github.com/numpy/numpy)
- SciPy (https://github.com/scipy/scipy)
- pandas (https://github.com/pandas-dev/pandas)
- tqdm (https://github.com/tqdm/tqdm)
- matplotlib (https://github.com/matplotlib/matplotlib)
- PDB2PQR (https://github.com/Electrostatics/pdb2pqr)
- docopt (https://github.com/docopt/docopt)

Optional external tools (installed separately; invoked via subprocess):

- PACKMOL (MIT License, https://github.com/m3g/packmol)
- MemPrO (GNU GPL v3, https://github.com/ShufflerBardOnTheEdge/MemPrO)

Additional utilities used by this workflow:

- PDBREMIX (MIT License, https://github.com/boscoh/pdbremix)
- charmmlipid2amber (GPL-2.0, from `AmberTools 2025`, https://ambermd.org/AmberTools.php)
- Martini 3 force field parameters (Apache-2.0, https://github.com/Martini-Force-Field-Initiative/martini-forcefields)


#############################################################
                 PACKMOL MEMGEN  2023.8.8 
#############################################################
Stephan N. Schott V.
2016-11-07



############ 
INSTALLATION
############

The main installation of Packmol Memgen is intended to be installed within AmberTools20. If that is not the case
the following instructions could be follow to make it run as a stand-alone, though it is not officially supported.

Dependencies:
Boost (tested with 1.54 or above)
gfortran & gcc (tested with 4.8 and above)
python 2.7 (or above)

The script should work as is. Includes a setup.py script which should work as usual:

``python setup.py install``

or

``pip install .``

You can specify a custom installation location with --home=<FOLDER> or --prefix=<FOLDER>, respectively.
Testing of the software can be performed in the example folder, by executing "./example.sh".
This folder can be easily cleaned by executing "./example.sh clean"



##############
CHANGELOG 
##############
2025.3.22
 - Added initial functionality to support Packmol's PBC option
2025.3.17
 - Added onlyleapin flag to generate a template leap input without parametrizing.
 - Modified memembed with Claude 3.7 suggestions to stop using boost.
 - Updated packmol to 20.16
2025.2.10
 - Force using propka for titration with_ph in pdb2pqr
2025.1.29
 - Bugfix calculating center of PDB. This could have caused issues in calculating volumes, mainly in solvated systems.
2024.3.27
 - Changed MEMEMBED to push TER to output file
2024.3.18:
 - Separation of function in classes
 - Better management of ppm3 alignment function
2024.2.9:
 - Fixed help typo
 - PACKMOL Update to fix FORCED output
2023.8.8:
 - Added siWT4 COM to avoid overlaps
 - Fixed deprecated np.float calls
 - Added ppm3 support and code
 - Shift water box if --dims is used to add at least 5A water on membrane surface
 - Add packmol finetuning parameters
2023.2.24:
 - Changed versioning to CalVer
 - Updated packmol
 - Added all amber supported ions and OPC3
 - Added option to generate HMR systems
 - Added control for minimization function and pmemd.cuda
 - Added option to use pdb2pqr to protonate, if available
 - Multiple small changes and cleanup
v1.2.3:
 - Added support for Lipid21 sphingolipids
 - Add channel_plug flag and fix 2LPG head atoms in memgen.parm
 - Add function to identify sterol lipid piercing within packed box
 - Limit packing of sterols in the rim under some conditions
v1.2.1:
 - Fixed using %v/v concentrations for solutes.
v1.2.0:
 - Added sterols ergosterol, stigmasterol, sitosterol, campesterol
 - Reformated and cleaned FF parameter files
 - Fixed bogus packing params
 - Added functionality for using alternate solvents
 - User provided memgen.parm or solvent.parm extend and not only replace parameters used by default
 - Added support for coarse graining systems with SIRAH 
 - Fixed bug that caused charmmlipid2amber to fail when packing a single residue molecule
 - Support for Lipid21 / Lipid17 can be selected.
v1.1.0:
 - Added parameters for cardiolipin headgroup
 - New PDBs generated with stereocenter constraints
 - Added --solute_prot_dist to set free diffusion simulations with ligands starting at a distance of a potential target.
 - Fixed rounding error for solute placement in water compartments
 - Added flags --apl_offset --self_assembly 
 - Small code changes
v1.0.8:
 - Added preliminary code for --double_span option for proteins with two transmembrane regions. Can be used with --preoriented given proper MEM atoms
 - Added preliminary code for using different surface geometries
 - Added --tight_box to use dimensions calculated during the packing process in the leap parametrization
v1.0.7:
 - Added APL values derived from Lipid17 DOPC:X 4:1 300K 1 bar. Flag to use old APL values added.
 - Allow to pass custom memgen.parm file with --memgen_parm
v1.0.5:
 - Changed pdb file distro
 - Added experimental pdbs for all possible combinations
 - Added PI & multiple protonation states parameters
 - memgen.parm to all missing lipids
v1.0.2:
 - Fixed the way dims and solute_inmem are handled
 - Included parameters for lysophospholipid heads PE PG PC // BETA! (pdbs and memgen.parm)
 - Fixed and updated internal charmmlipid2amber.csv
 - Added missing MG pdb
v1.0.1:
 - Added flags dims, gaff2 and solute_charge
 - Possible to specify concentrations as percentages. Depends on pdbremix at the moment! 
v1.0.0:
 - Modified POPC pdb to avoid bug that apparently caused it to be stuck in a maxima
 - Fixed bug while trying to find lipid APL with the same headgroup
 - Newest packmol version 18.169
 - Inclusion of boost 1.69 // compatible with gcc 8 compilers  // AmberTools19 includes boost!
 - Mac compatible compilation code 
v0.9.95:
 - Added writeout flag to control intermediate saves. Allows shorter nloops.
 - Adapted code to Leandro's nloop_all. 
 - New packmol version with bug fixes (18.104). Should fix isues with Windows WSL.
 - Python 3 friendly code.
 - Added cubic flag for solvating. reducebuild for protonating HIS.
 - Fixed multi bilayer z-dim size bug.
 - Changed solute_con to accept either number of molecules or concentrations.
v0.9.9:
 - Possibility to build stacked membrane systems by calling --lipids multiple times. PDB files to be embeded have to be included with --pdb now, to make possible to specify multiple inputs.
 - A charge imbalance can be specified between compartments (CompEL!)
 - If desired, a specific per leaflet composition can be specified.
 - The orientation by MEMEMBED is with the N-terminus to the "inside" (or to lower z values). Added flag (--n_ter) to set this per protein in case of multiple bilayers.
v0.9.8:
 - Modified print commands for logging statements.
 - Added optional grid building command (testing!)
 - Added functions that allow minimization and parametrization, given that AMBERHOME is defined
 - Added option to keep ligands after protein alignment
v0.9.7:
 - Check if ions are required before adding the line to packmol (avoids "bug" that adds one molecule even when the number is 0)
v0.9.6:
 - Possible to add solutes in water based on PDB input
 - Modified sigterm command for runs
 - Reduced the verbosity. Can be called with --verbose
v0.9.5:
 - Changed parsing system to argparse
 - Adapted to work with Wooey
 - Implemented charmmlipid2amber as a module // charmm and amber outputs
v0.9.1:
 - Now is possible to create membranes without proteins by setting the PDB as None
 - The xy dimensions can be set with the flag -distxy_fix. This flag is required if no PDB is given
v0.9:
 - Multiple lipids available! (-available_lipids for a list)
 - Detection of neutralization / salt concentration included
v0.7:
 - Volume estimation by building grid based on PDBREMIX
 - Search for AMBERHOME definition, using reduce if available
 - Automated alignment available based on MEMEMBED
 - Tiny code clean-up
v0.3:
 - Distance is measured considering worst case scenario (max x and y considered to be on the diagonal)
 - Added progress bar for all-together packing step
 - Filter for non-defined arguments
