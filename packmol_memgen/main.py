#!/usr/bin/env python

from __future__ import print_function
import os, sys, math, subprocess, random, argparse, shutil, atexit, signal, logging, shlex, glob, importlib.util
from pathlib import Path
import pandas as pd
import tarfile
script_path = os.path.abspath(os.path.dirname(__file__))+os.path.sep
sys.path.append(script_path)
from argparse import RawDescriptionHelpFormatter
from lib.utils import *
from lib.amber import *
import tqdm
from lib.pdbremix.lib.docopt import docopt
from lib.pdbremix import pdbatoms
from lib.pdbremix.rmsd import *
from lib.pdbremix.volume import volume
from lib.charmmlipid2amber.charmmlipid2amber import charmmlipid2amber

#CHECK CHANGELOG IN README.rst

try:
    os.environ['COLUMNS'] = str(shutil.get_terminal_size()[0])
except:
    try:
        from backports.shutil_get_terminal_size import get_terminal_size
        os.environ['COLUMNS'] = str(get_terminal_size()[0])
    except:
        os.environ['COLUMNS'] = "80"

def _prepend_uv_tool_dirs_to_path():
    # Ensure tools installed by `uv tool install` are discoverable via PATH.
    tool_names = ("packmol_memgen_minimal", "packmol-memgen-minimal")
    candidates = []

    uv_tool_bin_dir = os.environ.get("UV_TOOL_BIN_DIR", "").strip()
    if uv_tool_bin_dir:
        candidates.append(Path(uv_tool_bin_dir))

    uv_tool_dir = os.environ.get("UV_TOOL_DIR", "").strip()
    if uv_tool_dir:
        uv_tool_root = Path(uv_tool_dir)
        candidates.append(uv_tool_root / "bin")
        for name in tool_names:
            candidates.append(uv_tool_root / "tools" / name / "bin")

    default_root = Path.home() / ".local" / "share" / "uv"
    candidates.append(default_root / "bin")
    for name in tool_names:
        candidates.append(default_root / "tools" / name / "bin")

    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    for path in candidates:
        if path and path.exists():
            path_str = str(path)
            if path_str not in parts:
                parts.insert(0, path_str)
    os.environ["PATH"] = os.pathsep.join(parts)

_prepend_uv_tool_dirs_to_path()


explanation = """The script creates an input file for PACKMOL for creating a bilayer system with a protein inserted in it. The input pdb file will be protonated and oriented by default using pdb2pqr and MemPrO; the user is encouraged to check the input and output files carefully!  If the protein is preoriented, for example by using the PPM webserver from OPM (http://opm.phar.umich.edu/server.php), be sure to set the corresponding flag (--preoriented).  In some cases the packed system might crash during the first MD step. Changes in the box boundaries or repacking with --random as an argument might help.

 If you use this script, please cite the tools reported at the end of the run:

    - [PACKMOL-Memgen] Schott-Verdugo, S.; Gohlke, H. PACKMOL-Memgen: A Simple-To-Use, Generalized Workflow for Membrane-Protein–Lipid-Bilayer System Building. J. Chem. Inf. Model. 2019, 59 (6), 2522–2528. https://doi.org/10.1021/acs.jcim.9b00269.
    - [PACKMOL] Martínez, L.; Andrade, R.; Birgin, E. G.; Martínez, J. M. PACKMOL: A Package for Building Initial Configurations for Molecular Dynamics Simulations. J. Comput. Chem. 2009, 30 (13), 2157–2164. https://doi.org/10.1002/jcc.21224.
    - [MemPrO] Parrag, M.; Stansfeld, P. J. MemPrO: A Predictive Tool for Membrane Protein Orientation. J. Chem. Theory Comput. 2025. https://doi.org/10.1021/acs.jctc.5c01433.
    - [Martini] Souza, P. C. T.; Alessandri, R.; Barnoud, J.; et al. Martini 3: a general purpose force field for coarse-grained molecular dynamics. Nat Methods 2021, 18, 382–388. https://doi.org/10.1038/s41592-021-01098-3.
    - [PDB2PQR] Dolinsky, T. J.; Czodrowski, P.; Li, H.; Nielsen, J. E.; Jensen, J. H.; Klebe, G.; Baker, N. A. PDB2PQR: Expanding and Upgrading Automated Preparation of Biomolecular Structures for Molecular Simulations. Nucleic Acids Res. 2007, 35 (Web Server issue), W522–W525. https://doi.org/10.1093/nar/gkm276.
    - [PDB2PQR] Dolinsky, T. J.; Nielsen, J. E.; McCammon, J. A.; Baker, N. A. PDB2PQR: An Automated Pipeline for the Setup of Poisson–Boltzmann Electrostatics Calculations. Nucleic Acids Res. 2004, 32 (Web Server issue), W665–W667. https://doi.org/10.1093/nar/gkh381.

"""

explanation = explanation+"-"*int(os.environ['COLUMNS'])

short_help = "-h" in sys.argv

parser = argparse.ArgumentParser(prog="packmol-memgen", description = explanation, add_help=False, formatter_class=RawDescriptionHelpFormatter)
parser.add_argument("-h",     action="help", help="prints this help message and exits" if short_help else "prints a short help message and exits")
parser.add_argument("--help", action="help", help="prints an extended help message and exits" if short_help else "prints this help message and exits")
parser.add_argument("--available_lipids",action="store_true",     help="list of available lipids and corresponding charges")
parser.add_argument("--available_lipids_all",action="store_true", help="list all lipids including experimental. Huge output (>4000 lines)! Think about grepping the lipid you want (packmol-memgen --available_lipids_all | grep YOUR_LIPID_PATTERN)")
parser.add_argument("--available_solvents",action="store_true",     help="list of available solvents and corresponding charges")
parser.add_argument("--available_ions",action="store_true",     help="list of available ions and corresponding charges")
parser.add_argument("-l","--lipids",action="append",metavar="LIP1:LIP2//LIP3",help="Lipid(s) to be used for embeding the protein. It should be a single string separated by ':' . If different composition is used in leaflets, add '//' as a separator.[ex. CHL1:DOPC//DOPE for a lower leaflet with CHL1+DOPC and an upper leaflet with DOPE]. Can be defined multiple times for multi-bilayer systems (stacks 'up' or 'outside')")
parser.add_argument("-r","--ratio",action="append",metavar="R1:R2//R3", help="mixture ratio (set to 1 if only one lipid required). Must be in the same order and syntax as in lipids, and defined once per bilayer [ex. 1:2//1] ")
parser.add_argument("--solvents",type=str,    metavar="SOL1:SOL2",help="Solvent(s) to be used for the packing. As lipids, it should be a single string separated by ':'. Water by default.")
parser.add_argument("--solvent_ratio",type=str,metavar="SR1:SR2", help="mixture ratio (set to 1 if only one solvent required). Must be in the same order and syntax as in solvents")
parser.add_argument("--dist",         type=float, default=15.0,   help=argparse.SUPPRESS if short_help else "specify the minimum distance between the maxmin values for x y and z to the box boundaries. Default = 15 A. Worst case scenario is considered, so real distance could be larger")
parser.add_argument("--dist_wat", type=float, default=17.5,       help=argparse.SUPPRESS if short_help else "specify the width of the water layer over the membrane or protein in the z axis. Default = 17.5")
parser.add_argument("--distxy_fix",   type=float,                 help="specify the membrane patch side length in the x and y axis. Useful when only lipids are packed! By default is calculated flexibly depending on the protein")
parser.add_argument("--watorient", action="store_true",           help=argparse.SUPPRESS if short_help else "use 1.5 radius in packmol for water oxygen, to foster water orientation. Might help for packing particularly big water systems")
parser.add_argument("--noxy_cen",     action="store_false",       help=argparse.SUPPRESS if short_help else "disable default centering in the xy plane that ensures symmetric membrane building. Not recommended!")
parser.add_argument("--channel_plug",  type=float,                help=argparse.SUPPRESS if short_help else "establishes a cylindrical restraint on the lipids using the protein z height and the input value as xy radius. A value of 0 will use half of the protein radius. By default, no restraint is imposed.")
parser.add_argument("--self_assembly",      action="store_true",  help=argparse.SUPPRESS if short_help else "places lipids all over the packed box, and not in a bilayer.")
parser.add_argument("--xygauss",nargs=3,metavar=("C","D","H"),    help=argparse.SUPPRESS if short_help else "set parameters for a curved 2d gaussian in the xy plane. Parameters are uncertainty in x, uncertainty in y and gaussian height. By default, membranes are flat.")
parser.add_argument("--curvature",    type=float, default=None,   help=argparse.SUPPRESS if short_help else "set the curvature of the membrane patch. By default, membranes are flat.")
parser.add_argument("--curv_radius",  type=float, default=None,   help=argparse.SUPPRESS if short_help else "inverse of curvature. Set the curvature as if on a vesicle with the provided radius.")
parser.add_argument("--dims", nargs=3,metavar=("X","Y","Z"),      type=float,default=[0,0,0], help=argparse.SUPPRESS if short_help else "box dimensions vector for the  x y z  axes. Be sure to use dimensions that cover the complete protein to be packed!!")
parser.add_argument("--solvate",      action="store_true",        help=argparse.SUPPRESS if short_help else "solvate the system without adding lipids. Disables the flag --dist_wat, using only --dist to set the box size. Under development!")
parser.add_argument("--pbc",          action="store_true",        help=argparse.SUPPRESS if short_help else "use PBC option in packmol, and adapt constraints accordingly.")
parser.add_argument("--cubic",        action="store_true",        help=argparse.SUPPRESS if short_help else "cube shaped box. Only works with --solvate")
parser.add_argument("--vol",          action="store_true",        help=argparse.SUPPRESS if short_help else "do the lipid number estimation based on the volume occupied by the leaflet instead of APL. This might cause a great overestimation of the number of lipid molecules!")
parser.add_argument("--leaflet",      type=float, default=23.0,   help=argparse.SUPPRESS if short_help else "set desired leaflet width. 23 by default.")
parser.add_argument("--lip_offset",   type=float, default=1.0,    help=argparse.SUPPRESS if short_help else "factor that multiplies the x/y sizes for the lipid membrane segment. Might improve packing and handling by AMBER")
parser.add_argument("--apl_offset",   action="append",            help=argparse.SUPPRESS if short_help else "factor that multiplies the default APL values. Helpful if packing stretched membranes.")
parser.add_argument("--tailplane",    type=float,                 help=argparse.SUPPRESS if short_help else "sets the position BELOW which the CH3 carbon atoms in the tail should be. By default defined in parameter file")
parser.add_argument("--headplane",    type=float,                 help=argparse.SUPPRESS if short_help else "sets the position ABOVE which the PO4 phosphorus and N atoms in the polar head group should be.By default defined in parameter file")
parser.add_argument("--plot",         action="store_true",        help=argparse.SUPPRESS if short_help else "makes a simple plot of loop number vs GENCAN optimization function value, and outputs the values to GENCAN.dat")
parser.add_argument("--traj",         action="store_true",        help=argparse.SUPPRESS if short_help else "saves all intermediate steps into separate pdb files")
parser.add_argument("--notgridvol",   action="store_false",       help=argparse.SUPPRESS if short_help else "skips grid building for volume estimation, and the calculation is done just by estimating density")
parser.add_argument("--keep",         action="store_true",       help=argparse.SUPPRESS if short_help else "skips deleting temporary files")
parser.add_argument("--noprogress",   action="store_true",        help=argparse.SUPPRESS if short_help else "avoids the printing of progress bar with time estimation in the final stage. Recommended if the job is piped into a file")
parser.add_argument("--apl_exp",      action="store_true",        help=argparse.SUPPRESS if short_help else "use experimental APL where available, like AmberTools18 release. Kept for consistency with older versions. By default, terms estimated with Lipid17 are used")
parser.add_argument("--memgen_parm", type=str,                    help=argparse.SUPPRESS if short_help else "load custom memgen.parm file with APL and VOL values. Extends and overwrites default values")
parser.add_argument("--solvent_parm", type=str,                   help=argparse.SUPPRESS if short_help else "load custom solvent.parm file with densities and molecular weights. Extends and overwrites default values")
parser.add_argument("--overwrite",    action="store_true",        help=argparse.SUPPRESS if short_help else "overwrite, even if files are present")
parser.add_argument("--log",type=str,default="packmol-memgen.log",help=argparse.SUPPRESS if short_help else "log file name where detailed information is to be written")
parser.add_argument("-o","--output",type=str,                     help=argparse.SUPPRESS if short_help else "name of the PACKMOL generated PDB file")
parser.add_argument("--charmm",     action="store_true",          help=argparse.SUPPRESS if short_help else "the output will be in CHARMM format instead of AMBER. Works only for small subset of lipids (see --available_lipids)")
parser.add_argument("--translate", nargs=3, type=float, default=[0,0,0], help=argparse.SUPPRESS if short_help else "pass a vector as  x y z  to translate the oriented pdb. Ex. ' 0 0 4 '")
parser.add_argument("--sirah", action="store_true",               help=argparse.SUPPRESS if short_help else "use SIRAH lipids, and corase-grain protein input. Will adapt tolerance accordingly. Only small subset of lipids available!")
parser.add_argument("--verbose",    action="store_true",          help=argparse.SUPPRESS if short_help else "verbose mode")
parser.add_argument("--xponge",     action="store_true",          help=argparse.SUPPRESS if short_help else "postprocess ion names to Xponge-compatible identifiers")

parser.add_argument("--pdb2pqr",      action="store_true",        help=argparse.SUPPRESS if short_help else "uses pdb2pqr to protonate the protein structure")
parser.add_argument("--pdb2pqr_pH",   type=float, default=7.0,    help=argparse.SUPPRESS if short_help else "pH to be used by pdb2pqr to protonate the structure")
parser.add_argument("--notprotonate", action="store_false",       help=argparse.SUPPRESS if short_help else "skips protonation")

inputs = parser.add_argument_group('Inputs')
inputs.add_argument("-p","--pdb",           action="append",       help="PDB or PQR file(s) to embed. If many bilayers, it has to be specified once for each bilayer. 'None' can be specified and a bilayer without protein will be generated [ex. --pdb PDB1.pdb --pdb None --pdb PDB2.pdb (3 bilayers without protein in the middle)]. If no PDB is provided, the bilayer(s) will be membrane only (--distxy_fix has to be defined).")
inputs.add_argument("--solute",        action="append",            help=argparse.SUPPRESS if short_help else "adds pdb as solute into the water. Concentration has to be specified")
inputs.add_argument("--solute_con",    action="append",            help=argparse.SUPPRESS if short_help else "number of molecules/concentration to be used. Concentrations are specified in Molar by adding an 'M' as a suffix (Ex. 0.15M). If not added, a number of molecules is assumed.")
inputs.add_argument("--solute_charge", action="append",            help=argparse.SUPPRESS if short_help else "absolute charge of the included solute (Ex. -2). To be considered in the system neutralization")
inputs.add_argument("--solute_inmem",  action="store_true",        help=argparse.SUPPRESS if short_help else "solute should be added to membrane fraction")
inputs.add_argument("--solute_prot_dist",  type=float,             help=argparse.SUPPRESS if short_help else "establishes a cylindrical restraint using the protein xy radius and z height + the input value. A value of 0 will use the protein radius. By default, no restraint is imposed.")

embedopt = parser.add_argument_group('MemPrO options')
embedopt.add_argument("--preoriented",  action="store_true",          help="use this flag if the protein has been previosuly oriented and you want to avoid running MemPrO (i.e. from OPM)")
embedopt.add_argument("--double_span",  action="store_true",          help=argparse.SUPPRESS) #"orient protein twice, assuming it spans two membrane bilayer")
embedopt.add_argument("--n_ter",        action="append",              help=argparse.SUPPRESS if short_help else "'in' or 'out'. By default proteins are oriented with the n_ter oriented 'in' (or 'down'). relevant for multi layer system. If defined for one protein, it has to be defined for all of them, following previous order")
embedopt.add_argument("--keepligs",     action="store_true",          help=argparse.SUPPRESS if short_help else "MemPrO ignores HETATM records; use with care if you rely on ligands")
embedopt.add_argument("--mempro",type=str,                            help=argparse.SUPPRESS if short_help else "Path to MemPrO executable or MemPrO_Script.py")
embedopt.add_argument("--mempro_grid",type=int,default=36,            help=argparse.SUPPRESS if short_help else "MemPrO grid size (-ng)")
embedopt.add_argument("--mempro_iters",type=int,default=150,          help=argparse.SUPPRESS if short_help else "MemPrO minimization iterations (-ni)")
embedopt.add_argument("--mempro_rank",type=str,default="auto",choices=["auto","h","p"],help=argparse.SUPPRESS if short_help else "MemPrO rank mode (-rank)")
embedopt.add_argument("--mempro_args",type=str,                       help=argparse.SUPPRESS if short_help else "Extra arguments passed to MemPrO")
embedopt.add_argument("--mempro_curvature", action="store_true",      help=argparse.SUPPRESS if short_help else "use MemPrO curvature (-c) and set --curvature from Global curvature in info_rank_1.txt")
embedopt.add_argument("--no-keep-mempro", action="store_false", dest="keep_mempro", help=argparse.SUPPRESS if short_help else "remove MemPrO outputs and working folder during cleanup")

packmolopt = parser.add_argument_group('PACKMOL options')
packmolopt.add_argument("--nloop",       type=int,default=20,         help=argparse.SUPPRESS if short_help else "number of nloops for GENCAN routine in PACKMOL. PACKMOL MEMGEN uses 20 by default; you might consider increasing the number to improve packing. Increasing the number of components requires more GENCAN loops.")
packmolopt.add_argument("--nloop_all",   type=int,default=100,        help=argparse.SUPPRESS if short_help else "number of nloops for all-together packing. PACKMOL MEMGEN uses 100 by default.")
packmolopt.add_argument("--tolerance",   type=float,default=2.0,      help=argparse.SUPPRESS if short_help else "tolerance for detecting clashes between molecules in PACKMOL (defined as radius1+radius2). PACKMOL uses 2.0 by default.")
packmolopt.add_argument("--prot_rad",   type=float,default=1.5,       help=argparse.SUPPRESS if short_help else "radius considered for protein atoms to establish the tolerance for detecting clashes. PACKMOL MEMGEN uses 1.5 by default.")
packmolopt.add_argument("--writeout",                                 help=argparse.SUPPRESS if short_help else "frequency for writing intermediate results. PACKMOL uses 10 by default.")
packmolopt.add_argument("--notrun",       action="store_true",        help=argparse.SUPPRESS if short_help else "will not run PACKMOL, even if it's available")
packmolopt.add_argument("--random",       action="store_true",        help=argparse.SUPPRESS if short_help else "turns PACKMOL random seed generator on. If a previous packing failed in the minimization problem, repacking with this feature on might solve the problem.")
packmolopt.add_argument("--packall",  action="store_true",            help=argparse.SUPPRESS if short_help else "skips initial individual packing steps")
packmolopt.add_argument("--short_penalty",  action="store_true",      help=argparse.SUPPRESS if short_help else "add a short range penalty for heavily overlapping atoms with default PACKMOL options")
packmolopt.add_argument("--movebadrandom", action="store_true",       help=argparse.SUPPRESS if short_help else "randomizes positions of badly placed molecules in initial guess")
packmolopt.add_argument("--maxit",       type=int,default=20,         help=argparse.SUPPRESS if short_help else "number of GENCAN iterations per loop. 20 by default.")
packmolopt.add_argument("--movefrac",    type=float,default=0.05,     help=argparse.SUPPRESS if short_help else "fraction of molecules to be moved. 0.05 by default.")
packmolopt.add_argument("--packlog",type=str,default="packmol",       help=argparse.SUPPRESS if short_help else "prefix for generated PACKMOL input and log files")
packmolopt.add_argument("--packmol",type=str,                         help=argparse.SUPPRESS)
packmolopt.add_argument("--hexadecimal_indices", action="store_true", default=True, help=argparse.SUPPRESS if short_help else "use PACKMOL hexadecimal_indices output; will be converted to hybrid-36 in final PDB")
packmolopt.add_argument("--no-hexadecimal-indices", action="store_false", dest="hexadecimal_indices", help=argparse.SUPPRESS if short_help else "disable PACKMOL hexadecimal_indices output")

saltopt = parser.add_argument_group('Salts and charges')
saltopt.add_argument("--salt",        action="store_true",         help=argparse.SUPPRESS if short_help else "adds salt at a concentration of 0.15M by default. Salt is always added considering estimated charges for the system.")
saltopt.add_argument("--salt_c",default="K+",                      help=argparse.SUPPRESS if short_help else "cation to add. (K+ by default)")
saltopt.add_argument("--salt_a",default="Cl-",                     help=argparse.SUPPRESS if short_help else "anion to add. (Cl- by default)")
saltopt.add_argument("--saltcon", type=float, default=0.15,        help=argparse.SUPPRESS if short_help else "modifies the default concentration for KCl. [M]")
saltopt.add_argument("--salt_override",action="store_true",        help=argparse.SUPPRESS if short_help else "if the concentration of salt specified is less than the required to neutralize, will try to continue omitting the warning")
saltopt.add_argument("--nocounter",action="store_true",            help=argparse.SUPPRESS if short_help else "no counterions are added. System charge is handled by downstream tools")
saltopt.add_argument("--charge_pdb_delta", action="append",        help=argparse.SUPPRESS if short_help else "add a given formal charge value per pdb. Might be useful to compensate for charges of non-standard residues not accounted by the script. If many pdbs, it has to be specified once for each pdb.")

compel = parser.add_argument_group('Computational electophysiology')
compel.add_argument("--double",       action="store_true",        help=argparse.SUPPRESS if short_help else "asumes a stacked double bilayer system for CompEL. The composition in --lipids will be used for both bilayers flipping the leaflets")
compel.add_argument("--charge_imbalance", type=int, default=0,    help=argparse.SUPPRESS if short_help else "sets a charge imbalance between compartments (in electron charge units). A positive imbalance implies an increase (decrease) in cations (anions) in the central compartment.")
compel.add_argument("--imbalance_ion", type=str, default="cat", choices=["cat","an"], help=argparse.SUPPRESS if short_help else "sets if cations or anions are used to imbalance the system charges. ('cat' by default)")

logger = logging.getLogger("pmmg_log")

class PACKMOLMemgen(object):
    """
    Class that manages PACKMOL-Memgen arguments and initialization
    """
    def __init__(self, args=None):
        logger = logging.getLogger("pmmg_log")
        if args==None:
            args = parser.parse_args()

        #Get args in self 
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        self._used_tools = {"packmol-memgen"}

    def prepare(self):
 
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            print("--verbose for details of the packing process", file=sys.stderr)
        logger.debug("Execution line:     "+" ".join(sys.argv))
    
    
        #Path to local Packmol installation (http://www.ime.unicamp.br/~martinez/packmol/home.shtml)
        rep = "data"
        lib = "lib"
        self.run = not self.notrun
        if self.apl_exp:
            apl = "APL_EXP"
        else:
            apl = "APL_FF"
        packmol_inc = os.path.join(script_path, lib, "packmol", "packmol" + exe_suffix)
        self.packmol_cmd = None
        packmol_path = None
        if self.packmol is None:
            packmol_env = shutil.which("packmol" + exe_suffix) or shutil.which("packmol")
    
            if os.path.exists(packmol_inc):
                packmol_path = packmol_inc
            elif packmol_env:
                packmol_path = packmol_env
            else:
                packmol_path = None
        else:
            if self.packmol:
                packmol_path = self.packmol
    
        if packmol_path:
            if os.path.exists(packmol_path):
                self.packmol_cmd = [packmol_path]
            else:
                self.packmol_cmd = shlex.split(packmol_path)
        else:
            self.packmol_cmd = None
    
        if not self.packmol_cmd:
            miss = "Packmol not found. Execution will only create the packmol input script."
            logger.info("\n"+len(miss)*"#"+"\n"+miss+"\n"+len(miss)*"#"+"\n")
            self.run = False
    
    
        ############## PARSE PARAMS FROM REP #########
    
        pdbtar = tarfile.open(os.path.join(script_path, rep,"pdbs.tar.gz"),"r:gz")
        parms = open(os.path.join(script_path, rep, "memgen.parm"), "r")
        parmlines = parms.readlines()
        parms.close()
        if self.memgen_parm is not None:
            logger.debug("Loading custom memgen.parm:     "+self.memgen_parm)
            logger.debug("Parameters will be extended or overwritten over default values")
            parms = open(self.memgen_parm, "r")
            parmlines += parms.readlines()
            parms.close()
        self.parameters = {}
        ext_par = False
        for line in parmlines:
            if line[0] == "#":
                if line.startswith("#EXTENDED LIPIDS"):
                    ext_par = True
                continue
            else:
                values = line.split()
                self.parameters[values[0]]  = {}
                self.parameters[values[0]].update({
                                         "p_atm":values[1],
                                         "t_atm":values[2],
                                         "APL_FF":values[3],
                                         "APL_EXP":values[4],
                                         "V":values[5],
                                         "charge":values[6],
                                         "h_bound":values[7],
                                         "t_bound":values[8],
                                         "C":values[9],
                                         "VH":values[10],
                                         "charmm":values[11],
                                         "name":values[12],
                                         "ext":ext_par
                                         })
    
        if self.available_lipids or self.available_lipids_all:
            if self.memgen_parm is not None:
                logger.debug("List will include parameters in the parsed parameter file!: "+self.memgen_parm)
            if self.available_lipids:
                pformat = "{:<9}{:>6}{:>70}"
            else:
                pformat = "{:<9}{:>6}{:>130}"
            table_header= (pformat+"{:>15}").format("Lipid","Charge","Full-name","Comment")
            print("\n"+table_header, file=sys.stderr)
            print("-"*len(table_header), file=sys.stderr)
            for key in sorted(self.parameters):
                if not self.available_lipids_all and self.parameters[key]["ext"]:
                    continue
                elif is_number(key[0]) or key[-2:]=="CL" or "PI" in key or self.parameters[key]["ext"]:
                    print((pformat+" ***Parameters from PACKMOL-Memgen Lipid_ext").format(key,self.parameters[key]["charge"],self.parameters[key]["name"]))
                elif key[-2:] == "SM":
                    print((pformat+" ***Only available with Lipid21").format(key,self.parameters[key]["charge"],self.parameters[key]["name"]))
                elif key.startswith("si"):
                    print((pformat+" ***Only available with SIRAH").format(key,self.parameters[key]["charge"],self.parameters[key]["name"]))
                else:
                    print(pformat.format(key,self.parameters[key]["charge"],self.parameters[key]["name"]))
            exit()
    
    
        solvent_parms = open(os.path.join(script_path, rep, "solvent.parm"), "r")
        sparmlines = solvent_parms.readlines()
        solvent_parms.close()
        if self.solvent_parm is not None:
            logger.debug("Loading custom solvent.parm:     "+self.solvent_parm)
            solvent_parms = open(self.solvent_parm, "r")
            sparmlines += solvent_parms.readlines()
            solvent_parms.close()
        self.sparameters = {}
        for line in sparmlines:
            if line[0] == "#":
                continue
            values = line.split()
            self.sparameters[values[0]]  = {}
            self.sparameters[values[0]].update({
                                     "density":values[1],
                                     "MW":values[2],
                                     "charge":values[3],
                                     "name":values[4],
                                     "source":values[5].replace("_"," ") if len(values) == 6 else "-"
                                     })
    
        if self.available_solvents:
            pformat = "{:<9}{:>15}{:>15}{:>15}{:>40}{:>80}"
            table_header= pformat.format("Solvent","Density [g/cm3]","MW [g/mol]","Charge","Full-name","Comments")
            print("\n"+table_header, file=sys.stderr)
            print("-"*len(table_header), file=sys.stderr)
            for key in sorted(self.sparameters):
                print(pformat.format(key,self.sparameters[key]["density"],self.sparameters[key]["MW"],self.sparameters[key]["charge"],self.sparameters[key]["name"],self.sparameters[key]["source"]))
            exit()
    
        if self.available_ions:
            pformat = "{:<9}{:>15}"
            table_header= pformat.format("Ion","Charge")
            print("\n"+table_header, file=sys.stderr)
            print("-"*len(table_header), file=sys.stderr)
            for key in sorted(amber_ion_dict):
                print(pformat.format(key,amber_ion_dict[key][1]))
            exit()
    
        ############ ARGPARSE ARGUMENTS ###################
    
        if self.curv_radius is not None:
            self.curvature = 1/self.curv_radius
    
        if self.curvature is not None:
            sphere_radius = 1/self.curvature
            if self.vol:
                logger.critical("CRITICAL:\n  --vol and curvature are not compatible!")
                exit()
            if self.solvate:
                logger.critical("CRITICAL:\n  --solvate and curvature are not compatible!")
                exit()
    
        if not self.solvate and self.cubic:
            logger.warning("WARNING:--cubic only available with --solvate. Turned off!")
            self.cubic = False
    
        if self.solvate:
            self.preoriented = True
            z_cen = True
            pdb_prefix = "solvated"
            self.dist_wat = self.dist
        else:
            pdb_prefix = "bilayer"
            z_cen = False
    
        leaflet_z = self.leaflet # leaflet thickness
        lip_offset = self.lip_offset
        bound_tail = self.tailplane # plane that defines boundary to last carbon in aliphatic chain
        bound_head = self.headplane # plane that defines boundary to polar head
        lipids = self.lipids # lipids to be used
    
        self.onlymembrane = False
        if self.pdb is None or self.pdb.count("None") == len(self.pdb):
            self.onlymembrane = True
        else:
            for pdb in self.pdb:
                if not (pdb.endswith(".pdb") or pdb.endswith(".pqr")) and pdb != "None":
                    logger.critical("CRITICAL:\n  The input file can only be in PDB or PQR formats.")
                    exit()
            base_dir = os.path.dirname(script_path.rstrip(os.path.sep))
            def resolve_input_path(path, label):
                if path == "None":
                    return path
                if os.path.exists(path):
                    return os.path.abspath(path)
                if not os.path.isabs(path):
                    candidate = os.path.abspath(os.path.join(base_dir, path))
                    if os.path.exists(candidate):
                        logger.debug("Resolved %s path %s -> %s", label, path, candidate)
                        return candidate
                logger.critical("CRITICAL:\n  %s file not found: %s (cwd=%s)", label, path, os.getcwd())
                exit()
            self.pdb = [resolve_input_path(pdb, "PDB") for pdb in self.pdb]
        if self.double_span and self.pdb is None:
            logger.critical("CRITICAL:\n  --double_span requires a PDB file as input")
            exit()
        elif self.double_span and len(self.pdb) == 1:
            self.pdb.append("None")
    
        self.outfile = self.output
        if self.outfile is not None:
            if not self.outfile.endswith(".pdb"):
                self.outfile = self.outfile+".pdb"
        if not self.output and self.pdb is None:
            self.outfile = pdb_prefix+"_only.pdb"
        elif not self.output:
            self.outfile = pdb_prefix+"_"+"".join([os.path.basename(pdb)[:-4] for pdb in self.pdb])+".pdb"
    
        if lipids is not None:
            if any([self.parameters[l]["ext"] for lipid in lipids for leaf in lipid.split("//") for l in leaf.split(":")]):
                logger.info("The Lipid Force Field extension lipid_ext is required for this system.")
        saltcon = self.saltcon # salt concentration in M
        if not self.salt:
            if "--saltcon" in sys.argv and (self.distxy_fix is not None or self.pdb is not None):
                logger.error("ERROR:\n    You specified a salt concentration, but not the salt flag. Only neutralizing ions will be added")
            saltcon = 0
            if self.charge_imbalance != 0:
                logger.error("ERROR:\n    You specified a charge imbalance, but not the salt flag. No charge imbalance will be applied")
        override_salt = self.salt_override
        #Check for cation
        if self.sirah:
            self.ion_dict = {"K+":("siKW",1,"KW"),"Na+":("siNaW",1,"NaW"),"Ca2+":("siCaX",2,"CaX"),"Cl-":("siClW",-1,"ClW")}
        else:
            self.ion_dict = amber_ion_dict 
        if self.salt_c not in self.ion_dict:
            logger.error("ERROR:\n    The specified cation option is no available at the moment")
            exit()
        else:
            if self.ion_dict[self.salt_c][1] < 0:
                logger.error("ERROR:\n    The specified cation is really an anion")
                exit()
            cation = self.ion_dict[self.salt_c][0]
        if self.salt_a not in self.ion_dict:
            logger.error("ERROR:\n    The specified cation option is no available at the moment")
            exit()
        else:
            if self.ion_dict[self.salt_a][1] > 0:
                logger.error("ERROR:\n    The specified anion is really a cation")
                exit()
            anion = self.ion_dict[self.salt_a][0] #Maybe more alternatives at some point? ILs?
        distance = self.dist # distance from the protein to the box boundaries to XY
        distance_wat = self.dist_wat # minimum distance from the surface of the membrane to the box boundary on Z
    
        if (self.distxy_fix is not None and self.dims != [0,0,0]):
            logger.error("ERROR:\n    --distxy_fix and --dims should not be used together! Use -h for help.")
            exit()
    
        asym = False
        if self.distxy_fix is not None and self.dims == [0,0,0]:
            self.dims = [self.distxy_fix,self.distxy_fix,0]
        elif self.dims == [0,0,0]:
            self.dims = None
        else:
            asym = True
    
        if (self.distxy_fix is None and self.dims is None) and self.onlymembrane:
            logger.error("ERROR:\n    No PDB file given or fixed XY dimensions specified. Check --distxy_fix and --dims for help (-h/--help).")
            exit()
        if self.writeout is None:
            if int(self.nloop) < 10 or int(self.nloop_all) < 10:
                logger.error("ERROR:\n    nloop and nloop_all have to be bigger than the writeout frequency (every 10 loops by default). You can modify this with --writeout")
                exit()
        else:
            if int(self.nloop) < int(self.writeout) or int(self.nloop_all) < int(self.writeout):
                logger.error("ERROR:\n    nloop and nloop_all have to be bigger than the writeout frequency! Modify the used values.")
                exit()
        protonate = self.notprotonate
        grid_calc = self.notgridvol
        self.delete = not self.keep
    
        # JSwails suggestion//Check if in a tty. Turn off progress bar if that's the case.
        if not os.isatty(sys.stdin.fileno()):
            self.noprogress = True
    
        # Make a list with created files for later deletion
        self.created        = []
        self.created_notrun = []
        self.created_mempro = []

        ###############################################
        ###############################################
        ###############################################
    
        ############ SEARCH LIBS, PDB, MEMPRO ###################
    
        if protonate and not self.pdb2pqr and pdb2pqr:
            self.pdb2pqr = True
            logger.debug("Defaulting to pdb2pqr for protonation.")
        if protonate and not self.pdb2pqr:
            logger.error("ERROR:\n    Protonation requested, but pdb2pqr is not available.")
            exit()

        if not os.path.exists(os.path.join(script_path, rep)):
            logger.critical("CRITICAL:\n  The data folder for using the script is missing. Check the path for the script!")
            exit()
    
        if self.pdb is not None:
            if self.pdb[0] == "None" and not self.onlymembrane:
                logger.error("ERROR:\n    Please specify first the protein PDB file!")
                exit()
            for pdb in self.pdb:
                if not os.path.exists(pdb) and pdb != "None":
                    logger.error("ERROR:\n    Either the options were wrongly used or the file "+pdb+" doesn't exist!")
                    exit()
    
        mempro_cmd = self.mempro
        if mempro_cmd is None:
            mempro_cmd = (
                shutil.which("mempro")
                or shutil.which("MemPrO")
                or shutil.which("MemPro")
            )
        if mempro_cmd:
            self.mempro = mempro_cmd
        else:
            self.mempro = ""
    
        if not self.mempro:
            miss = (
                "MemPrO not found. Protein orientation will not be available unless --preoriented is set.\n"
                "Install it with: pip install packmol-memgen-minimal[mempro]"
            )
            logger.warning("\n"+len(miss)*"#"+"\n"+miss+"\n"+len(miss)*"#"+"\n")
            self.mempro = False
            
        # logger.info(self.mempro if self.mempro else "MemPrO not used")
    
        if not os.path.exists(os.path.join(script_path, lib, "pdbremix")):
            logger.warning("WARNING:PDBREMIX lib not available. Volume estimation will be done based on estimated density")
        else:
            grid_avail = True
    
        if self.solvents is None:
            self.solvents = "WAT"
        if len(self.solvents.split(":")) == 1:
            self.solvent_ratio = "1"
        elif len(self.solvents.split(":")) > 1 and self.solvent_ratio is None:
            logger.error("ERROR:\n    If using solvent mixtures, a solvent ratio has to be specified")
            exit()
        if len(self.solvents.split(":")) != len(self.solvent_ratio.split(":")):
            logger.error("ERROR:\n    Amount of solvent types and solvent ratios doesn't fit! Check your input")
            exit()
    
        if self.sirah:
            logger.info("Using parameters to pack a SIRAH system")
            self.solvents = "siWT4"
            self.solvent_ratio = "1"
            if "--prot_rad" not in sys.argv:
                self.prot_rad = 2
            if "--tolerance" not in sys.argv:
                self.tolerance = 3
    
        for solvent in self.solvents.split(":"):
            if solvent not in self.sparameters:
                logger.error("ERROR:\n    Selected solvent %s parameters not available. Check --available_solvents" % (solvent))
                exit()
    
        solvent_ratios = [float(ratio) for ratio in self.solvent_ratio.split(":")]
        solvent_density = sum([float(self.sparameters[solvent]["density"])*solvent_ratios[i] for i, solvent in enumerate(self.solvents.split(":"))])/sum(solvent_ratios)
        solvent_con     = sum([(solvent_ratios[i]*float(self.sparameters[solvent]["density"])*avogadro)/(float(self.sparameters[solvent]["MW"])*10**24) for i, solvent in enumerate(self.solvents.split(":"))])/sum(solvent_ratios)
    
        if lipids is None:
            lipids = ["DOPC"]
        if self.double and len(lipids) == 1:
            lipids = lipids+["//".join(reversed(lipids[0].split("//")))]
        elif self.double and len(lipids) != 1:
            logger.error("ERROR:\n    --double requires a single definition of lipid composition")
            exit()
        self.lipids = lipids # Keep the class list up to date
        if self.double_span and len(lipids) != 2:
            logger.error("ERROR:\n    --double_span requires a definition of lipid composition per bilayer")
            exit()
        if self.ratio is None:
            logger.info("No ratio provided (--ratio).")
            self.ratio  = ["1"]*len(lipids)
        if self.double and len(lipids) == 2 and len(self.ratio) == 1:
            self.ratio = self.ratio+["//".join(reversed(self.ratio[0].split("//")))]
    
        if len(lipids) != len(self.ratio):
            logger.error("ERROR:\n    Number of defined bilayer lipids and ratios doesn't fit! Check your input")
            exit()
    
        if self.n_ter is None and self.double:
            self.n_ter = ["in","out"]
        elif self.n_ter is None:
            self.n_ter = ["in"]*len(lipids)
    
        elif len(self.n_ter) != len(self.pdb):
            logger.error("ERROR:\n    Number of specified orientations and bilayers doesn't fit! Check your input")
            exit()
        else:
            for n, pdb in enumerate(self.pdb):
                if pdb != "None" and not (self.n_ter[n] == "in" or self.n_ter[n] == "out"):
                    logger.error("ERROR:\n    The orientation has to be 'in' or 'out' unless pdb is 'None'")
                    exit()
    
        if self.pdb is not None:
            if self.charge_pdb_delta is None:
                self.charge_pdb_delta = [0]*len(self.pdb)
            elif len(self.charge_pdb_delta) != len(self.pdb):
                logger.error("ERROR:\n    Number of specified charge deltas and pdbs doesn't fit! Check your input")
                exit()
            else:
                self.charge_pdb_delta = [int(i) for i in self.charge_pdb_delta]
    
        if self.solute_prot_dist is not None and self.onlymembrane:
            logger.error("ERROR:\n    A solute distance can only be specified with respect to an included PDB!")
            exit()
    
        if self.channel_plug is not None and self.onlymembrane:
            logger.error("ERROR:\n    A channel plug can only be specified with respect to an included PDB!")
            exit()
    
        if self.solute_charge is None and self.solute is not None:
            self.solute_charge = [0]*len(self.solute)
    
        elif self.solute_charge is not None and self.solute is not None:
            if len(self.solute_charge) != len(self.solute):
                logger.error("ERROR:\n    Number of specified solute charges and solutes doesn't fit! Check your input")
                exit()
            else:
                for n, solute in enumerate(self.solute):
                    try:
                        self.solute_charge[n] = int(self.solute_charge[n])
                    except:
                        logger.error("ERROR:\n    Solute charges have to be integers")
                        exit()
        apl_offset   = {}
        composition  = {}
        self.sterols_PI_used = False
        for bilayer in range(len(lipids)):
            apl_offset[bilayer]  = {}
            composition[bilayer] = {}
            if "//" in lipids[bilayer]:
                if len(lipids[bilayer].split("//")) != 2 or len(self.ratio[bilayer].split("//")) != 2:
                    logger.error("ERROR:\n    If different leaflet compositions used, ratios have to be specified explicitly, and both must be separated by one '//' only!")
                    exit()
                for leaflet in range(2):
                    if len(lipids[bilayer].split("//")[leaflet].split(":")) != len(self.ratio[bilayer].split("//")[leaflet].split(":")):
                        logger.error("ERROR:\n    Amount of lipid types and ratios doesn't fit! Check your input")
                        exit()
                    apl_offset[bilayer][leaflet] = {}
                    if self.apl_offset is None:
                        apl_offset[bilayer][leaflet] = 1.0
                    elif len(self.apl_offset) != len(lipids):
                        logger.error("ERROR:\n    If apl_offset is specified, it has to be set once per bilayer")
                    elif "//" in self.apl_offset[bilayer]:
                        apl_offset[bilayer][leaflet] = float(self.apl_offset[bilayer].split("//")[leaflet])
                    else:
                        apl_offset[bilayer][leaflet] = float(self.apl_offset[bilayer])
                    composition[bilayer][leaflet]={}
                    ratio_total = sum([float(x) for x in self.ratio[bilayer].split("//")[leaflet].split(":")])
                    for n, lipid in enumerate(lipids[bilayer].split("//")[leaflet].split(":")):
                        if self.sirah:
                            composition[bilayer][leaflet]["si"+lipid] = float(self.ratio[bilayer].split("//")[leaflet].split(":")[n])/ratio_total
                        else:
                            composition[bilayer][leaflet][lipid] = float(self.ratio[bilayer].split("//")[leaflet].split(":")[n])/ratio_total
            else:
                if len(lipids[bilayer].split(":")) != len(self.ratio[bilayer].split(":")):
                    logger.error("ERROR:\n    Amount of lipid types and ratios doesn't fit! Check your input")
                    exit()
                if self.apl_offset is None:
                    apl_offset[bilayer][0] = apl_offset[bilayer][1] = 1.0
                elif len(self.apl_offset) != len(lipids):
                    logger.error("ERROR:\n    If apl_offset is specified, it has to be set once per bilayer")
                elif "//" in self.apl_offset[bilayer]:
                    apl_offset[bilayer][0], apl_offset[bilayer][1] = map(float,self.apl_offset[bilayer].split("//"))
                else:
                    apl_offset[bilayer][0] = apl_offset[bilayer][1] = float(self.apl_offset[bilayer])
                composition[bilayer][0]={} ; composition[bilayer][1]={}
                ratio_total = sum([float(x) for x in self.ratio[bilayer].split(":")])
                for n, lipid in enumerate(lipids[bilayer].split(":")):
                    if self.sirah:
                        composition[bilayer][0]["si"+lipid] = composition[bilayer][1]["si"+lipid] = float(self.ratio[bilayer].split(":")[n])/ratio_total
                    else:
                        composition[bilayer][0][lipid] = composition[bilayer][1][lipid] = float(self.ratio[bilayer].split(":")[n])/ratio_total
    
        for key in [x for leaflet in composition[bilayer] for bilayer in composition for x in composition[bilayer][leaflet]]:
            if key in sterols_PI or "PI" in key:
                self.sterols_PI_used = True
            if key not in self.parameters:
                logger.critical("CRITICAL:\n  Parameters missing for "+key+". Please check the file memgen.parm")
                exit()
            if self.charmm and self.parameters[key]["charmm"] == "N":
                logger.critical("CRITICAL:\n  Lipid "+key+" not available in CHARMM format.")
                exit()
            if not os.path.exists(os.path.join(script_path, rep, "pdbs", key +".pdb")) and key+".pdb" not in pdbtar.getnames():
                logger.warning("WARNING:\n  PDB file for "+key+" not found in repo!")
                if not os.path.exists(key+".pdb"):
                    logger.critical("CRITICAL:\n  PDB file for "+key+" not found in local folder!")
                    exit()
    
        if not self.onlymembrane:
            if len(composition) != len(self.pdb):
                if self.double and len(composition) == 2 and len(self.pdb) == 1:
                    logger.debug("Using the pdb file for both bilayers")
                    self.pdb = self.pdb*2
                else:
                    logger.critical("CRITICAL:\n  The number of provided PDB files doesnt fit with the number of bilayers. Pass --pdb \"None\" if you want an empty bilayer.")
                    exit()
    
        if self.solute is not None:
            if self.solute_con is not None:
                if len(self.solute) != len(self.solute_con):
                    logger.critical("CRITICAL:\n  Number of solutes and concentrations/number of molecules is not the same! Please provide a concentration for each solute in the respective order.")
                    exit()
            else:
                logger.error("ERROR:\n    Concentrations/number of molecules have to be provided.")
                exit()
    
    
        if self.solute is not None:
            for i, sol in enumerate(self.solute):
                if not os.path.exists(self.solute[i]):
                    logger.error(self.solute[i]+" not found!")
                    exit()
                logger.info("Extra solute PDB         = %-9s" % (self.solute[i]))
                logger.info("Solute to be added       = %-9s" % (self.solute_con[i]))
    
    
        ############################### SCRIPT HEADER ################################
    
        content_header = content_prot = content_lipid = content_solvent = content_ion = content_solute = ""
        content_header += "tolerance "+str(self.tolerance)+"\n"
        content_header += "filetype pdb\n"
        content_header += "output "+self.outfile+"\n\n"
        if self.writeout is not None:
            content_header += "writeout "+self.writeout+"\n"
        if self.traj:
            if self.writeout is None:
                content_header += "writeout 1\n"
            content_header += "writebad\n\n"
    
        if self.random:
            content_header += "seed -1\n"
        if self.packall:
            content_header += "packall\n"
        if self.maxit != 20:
            content_header += "maxit "+str(self.maxit)+"\n"
        if self.movefrac != 0.05:
            content_header += "movefrac "+str(self.movefrac)+"\n"
        if self.short_penalty:
            content_header += "use_short_tol\n"
            content_header += "short_tol_dist "+str(self.tolerance/2)+"\n"
            content_header += "short_tol_scale 5\n"
        if not self.charmm:
             content_header += "add_amber_ter\n"
        if self.hexadecimal_indices:
            content_header += "hexadecimal_indices\n"
        content_header += "amber_ter_preserve\n"
        if self.movebadrandom:
            content_header += "movebadrandom\n"

    
        content_header += "nloop "+str(self.nloop_all)+"\n\n"
    
        pond_lip_vol_dict      = {}
        pond_lip_apl_dict      = {}
        lipnum_dict            = {}
        lipnum_area_dict       = {}
    
        X_min   = X_max = X_len = Y_min = Y_max = Y_len = 0
        Z_dim   = []
        memvol  = []
        solvol  = []
        charges = []
        if not self.onlymembrane:
            chain_nr = 0
            chain_index = list(string.ascii_uppercase)+list(string.ascii_lowercase)
            for n,pdb in enumerate(self.pdb):
                if pdb != "None":
                    if not self.verbose:
                        logger.info("Preprocessing "+pdb+". This might take a minute.")
                    if not self.preoriented and self.mempro:
                        if self.double_span:
                            logger.debug("Attempting to orient double span protein using MemPrO...")
                            self._used_tools.add("mempro")
                            self._used_tools.add("martini")
                            pdb, z_offset_ds = self.mempro_align(pdb,keepligs=self.keepligs,verbose=self.verbose,overwrite=self.overwrite,n_ter=self.n_ter[n], double_span=True)
                            z_offset_ds = np.abs(z_offset_ds)
                            pdb_ds = pdb
                        else:
                            logger.debug("Orienting the protein using MemPrO...")
                            self._used_tools.add("mempro")
                            self._used_tools.add("martini")
                            pdb = self.mempro_align(pdb,keepligs=self.keepligs,verbose=self.verbose,overwrite=self.overwrite,n_ter=self.n_ter[n])
                        if self.keep_mempro:
                            self.created_mempro.append(pdb)
                        else:
                            self.created.append(pdb)
                    if protonate:
                        logger.debug("Adding protons using pdb2pqr at pH "+str(self.pdb2pqr_pH)+"...")
                        self._used_tools.add("pdb2pqr")
                        pdb =  pdb2pqr_protonate(pdb,overwrite=self.overwrite,pH=self.pdb2pqr_pH)
                        self.created.append(pdb)                 
                    elif self.preoriented and self.double_span:
                        ds_oriented = pdb_parse(pdb, onlybb=False)
                        try:
                            z_dist = ds_oriented[('MEM', 1, 'X')][('MEM', 1)][2]+ds_oriented[('MEM', 2, 'X')][('MEM', 2)][2]
                        except:
                            logger.critical("CRITICAL:\n  If using a preoriented double spanning PDB, it has to include MEM atoms!")
                            exit()
                        if z_dist < 0:
                            ds_oriented = translate_pdb(ds_oriented,vec=[0,0,z_dist])
                            z_dist *= -1
                        del ds_oriented[('MEM', 1, 'X')]
                        del ds_oriented[('MEM', 2, 'X')]
    
                        pdb_write(ds_oriented, outfile="ds_temp.pdb")
                        pdb = pdb_ds = "ds_temp.pdb"
    
                        z_offset_ds = z_dist
    
                    if grid_calc:
                        logger.debug("Estimating the volume by building a grid (PDBREMIX)...")
                        grid = self.pdbvol(pdb)
                        logger.debug("PDBREMIX grid done: grid_file=%s vol=%s", grid[0], grid[1])
                        self.created.append(grid[0])
                    else:
                        grid = (None,None)
    
                    ############## FAST VALUE ESTIMATION FROM PDB #####################
                    mem_params = MembraneParams(pdb,leaflet_z,grid[0],move=True, move_vec=self.translate, xy_cen=self.noxy_cen, z_cen=z_cen, outpdb="PROT"+str(n)+".pdb",chain=chain_index[chain_nr],renumber=True)
                    minmax, max_rad, charge_prot, vol, memvol_up, memvol_down, solvol_up, solvol_down, density, mass, chains = mem_params.measure()
                    chain_nr += chains
                    charge_prot += self.charge_pdb_delta[n]
                    charges.append(charge_prot)
                    if self.double:
                        mem_params = MembraneParams(pdb,leaflet_z,grid[0],move=True, move_vec=self.translate, xy_cen=self.noxy_cen, z_cen=z_cen, outpdb="PROT"+str(n+1)+".pdb",chain=chain_index[chain_nr],renumber=True)
                        minmax, max_rad, charge_prot, vol, memvol_up, memvol_down, solvol_up, solvol_down, density, mass, chains = mem_params.measure()
                        chain_nr += chains
                        charge_prot += self.charge_pdb_delta[n]
                        charges.append(charge_prot)
    
    
                elif pdb == "None" and self.double_span:
                    logger.debug("Estimating for second mem!")
                    mem_params = MembraneParams(pdb_ds,leaflet_z,grid[0],move=True, move_vec=[0,0,-z_offset_ds], xy_cen=self.noxy_cen, z_cen=z_cen, outpdb="PROT"+str(n)+".pdb",chain=chain_index[chain_nr],renumber=True)
                    minmax, max_rad, charge_prot, vol, memvol_up, memvol_down, solvol_up, solvol_down, density, mass, chains = mem_params.measure()
                else:
                    if not self.verbose:
                        logger.debug("Building membrane without protein!")
                    if self.dims is not None:
                        minmax      = [-self.dims[0]/2,-self.dims[1]/2,-leaflet_z,self.dims[0]/2,self.dims[1]/2,leaflet_z]
                    else:
                        minmax      = minmax[0:2]+[-leaflet_z]+minmax[3:5]+[leaflet_z]
                    charge_prot      = 0
                    charges.append(charge_prot)
                    vol              = 0
                    memvol_up        = 0
                    memvol_down      = 0
                    solvol_up        = 0
                    solvol_down      = 0
                    density          = 0
                    mass             = 0
    
                solvol.append((solvol_down,solvol_up))
                memvol.append((memvol_down,memvol_up))
                if self.double:
                    memvol.append((memvol_down,memvol_up))
                #    solvol.append((solvol_down,solvol_up))
    
                if self.double_span and pdb == "None":
                    charges.append(0)
                    Z_dim.append((-z_offset_ds/2,z_max-z_offset_ds,z_max-z_offset_ds/2))
                    continue
    
                com = [(minmax[0]+minmax[3])/2,(minmax[1]+minmax[4])/2,(minmax[2]+minmax[5])/2]
    
                if self.dims is not None:
                    pdbx_min     = com[0]-self.dims[0]/2
                    pdbx_max     = com[0]+self.dims[0]/2
                    pdby_min     = com[1]-self.dims[1]/2
                    pdby_max     = com[1]+self.dims[1]/2
                    pdbz_min     = com[2]-self.dims[2]/2
                    pdbz_max     = com[2]+self.dims[2]/2
    
                else:
                    pdbx_min     = minmax[0]
                    pdbx_max     = minmax[3]
                    pdby_min     = minmax[1]
                    pdby_max     = minmax[4]
                    pdbz_min     = minmax[2]
                    pdbz_max     = minmax[5]
    
                pdbx_len     = pdbx_max-pdbx_min
                pdby_len     = pdby_max-pdby_min
                pdbz_len     = pdbz_max-pdbz_min
    
                if not asym:
                    if self.dims is None:
        #                max_side_len = math.sqrt(pdbx_len**2+pdby_len**2)+2*distance
                        max_side_len = 2*max_rad+2*distance
                    else:
                        max_side_len = max(self.dims[:2])
                    if self.cubic:
                        max_side_len = max(max_side_len,pdbz_len+2*distance)
                    diff_x       = max_side_len - pdbx_len
                    diff_y       = max_side_len - pdby_len
                    diff_z       = max_side_len - pdbz_len
                else:
                    #if setting box dimensions by hand, check that box boundaries leave some space to add water
                    if pdbz_min > -(leaflet_z+5):
                        correct_z = pdbz_min+(leaflet_z+5)
                        pdbz_min = -(leaflet_z+5)
                        pdbz_max = pdbz_max-correct_z
                    if pdbz_max < (leaflet_z+5):
                        correct_z = pdbz_max-(leaflet_z+5)
                        pdbz_max = (leaflet_z+5)
                        pdbz_min = pdbz_min-correct_z
                    if pdbz_max < minmax[5]+(self.dist_wat-5) or pdbz_min > minmax[2]-(self.dist_wat-5):
                        logger.critical("CRITICAL: Set dimensions and protein location don't allow to have a proper water surface on the membrane. Increase the the used dimensions.")
                        exit()
                    diff_x = diff_y = diff_z = 0
                x_min        = pdbx_min-diff_x/2
                x_max        = pdbx_max+diff_x/2
                y_min        = pdby_min-diff_y/2
                y_max        = pdby_max+diff_y/2
                z_min        = pdbz_min-diff_z/2
                z_max        = pdbz_max+diff_z/2
                x_len        = x_max-x_min
                y_len        = y_max-y_min
                z_len        = z_max-z_min
    
                if not self.cubic and not asym:
                    z_min = pdbz_min-distance_wat
                    if z_min > -(leaflet_z+distance_wat):
                        z_min = -(leaflet_z+distance_wat)
                    z_max = pdbz_max+distance_wat
                    if z_max < (leaflet_z+distance_wat):
                        z_max =  (leaflet_z+distance_wat)
                    if self.xygauss:
                        z_max += float(self.xygauss[2])
                    z_len = z_max-z_min
    
                prot_data2= """
     Estimated values for input protein:
    
     Input PDB                = %-9s
     Charge                   = %-9s
     Mass                     = %-9s    Da
     Density                  = %-9s    Da/A^3
     Estimated volume         = %-9s    A^3
         in upper leaflet     = %-9s    A^3
         in lower leaflet     = %-9s    A^3
         in upper water box   = %-9s    A^3
         in lower water box   = %-9s    A^3
                """
                logger.debug(prot_data2 % ( pdb, charge_prot, mass, round(density,2), round(vol,2), round(memvol_up,2), round(memvol_down,2), round(solvol_up), round(solvol_down,2)))
    
    
                if x_len > X_len:
                    X_min = x_min; X_max = x_max; X_len = x_len
                if y_len > Y_len:
                    Y_min = y_min; Y_max = y_max; Y_len = y_len
                if self.double_span and pdb != "None":
                    Z_dim.append((z_min,z_offset_ds/2,z_offset_ds/2-z_min))
                else:
                    Z_dim.append((z_min,z_max,z_len))
                if self.double:
                    Z_dim.append((z_min,z_max,z_len))
                    break

        else:
            if self.dims is not None:
                minmax      = [-self.dims[0]/2,-self.dims[1]/2,-self.dims[2]/2,self.dims[0]/2,self.dims[1]/2,self.dims[2]/2]
                z_min = minmax[2]
                z_max = minmax[5]
            else:
                z_min = minmax[2]-distance_wat
                z_max = minmax[5]+distance_wat
    
            charge_prot      = 0
            charges          = [charge_prot]*len(composition)
            vol              = 0
            memvol_up        = 0
            memvol_down      = 0
            solvol_up        = 0
            solvol_down      = 0
            density          = 0
            mass             = 0
    
            memvol = [(memvol_down,memvol_up)]*len(composition)
            solvol = [(solvol_down,solvol_up)]*len(composition)
    
            pdbx_len     = minmax[3]-minmax[0]
            pdby_len     = minmax[4]-minmax[1]
    
            if not asym:
                max_side_len = max(pdbx_len,pdby_len)
                diff_x       = max_side_len - pdbx_len
                diff_y       = max_side_len - pdby_len
            else:
                diff_x = diff_y = 0
            x_min        = minmax[0]-diff_x/2
            x_max        = minmax[3]+diff_x/2
            y_min        = minmax[1]-diff_y/2
            y_max        = minmax[4]+diff_y/2
            x_len        = x_max-x_min
            y_len        = y_max-y_min
    
    
            if not self.solvate:
                if z_min > -(leaflet_z+distance_wat):
                    z_min = -(leaflet_z+distance_wat)
                if z_max < (leaflet_z+distance_wat):
                    z_max =  (leaflet_z+distance_wat)
                if self.xygauss:
                    z_max += float(self.xygauss[2])
            z_len = z_max-z_min
    
            if x_len > X_len:
                X_min = x_min; X_max = x_max; X_len = x_len
            if y_len > Y_len:
                Y_min = y_min; Y_max = y_max; Y_len = y_len
            Z_dim = [(z_min,z_max,z_len)]*len(composition)

        if X_min < -1000 or X_max > 1000 or Y_min < -1000 or Y_max > 1000 or Z_dim[0][0] < -1000 or Z_dim[0][0]+sum([zdim[2] for zdim in Z_dim]) > 1000:
            logger.warning("WARNING:The size of the system is bigger than the default accepted values for PACKMOL. The flag sidemax will be added.")
            content_header += "sidemax "+str(max(abs(max(X_max,Y_max,Z_dim[0][0]+sum([zdim[2] for zdim in Z_dim]))),abs(min(X_min,Y_min,Z_dim[0][0]))))+"\n"

        if self.pbc:
            content_header += f"pbc {X_min:.2f} {Y_min:.2f} {Z_dim[0][0]:.2f} {X_max:.2f} {Y_max:.2f} {Z_dim[0][0]+sum([zdim[2] for zdim in Z_dim]):.2f} \n\n"
    
        prot_data1 = """
     Information for packing:
    
     Input PDB(s)                     = %-9s
     Output PDB                       = %-9s
     Packmol output and log prefix    = %-9s
     Lipids                           = %-9s
     Lipid ratio                      = %-9s
     Solvents                         = %-9s
     Solvent ratio                    = %-9s
     Salt concentration            (M)= %-9s
     Distance to boundaries        (A)= %-9s
     Minimum water distance        (A)= %-9s
     Packmol loops                    = %-9s
     Packmol loops for All-together   = %-9s"""
    
        if not self.solvate:
            logger.info(prot_data1 % ( self.pdb, self.outfile, self.packlog, lipids, self.ratio, self.solvents, self.solvent_ratio, saltcon, distance, distance_wat, self.nloop, self.nloop_all))
        else:
            logger.info(prot_data1 % ( self.pdb, self.outfile, self.packlog, "-", "-", self.solvents, self.solvent_ratio, saltcon, distance, distance_wat, self.nloop, self.nloop_all))
        if self.curvature is not None:
            logger.info(" Membrane curvature          (1/A)= %-9s " % (self.curvature))
        elif self.xygauss:
            logger.info(" XY Gauss curvature (s_x,s_y,h)   = %s %s %s " % tuple(self.xygauss))
    
        box_info = """
     Box information:
     x_min                    = %-9s
     x_max                    = %-9s
     x_len                    = %-9s
    
     y_min                    = %-9s
     y_max                    = %-9s
     y_len                    = %-9s
    
     z_min                    = %-9s
     z_max                    = %-9s
     z_len                    = %-9s
     """
        logger.debug(box_info % ( X_min, X_max, X_len, Y_min, Y_max, Y_len, Z_dim[0][0], Z_dim[0][0]+sum([zdim[2] for zdim in Z_dim]), sum([zdim[2] for zdim in Z_dim]) ))
        self.X_len = X_len
        self.Y_len = Y_len
        self.Z_dim = Z_dim
    
        if os.path.exists(self.outfile) and not self.overwrite:
            logger.info("Packed PDB "+self.outfile+" found. Skipping PACKMOL")
            return False
        else:
            z_offset = 0
    
            for n, bilayer in enumerate(composition):
                if bilayer > 0:
                    if self.double_span:
                        z_offset = z_offset_ds
                    else:
                        z_offset = z_offset+Z_dim[bilayer-1][1]-Z_dim[bilayer][0]
    
                ################################## PROTEIN ###################################
    
                if not self.onlymembrane:
                    if self.pdb[bilayer] != "None":
                        self.created_notrun.append("PROT"+str(bilayer)+".pdb")
                        if self.sirah:
                            self.created_notrun.append("PROT"+str(bilayer)+"_cg.pdb")
                            content_prot += "structure PROT"+str(bilayer)+"_cg.pdb\n"
                        else:
                            content_prot += "structure PROT"+str(bilayer)+".pdb\n"
                        content_prot += "  number 1\n"
                        content_prot += "  fixed 0. 0. "+str(z_offset)+" 0. 0. 0.\n"
                        content_prot += "  radius "+str(self.prot_rad)+"\n"
                        content_prot += "end structure\n\n"
    
                ################################ LIPIDS ######################################
    
                if self.curvature is not None:
                    lipid_vol_up           = sphere_integral_square(X_min,X_max,Y_min,Y_max,r1=sphere_radius+z_offset,r2=sphere_radius+z_offset+leaflet_z,c=-sphere_radius)-memvol[bilayer][1]
                    lipid_vol_down         = sphere_integral_square(X_min,X_max,Y_min,Y_max,r1=sphere_radius+z_offset-leaflet_z,r2=sphere_radius+z_offset,c=-sphere_radius)-memvol[bilayer][0]
                elif self.xygauss:
                    lipid_vol_up           = gauss_integral_square(X_min,X_max,Y_min,Y_max,*map(float,self.xygauss),g1=z_offset,g2=z_offset+leaflet_z)-memvol[bilayer][1]
                    lipid_vol_down         = gauss_integral_square(X_min,X_max,Y_min,Y_max,*map(float,self.xygauss),g1=z_offset-leaflet_z,g2=z_offset)-memvol[bilayer][0]
                else:
                    lipid_vol_up           = ((X_len+2*lip_offset)*(Y_len+2*lip_offset)*leaflet_z)-memvol[bilayer][1]
                    lipid_vol_down         = ((X_len+2*lip_offset)*(Y_len+2*lip_offset)*leaflet_z)-memvol[bilayer][0]
                lipid_vol              = (lipid_vol_down,lipid_vol_up)
                lipid_area             = X_len*Y_len # if curvature, will be defined in loop
                if self.xygauss:
                   lipid_area          = gauss_rectangle_area(X_min,X_max,Y_min,Y_max, *map(float,self.xygauss))
                pond_lip_vol_dict[bilayer]      = {}
                pond_lip_apl_dict[bilayer]      = {}
                lipnum_dict[bilayer]            = {}
                lipnum_area_dict[bilayer]       = {}
                for leaflet in composition[bilayer]:
                    if self.curvature is not None:
                        if leaflet < 1:
                            #lipid_area    = sphere_rectangle_area(sphere_radius+z_offset-leaflet_z,sphere_dist(sphere_radius,X_len),sphere_dist(sphere_radius,Y_len))
                            lipid_area    = sphere_rectangle_area(sphere_radius+z_offset-leaflet_z,X_len,Y_len)
                            logger.debug("Curvature lower leaflet area:"+str(lipid_area))
                        else:
                            #lipid_area    = sphere_rectangle_area(sphere_radius+z_offset+leaflet_z,sphere_dist(sphere_radius,X_len),sphere_dist(sphere_radius,Y_len))
                            lipid_area    = sphere_rectangle_area(sphere_radius+z_offset+leaflet_z,X_len,Y_len)
                            logger.debug("Curvature upper leaflet area:"+str(lipid_area))
                    pond_lip_vol = 0
                    pond_lip_apl = 0
                    for lipid in composition[bilayer][leaflet]:
                        if self.parameters[lipid]["V"] == "XXXX":
                            if self.parameters[lipid]["C"] == "X":
                                logger.critical("CRITICAL:\n  Volume not specified and can not be estimated! Check the parameter file")
                                exit()
                            nCH3,nCH2,nCH = list(map(int,self.parameters[lipid]["C"].split(":")))
                            self.parameters[lipid]["V"] = str(int(round(float(self.parameters[lipid]["VH"])+nCH3*2*VCH2+nCH2*VCH2+nCH*VCH)))         #doi:10.1016/j.bbamem.2005.07.006
                            logger.debug("Experimental value for "+lipid+" volume not available in parm file. Using estimated "+self.parameters[lipid]["V"]+" A^3 instead...")
                        if self.parameters[lipid][apl] == "XX":
                            try:
                                self.parameters[lipid][apl] = max([float(self.parameters[lip][apl]) for lip in self.parameters if (is_number(self.parameters[lip][apl]) and lip.endswith(lipid[-2:]))])
                                logger.debug("Taking maximal APL of lipids with headgroup "+lipid[-2:])
                            except:
                                self.parameters[lipid][apl] = "75"
                                logger.debug("No other lipid with same headgroup has APL. Setting APL of 75 to "+lipid[-2:])
                            logger.debug("Value for "+lipid+" area per lipid not available in parm file. Using "+str(self.parameters[lipid][apl])+" A^2 instead...")
                        pond_lip_vol += composition[bilayer][leaflet][lipid]*int(self.parameters[lipid]["V"])
                        pond_lip_apl += composition[bilayer][leaflet][lipid]*int(self.parameters[lipid][apl])
                    pond_lip_apl = pond_lip_apl*apl_offset[bilayer][leaflet]
                    lipnum = lipid_vol[leaflet]/pond_lip_vol
                    lipnum_area = lipid_area/pond_lip_apl-(memvol[bilayer][leaflet]/pond_lip_vol)
                    pond_lip_vol_dict[bilayer][leaflet]      = pond_lip_vol
                    pond_lip_apl_dict[bilayer][leaflet]      = pond_lip_apl
                    lipnum_dict[bilayer][leaflet]            = lipnum
                    lipnum_area_dict[bilayer][leaflet]       = lipnum_area
    
                charge_lip = charge_solute = 0
                if not self.vol:
                    lipnum_dict = lipnum_area_dict
    
                ################################ SOLUTE ######################################
    
                if not self.solvate:
                    if self.solute is not None and self.solute_inmem:
                        for i,sol in enumerate(self.solute):
                                logger.info("Adding "+self.solute_con[i]+" "+self.solute[i]+" to the lipid volume")
                                grid_file, solute_vol      = self.pdbvol(self.solute[i])
                                self.created.append(grid_file)
                                if self.solute_con[i].endswith("M") and is_number(self.solute_con[i][:-1]):
                                    solute_num      = int(float(self.solute_con[i][:-1])*((sum(lipid_vol)*avogadro/(1*10**27))))
                                elif self.solute_con[i].endswith("%") and is_number(self.solute_con[i][:-1]):
                                    solute_num      = int(float(self.solute_con[i][:-1])/100*((sum(lipid_vol)/solute_vol)))
                                    solute_vol_tot  = int(float(self.solute_con[i][:-1])/100*(sum(lipid_vol)))
                                elif is_number(self.solute_con[i]):
                                    try:
                                        int(self.solute_con[i])
                                    except:
                                        logger.error("ERROR:\n    A number less than 1 is specified. If a concentration was intended, add M/% as a suffix!")
                                        exit()
                                    solute_num      = int(self.solute_con[i])
                                else:
                                    logger.error("ERROR:\n    The format used to specify the number of molecules is not correct! It has to be a number or a number with an M/% suffix.")
                                    exit()
                                if solute_num < 1:
                                    logger.error("ERROR:\n    The solute concentration is too low for the calculated box size. Try increasing the concentration.")
                                    exit()
                                if not self.solute_con[i].endswith("%"):
                                    solute_vol_tot  = solute_vol*solute_num
                                charge_solute += self.solute_charge[i]*solute_num
                                for leaflet in lipnum_dict[bilayer]:
                                    lipnum_dict[bilayer][leaflet] -= int((solute_vol_tot)/(2*pond_lip_vol_dict[bilayer][leaflet]))
    
                                content_solute += "structure "+self.solute[i]+"\n"
                                content_solute += "  number "+str(int(solute_num))+"\n"
                                if self.solute_prot_dist is not None and self.pdb[bilayer] != "None":
                                    content_solute += "  outside cylinder 0. 0. "+str(minmax[2]-self.solute_prot_dist)+" 0. 0. 1. "+str(max_rad+self.solute_prot_dist)+" "+str(minmax[5]-minmax[2]+self.solute_prot_dist)+" \n"
                                if self.curvature is not None:
                                    if not self.pbc:
                                        content_solute += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                                    content_solute += "  inside sphere 0. 0. "+str(-sphere_radius)+" "+str(z_offset+sphere_radius+leaflet_z)+"\n"
                                    content_solute += "  outside sphere 0. 0. "+str(-sphere_radius)+" "+str(z_offset+sphere_radius-leaflet_z)+"\n"
                                elif self.xygauss:
                                    if not self.pbc:
                                        content_solute += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                                    content_solute += "  below xygauss 0. 0. "+self.xygauss[0]+" "+self.xygauss[1]+" "+str(z_offset+leaflet_z)+" "+self.xygauss[2]+"\n"
                                    content_solute += "  over xygauss 0. 0. "+self.xygauss[0]+" "+self.xygauss[1]+" "+str(z_offset-leaflet_z)+" "+self.xygauss[2]+"\n"
                                elif not self.pbc:
                                    content_solute += "  inside box "+str(round(X_min*lip_offset,2))+" "+str(round(Y_min*lip_offset,2))+" "+str(z_offset-leaflet_z)+" "+str(round(X_max*lip_offset,2))+" "+str(round(Y_max*lip_offset,2))+" "+str(z_offset+leaflet_z)+"\n"
                                content_solute += "end structure\n\n"
                    elif (self.solute is not None or self.solute_con is not None) and self.solute_inmem:
                        logger.error("ERROR:\n    The solute parameters are incomplete. Please include both a solute pdb and a concentration.")
                        exit()
    
    
                if not self.solvate:
                    for leaflet in composition[bilayer]:
                        for lipid in sorted(composition[bilayer][leaflet], reverse=True, key=lambda x: composition[bilayer][leaflet][x]):
                            if self.headplane is None:
                                bound_head=float(self.parameters[lipid]["h_bound"])
                                if bound_head >= leaflet_z:
                                    logger.error("ERROR:\n    The boundary for "+lipid+" head group is out of the space delimited for the membrane by the membrane width "+leaflet_z+". Please consider increasing the value!")
                                    exit()
                            if self.tailplane is None:
                                bound_tail=float(self.parameters[lipid]["t_bound"])
                                if bound_tail <=0:
                                    logger.error("ERROR:\n    The boundary for "+lipid+" tail is out of the space delimited for the membrane by the membrane center at the z axis origin (it must be a positive value greater than 0). Please consider increasing the value!")
                                    exit()
                            if not os.path.isfile("./"+lipid+".pdb"):
                                try:
                                    shutil.copy(script_path+rep+"/pdbs/"+lipid+".pdb", "./")
                                except:
                                    pdbtar.extract(lipid+".pdb")
                                self.created_notrun.append(lipid+".pdb")
    
                            content_lipid += "structure "+lipid+".pdb\n"
                            sp = 0 
                            if lipid in sterols_PI or "PI" in lipid: ## Avoid ring piercing by lipid tails, adding extra radius. Later remove those piercing despite this (find_piercing_lipids). Add inositol rings as well!
                                if "PI" in lipid:
                                    _pdb = pdb_parse_TER(lipid+".pdb", onlybb=False)
                                    _atom_string = " ".join([ str(_a[3]) for _a in _pdb[list(_pdb.keys())[0]].keys() if _a[0] == "PI" and _a[2] in PI_ring_probe])
                                    content_lipid += f"  atoms {_atom_string}\n"
                                    content_lipid +=  "    radius 1.5\n"
                                    content_lipid +=  "  end atoms\n"
                                else:
                                    content_lipid += "  radius 1.5\n"
                                if sum([composition[bilayer][leaflet][l] for l in composition[bilayer][leaflet] if l in sterols_PI or "PI" in l]) < 0.5 and (X_len > 50 and Y_len > 50): # only enable if less then half of the composition of the leaflet are sterols
                                    logger.info("Adding 5A padding to the sterol or PI %s components to avoid piercing in the periodic boundary" % (lipid))
                                    sp = 5 #Safety padding to avoid having sterols in the system rim and piercing from periodic image. Taking about "a lipid" width
                                else:
                                    logger.warning("Sterols and/or PI are used, and the concentration is too high or the system too small to add proper side padding. Check that your system doesn't have pierced sterols or PI in the periodic boundary once the packing is finished")
                            content_lipid += "  nloop "+str(self.nloop)+"\n"
                            if int(round(composition[bilayer][leaflet][lipid]*lipnum_dict[bilayer][leaflet])) < 1:
                                logger.error("ERROR:\n    The ratio for lipid "+lipid+" is too small for the given/estimated system size. Either increase the ratio or make the system bigger!")
                                exit()
                            content_lipid += "  number "+str(int(round(composition[bilayer][leaflet][lipid]*lipnum_dict[bilayer][leaflet])))+"\n"
    
                            if not self.pbc:
                                if self.curvature is not None or self.xygauss or self.self_assembly:
                                    content_lipid += "  inside box "+str(round(X_min*lip_offset,2)+sp)+" "+str(round(Y_min*lip_offset,2)+sp)+" "+str(Z_dim[bilayer][0]+z_offset)+" "+str(round(X_max*lip_offset,2)-sp)+" "+str(round(Y_max*lip_offset,2)-sp)+" "+str(Z_dim[bilayer][1]+z_offset)+"\n"
                                else:
                                    content_lipid += "  inside box "+str(round(X_min*lip_offset,2)+sp)+" "+str(round(Y_min*lip_offset,2)+sp)+" "+str(z_offset-leaflet_z+leaflet_z*leaflet)+" "+str(round(X_max*lip_offset,2)-sp)+" "+str(round(Y_max*lip_offset,2)-sp)+" "+str(z_offset+leaflet_z*leaflet)+"\n"
    
                            if self.channel_plug is not None and self.pdb[bilayer] != "None":
                                if self.channel_plug == 0:
                                    self.channel_plug = max_rad/2
                                content_lipid += "  outside cylinder 0. 0. "+str(minmax[2])+" 0. 0. 1. "+str(self.channel_plug)+" "+str(minmax[5]-minmax[2])+" \n"
    
                            if not self.self_assembly:
                                content_lipid += "  atoms "+" ".join([p_atm for p_atm in self.parameters[lipid]["p_atm"].split(",")])+"\n"
                                if leaflet == 0:
                                    if self.curvature is not None:
                                        content_lipid += "    inside sphere 0. 0. "+str(-sphere_radius)+" "+str(sphere_radius-bound_head+z_offset)+"\n"
                                    elif self.xygauss:
                                        content_lipid += "    below xygauss 0. 0. "+self.xygauss[0]+" "+self.xygauss[1]+" "+str(-bound_head+z_offset)+" "+self.xygauss[2]+"\n"
                                    else:
                                        content_lipid += "    below plane 0. 0. 1. "+str(-bound_head+z_offset)+"\n"
                                else:
                                    if self.curvature is not None:
                                        content_lipid += "    outside sphere 0. 0. "+str(-sphere_radius)+" "+str(sphere_radius+bound_head+z_offset)+"\n"
                                    elif self.xygauss:
                                        content_lipid += "    over xygauss 0. 0. "+self.xygauss[0]+" "+self.xygauss[1]+" "+str(bound_head+z_offset)+" "+self.xygauss[2]+"\n"
                                    else:
                                        content_lipid += "    over plane 0. 0. 1. "+str(bound_head+z_offset)+"\n"
                                content_lipid += "  end atoms\n"
                                content_lipid += "  atoms "+" ".join([t_atm for t_atm in self.parameters[lipid]["t_atm"].split(",")])+"\n"
                                if leaflet == 0:
                                    if self.curvature is not None:
                                        content_lipid += "    outside sphere 0. 0. "+str(-sphere_radius)+" "+str(sphere_radius-bound_tail+z_offset)+"\n"
                                    elif self.xygauss:
                                        content_lipid += "    over xygauss 0. 0. "+self.xygauss[0]+" "+self.xygauss[1]+" "+str(-bound_tail+z_offset)+" "+self.xygauss[2]+"\n"
                                    else:
                                        content_lipid += "    over plane 0. 0. 1. "+str(-bound_tail+z_offset)+"\n"
                                else:
                                    if self.curvature is not None:
                                        content_lipid += "    inside sphere 0. 0. "+str(-sphere_radius)+" "+str(sphere_radius+bound_tail+z_offset)+"\n"
                                    elif self.xygauss:
                                        content_lipid += "    below xygauss 0. 0. "+self.xygauss[0]+" "+self.xygauss[1]+" "+str(bound_tail+z_offset)+" "+self.xygauss[2]+"\n"
                                    else:
                                        content_lipid += "    below plane 0. 0. 1. "+str(bound_tail+z_offset)+"\n"
                                content_lipid += "  end atoms\n"
                            content_lipid += "end structure\n\n"
    
                ######################## WATER && SOLUTE #####################################
    
                for solvent in self.solvents.split(":"):
                    solvent_pdb = solvent+".pdb"
                    if os.path.isfile(solvent_pdb):
                        logger.info("Using "+solvent_pdb+" in the folder")
                    else:
                        try:
                            shutil.copy(os.path.join(script_path, rep, "pdbs", solvent_pdb), "./")
                        except:
                            pdbtar.extract(solvent_pdb)
                        self.created_notrun.append(solvent_pdb)
                    if not os.path.isfile(solvent_pdb):
                        logger.critical("CRITICAL:"+solvent_pdb+" is not to found in the folder!")
                        exit()
    
                if self.curvature is not None:
                    solvent_vol_up           = sphere_integral_square(X_min,X_max,Y_min,Y_max,r1=sphere_radius+z_offset+leaflet_z,z_max=Z_dim[bilayer][1]+z_offset,c=-sphere_radius)-solvol[bilayer][1]
                    solvent_vol_down         = sphere_integral_square(X_min,X_max,Y_min,Y_max,z_min=Z_dim[bilayer][0]+z_offset,r2=sphere_radius+z_offset-leaflet_z,c=-sphere_radius)-solvol[bilayer][0]
                elif self.xygauss:
                    solvent_vol_up           = gauss_integral_square(X_min,X_max,Y_min,Y_max,*map(float,self.xygauss),g1=z_offset+leaflet_z,z_max=Z_dim[bilayer][1]+z_offset)-solvol[bilayer][1]
                    solvent_vol_down         = gauss_integral_square(X_min,X_max,Y_min,Y_max,*map(float,self.xygauss),z_min=Z_dim[bilayer][0]+z_offset,g2=z_offset-leaflet_z)-solvol[bilayer][0]
                else:
                    solvent_vol_up   = (X_len*Y_len*(abs(Z_dim[bilayer][1])-leaflet_z))-solvol[bilayer][1]
                    solvent_vol_down = (X_len*Y_len*(abs(Z_dim[bilayer][0])-leaflet_z))-solvol[bilayer][0]
                if self.solvate:
                    solvent_vol_up   = solvent_vol_up+lipid_vol_up
                    solvent_vol_down = solvent_vol_down+lipid_vol_down
                solvent_vol_tot = solvent_vol_up+solvent_vol_down
                watnum_up = int(solvent_vol_up*solvent_con)
                watnum_down = int(solvent_vol_down*solvent_con)
    
                if self.solute is not None and not self.solute_inmem:
                    for i,sol in enumerate(self.solute):
                        logger.info("Adding "+self.solute_con[i]+" "+self.solute[i]+" to the water volume")
                        grid_file, solute_vol      = self.pdbvol(self.solute[i])
                        self.created.append(grid_file)
                        if self.solute_con[i].endswith("M") and is_number(self.solute_con[i][:-1]):
                            solute_up       = int(float(self.solute_con[i][:-1])*((solvent_vol_up*avogadro/(1*10**27))))
                            solute_down     = int(float(self.solute_con[i][:-1])*((solvent_vol_down*avogadro/(1*10**27))))
                        elif self.solute_con[i].endswith("%") and is_number(self.solute_con[i][:-1]):
                            solute_up       = int(float(self.solute_con[i][:-1])/100*((solvent_vol_up/solute_vol)))
                            solute_down     = int(float(self.solute_con[i][:-1])/100*((solvent_vol_down/solute_vol)))
                            solute_vol_up   = int(float(self.solute_con[i][:-1])/100*(solvent_vol_up))
                            solute_vol_down = int(float(self.solute_con[i][:-1])/100*(solvent_vol_down))
                        elif is_number(self.solute_con[i]):
                            try:
                                int(self.solute_con[i])
                            except:
                                logger.error("ERROR:\n    A number less than 1 is specified. If a concentration was intended, add M as a suffix!")
                                exit()
                            solute_up, solute_down = distribute_integer(int(self.solute_con[i]),[solvent_vol_up,solvent_vol_down])
                        else:
                            logger.error("ERROR:\n    The format used to specify the number of molecules is not correct! It has to be a number or a number with an M/% suffix.")
                            exit()
                        if not self.solute_con[i].endswith("%"):
                            solute_vol_up   = solute_vol*solute_up
                            solute_vol_down = solute_vol*solute_down
                        watnum_up       = watnum_up-int((solute_vol_up)*solvent_con)
                        watnum_down     = watnum_down-int((solute_vol_down)*solvent_con)
                        if solute_up+solute_down < 1:
                            logger.error("ERROR:\n    The solute concentration is too low for the calculated box size. Try increasing the concentration.")
                            exit()
                        charge_solute += self.solute_charge[i]*(solute_down+solute_up)
                        if not self.solvate and not self.self_assembly:
                            if solute_down > 0:
                                content_solute += "structure "+self.solute[i]+"\n"
                                content_solute += "  nloop "+str(self.nloop)+"\n"
                                content_solute += "  number "+str(solute_down)+"\n"
                                if self.solute_prot_dist is not None and self.pdb[bilayer] != "None":
                                    content_solute += "  outside cylinder 0. 0. "+str(minmax[2]-self.solute_prot_dist)+" 0. 0. 1. "+str(max_rad+self.solute_prot_dist)+" "+str(minmax[5]-minmax[2]+self.solute_prot_dist)+" \n"
                                if self.curvature is not None:
                                    if not self.pbc:
                                        content_solute += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                                    content_solute += "  inside sphere 0. 0. "+str(-sphere_radius)+" "+str(sphere_radius-leaflet_z+z_offset)+"\n"
                                elif self.xygauss:
                                    if not self.pbc:                                    
                                        content_solute += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                                    content_solute += "  below xygauss 0. 0. "+self.xygauss[0]+" "+self.xygauss[1]+" "+str(-leaflet_z+z_offset)+" "+self.xygauss[2]+"\n"
                                else:
                                    if self.pbc:
                                        content_solute += "  below plane 0. 0. 1. "+str(-leaflet_z+z_offset)+"\n" 
                                    else:
                                        content_solute += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(-leaflet_z+z_offset)+"\n"
                                content_solute += "end structure\n\n"
                            if solute_up > 0:
                                content_solute += "structure "+self.solute[i]+"\n"
                                content_solute += "  nloop "+str(self.nloop)+"\n"
                                content_solute += "  number "+str(solute_up)+"\n"
                                if self.solute_prot_dist is not None and self.pdb[bilayer] != "None":
                                    content_solute += "  outside cylinder 0. 0. "+str(minmax[2]-self.solute_prot_dist)+" 0. 0. 1. "+str(max_rad+self.solute_prot_dist)+" "+str(minmax[5]-minmax[2]+self.solute_prot_dist)+" \n"
                                if self.curvature is not None:
                                    if not self.pbc:
                                        content_solute += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                                    content_solute += "  outside sphere 0. 0. "+str(-sphere_radius)+" "+str(sphere_radius+leaflet_z+z_offset)+"\n"
                                elif self.xygauss:
                                    if not self.pbc:
                                        content_solute += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                                    content_solute += "  below xygauss 0. 0. "+self.xygauss[0]+" "+self.xygauss[1]+" "+str(leaflet_z+z_offset)+" "+self.xygauss[2]+"\n"
                                else:
                                    if self.pbc:
                                        content_solute += "  above plane 0. 0. 1. "+str(leaflet_z+z_offset)+"\n" 
                                    else:
                                        content_solute += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(leaflet_z+z_offset)+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(round(Z_dim[bilayer][1]+z_offset,2))+"\n"
                                content_solute += "end structure\n\n"
                        else:
                            content_solute += "structure "+self.solute[i]+"\n"
                            content_solute += "  nloop "+str(self.nloop)+"\n"
                            content_solute += "  number "+str(solute_down+solute_up)+"\n"
                            if self.solute_prot_dist is not None and self.pdb[bilayer] != "None":
                                    content_solute += "  outside cylinder 0. 0. "+str(minmax[2]-self.solute_prot_dist)+" 0. 0. 1. "+str(max_rad+self.solute_prot_dist)+" "+str(minmax[5]-minmax[2]+self.solute_prot_dist)+" \n"
                            if not self.pbc:
                                content_solute += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(round(Z_dim[bilayer][1]+z_offset,2))+"\n"
                            content_solute += "end structure\n\n"
    
                elif (self.solute is not None or self.solute_con is not None) and not self.solute_inmem:
                    logger.error("ERROR:\n    The solute parameters are incomplete. Please include both a solute pdb and a concentration.")
                    exit()
    
                ################################# SALT & CHARGES #############################
    
                pos_up = pos_down = 0
                neg_up = neg_down = 0
                charge = charges[bilayer]
                if not self.solvate:
                    for leaflet in composition[bilayer]:
                        for lipid in composition[bilayer][leaflet]:
                            charge_lip += int(round(composition[bilayer][leaflet][lipid]*lipnum_dict[bilayer][leaflet]))*int(self.parameters[lipid]["charge"])
                if charge_lip != 0:
                    logger.debug("The lipids contribute a charge of "+str(charge_lip)+" to the system. It will be considered for the neutralization.")
                    charge += charge_lip
                if charge_solute != 0:
                    logger.debug("The solutes contribute a charge of "+str(charge_solute)+" to the system. It will be considered for the neutralization.")
                    charge += charge_solute
                if charge > 0:
                    neg_up = int(round(charge*(solvent_vol_up/(solvent_vol_tot))))
                    neg_down = int(round(charge*(solvent_vol_down/(solvent_vol_tot))))
                else:
                    pos_up = int(round((abs(charge)/self.ion_dict[self.salt_c][1])*(solvent_vol_up/(solvent_vol_tot))))
                    pos_down = int(round((abs(charge)/self.ion_dict[self.salt_c][1])*(solvent_vol_down/(solvent_vol_tot))))
                con_pos = (pos_up+pos_down)/((solvent_vol_tot)*avogadro/(1*10**27))
                con_neg = (neg_up+neg_down)/((solvent_vol_tot)*avogadro/(1*10**27))
                if self.salt:
                    if con_pos > saltcon or con_neg > saltcon*self.ion_dict[self.salt_c][1]:
                #        print pos_up, pos_down, neg_up, neg_down, saltnum_up, saltnum_down
                        logger.warning("""WARNING:
                The concentration of ions required to neutralize the system is higher than the concentration specified.
                Either increase the salt concentration by using the --saltcon flag or run the script without the --salt flag.""")
                        logger.info("Positive ion concentration: "+str(round(con_pos,3)))
                        logger.info("Negative ion concentration: "+str(round(con_neg,3)))
                        logger.info("Salt concentration specified: "+str(saltcon))
                        if override_salt:
                            if self.verbose:
                                logger.info("Overriding salt concentration...")
                            saltcon = max(con_pos,con_neg)
                            pass
                        else:
                            exit()
                    saltnum_up   = int(solvent_vol_up*saltcon*avogadro/(1*10**27))
                    saltnum_down = int(solvent_vol_down*saltcon*avogadro/(1*10**27))
    
                    if abs(charge)/2 < min(saltnum_up,saltnum_down):
                        pos_up   += saltnum_up-abs(charge)/(self.ion_dict[self.salt_c][1]*2)
                        pos_down += saltnum_down-abs(charge)/(self.ion_dict[self.salt_c][1]*2)
                        neg_up   += saltnum_up*self.ion_dict[self.salt_c][1]-abs(charge)/2
                        neg_down += saltnum_down*self.ion_dict[self.salt_c][1]-abs(charge)/2
    
                    if self.charge_imbalance != 0:
                        if n % 2 == 0:
                            if self.imbalance_ion == "cat":
                                pos_up   += self.charge_imbalance/self.ion_dict[self.salt_c][1]
                                pos_down -= self.charge_imbalance/self.ion_dict[self.salt_c][1]
                            else:
                                neg_up   -= self.charge_imbalance
                                neg_down += self.charge_imbalance
                        else:
                            if self.imbalance_ion == "cat":
                                pos_up   -= self.charge_imbalance/self.ion_dict[self.salt_c][1]
                                pos_down += self.charge_imbalance/self.ion_dict[self.salt_c][1]
                            else:
                                neg_up   += self.charge_imbalance
                                neg_down -= self.charge_imbalance
    
                    charge_data= """
    Salt and charge info:
    Upper positive charges   = %-9s
    Lower positive charges   = %-9s
    Upper negative charges   = %-9s
    Lower negative charges   = %-9s
    Charge imbalance         = %-9s
    Upper salt number        = %-9s
    Lower salt number        = %-9s
                """
                    logger.debug(charge_data % ( pos_up, pos_down, neg_up, neg_down, self.charge_imbalance, saltnum_up, saltnum_down ))
    
                    checkup = [pos_up,pos_down,neg_up,neg_down]
                    if any(v < 0 for v in checkup):
                        logger.critical("CRITICAL:\n  The applied charge imbalance caused a negative number of ions! Check your input")
                        exit()
    
    
                #    new_con_neg = (neg_up+neg_down)/(((solvent_vol_tot)*avogadro/(1*10**27)))
                #    new_con_pos = (pos_up+pos_down)/(((solvent_vol_tot)*avogadro/(1*10**27)))
    
                try:
                    shutil.copy(os.path.join(script_path, rep, "pdbs", cation+".pdb"), "./")
                except:
                    pdbtar.extract(cation+".pdb")
                self.created_notrun.append(cation+".pdb")
                try:
                    shutil.copy(os.path.join(script_path, rep, "pdbs", anion+".pdb"), "./")
                except:
                    pdbtar.extract(anion+".pdb")
                self.created_notrun.append(anion+".pdb")
                if not self.solvate and not self.self_assembly:
                    if pos_down > 0:
                        content_ion += "structure "+cation+".pdb\n"
                        content_ion += "  nloop "+str(self.nloop)+"\n"
                        content_ion += "  number "+str(int(pos_down))+"\n"
                        if self.curvature is not None:
                            if not self.pbc:
                                content_ion += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                            content_ion += "  inside sphere 0. 0. "+str(-sphere_radius)+" "+str(sphere_radius-leaflet_z+z_offset)+"\n"
                        elif self.xygauss:
                            if not self.pbc:
                                content_ion += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                            content_ion += "  below xygauss 0. 0. "+self.xygauss[0]+" "+self.xygauss[1]+" "+str(-leaflet_z+z_offset)+" "+self.xygauss[2]+"\n"
                        else:
                            if self.pbc:
                                content_ion += "  below plane 0. 0. 1. "+str(-leaflet_z+z_offset)+"\n" 
                                if bilayer > 0:
                                    content_ion += "  above plane 0. 0. 1. "+str(round(Z_dim[bilayer][0]+z_offset,2))+"\n"
                            else:
                                content_ion += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(-leaflet_z+z_offset)+"\n"
                        content_ion += "end structure\n\n"
                    if pos_up > 0:
                        content_ion += "structure "+cation+".pdb\n"
                        content_ion += "  nloop "+str(self.nloop)+"\n"
                        content_ion += "  number "+str(int(pos_up))+"\n"
                        if self.curvature is not None:
                            if not self.pbc:
                                content_ion += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                            content_ion += "  outside sphere 0. 0. "+str(-sphere_radius)+" "+str(sphere_radius+leaflet_z+z_offset)+"\n"
                        elif self.xygauss:
                            if not self.pbc:
                                content_ion += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                            content_ion += "  over xygauss 0. 0. "+self.xygauss[0]+" "+self.xygauss[1]+" "+str(leaflet_z+z_offset)+" "+self.xygauss[2]+"\n"
                        else:
                            if self.pbc:
                                content_ion += "  above plane 0. 0. 1. "+str(leaflet_z+z_offset)+"\n"
                                if bilayer+1 < len(composition):
                                    content_ion += "  below plane 0. 0. 1. "+str(round(Z_dim[bilayer][1]+z_offset,2))+"\n"
                            else:
                                content_ion += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(leaflet_z+z_offset)+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(round(Z_dim[bilayer][1]+z_offset,2))+"\n"
                        content_ion += "end structure\n\n"
                    if neg_down > 0:
                        content_ion += "structure "+anion+".pdb\n"
                        content_ion += "  nloop "+str(self.nloop)+"\n"
                        content_ion += "  number "+str(int(neg_down))+"\n"
                        if self.curvature is not None:
                            if not self.pbc:
                                content_ion += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                            content_ion += "  inside sphere 0. 0. "+str(-sphere_radius)+" "+str(sphere_radius-leaflet_z+z_offset)+"\n"
                        elif self.xygauss:
                            if not self.pbc:
                                content_ion += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                            content_ion += "  below xygauss 0. 0. "+self.xygauss[0]+" "+self.xygauss[1]+" "+str(-leaflet_z+z_offset)+" "+self.xygauss[2]+"\n"
                        else:
                            if self.pbc:
                                content_ion += "  below plane 0. 0. 1. "+str(-leaflet_z+z_offset)+"\n"
                                if bilayer > 0:
                                    content_ion += "  above plane 0. 0. 1. "+str(round(Z_dim[bilayer][0]+z_offset,2))+"\n"
                            else:
                                content_ion += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(-leaflet_z+z_offset)+"\n"
                        content_ion += "end structure\n\n"
                    if neg_up > 0:
                        content_ion += "structure "+anion+".pdb\n"
                        content_ion += "  nloop "+str(self.nloop)+"\n"
                        content_ion += "  number "+str(int(neg_up))+"\n"
                        if self.curvature is not None:
                            if not self.pbc:
                                content_ion += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                            content_ion += "  outside sphere 0. 0. "+str(-sphere_radius)+" "+str(sphere_radius+leaflet_z+z_offset)+"\n"
                        elif self.xygauss:
                            if not self.pbc:
                                content_ion += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                            content_ion += "  over xygauss 0. 0. "+self.xygauss[0]+" "+self.xygauss[1]+" "+str(leaflet_z+z_offset)+" "+self.xygauss[2]+"\n"
                        else:
                            if self.pbc:
                                content_ion += "  above plane 0. 0. 1. "+str(leaflet_z+z_offset)+"\n"
                                if bilayer+1 < len(composition):
                                    content_ion += "  below plane 0. 0. 1. "+str(round(Z_dim[bilayer][1]+z_offset,2))+"\n"
                            else:
                                content_ion += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(leaflet_z+z_offset)+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(round(Z_dim[bilayer][1]+z_offset,2))+"\n"
                        content_ion += "end structure\n\n"
                else:
                    if pos_down+pos_up > 0:
                        content_ion += "structure "+cation+".pdb\n"
                        content_ion += "  nloop "+str(self.nloop)+"\n"
                        content_ion += "  number "+str(int(pos_down+pos_up))+"\n"
                        if not self.pbc:
                            content_ion += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(round(Z_dim[bilayer][1]+z_offset,2))+"\n"
                        content_ion += "end structure\n\n"
                    if neg_down+neg_up > 0:
                        content_ion += "structure "+anion+".pdb\n"
                        content_ion += "  nloop "+str(self.nloop)+"\n"
                        content_ion += "  number "+str(int(neg_down+neg_up))+"\n"
                        if not self.pbc:
                            content_ion += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(round(Z_dim[bilayer][1]+z_offset,2))+"\n"
                        content_ion += "end structure\n\n"
    
                for i, solvent in enumerate(self.solvents.split(":")):
                    solvent_pdb   = solvent+".pdb"
                    solvents_up   = int((watnum_up/solvent_con)*(solvent_ratios[i]/float(sum(solvent_ratios)))*(avogadro/(1*10**24))*(float(self.sparameters[solvent]["density"])/float(self.sparameters[solvent]["MW"])))
                    solvents_down = int((watnum_down/solvent_con)*(solvent_ratios[i]/float(sum(solvent_ratios)))*(avogadro/(1*10**24))*(float(self.sparameters[solvent]["density"])/float(self.sparameters[solvent]["MW"])))

                    content_solvent_header = "structure "+solvent_pdb+"\n"
                    if self.sirah:
                        content_solvent_header += "  atoms 5\n"
                        content_solvent_header += "    radius 2.5\n"
                        content_solvent_header += "  end atoms\n"
                    elif solvent == "WAT" and self.watorient:
                        content_solvent_header += "  atoms 1\n"
                        content_solvent_header += "    radius 1.5\n"
                        content_solvent_header += "  end atoms\n"
                    content_solvent_header += "  nloop "+str(self.nloop)+"\n"

                    if not self.solvate and not self.self_assembly:
                        if solvents_down > 0:
                            content_solvent += content_solvent_header
                            content_solvent += "  number "+str(solvents_down)+"\n" # deleted -pos_down-neg_down. Was it necessary?
                            if self.curvature is not None:
                                if not self.pbc:
                                    content_solvent += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                                content_solvent += "  inside sphere 0. 0. "+str(-sphere_radius)+" "+str(sphere_radius-leaflet_z+z_offset)+"\n"
                            elif self.xygauss:
                                if not self.pbc:
                                    content_solvent += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                                content_solvent += "  below xygauss 0. 0. "+self.xygauss[0]+" "+self.xygauss[1]+" "+str(-leaflet_z+z_offset)+" "+self.xygauss[2]+"\n"
                            else:
                                if self.pbc:
                                    content_solvent += "  below plane 0. 0. 1. "+str(-leaflet_z+z_offset)+"\n"
                                    if bilayer > 0:
                                        content_solvent += "  above plane 0. 0. 1. "+str(round(Z_dim[bilayer][0]+z_offset,2))+"\n"
                                else:
                                    content_solvent += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(-leaflet_z+z_offset)+"\n"
                            content_solvent += "end structure\n\n"
                        if solvents_up > 0:
                            content_solvent += content_solvent_header
                            content_solvent += "  number "+str(solvents_up)+"\n" # deleted -pos_up-neg_up. Was it necessary?
                            if self.curvature is not None:
                                if not self.pbc:
                                    content_solvent += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                                content_solvent += "  outside sphere 0. 0. "+str(-sphere_radius)+" "+str(sphere_radius+leaflet_z+z_offset)+"\n"
                            elif self.xygauss:
                                if not self.pbc:
                                    content_solvent += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(Z_dim[bilayer][1])+"\n"
                                content_solvent += "  over xygauss 0. 0. "+self.xygauss[0]+" "+self.xygauss[1]+" "+str(leaflet_z+z_offset)+" "+self.xygauss[2]+"\n"
                            else:
                                if self.pbc:
                                    content_solvent += "  above plane 0. 0. 1. "+str(leaflet_z+z_offset)+"\n"
                                    if bilayer+1 < len(composition):
                                        content_solvent += "  below plane 0. 0. 1. "+str(round(Z_dim[bilayer][1]+z_offset,2))+"\n"
                                else:
                                    content_solvent += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(leaflet_z+z_offset)+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(round(Z_dim[bilayer][1]+z_offset,2))+"\n"
                            content_solvent += "end structure\n\n"
                    elif solvents_down+solvents_up > 0:
                        content_solvent += content_solvent_header
                        content_solvent += "  number "+str(solvents_down+solvents_up)+"\n" # deleted -pos-neg. Was it necessary?
                        if not self.pbc:
                            content_solvent += "  inside box "+str(round(X_min,2))+" "+str(round(Y_min,2))+" "+str(round(Z_dim[bilayer][0]+z_offset,2))+" "+str(round(X_max,2))+" "+str(round(Y_max,2))+" "+str(round(Z_dim[bilayer][1]+z_offset,2))+"\n"
                        content_solvent += "end structure\n\n"
    
            if self.nocounter:
                logger.info("Ions will not be added. System charge will be handled downstream.")
                content_ion = ""
            self.contents = content_header+content_prot+content_lipid+content_solvent+content_ion+content_solute

            if not self.solvate:
                box_data = """
                Lower water box vol    = %-9s A^3
                Upper water box vol    = %-9s A^3
                Mem. upper leaflet vol = %-9s A^3
                Mem. lower leaflet vol = %-9s A^3
    
                """
    
                logger.debug(box_data % (round(solvent_vol_down,2), round(solvent_vol_up,2), round(lipid_vol_up,2), round(lipid_vol_down,2)))
            else:
                box_data = """
                Water box vol    = %-9s A^3
    
                """
    
                logger.debug(box_data % (round(solvent_vol_down+solvent_vol_up,2)))

            return True
    
    

    ############## SET OF USED FUNCTIONS  #####################
    def mempro_align(self,pdb,keepligs=False,double_span=False,verbose=False,overwrite=False,n_ter="in"):
        output = pdb[:-4]+n_ter+"_MEMPRO.pdb"
        pdb_base = os.path.basename(pdb)
        tmp_prefix = "" if self.keep_mempro else "_tmp_"
        tmp_folder = os.path.abspath(tmp_prefix+pdb_base[:-4]+n_ter+"_MEMPRO")
        out_dir = tmp_folder + os.path.sep
        info_path = os.path.join(tmp_folder, "Rank_1", "info_rank_1.txt")
        if os.path.exists(output) and not overwrite:
            logger.info("MemPrO output exists at %s; skipping MemPrO execution.", output)
            self._apply_mempro_curvature(info_path)
            if double_span:
                if not os.path.exists(info_path):
                    logger.warning(
                        "MemPrO info_rank_1.txt not found for cached output; "
                        "using z_offset=0.0. Rerun with --overwrite to regenerate."
                    )
                    return (output, 0.0)
                z_offset = None
                with open(info_path, "r") as f:
                    for line in f:
                        if line.startswith("Iter-Membrane distance"):
                            try:
                                z_offset = float(line.split(":")[1].split()[0])
                            except:
                                z_offset = None
                            break
                if z_offset is None:
                    logger.warning(
                        "Could not parse inter-membrane distance from cached MemPrO output; "
                        "using z_offset=0.0. Rerun with --overwrite to regenerate."
                    )
                    z_offset = 0.0
                return (output, z_offset)
            return output
        if not os.path.exists(output) or overwrite:
            if not os.path.exists(tmp_folder):
                os.mkdir(tmp_folder)
            if self.keep_mempro:
                self.created_mempro.append(tmp_folder)
            else:
                self.created.append(tmp_folder)
            if "NUM_CPU" not in os.environ or not os.environ["NUM_CPU"].strip():
                os.environ["NUM_CPU"] = str(os.cpu_count() or 1)
            if "PATH_TO_MARTINI" not in os.environ or not os.environ["PATH_TO_MARTINI"].strip():
                martini_candidates = sorted(glob.glob(os.path.join(script_path, "data", "*martini*.itp")))
                if martini_candidates:
                    os.environ["PATH_TO_MARTINI"] = martini_candidates[0]
                else:
                    logger.critical("CRITICAL:\n  PATH_TO_MARTINI is required for MemPrO. Place a martini itp in data/ or set the env var.")
                    exit()
            if "PATH_TO_INSANE" not in os.environ or not os.environ["PATH_TO_INSANE"].strip():
                insane_path = None
                for mod_name in ("mempro", "MemPrO"):
                    spec = importlib.util.find_spec(mod_name)
                    if spec and spec.submodule_search_locations:
                        base = spec.submodule_search_locations[0]
                        candidate = os.path.join(base, "Insane4MemPrO.py")
                        if os.path.exists(candidate):
                            insane_path = candidate
                            break
                if insane_path:
                    os.environ["PATH_TO_INSANE"] = insane_path
            if not self.mempro:
                logger.critical("CRITICAL:\n  MemPrO executable not found. Use --mempro to point to MemPrO.")
                exit()
            if keepligs:
                logger.warning("WARNING: MemPrO ignores HETATM records; ligands may be dropped during orientation.")
            if os.path.isfile(self.mempro) and self.mempro.endswith(".py"):
                cmd = [sys.executable, self.mempro]
            else:
                cmd = [self.mempro]
            cmd += ["-f", pdb, "-o", out_dir, "-ng", str(self.mempro_grid), "-ni", str(self.mempro_iters), "-rank", self.mempro_rank]
            if double_span:
                cmd.append("-dm")
            if self.mempro_curvature:
                cmd.append("-c")
            if self.mempro_args:
                cmd += shlex.split(self.mempro_args)
            if verbose:
                logger.info("Running MemPrO: %s", " ".join(cmd))
            result = subprocess.run(cmd)
            if result.returncode != 0:
                logger.critical("CRITICAL:\n  MemPrO failed to orient the protein. Check MemPrO logs and inputs.")
                exit()
            oriented = os.path.join(tmp_folder, "Rank_1", "oriented_rank_1.pdb")
            if not os.path.exists(oriented):
                logger.critical("CRITICAL:\n  MemPrO output not found at %s", oriented)
                exit()
            shutil.copy(oriented, output)
            cleaned = []
            with open(output, "r") as f:
                for line in f:
                    if not (line.startswith("ATOM") or line.startswith("HETATM")):
                        continue
                    if line[17:20].strip() == "DUM":
                        continue
                    cleaned.append(line)
            with open(output, "w") as f:
                f.writelines(cleaned)
        self._apply_mempro_curvature(info_path)
        if double_span:
            if not os.path.exists(info_path):
                logger.critical("CRITICAL:\n  MemPrO info_rank_1.txt not found for double_span orientation.")
                exit()
            z_offset = None
            with open(info_path, "r") as f:
                for line in f:
                    if line.startswith("Iter-Membrane distance"):
                        try:
                            z_offset = float(line.split(":")[1].split()[0])
                        except:
                            z_offset = None
                        break
            if z_offset is None:
                logger.critical("CRITICAL:\n  Could not parse inter-membrane distance from MemPrO output.")
                exit()
            return (output, z_offset)
        return output

    def _parse_mempro_global_curvature(self, info_path):
        if not os.path.exists(info_path):
            return None
        with open(info_path, "r") as handle:
            for line in handle:
                if "Global curvature" not in line:
                    continue
                try:
                    return float(line.split(":")[1].split()[0])
                except:
                    return None
        return None

    def _apply_mempro_curvature(self, info_path):
        if not self.mempro_curvature:
            return
        curvature = self._parse_mempro_global_curvature(info_path)
        if curvature is None:
            logger.warning("MemPrO Global curvature not found in %s; --curvature unchanged.", info_path)
            return
        if abs(curvature) < 1e-9:
            logger.warning("MemPrO Global curvature is ~0.0; treating membrane as flat.")
            return
        if self.curvature is not None:
            logger.info("MemPrO Global curvature available; keeping user-provided --curvature=%s.", self.curvature)
            return
        if self.vol or self.solvate:
            logger.critical("CRITICAL:\n  MemPrO curvature is not compatible with --vol or --solvate.")
            exit()
        self.curvature = curvature
        self.curv_radius = 1 / curvature
        logger.info("Using MemPrO Global curvature as --curvature: %s", curvature)

    def pdbvol(self,pdb,spacing=0.5,overwrite=False):
        output = pdb[:-4]+".grid.pdb"
        if os.path.exists(output) and not overwrite:
            filelength = len(open(output,"r").readlines())
            vol        = filelength*spacing**3
            return (output, vol)
        else:
            logger.debug("PDBREMIX grid start: pdb=%s spacing=%s output=%s", pdb, spacing, output)
            filelines = open(pdb,"r").readlines()
            temp_pdb = []
            for line in filelines:
                if (line[0:4] == "ATOM" or line[0:6] == "HETATM") and  line[17:20].strip() != "DUM":
                    temp_pdb.append(line)
            logger.debug("PDBREMIX grid: input lines=%s kept=%s", len(filelines), len(temp_pdb))
            temp = open("temp.pdb","w").writelines(temp_pdb)
            logger.debug("PDBREMIX grid: temp.pdb written, reading with pdbremix")
            atoms = pdbatoms.read_pdb("temp.pdb")
            pdbatoms.add_radii(atoms)
            logger.debug("PDBREMIX grid: calling volume()")
            vol = volume(atoms, spacing, output, verbose=False)
            logger.debug("PDBREMIX grid: volume() done, vol=%s", vol)
            os.remove("temp.pdb")
            return (output, vol)

    def run_packmol(self):
        with open(self.packlog+".inp","w+") as script:
            script.write(self.contents)
            script.seek(0, os.SEEK_SET)
            logger.debug("Script for packmol written to "+self.packlog+".inp")

            if self.packmol_cmd and self.run:
                self._used_tools.add("packmol")
                logger.info("\nRunning Packmol...")
                log = open(self.packlog+".log","w")
                p = subprocess.Popen(self.packmol_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=script)
                global pid
                pid = p.pid
                def kill_child():
                    if pid is None:
                        pass
                    else:
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except:
                            pass
                atexit.register(kill_child)
                GENCAN_track = 0
                TOTGENCAN    = 0
                FRAME        = 0
                energy_values = []
                starting = True
                all_together = False
                progress_bar = True
                step = 1
                while True:
                    output = p.stdout.readline().decode('utf-8')
                    log.write(output)
                    if output == '' and p.poll() is not None:
                        break
                    if "Number of independent structures:" in output.strip():
                        structs = output.strip().split()[-1]
                        if not self.packall:
                            if not self.onlymembrane:
                                TOTAL = int(self.nloop)*(int(structs)-1)+self.nloop_all
                            else:
                                TOTAL = int(self.nloop)*(int(structs))+self.nloop_all
                        else:
                            TOTAL = int(self.nloop_all)
                        mag   = int(math.log10(TOTAL))+2
                        if not self.noprogress:
                            p_bar = tqdm.tqdm(total=TOTAL)

                    if "Starting GENCAN loop"  in output.strip():
                        if GENCAN_track < self.nloop+1:
                            GENCAN_track += 1; TOTGENCAN += 1
                        if self.noprogress:
                            if GENCAN_track % 10 == 0:
                                sys.stdout.write(".")
                                sys.stdout.flush()
                        if not self.noprogress and GENCAN_track <= self.nloop:
                            p_bar.update(1)
                    if "Packing molecules " in output.strip():
                        if self.noprogress:
#                            sys.stdout.write("\033[F")
                            sys.stdout.write("\r" +"\nProcessing segment "+output.strip().split()[-1]+" of "+structs)
                            sys.stdout.flush()
                        else:
                            if int(self.nloop)-GENCAN_track > 0 and not starting:
                                p_bar.update(int(self.nloop)-GENCAN_track)
                            else:
                                starting = False
                            p_bar.set_description("Molecule segment %s/%s" % (output.strip().split()[-1],structs))
                        GENCAN_track = 0
                    if ("Current solution written to file" in output.strip() or "Writing current (perhaps bad) structure to file" in output.strip()) and self.traj:
                        FRAME += 1
                        shutil.move(self.outfile,self.outfile.replace(".pdb","_"+("{:0"+str(mag)+"d}").format(FRAME)+".pdb"))
                    if "Packing all molecules together" in output.strip():
                        all_together = True
                        if not self.noprogress:
                            if int(self.nloop)-GENCAN_track > 0 and not self.packall:
                                p_bar.update(int(self.nloop)-GENCAN_track)
                            p_bar.set_description("All-together Packing")
                        logger.debug("\nIndividual packing processes complete. Initiating all-together packing. This might take a while!")
                        sys.stdout.write("\033[F")
                        sys.stdout.write("\r" +"\nAll-together Packing")
                        sys.stdout.flush()
                        GENCAN_track = 0
                        self.nloop = self.nloop_all
                    if "All-type function" in output.strip():
                        fnx_value = float(output.strip().split()[-1])
                        energy_values.append((TOTGENCAN,fnx_value))
                    if "Function value from last GENCAN" in output.strip() and all_together:
                        fnx_value = float(output.strip().split()[-1])
                        energy_values.append((TOTGENCAN,fnx_value))
                    if "Success!" in output.strip():
                        if not self.noprogress:
                            p_bar.update(int(self.nloop)-GENCAN_track)

                if self.run and self.plot:
                    try:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        matplotlib.rcParams["font.sans-serif"]='Arial'
                        plt.plot(*list(zip(*energy_values)),color="black")
                        plt.xlabel('Iteration')
                        plt.ylabel('Objective function')
                        plt.savefig(self.outfile+'.png')
                    except:
                        logger.error("ERROR:\n    Matplotlib could not be imported. Check that you have a working version.")
                    with open("GENCAN.log","w") as gencanlog:
                        gencanlog.write("\n".join("%s %s" % x for x in energy_values))

                if not os.path.exists(self.outfile):
                    logger.critical("CRITICAL:\n  No output file generated by PACKMOL. Check packmol.log and if the initial set of constraints are adequate for your system (e.g. leaflet size, tail and head planes)!")
                    exit()

                if self.sirah:
                    fixwt4 = []
                    with open(self.outfile, "r") as f:
                        for line in f:
                            if "COM WT4" in line:
                                continue
                            fixwt4.append(line)
                    with open(self.outfile, "w") as f:
                        f.writelines(fixwt4)
            else:
                logger.info("The script generated can be run by using a packmol executable (e.g. 'packmol < memgen.inp').")
                if self.sirah:
                    logger.info("An artificial 'COM WT4' atom was added to avoid clashes. Make sure to delete it post packing and before parametrizing with 'grep -v \"COM WT4\"'")
                exit()

    def postprocess(self):
        if not self.charmm:
            logger.info("Transforming to AMBER")
            charmmlipid2amber(self.outfile,self.outfile, os.path.join(script_path, "lib", "charmmlipid2amber", "charmmlipid2amber.csv"))
    
            if self.sterols_PI_used:
                logger.info("Sterols and/or PI residues were packed. Potential piercing lipid tails will be searched")
                try:
                    to_remove = find_piercing_lipids(self.outfile, verbose=True, hexadecimal_indices=self.hexadecimal_indices)
                    try:                
                        if len(to_remove) > 0:
                           self.outfile = remove_piercing_lipids(self.outfile, to_remove, outfile=self.outfile.replace(".pdb","_noclash.pdb"), verbose=True, hexadecimal_indices=self.hexadecimal_indices)
                    except:
                        logger.warning("Lipid piercing removal failed. Check your structure manually in case of clashing lipid tails!")
                except:
                    logger.warning("Lipid piercing finder failed. Check your structure manually in case of clashing lipid tails!")

        if self.hexadecimal_indices:
            convert_pdb_indices_to_hybrid36(self.outfile, atom_base=16, res_base=16)
        if self.xponge:
            logger.info("Applying Xponge ion names for compatibility with Xponge")
            apply_xponge_ion_names(self.outfile)
    
    def cleanup(self):
        if self.delete:
            logger.debug("Deleting temporary files...")
            for file in self.created:
                try:
                    if os.path.isfile(file):
                        os.remove(file)
                    elif os.path.isdir(file):
                        shutil.rmtree(file) 
                except:
                    pass
            if self.run:
                for file in self.created_notrun:
                    try:
                        os.remove(file)
                    except:
                        pass
    
        warn = "#Packing process finished. Check your final structure, particularly for lipids inserted in proteins, protein tunnels or piercing rings!#"
        print("\n"+"#"*len(warn))
        logger.info(warn)
        print("#"*len(warn))
        self._emit_references()
        print("DONE!")

    def _emit_references(self):
        references = {
            "packmol-memgen": [
                "Schott-Verdugo, S.; Gohlke, H. "
                "PACKMOL-Memgen: A Simple-To-Use, Generalized Workflow for Membrane-Protein–Lipid-Bilayer System Building. "
                "J. Chem. Inf. Model. 2019, 59 (6), 2522–2528. https://doi.org/10.1021/acs.jcim.9b00269."
            ],
            "packmol": [
                "Martínez, L.; Andrade, R.; Birgin, E. G.; Martínez, J. M. "
                "PACKMOL: A Package for Building Initial Configurations for Molecular Dynamics Simulations. "
                "J. Comput. Chem. 2009, 30 (13), 2157–2164. https://doi.org/10.1002/jcc.21224."
            ],
            "mempro": [
                "Parrag, M.; Stansfeld, P. J. "
                "MemPrO: A Predictive Tool for Membrane Protein Orientation. "
                "J. Chem. Theory Comput. 2025. https://doi.org/10.1021/acs.jctc.5c01433."
            ],
            "martini": [
                "Souza, P. C. T.; Alessandri, R.; Barnoud, J.; et al. "
                "Martini 3: a general purpose force field for coarse-grained molecular dynamics. "
                "Nat Methods 2021, 18, 382–388. https://doi.org/10.1038/s41592-021-01098-3."
            ],
            "pdb2pqr": [
                "Dolinsky, T. J.; Czodrowski, P.; Li, H.; Nielsen, J. E.; Jensen, J. H.; Klebe, G.; Baker, N. A. "
                "PDB2PQR: Expanding and Upgrading Automated Preparation of Biomolecular Structures for Molecular Simulations. "
                "Nucleic Acids Res. 2007, 35 (Web Server issue), W522–W525. https://doi.org/10.1093/nar/gkm276.",
                "Dolinsky, T. J.; Nielsen, J. E.; McCammon, J. A.; Baker, N. A. "
                "PDB2PQR: An Automated Pipeline for the Setup of Poisson–Boltzmann Electrostatics Calculations. "
                "Nucleic Acids Res. 2004, 32 (Web Server issue), W665–W667. https://doi.org/10.1093/nar/gkh381.",
            ],
        }
        used = [tool for tool in ("packmol-memgen", "packmol", "mempro", "martini", "pdb2pqr") if tool in self._used_tools]
        if not used:
            return
        tool_labels = {
            "packmol-memgen": "PACKMOL-Memgen",
            "packmol": "PACKMOL",
            "mempro": "MemPrO",
            "martini": "Martini",
            "pdb2pqr": "PDB2PQR",
        }
        header = "References (for tools used in this run):"
        logger.info(header)
        for tool in used:
            entries = references.get(tool, [])
            for entry in entries:
                label = tool_labels.get(tool, tool)
                line = "- [" + label + "] " + entry
                logger.info(line)

    def run_all(self):
        pack = self.prepare()
        if pack:
            self.run_packmol()
        self.postprocess()
        self.cleanup()

    def lipids_df(self):
        try:
            return pd.DataFrame.from_dict(self.parameters).T
        except:
            print("Run prepare() to populate the lipid parameters")

    def solvents_df(self):
        try:
            return pd.DataFrame.from_dict(self.sparameters).T
        except:
            print("Run prepare() to populate the solvent parameters")

    def ions_df(self):
        return pd.DataFrame.from_dict(amber_ion_dict).T

def _setup_logging(log_path):
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    streamer = logging.StreamHandler()
    streamer.setLevel(logging.INFO)
    loghandle = logging.FileHandler(log_path, mode="a")
    loghandle.setLevel(logging.DEBUG)
    loghandle.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:\n%(message)s',datefmt='%m/%d/%Y %I:%M:%S %p'))
    logger.addHandler(streamer)
    logger.addHandler(loghandle)


def cli():
    args = parser.parse_args()
    _setup_logging(args.log)
    pmg = PACKMOLMemgen(args)
    pmg.run_all()


if __name__ == "__main__":
    cli()
