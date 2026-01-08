#!/usr/bin/python

from __future__ import print_function
import os, sys, math, string, copy, random, shlex, subprocess, time
import numpy as np
import logging
import contextlib
import warnings
from scipy import integrate
from .pdbremix import data

logger = logging.getLogger("pmmg_log")

#Load pdb2pqr functions if available
try:
#    from pdb2pqr.main import main_driver as pdb2pqr
#    from pdb2pqr.main import build_main_parser as pdb2pqr_args
    import shutil
    pdb2pqr = shutil.which("pdb2pqr30") or shutil.which("pdb2pqr")
    if not pdb2pqr and shutil.which("uv"):
        pdb2pqr = "uv run pdb2pqr"
except:
    logger.debug("PDB2PQR not available. Protonation with --pdb2pqr will not be available")
    pdb2pqr = False

VCH           = 21.65 # A^3
VCH2          = 27.03 # A^3
avogadro = 6.02214086*10**23
residues = {"CYS","CYX","CYM","MET","HIS","HSD","HIE","HID","HIP","HSE","SER","GLN","ASP","ASH","GLU","GLH","TYR","THR","ALA","LEU","ILE","PHE","TRP","ARG","ASN","LYS","LYN","VAL","PRO","GLY"}
cgatoms  = {"CA","CB","C","N","O"}
charged  = {"ASP":-1,"GLU":-1,"LYS":1,"ARG":1,"HIP":1,"Cl-":-1,"MG":2,"Na+":1,"CA":2,"OHE":-0.308100,
            "A":-1,"A5":-0.3081,"A3":-0.6919,"DA":-1,"DA5":-0.3079,"DA3":-0.6921,
            "C":-1,"C5":-0.3081,"C3":-0.6919,"DC":-1,"DC5":-0.3079,"DC3":-0.6921,
            "G":-1,"G5":-0.3081,"G3":-0.6919,"DG":-1,"DG5":-0.3079,"DG3":-0.6921,
            "U":-1,"U5":-0.3081,"U3":-0.6919,"DT":-1,"DT5":-0.3079,"DT3":-0.6921,
            "PTR":-2,"SEP":-2,"TPO":-2,"Y1P":-1,"S1P":-1,"T1P":-1,"H1D": 0,"H2D":-1,"H1E": 0,"H2E":-1,
            "NME":1,"ACE":-1} #These two are actually neutral, but by being added, the opposite terminal end charge is not neutralized. Should work as long as no custom terminal ends or protein constructs are used.

tails = {"LAL","MY","PA","ST","OL","AR","DHA","SA"}

sterols_PI = {"CHL1","ERG","CAM","SIT","STI","PI"}
sterol_ring_probes = [["C1","C2","C3","C4","C5","C10"],["C5","C6","C7","C8","C9","C10"],["C8","C9","C11","C12","C13","C14"],["C13","C14","C15","C16","C17"]]
PI_ring_probe = ["C31","C32","C33","C34","C35","C36"]

#masses   = {"C": 12, "S": 32, "O": 16, "H": 1, "N": 14}

def pdb2pqr_protonate(pdb,overwrite=False,ffout='AMBER',pH=7.0):
    if not pdb2pqr:
        logger.critical("PDB2PQR module was not found. Use a different method to protonate your system")
        exit()
    output_pqr = pdb[:-4]+"_H.pqr"
    output_pdb = pdb[:-4]+"_H.pdb"
    if os.path.exists(output_pdb) and os.path.exists(output_pqr) and not overwrite:
        return output_pdb
#As pdb2pqr call logging.basicConfig in the main function, it disrupts the logging setup. Calling in os.system to avoid issues
#    with open("pdb2pqr.log", "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
#        pdb2pqr_parser = pdb2pqr_args()
#        args = pdb2pqr_parser.parse_args(['--pdb-output='+output_pdb,'--ff=PARSE','--ffout=AMBER','--with-ph='+str(pH),pdb,output_pqr])
#        pdb2pqr(args)
    log_path = os.path.join(os.path.dirname(output_pdb) or ".", "pdb2pqr.log")
    cmd = shlex.split(pdb2pqr) + [
        '--pdb-output='+output_pdb,
        '--ff=PARSE',
        '--ffout=AMBER',
        '--titration-state-method=propka',
        '--with-ph='+str(pH),
        pdb,
        output_pqr,
    ]
    with open(log_path, "w") as log_handle:
        result = subprocess.run(cmd, stdout=log_handle, stderr=log_handle)
    if result.returncode != 0:
        logger.critical("CRITICAL:\n PDB2PQR failed! Check %s", log_path)
        exit()
    for _ in range(5):
        if os.path.exists(output_pdb) and os.stat(output_pdb).st_size > 0:
            break
        time.sleep(0.2)
    if not os.path.exists(output_pdb) or os.stat(output_pdb).st_size == 0:
        logger.critical("CRITICAL:\n PDB2PQR did not create %s. Check %s and input paths.", output_pdb, log_path)
        exit()
    return output_pdb

def estimated_density(MW):
    density = 1.41 + 0.145*math.exp(float(-MW)/13000)  #Protein Sci. 2004 Oct; 13(10):2825-2828
    return density

def distribute_integer(integer, fracs):
    dist = []
    tot = sum(fracs)
    for frac in fracs:
        d = int(round(integer*frac/tot))
        dist.append(d)
        tot -= frac
        integer -= d
    return dist

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def vector_angle(vec1,vec2, ref=[1,1,0]):
    cross = np.cross(vec1,vec2)
    dot = np.dot(vec1,vec2)
    dot /= np.linalg.norm(vec1)*np.linalg.norm(vec2)
    ori = np.dot(cross, ref)
    return np.arccos(dot)

def align_vectors(vec1,vec2, R=None):
    if np.allclose(vec1,np.array([0,0,0])):
        return np.array([0,0,0])
    angle = vector_angle(vec1,vec2)
    if R is None:
        R = rotation_matrix(cross,angle)
    return np.matmul(vec1,R)

def rotate_pdb(pdb_ori,tip_num,pivot_num, ref=[0,0,1],randomize=False,randomize_vec=[0,0,1]):
    pdb = copy.deepcopy(pdb_ori)
    tip = []
    pivot = []
    for res in pdb:
        for atom in pdb[res]:
            if isinstance(tip_num, list):
                if atom[1] in tip_num:
                    tip.append(pdb[res][atom])
            else:
                if atom[1] == tip_num:
                    tip = pdb[res][atom]
            if isinstance(pivot_num,list):
                if atom[1] in pivot_num:
                    pivot.append(pdb[res][atom])
            else:
                if atom[1] == pivot_num:
                    pivot = pdb[res][atom]
    if isinstance(tip_num, list):
        tip = np.mean(tip,axis=0)
    if isinstance(pivot_num,list):
        pivot = np.mean(pivot,axis=0)
    angle = vector_angle(tip-pivot,np.array(ref))
    axis = np.cross(tip-pivot,np.array(ref))   # vector perpendicular to v1 v2 plane
    R = rotation_matrix(axis,angle)
    for res in pdb:
        if randomize:
            angle = random.uniform(0,6.28)
            axis = np.array([0,0,1])
            R = np.matmul(R,rotation_matrix(axis,angle))
        for atom in pdb[res]:
            pdb[res][atom] = align_vectors(pdb[res][atom]-pivot,np.array(ref),R)+pivot
    return pdb

def randomize_pdb(pdb_ori,tip_num,pivot_num, ref=[0,0,1]):
    pdb = copy.deepcopy(pdb_ori)
    tip = []
    pivot = []
    for res in pdb:
        for atom in pdb[res]:
            if isinstance(tip_num, list):
                if atom[1] in tip_num:
                    tip.append(pdb[res][atom])
            else:
                if atom[1] == tip_num:
                    tip = pdb[res][atom]
            if isinstance(pivot_num,list):
                if atom[1] in pivot_num:
                    pivot.append(pdb[res][atom])
            else:
                if atom[1] == pivot_num:
                    pivot = pdb[res][atom]
    if isinstance(tip_num, list):
        tip = np.mean(tip,axis=0)
    if isinstance(pivot_num,list):
        pivot = np.mean(pivot,axis=0)
    angle = random.uniform(0,6.28)
    axis = np.array(ref)
    R = rotation_matrix(axis,angle)
    if len(pdb[res]) > 1:
        for res in pdb:
            for atom in pdb[res]:
                pdb[res][atom] = align_vectors(pdb[res][atom]-pivot,np.array(ref),R)+pivot
    return pdb

def translate_pdb(pdb_ori,target=None,ref_atm=None,vec=None):
    pdb = copy.deepcopy(pdb_ori)
    if target is not None and ref_atm is not None:
        for res in pdb:
            for atom in pdb[res]:
                if atom[1] == ref_atm:
                    ref = pdb[res][atom]
        tran_vec = ref-target
    elif vec is not None:
        tran_vec = vec
    else:
        print("A target and reference atom, or a translation vector has to be provided")
        exit()
    for res in pdb:
        for atom in pdb[res]:
            pdb[res][atom] = pdb[res][atom]-tran_vec
    return pdb

def superimpose_pdb(pdb1,pdb2):

    points = []

    count = 0
    while count < 3:
        key   = list(pdb1.keys())[count]
        key2  = list(pdb1[key].keys())[count]
        points.append([pdb1[key][key2],pdb2[key][key2]])
        count += 1

    trans_vec = points[0][1]-points[0][0]
#    print(trans_vec)

    for n, point in enumerate(points):
        points[n][1] = points[n][1]-trans_vec

    angle = vector_angle(points[1][1]-points[0][0],points[1][0]-points[0][0])
    axis = np.cross(points[1][1]-points[0][0],points[1][0]-points[0][0])

    R = rotation_matrix(axis,angle)


    if not np.allclose(align_vectors(points[1][1]-points[0][0],points[1][0]-points[0][0],R),points[1][0]-points[0][0],atol=1e-01):
        print("Flipping!")
        R = rotation_matrix(axis,-angle)

    pdb3 = translate_pdb(pdb2,vec=trans_vec)

    pdb_write(pdb3,outfile="trans.pdb")

    for res in pdb3:
        for atom in pdb3[res]:
            pdb3[res][atom] = align_vectors(pdb3[res][atom]-points[0][0],points[1][0]-points[0][0],R)+points[0][0]

    for n, point in enumerate(points):
        points[n][1] = align_vectors(points[n][1]-points[0][0],points[1][0]-points[0][0],R)+points[0][0]

    pdb_write(pdb3,outfile="rot1.pdb")

    axis = points[1][0]-points[0][0]

    axis_n     = axis/np.linalg.norm(axis)
    vec_proj1  = np.dot((points[2][1]-points[0][0]),axis_n)
    vec_proj2  = np.dot((points[2][0]-points[0][0]),axis_n)
    line_point1 = points[0][0] + vec_proj1*axis_n
    line_point2 = points[0][0] + vec_proj2*axis_n

    angle = vector_angle(points[2][1]-line_point1,points[2][0]-line_point1)

    R = rotation_matrix(axis,angle)


    if not np.allclose(align_vectors(points[2][1]-line_point1,points[2][0]-line_point1,R),points[2][0]-line_point1,atol=1e-02):
#        print("Flipping2!")
        R = rotation_matrix(axis,-angle)

    for res in pdb3:
        for atom in pdb3[res]:
            pdb3[res][atom] = align_vectors(pdb3[res][atom]-line_point1,points[1][0]-line_point1,R)+line_point1

    return pdb3

def sphere_dist(rad, dx, dy=0, dz=0):
    return rad*2*np.arcsin(np.sqrt(dx**2+dy**2+dz**2)/(2*rad))

def sphere_rectangle_area(rad, a, b):
    a = sphere_dist(rad, a)
    b = sphere_dist(rad, b)
    return rad**2*(2*np.pi-4*np.arccos(np.tan(a/(2*rad))*np.tan(b/(2*rad))))

def sphere_integral(a,b,c,r):
    f = lambda z, y, x: 1
    return integrate.tplquad(f, a-r, a+r,
                lambda x: -np.sqrt(r**2-(x-a)**2)+b,             lambda x: np.sqrt(r**2-(x-a)**2)+b,
                lambda x, y: -np.sqrt(r**2-(x-a)**2-(y-b)**2)+c, lambda x, y: np.sqrt(r**2-(x-a)**2-(y-b)**2)+c)[0]

def sphere_integral_square(x_min,x_max,y_min,y_max,z_min=None, z_max=None,r1=None,r2=None, a=0,b=0,c=0):
    f = lambda z, y, x: 1
    if r1 is not None and r2 is not None:
        if r1**2-(x_max-a)**2-(y_max-b)**2 < 0:
            print("Radius is too small for given dimensions!")
            raise ValueError
        return integrate.tplquad(f, x_min, x_max,
                lambda x: y_min,             lambda x: y_max,
                lambda x, y: np.sqrt(r1**2-(x-a)**2-(y-b)**2)+c, lambda x, y: np.sqrt(r2**2-(x-a)**2-(y-b)**2)+c)[0]
    elif z_min is not None and r2 is not None:
        if r2**2-(x_max-a)**2-(y_max-b)**2 < 0:
            print("Radius is too small for given dimensions!")
            raise ValueError
        return integrate.tplquad(f, x_min, x_max,
                lambda x: y_min,             lambda x: y_max,
                lambda x, y: z_min, lambda x, y: np.sqrt(r2**2-(x-a)**2-(y-b)**2)+c)[0]
    elif z_max is not None and r1 is not None:
        if r1**2-(x_max-a)**2-(y_max-b)**2 < 0:
            print("Radius is too small for given dimensions!")
            raise ValueError
        return integrate.tplquad(f, x_min, x_max,
                lambda x: y_min,             lambda x: y_max,
                lambda x, y: np.sqrt(r1**2-(x-a)**2-(y-b)**2)+c, lambda x, y: z_max)[0]
    else:
        raise ValueError

def gauss_rectangle_area(x_min,x_max,y_min,y_max,b,d,h,a=0,c=0):
    f = lambda y, x: np.sqrt(1 + (-h*(x-a)*np.exp(-(x-a)**2/(2*b**2)-(y-c)**2/(2*d**2))/b**2)**2 + (-h*(y-c)*np.exp(-(x-a)**2/(2*b**2)-(y-c)**2/(2*d**2))/d**2)**2)
    return integrate.dblquad(f, x_min, x_max,
                lambda x: y_min,lambda x: y_max)[0]

def gauss_integral_square(x_min,x_max,y_min,y_max,b,d,h,a=0,c=0,z_min=None, z_max=None, g1=None, g2=None):
    f = lambda z, y, x: 1
    f1= lambda y, x: h*np.exp(-(x-a)**2/(2*b**2)-(y-c)**2/(2*d**2))+g1
    f2= lambda y, x: h*np.exp(-(x-a)**2/(2*b**2)-(y-c)**2/(2*d**2))+g2
    if g1 is not None and g2 is not None:
        if g2-g1 < 0:
            raise ValueError
        return integrate.tplquad(f, x_min, x_max,
                lambda x: y_min,             lambda x: y_max,
                f1, f2)[0]
    elif z_min is not None and g2 is not None:
        if g2-z_min < 0:
            raise ValueError
        return integrate.tplquad(f, x_min, x_max,
                lambda x: y_min,             lambda x: y_max,
                lambda x, y: z_min, f2)[0]
    elif z_max is not None and g1 is not None:
        if z_max-g1 < 0:
            raise ValueError
        return integrate.tplquad(f, x_min, x_max,
                lambda x: y_min,             lambda x: y_max,
                f1, lambda x, y: z_max)[0]
    else:
        raise ValueError

class MembraneParams(object):
    """
    A class to store membrane params corresponding to a PDB
    """
    def __init__(self,pdb, leaflet_z, grid=None, move=False, move_vec=[0,0,0], xy_cen=False, z_cen = False, outpdb="PROT.pdb", chain=" ",renumber=False):
        #Getting variables into class attributes
        self.pdb = pdb
        self.leaflet_z = leaflet_z
        self.grid = grid
        self.move = move
        self.move_vec = move_vec
        self.xy_cen = xy_cen
        self.z_cen = z_cen
        self.outpdb = outpdb
        self.chain = chain
        self.renumber = renumber


        #Variables to store values after
        self.density = None
        self.chains = 1
        self.charge = 0
        self.mass = 0
        self.hydrogens = 0 
        self.use_hex = False
        self.mem_atoms_mass_up = 0
        self.mem_atoms_mass_down = 0
        self.solv_atoms_mass_up = 0
        self.solv_atoms_mass_down = 0
        self.new_pdb = []                    
        self.x_mem = []
        self.y_mem = []
        self.pdblines = None
        self.minmax = None

        self.x = []
        self.y = []
        self.z = []

        self._x_cen = 0
        self._y_cen = 0
        self._z_cen = 0

    def read_pdb(self):
        "Read pdb lines into class"
        file = open(self.pdb, "r")
        self.pdblines = file.readlines()
        file.close()

    def write_pdb(self): 
        new_file = open(self.outpdb,"w")
        new_file.writelines(self.new_pdb)
        new_file.close()

    def xyz_center(self):
        """
        Calculate "center" of pdb from coordinate max min average
        """
        x = []
        y = []
        z = []
        for line in self.pdblines:
            if (line[0:4] == "ATOM" or line[0:6] == "HETATM") and  line[17:20].strip() != "DUM":
                x_coord = float(line[30:38])+self.move_vec[0]
                y_coord = float(line[38:46])+self.move_vec[1]
                z_coord = float(line[46:54])+self.move_vec[2]
                x.append(float(x_coord))
                y.append(float(y_coord))
                z.append(float(z_coord))
        self._x_cen = (max(x)+min(x))/2
        self._y_cen = (max(y)+min(y))/2
        self._z_cen = (max(z)+min(z))/2

    def pdb_reindex(self):
        """
        Go over pdblines, renumber, add chain ids and skip as needed
        """
        last_chain   = None
        last_resnum  = None
        last_type    = None
        track        = None
        chain_list   = list(string.ascii_uppercase)+list(string.ascii_lowercase)+list(map(str,range(0,10)))
        resnum_index = 1
        chain = self.chain

        #Calcualte PDB center to be used later
        self.xyz_center()

        if chain == " ":
            chain_index  = 0
        else:
            try:
                chain_index = chain_list.index(chain)
            except:
                print("Chain ID not found in list")
                chain_index  = 0
        for line in self.pdblines:
            if line[0:3] == "TER":
                self.new_pdb.append("TER\n")
                continue
            if (line[0:4] == "ATOM" or line[0:6] == "HETATM") and  line[17:20].strip() != "DUM":
                if last_type is not None and line[0:6].strip() != last_type:
                    if not line.startswith("TER") and not self.new_pdb[-1].startswith("TER"):
                        self.new_pdb.append("TER\n")
                last_type = line[0:6].strip()
                residue = line[17:21].strip()
                atomnum = int(line[6:11].strip())
                atomname = line[12:16].strip()
                resnum = int(line[22:26].strip())
                if self.renumber and resnum != last_resnum:
                    last_resnum   = resnum
                    resnum_new    = resnum_index
                    resnum_index += 1
                    if resnum_new > 9999:
                        resnum_new = ((resnum_new-1)%9999)+1
                if self.renumber:
                    resnum = resnum_new
                segid  = line[72:76].strip()
                if residue == "ILE" and atomname == "CD":
                    atomname = "CD1"
                if residue == "CYM":
                    if atomname == "HN1" or atomname == "HB1":
                        continue
                if atomname == "OT1":
                    atomname = "O"
                if atomname == "OT2":
                    atomname = "OXT"
                if len(atomname) == 3:
                    ali = ">"
                else:
                    ali = "^"
                if last_chain is not None and last_chain != line[21:22]:
                    if not line.startswith("TER") and not self.new_pdb[-1].startswith("TER"):
                        self.new_pdb.append("TER\n")
                    self.chains += 1
                    chain_index += 1
                    chain = chain_list[chain_index]
                last_chain = line[21:22]
                if not self.move and not self.xy_cen:
                    x_coord = float(line[30:38])
                    y_coord = float(line[38:46])
                    z_coord = float(line[46:54])
                elif not self.move:
                    x_coord = float(line[30:38])-self._x_cen
                    y_coord = float(line[38:46])-self._y_cen
                    z_coord = float(line[46:54])
                    if self.z_cen:
                        z_coord = float(line[46:54])-self._z_cen
                elif not self.xy_cen:
                    x_coord = float(line[30:38])+self.move_vec[0]
                    y_coord = float(line[38:46])+self.move_vec[1]
                    z_coord = float(line[46:54])+self.move_vec[2]
                else:
                    x_coord = float(line[30:38])+self.move_vec[0]-self._x_cen
                    y_coord = float(line[38:46])+self.move_vec[1]-self._y_cen
                    z_coord = float(line[46:54])+self.move_vec[2]
                    if self.z_cen:
                        z_coord = float(line[46:54])+self.move_vec[2]-self._z_cen
                self.x.append(float(x_coord))
                self.y.append(float(y_coord))
                self.z.append(float(z_coord))
                line = line[0:6]+"{:>5d} {:{align}4} {:<4}{:1}{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}      {:<4}{:>2}\n".format( atomnum,atomname,residue,chain,resnum,x_coord,y_coord,z_coord,1,0,segid,atomname[0],align=ali)
                self.new_pdb.append(line)
    
                if line[17:20].strip() in charged and track != line[22:26].strip():
                    self.charge += charged[line[17:20].strip()]
                    track =  line[22:26].strip()
                #Add masses according to z location
                if line[0:4] == "ATOM":
                    element = data.guess_element(residue,atomname)
                    if element in data.masses:
                        self.mass += data.masses[element]
                        if element == "H":
                            self.hydrogens += 1
                        if self.z[-1] > self.leaflet_z:
                            self.solv_atoms_mass_up += data.masses[element]
                        elif self.z[-1] < -self.leaflet_z:
                            self.solv_atoms_mass_down += data.masses[element]
                        else:
                            if self.z[-1] >= 0:
                                self.mem_atoms_mass_up +=  data.masses[element]
                            else:
                                self.mem_atoms_mass_down +=  data.masses[element]
                            self.x_mem.append(self.x[-1])
                            self.y_mem.append(self.y[-1])
                    else:
                        print("Atom "+element+" mass will not be considered!\n")

    def read_grid(self):
        file = open(self.grid,"r").readlines()
        solv_up = 0
        solv_down = 0
        mem_up = 0
        mem_down = 0
        for line in file:
            coord = float(line[46:54])
            if coord+self.move_vec[2] > self.leaflet_z:
                solv_up += 1
            elif coord+self.move_vec[2] > 0:
                mem_up += 1
            elif coord+self.move_vec[2] >= -self.leaflet_z:
                mem_down += 1
            elif coord+self.move_vec[2] < -self.leaflet_z:
                solv_down += 1
            else:
                print("Coordinate not assigned?") #Shouldn't be called
        self.mem_vol_up = mem_up/8
        self.mem_vol_down = mem_down/8
        self.solv_vol_up = solv_up/8
        self.solv_vol_down = solv_down/8
        self.volume = self.mem_vol_up+self.mem_vol_down+self.solv_vol_up+self.solv_vol_down

    def estimated_atoms(self):
        est_density = estimated_density(self.mass)
        self.density = est_density*avogadro/10**24
        self.volume = self.mass/self.density
        self.mem_vol_up = self.mem_atoms_mass_up/self.density
        self.mem_vol_down = self.mem_atoms_mass_down/self.density
        self.solv_vol_up = self.solv_atoms_mass_up/self.density
        self.solv_vol_down = self.solv_atoms_mass_down/self.density

    def measure(self):

        self.read_pdb()
        self.pdb_reindex()
        self.write_pdb()

        if self.hydrogens == 0:
            print("Protein doesn't look to be protonated! Please consider that this will cause a bad estimation of the volume and of the packing process!\n\n")
    
        #### MAXIMUM PROTEIN XY RADIUS ###
    
        mean_x  = sum(self.x)/len(self.x)
        mean_y  = sum(self.y)/len(self.y)
        self.max_rad = max([math.sqrt((self.x[n]-mean_x)**2+(self.y[n]-mean_y)**2) for n, _ in enumerate(self.x)])
    
        #### IF GRID VOL CALCULATION, REPLACE ESTIMATION ####
   
        self.estimated_atoms() 
        if self.grid != None:
            self.read_grid()
 
        self.x.sort()
        self.y.sort()
        self.z.sort()
        self.x_mem.sort()
        self.y_mem.sort()
        self.minmax = [self.x[0], self.y[0], self.z[0], self.x[-1], self.y[-1], self.z[-1]]
        try:
            area_est = (((self.x_mem[-1]-self.x_mem[0])+(self.y_mem[-1]-self.y_mem[0]))/4)**2*math.pi # Maybe estimate the protein area in the membrane...(Not used ATM)
        except:
            print("WARNING! The protein doesn't have atoms sitting in the membrane! Make sure that it was correctly aligned and that the placement is as intended!")
        return [self.x[0], self.y[0], self.z[0], self.x[-1], self.y[-1], self.z[-1]], self.max_rad, self.charge, self.volume, self.mem_vol_up, self.mem_vol_down, self.solv_vol_up, self.solv_vol_down, self.density, self.mass, self.chains

    def __repr__(self):
        return f"<MembraneParam PDB:{self.pdb}>"
        #
        
def is_number(num):
    try:
        float(num)
        return True
    except:
        return False


def pdb_parse(pdbfile, onlybb=True):
    CA_CB = {}
    pdb = open(pdbfile,"r").readlines()
    for line in pdb:
        if (line.startswith("ATOM") or line.startswith("HETATM")):
            residue = line[17:21].strip()
            atomnum = int(line[6:11].strip())
            atomname = line[12:16].strip()
            resnum = int(line[22:26].strip())
            chain = line[21:22]
            id = (residue,resnum,chain)
            if atomname in cgatoms and residue in residues and onlybb:
                if id not in CA_CB:
                    CA_CB[id]= {}
                CA_CB[id][(atomname,atomnum)] = np.array([float(line[30:38].strip()),float(line[38:46].strip()),float(line[46:54].strip())])
            if not onlybb:
                if id not in CA_CB:
                    CA_CB[id]= {}
                CA_CB[id][(atomname,atomnum)] = np.array([float(line[30:38].strip()),float(line[38:46].strip()),float(line[46:54].strip())])
    return CA_CB

def pdb_write(CA_CB, outfile="test.pdb"):
    handle = open(outfile,"w")
    for res in sorted(CA_CB,key=lambda x:(x[2],x[1])):
        for atom in sorted(CA_CB[res], key=lambda x:x[1]):
            handle.write("ATOM  {:>5d} {:>4} {:>3}{:>2}{:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}   1.00  0.00           {:1}\n".format(atom[1],"{:<3}".format(atom[0]),res[0],res[2],res[1],CA_CB[res][atom][0],CA_CB[res][atom][1],CA_CB[res][atom][2],atom[0][0]))
    handle.close()

def pdb_parse_TER(pdbfile, onlybb=True, noH=True, filter_res=None, filter_atm=None):
    CA_CB = {}
    pdb = open(pdbfile,"r").readlines()
    molnum = 1
    tracker = 1
    hex_switch = False
    atomlimit = False
    for line in pdb:
        if line.startswith("TER"):
            molnum += 1
        if (line.startswith("ATOM") or line.startswith("HETATM")):
            if line[6:11].strip() == "*****" and not atomlimit:
                logger.warning("Found atom number limit '*****'. Atom number parsing will be unreliable")
                atomlimit = True
            residue = line[17:21].strip()
            if not str(line[6:11].strip()).isnumeric():
                hex_switch = True
            if atomlimit:
                atomnum = atomnum + 1
            else:
                atomnum = int(line[6:11].strip(),16) if hex_switch else int(line[6:11].strip()) # asume hex 16 if parsing packmol
            atomname = line[12:16].strip()
            resnum = int(line[22:26].strip())
            chain = line[21:22]
            id = (molnum,chain)
            if atomname in cgatoms and residue in residues and onlybb:
                if id not in CA_CB:
                    CA_CB[id]= {}
                CA_CB[id][(residue, resnum, atomname, atomnum, tracker)] = np.array([float(line[30:38].strip()),float(line[38:46].strip()),float(line[46:54].strip())])
            if not onlybb:
                if noH:
                    if atomname.startswith("H"):
                        continue
                if filter_res != None:
                    if residue not in filter_res:
                        continue
                if filter_atm != None:
                    if atomname not in filter_atm:
                        continue
                if id not in CA_CB:
                    CA_CB[id]= {}
                CA_CB[id][(residue, resnum, atomname, atomnum, tracker)] = np.array([float(line[30:38].strip()),float(line[38:46].strip()),float(line[46:54].strip())])
        tracker += 1
    return CA_CB



def pdb_write_TER(CA_CB, outfile="test.pdb"):
    handle = open(outfile,"w")
    for mol in sorted(CA_CB,key=lambda x:x[0]):
        for atom in sorted(CA_CB[mol], key=lambda x:(x[4],x[1])): # Packmol output comes serialized first and foremost by atomnumber
            if atom[3] < 100000:
                handle.write("ATOM  {:>5d} {:>4} {:>3}{:>2}{:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}   1.00  0.00           {:1}\n".format(atom[3],"{:<3}".format(atom[2]),atom[0],mol[1],atom[1],CA_CB[mol][atom][0],CA_CB[mol][atom][1],CA_CB[mol][atom][2],atom[0][0]))
            else:
                handle.write("ATOM  {:>5} {:>4} {:>3}{:>2}{:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}   1.00  0.00           {:1}\n".format(hex(atom[3]),"{:<3}".format(atom[2]),atom[0],mol[1],atom[1],CA_CB[mol][atom][0],CA_CB[mol][atom][1],CA_CB[mol][atom][2],atom[0][0]))
        handle.write("TER\n")
    handle.write("END\n")
    handle.close()
    return outfile

def find_piercing_lipids(pdb, outfile="noclash.pdb", verbose=False):
    tails_dict  = pdb_parse_TER(pdb, onlybb=False, filter_res=tails)
    sterol_PI_dict = pdb_parse_TER(pdb, onlybb=False, filter_res=sterols_PI)

    midpoints = np.zeros((len(tails_dict),50,3))+np.inf
    midpointmap = {}

    ringpoints = np.zeros((len(sterol_PI_dict),5,3))+np.inf
    ringmap    = {}

    for i,r in enumerate(tails_dict):
        bond_idx = 0
        if i not in midpointmap:
            midpointmap[i] = r
        search_keys = list(tails_dict[r].keys())
        for x,a in enumerate(search_keys):
            for b in search_keys[x+1:]:
                if np.linalg.norm(tails_dict[r][a]-tails_dict[r][b]) < 1.7: # C-C bond length shouldn't be larger then 1.59A / 1.7 just in case
                    midpoints[i,bond_idx] = np.mean([tails_dict[r][a],tails_dict[r][b]], axis=0)
                    bond_idx += 1

    for i,r in enumerate(sterol_PI_dict):
        if i not in ringmap:
            ringmap[i] = r
        for ring_idx, ring in enumerate(sterol_ring_probes):
            ring_coords = []
            for ring_atom in ring:
                for a in sterol_PI_dict[r]:
                    #Have to check PI and sterols independently, as PI has same atomnames as sterol rings
                    # structure of dict key ('PI', 2, 'P31', 63, 25)  resname, resnum, atomname, atomnum, internal_idx
                    if a[2].strip() == ring_atom and not a[0] == "PI":
                        ring_coords.append(sterol_PI_dict[r][a])
            if len(ring_coords) > 0:
                ringpoints[i,ring_idx] = np.mean(ring_coords,axis=0)
        #Now check for PI rings
        ring_coords = []
        for ring_atom in PI_ring_probe:
            for a in sterol_PI_dict[r]:
                if a[2].strip() == ring_atom and a[0] == "PI":
                    ring_coords.append(sterol_PI_dict[r][a])
        if len(ring_coords) > 0:
            ringpoints[i,4] = np.mean(ring_coords,axis=0)


    to_remove = []

    for i,sterol in enumerate(ringpoints):
        for ring_center in sterol:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pierce_dist = np.linalg.norm(midpoints-ring_center,axis=2)
            for pierce  in np.argwhere(pierce_dist < 2.5):   # H to H benzene "ring diameter" should be about 4.963A (from quick Avogadro min). Distance of the center of an aliphatic bond should be farther then this.
                to_remove.append(midpointmap[pierce[0]])
    to_remove = set(to_remove)

    if len(to_remove) > 0:
        logger.debug("The following lipids have clashing tails with sterols:")
    else:
        logger.debug("No piercing lipid found!")
    for clash in to_remove:
        tr_names  = [i[0] for i in set([clash_res[:2] for clash_res in tails_dict[clash].keys()])]
        tr_resids = [i[1] for i in set([clash_res[:2] for clash_res in tails_dict[clash].keys()])]
        logger.debug("Resnames:%s, Resids:%s" % (tr_names,tr_resids))
    return to_remove


def remove_piercing_lipids(pdb, to_remove, outfile="noclash.pdb", verbose=False):
    original_dict  = pdb_parse_TER(pdb, onlybb=False, noH=False)

    if verbose:
        logger.info("Removing clashing lipids")
    for clash in to_remove:
        del original_dict[clash]

    return pdb_write_TER(original_dict, outfile=outfile)



            
if __name__ == "__main__":
    pdb = sys.argv[1]
    print(measure_parms(pdb,23,None))
    if "-move" in sys.argv:
        vec = [float(i) for i in sys.argv[sys.argv.index("-move")+1].split(",")]
        print(measure_parms(pdb,23,move=True, move_vec=vec))
    if "-cen" in sys.argv:
        print(measure_parms(pdb,23,xy_cen=True))
