"""
PDB file I/O for pymemembed.

This module handles parsing and writing PDB files. Pure Python implementation
(not JIT-compiled) since file I/O is not performance-critical.
"""

import numpy as np
from scipy.spatial.distance import pdist


# Amino acid residue name to index mapping (0-19)
RESIDUE_MAP = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

RESIDUE_NAMES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']


class PDB:
    """
    PDB file parser and writer for MEMEMBED.

    Extracts CA/CB backbone atoms for energy calculation and writes
    transformed PDB files with membrane plane markers.
    """

    def __init__(self, pdb_file, chains=None):
        """
        Initialize PDB parser.

        Args:
            pdb_file: Path to input PDB file
            chains: List of chain IDs to process (default: all chains)
        """
        self.pdb_file = pdb_file
        self.chains = chains or []
        self.all_atoms = []
        self.backbone = {
            'x': [],
            'y': [],
            'z': [],
            'res': []
        }
        self.max_x = 0.0
        self.max_y = 0.0
        self.max_z = 0.0
        self._parse()
        self._apply_origin_shift()

    def _parse(self):
        """
        Parse PDB file and extract CA/CB atoms.

        For each residue, extracts:
        - CA atom for glycine (no CB)
        - CB atom for all other residues

        Populates self.all_atoms (all ATOM lines) and self.backbone
        (NumPy arrays for x, y, z coordinates and residue type indices).
        """
        with open(self.pdb_file, 'r') as f:
            for line in f:
                # Store all ATOM and TER lines for output
                if line.startswith('ATOM') or line.startswith('TER'):
                    self.all_atoms.append(line)

                # Process only ATOM lines
                if not line.startswith('ATOM'):
                    continue

                # Check chain filter
                chain_id = line[21:22]
                if self.chains and chain_id not in self.chains:
                    continue

                # Extract atom and residue names
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()

                # Skip non-standard residues
                if res_name not in RESIDUE_MAP:
                    continue

                # Extract CA (all residues) or CB (non-glycine)
                extract_atom = False
                if atom_name == 'CA' and res_name == 'GLY':
                    extract_atom = True
                elif atom_name == 'CB' and res_name != 'GLY':
                    extract_atom = True

                if extract_atom:
                    # Parse coordinates
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    res_idx = RESIDUE_MAP[res_name]

                    # Store in backbone
                    self.backbone['x'].append(x)
                    self.backbone['y'].append(y)
                    self.backbone['z'].append(z)
                    self.backbone['res'].append(res_idx)

        # Convert to NumPy arrays for Numba compatibility
        for key in ['x', 'y', 'z', 'res']:
            self.backbone[key] = np.array(self.backbone[key], dtype=np.float64 if key != 'res' else np.int32)

        if len(self.backbone['x']) == 0:
            raise ValueError(f"No CA/CB atoms found in {self.pdb_file}")

    def _apply_origin_shift(self):
        """
        Apply origin shift to backbone coordinates (matches C++ origin_shift()).

        Finds max X, Y, Z from backbone and subtracts these values to center
        the protein. This is called before optimization in C++ MEMEMBED.
        """
        if len(self.backbone['x']) == 0:
            return

        # Find max values (matching C++ pdb.cpp:736-748)
        self.max_x = float(np.max(self.backbone['x']))
        self.max_y = float(np.max(self.backbone['y']))
        self.max_z = float(np.max(self.backbone['z']))

        # Shift backbone coords by subtracting max values (matching C++ pdb.cpp:752-756)
        self.backbone['x'] = self.backbone['x'] - self.max_x
        self.backbone['y'] = self.backbone['y'] - self.max_y
        self.backbone['z'] = self.backbone['z'] - self.max_z

        # Compute max pairwise distance among backbone atoms (matching C++ get_maxcdist())
        coords = np.column_stack([self.backbone['x'], self.backbone['y'], self.backbone['z']])
        self.max_c_dist = float(np.max(pdist(coords)))

    def get_backbone_arrays(self):
        """
        Get backbone arrays for energy calculation.

        Returns:
            tuple: (x, y, z, res) as NumPy arrays
        """
        return (
            self.backbone['x'],
            self.backbone['y'],
            self.backbone['z'],
            self.backbone['res']
        )

    def write_oriented_pdb(self, output_file, xrt, yrt, z_trans, energy,
                          polar_headgroups=False, extra_shift=0.0, cyto_shift=0.0, flip=False):
        """
        Write transformed PDB with membrane plane markers.

        Args:
            output_file: Output PDB file path
            xrt: X-axis rotation (radians)
            yrt: Y-axis rotation (radians)
            z_trans: Z-axis translation (Angstroms)
            energy: Membrane energy value
            polar_headgroups: If True, add ±24 Å markers for polar headgroups
            extra_shift: Extra shift for extracellular side (Angstroms)
            cyto_shift: Shift for cytoplasmic side (Angstroms)
            flip: If True, apply 180° X-rotation after other transformations
        """
        with open(output_file, 'w') as f:
            # Write header with optimization results
            f.write(f"HEADER MEMBRANE_ENERGY {energy:.10f}\n")
            f.write(f"HEADER X_ROTATION      {np.degrees(xrt):.6f}\n")
            f.write(f"HEADER Y_ROTATION      {np.degrees(yrt):.6f}\n")
            f.write(f"HEADER Z_TRANSLATION   {z_trans:.6f}\n")

            # Track min/max X/Y for membrane markers
            max_x = -1e6
            max_y = -1e6
            min_x = 1e6
            min_y = 1e6

            # Transform and write all atoms
            for line in self.all_atoms:
                if line.startswith('TER'):
                    f.write(line)
                    continue

                # Parse coordinates
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                # Apply transformations
                x_new, y_new, z_new = self._transform_point(x, y, z, xrt, yrt, z_trans, flip)

                # Update bounds
                max_x = max(max_x, x_new)
                max_y = max(max_y, y_new)
                min_x = min(min_x, x_new)
                min_y = min(min_y, y_new)

                # Update coordinates in line
                new_line = line[:30] + f"{x_new:8.3f}{y_new:8.3f}{z_new:8.3f}" + line[54:]
                f.write(new_line)

            # Add membrane dummy atoms
            self._write_membrane_markers(f, min_x, max_x, min_y, max_y,
                                         polar_headgroups, extra_shift, cyto_shift)

#        print(f"Written {output_file}")

    def _transform_point(self, x, y, z, xrt, yrt, z_trans, flip=False):
        """
        Apply rotation and translation to a single point.

        Matches C++ transform_atom() in pdb.cpp:511-567

        Args:
            x, y, z: Original coordinates
            xrt: X-axis rotation (radians)
            yrt: Y-axis rotation (radians)
            z_trans: Z-axis translation
            flip: If True, apply 180° X-rotation after other transformations

        Returns:
            tuple: (x_new, y_new, z_new)
        """
        # Apply origin shift first (matching C++ pdb.cpp:516-518)
        x -= self.max_x
        y -= self.max_y
        z -= self.max_z

        # Pre-compute trig values
        cx = np.cos(xrt)
        sx = np.sin(xrt)
        cy = np.cos(yrt)
        sy = np.sin(yrt)

        # X-axis rotation (matching C++ pdb.cpp:542-546)
        yp = y * cx - z * sx
        zp = y * sx + z * cx
        y = yp
        z = zp

        # Y-axis rotation (matching C++ pdb.cpp:548-552)
        zp = z * cy - x * sy
        xp = z * sy + x * cy
        z = zp
        x = xp

        # Z-axis translation (matching C++ pdb.cpp:554)
        z += z_trans

        # Apply flip if requested (matching C++ pdb.cpp:556-561)
        if flip:
            # X-axis rotation by 180° (π radians)
            # cos(π) = -1, sin(π) = 0
            y = -y
            z = -z

        return x, y, z

    def _write_membrane_markers(self, f, min_x, max_x, min_y, max_y,
                                polar_headgroups, extra_shift, cyto_shift):
        """
        Write membrane dummy atoms (HETATM records).

        Adds grid of O atoms at extracellular side and N atoms at cytoplasmic side.

        Args:
            f: Output file handle
            min_x, max_x, min_y, max_y: Bounding box
            polar_headgroups: If True, add ±24 Å markers
            extra_shift: Extra shift for extracellular side
            cyto_shift: Shift for cytoplasmic side
        """
        # Expand bounds by 8 Å
        max_x += 8
        max_y += 8
        min_x -= 8
        min_y -= 8

        count = 1000
        grid_spacing = 2.0  # 2 Å grid

        # Generate grid points
        for i in range(int((max_x - min_x) / grid_spacing) + 1):
            for j in range(int((max_y - min_y) / grid_spacing) + 1):
                x = min_x + (i * grid_spacing)
                y = min_y + (j * grid_spacing)

                # Hydrophobic core markers (±15 Å)
                # Extracellular side (O atoms)
                z = 15.0 + extra_shift
                f.write(f"HETATM {count:4d}  O   DUM  {count:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
                count += 1
                if count > 9999:
                    count = 1000

                # Cytoplasmic side (N atoms)
                z = -15.0 - cyto_shift
                f.write(f"HETATM {count:4d}  N   DUM  {count:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
                count += 1
                if count > 9999:
                    count = 1000

                # Polar headgroup markers (±24 Å)
                if polar_headgroups:
                    z = 24.0 + extra_shift
                    f.write(f"HETATM {count:4d}  O   DUM  {count:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
                    count += 1
                    if count > 9999:
                        count = 1000

                    z = -24.0 - cyto_shift
                    f.write(f"HETATM {count:4d}  N   DUM  {count:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
                    count += 1
                    if count > 9999:
                        count = 1000
