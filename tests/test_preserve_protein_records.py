import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "packmol_memgen" / "main.py"
SPEC = importlib.util.spec_from_file_location("packmol_memgen_main", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

PACKMOLMemgen = MODULE.PACKMOLMemgen
parser = MODULE.parser


PDB_TEXT = """\
ATOM      1  N   GLY B  10      11.104  13.207   9.599  1.00 20.00           N
ATOM      2  CA  GLY B  10      12.560  13.207   9.599  1.00 20.00           C
ATOM      3  C   GLY B  10      13.000  14.500  10.000  1.00 20.00           C
ATOM      4  OT2 GLY B  10      14.200  14.700  10.200  1.00 20.00           O
END
"""


def test_preserve_protein_records_cli_keeps_protein_records(tmp_path: Path):
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text(PDB_TEXT)

    args = parser.parse_args(
        [
            "--solvate",
            "--notrun",
            "--notprotonate",
            "--preserve-protein-records",
            "--noxy_cen",
            "-p",
            str(pdb_path),
            "--outdir",
            str(tmp_path),
        ]
    )

    pmg = PACKMOLMemgen(args)
    pmg.prepare()

    prot_path = tmp_path / "PROT0.pdb"
    assert prot_path.exists()

    output = prot_path.read_text()
    assert "TER\n" not in output
    assert " OT2 " in output
    assert " OXT " not in output
    assert "GLY B  10" in output
    assert "structure " + str(prot_path) in pmg.contents
