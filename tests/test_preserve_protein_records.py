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


def test_default_cli_preserves_protein_records_and_skips_protonation(tmp_path: Path):
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text(PDB_TEXT)

    args = parser.parse_args(
        [
            "--solvate",
            "--notrun",
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


def test_rewrite_and_protonate_flags_restore_legacy_behavior(tmp_path: Path):
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text(PDB_TEXT)

    args = parser.parse_args(
        [
            "--solvate",
            "--notrun",
            "--noxy_cen",
            "--rewrite-protein-records",
            "--protonate",
            "-p",
            str(pdb_path),
            "--outdir",
            str(tmp_path),
        ]
    )

    assert args.preserve_protein_records is False
    assert args.notprotonate is True


def test_mempro_windows_retry_uses_single_cpu_without_mutating_environment(monkeypatch):
    calls = []

    class Result:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(cmd, env=None):
        calls.append((cmd, env))
        return Result(1 if len(calls) == 1 else 0)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(MODULE.os, "name", "nt")
    monkeypatch.setenv("NUM_CPU", "20")

    pmg = object.__new__(PACKMOLMemgen)
    result = pmg._run_mempro_command(["mempro", "-f", "input.pdb"])

    assert result.returncode == 0
    assert len(calls) == 2
    assert calls[0][1] is None
    assert calls[1][1]["NUM_CPU"] == "1"
    assert MODULE.os.environ["NUM_CPU"] == "20"


def test_mempro_default_membrane_thickness_is_double_leaflet(tmp_path: Path, monkeypatch):
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text(PDB_TEXT)
    captured = {}

    class Result:
        returncode = 0

    def fake_run(cmd, verbose=False):
        captured["cmd"] = cmd
        oriented = tmp_path / "_tmp_input_in_MEMPRO" / "Rank_1" / "oriented_rank_1.pdb"
        oriented.parent.mkdir(parents=True, exist_ok=True)
        oriented.write_text(PDB_TEXT)
        return Result()

    def fake_write(source_pdb, oriented_pdb, output_pdb, preserve_records=False):
        Path(output_pdb).write_text(PDB_TEXT)

    pmg = object.__new__(PACKMOLMemgen)
    pmg.outdir = str(tmp_path)
    pmg.keep_mempro = False
    pmg.created = []
    pmg.created_mempro = []
    pmg.martini = False
    pmg.build_system = None
    pmg.build_arguments = None
    pmg.mempro = "mempro"
    pmg.mempro_grid = 36
    pmg.mempro_iters = 150
    pmg.mempro_rank = "auto"
    pmg.mempro_args = None
    pmg.leaflet = 23.0
    pmg.mempro_curvature = False
    pmg._used_tools = set()
    pmg._run_mempro_command = fake_run
    pmg._write_mempro_aligned_pdb = fake_write
    pmg._apply_mempro_curvature = lambda info_path: None

    output = pmg.mempro_align(str(pdb_path), overwrite=True)

    mt_index = captured["cmd"].index("-mt")
    assert captured["cmd"][mt_index + 1] == "46.0"
    assert pmg.leaflet == 23.0
    assert output == str(tmp_path / "input_in_MEMPRO.pdb")
