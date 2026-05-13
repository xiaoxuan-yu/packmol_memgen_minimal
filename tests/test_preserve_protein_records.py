import importlib.util
import types
from pathlib import Path

import pytest


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

TRANSFORM_SOURCE_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 10.00           C
ATOM      3  C   ALA A   1       1.200   0.800   0.100  1.00 10.00           C
ATOM      4  N   GLY A   2       1.900   1.400   0.200  1.00 10.00           N
ATOM      5  CA  GLY A   2       2.500   2.000   0.500  1.00 10.00           C
ATOM      6  C   GLY A   2       3.400   2.100   1.100  1.00 10.00           C
ATOM      7  N   SER A   3       0.100   2.400   1.700  1.00 10.00           N
ATOM      8  CA  SER A   3       0.500   3.000   2.000  1.00 10.00           C
ATOM      9  C   SER A   3       1.100   3.700   1.400  1.00 10.00           C
HETATM   10  C1  LIG A 101       2.200   3.800   2.700  1.00 20.00           C
END
"""

TRANSFORM_ORIENTED_PDB = """\
ATOM      1  N   ALA A   1      10.000  -2.000   3.000  1.00 10.00           N
ATOM      2  CA  ALA A   1      11.000  -2.000   3.000  1.00 10.00           C
ATOM      3  C   ALA A   1      11.200  -1.200   3.100  1.00 10.00           C
ATOM      4  N   GLY A   2      11.900  -0.600   3.200  1.00 10.00           N
ATOM      5  CA  GLY A   2      12.500   0.000   3.500  1.00 10.00           C
ATOM      6  C   GLY A   2      13.400   0.100   4.100  1.00 10.00           C
ATOM      7  N   SER A   3      10.100   0.400   4.700  1.00 10.00           N
ATOM      8  CA  SER A   3      10.500   1.000   5.000  1.00 10.00           C
ATOM      9  C   SER A   3      11.100   1.700   4.400  1.00 10.00           C
HETATM 1010  O   DUM 1010      10.000  10.000  15.000  1.00  0.00           O
END
"""

DOUBLE_SPAN_SOURCE_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000  30.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       1.000   0.000  30.000  1.00 10.00           C
ATOM      3  C   ALA A   1       1.500   0.500  30.000  1.00 10.00           C
ATOM      4  N   GLY A   2       0.000   1.500  34.000  1.00 10.00           N
ATOM      5  CA  GLY A   2       1.000   1.500  34.000  1.00 10.00           C
ATOM      6  C   GLY A   2       1.500   2.000  34.000  1.00 10.00           C
ATOM      7  N   SER A   3       0.000   3.000  38.000  1.00 10.00           N
ATOM      8  CA  SER A   3       1.000   3.000  38.000  1.00 10.00           C
ATOM      9  C   SER A   3       1.500   3.500  38.000  1.00 10.00           C
HETATM   10  C1  LIG A 101       2.200   3.800  40.700  1.00 20.00           C
END
"""

DOUBLE_SPAN_STAGE1_RAW = """\
HEADER MEMBRANE_ENERGY -1.0000000000
ATOM      1  N   ALA A   1       0.000   0.000  30.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       1.000   0.000  30.000  1.00 10.00           C
ATOM      3  C   ALA A   1       1.500   0.500  30.000  1.00 10.00           C
ATOM      4  N   GLY A   2       0.000   1.500  34.000  1.00 10.00           N
ATOM      5  CA  GLY A   2       1.000   1.500  34.000  1.00 10.00           C
ATOM      6  C   GLY A   2       1.500   2.000  34.000  1.00 10.00           C
ATOM      7  N   SER A   3       0.000   3.000  38.000  1.00 10.00           N
ATOM      8  CA  SER A   3       1.000   3.000  38.000  1.00 10.00           C
ATOM      9  C   SER A   3       1.500   3.500  38.000  1.00 10.00           C
HETATM 1000  O   DUM 1000      -2.000  -2.000  15.000  1.00  0.00           O
HETATM 1001  O   DUM 1001       2.000   2.000  15.000  1.00  0.00           O
HETATM 1002  N   DUM 1002      -2.000  -2.000 -15.000  1.00  0.00           N
HETATM 1003  N   DUM 1003       2.000   2.000 -15.000  1.00  0.00           N
END
"""

DOUBLE_SPAN_STAGE2_RAW = """\
HEADER MEMBRANE_ENERGY -2.0000000000
ATOM      1  N   ALA A   1       5.000   0.000  12.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       6.000   0.000  12.000  1.00 10.00           C
ATOM      3  C   ALA A   1       6.500   0.500  12.000  1.00 10.00           C
ATOM      4  N   GLY A   2       5.000   1.500  14.000  1.00 10.00           N
ATOM      5  CA  GLY A   2       6.000   1.500  14.000  1.00 10.00           C
ATOM      6  C   GLY A   2       6.500   2.000  14.000  1.00 10.00           C
ATOM      7  N   SER A   3       5.000   3.000  16.000  1.00 10.00           N
ATOM      8  CA  SER A   3       6.000   3.000  16.000  1.00 10.00           C
ATOM      9  C   SER A   3       6.500   3.500  16.000  1.00 10.00           C
ATOM     10  MEM MEM X   1       5.500   1.500  10.000  1.00  0.00           M
HETATM 1000  O   DUM 1000      -2.000  -2.000  15.000  1.00  0.00           O
HETATM 1001  O   DUM 1001       2.000   2.000  15.000  1.00  0.00           O
HETATM 1002  N   DUM 1002      -2.000  -2.000   1.000  1.00  0.00           N
HETATM 1003  N   DUM 1003       2.000   2.000   1.000  1.00  0.00           N
END
"""


def build_pmg(argv):
    args = parser.parse_args(argv)
    return PACKMOLMemgen(args)


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


def test_default_orientation_backend_is_mempro():
    args = parser.parse_args([])

    assert args.orientation_backend == "mempro"
    assert args.orientation_backend_explicit is False


def test_pymemembed_backend_parses():
    args = parser.parse_args(["--orientation-backend", "pymemembed"])

    assert args.orientation_backend == "pymemembed"
    assert args.orientation_backend_explicit is True


def test_preoriented_conflicts_with_explicit_backend():
    pmg = build_pmg(["--preoriented", "--orientation-backend", "mempro"])

    with pytest.raises(SystemExit):
        pmg._normalize_orientation_backend()


def test_pymemembed_rejects_mempro_only_flags():
    pmg = build_pmg(["--orientation-backend", "pymemembed", "--mempro_curvature"])
    pmg._normalize_orientation_backend()

    with pytest.raises(SystemExit):
        pmg._validate_orientation_backend_options()


def test_pymemembed_allows_double_span():
    pmg = build_pmg(["--orientation-backend", "pymemembed", "--double_span"])
    pmg._normalize_orientation_backend()
    pmg._validate_orientation_backend_options()


@pytest.mark.parametrize(
    ("backend", "expected_output", "called"),
    [
        ("mempro", "mempro_out.pdb", "mempro"),
        ("pymemembed", "pymemembed_out.pdb", "pymemembed"),
        ("preoriented", "input.pdb", None),
    ],
)
def test_orientation_dispatch_uses_selected_backend(monkeypatch, backend, expected_output, called):
    pmg = build_pmg([])
    pmg.orientation_backend = backend
    pmg.double_span = False
    pmg.keepligs = False
    pmg.martini = False
    seen = []

    def fake_mempro(*args, **kwargs):
        seen.append("mempro")
        return "mempro_out.pdb"

    def fake_pymemembed(*args, **kwargs):
        seen.append("pymemembed")
        return "pymemembed_out.pdb"

    monkeypatch.setattr(pmg, "mempro_align", fake_mempro)
    monkeypatch.setattr(pmg, "pymemembed_align", fake_pymemembed)

    output = pmg._orient_protein("input.pdb", verbose=False, overwrite=False, n_ter="in", preserve_records=True)

    assert output == expected_output
    assert seen == ([] if called is None else [called])


def test_orientation_dispatch_passes_double_span_to_pymemembed(monkeypatch):
    pmg = build_pmg(["--orientation-backend", "pymemembed", "--double_span"])
    pmg.orientation_backend = "pymemembed"
    pmg.double_span = True
    seen = {}

    def fake_pymemembed(*args, **kwargs):
        seen["double_span"] = kwargs["double_span"]
        return ("pymemembed_out_double.pdb", 18.0)

    monkeypatch.setattr(pmg, "pymemembed_align", fake_pymemembed)

    output = pmg._orient_protein("input.pdb", verbose=False, overwrite=False, n_ter="in", preserve_records=True)

    assert output == ("pymemembed_out_double.pdb", 18.0)
    assert seen["double_span"] is True


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


def test_pymemembed_missing_numba_gives_clear_error(monkeypatch):
    pmg = build_pmg([])

    def fake_import_module(name):
        exc = ModuleNotFoundError("No module named 'numba'")
        exc.name = "numba"
        raise exc

    monkeypatch.setattr(MODULE.importlib, "import_module", fake_import_module)

    with pytest.raises(SystemExit):
        pmg._require_pymemembed_module()


def test_pymemembed_align_writes_output_and_uses_cache(tmp_path: Path, monkeypatch):
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text(PDB_TEXT)

    def fake_memembed_align(**kwargs):
            Path(kwargs["output_file"]).write_text(PDB_TEXT)
            return {
                "energy": 1.0,
                "x_rotation": 0.0,
                "y_rotation": 0.0,
                "z_translation": 0.0,
                "n_calls": 10,
                "output_file": kwargs["output_file"],
                "method": kwargs["method"],
            }
    fake_package = types.SimpleNamespace(__name__="fakepkg", memembed_align=fake_memembed_align)

    class FakeWrapper:
        @staticmethod
        def run_ga_multi(**kwargs):
            Path(kwargs["output_file"]).write_text(PDB_TEXT)
            return {
                "energy": 1.0,
                "x_rotation": 0.0,
                "y_rotation": 0.0,
                "z_translation": 0.0,
                "n_calls": 10,
                "output_file": kwargs["output_file"],
                "method": "ga_multi",
            }

        @staticmethod
        def write_memembed_log(log_file, result, opt, barrel, n_ter):
            Path(log_file).write_text("ok\n")

    def fake_import_module(name):
        if name == "fakepkg.memembed_wrapper":
            return FakeWrapper
        raise AssertionError(name)

    pmg = build_pmg(["--orientation-backend", "pymemembed", "--outdir", str(tmp_path)])
    pmg.keep_mempro = False
    pmg.created = []
    pmg.created_mempro = []
    pmg.pymemembed_force_span = False
    pmg.pymemembed_barrel = False
    pmg.pymemembed_threads = 2
    pmg.pymemembed_runs = 5
    pmg.pymemembed_search = 0
    pmg.pymemembed_max_calls = 123
    pmg.pymemembed_polar_headgroups = False
    monkeypatch.setattr(pmg, "_require_pymemembed_module", lambda: fake_package)
    monkeypatch.setattr(MODULE.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        pmg,
        "_write_aligned_pdb_from_reference",
        lambda source, oriented, output, preserve_records=False, tool_name="orientation": Path(output).write_text("OUT\n"),
    )

    output = pmg.pymemembed_align(str(pdb_path), overwrite=True, n_ter="in")

    assert output == str(tmp_path / "input_in_PYMEMEMBED.pdb")
    assert Path(output).read_text() == "OUT\n"

    monkeypatch.setattr(
        pmg,
        "_require_pymemembed_module",
        lambda: (_ for _ in ()).throw(AssertionError("cache path should not import pymemembed")),
    )
    cached = pmg.pymemembed_align(str(pdb_path), overwrite=False, n_ter="in")
    assert cached == output


def test_pymemembed_double_span_align_runs_second_stage_and_uses_cache(tmp_path: Path, monkeypatch):
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text(DOUBLE_SPAN_SOURCE_PDB)
    calls = []

    def fake_memembed_align(**kwargs):
        calls.append(kwargs["pdb_file"])
        output_path = Path(kwargs["output_file"])
        if output_path.name == "oriented_embed_raw.pdb":
            output_path.write_text(DOUBLE_SPAN_STAGE1_RAW)
        elif output_path.name == "double_span_stage2_raw.pdb":
            output_path.write_text(DOUBLE_SPAN_STAGE2_RAW)
        else:
            raise AssertionError(output_path)
        return {
            "energy": 1.0,
            "x_rotation": 0.0,
            "y_rotation": 0.0,
            "z_translation": 0.0,
            "n_calls": 10,
            "output_file": kwargs["output_file"],
            "method": kwargs["method"],
        }

    fake_package = types.SimpleNamespace(__name__="fakepkg", memembed_align=fake_memembed_align)

    class FakeWrapper:
        @staticmethod
        def run_ga_multi(**kwargs):
            raise AssertionError("ga_multi should not be used in this test")

        @staticmethod
        def write_memembed_log(log_file, result, opt, barrel, n_ter):
            Path(log_file).write_text("ok\n")

    def fake_import_module(name):
        if name == "fakepkg.memembed_wrapper":
            return FakeWrapper
        raise AssertionError(name)

    pmg = build_pmg(["--orientation-backend", "pymemembed", "--double_span", "--outdir", str(tmp_path)])
    pmg.keep_mempro = False
    pmg.created = []
    pmg.created_mempro = []
    pmg.pymemembed_force_span = False
    pmg.pymemembed_barrel = False
    pmg.pymemembed_threads = 2
    pmg.pymemembed_runs = 5
    pmg.pymemembed_search = 0
    pmg.pymemembed_max_calls = 123
    pmg.pymemembed_polar_headgroups = False
    monkeypatch.setattr(pmg, "_require_pymemembed_module", lambda: fake_package)
    monkeypatch.setattr(MODULE.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(MODULE, "superimpose_pdb", lambda pdb1, pdb2: pdb2)
    monkeypatch.setattr(
        pmg,
        "_write_aligned_pdb_from_reference",
        lambda source, oriented, output, preserve_records=False, tool_name="orientation": Path(output).write_text("OUT\n"),
    )

    output, z_offset = pmg.pymemembed_align(str(pdb_path), overwrite=True, n_ter="in", double_span=True, keepligs=True)

    assert output == str(tmp_path / "input_in_PYMEMEMBED_double.pdb")
    assert z_offset == pytest.approx(18.0)
    assert Path(output).read_text() == "OUT\n"
    stage_dir = tmp_path / "_tmp_input_in_PYMEMEMBED"
    assert (stage_dir / "double_span_stage1_input.pdb").exists()
    assert (stage_dir / "double_span_stage2_raw.pdb").exists()
    assert (stage_dir / "double_span_presuper.pdb").exists()
    assert (stage_dir / "double_span_aligned_raw.pdb").exists()
    assert calls == [str(pdb_path), str(stage_dir / "double_span_stage1_input.pdb")]

    monkeypatch.setattr(
        pmg,
        "_require_pymemembed_module",
        lambda: (_ for _ in ()).throw(AssertionError("cache path should not import pymemembed")),
    )
    cached_output, cached_z_offset = pmg.pymemembed_align(str(pdb_path), overwrite=False, n_ter="in", double_span=True)
    assert cached_output == output
    assert cached_z_offset == pytest.approx(18.0)


def test_pymemembed_double_span_fails_without_first_stage_dum_markers(tmp_path: Path, monkeypatch):
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text(DOUBLE_SPAN_SOURCE_PDB)

    def fake_memembed_align(**kwargs):
        Path(kwargs["output_file"]).write_text(DOUBLE_SPAN_SOURCE_PDB)
        return {
            "energy": 1.0,
            "x_rotation": 0.0,
            "y_rotation": 0.0,
            "z_translation": 0.0,
            "n_calls": 10,
            "output_file": kwargs["output_file"],
            "method": kwargs["method"],
        }

    fake_package = types.SimpleNamespace(__name__="fakepkg", memembed_align=fake_memembed_align)

    class FakeWrapper:
        @staticmethod
        def run_ga_multi(**kwargs):
            raise AssertionError("ga_multi should not be used in this test")

        @staticmethod
        def write_memembed_log(log_file, result, opt, barrel, n_ter):
            Path(log_file).write_text("ok\n")

    pmg = build_pmg(["--orientation-backend", "pymemembed", "--double_span", "--outdir", str(tmp_path)])
    pmg.keep_mempro = False
    pmg.created = []
    pmg.created_mempro = []
    monkeypatch.setattr(pmg, "_require_pymemembed_module", lambda: fake_package)
    monkeypatch.setattr(MODULE.importlib, "import_module", lambda name: FakeWrapper if name == "fakepkg.memembed_wrapper" else (_ for _ in ()).throw(AssertionError(name)))

    with pytest.raises(SystemExit):
        pmg.pymemembed_align(str(pdb_path), overwrite=True, n_ter="in", double_span=True)


def test_pymemembed_double_span_fails_without_second_stage_dum_markers(tmp_path: Path, monkeypatch):
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text(DOUBLE_SPAN_SOURCE_PDB)

    def fake_memembed_align(**kwargs):
        output_path = Path(kwargs["output_file"])
        if output_path.name == "oriented_embed_raw.pdb":
            output_path.write_text(DOUBLE_SPAN_STAGE1_RAW)
        else:
            output_path.write_text(DOUBLE_SPAN_STAGE2_RAW.replace("HETATM 1000  O   DUM 1000      -2.000  -2.000  15.000  1.00  0.00           O\n", "").replace("HETATM 1001  O   DUM 1001       2.000   2.000  15.000  1.00  0.00           O\n", ""))
        return {
            "energy": 1.0,
            "x_rotation": 0.0,
            "y_rotation": 0.0,
            "z_translation": 0.0,
            "n_calls": 10,
            "output_file": kwargs["output_file"],
            "method": kwargs["method"],
        }

    fake_package = types.SimpleNamespace(__name__="fakepkg", memembed_align=fake_memembed_align)

    class FakeWrapper:
        @staticmethod
        def run_ga_multi(**kwargs):
            raise AssertionError("ga_multi should not be used in this test")

        @staticmethod
        def write_memembed_log(log_file, result, opt, barrel, n_ter):
            Path(log_file).write_text("ok\n")

    pmg = build_pmg(["--orientation-backend", "pymemembed", "--double_span", "--outdir", str(tmp_path)])
    pmg.keep_mempro = False
    pmg.created = []
    pmg.created_mempro = []
    monkeypatch.setattr(pmg, "_require_pymemembed_module", lambda: fake_package)
    monkeypatch.setattr(MODULE.importlib, "import_module", lambda name: FakeWrapper if name == "fakepkg.memembed_wrapper" else (_ for _ in ()).throw(AssertionError(name)))
    monkeypatch.setattr(MODULE, "superimpose_pdb", lambda pdb1, pdb2: pdb2)
    monkeypatch.setattr(
        pmg,
        "_write_aligned_pdb_from_reference",
        lambda source, oriented, output, preserve_records=False, tool_name="orientation": Path(output).write_text("OUT\n"),
    )

    with pytest.raises(SystemExit):
        pmg.pymemembed_align(str(pdb_path), overwrite=True, n_ter="in", double_span=True)


def test_generic_alignment_writer_preserves_records_with_transformed_coordinates(tmp_path: Path):
    source_path = tmp_path / "source.pdb"
    oriented_path = tmp_path / "oriented.pdb"
    output_path = tmp_path / "output.pdb"
    source_path.write_text(TRANSFORM_SOURCE_PDB)
    oriented_path.write_text(TRANSFORM_ORIENTED_PDB)

    pmg = object.__new__(PACKMOLMemgen)
    pmg._write_aligned_pdb_from_reference(
        str(source_path),
        str(oriented_path),
        str(output_path),
        preserve_records=True,
        tool_name="pymemembed",
    )

    output = output_path.read_text()
    assert "HETATM   10  C1  LIG A 101" in output
    assert "      13.496" in output
    assert " 1010 " not in output
    assert "       2.200   3.800   2.700" not in output
