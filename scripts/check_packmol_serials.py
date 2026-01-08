#!/usr/bin/env python3
import argparse
from pathlib import Path

from packmol_memgen.lib import utils


def iter_atom_lines(lines):
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            yield line


def parse_packmol_serials(lines, packmol_hex_after):
    atom_index = 0
    hex_switch = False
    atomlimit = False
    atomnum = 0
    for line in iter_atom_lines(lines):
        atom_index += 1
        field = line[6:11].strip()
        if field == "*****" and not atomlimit:
            atomlimit = True
        if not field.isnumeric():
            hex_switch = True
        if atomlimit:
            atomnum += 1
        else:
            if packmol_hex_after is not None and atom_index > packmol_hex_after:
                atomnum = int(field, 16)
            else:
                atomnum = int(field, 16) if hex_switch else int(field)
        yield atom_index, field, atomnum


def main():
    parser = argparse.ArgumentParser(
        description="Check packmol serial parsing and hybrid-36 writing."
    )
    parser.add_argument("input_pdb", help="Input PDB from packmol")
    parser.add_argument(
        "-o",
        "--output",
        default="out_hybrid36.pdb",
        help="Output PDB with hybrid-36 serials",
    )
    parser.add_argument(
        "--packmol-hex-after",
        type=int,
        default=99999,
        help="Atom index after which packmol switches to hex",
    )
    args = parser.parse_args()

    input_path = Path(args.input_pdb)
    output_path = Path(args.output)
    lines = input_path.read_text(encoding="utf-8").splitlines()

    parsed = list(parse_packmol_serials(lines, args.packmol_hex_after))
    atom_count = len(parsed)
    print(f"Atoms: {atom_count}")

    if atom_count == 0:
        print("No ATOM/HETATM records found.")
        return 0

    if atom_count <= args.packmol_hex_after:
        print(
            f"Atom count <= {args.packmol_hex_after}; "
            "cannot validate packmol hex switch region."
        )
    else:
        for idx in (args.packmol_hex_after - 1, args.packmol_hex_after, args.packmol_hex_after + 1):
            if idx <= 0 or idx > atom_count:
                continue
            atom_index, raw_field, atomnum = parsed[idx - 1]
            encoded = utils.hy36encode(5, atomnum)
            print(
                f"Index {atom_index}: input='{raw_field}' parsed={atomnum} "
                f"-> hybrid-36='{encoded}'"
            )

    ca_cb = utils.pdb_parse_TER(
        str(input_path),
        onlybb=False,
        noH=False,
        packmol_hex_after=args.packmol_hex_after,
    )
    utils.pdb_write_TER(ca_cb, outfile=str(output_path))
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
