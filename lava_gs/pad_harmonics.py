"""Add zero-valued unused SH coefficients for the shared degree-3 scene loader."""

import argparse
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ply", type=Path)
    args = p.parse_args()
    source = PlyData.read(args.ply)
    vertex = source["vertex"].data
    names = list(vertex.dtype.names)
    missing = ["f_rest_" + str(i) for i in range(45) if "f_rest_" + str(i) not in names]
    if not missing:
        print("All SH fields already present")
        return
    assert len(source.elements) == 1
    assert all(
        vertex.dtype.fields[n][0].kind == "f"
        and vertex.dtype.fields[n][0].itemsize == 4
        for n in names
    )
    dtype = np.dtype([(n, "<f4") for n in names + missing])
    target = args.ply.with_suffix(".working.ply")
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex "
        + str(len(vertex))
        + "\n"
    )
    header += (
        "".join("property float " + n + "\n" for n in names + missing) + "end_header\n"
    )
    with target.open("wb") as stream:
        stream.write(header.encode("ascii"))
        for start in range(0, len(vertex), 100000):
            block = vertex[start : start + 100000]
            extended = np.zeros(len(block), dtype=dtype)
            for name in names:
                extended[name] = block[name]
            stream.write(extended.tobytes())
    native = args.ply.with_suffix(".native.ply")
    if native.exists():
        raise FileExistsError(native)
    args.ply.rename(native)
    target.rename(args.ply)
    args.ply.with_suffix(".padding.json").write_text(
        json.dumps(
            dict(
                native_ply=str(native),
                output_ply=str(args.ply),
                vertices=len(vertex),
                added_zero_columns=missing,
                appearance_change=False,
            ),
            indent=2,
        )
    )
    print("Padded", len(vertex), "vertices with", len(missing), "zero SH columns")


if __name__ == "__main__":
    main()
