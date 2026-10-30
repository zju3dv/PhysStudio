"""Download, convert, and prepare a public SuperSplat lava scene."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from physstudio.paths import ROOT, data_path
from .fetch_scene import download
from .prepare_scene import prepare


SOURCES = {
    "zohar": ("7aa4d3d4", 0),
    "lands_end": ("9e2051b8", 0),
    "calico": ("5d517144", 1),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SOURCES, required=True)
    parser.add_argument("--converter", type=Path)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    root = data_path("lava_gs", args.scene)
    ply = root / "point_cloud/iteration_30000/point_cloud.ply"
    scene_hash, lod = SOURCES[args.scene]
    if not ply.is_file():
        converter = args.converter or (
            ROOT / ".cache/lava_tools/node_modules/.bin/splat-transform"
        )
        if not args.download_only and not converter.is_file():
            installed = shutil.which("splat-transform")
            if installed and args.converter is None:
                converter = Path(installed)
            else:
                parser.error(
                    "Install the pinned converter listed in lava_gs/README.md or pass --converter."
                )
        download(scene_hash, lod, root)
        if args.download_only:
            prepare(args.scene)
            return
        ply.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(converter.resolve()),
                "--select-lod",
                str(lod),
                str(root / "lod-meta.json"),
                str(ply),
            ],
            check=True,
            cwd=ROOT,
        )
    if args.scene == "calico":
        subprocess.run(
            [sys.executable, "-m", "lava_gs.pad_harmonics", str(ply)],
            check=True,
            cwd=ROOT,
        )
    prepare(args.scene)
    print("Ready:", ply)


if __name__ == "__main__":
    main()
