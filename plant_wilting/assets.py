"""Prepare released DiffWind plants without changing the original assets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from enlivening_gs.assets import read_ply
from physstudio.paths import data_path


SCENES = ("potted_plant", "ficus", "alocasia")
ROOT = data_path("plant_wilting", "prepared")


def prepare(scene, output_root=ROOT):
    source_dir = data_path("plant_wilting", "source", scene)
    source = source_dir / "model.ply"
    cfg = json.loads((source_dir / "config.json").read_text())
    data = read_ply(source)
    xyz = data["means"]
    labels = np.zeros(len(xyz), np.int32)
    offset = 0
    for group, count in enumerate(cfg["physics"]["part_counts"]):
        labels[offset:offset + count] = group
        offset += count
    if scene == "ficus":
        static = labels >= 1
    elif scene == "alocasia":
        static = (labels == 2) | (labels == 3)
    else:
        static = (labels == 0) & (xyz[:, 2] < -0.082)
    lo, hi = np.quantile(xyz, [.001, .999], axis=0)
    scale = 1.65 / (hi[2] - lo[2])
    center = np.r_[np.median(xyz[static, :2], axis=0), lo[2]]
    xyz = (xyz - center) * scale
    xyz[:, 2] += .025
    data["means"] = xyz
    data["axes"] *= scale
    keep = (data["opacity"] > .025) & np.isfinite(xyz).all(axis=1)
    keep &= np.linalg.norm(data["axes"], axis=1).max(axis=1) < .12
    keep &= (xyz[:, 2] > -.035) & (xyz[:, 2] < 1.85)
    data = {k: v[keep].astype(np.float32) for k, v in data.items()}
    data["static"] = static[keep]
    data["part"] = labels[keep]
    data["source_rotation"] = np.eye(3, dtype=np.float32)
    out = Path(output_root) / scene
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "gaussians.npz", **data)
    manifest = dict(scene=scene, source=str(source),
                    repository="https://github.com/zju3dv/DiffWind",
                    sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    source_count=len(keep), selected_count=int(keep.sum()),
                    scale=float(scale), center=center.tolist(), static_count=int(static[keep].sum()))
    (out / "source.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=SCENES, nargs="+", default=list(SCENES))
    p.add_argument("--output-root", default=str(ROOT))
    args = p.parse_args()
    for scene in args.scene:
        prepare(scene, args.output_root)
