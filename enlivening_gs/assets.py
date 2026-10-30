"""Public pretrained splats and explicit source-to-simulation transforms."""

from physstudio.paths import data_path
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation

PREPARED_ROOT = data_path("enlivening_gs")


def read_ply(path):
    v = PlyData.read(str(path))["vertex"]

    def a(names):
        return np.stack([v[n] for n in names], -1).astype(np.float64)

    q = a(["rot_" + str(i) for i in range(4)])
    r = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    s = np.exp(a(["scale_" + str(i) for i in range(3)]))
    return dict(
        means=a(["x", "y", "z"]),
        axes=r * s[:, None, :],
        opacity=1 / (1 + np.exp(-np.clip(a(["opacity"])[:, 0], -40, 40))),
        sh=np.concatenate(
            [
                a(["f_dc_" + str(i) for i in range(3)])[:, None],
                a(["f_rest_" + str(i) for i in range(45)])
                .reshape(-1, 3, 15)
                .transpose(0, 2, 1),
            ],
            1,
        ),
    )


def align_normal(n):
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    c = n @ z
    if c < -0.9999:
        return Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def prepare(name, height=1.25):
    repository = "https://github.com/XPandora/PhysGaussian"
    path = next(data_path("gaussian_splashing", name).rglob("point_cloud.ply"))
    a = read_ply(path)
    count = len(a["means"])
    keep = (a["opacity"] > 0.035) & np.isfinite(a["means"]).all(-1)
    rotation = np.eye(3)
    plane = None
    if name == "bread":
        rotation = Rotation.from_euler("y", 25, degrees=True).as_matrix()
    if name == "vasedeck":
        # The authors' released selection is in their rotated coordinates.
        cfg = json.loads((path.parents[3] / "source_config.json").read_text())
        r = Rotation.from_euler("x", 135, degrees=True).as_matrix()
        pos = a["means"] @ r.T
        bounds = np.array(cfg["sim_area"]).reshape(3, 2)
        keep &= ((pos > bounds[:, 0]) & (pos < bounds[:, 1])).all(-1)
        from .plane import fit_vase_plane

        plane = fit_vase_plane(path, a)
        rotation = align_normal(np.array(plane["normal"]))
        keep &= a["means"] @ np.array(plane["normal"]) + plane["offset"] > 0.025
    a = {k: v[keep] for k, v in a.items()}
    a["means"] = a["means"] @ rotation.T
    a["axes"] = rotation @ a["axes"]
    lo, hi = np.quantile(a["means"], [0.002, 0.998], axis=0)
    extent = hi - lo
    keep = (np.abs(a["means"] - (lo + hi) / 2) < extent * 0.56).all(-1)
    keep &= np.linalg.norm(a["axes"], axis=1).max(-1) < extent.max() * 0.065
    a = {k: v[keep] for k, v in a.items()}
    scale = height / (hi[2] - lo[2])
    center = (lo + hi) / 2
    a["means"] = (a["means"] - center) * scale
    a["axes"] *= scale
    # Plane-prior segmentation: shrink only Gaussians crossing the cut plane.
    # For isolated white-background assets the support plane is their bottom.
    support = np.max(np.abs(3 * a["axes"][:, 2, :]), axis=-1)
    bottom = np.quantile(a["means"][:, 2] - support, 0.002)
    cut = a["means"][:, 2] < bottom + 1e-5
    a = {k: v[~cut] for k, v in a.items()}
    a["means"][:, 2] -= bottom
    support = np.max(np.abs(3 * a["axes"][:, 2, :]), axis=-1)
    factor = np.minimum(1.0, a["means"][:, 2] / np.maximum(support, 1e-12))
    # Coordinate shrink is a positive affine covariance transform.
    a["axes"][:, 2, :] *= factor[:, None]
    out = PREPARED_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "gaussians.npz",
        **{k: v.astype(np.float32) for k, v in a.items()},
        source_rotation=rotation.astype(np.float32),
    )
    manifest = dict(
        name=name,
        repository=repository,
        source=str(path.resolve()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_count=count,
        selected_count=len(a["means"]),
        height=height,
        source_rotation=rotation.tolist(),
        source_center=center.tolist(),
        scale=scale,
        bottom=float(bottom),
        plane=plane,
        cut_regularized=int((factor < 1).sum()),
    )
    (out / "source.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=["wolf", "bread", "vasedeck"], required=True)
    args = p.parse_args()
    prepare(args.scene)
