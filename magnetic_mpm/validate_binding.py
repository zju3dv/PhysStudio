"""Check exact affine reproduction and covariance / SH frame transport."""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from gaussian_splashing.assets import Object
from .render import BoundGaussians


@torch.no_grad()
def main(args):
    root = Path(args.input)
    cfg = json.loads((root / "config.json").read_text())
    tr = np.load(root / "trajectory.npz")
    rest = tr["rest"]
    obj = Object(
        cfg["asset"]["name"], height=cfg["asset"]["height"], device=args.device
    )
    center = np.asarray(cfg["asset"]["center"], dtype="f")
    binding = BoundGaussians(obj, rest, center, cfg["dx"])
    rot = Rotation.from_euler("xyz", [0.22, -0.37, 0.46]).as_matrix().astype("f")
    affine = rot @ np.diag([1.12, 0.89, 1.04]).astype("f")
    trans = np.array([0.027, -0.019, 0.034], dtype="f")
    state = dict(
        positions=rest @ affine.T + trans,
        deformation=np.broadcast_to(affine, (len(rest), 3, 3)).copy(),
    )
    means, cov, r = binding.deform(state)
    a = torch.tensor(affine, device=args.device)
    t = torch.tensor(trans, device=args.device)
    expected_means = (obj.local + torch.tensor(center, device=args.device)) @ a.T + t
    expected_cov = a @ obj.cov @ a.T
    mean_error = float((means - expected_means).abs().max())
    covariance_error = float(
        torch.linalg.vector_norm(cov - expected_cov)
        / torch.linalg.vector_norm(expected_cov)
    )
    rotation_error = float((r - torch.tensor(rot, device=args.device)).abs().max())
    assert mean_error < 2.0e-6 and covariance_error < 2.0e-5 and rotation_error < 2.0e-5
    result = dict(
        gaussians=len(means),
        affine_position_max_error_m=mean_error,
        covariance_relative_error=covariance_error,
        polar_rotation_max_error=rotation_error,
    )
    (root / "binding_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--device", default="cuda:0")
    main(p.parse_args())
