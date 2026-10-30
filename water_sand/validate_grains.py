"""Check material attachment, positive grain covariances, and sparse visibility."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import warp as wp
from .render_granular import GranularRenderer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads((args.input / "config.json").read_text())
    rest = np.load(args.input / "initial.npz")["positions"]
    with np.load(args.input / "states" / "00000.npz") as file:
        state = dict(file)
    wp.init()
    wp.set_device("cuda:0")
    renderer = GranularRenderer(cfg, rest, 320, 200, grains=8)
    x, cov, _, _, _ = renderer.sand_surface(state)
    first = x[renderer.base_count :].cpu().numpy()
    keys = renderer.grain_keys.copy()
    minimum = float(torch.linalg.eigvalsh(cov).min())
    shifted = dict(
        state, positions=state["positions"] + np.array([1, 0, 0], np.float32)
    )
    x2, _, _, _, _ = renderer.sand_surface(shifted)
    second = x2[renderer.base_count :].cpu().numpy()
    shared, a, b = np.intersect1d(keys, renderer.grain_keys, return_indices=True)
    error = np.linalg.norm(second[b] - first[a] - np.array([cfg["dx"], 0, 0]), axis=-1)
    assert len(shared) >= len(keys) * 0.999
    assert error.max() < 2e-4 and minimum > 0
    isolated = np.array(
        [[cfg["shape"][0] * 0.45, cfg["shape"][1] * 0.45, cfg["shape"][2] * 0.45]],
        np.float32,
    )
    lone = GranularRenderer(cfg, isolated, 320, 200, grains=8)
    simple = dict(
        positions=isolated,
        moisture=np.zeros(1, np.float32),
        deformation=np.eye(3, dtype=np.float32)[None],
        phi=np.zeros(cfg["shape"], np.float32),
    )
    sx, scov, _, _, _ = lone.sand_surface(simple)
    assert lone.base_count == 0 and len(sx) == 8
    assert torch.isfinite(sx).all() and torch.linalg.eigvalsh(scov).min() > 0
    result = dict(
        passed=True,
        matched_grain_ids=len(shared),
        max_rigid_translation_error_meters=float(error.max()),
        mean_rigid_translation_error_meters=float(error.mean()),
        covariance_min_eigenvalue=minimum,
        isolated_material_point_gaussians=len(sx),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
