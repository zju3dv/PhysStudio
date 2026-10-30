"""Friction complementarity and affine/moment invariants of Gaussian binding."""

from physstudio.paths import data_path
import json
from pathlib import Path
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from .mesh import Cage
from .gaussians import GaussianBody, split_moments
from .contact import forward_lcp, forward_lcp_reference


def main():
    wp.init()
    rng = np.random.default_rng(48)
    result = {}
    for name, q, gap in [
        ("sliding", np.array([1.2, 0.3, -2.0]), np.zeros(1)),
        ("separated", np.array([0.0, 0.0, -1.0]), np.array([0.1])),
    ]:
        f, v, _ = forward_lcp(q, np.eye(3), gap, 0.01, 0.5)
        assert abs(f[:, :2]).sum() <= 0.5 * f[:, 2].sum() + 1e-9
        assert np.min(v[:, 2] + gap / 0.01) > -1e-9
        assert np.max(abs(f[:, 2] * (v[:, 2] + gap / 0.01))) < 1e-9
        result[name] = dict(impulse=f.tolist(), velocity=v.tolist())
    n = 20
    b = rng.normal(size=(3 * n, 3 * n))
    w = b @ b.T + np.eye(3 * n) * 5
    q = rng.normal(size=3 * n)
    q[2::3] -= 2
    gap = rng.uniform(-0.001, 0.003, n)
    f, v, _ = forward_lcp(q, w, gap, 0.01, 0.6)
    fr, vr, _ = forward_lcp_reference(q, w, gap, 0.01, 0.6)
    result["compiled_lcp_max_error"] = float(max(abs(f - fr).max(), abs(v - vr).max()))
    assert result["compiled_lcp_max_error"] < 1e-9
    data = np.load(data_path("enlivening_gs/bread/gaussians.npz"))
    cage = Cage(data["means"], data["axes"], 7)
    body = GaussianBody(cage, data)
    affine = Rotation.from_rotvec([0.3, -0.2, 0.15]).as_matrix() @ np.diag(
        [1.1, 0.9, 1.05]
    )
    shift = np.array([0.2, 0.3, 0.4])
    means, cov, rotation, angle = body.deform(cage.x @ affine.T + shift)
    covref = affine @ (body.axes @ body.axes.transpose(1, 2)).cpu().numpy() @ affine.T
    result["affine_mean_max_error"] = float(
        abs(means.cpu().numpy() - (data["means"] @ affine.T + shift)).max()
    )
    result["affine_covariance_max_error"] = float(
        abs(cov.cpu().numpy() - covref - np.eye(3) * np.exp(-20)).max()
    )
    result["affine_bidirectional_max_angle"] = float(angle.max())
    assert result["affine_mean_max_error"] < 1e-9
    assert result["affine_covariance_max_error"] < 1e-9
    assert result["affine_bidirectional_max_angle"] < 1e-4
    means, axes, opacity = split_moments(
        body.means[:1000], body.axes[:1000], body.opacity[:1000]
    )
    n = 1000
    m1 = (means[:n] + means[n:]) / 2
    c1 = (
        axes[:n] @ axes[:n].transpose(1, 2)
        + (means[:n] - m1)[:, :, None] * (means[:n] - m1)[:, None, :]
    )
    result["split_mean_error"] = float(abs(m1 - body.means[:n]).max())
    result["split_covariance_error"] = float(
        abs(c1 - body.axes[:n] @ body.axes[:n].transpose(1, 2)).max()
    )
    result["split_mass_error"] = float(
        abs(opacity[:n] + opacity[n:] - body.opacity[:n]).max()
    )
    assert (
        max(
            result["split_mean_error"],
            result["split_covariance_error"],
            result["split_mass_error"],
        )
        < 1e-10
    )
    path = Path("outputs/enlivening_gs/contact_binding_validation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
