"""Independent free-fall, plastic impact, friction and deposition checks."""

import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
import warp as wp
from .mpm import SnowMPM


def penetration(mpm):
    s = mpm.state()
    sd = map_coordinates(
        mpm.phi.numpy(), ((s["x"] - mpm.origin) / mpm.dx).T, order=1, mode="nearest"
    )
    return float(max(0, -sd.min()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    results = {}
    dx = 0.025
    shape = (48, 32, 48)
    origin = np.array([-0.2, -0.4, -0.1], "f")
    phi = np.broadcast_to(
        origin[2] + np.arange(shape[2])[None, None, :] * dx, shape
    ).copy()
    axis = np.arange(5) * dx * 0.5
    positions = (
        np.stack(np.meshgrid(axis + 0.3, axis - 0.03, axis + 0.35, indexing="ij"), -1)
        .reshape(-1, 3)
        .astype("f")
    )
    free = SnowMPM(
        positions, phi, origin, dx, (dx * 0.5) ** 3, dt=0.0005, extra_capacity=0
    )
    free.advance(0.04)
    s = free.state()
    vz = float(s["v"][:, 2].mean())
    results["free_fall"] = dict(
        vz=vz,
        expected=-9.81 * 0.04,
        relative_error=abs(vz + 9.81 * 0.04) / (0.04 * 9.81),
    )
    assert results["free_fall"]["relative_error"] < 0.005
    for friction in (0.0, 0.7):
        m = SnowMPM(
            positions,
            phi,
            origin,
            dx,
            (dx * 0.5) ** 3,
            dt=0.0005,
            friction=friction,
            extra_capacity=8,
        )
        wp.to_torch(m.v)[: m.count, 0] = 0.6
        initial = positions.mean(0)
        history = []
        for t in range(21):
            m.advance(0.025)
            history.append(m.diagnostics())
        state = m.state()
        error = penetration(m)
        result = dict(
            history=history,
            penetration_m=error,
            displacement=(state["x"].mean(0) - initial).tolist(),
            final_x_velocity=float(state["v"][:, 0].mean()),
            final=m.diagnostics(),
        )
        results[f"friction_{friction}"] = result
        assert error < 0.001
        assert (
            abs(result["final"]["mass_kg"] - result["final"]["initial_mass_kg"])
            < 1.0e-7
        )
        assert result["final"]["elastic_stretch_min"] >= 1 - m.theta_c - 0.001
        assert result["final"]["elastic_stretch_max"] <= 1 + m.theta_s + 0.001
        assert result["final"]["jp_min"] < 0.999
        np.savez_compressed(a.output / f"impact_mu_{friction}.npz", **state)
        print(
            json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True
        )
    assert (
        results["friction_0.7"]["displacement"][0]
        < results["friction_0.0"]["displacement"][0] * 0.9
    )
    m.deposit(np.array([[0.5, 0, 0.4]], "f"), np.array([[0, 0, -0.3]], "f"), 0.00001)
    m.advance(0.01)
    expected = m.initial_mass + 0.00001 * m.density
    results["deposition"] = dict(
        actual_mass=m.diagnostics()["mass_kg"], expected_mass=expected
    )
    assert abs(results["deposition"]["actual_mass"] - expected) < 1.0e-6
    results["passed"] = True
    (a.output / "validation.json").write_text(json.dumps(results, indent=2))
    print("All physical checks passed", flush=True)


if __name__ == "__main__":
    main()
