"""Numerical and visibility checks for the electrical-discharge implementation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import warp as wp
from .solver import Poisson, growth_weights, laplacian
from .render import raster_channels


def numerical():
    n = 24
    dx = 0.1
    shape = (n, n, n)
    z = np.arange(n)[None, None, :]
    exact = np.broadcast_to(dx * dx * z * (2 * n - 1 - z), shape).copy()
    rho = np.full(shape, 2.0, dtype="f")
    fixed = np.zeros(shape, dtype="i")
    fixed[:, :, 0] = 1
    values = np.zeros(shape, dtype="f")
    pde = Poisson(rho, fixed, values, dx, iterations=48)
    phi = pde.solve(60)
    manufactured = float(np.max(np.abs(phi - exact)) / np.max(exact))
    assert manufactured < 5e-5, manufactured
    values[fixed != 0] = 17.0
    pde.boundaries(fixed, values)
    shifted = pde.solve(60)
    gauge_phi = float(np.max(np.abs(shifted - phi - 17.0)))
    assert gauge_phi < 0.002, gauge_phi
    p = np.array([-0.2, 1.0, -3.0, 7.0])
    q = np.array([1.0, 0.2, -2.0, -1.0])
    d = np.array([1.0, 0.3, 0.4, 2.0])
    weights = growth_weights(p, q, d, -1, 0.1, 2.3)
    shifted_weights = growth_weights(p + 199.0, q + 199.0, d, -1, 0.1, 2.3)
    gauge_weights = float(np.max(np.abs(weights - shifted_weights)))
    assert gauge_weights < 1e-10 and (weights >= 0).all()
    assert not growth_weights(p, q, d, 1, 100.0, 2.0).any()
    # A raised grounded conductor concentrates the surface electric field.
    fixed = np.zeros(shape, dtype="i")
    fixed[:, :, 0] = 1
    rod = fixed.copy()
    rod[n // 2 - 1 : n // 2 + 2, n // 2 - 1 : n // 2 + 2, :10] = 1
    values = np.zeros(shape, dtype="f")
    flat = Poisson(-rho, fixed, values, dx, iterations=48).solve(60)
    raised = Poisson(-rho, rod, values, dx, iterations=48).solve(60)
    ratio = float(abs(raised[n // 2, n // 2, 10]) / abs(flat[n // 2, n // 2, 1]))
    assert ratio > 1.2, ratio
    # Camera-space depth rejects emission behind opaque scene surfaces.
    a = wp.array(np.array([[-0.2, 0.0, 3.0]], dtype="f"), dtype=wp.vec3)
    b = wp.array(np.array([[0.2, 0.0, 3.0]], dtype="f"), dtype=wp.vec3)
    power = wp.array(np.array([1.0], dtype="f"))
    radius = wp.array(np.array([0.01], dtype="f"))
    surface = wp.array(np.full((64, 64), 2.0, dtype="f"))
    emission = wp.zeros((64, 64), dtype=wp.vec3)
    view = wp.mat44(*np.eye(4).reshape(-1).tolist())
    wp.launch(
        raster_channels,
        (1, 128),
        [a, b, power, radius, surface, emission, view, 80.0, 80.0, 32.0, 32.0, 0.0],
    )
    wp.synchronize()
    hidden = float(np.max(emission.numpy()))
    assert hidden == 0.0, hidden
    a.assign(np.array([[-0.2, 0.0, 1.0]], dtype="f"))
    b.assign(np.array([[0.2, 0.0, 1.0]], dtype="f"))
    wp.launch(
        raster_channels,
        (1, 128),
        [a, b, power, radius, surface, emission, view, 80.0, 80.0, 32.0, 32.0, 0.0],
    )
    wp.synchronize()
    visible = float(np.max(emission.numpy()))
    assert visible > 0.1, visible
    return dict(
        manufactured_relative_max_error=manufactured,
        potential_gauge_max_error=gauge_phi,
        growth_gauge_max_error=gauge_weights,
        conductor_field_enhancement=ratio,
        occluded_emission_max=hidden,
        visible_emission_max=visible,
    )


def audit(path):
    geometry = np.load(path / "geometry.npz")
    cfg = json.loads((path / "config.json").read_text())
    reports = []
    for f in sorted(path.glob("event_*.npz")):
        d = np.load(f)
        r = json.loads(f.with_suffix(".json").read_text())
        assert all(np.isfinite(d[k]).all() for k in d.files)
        parents = d["parents"]
        assert parents[0] == -1 and (parents[1:] < np.arange(1, len(parents))).all()
        assert (parents[1:] >= 0).all()
        lengths = np.linalg.norm(
            d["positions"][1:] - d["positions"][parents[1:]], axis=-1
        )
        assert lengths.max() <= np.sqrt(3) * float(d["dx"]) * 1.0001
        cells = np.rint((d["positions"] - d["origin"]) / float(d["dx"]) - 0.5).astype(
            int
        )
        inside = d["solid"][tuple(cells.T)]
        assert np.array_equal(np.where(inside)[0], d["contact"])
        assert len(d["contact"]) > 0
        assert len(np.unique(cells, axis=0)) == len(cells)
        assert r["poisson_relative_rms"] < 0.003, r["poisson_relative_rms"]
        before = [
            v["relative_imbalance"]
            for v in r["charge_log"]
            if not any(v["grounded"].values())
        ]
        assert max(before) < 0.035, max(before)
        induced = -laplacian(d["phi"], float(d["dx"])) - d["rho"]
        fixed = d["solid"].copy()
        fixed[tuple(cells.T)] = True
        ambient = float(d["rho"].sum()) * float(d["dx"]) ** 3
        boundary = float(induced[fixed].sum()) * float(d["dx"]) ** 3
        charge_closure = abs(ambient + boundary) / max(abs(ambient), 1e-10)
        assert charge_closure < 0.005, charge_closure
        reports.append(
            dict(
                event=f.name,
                nodes=len(parents),
                contacts=d["positions"][d["contact"]].tolist(),
                poisson_relative_rms=r["poisson_relative_rms"],
                max_precontact_imbalance=max(before),
                ambient_charge=ambient,
                earth_and_channel_charge=boundary,
                relative_charge_closure=charge_closure,
            )
        )
    assert len(reports) == cfg["events"]
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--simulations", type=Path, nargs="*", default=[])
    args = p.parse_args()
    torch.set_num_threads(4)
    wp.init()
    result = dict(
        numerical=numerical(), scenes={str(s): audit(s) for s in args.simulations}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
