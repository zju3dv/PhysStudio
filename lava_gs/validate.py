"""Numerical checks for mass, free fall, heat balance and thermal rheology."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from .mpm import LavaMPM


def make(
    temperature=1450.0, exchange=0.0, conductivity=2.2, gravity=(0, 0, -9.81), air=False
):
    dx = 0.06
    origin = np.array([-1.0, -1.0, -0.36], dtype="f")
    shape = (36, 36, 34)
    phi = (
        np.broadcast_to(origin[2] + np.arange(shape[2])[None, None, :] * dx, shape)
        .copy()
        .astype("f")
    )
    if air:
        phi[:] = 10.0
    x = np.stack(
        np.meshgrid(
            np.arange(-0.18, 0.18, 0.03),
            np.arange(-0.18, 0.18, 0.03),
            np.arange(0.15, 0.69, 0.03),
            indexing="ij",
        ),
        -1,
    ).reshape(-1, 3)
    geo = dict(phi=phi, origin=origin, dx=dx)
    temp = np.full(len(x), temperature, dtype="f")
    return LavaMPM(
        x,
        temp,
        geo,
        capacity=len(x),
        eta=900.0,
        friction=0.06,
        exchange=exchange,
        conductivity=conductivity,
        gravity=gravity,
    ), geo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    results = {}
    m, g = make(air=True)
    m.advance(0.05)
    s = m.state()
    m.transfer(0.0)
    mass = m.mass.numpy().sum(dtype=np.float64)
    expected = m.count * m.rho * m.volume
    error = float(np.max(np.abs(s["v"] - np.array([0, 0, -9.81 * 0.05]))))
    results["free_fall"] = dict(
        max_velocity_error_m_s=error, relative_mass_error=abs(mass / expected - 1)
    )
    assert error < 1.0e-4 and abs(mass / expected - 1) < 2.0e-6
    m, g = make(air=True, gravity=(0, 0, 0), conductivity=1000.0)
    import warp as wp

    temp = np.where(m.state()["x"][:, 0] < 0, 1500.0, 1100.0).astype("f")
    wp.to_torch(m.temp).copy_(torch.as_tensor(temp, device=m.device))
    old = float(temp.mean())
    variance = float(temp.var())
    for _ in range(10):
        m.advance(1 / 30)
    new = m.state()["temperature"]
    results["closed_heat"] = dict(
        initial_mean_k=old,
        final_mean_k=float(new.mean()),
        relative_energy_error=abs(float(new.mean()) / old - 1),
        initial_variance=variance,
        final_variance=float(new.var()),
        minimum=float(new.min()),
        maximum=float(new.max()),
    )
    assert abs(float(new.mean()) / old - 1) < 2.0e-5 and new.var() < variance
    assert new.min() > 1099.9 and new.max() < 1500.1
    rheology = {}
    for label, t in [("hot", 1490.0), ("cold", 950.0)]:
        m, g = make(temperature=t)
        initial = m.state()["x"].copy()
        for _ in range(30):
            m.advance(1 / 30)
        s = m.state()
        sd = map_coordinates(g["phi"], ((s["x"] - g["origin"]) / g["dx"]).T, order=1)
        extent = np.ptp(s["x"][:, :2], axis=0)
        rheology[label] = dict(
            horizontal_area_m2=float(np.prod(extent)),
            max_height_m=float(s["x"][:, 2].max()),
            min_collider_distance_m=float(sd.min()),
            **m.diagnostics(),
        )
        assert sd.min() > -0.001
    results["rheology"] = rheology
    assert (
        rheology["hot"]["horizontal_area_m2"]
        > rheology["cold"]["horizontal_area_m2"] * 1.1
    )
    m, g = make(exchange=20.0)
    initial = m.state()["temperature"].mean()
    for _ in range(10):
        m.advance(1 / 30)
    final = m.state()["temperature"]
    results["cooling"] = dict(
        initial_mean_k=float(initial),
        final_mean_k=float(final.mean()),
        min_k=float(final.min()),
        max_k=float(final.max()),
    )
    assert final.mean() < initial and final.max() < initial + 0.1 and final.min() > 300
    m, g = make(air=True, gravity=(0, 0, 0))
    m.bulk = 0.0
    checker = np.where(np.arange(m.count) % 2, 0.85, 1.15).astype("f")
    wp.to_torch(m.j).copy_(torch.as_tensor(checker, device=m.device))
    m.advance(0.05)
    projected = m.state()["j"]
    results["volume_projection"] = dict(
        initial_mean=float(checker.mean()),
        final_mean=float(projected.mean()),
        relative_volume_error=abs(float(projected.mean() / checker.mean()) - 1),
        initial_variance=float(checker.var()),
        final_variance=float(projected.var()),
    )
    assert abs(projected.mean() - checker.mean()) < 2.0e-6
    assert projected.var() < checker.var() * 0.2
    m, g = make()
    m.bulk = 1.2e6
    wp.to_torch(m.v)[:, 2] = -8.0
    for _ in range(12):
        m.advance(1 / 30)
        m.diagnostics()
    s = m.state()
    sd = map_coordinates(g["phi"], ((s["x"] - g["origin"]) / g["dx"]).T, order=1)
    results["high_speed_impact"] = dict(
        **m.diagnostics(), min_collider_distance_m=float(sd.min())
    )
    assert sd.min() > -0.001
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=True, checks=results), indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
