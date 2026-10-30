"""Check physical drag relaxation and uptake rates across grid resolutions."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import warp as wp
from .common import Grid
from .solver import Mixture
from . import sand


@wp.kernel(enable_backward=False)
def exchange(g: Grid, node: int):
    g.momentum[node] = g.mass[node] * g.vs[node]
    g.u[node] -= g.drag[node] / g.eps[node]


def relaxation(dx, fixed=True):
    shape = (9, 9, 9)
    node = 4 * 81 + 4 * 9 + 4
    ijk = np.stack(np.meshgrid(*[np.arange(9)] * 3, indexing="ij"), -1).reshape(-1, 3)
    wall = np.any((ijk < 3) | (ijk >= 6), axis=-1).astype(np.int32)
    cfg = dict(
        shape=shape,
        dx=dx,
        reference_length=2.52 / dx,
        gravity=9.8,
        lattice_velocity=0.1,
        rho_s=1500.0,
        young=140000.0,
        cohesion=0.01,
        ppc=2,
        retention=False,
        mode="coupled",
        warmup=0,
        settle=0,
        conservative=True,
        absorption_per_second=7.4542605,
    )
    if fixed:
        cfg["drag_reference_dx"] = 0.04
    phi = (1 - wall).astype(np.float32)
    sim = Mixture(
        cfg,
        wall,
        phi,
        np.zeros((len(wall), 3), np.float32),
        np.array([[4, 4, 4]], np.float32),
    )
    g = sim.g
    g.gravity = 0.0
    g.stressforce.zero_()
    g.momentum.zero_()
    delta = 0.30
    ms = delta * g.rho_s
    mf = 1 - delta
    mass = np.zeros(len(wall), np.float32)
    mass[node] = ms
    g.mass.assign(mass)
    vel = np.zeros((len(wall), 3), np.float32)
    vel[node, 0] = 1 / sim.velocity_scale
    g.u.assign(vel)
    count = round(0.2 / sim.dt)
    for _ in range(count):
        sim.launch(sand.grid_update)
        wp.launch(exchange, 1, inputs=[g, node], device="cuda:0")
    vs = float(g.vs.numpy()[node, 0]) * sim.velocity_scale
    vf = float(g.u.numpy()[node, 0]) * sim.velocity_scale
    area = math.pi * (3 * delta / (4 * math.pi)) ** (2 / 3)
    reference = 0.04 if fixed else dx
    beta = 0.5 * 0.39 * mf**-3.7 * area * (1 / ms + 1 / mf) / reference
    expected = 1 / (1 + beta * count * sim.dt)
    return dict(
        dx=dx,
        time=count * sim.dt,
        fluid_velocity=vf,
        sand_velocity=vs,
        relative_velocity=vf - vs,
        analytic_relative_velocity=expected,
        analytic_error=abs(vf - vs - expected),
        momentum_error=abs(ms * vs + mf * vf - mf),
        absorption_per_second=g.absorb_rate / sim.dt,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wp.init()
    wp.set_device("cuda:0")
    fixed = [relaxation(dx) for dx in [0.04, 3.6 / 122, 0.02]]
    legacy = relaxation(3.6 / 122, False)
    assert max(x["analytic_error"] for x in fixed) < 2e-5
    assert max(x["momentum_error"] for x in fixed) < 2e-5
    assert max(abs(x["absorption_per_second"] - 7.4542605) for x in fixed) < 1e-5
    report = dict(
        passed=True, fixed_physical_drag=fixed, unscaled_fine_grid_drag=legacy
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
