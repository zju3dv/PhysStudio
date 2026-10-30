"""Independent lattice moments, porous equilibrium, free fall, and water ledger."""

import argparse
import json
from pathlib import Path
import numpy as np
import warp as wp
from .common import population
from .scenes import initial
from .solver import Mixture
from . import fluid


@wp.kernel(enable_backward=False)
def evaluate_populations(
    velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    out: wp.array2d(dtype=float),
):
    i, q = wp.tid()
    out[i, q] = population(q, velocities[i], forces[i], wp.mat33(0.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wp.init()
    wp.set_device("cuda:0")
    checks = {}
    rng = np.random.default_rng(11)
    u = rng.uniform(-0.06, 0.06, (20, 3)).astype(np.float32)
    f = rng.uniform(-0.001, 0.001, (20, 3)).astype(np.float32)
    out = wp.zeros((len(u), 27), dtype=float)
    wp.launch(
        evaluate_populations,
        (len(u), 27),
        inputs=[wp.array(u, dtype=wp.vec3), wp.array(f, dtype=wp.vec3), out],
    )
    pop = out.numpy().astype(np.float64)
    c = np.stack(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij"), -1).reshape(-1, 3)
    second = np.einsum("nq,qi,qj->nij", pop, c, c)
    expected = (
        np.eye(3)[None] / 3
        + u[:, :, None] * u[:, None, :]
        + 0.5 * (u[:, :, None] * f[:, None, :] + f[:, :, None] * u[:, None, :])
    )
    checks["d3q27_moments"] = dict(
        mass=float(abs(pop.sum(1) - 1).max()),
        momentum=float(abs(pop @ c - u - 0.5 * f).max()),
        second=float(abs(second - expected).max()),
    )
    assert max(checks["d3q27_moments"].values()) < 2.0e-6

    cfg, wall, phi, u, x = initial("breach", 32, 1, "fluid")
    cfg.update(warmup=0, settle=0, retention=False)
    phi[:] = 1
    phi[wall == 1] = 0
    u[:] = 0
    sim = Mixture(cfg, wall, phi, u, x)
    sim.g.gravity = 0.0
    sim.g.sigma = 0.0
    points = np.stack(
        np.meshgrid(*(np.arange(n) for n in cfg["shape"]), indexing="ij"), -1
    )
    eps = np.where(points[..., 0] < 16, 0.36, 1.0).astype(np.float32).reshape(-1)
    sim.g.eps.assign(eps)
    sim.g.eps_eff.assign(eps)
    sim.g.eps_old.assign(eps)
    sim.launch(fluid.initialize_pore_water)
    sim.step(100)
    err = float(abs(sim.g.phi.numpy()[wall == 0] - 1).max())
    checks["static_porous_equilibrium"] = dict(
        max_phase_error=err,
        max_speed=float(np.linalg.norm(sim.g.u.numpy(), axis=-1).max()),
    )
    assert err < 2.0e-4

    cfg, wall, phi, u, x = initial("sediment", 36, 2, "sand")
    cfg.update(warmup=0, settle=0, retention=False)
    sim = Mixture(cfg, wall, phi, u, x)
    start_x = sim.s.x.numpy().mean(0)
    start_v = sim.s.v.numpy().mean(0)
    count = 80
    sim.step(count)
    expected = (
        start_x
        + count * start_v
        + np.array([0, 0, -0.5 * sim.g.gravity * count * (count + 1)])
    )
    poserror = float(np.linalg.norm(sim.s.x.numpy().mean(0) - expected))
    diag = sim.diagnostics()
    checks["sand_free_fall"] = dict(
        center_error_cells=poserror, mass_error=abs(diag["sand_mass_relative_error"])
    )
    assert poserror < 0.003 and checks["sand_free_fall"]["mass_error"] < 2.0e-6

    cfg, wall, phi, u, x = initial("breach", 40, 2, "coupled")
    cfg.update(warmup=40, settle=100, retention=True)
    sim = Mixture(cfg, wall, phi, u, x)
    sim.step(800)
    diag = sim.diagnostics()
    checks["coupled_retention"] = diag
    assert diag["finite"] and diag["particles_inside"] and diag["bound_water"] > 0
    assert (
        abs(diag["water_relative_error"]) < 1.0e-4
        and abs(diag["sand_mass_relative_error"]) < 2.0e-6
    )
    assert -0.02 < diag["phi_min"] and diag["phi_max"] < 1.2
    result = dict(passed=True, checks=checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
