"""Numerical checks for density projection and fluid/solid contact momentum."""

import json
from pathlib import Path
import numpy as np
import warp as wp
from .physics import PBF, compute_lambda, compute_delta, project
from .demo import lattice


def main():
    wp.init()
    wp.set_device("cuda:0")
    s = 0.04
    x = lattice([-0.2, -0.2, 0.5], [0.2, 0.2, 0.9], s) * np.array(
        [0.8, 0.8, 1.0], np.float32
    )
    sdf = np.ones((8, 8, 8), np.float32)
    sim = PBF(
        x,
        s,
        ([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]),
        sdf,
        np.array([-1.0, -1.0, -1.0]),
        0.1,
        [0.0, 0.0, 0.0],
        np.array([[0.0, 0.0, 0.0]], np.float32),
        contact=False,
    )

    def density():
        sim.grid.build(sim.x, sim.h)
        sim.launch(
            compute_lambda,
            sim.n,
            [sim.x, sim.grid.id, sim.h, sim.norm, sim.lam, sim.rho],
        )
        return sim.rho.numpy()

    before = density()
    for _ in range(12):
        density()
        sim.launch(
            compute_delta,
            sim.n,
            [sim.x, sim.grid.id, sim.h, sim.norm, s, sim.lam, sim.delta, 1.0],
        )
        sim.launch(
            project,
            sim.n,
            [
                sim.x,
                sim.delta,
                sim.lo,
                sim.hi,
                sim.sdf,
                sim.origin,
                sim.sdf_dx,
                sim.center,
                sim.rotation,
                0,
                0.0,
                wp.vec3(0.0),
                sim.particle_mass,
                sim.bdp,
                sim.bdr,
                s * 0.36,
            ],
        )
    after = density()
    assert after.max() < before.max() * 0.8, (before.max(), after.max())
    assert np.linalg.norm((sim.x.numpy() - x).mean(0)) < 1e-5
    # One particle inside a sphere, with the rigid body free of other contacts.
    dx = 0.025
    origin = np.array([-0.5, -0.5, -0.5], np.float32)
    xyz = np.stack(np.meshgrid(*[np.arange(41) * dx - 0.5] * 3, indexing="ij"), -1)
    sdf = (np.linalg.norm(xyz, axis=-1) - 0.30).astype(np.float32)
    contact = PBF(
        np.array([[0.28, 0.0, 0.0]], np.float32),
        s,
        ([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]),
        sdf,
        origin,
        dx,
        [0.0, 0.0, 0.0],
        np.array([[-0.3, 0, 0], [0.3, 0, 0]], np.float32),
        body_mass=2.0,
    )
    old = contact.x.numpy()
    contact.launch(
        project,
        1,
        [
            contact.x,
            contact.delta,
            contact.lo,
            contact.hi,
            contact.sdf,
            contact.origin,
            contact.sdf_dx,
            contact.center,
            contact.rotation,
            1,
            contact.im,
            contact.ii,
            contact.particle_mass,
            contact.bdp,
            contact.bdr,
            s * 0.36,
        ],
    )
    residual = (contact.x.numpy() - old)[
        0
    ] * contact.particle_mass + contact.bdp.numpy()[0] * 2.0
    assert np.linalg.norm(residual) < 1e-7, residual
    result = dict(
        compressed_density_max_before=float(before.max()),
        compressed_density_max_after=float(after.max()),
        contact_mass_weighted_displacement_residual=float(np.linalg.norm(residual)),
        status="passed",
    )
    path = Path("outputs/gaussian_splashing/numerical_validation.json")
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
