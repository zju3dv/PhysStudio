"""Procedural 3D initial conditions; all geometry is simulated after release."""

import numpy as np


def sand_mask(scene, p):
    x, y, z = p.T
    if scene == "breach":
        height = 1.18 - 0.30 * np.exp(-(((y - 1.15) / 0.24) ** 4))
        half = 0.19 + 0.38 * np.maximum(height - z, 0)
        return (
            (abs(x - 1.50) < half)
            & (z > 0.02)
            & (z < height)
            & (y > 0.015)
            & (y < 2.28)
        )
    if scene == "castle":
        result = (
            (abs(x - 2.40) < 0.78) & (abs(y - 1.15) < 0.82) & (z < 0.16) & (z > 0.02)
        )
        core = (abs(x - 2.40) < 0.45) & (abs(y - 1.15) < 0.45) & (z > 0.10) & (z < 0.83)
        # A front gate is an actual void in the granular initial geometry.
        gate = (x < 2.20) & (abs(y - 1.15) < 0.18) & (z < 0.40)
        result |= core & ~gate
        for cx in [1.87, 2.93]:
            for cy in [0.58, 1.72]:
                r = np.hypot(x - cx, y - cy)
                theta = np.arctan2(y - cy, x - cx)
                top = 1.05 + 0.15 * (np.cos(theta * 8) > -0.15)
                tower = (r < 0.255) & (z > 0.08) & (z < top)
                tower &= ~((r < 0.145) & (z > 0.91))
                result |= tower
        return result
    if scene == "sediment":
        result = np.zeros(len(p), bool)
        for center, radius in [
            ((0.85, 0.70, 1.38), 0.28),
            ((1.80, 1.50, 1.57), 0.30),
            ((2.72, 0.80, 1.40), 0.32),
        ]:
            result |= np.linalg.norm(p - center, axis=1) < radius
        return result
    raise ValueError(scene)


def initial(scene, resolution=96, ppc=2, mode="coupled", seed=17):
    nx = int(resolution)
    dx = 3.6 / (nx - 6)
    ny, nz = int(round(2.3 / dx)) + 6, int(round(2.05 / dx)) + 6
    shape = (nx, ny, nz)
    ijk = np.stack(
        np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), -1
    ).reshape(-1, 3)
    xyz = (ijk - 3) * dx
    wall = np.any((ijk < 3) | (ijk >= np.array(shape) - 3), axis=1)
    x, y, z = xyz.T
    phi = np.zeros(len(ijk), np.float32)
    velocity = np.zeros((len(ijk), 3), np.float32)
    if mode != "sand":
        if scene == "breach":
            left = 1.50 - (0.19 + 0.38 * np.maximum(1.18 - z, 0)) - 0.04
            phi = ((x < left) & (z < 1.31)).astype(np.float32)
        elif scene == "castle":
            phi = ((x < 0.91) & (z < 1.24)).astype(np.float32)
            velocity[:, 0] = 0.8 * phi
        else:
            phi = (z < 0.86).astype(np.float32)
    phi[wall] = 0
    velocity[wall] = 0
    positions = []
    for a in range(ppc):
        for b in range(ppc):
            for c in range(ppc):
                sub = ijk + np.array(
                    [(a + 0.5) / ppc, (b + 0.5) / ppc, (c + 0.5) / ppc]
                )
                physical = (sub - 3) * dx
                selected = sand_mask(scene, physical) & ~wall
                positions.append(sub[selected])
    positions = np.concatenate(positions).astype(np.float32)
    if mode == "fluid":
        positions = positions[:1]
    rng = np.random.default_rng(seed)
    positions += rng.uniform(-0.015, 0.015, positions.shape).astype(np.float32)
    cfg = dict(
        scene=scene,
        mode=mode,
        shape=shape,
        dx=dx,
        ppc=ppc,
        seed=seed,
        rho_s=1500.0 if scene != "sediment" else 1800.0,
        young=140000.0 if scene != "sediment" else 80000.0,
        cohesion=0.01 if scene != "sediment" else 0.002,
        reference_length=ny,
        lattice_velocity=0.10,
        gravity=9.8,
        retention=True,
        frames=180,
        steps_per_frame=32,
        fps=30,
        warmup=60,
        settle=100,
    )
    return cfg, wall.astype(np.int32), phi, velocity, positions
