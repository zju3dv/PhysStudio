"""Warp MLS-MPM, particle-carried magnetic density and Maxwell force transfer.

Quadratic APIC/MLS transfers, compressible Newtonian liquid (paper eq. 19--21)
or Neo-Hookean solid (eq. 18). Magnetic forces use the cell integration
weights (1/6, 2/3, 1/6) from Appendix A.2, not a point attraction force.
"""

import numpy as np
import torch
import warp as wp
from .field import Field, MS


@wp.func
def weight(f: float):
    return wp.vec3(
        0.5 * (1.5 - f) * (1.5 - f),
        0.75 - (f - 1.0) * (f - 1.0),
        0.5 * (f - 0.5) * (f - 0.5),
    )


@wp.kernel(enable_backward=False)
def p2g(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    c: wp.array(dtype=wp.mat33),
    f: wp.array(dtype=wp.mat33),
    mass: wp.array3d(dtype=float),
    momentum: wp.array3d(dtype=wp.vec3),
    saturation: wp.array3d(dtype=float),
    occupied: wp.array3d(dtype=int),
    origin: wp.vec3,
    dx: float,
    dt: float,
    density: float,
    volume: float,
    bulk: float,
    viscosity: float,
    shear: float,
    ms: float,
    material: int,
):
    p = wp.tid()
    grid = (x[p] - origin) / dx
    base = wp.vec3i(
        wp.int32(wp.floor(grid[0] - 0.5)),
        wp.int32(wp.floor(grid[1] - 0.5)),
        wp.int32(wp.floor(grid[2] - 0.5)),
    )
    fx = grid - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    wx, wy, wz = weight(fx[0]), weight(fx[1]), weight(fx[2])
    j = wp.determinant(f[p])
    sig = wp.mat33(0.0)
    identity = wp.identity(n=3, dtype=float)
    if material == 0:
        sig = bulk * (j - 1.0) * identity + viscosity * (c[p] + wp.transpose(c[p]))
    else:
        sig = (
            shear * (f[p] * wp.transpose(f[p]) - identity)
            + bulk * wp.log(wp.max(j, 1.0e-12)) * identity
        ) / j
        sig += viscosity * (c[p] + wp.transpose(c[p]))
    pmass = density * volume
    affine = pmass * c[p] - dt * (4.0 / (dx * dx)) * volume * j * sig
    for a in range(3):
        for b in range(3):
            for d in range(3):
                w = wx[a] * wy[b] * wz[d]
                offset = (wp.vec3(float(a), float(b), float(d)) - fx) * dx
                i, jj, k = base[0] + a, base[1] + b, base[2] + d
                wp.atomic_add(mass, i, jj, k, w * pmass)
                wp.atomic_add(momentum, i, jj, k, w * (pmass * v[p] + affine * offset))
                wp.atomic_add(saturation, i, jj, k, w * ms * volume / (dx * dx * dx))
    nearest = wp.vec3i(
        wp.int32(wp.floor(grid[0] + 0.5)),
        wp.int32(wp.floor(grid[1] + 0.5)),
        wp.int32(wp.floor(grid[2] + 0.5)),
    )
    wp.atomic_add(occupied, nearest[0], nearest[1], nearest[2], 1)


@wp.kernel(enable_backward=False)
def integrate_force(
    raw: wp.array3d(dtype=wp.vec3),
    occupied: wp.array3d(dtype=int),
    out: wp.array3d(dtype=wp.vec3),
    dx: float,
    n: int,
):
    i, j, k = wp.tid()
    weights = wp.vec3(1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
    force = wp.vec3(0.0)
    if i > 0 and j > 0 and k > 0 and i < n - 1 and j < n - 1 and k < n - 1:
        for a in range(3):
            for b in range(3):
                for d in range(3):
                    u, v, w = i + a - 1, j + b - 1, k + d - 1
                    if occupied[u, v, w] > 0:
                        force += weights[a] * weights[b] * weights[d] * raw[u, v, w]
    out[i, j, k] = force * dx * dx * dx


@wp.func
def sdf_sample(sdf: wp.array3d(dtype=float), origin: wp.vec3, dx: float, x: wp.vec3):
    g = (x - origin) / dx
    base = wp.vec3i(
        wp.int32(wp.floor(g[0])), wp.int32(wp.floor(g[1])), wp.int32(wp.floor(g[2]))
    )
    frac = g - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    val = float(1.0)
    if (
        base[0] >= 0
        and base[1] >= 0
        and base[2] >= 0
        and base[0] < sdf.shape[0] - 1
        and base[1] < sdf.shape[1] - 1
        and base[2] < sdf.shape[2] - 1
    ):
        val = 0.0
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    wa = 1.0 - frac[0]
                    wb = 1.0 - frac[1]
                    wc = 1.0 - frac[2]
                    if a == 1:
                        wa = frac[0]
                    if b == 1:
                        wb = frac[1]
                    if c == 1:
                        wc = frac[2]
                    val += wa * wb * wc * sdf[base[0] + a, base[1] + b, base[2] + c]
    return val


@wp.kernel(enable_backward=False)
def grid_update(
    mass: wp.array3d(dtype=float),
    momentum: wp.array3d(dtype=wp.vec3),
    magnetic: wp.array3d(dtype=wp.vec3),
    velocity: wp.array3d(dtype=wp.vec3),
    origin: wp.vec3,
    dx: float,
    dt: float,
    n: int,
    floor: float,
    gravity: float,
    friction: float,
    clamp_height: float,
    magnet_center: wp.vec3,
    magnet_velocity: wp.vec3,
    magnet_radius: float,
    sdf: wp.array3d(dtype=float),
    sdf_origin: wp.vec3,
    sdf_dx: float,
    use_sdf: int,
):
    i, j, k = wp.tid()
    vel = wp.vec3(0.0)
    m = mass[i, j, k]
    if m > 1.0e-9:
        vel = (momentum[i, j, k] + dt * magnetic[i, j, k]) / m + wp.vec3(
            0.0, 0.0, dt * gravity
        )
        pos = origin + dx * wp.vec3(float(i), float(j), float(k))
        if pos[2] < floor + dx and vel[2] < 0.0:
            horizontal = wp.vec3(vel[0], vel[1], 0.0)
            speed = wp.length(horizontal)
            horizontal *= wp.max(0.0, 1.0 + friction * vel[2] / wp.max(speed, 1.0e-10))
            vel = horizontal
        if i < 3 and vel[0] < 0.0:
            vel[0] = 0.0
        if j < 3 and vel[1] < 0.0:
            vel[1] = 0.0
        if k < 3 and vel[2] < 0.0:
            vel[2] = 0.0
        if i > n - 4 and vel[0] > 0.0:
            vel[0] = 0.0
        if j > n - 4 and vel[1] > 0.0:
            vel[1] = 0.0
        if k > n - 4 and vel[2] > 0.0:
            vel[2] = 0.0
        dist = wp.length(pos - magnet_center)
        if magnet_radius > 0.0 and dist < magnet_radius + dx * 0.5:
            normal = (pos - magnet_center) / wp.max(dist, 1.0e-8)
            vn = wp.dot(vel - magnet_velocity, normal)
            if vn < 0.0:
                vel -= vn * normal
        if use_sdf == 1:
            sd = sdf_sample(sdf, sdf_origin, sdf_dx, pos)
            if sd < dx * 0.5:
                h = sdf_dx * 0.5
                grad = wp.vec3(
                    sdf_sample(sdf, sdf_origin, sdf_dx, pos + wp.vec3(h, 0.0, 0.0))
                    - sdf_sample(sdf, sdf_origin, sdf_dx, pos - wp.vec3(h, 0.0, 0.0)),
                    sdf_sample(sdf, sdf_origin, sdf_dx, pos + wp.vec3(0.0, h, 0.0))
                    - sdf_sample(sdf, sdf_origin, sdf_dx, pos - wp.vec3(0.0, h, 0.0)),
                    sdf_sample(sdf, sdf_origin, sdf_dx, pos + wp.vec3(0.0, 0.0, h))
                    - sdf_sample(sdf, sdf_origin, sdf_dx, pos - wp.vec3(0.0, 0.0, h)),
                )
                normal = wp.normalize(grad)
                vn = wp.dot(vel, normal)
                if vn < 0.0:
                    vel -= vn * normal
        if pos[2] < clamp_height:
            vel = wp.vec3(0.0)
    velocity[i, j, k] = vel


@wp.kernel(enable_backward=False)
def g2p(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    c: wp.array(dtype=wp.mat33),
    f: wp.array(dtype=wp.mat33),
    gv: wp.array3d(dtype=wp.vec3),
    origin: wp.vec3,
    dx: float,
    dt: float,
    material: int,
):
    p = wp.tid()
    grid = (x[p] - origin) / dx
    base = wp.vec3i(
        wp.int32(wp.floor(grid[0] - 0.5)),
        wp.int32(wp.floor(grid[1] - 0.5)),
        wp.int32(wp.floor(grid[2] - 0.5)),
    )
    fx = grid - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    wx, wy, wz = weight(fx[0]), weight(fx[1]), weight(fx[2])
    vel = wp.vec3(0.0)
    affine = wp.mat33(0.0)
    for a in range(3):
        for b in range(3):
            for d in range(3):
                w = wx[a] * wy[b] * wz[d]
                offset = (wp.vec3(float(a), float(b), float(d)) - fx) * dx
                velocity = gv[base[0] + a, base[1] + b, base[2] + d]
                vel += w * velocity
                affine += (4.0 / (dx * dx)) * w * wp.outer(velocity, offset)
    v[p] = vel
    c[p] = affine
    x[p] += dt * vel
    next_f = (wp.identity(n=3, dtype=float) + dt * affine) * f[p]
    if material == 0:
        j = wp.determinant(next_f)
        next_f = wp.identity(n=3, dtype=float) * wp.pow(j, 1.0 / 3.0)
    f[p] = next_f


class MPM:
    def __init__(
        self,
        positions,
        n=48,
        length=0.48,
        origin=(-0.24, -0.24, -0.06),
        dt=0.0002,
        density=2500.0,
        bulk=2.0e5,
        viscosity=12.0,
        shear=2.0e4,
        material="fluid",
        gravity=-9.81,
        floor=0.0,
        ms=MS,
        device="cuda:0",
        volume=None,
        clamp_height=-1.0,
    ):
        self.n, self.dx, self.origin, self.dt = (
            n,
            length / n,
            np.asarray(origin, np.float32),
            dt,
        )
        self.density, self.bulk, self.viscosity, self.shear, self.ms = (
            density,
            bulk,
            viscosity,
            shear,
            ms,
        )
        self.material = int(material == "solid")
        self.gravity = gravity
        self.floor = floor
        self.device = device
        self.clamp_height = clamp_height
        self.volume = (self.dx / 2.0) ** 3 if volume is None else volume
        self.field = Field(n, self.dx, self.origin, device)
        self.x = wp.array(positions.astype("f"), dtype=wp.vec3, device=device)
        self.v = wp.zeros(len(positions), dtype=wp.vec3, device=device)
        self.c = wp.zeros(len(positions), dtype=wp.mat33, device=device)
        self.f = wp.array(
            np.tile(np.eye(3, dtype="f"), (len(positions), 1, 1)),
            dtype=wp.mat33,
            device=device,
        )
        self.mass = wp.zeros((n, n, n), dtype=float, device=device)
        self.occupied = wp.zeros((n, n, n), dtype=int, device=device)
        self.momentum, self.velocity, self.magnetic = [
            wp.zeros((n, n, n), dtype=wp.vec3, device=device) for _ in range(3)
        ]
        self.sdf = wp.zeros((2, 2, 2), dtype=float, device=device)
        self.sdf_origin = wp.vec3(0.0)
        self.sdf_dx = 1.0
        self.use_sdf = 0
        self.time = 0.0
        self.steps = 0

    def set_collider(self, sdf, origin, dx):
        self.sdf = wp.array(sdf.astype("f"), dtype=float, device=self.device)
        self.sdf_origin = wp.vec3(*origin)
        self.sdf_dx = dx
        self.use_sdf = 1

    def transfer(self):
        for a in (self.mass, self.occupied, self.momentum, self.field.saturation):
            a.zero_()
        wp.launch(
            p2g,
            len(self.x),
            [
                self.x,
                self.v,
                self.c,
                self.f,
                self.mass,
                self.momentum,
                self.field.saturation,
                self.occupied,
                wp.vec3(*self.origin),
                self.dx,
                self.dt,
                self.density,
                self.volume,
                self.bulk,
                self.viscosity,
                self.shear,
                self.ms,
                self.material,
            ],
            device=self.device,
        )

    def step(
        self,
        center,
        radius,
        intensity,
        magnet_velocity=(0.0, 0.0, 0.0),
        update_field=True,
        uniform=(0.0, 0.0, 0.0),
    ):
        self.transfer()
        if update_field:
            self.field.set_source(center, radius, intensity, uniform)
            stats = self.field.solve()
            if not stats["converged"]:
                raise RuntimeError("Magnetic Newton solve failed: " + str(stats))
        wp.launch(
            integrate_force,
            (self.n,) * 3,
            [self.field.force_density, self.occupied, self.magnetic, self.dx, self.n],
            device=self.device,
        )
        wp.launch(
            grid_update,
            (self.n,) * 3,
            [
                self.mass,
                self.momentum,
                self.magnetic,
                self.velocity,
                wp.vec3(*self.origin),
                self.dx,
                self.dt,
                self.n,
                self.floor,
                self.gravity,
                0.15,
                self.clamp_height,
                wp.vec3(*center),
                wp.vec3(*magnet_velocity),
                radius,
                self.sdf,
                self.sdf_origin,
                self.sdf_dx,
                self.use_sdf,
            ],
            device=self.device,
        )
        wp.launch(
            g2p,
            len(self.x),
            [
                self.x,
                self.v,
                self.c,
                self.f,
                self.velocity,
                wp.vec3(*self.origin),
                self.dx,
                self.dt,
                self.material,
            ],
            device=self.device,
        )
        self.time += self.dt
        self.steps += 1

    def state(self):
        x, v, f = self.x.numpy(), self.v.numpy(), self.f.numpy()
        j = np.linalg.det(f)
        if (
            not np.isfinite(x).all()
            or not np.isfinite(v).all()
            or not np.isfinite(f).all()
            or j.min() <= 0
        ):
            raise FloatingPointError("Invalid MPM state")
        grid = (x - self.origin) / self.dx
        if grid.min() < 1.5 or grid.max() > self.n - 2.5:
            raise FloatingPointError("Particles left the supported MPM grid")
        return dict(positions=x, velocities=v, deformation=f)

    def diagnostics(self, state):
        x, v, f = [state[k] for k in ("positions", "velocities", "deformation")]
        j = np.linalg.det(f)
        return dict(
            time=self.time,
            steps=self.steps,
            particles=len(x),
            mass_kg=len(x) * self.volume * self.density,
            com=x.mean(0).tolist(),
            height_max=float(x[:, 2].max()),
            height_p99=float(np.quantile(x[:, 2], 0.99)),
            min_j=float(j.min()),
            max_j=float(j.max()),
            mean_j=float(j.mean()),
            speed_max=float(np.linalg.norm(v, axis=-1).max()),
            field_relative_residual=self.field.last_stats.get("relative_residual"),
        )
