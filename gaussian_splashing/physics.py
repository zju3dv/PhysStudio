"""GPU position-based fluids with SDF contact and a coupled rigid body.

Density projection follows Macklin & Mueller (2013). Geometry is in metres,
time in seconds, density in kg/m^3. The rigid body exchanges equal/opposite
contact corrections with fluid particles; static boundaries have infinite mass.
"""

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def weight(r: wp.vec3, h: float):
    q = wp.max(0.0, 1.0 - wp.dot(r, r) / (h * h))
    return q * q * q


@wp.func
def gradient(r: wp.vec3, h: float, norm: float):
    # PBF uses the spiky pressure gradient: unlike the poly6 derivative it
    # retains a separating force for nearly coincident neighbors.
    length = wp.length(r)
    result = wp.vec3(0.0)
    if length > 0.000001 and length < h:
        q = 1.0 - length / h
        result = -9.142857142857 * q * q / (h * norm * length) * r
    return result


@wp.func
def sdf_value(p: wp.vec3, field: wp.array3d(dtype=float), origin: wp.vec3, dx: float):
    q = (p - origin) / dx
    i = wp.clamp(int(wp.floor(q[0])), 0, field.shape[0] - 2)
    j = wp.clamp(int(wp.floor(q[1])), 0, field.shape[1] - 2)
    k = wp.clamp(int(wp.floor(q[2])), 0, field.shape[2] - 2)
    f = wp.vec3(
        wp.clamp(q[0] - float(i), 0.0, 1.0),
        wp.clamp(q[1] - float(j), 0.0, 1.0),
        wp.clamp(q[2] - float(k), 0.0, 1.0),
    )
    result = float(0.0)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                wa = 1.0 - f[0]
                wb = 1.0 - f[1]
                wc = 1.0 - f[2]
                if a == 1:
                    wa = f[0]
                if b == 1:
                    wb = f[1]
                if c == 1:
                    wc = f[2]
                result += wa * wb * wc * field[i + a, j + b, k + c]
    # SDF grid is padded. Outside its bounds cannot collide with the body.
    if (
        q[0] < 1.0
        or q[1] < 1.0
        or q[2] < 1.0
        or q[0] > float(field.shape[0] - 2)
        or q[1] > float(field.shape[1] - 2)
        or q[2] > float(field.shape[2] - 2)
    ):
        result = 10.0
    return result


@wp.kernel
def predict(
    x: wp.array(dtype=wp.vec3),
    old: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    old[i] = x[i]
    v[i] = v[i] + wp.vec3(0.0, 0.0, -9.81) * dt
    x[i] = x[i] + v[i] * dt


@wp.kernel
def compute_lambda(
    x: wp.array(dtype=wp.vec3),
    grid: wp.uint64,
    h: float,
    norm: float,
    lam: wp.array(dtype=float),
    rho: wp.array(dtype=float),
):
    i = wp.tid()
    p = x[i]
    density = float(0.0)
    gsum = wp.vec3(0.0)
    g2 = float(0.0)
    for j in wp.hash_grid_query(grid, p, h):
        r = p - x[j]
        density += weight(r, h) / norm
        g = gradient(r, h, norm)
        gsum += g
        g2 += wp.dot(g, g)
    rho[i] = density
    lam[i] = -wp.max(density - 1.0, 0.0) / (g2 + wp.dot(gsum, gsum) + 0.02 / (h * h))


@wp.kernel
def compute_delta(
    x: wp.array(dtype=wp.vec3),
    grid: wp.uint64,
    h: float,
    norm: float,
    spacing: float,
    lam: wp.array(dtype=float),
    delta: wp.array(dtype=wp.vec3),
    dt_scale: float,
):
    i = wp.tid()
    p = x[i]
    d = wp.vec3(0.0)
    for j in wp.hash_grid_query(grid, p, h):
        if i != j:
            r = p - x[j]
            w = weight(r, h) / 0.753571
            tensile = -0.00003 * h * h * w * w * w * w * dt_scale
            d += (lam[i] + lam[j] + tensile) * gradient(r, h, norm)
            l = wp.length(r)
            if l > 0.00001 and l < spacing * 0.65:
                d += 0.02 * dt_scale * (spacing * 0.65 - l) * r / l
    # Only guards a pathological unresolved contact, not normal gravity speeds.
    l = wp.length(d)
    if l > spacing * 0.4:
        d = d * (spacing * 0.4 / l)
    delta[i] = d


@wp.kernel
def project(
    x: wp.array(dtype=wp.vec3),
    delta: wp.array(dtype=wp.vec3),
    lo: wp.vec3,
    hi: wp.vec3,
    sdf: wp.array3d(dtype=float),
    origin: wp.vec3,
    dx: float,
    center: wp.array(dtype=wp.vec3),
    rotation: wp.array(dtype=wp.quat),
    contact: int,
    inverse_mass: float,
    inverse_inertia: wp.vec3,
    particle_mass: float,
    body_dp: wp.array(dtype=wp.vec3),
    body_dr: wp.array(dtype=wp.vec3),
    radius: float,
):
    i = wp.tid()
    p = x[i] + delta[i]
    if contact > 0:
        rel = p - center[0]
        q = wp.quat_rotate_inv(rotation[0], rel)
        dist = sdf_value(q, sdf, origin, dx) - radius
        if dist < 0.0:
            eps = dx * 0.6
            n = wp.vec3(
                sdf_value(q + wp.vec3(eps, 0.0, 0.0), sdf, origin, dx)
                - sdf_value(q - wp.vec3(eps, 0.0, 0.0), sdf, origin, dx),
                sdf_value(q + wp.vec3(0.0, eps, 0.0), sdf, origin, dx)
                - sdf_value(q - wp.vec3(0.0, eps, 0.0), sdf, origin, dx),
                sdf_value(q + wp.vec3(0.0, 0.0, eps), sdf, origin, dx)
                - sdf_value(q - wp.vec3(0.0, 0.0, eps), sdf, origin, dx),
            )
            n = wp.quat_rotate(rotation[0], wp.normalize(n))
            arm = wp.cross(rel, n)
            local_arm = wp.quat_rotate_inv(rotation[0], arm)
            inertia_arm = wp.cw_mul(local_arm, inverse_inertia)
            effective = 1.0 + particle_mass * (
                inverse_mass + wp.dot(local_arm, inertia_arm)
            )
            correction = -dist / effective * n
            # Bound a single correction to retain neighbor-list accuracy.
            l = wp.length(correction)
            if l > 2.0 * radius:
                correction = correction * (2.0 * radius / l)
            p += correction
            wp.atomic_add(body_dp, 0, -particle_mass * inverse_mass * correction)
            torque = wp.quat_rotate_inv(rotation[0], wp.cross(rel, correction))
            wp.atomic_add(
                body_dr,
                0,
                wp.quat_rotate(
                    rotation[0], -particle_mass * wp.cw_mul(torque, inverse_inertia)
                ),
            )
    p = wp.vec3(
        wp.clamp(p[0], lo[0] + radius, hi[0] - radius),
        wp.clamp(p[1], lo[1] + radius, hi[1] - radius),
        wp.clamp(p[2], lo[2] + radius, hi[2] - radius),
    )
    x[i] = p


@wp.kernel
def body_predict(
    center: wp.array(dtype=wp.vec3),
    rot: wp.array(dtype=wp.quat),
    vel: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
    old: wp.array(dtype=wp.vec3),
    oldq: wp.array(dtype=wp.quat),
    dt: float,
):
    old[0] = center[0]
    oldq[0] = rot[0]
    vel[0] = vel[0] + wp.vec3(0.0, 0.0, -9.81) * dt
    center[0] += vel[0] * dt
    w = omega[0]
    angle = wp.length(w) * dt
    if angle > 0.000001:
        rot[0] = wp.normalize(wp.quat_from_axis_angle(wp.normalize(w), angle) * rot[0])


@wp.kernel
def body_project(
    center: wp.array(dtype=wp.vec3),
    rot: wp.array(dtype=wp.quat),
    dp: wp.array(dtype=wp.vec3),
    dr: wp.array(dtype=wp.vec3),
    corners: wp.array(dtype=wp.vec3),
    lower: wp.vec3,
    upper: wp.vec3,
):
    center[0] += dp[0]
    angle = wp.length(dr[0])
    if angle > 0.000001:
        rot[0] = wp.normalize(
            wp.quat_from_axis_angle(dr[0] / angle, wp.min(angle, 0.08)) * rot[0]
        )
    # Container contacts for the rigid collision proxy.
    for k in range(corners.shape[0]):
        p = center[0] + wp.quat_rotate(rot[0], corners[k])
        center[0] += wp.vec3(
            wp.max(lower[0] - p[0], 0.0) + wp.min(upper[0] - p[0], 0.0),
            wp.max(lower[1] - p[1], 0.0) + wp.min(upper[1] - p[1], 0.0),
            wp.max(lower[2] - p[2], 0.0),
        )


@wp.kernel
def body_velocity(
    center: wp.array(dtype=wp.vec3),
    rot: wp.array(dtype=wp.quat),
    old: wp.array(dtype=wp.vec3),
    oldq: wp.array(dtype=wp.quat),
    vel: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
    dt: float,
):
    vel[0] = (center[0] - old[0]) / dt * 0.999
    q = rot[0] * wp.quat_inverse(oldq[0])
    if q[3] < 0.0:
        q = wp.quat(-q[0], -q[1], -q[2], -q[3])
    omega[0] = wp.vec3(q[0], q[1], q[2]) * 2.0 / dt * 0.997


@wp.kernel
def velocity(
    x: wp.array(dtype=wp.vec3),
    old: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    v[i] = (x[i] - old[i]) / dt


@wp.kernel
def smooth_velocity(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
    grid: wp.uint64,
    h: float,
    norm: float,
    viscosity: float,
):
    i = wp.tid()
    dv = wp.vec3(0.0)
    for j in wp.hash_grid_query(grid, x[i], h):
        dv += (v[j] - v[i]) * weight(x[i] - x[j], h) / norm
    out[i] = v[i] + viscosity * dv


class PBF:
    def __init__(
        self,
        positions,
        spacing,
        bounds,
        sdf,
        origin,
        sdf_dx,
        body_center,
        body_points,
        body_mass=0.0,
        contact=True,
        velocities=None,
        device="cuda:0",
    ):
        self.device = device
        self.n = len(positions)
        self.spacing = spacing
        self.h = spacing * 2.3
        self.lo, self.hi = wp.vec3(*bounds[0]), wp.vec3(*bounds[1])
        lattice = np.stack(
            np.meshgrid(*[np.arange(-3, 4) * spacing] * 3, indexing="ij"), -1
        ).reshape(-1, 3)
        self.norm = float(
            np.maximum(1 - (lattice * lattice).sum(-1) / self.h**2, 0) ** 3
            @ np.ones(len(lattice))
        )
        self.x = wp.array(
            np.asarray(positions, np.float32), dtype=wp.vec3, device=device
        )
        self.old = wp.clone(self.x)
        self.v = wp.array(
            np.zeros_like(positions) if velocities is None else velocities,
            dtype=wp.vec3,
            device=device,
        )
        self.tmp = wp.zeros_like(self.v)
        self.delta = wp.zeros_like(self.v)
        self.lam = wp.zeros(self.n, dtype=float, device=device)
        self.rho = wp.zeros_like(self.lam)
        self.grid = wp.HashGrid(128, 128, 128, device=device)
        self.sdf = wp.array(sdf.astype(np.float32), dtype=float, device=device)
        self.origin = wp.vec3(*origin)
        self.sdf_dx = sdf_dx
        self.center = wp.array([body_center], dtype=wp.vec3, device=device)
        self.rotation = wp.array([[0.0, 0.0, 0.0, 1.0]], dtype=wp.quat, device=device)
        self.bc_old = wp.clone(self.center)
        self.bq_old = wp.clone(self.rotation)
        self.bv = wp.zeros(1, dtype=wp.vec3, device=device)
        self.bw = wp.zeros_like(self.bv)
        self.bdp = wp.zeros_like(self.bv)
        self.bdr = wp.zeros_like(self.bv)
        self.corners = wp.array(
            body_points.astype(np.float32), dtype=wp.vec3, device=device
        )
        self.contact = int(contact)
        self.mass = body_mass
        self.im = 1.0 / body_mass if body_mass > 0 else 0.0
        var = np.mean(body_points**2, axis=0)
        self.ii = wp.vec3(
            *(
                1 / (body_mass * (var.sum() - var) + 1e-8)
                if body_mass > 0
                else np.zeros(3)
            )
        )
        self.particle_mass = 1000 * spacing**3
        self.steps = 0

    def launch(self, k, n, args):
        wp.launch(k, n, inputs=args, device=self.device)

    def step(self, dt=1 / 180, iterations=5, viscosity=0.035, moving_body=False):
        self.launch(predict, self.n, [self.x, self.old, self.v, dt])
        if moving_body:
            self.launch(
                body_predict,
                1,
                [
                    self.center,
                    self.rotation,
                    self.bv,
                    self.bw,
                    self.bc_old,
                    self.bq_old,
                    dt,
                ],
            )
        for _ in range(iterations):
            # Contact can move particles across hash cells. Rebuild before
            # every density iteration so the query and positions agree.
            self.grid.build(self.x, self.h)
            self.launch(
                compute_lambda,
                self.n,
                [self.x, self.grid.id, self.h, self.norm, self.lam, self.rho],
            )
            self.launch(
                compute_delta,
                self.n,
                [
                    self.x,
                    self.grid.id,
                    self.h,
                    self.norm,
                    self.spacing,
                    self.lam,
                    self.delta,
                    (dt * 240) ** 2,
                ],
            )
            self.bdp.zero_()
            self.bdr.zero_()
            self.launch(
                project,
                self.n,
                [
                    self.x,
                    self.delta,
                    self.lo,
                    self.hi,
                    self.sdf,
                    self.origin,
                    self.sdf_dx,
                    self.center,
                    self.rotation,
                    self.contact,
                    self.im if moving_body else 0.0,
                    self.ii if moving_body else wp.vec3(0.0),
                    self.particle_mass,
                    self.bdp,
                    self.bdr,
                    self.spacing * 0.36,
                ],
            )
            if moving_body:
                self.launch(
                    body_project,
                    1,
                    [
                        self.center,
                        self.rotation,
                        self.bdp,
                        self.bdr,
                        self.corners,
                        self.lo,
                        self.hi,
                    ],
                )
        if self.contact and not moving_body:
            # Complete contact after density projection, including fast droplets.
            self.delta.zero_()
            for _ in range(2):
                self.launch(
                    project,
                    self.n,
                    [
                        self.x,
                        self.delta,
                        self.lo,
                        self.hi,
                        self.sdf,
                        self.origin,
                        self.sdf_dx,
                        self.center,
                        self.rotation,
                        self.contact,
                        0.0,
                        wp.vec3(0.0),
                        self.particle_mass,
                        self.bdp,
                        self.bdr,
                        self.spacing * 0.36,
                    ],
                )
        self.launch(velocity, self.n, [self.x, self.old, self.v, dt])
        self.grid.build(self.x, self.h)
        self.launch(
            smooth_velocity,
            self.n,
            [self.x, self.v, self.tmp, self.grid.id, self.h, self.norm, viscosity],
        )
        self.v, self.tmp = self.tmp, self.v
        if moving_body:
            self.launch(
                body_velocity,
                1,
                [
                    self.center,
                    self.rotation,
                    self.bc_old,
                    self.bq_old,
                    self.bv,
                    self.bw,
                    dt,
                ],
            )
        self.steps += 1

    def snapshot(self):
        return dict(
            positions=self.x.numpy(),
            velocities=self.v.numpy(),
            density=self.rho.numpy(),
            body_center=self.center.numpy()[0],
            body_quat=self.rotation.numpy()[0],
        )
