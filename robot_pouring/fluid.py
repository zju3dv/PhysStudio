"""Batched, incompressible particle water with moving hollow-vessel contact.

Positions use separated world origins for neighbor lookup. Every interaction
also checks the world id, so a hash collision cannot couple environments.
MuJoCo receives the opposite cup boundary impulse on the next substep.
"""
import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.struct
class Fluid:
    x: wp.array(dtype=wp.vec3)
    old: wp.array(dtype=wp.vec3)
    v: wp.array(dtype=wp.vec3)
    next_v: wp.array(dtype=wp.vec3)
    delta: wp.array(dtype=wp.vec3)
    absorbed_pos: wp.array(dtype=wp.vec3)
    lam: wp.array(dtype=float)
    rho: wp.array(dtype=float)
    status: wp.array(dtype=int)
    initial: wp.array(dtype=wp.vec3)
    cup_pos: wp.array(dtype=wp.vec3)
    cup_rot: wp.array(dtype=wp.mat33)
    prev_pos: wp.array(dtype=wp.vec3)
    prev_rot: wp.array(dtype=wp.mat33)
    target: wp.array(dtype=wp.vec3)
    force: wp.array(dtype=wp.vec3)
    torque: wp.array(dtype=wp.vec3)
    metrics: wp.array2d(dtype=float)
    mask: wp.array(dtype=bool)
    fill: wp.array(dtype=float)
    initial_count: wp.array(dtype=int)
    failures: wp.array(dtype=int)
    absorption_attempts: wp.array(dtype=int)
    grid: wp.uint64
    n: int
    h: float
    dx: float
    norm: float
    dt: float
    mass: float
    viscosity: float
    cup_radius: float
    cup_height: float
    wall: float
    target_radius: float
    target_height: float
    absorb: int


@wp.func
def origin(e: int):
    return wp.vec3(3.0 * float(e % 16), 3.0 * float(e // 16), 0.0)


@wp.func
def weight(r: wp.vec3, h: float):
    a = wp.max(1.0 - wp.dot(r, r) / (h * h), 0.0)
    return a * a * a


@wp.func
def gradient(r: wp.vec3, h: float, norm: float):
    length = wp.length(r)
    result = wp.vec3(0.0)
    if length > 1.e-9 and length < h:
        a = 1.0 - length / h
        # PBF uses the spiky pressure gradient to avoid the vanishing poly6
        # derivative at close range (particle pairing at a solid boundary).
        result = (-9.142857142857 * a * a / (h * norm * length)) * r
    return result


@wp.func
def box2(q: wp.vec2):
    return wp.length(wp.max(q, wp.vec2(0.0))) + wp.min(wp.max(q[0], q[1]), 0.0)


@wp.func
def vessel_distance(p: wp.vec3, radius: float, height: float, wall: float):
    r = wp.length(wp.vec2(p[0], p[1]))
    # Union of a cylindrical bottom and an annular wall, open at the top.
    floor = box2(wp.vec2(r - radius - wall, wp.abs(p[2] + wall / 2.0) - wall / 2.0))
    side = box2(wp.vec2(wp.abs(r - radius - wall / 2.0) - wall / 2.0,
                        wp.abs(p[2] - height / 2.0) - height / 2.0))
    return wp.min(floor, side)


@wp.func
def vessel_projection(p: wp.vec3, radius: float, height: float, wall: float, pr: float):
    out = wp.vec3(0.0)
    # At the inner bottom corner, two constraints can be active. Re-evaluate
    # after each correction instead of accepting one blended finite-difference
    # normal, which otherwise leaves a fraction of the particle radius inside.
    q = p
    for _ in range(3):
        d = vessel_distance(q, radius, height, wall)
        if d < pr:
            eps = float(0.00001)
            n = wp.vec3(
                vessel_distance(q + wp.vec3(eps, 0.0, 0.0), radius, height, wall) - vessel_distance(q - wp.vec3(eps, 0.0, 0.0), radius, height, wall),
                vessel_distance(q + wp.vec3(0.0, eps, 0.0), radius, height, wall) - vessel_distance(q - wp.vec3(0.0, eps, 0.0), radius, height, wall),
                vessel_distance(q + wp.vec3(0.0, 0.0, eps), radius, height, wall) - vessel_distance(q - wp.vec3(0.0, 0.0, eps), radius, height, wall))
            delta = (pr - d) * wp.normalize(n)
            out += delta
            q += delta
    return out


@wp.kernel
def copy_pose(g: Fluid, xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33), cup: int):
    e = wp.tid()
    g.prev_pos[e] = g.cup_pos[e]
    g.prev_rot[e] = g.cup_rot[e]
    g.cup_pos[e] = xpos[e, cup]
    g.cup_rot[e] = xmat[e, cup]


@wp.kernel
def reset_particles(g: Fluid):
    i = wp.tid()
    e, j = i // g.n, i % g.n
    if not g.mask[e]:
        return
    p = g.initial[j]
    g.status[i] = 0
    if p[2] > g.fill[e]:
        g.status[i] = 3
        p = wp.vec3(0.0, 0.0, -2.0)
    else:
        wp.atomic_add(g.initial_count, e, 1)
    g.x[i] = origin(e) + g.cup_pos[e] + g.cup_rot[e] * p
    g.old[i] = g.x[i]
    g.v[i] = wp.vec3(0.0)
    g.next_v[i] = wp.vec3(0.0)
    g.delta[i] = wp.vec3(0.0)
    g.absorbed_pos[i] = wp.vec3(0.0)
    g.lam[i] = 0.0
    g.rho[i] = 0.0


@wp.kernel
def predict(g: Fluid):
    i = wp.tid()
    g.old[i] = g.x[i]
    if g.status[i] < 2:
        g.v[i] += g.dt * wp.vec3(0.0, 0.0, -9.81)
        g.x[i] += g.dt * g.v[i]


@wp.kernel
def density(g: Fluid):
    i = wp.tid()
    g.lam[i] = 0.0
    if g.status[i] >= 2:
        return
    p = g.x[i]
    den, g2 = float(0.0), float(0.0)
    grad = wp.vec3(0.0)
    for j in wp.hash_grid_query(g.grid, p, g.h):
        if j // g.n == i // g.n and g.status[j] < 2:
            r = p - g.x[j]
            den += weight(r, g.h) / g.norm
            gj = gradient(r, g.h, g.norm)
            grad += gj
            if j != i:
                g2 += wp.dot(gj, gj)
    g.rho[i] = den
    g.lam[i] = -wp.max(den - 1.0, 0.0) / (g2 + wp.dot(grad, grad) + 0.01 / (g.h * g.h))


@wp.kernel
def correct(g: Fluid):
    i = wp.tid()
    g.delta[i] = wp.vec3(0.0)
    if g.status[i] >= 2:
        return
    p, delta = g.x[i], wp.vec3(0.0)
    for j in wp.hash_grid_query(g.grid, p, g.h):
        if j != i and j // g.n == i // g.n and g.status[j] < 2:
            r = p - g.x[j]
            # Small PBF artificial-pressure term prevents lattice clumping.
            scorr = -0.0001 * g.h * g.h * wp.pow(weight(r, g.h) / 0.753571, 4.0)
            delta += (g.lam[i] + g.lam[j] + scorr) * gradient(r, g.h, g.norm)
            length = wp.length(r)
            if length > 1.e-9 and length < 0.65 * g.dx:
                delta += 0.25 * (0.65 * g.dx - length) * r / length
    length = wp.length(delta)
    if length > 0.35 * g.dx:
        delta *= 0.35 * g.dx / length
    g.delta[i] = delta


@wp.kernel
def project(g: Fluid, use_delta: int):
    i = wp.tid()
    if g.status[i] >= 2:
        return
    e = i // g.n
    p = g.x[i] - origin(e)
    if use_delta == 1:
        p += g.delta[i]
    pr = 0.46 * g.dx
    rot, center = g.cup_rot[e], g.cup_pos[e]
    local = wp.transpose(rot) * (p - center)
    delta = rot * vessel_projection(local, g.cup_radius, g.cup_height, g.wall, pr)
    p += delta
    impulse = -g.mass * delta / g.dt
    wp.atomic_add(g.force, e, impulse / g.dt)
    wp.atomic_add(g.torque, e, wp.cross(p - center, impulse / g.dt))
    target_local = p - g.target[e]
    p += vessel_projection(target_local, g.target_radius, g.target_height, 0.008, pr)
    p[2] = wp.max(p[2], pr)
    # The soil sink is localized to the pot interior; each particle is absorbed
    # once and remains in the mass ledger, rather than disappearing uncounted.
    t = p - g.target[e]
    if g.absorb == 1:
        rr = wp.length(wp.vec2(t[0], t[1]))
        if rr < 0.007 + pr and rr > 1.e-7 and t[2] >= 0.05 and t[2] < 0.27:
            p += (0.007 + pr - rr) * wp.vec3(t[0] / rr, t[1] / rr, 0.0)
        if g.status[i] == 0 and rr < g.target_radius - pr and t[2] < 0.05 + pr:
            attempt = wp.atomic_add(g.absorption_attempts, e, 1)
            if attempt < int(0.75 * float(g.initial_count[e])):
                g.status[i] = 2
                g.absorbed_pos[i] = p
                p = wp.vec3(0.0, 0.0, -2.0)
            else:
                p[2] = wp.max(p[2], g.target[e][2] + 0.05 + pr)
    g.x[i] = p + origin(e)


@wp.kernel
def velocity(g: Fluid):
    i = wp.tid()
    if g.status[i] >= 2:
        g.v[i] = wp.vec3(0.0)
        return
    v = (g.x[i] - g.old[i]) / g.dt
    # Dissipative floor slip, never applied to airborne water.
    if g.x[i][2] < 0.46 * g.dx + 1.e-5:
        v[0] *= 0.96
        v[1] *= 0.96
    g.v[i] = v
    if not wp.isfinite(v[0]) or not wp.isfinite(v[1]) or not wp.isfinite(v[2]) or wp.length(v) > 15.0:
        wp.atomic_add(g.failures, i // g.n, 1)


@wp.kernel
def viscosity(g: Fluid):
    i = wp.tid()
    v, delta = g.v[i], wp.vec3(0.0)
    if g.status[i] < 2:
        for j in wp.hash_grid_query(g.grid, g.x[i], g.h):
            if i // g.n == j // g.n and g.status[j] < 2:
                delta += weight(g.x[i] - g.x[j], g.h) / g.norm * (g.v[j] - v)
    g.next_v[i] = v + g.viscosity * delta


@wp.kernel
def classify(g: Fluid):
    i = wp.tid()
    e = i // g.n
    status = g.status[i]
    if status == 3:
        return
    p = g.x[i] - origin(e)
    q = wp.transpose(g.cup_rot[e]) * (p - g.cup_pos[e])
    t = p - g.target[e]
    inside_target = wp.length(wp.vec2(t[0], t[1])) < g.target_radius and t[2] >= 0.0 and t[2] < g.target_height
    inside_cup = wp.length(wp.vec2(q[0], q[1])) < g.cup_radius and q[2] >= 0.0 and q[2] < g.cup_height
    outside = wp.abs(p[0]) > 1.4 or wp.abs(p[1]) > 1.1 or p[2] < -0.05
    if status == 0 and ((p[2] < g.dx and not inside_target and not inside_cup) or outside):
        status = 1
        g.status[i] = 1
    # Columns: source, target, flight, spilled, absorbed, kinetic energy,
    # density sum, moving particle count. The first five partition initial mass.
    col = int(2)
    if status == 1:
        col = 3
    elif status == 2:
        col = 4
    elif inside_cup:
        col = 0
    elif inside_target:
        col = 1
    wp.atomic_add(g.metrics, e, col, 1.0)
    if status < 2:
        wp.atomic_add(g.metrics, e, 5, 0.5 * g.mass * wp.dot(g.v[i], g.v[i]))
        wp.atomic_add(g.metrics, e, 6, g.rho[i])
        wp.atomic_add(g.metrics, e, 7, 1.0)


@wp.kernel
def apply_wrench(g: Fluid, xfrc: wp.array2d(dtype=wp.spatial_vector), xipos: wp.array2d(dtype=wp.vec3), cup: int, enabled: int):
    e = wp.tid()
    f, t = wp.vec3(0.0), wp.vec3(0.0)
    if enabled == 1:
        f = g.force[e]
        # MuJoCo's external moment is about the body's inertial COM.
        t = g.torque[e] + wp.cross(g.cup_pos[e] - xipos[e, cup], f)
    xfrc[e, cup] = wp.spatial_vector(f[0], f[1], f[2], t[0], t[1], t[2])


class ParticleWater:
    def __init__(self, cfg):
        self.cfg = cfg
        dx = cfg.spacing
        axis = np.arange(-cfg.cup_radius, cfg.cup_radius + dx / 2, dx)
        z = np.arange(dx * .6, cfg.fill_height, dx)
        points = np.stack(np.meshgrid(axis, axis, z, indexing="ij"), -1).reshape(-1, 3)
        points = points[np.linalg.norm(points[:, :2], axis=1) < cfg.cup_radius - dx * .6].astype(np.float32)
        g = self.g = Fluid()
        g.n = len(points)
        n, b = g.n * cfg.num_envs, cfg.num_envs
        self.grid = wp.HashGrid(128, 128, 64, device=cfg.device)
        self.grid.reserve(n)
        g.grid = self.grid.id
        for name in ("x", "old", "v", "next_v", "delta", "absorbed_pos"):
            setattr(g, name, wp.zeros(n, dtype=wp.vec3, device=cfg.device))
        for name in ("lam", "rho"):
            setattr(g, name, wp.zeros(n, dtype=float, device=cfg.device))
        g.status = wp.zeros(n, dtype=int, device=cfg.device)
        g.initial = wp.array(points, dtype=wp.vec3, device=cfg.device)
        for name in ("cup_pos", "prev_pos", "target", "force", "torque"):
            setattr(g, name, wp.zeros(b, dtype=wp.vec3, device=cfg.device))
        for name in ("cup_rot", "prev_rot"):
            setattr(g, name, wp.zeros(b, dtype=wp.mat33, device=cfg.device))
        g.metrics = wp.zeros((b, 8), dtype=float, device=cfg.device)
        g.mask = wp.zeros(b, dtype=bool, device=cfg.device)
        g.fill = wp.zeros(b, dtype=float, device=cfg.device)
        g.initial_count = wp.zeros(b, dtype=int, device=cfg.device)
        g.failures = wp.zeros(b, dtype=int, device=cfg.device)
        g.absorption_attempts = wp.zeros(b, dtype=int, device=cfg.device)
        g.dx, g.h, g.dt = dx, 2.4 * dx, cfg.dt
        lattice = np.stack(np.meshgrid(*([np.arange(-3, 4)] * 3), indexing="ij"), -1).reshape(-1, 3) * dx
        g.norm = float(np.maximum(1 - np.sum(lattice ** 2, axis=1) / g.h ** 2, 0) ** 3 @ np.ones(len(lattice)))
        g.mass = cfg.density * dx ** 3
        g.viscosity = cfg.viscosity
        g.cup_radius, g.cup_height, g.wall = cfg.cup_radius, cfg.cup_height, cfg.cup_wall
        g.target_radius, g.target_height = cfg.target_radius, cfg.target_height
        g.absorb = int(cfg.task == "flower")
        self.count = n

    def step(self):
        g = self.g
        g.force.zero_(); g.torque.zero_()
        wp.launch(predict, self.count, [g], device=self.cfg.device)
        wp.launch(project, self.count, [g, 0], device=self.cfg.device)
        for _ in range(self.cfg.iterations):
            self.grid.build(g.x, g.h)
            wp.launch(density, self.count, [g], device=self.cfg.device)
            wp.launch(correct, self.count, [g], device=self.cfg.device)
            wp.launch(project, self.count, [g, 1], device=self.cfg.device)
        wp.launch(velocity, self.count, [g], device=self.cfg.device)
        wp.launch(viscosity, self.count, [g], device=self.cfg.device)
        wp.copy(g.v, g.next_v)

    def measure(self):
        self.g.metrics.zero_()
        wp.launch(classify, self.count, [self.g], device=self.cfg.device)
