"""Thermoviscoplastic MLS-MPM with split volumetric/deviatoric stress.

Temperature is transported conservatively on the MPM grid. Cooling changes
Maxwell relaxation and the yield surface; scene colliders remain rigid.
"""

import math
import numpy as np
import torch
import warp as wp
from snowy_gs.mpm import weights, sample, normal, contact


@wp.func
def viscosity(temperature: float, eta_hot: float):
    return eta_hot * wp.exp(
        wp.clamp(11000.0 * (1.0 / wp.max(temperature, 650.0) - 1.0 / 1450.0), -2.0, 6.0)
    )


@wp.kernel(enable_backward=False)
def p2g(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    c: wp.array(dtype=wp.mat33),
    fe: wp.array(dtype=wp.mat33),
    j: wp.array(dtype=float),
    temp: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    momentum: wp.array3d(dtype=wp.vec3),
    heat: wp.array3d(dtype=float),
    volume_moment: wp.array3d(dtype=float),
    o: wp.vec3,
    dx: float,
    dt: float,
    rho: float,
    volume: float,
    bulk: float,
    mu: float,
):
    p = wp.tid()
    g = (x[p] - o) / dx
    b = wp.vec3i(
        int(wp.floor(g[0] - 0.5)), int(wp.floor(g[1] - 0.5)), int(wp.floor(g[2] - 0.5))
    )
    f = g - wp.vec3(float(b[0]), float(b[1]), float(b[2]))
    wx, wy, wz = weights(f[0]), weights(f[1]), weights(f[2])
    eye = wp.identity(n=3, dtype=float)
    bb = fe[p] * wp.transpose(fe[p])
    # W_vol = 0.5 K (log J)^2 gives increasing resistance to compression.
    tau = mu * (bb - wp.trace(bb) / 3.0 * eye) + bulk * wp.log(j[p]) * eye
    m = rho * volume
    affine = m * c[p] - dt * volume * 4.0 / (dx * dx) * tau
    for a in range(3):
        for d in range(3):
            for e in range(3):
                ix, iy, iz = b[0] + a, b[1] + d, b[2] + e
                w = wx[a] * wy[d] * wz[e]
                delta = (wp.vec3(float(a), float(d), float(e)) - f) * dx
                wp.atomic_add(mass, ix, iy, iz, w * m)
                wp.atomic_add(momentum, ix, iy, iz, w * (m * v[p] + affine * delta))
                wp.atomic_add(heat, ix, iy, iz, w * m * temp[p])
                wp.atomic_add(volume_moment, ix, iy, iz, w * m * j[p])


@wp.kernel(enable_backward=False)
def grid(
    mass: wp.array3d(dtype=float),
    momentum: wp.array3d(dtype=wp.vec3),
    heat: wp.array3d(dtype=float),
    gv: wp.array3d(dtype=wp.vec3),
    gt: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    o: wp.vec3,
    dx: float,
    dt: float,
    gravity: wp.vec3,
    friction: float,
    stick: float,
):
    i, j, k = wp.tid()
    vel = wp.vec3(0.0)
    t = float(300.0)
    if mass[i, j, k] > 1.0e-9:
        vel = momentum[i, j, k] / mass[i, j, k] + dt * gravity
        t = heat[i, j, k] / mass[i, j, k]
        pos = o + dx * wp.vec3(float(i), float(j), float(k))
        sd = phi[i, j, k]
        if sd < dx * 0.45:
            n = normal(phi, o, dx, pos)
            vel = contact(vel, n, friction)
            vel *= wp.exp(-stick * dt * wp.clamp(1.0 - sd / (dx * 0.45), 0.0, 1.0))
        if i < 3 and vel[0] < 0.0:
            vel[0] = 0.0
        if j < 3 and vel[1] < 0.0:
            vel[1] = 0.0
        if k < 3 and vel[2] < 0.0:
            vel[2] = 0.0
        if i > mass.shape[0] - 4 and vel[0] > 0.0:
            vel[0] = 0.0
        if j > mass.shape[1] - 4 and vel[1] > 0.0:
            vel[1] = 0.0
        if k > mass.shape[2] - 4 and vel[2] > 0.0:
            vel[2] = 0.0
    gv[i, j, k] = vel
    gt[i, j, k] = t


@wp.kernel(enable_backward=False)
def g2p(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    c: wp.array(dtype=wp.mat33),
    fe: wp.array(dtype=wp.mat33),
    j: wp.array(dtype=float),
    temp: wp.array(dtype=float),
    skin_age: wp.array(dtype=float),
    gv: wp.array3d(dtype=wp.vec3),
    mass: wp.array3d(dtype=float),
    volume_moment: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    o: wp.vec3,
    dx: float,
    dt: float,
    mu: float,
    eta: float,
    yield_hot: float,
    friction: float,
    volume_relaxation: float,
):
    p = wp.tid()
    g = (x[p] - o) / dx
    b = wp.vec3i(
        int(wp.floor(g[0] - 0.5)), int(wp.floor(g[1] - 0.5)), int(wp.floor(g[2] - 0.5))
    )
    f = g - wp.vec3(float(b[0]), float(b[1]), float(b[2]))
    wx, wy, wz = weights(f[0]), weights(f[1]), weights(f[2])
    vel = wp.vec3(0.0)
    affine = wp.mat33(0.0)
    projected_j = float(0.0)
    for a in range(3):
        for d in range(3):
            for e in range(3):
                vv = gv[b[0] + a, b[1] + d, b[2] + e]
                w = wx[a] * wy[d] * wz[e]
                delta = (wp.vec3(float(a), float(d), float(e)) - f) * dx
                vel += w * vv
                affine += 4.0 / (dx * dx) * w * wp.outer(vv, delta)
                gm = mass[b[0] + a, b[1] + d, b[2] + e]
                if gm > 1.0e-12:
                    projected_j += w * volume_moment[b[0] + a, b[1] + d, b[2] + e] / gm
    point = x[p] + dt * vel
    sd = sample(phi, o, dx, point)
    if sd < dx * 0.10:
        n = normal(phi, o, dx, point)
        point += (dx * 0.10 - sd) * n
        vel = contact(vel, n, friction)
    x[p] = point
    v[p] = vel
    c[p] = affine
    # Surface renewal is an appearance state: stretching/shearing exposes
    # fresh material, while resting surfaces develop a thicker cooled skin.
    symmetric = 0.5 * (affine + wp.transpose(affine))
    deviatoric = symmetric - wp.trace(symmetric) / 3.0 * wp.identity(n=3, dtype=float)
    strain_rate = wp.sqrt(wp.trace(deviatoric * wp.transpose(deviatoric)))
    skin_age[p] = (skin_age[p] + dt) * wp.exp(-1.5 * dt * strain_rate)
    # Volume and isochoric shape are updated separately. Scalar J remains
    # positive under the exponential discrete continuity update.
    # Mass-weighted P2G/G2P averaging removes unresolved pressure modes.
    # Its partition-of-unity weights preserve the sum of particle volumes
    # before the continuity update; no per-particle volume clamp is used.
    blend = float(0.0)
    if volume_relaxation > 0.0:
        blend = 1.0 - wp.exp(-dt / volume_relaxation)
    j[p] = ((1.0 - blend) * j[p] + blend * projected_j) * wp.exp(dt * wp.trace(affine))
    trial = (wp.identity(n=3, dtype=float) + dt * affine) * fe[p]
    u = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    sv = wp.mat33(0.0)
    wp.svd3(trial, u, sig, sv)
    log_s = wp.vec3(
        wp.log(wp.max(sig[0], 1.0e-6)),
        wp.log(wp.max(sig[1], 1.0e-6)),
        wp.log(wp.max(sig[2], 1.0e-6)),
    )
    dev = log_s - wp.vec3((log_s[0] + log_s[1] + log_s[2]) / 3.0)
    dev *= wp.exp(-mu * dt / viscosity(temp[p], eta))
    cold = wp.clamp((1350.0 - temp[p]) / 350.0, 0.0, 1.0)
    yield_stress = yield_hot + 18000.0 * cold * cold
    limit = yield_stress / (2.0 * mu)
    dev *= wp.min(1.0, limit / wp.max(wp.length(dev), 1.0e-8))
    fe[p] = (
        u
        * wp.diag(wp.vec3(wp.exp(dev[0]), wp.exp(dev[1]), wp.exp(dev[2])))
        * wp.transpose(sv)
    )


@wp.kernel(enable_backward=False)
def thermal(
    mass: wp.array3d(dtype=float),
    initial: wp.array3d(dtype=float),
    current: wp.array3d(dtype=float),
    updated: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    dt: float,
    dx: float,
    rho: float,
    cp: float,
    conductivity: float,
    ambient: float,
    ground: float,
    exchange: float,
):
    i, j, k = wp.tid()
    m = mass[i, j, k]
    value = float(ambient)
    if m > 1.0e-9:
        diagonal = m * cp
        rhs = diagonal * initial[i, j, k]
        fill = wp.min(1.0, m / (rho * dx * dx * dx))
        for side in range(6):
            ii, jj, kk = i, j, k
            if side == 0:
                ii -= 1
            if side == 1:
                ii += 1
            if side == 2:
                jj -= 1
            if side == 3:
                jj += 1
            if side == 4:
                kk -= 1
            if side == 5:
                kk += 1
            if (
                ii >= 0
                and jj >= 0
                and kk >= 0
                and ii < mass.shape[0]
                and jj < mass.shape[1]
                and kk < mass.shape[2]
            ):
                m2 = mass[ii, jj, kk]
                if m2 > 1.0e-9:
                    fill2 = wp.min(1.0, m2 / (rho * dx * dx * dx))
                    conduct = (
                        dt
                        * conductivity
                        * dx
                        * 2.0
                        * fill
                        * fill2
                        / wp.max(fill + fill2, 1.0e-9)
                    )
                    diagonal += conduct
                    rhs += conduct * current[ii, jj, kk]
                else:
                    # Linearize radiation at the old temperature to keep the
                    # implicit boundary exchange positive and monotone.
                    target = ambient
                    h = float(18.0)
                    if phi[ii, jj, kk] < dx * 0.1:
                        target = ground
                        h = 220.0
                    t = initial[i, j, k]
                    h += (
                        0.92 * 5.670374419e-8 * (t + target) * (t * t + target * target)
                    )
                    conduct = dt * exchange * h * dx * dx * fill
                    diagonal += conduct
                    rhs += conduct * target
        value = rhs / diagonal
    updated[i, j, k] = value


@wp.kernel(enable_backward=False)
def temperature_g2p(
    x: wp.array(dtype=wp.vec3),
    temp: wp.array(dtype=float),
    old: wp.array3d(dtype=float),
    new: wp.array3d(dtype=float),
    o: wp.vec3,
    dx: float,
):
    p = wp.tid()
    g = (x[p] - o) / dx
    b = wp.vec3i(
        int(wp.floor(g[0] - 0.5)), int(wp.floor(g[1] - 0.5)), int(wp.floor(g[2] - 0.5))
    )
    f = g - wp.vec3(float(b[0]), float(b[1]), float(b[2]))
    wx, wy, wz = weights(f[0]), weights(f[1]), weights(f[2])
    delta = float(0.0)
    for a in range(3):
        for d in range(3):
            for e in range(3):
                i, j, k = b[0] + a, b[1] + d, b[2] + e
                delta += wx[a] * wy[d] * wz[e] * (new[i, j, k] - old[i, j, k])
    temp[p] = wp.max(300.0, temp[p] + delta)


class LavaMPM:
    def __init__(
        self,
        positions,
        temperature,
        geo,
        velocity=None,
        capacity=None,
        device="cuda:0",
        density=2600.0,
        bulk=4.0e5,
        shear=4.0e4,
        eta=2200.0,
        yield_hot=700.0,
        dt=0.0015,
        friction=0.32,
        exchange=20.0,
        conductivity=2.2,
        volume_relaxation=0.02,
        gravity=(0.0, 0.0, -9.81),
    ):
        wp.init()
        wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()), device=device)
        self.device = device
        self.origin = np.array(geo["origin"], "f")
        self.dx = float(geo["dx"])
        self.shape = geo["phi"].shape
        self.rho = density
        self.bulk = bulk
        self.mu = shear
        self.eta = eta
        self.yield_hot = yield_hot
        self.dt = dt
        self.friction = friction
        self.exchange = exchange
        self.conductivity = conductivity
        self.volume_relaxation = volume_relaxation
        self.gravity = gravity
        self.count = len(positions)
        self.initial_count = self.count
        self.capacity = capacity or self.count + 80000
        self.volume = (self.dx * 0.5) ** 3
        self.time = 0.0
        self.steps = 0
        self.injected = 0
        self.x = wp.zeros(self.capacity, dtype=wp.vec3, device=device)
        self.v = wp.zeros_like(self.x)
        self.reference = wp.zeros_like(self.x)
        self.c = wp.zeros(self.capacity, dtype=wp.mat33, device=device)
        self.fe = wp.array(
            np.tile(np.eye(3, dtype="f"), (self.capacity, 1, 1)),
            dtype=wp.mat33,
            device=device,
        )
        self.j = wp.ones(self.capacity, dtype=float, device=device)
        self.temp = wp.zeros_like(self.j)
        self.birth = wp.zeros_like(self.j)
        self.skin_age = wp.zeros_like(self.j)
        self.vol = wp.full(self.capacity, self.volume, dtype=float, device=device)
        self.phi = wp.array(np.asarray(geo["phi"], "f"), dtype=float, device=device)
        self.mass = wp.zeros(self.shape, dtype=float, device=device)
        self.momentum = wp.zeros(self.shape, dtype=wp.vec3, device=device)
        self.gv = wp.zeros_like(self.momentum)
        self.heat = wp.zeros_like(self.mass)
        self.volume_moment = wp.zeros_like(self.mass)
        self.gt = wp.zeros_like(self.mass)
        self.ta = wp.zeros_like(self.mass)
        self.tb = wp.zeros_like(self.mass)
        self.count = 0
        self.append(positions, temperature, velocity)
        self.injected = 0

    def append(self, positions, temperature, velocity=None):
        n = len(positions)
        if self.count + n > self.capacity:
            raise ValueError("Particle capacity exceeded")
        if n == 0:
            return
        sl = slice(self.count, self.count + n)
        p = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        wp.to_torch(self.x)[sl] = p
        wp.to_torch(self.reference)[sl] = p
        wp.to_torch(self.temp)[sl] = torch.as_tensor(
            temperature, dtype=torch.float32, device=self.device
        )
        if velocity is not None:
            wp.to_torch(self.v)[sl] = torch.as_tensor(
                velocity, dtype=torch.float32, device=self.device
            )
        wp.to_torch(self.birth)[sl] = self.time
        self.count += n
        self.injected += n

    def transfer(self, dt):
        self.mass.zero_()
        self.momentum.zero_()
        self.heat.zero_()
        self.volume_moment.zero_()
        wp.launch(
            p2g,
            self.count,
            [
                self.x,
                self.v,
                self.c,
                self.fe,
                self.j,
                self.temp,
                self.mass,
                self.momentum,
                self.heat,
                self.volume_moment,
                wp.vec3(*self.origin),
                self.dx,
                dt,
                self.rho,
                self.volume,
                self.bulk,
                self.mu,
            ],
            device=self.device,
        )
        wp.launch(
            grid,
            self.shape,
            [
                self.mass,
                self.momentum,
                self.heat,
                self.gv,
                self.gt,
                self.phi,
                wp.vec3(*self.origin),
                self.dx,
                dt,
                wp.vec3(*self.gravity),
                self.friction,
                2.0,
            ],
            device=self.device,
        )

    def advance(self, duration, emitter=None):
        speed = (
            float(
                torch.linalg.vector_norm(
                    wp.to_torch(self.v)[: self.count], dim=-1
                ).max()
            )
            if self.count
            else 0.0
        )
        min_j = float(wp.to_torch(self.j)[: self.count].min()) if self.count else 1.0
        acoustic_bulk = self.bulk * max(1.0, (1.0 - math.log(min_j)) / min_j)
        step = min(
            self.dt,
            0.22
            * self.dx
            / (speed + math.sqrt((acoustic_bulk + 4 * self.mu / 3) / self.rho)),
        )
        n = math.ceil(duration / step)
        dt = duration / n
        for _ in range(n):
            if emitter is not None:
                emitter.advance(self, dt)
            self.transfer(dt)
            wp.launch(
                g2p,
                self.count,
                [
                    self.x,
                    self.v,
                    self.c,
                    self.fe,
                    self.j,
                    self.temp,
                    self.skin_age,
                    self.gv,
                    self.mass,
                    self.volume_moment,
                    self.phi,
                    wp.vec3(*self.origin),
                    self.dx,
                    dt,
                    self.mu,
                    self.eta,
                    self.yield_hot,
                    self.friction,
                    self.volume_relaxation,
                ],
                device=self.device,
            )
            self.time += dt
            self.steps += 1
        # Re-rasterize at the updated positions before the thermal split.
        self.transfer(0.0)
        wp.copy(self.ta, self.gt)
        for _ in range(12):
            wp.launch(
                thermal,
                self.shape,
                [
                    self.mass,
                    self.gt,
                    self.ta,
                    self.tb,
                    self.phi,
                    duration,
                    self.dx,
                    self.rho,
                    1200.0,
                    self.conductivity,
                    300.0,
                    650.0,
                    self.exchange,
                ],
                device=self.device,
            )
            self.ta, self.tb = self.tb, self.ta
        wp.launch(
            temperature_g2p,
            self.count,
            [self.x, self.temp, self.gt, self.ta, wp.vec3(*self.origin), self.dx],
            device=self.device,
        )

    def state(self):
        n = self.count
        return dict(
            x=self.x.numpy()[:n],
            v=self.v.numpy()[:n],
            fe=self.fe.numpy()[:n],
            j=self.j.numpy()[:n],
            temperature=self.temp.numpy()[:n],
            birth=self.birth.numpy()[:n],
            skin_age=self.skin_age.numpy()[:n],
            reference=self.reference.numpy()[:n],
            time=np.float64(self.time),
        )

    def diagnostics(self):
        s = self.state()
        assert all(np.isfinite(v).all() for v in s.values())
        q = (s["x"] - self.origin) / self.dx
        assert (q > 1.5).all() and (q < np.array(self.shape) - 2.5).all(), (
            "Particle left MPM support"
        )
        assert s["j"].min() > 0.3 and s["j"].max() < 2.5, "Invalid volumetric strain"
        return dict(
            time=self.time,
            steps=self.steps,
            particles=self.count,
            injected=self.injected,
            mass_kg=self.count * self.volume * self.rho,
            min_j=float(s["j"].min()),
            max_j=float(s["j"].max()),
            mean_j=float(s["j"].mean()),
            temperature_min=float(s["temperature"].min()),
            temperature_max=float(s["temperature"].max()),
            temperature_mean=float(s["temperature"].mean()),
            max_speed=float(np.linalg.norm(s["v"], axis=-1).max()),
        )
