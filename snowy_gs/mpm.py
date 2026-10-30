"""Explicit quadratic MLS/APIC MPM with Stomakhin et al. (2013) snow.

Fe = U diag(s) Vt; plastic return clamps elastic principal stretches, stores
Jp *= det(Fe_trial)/det(Fe_return), and hardens Lamé parameters by exp(h(1-Jp)).
The grid receives -dt V0 (dPsi/dFe) Fe^T grad(w). All geometry is in metres.
"""

import math
import numpy as np
import torch
import warp as wp


@wp.func
def weights(f: float):
    return wp.vec3(
        0.5 * (1.5 - f) * (1.5 - f),
        0.75 - (f - 1.0) * (f - 1.0),
        0.5 * (f - 0.5) * (f - 0.5),
    )


@wp.func
def sample(phi: wp.array3d(dtype=float), o: wp.vec3, dx: float, x: wp.vec3):
    g = (x - o) / dx
    i = wp.vec3i(int(wp.floor(g[0])), int(wp.floor(g[1])), int(wp.floor(g[2])))
    f = g - wp.vec3(float(i[0]), float(i[1]), float(i[2]))
    value = float(10.0)
    if (
        i[0] >= 0
        and i[1] >= 0
        and i[2] >= 0
        and i[0] < phi.shape[0] - 1
        and i[1] < phi.shape[1] - 1
        and i[2] < phi.shape[2] - 1
    ):
        value = 0.0
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
                    value += wa * wb * wc * phi[i[0] + a, i[1] + b, i[2] + c]
    return value


@wp.func
def normal(phi: wp.array3d(dtype=float), o: wp.vec3, dx: float, x: wp.vec3):
    h = dx * 0.5
    n = wp.vec3(
        sample(phi, o, dx, x + wp.vec3(h, 0.0, 0.0))
        - sample(phi, o, dx, x - wp.vec3(h, 0.0, 0.0)),
        sample(phi, o, dx, x + wp.vec3(0.0, h, 0.0))
        - sample(phi, o, dx, x - wp.vec3(0.0, h, 0.0)),
        sample(phi, o, dx, x + wp.vec3(0.0, 0.0, h))
        - sample(phi, o, dx, x - wp.vec3(0.0, 0.0, h)),
    )
    if wp.length(n) < 1.0e-7:
        n = wp.vec3(0.0, 0.0, 1.0)
    return wp.normalize(n)


@wp.func
def contact(v: wp.vec3, n: wp.vec3, friction: float):
    vn = wp.dot(v, n)
    out = v
    if vn < 0.0:
        tang = v - vn * n
        out = tang * wp.max(0.0, 1.0 + friction * vn / wp.max(wp.length(tang), 1.0e-8))
    return out


@wp.kernel(enable_backward=False)
def p2g(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    c: wp.array(dtype=wp.mat33),
    fe: wp.array(dtype=wp.mat33),
    jp: wp.array(dtype=float),
    vol: wp.array(dtype=float),
    gm: wp.array3d(dtype=float),
    gp: wp.array3d(dtype=wp.vec3),
    origin: wp.vec3,
    dx: float,
    dt: float,
    rho: float,
    mu: float,
    lam: float,
    hardening: float,
):
    p = wp.tid()
    if vol[p] > 0.0:
        g = (x[p] - origin) / dx
        base = wp.vec3i(
            int(wp.floor(g[0] - 0.5)),
            int(wp.floor(g[1] - 0.5)),
            int(wp.floor(g[2] - 0.5)),
        )
        f = g - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
        wx, wy, wz = weights(f[0]), weights(f[1]), weights(f[2])
        u = wp.mat33(0.0)
        s = wp.vec3(0.0)
        vmat = wp.mat33(0.0)
        wp.svd3(fe[p], u, s, vmat)
        r = u * wp.transpose(vmat)
        je = wp.determinant(fe[p])
        h = wp.exp(hardening * (1.0 - jp[p]))
        stress = 2.0 * mu * h * (fe[p] - r) * wp.transpose(fe[p]) + lam * h * (
            je - 1.0
        ) * je * wp.identity(n=3, dtype=float)
        m = rho * vol[p]
        a = m * c[p] - dt * vol[p] * 4.0 / (dx * dx) * stress
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    ix, iy, iz = base[0] + i, base[1] + j, base[2] + k
                    if (
                        ix >= 0
                        and iy >= 0
                        and iz >= 0
                        and ix < gm.shape[0]
                        and iy < gm.shape[1]
                        and iz < gm.shape[2]
                    ):
                        w = wx[i] * wy[j] * wz[k]
                        d = (wp.vec3(float(i), float(j), float(k)) - f) * dx
                        wp.atomic_add(gm, ix, iy, iz, w * m)
                        wp.atomic_add(gp, ix, iy, iz, w * (m * v[p] + a * d))


@wp.kernel(enable_backward=False)
def grid(
    gm: wp.array3d(dtype=float),
    gp: wp.array3d(dtype=wp.vec3),
    gv: wp.array3d(dtype=wp.vec3),
    phi: wp.array3d(dtype=float),
    o: wp.vec3,
    dx: float,
    dt: float,
    friction: float,
    acceleration: wp.vec3,
):
    i, j, k = wp.tid()
    vel = wp.vec3(0.0)
    if gm[i, j, k] > 1.0e-12:
        vel = gp[i, j, k] / gm[i, j, k] + dt * acceleration
        x = o + dx * wp.vec3(float(i), float(j), float(k))
        if phi[i, j, k] < 0.0:
            vel = contact(vel, normal(phi, o, dx, x), friction)
        if i < 3 and vel[0] < 0.0:
            vel[0] = 0.0
        if j < 3 and vel[1] < 0.0:
            vel[1] = 0.0
        if k < 3 and vel[2] < 0.0:
            vel[2] = 0.0
        if i > gm.shape[0] - 4 and vel[0] > 0.0:
            vel[0] = 0.0
        if j > gm.shape[1] - 4 and vel[1] > 0.0:
            vel[1] = 0.0
        if k > gm.shape[2] - 4 and vel[2] > 0.0:
            vel[2] = 0.0
    gv[i, j, k] = vel


@wp.kernel(enable_backward=False)
def g2p(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    c: wp.array(dtype=wp.mat33),
    fe: wp.array(dtype=wp.mat33),
    ft: wp.array(dtype=wp.mat33),
    jp: wp.array(dtype=float),
    vol: wp.array(dtype=float),
    gv: wp.array3d(dtype=wp.vec3),
    phi: wp.array3d(dtype=float),
    origin: wp.vec3,
    dx: float,
    dt: float,
    theta_c: float,
    theta_s: float,
    friction: float,
):
    p = wp.tid()
    if vol[p] > 0.0:
        g = (x[p] - origin) / dx
        base = wp.vec3i(
            int(wp.floor(g[0] - 0.5)),
            int(wp.floor(g[1] - 0.5)),
            int(wp.floor(g[2] - 0.5)),
        )
        f = g - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
        wx, wy, wz = weights(f[0]), weights(f[1]), weights(f[2])
        vel = wp.vec3(0.0)
        affine = wp.mat33(0.0)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    ix, iy, iz = base[0] + i, base[1] + j, base[2] + k
                    if (
                        ix >= 0
                        and iy >= 0
                        and iz >= 0
                        and ix < gv.shape[0]
                        and iy < gv.shape[1]
                        and iz < gv.shape[2]
                    ):
                        w = wx[i] * wy[j] * wz[k]
                        d = (wp.vec3(float(i), float(j), float(k)) - f) * dx
                        vel += w * gv[ix, iy, iz]
                        affine += w * 4.0 / (dx * dx) * wp.outer(gv[ix, iy, iz], d)
        trial = (wp.identity(n=3, dtype=float) + dt * affine) * fe[p]
        u = wp.mat33(0.0)
        s = wp.vec3(0.0)
        vm = wp.mat33(0.0)
        wp.svd3(trial, u, s, vm)
        returned = wp.vec3(
            wp.clamp(s[0], 1.0 - theta_c, 1.0 + theta_s),
            wp.clamp(s[1], 1.0 - theta_c, 1.0 + theta_s),
            wp.clamp(s[2], 1.0 - theta_c, 1.0 + theta_s),
        )
        # Finite compaction / loose-powder density range. Detached, very thin
        # layers otherwise accumulate unlimited tensile plastic volume while
        # their stress has already decayed to zero.
        jp[p] = wp.clamp(
            jp[p] * (s[0] * s[1] * s[2]) / (returned[0] * returned[1] * returned[2]),
            0.6,
            2.0,
        )
        fe[p] = u * wp.diag(returned) * wp.transpose(vm)
        # Recondition the visual quadrature after plastic flow. The constitutive
        # law retains Fe and Jp; a voxel surface is reconstructed from mass.
        ft[p] = fe[p] * wp.pow(jp[p], 1.0 / 3.0)
        pos = x[p] + dt * vel
        for it in range(2):
            sd = sample(phi, origin, dx, pos)
            if sd < 0.002:
                n = normal(phi, origin, dx, pos)
                pos += (0.002 - sd) * n
                vel = contact(vel, n, friction)
        x[p] = pos
        v[p] = vel
        c[p] = affine


class SnowMPM:
    def __init__(
        self,
        positions,
        phi,
        origin,
        dx,
        volume,
        density=300.0,
        young=20000.0,
        poisson=0.2,
        hardening=10.0,
        theta_c=0.025,
        theta_s=0.0075,
        friction=0.45,
        dt=0.001,
        device="cuda:0",
        extra_capacity=12000,
    ):
        wp.init()
        wp.set_stream(
            wp.stream_from_torch(torch.cuda.current_stream(device)), device=device
        )
        self.device = device
        self.origin = np.asarray(origin, np.float32)
        self.dx = dx
        self.dt = dt
        self.density = density
        self.young = young
        self.hardening = hardening
        self.mu = young / (2 * (1 + poisson))
        self.lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        self.theta_c = theta_c
        self.theta_s = theta_s
        self.friction = friction
        self.count = len(positions)
        self.capacity = self.count + extra_capacity
        p = np.zeros((self.capacity, 3), "f")
        p[: self.count] = positions
        self.x = wp.array(p, dtype=wp.vec3, device=device)
        self.v = wp.zeros(self.capacity, dtype=wp.vec3, device=device)
        self.c = wp.zeros(self.capacity, dtype=wp.mat33, device=device)
        eye = np.tile(np.eye(3, dtype="f"), (self.capacity, 1, 1))
        self.fe = wp.array(eye, dtype=wp.mat33, device=device)
        self.ft = wp.array(eye, dtype=wp.mat33, device=device)
        self.jp = wp.ones(self.capacity, dtype=float, device=device)
        vols = np.zeros(self.capacity, "f")
        vols[: self.count] = volume
        self.vol = wp.array(vols, dtype=float, device=device)
        self.phi = wp.array(
            np.ascontiguousarray(phi, dtype="f"), dtype=float, device=device
        )
        self.shape = phi.shape
        self.gm = wp.zeros(phi.shape, dtype=float, device=device)
        self.gp = wp.zeros(phi.shape, dtype=wp.vec3, device=device)
        self.gv = wp.zeros(phi.shape, dtype=wp.vec3, device=device)
        self.time = 0.0
        self.steps = 0
        self.deposited = 0
        self.initial_mass = float(vols.sum() * density)

    def deposit(self, positions, velocity, volume):
        n = len(positions)
        if not n:
            return
        if self.count + n > self.capacity:
            raise RuntimeError("Increase MPM deposition capacity")
        sl = slice(self.count, self.count + n)
        wp.to_torch(self.x)[sl] = torch.as_tensor(positions, device=self.device)
        wp.to_torch(self.v)[sl] = torch.as_tensor(velocity, device=self.device)
        wp.to_torch(self.vol)[sl] = volume
        self.count += n
        self.deposited += n

    def advance(self, duration, acceleration=(0.0, 0.0, -9.81)):
        jp = wp.to_torch(self.jp)[: self.count]
        speed = float(wp.to_torch(self.v)[: self.count].norm(dim=-1).max())
        hard = math.exp(self.hardening * (1.0 - float(jp.min())))
        wave = math.sqrt((self.lam + 2 * self.mu) * hard / self.density)
        dt = min(self.dt, 0.25 * self.dx / (wave + speed + 1.0e-6))
        n = math.ceil(duration / dt)
        dt = duration / n
        for _ in range(n):
            self.gm.zero_()
            self.gp.zero_()
            wp.launch(
                p2g,
                self.count,
                [
                    self.x,
                    self.v,
                    self.c,
                    self.fe,
                    self.jp,
                    self.vol,
                    self.gm,
                    self.gp,
                    wp.vec3(*self.origin),
                    self.dx,
                    dt,
                    self.density,
                    self.mu,
                    self.lam,
                    self.hardening,
                ],
                device=self.device,
            )
            wp.launch(
                grid,
                self.shape,
                [
                    self.gm,
                    self.gp,
                    self.gv,
                    self.phi,
                    wp.vec3(*self.origin),
                    self.dx,
                    dt,
                    self.friction,
                    wp.vec3(*acceleration),
                ],
                device=self.device,
            )
            wp.launch(
                g2p,
                self.count,
                [
                    self.x,
                    self.v,
                    self.c,
                    self.fe,
                    self.ft,
                    self.jp,
                    self.vol,
                    self.gv,
                    self.phi,
                    wp.vec3(*self.origin),
                    self.dx,
                    dt,
                    self.theta_c,
                    self.theta_s,
                    self.friction,
                ],
                device=self.device,
            )
        self.time += duration
        self.steps += n

    def state(self):
        return {
            name: wp.to_torch(getattr(self, name))[: self.count]
            .detach()
            .cpu()
            .numpy()
            .copy()
            for name in ("x", "v", "c", "fe", "ft", "jp", "vol")
        }

    def diagnostics(self):
        s = self.state()
        sing = np.linalg.svd(s["fe"], compute_uv=False)
        if not all(np.isfinite(v).all() for v in s.values()) or (s["jp"] <= 0).any():
            raise FloatingPointError("Invalid snow state")
        if np.linalg.det(s["ft"]).min() <= 0:
            raise FloatingPointError("Snow deformation inverted")
        return dict(
            time=self.time,
            steps=self.steps,
            particles=self.count,
            deposited=self.deposited,
            mass_kg=float(s["vol"].sum() * self.density),
            initial_mass_kg=self.initial_mass,
            jp_min=float(s["jp"].min()),
            jp_max=float(s["jp"].max()),
            elastic_stretch_min=float(sing.min()),
            elastic_stretch_max=float(sing.max()),
            max_speed=float(np.linalg.norm(s["v"], axis=-1).max()),
        )
