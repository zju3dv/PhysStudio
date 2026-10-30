"""Quadratic APIC, Hencky Drucker-Prager plasticity, and retained water.

The constitutive and retention updates extend the MIT-licensed 2D reference
by Xiaoyu Xiao to three dimensions. See LICENSE.reference.
"""

import warp as wp
from .common import Grid, Sand, weights, index, coord, rho


@wp.kernel(enable_backward=False)
def clear_grid(g: Grid):
    n = wp.tid()
    g.mass[n] = 0.0
    g.momentum[n] = wp.vec3(0.0)
    g.stressforce[n] = wp.vec3(0.0)
    g.bound[n] = 0.0
    g.capacity[n] = 0.0
    g.bound_next[n] = 0.0
    g.absorbed[n] = 0.0
    g.drag[n] = wp.vec3(0.0)


@wp.kernel(enable_backward=False)
def p2g(g: Grid, s: Sand):
    p = wp.tid()
    x = s.x[p]
    base = wp.vec3i(
        int(wp.floor(x[0] - 0.5)), int(wp.floor(x[1] - 0.5)), int(wp.floor(x[2] - 0.5))
    )
    if (
        base[0] < 0
        or base[1] < 0
        or base[2] < 0
        or base[0] > g.nx - 3
        or base[1] > g.ny - 3
        or base[2] > g.nz - 3
    ):
        return
    f = x - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    wx, wy, wz = weights(f[0]), weights(f[1]), weights(f[2])
    u, sig, v = wp.mat33(0.0), wp.vec3(0.0), wp.mat33(0.0)
    wp.svd3(s.fe[p], u, sig, v)
    e = wp.vec3(
        wp.log(wp.max(sig[0], 1.0e-8)),
        wp.log(wp.max(sig[1], 1.0e-8)),
        wp.log(wp.max(sig[2], 1.0e-8)),
    )
    tau = (
        u
        * wp.diag(2.0 * g.mu * e + g.lam * (e[0] + e[1] + e[2]) * wp.vec3(1.0))
        * wp.transpose(u)
    )
    force = -4.0 * g.pvol * tau
    m = g.rho_s * g.pvol
    for a in range(3):
        for b in range(3):
            for c in range(3):
                offset = wp.vec3i(a, b, c)
                n = index(g, base + offset)
                w = wx[a] * wy[b] * wz[c]
                d = wp.vec3(float(a), float(b), float(c)) - f
                wp.atomic_add(g.mass, n, w * m)
                wp.atomic_add(g.momentum, n, w * m * (s.v[p] + s.c[p] * d))
                wp.atomic_add(g.stressforce, n, w * (force * d))
                wp.atomic_add(g.bound, n, w * s.water[p])
                wp.atomic_add(
                    g.capacity, n, w * wp.max(0.0, 0.293 * g.pvol - s.water[p])
                )


@wp.kernel(enable_backward=False)
def grid_update(g: Grid):
    n = wp.tid()
    g.eps_old[n] = g.eps_eff[n]
    m = g.mass[n]
    vel = wp.vec3(0.0)
    delta = m / g.rho_s
    packed = wp.min(delta, 0.64)
    eps = 1.0 - packed
    g.delta[n] = delta
    g.eps[n] = eps
    g.eps_eff[n] = eps
    if m > 1.0e-10:
        vel = g.momentum[n] / m
        if g.retention == 1:
            old = g.bound[n]
            cap = packed * 0.293
            effective_old = wp.min(old, cap)
            free = (eps - effective_old) * wp.clamp(g.phi[n], 0.0, 1.0)
            absorbed = wp.max(
                0.0,
                wp.min(
                    wp.min(wp.min(cap - effective_old, g.absorb_rate * cap), free),
                    g.capacity[n],
                ),
            )
            g.absorbed[n] = absorbed
            g.bound_next[n] = old + absorbed
            g.eps_eff[n] = eps - wp.min(old + absorbed, cap)
        fd = wp.vec3(0.0)
        density = float(0.0)
        if g.coupled == 1:
            density = rho(g, g.phi[n])
            rel = g.u[n] - vel
            area = 3.141592653589793 * wp.pow(
                3.0 * delta / (4.0 * 3.141592653589793), 2.0 / 3.0
            )
            coeff = 0.5 * 0.39 * wp.pow(eps, -3.7) * density * area * wp.length(rel)
            # The reference treats a grid parcel as its drag representative.
            # Fix its physical length when comparing different grid sizes.
            coeff *= g.drag_scale
            # Pairwise backward-Euler drag avoids explicit overshoot in dilute cells.
            # The exact same impulse is applied with opposite sign to the fluid.
            coeff /= 1.0 + coeff * (1.0 / m + 1.0 / (eps * density))
            fd = coeff * rel
            g.drag[n] = fd
        vel += g.stressforce[n] / m + fd / m
        vel[2] -= g.gravity * (m - delta * density) / m
        pos = coord(g, n)
        for axis in range(3):
            bound = g.nx
            if axis == 1:
                bound = g.ny
            if axis == 2:
                bound = g.nz
            normal = wp.vec3(0.0)
            if pos[axis] < 3 and vel[axis] < 0.0:
                normal[axis] = 1.0
            if pos[axis] >= bound - 3 and vel[axis] > 0.0:
                normal[axis] = -1.0
            vn = wp.dot(vel, normal)
            if vn < 0.0:
                tangent = vel - vn * normal
                norm = wp.length(tangent)
                vel = tangent * wp.max(
                    0.0, 1.0 + g.friction * vn / wp.max(norm, 1.0e-12)
                )
    g.vs[n] = vel


@wp.kernel(enable_backward=False)
def g2p(g: Grid, s: Sand):
    p = wp.tid()
    x = s.x[p]
    base = wp.vec3i(
        int(wp.floor(x[0] - 0.5)), int(wp.floor(x[1] - 0.5)), int(wp.floor(x[2] - 0.5))
    )
    if (
        base[0] < 0
        or base[1] < 0
        or base[2] < 0
        or base[0] > g.nx - 3
        or base[1] > g.ny - 3
        or base[2] > g.nz - 3
    ):
        return
    f = x - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    wx, wy, wz = weights(f[0]), weights(f[1]), weights(f[2])
    vel, affine = wp.vec3(0.0), wp.mat33(0.0)
    moisture, water_delta = float(0.0), float(0.0)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                offset = wp.vec3i(a, b, c)
                n = index(g, base + offset)
                w = wx[a] * wy[b] * wz[c]
                d = wp.vec3(float(a), float(b), float(c)) - f
                vv = g.vs[n]
                vel += w * vv
                affine += 4.0 * w * wp.outer(vv, d)
                ph = wp.clamp(g.phi[n], 0.0, 1.0)
                moisture += w * (g.eps_eff[n] * ph + g.bound_next[n])
                water_delta += (
                    w
                    * g.absorbed[n]
                    * wp.max(0.0, 0.293 * g.pvol - s.water[p])
                    / wp.max(g.capacity[n], 1.0e-10)
                )
    s.x[p] = x + vel
    s.v[p] = vel
    s.c[p] = affine
    if g.retention == 1:
        s.water[p] = wp.clamp(s.water[p] + water_delta, 0.0, 0.293 * g.pvol)
    s.saturation[p] = moisture
    trial = (wp.identity(n=3, dtype=float) + affine) * s.fe[p]
    u, sig, v = wp.mat33(0.0), wp.vec3(0.0), wp.mat33(0.0)
    wp.svd3(trial, u, sig, v)
    e0 = wp.vec3(
        wp.log(wp.max(sig[0], 1.0e-8)),
        wp.log(wp.max(sig[1], 1.0e-8)),
        wp.log(wp.max(sig[2], 1.0e-8)),
    )
    cohesive = g.cohesion * wp.clamp(1.0 - moisture / 0.36, 0.0, 1.0)
    if g.retention == 1:
        value = wp.clamp(moisture, 0.0, 0.3)
        cohesive = 0.00382 + (0.00882 - 0.00382) * value / 0.108
        if value >= 0.108:
            cohesive = 0.00882 + (0.01 - 0.00882) * (value - 0.108) / (0.18752 - 0.108)
        if value >= 0.18752:
            cohesive = 0.01 * (1.0 - (value - 0.18752) / (0.3 - 0.18752))
        cohesive *= g.cohesion / 0.01
    e = e0 + (s.plastic[p] / 3.0 - cohesive / (3.0 * s.alpha[p])) * wp.vec3(1.0)
    trace = e[0] + e[1] + e[2]
    dev = e - trace / 3.0 * wp.vec3(1.0)
    norm = wp.length(dev)
    gamma = norm + (3.0 * g.lam + 2.0 * g.mu) / (2.0 * g.mu) * trace * s.alpha[p]
    e_new, dq = e0, float(0.0)
    if norm <= 1.0e-12 or trace > 0.0:
        e_new, dq = wp.vec3(0.0), wp.length(e)
    elif gamma > 0.0:
        e_new, dq = e - gamma / norm * dev, gamma
    projected = (
        u
        * wp.diag(wp.vec3(wp.exp(e_new[0]), wp.exp(e_new[1]), wp.exp(e_new[2])))
        * wp.transpose(v)
    )
    s.fe[p] = projected
    s.plastic[p] += (e0[0] + e0[1] + e0[2]) - (e_new[0] + e_new[1] + e_new[2])
    hard = s.hardening[p] + dq
    s.hardening[p] = hard
    angle = (35.0 + (9.0 * hard - 10.0) * wp.exp(-0.2 * hard)) * 0.017453292519943295
    sine = wp.sin(angle)
    s.alpha[p] = 0.816496580927726 * 2.0 * sine / (3.0 - sine)
