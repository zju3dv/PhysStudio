"""Velocity-based D3Q27 central moments and conservative D3Q7 phase field.

Xiao et al. 2026, Sections 4.1, 4.3, and 4.4. The Warp implementation uses
runtime direction loops and graph replay, also used by OpenHOMELBM's 3D code.
"""

import warp as wp
from .common import (
    Grid,
    coord,
    neighbor,
    lattice,
    direction,
    weight,
    rho,
    nu,
    population,
    phase_direction,
    phase_weight,
)


@wp.kernel(enable_backward=False)
def initialize(g: Grid):
    n = wp.tid()
    for q in range(g.q):
        g.f[q, n] = population(q, g.u[n], wp.vec3(0.0), wp.mat33(0.0))
    for q in range(7):
        c = phase_direction(q)
        cv = wp.vec3(float(c[0]), float(c[1]), float(c[2]))
        g.h[q, n] = phase_weight(q) * g.phi[n] * (1.0 + 4.0 * wp.dot(cv, g.u[n]))
    g.eps[n] = 1.0
    g.eps_eff[n] = 1.0
    g.eps_old[n] = 1.0


@wp.kernel(enable_backward=False)
def gradients_forces(g: Grid):
    n = wp.tid()
    if g.wall[n] != 0:
        return
    pos = coord(g, n)
    phi = float(g.phi[n])
    eps = float(g.eps_eff[n])
    p = float(g.pressure[n])
    u = wp.vec3(g.u[n])
    grad = wp.vec3(0.0)
    ge = wp.vec3(0.0)
    gp = wp.vec3(0.0)
    du = wp.mat33(0.0)
    lap = float(0.0)
    for q in range(g.q):
        c, cv = lattice(q), direction(q)
        nb = neighbor(g, pos + c)
        nb2 = neighbor(g, pos + 2 * c)
        ph1, ph2, e1, p1, u1 = phi, phi, eps, p, wp.vec3(0.0)
        if nb >= 0:
            ph1 = float(g.phi[nb])
            e1 = float(g.eps_eff[nb])
            p1 = float(g.pressure[nb])
            u1 = wp.vec3(g.u[nb])
        if nb2 >= 0:
            ph2 = g.phi[nb2]
        w = weight(q)
        grad += w * cv * (2.0 * (ph1 - phi) + 0.5 * (4.0 * ph1 - ph2))
        lap += 6.0 * w * (ph1 - phi)
        ge += 3.0 * w * cv * (e1 - eps)
        gp += 3.0 * w * cv * (p1 - p)
        du += 3.0 * w * wp.outer(u1 - u, cv)
    ph = wp.clamp(phi, 0.0, 1.0)
    density = rho(g, ph)
    strain = du + wp.transpose(du)
    art = g.cd * g.cd * wp.sqrt(0.5 * wp.trace(strain * wp.transpose(strain)))
    viscosity = nu(g, ph)
    beta, kappa = 12.0 * g.sigma / g.interface, 1.5 * g.sigma * g.interface
    chem = 4.0 * beta * ph * (ph - 1.0) * (ph - 0.5) - kappa * lap
    force = wp.vec3(0.0, 0.0, -g.gravity) + chem * grad / density
    force += (viscosity + art) * (
        strain * ((1.0 - g.rho_air) * grad / density + ge / eps)
    )
    # Reference stabilization for the low-density gas phase.
    if ph < 0.1:
        gp *= 0.1
    force -= gp / density
    # Axis-aligned numerical flux correction from the released solver.
    fa = wp.vec3(0.0)
    for a in range(3):
        for sign in range(2):
            c = wp.vec3i(0)
            c[a] = 1 - 2 * sign
            nb = neighbor(g, pos + c)
            if nb >= 0:
                un = g.u[nb]
                speed = wp.abs(u[a] + un[a]) * 0.5
                if speed > 8.0 * viscosity:
                    fa += (un - u) * speed
    force += 0.5 * fa
    if g.coupled == 1:
        force -= g.drag[n] / (g.eps[n] * density)
    g.force[n] = force
    g.ut[n] = u + 0.5 * force
    g.viscosity[n] = viscosity + art
    g.gradphi[n] = grad
    g.gradeps[n] = ge
    source = float(0.0)
    if g.coupled == 1:
        source = -phi / eps * (wp.dot(u, ge) + eps - g.eps_old[n])
    g.source[n] = source
    total = phi * eps
    if g.retention == 1:
        total += g.bound_next[n]
    wp.atomic_add(g.totals, 1, wp.float64(total))
    wp.atomic_add(g.totals, 2, wp.float64(source * eps))


@wp.kernel(enable_backward=False)
def volume_control(g: Grid, initialize_reference: int):
    if initialize_reference == 1:
        g.totals[0] = g.totals[1]
    ratio = float(1.0)
    error = float(g.totals[0] - g.totals[1])
    source = float(g.totals[2])
    # The released reference enables the proportional controller after 4% drift.
    if wp.abs(source) > 1.0e-9 and wp.abs(error) > 0.04 * float(g.totals[0]):
        ratio = wp.clamp(error / source, -1.0, 1.0)
    g.control[0] = ratio
    g.totals[1] = wp.float64(0.0)
    g.totals[2] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def collide_flow(g: Grid):
    n = wp.tid()
    if g.wall[n] != 0:
        return
    pos = coord(g, n)
    u = g.ut[n]
    moment = wp.mat33(0.0)
    for q in range(g.q):
        c = direction(q) - u
        moment += g.f[q, n] * wp.outer(c, c)
    dev = moment - (wp.trace(moment) / 3.0) * wp.identity(n=3, dtype=float)
    dev *= 1.0 - 1.0 / (3.0 * g.viscosity[n] + 0.5)
    for q in range(g.q):
        value = population(q, u, g.force[n], dev)
        nb = neighbor(g, pos + lattice(q))
        if nb >= 0:
            g.fn[q, nb] = value
        else:
            g.fn[26 - q, n] = value


@wp.kernel(enable_backward=False)
def collide_phase(g: Grid, warmup: int):
    n = wp.tid()
    if g.wall[n] != 0:
        return
    pos = coord(g, n)
    phi = float(g.phi[n])
    u = wp.vec3(g.ut[n])
    if warmup == 1:
        u = wp.vec3(0.0)
    flux = wp.vec3(0.0)
    for q in range(7):
        c = phase_direction(q)
        flux += g.h[q, n] * wp.vec3(float(c[0]), float(c[1]), float(c[2]))
    rate = 1.0 / (4.0 * g.mobility + 0.5)
    compression = 4.0 * phi * (1.0 - phi) / g.interface
    normal = g.gradphi[n] / (wp.length(g.gradphi[n]) + 1.0e-14)
    transported = phi
    interface_force = compression * normal
    source = g.source[n] * g.control[0]
    if g.retention == 1:
        source -= wp.min(wp.max(phi + source, 0.0), g.absorbed[n] / g.eps_eff[n])
    if warmup == 1:
        source = 0.0
    if g.conservative == 1 and warmup == 0:
        # Transport physical pore water q=epsilon_hat*phi. Symmetric pore-link
        # conductances conserve water and keep constant phi at zero-flux
        # equilibrium across an abrupt porosity jump.
        transported = g.eps_eff[n] * phi
        # Link transmissivities below represent the porosity part of the
        # diffusion operator directly, including abrupt packed-sand boundaries.
        interface_force = g.eps_eff[n] * compression * normal
        source = -g.absorbed[n]
    correction = (1.0 - rate) * (flux - transported * u) + (
        1.0 - 0.5 * rate
    ) * 0.25 * interface_force
    for q in range(7):
        c = phase_direction(q)
        cv = wp.vec3(float(c[0]), float(c[1]), float(c[2]))
        value = phase_weight(q) * (
            transported * (1.0 + 4.0 * wp.dot(cv, u))
            + 4.0 * wp.dot(cv, correction)
            + source
        )
        nb = neighbor(g, pos + c)
        if nb >= 0:
            transmission = float(1.0)
            if g.conservative == 1 and warmup == 0:
                transmission = wp.min(1.0, g.eps_eff[nb] / g.eps_eff[n])
            wp.atomic_add(g.hn, q, nb, value * transmission)
            opposite = int(0)
            if q > 0:
                opposite = q + 1 - 2 * ((q - 1) % 2)
            wp.atomic_add(g.hn, opposite, n, value * (1.0 - transmission))
        else:
            opposite = int(0)
            if q > 0:
                opposite = q + 1 - 2 * ((q - 1) % 2)
            wp.atomic_add(g.hn, opposite, n, value)


@wp.kernel(enable_backward=False)
def macros(g: Grid, warmup: int):
    n = wp.tid()
    if g.wall[n] == 0:
        phi, u = float(0.0), wp.vec3(0.0)
        for q in range(7):
            g.h[q, n] = g.hn[q, n]
            phi += g.hn[q, n]
        if g.conservative == 1 and warmup == 0:
            phi /= g.eps_eff[n]
        g.phi[n] = phi
        if warmup == 0:
            for q in range(g.q):
                g.f[q, n] = g.fn[q, n]
                u += direction(q) * g.fn[q, n]
            g.u[n] = u


@wp.kernel(enable_backward=False)
def initialize_pore_water(g: Grid):
    n = wp.tid()
    if g.conservative == 1 and g.wall[n] == 0:
        for q in range(7):
            g.h[q, n] *= g.eps_eff[n]


@wp.kernel(enable_backward=False)
def refresh_phase(g: Grid):
    n = wp.tid()
    if g.conservative == 1 and g.wall[n] == 0:
        value = float(0.0)
        for q in range(7):
            value += g.h[q, n]
        g.phi[n] = value / g.eps_eff[n]


@wp.kernel(enable_backward=False)
def pressure_update(g: Grid):
    n = wp.tid()
    if g.wall[n] != 0:
        return
    p = coord(g, n)
    total = float(0.0)
    for q in range(g.q):
        total += g.f[q, n]
    value = g.pressure[n] + rho(g, g.phi[n]) / 3.0 * (total - 1.0)
    if g.coupled == 1:
        eps = float(g.eps_eff[n])
        vs = wp.vec3(g.vs[n])
        if wp.length(vs) > 0.3:
            vs = wp.normalize(vs) * 0.3
        grad, div = wp.vec3(0.0), float(0.0)
        for q in range(g.q):
            c = direction(q)
            nb = neighbor(g, p + lattice(q))
            e1, v1 = eps, wp.vec3(0.0)
            if nb >= 0:
                e1 = float(g.eps_eff[nb])
                v1 = wp.vec3(g.vs[nb])
            if wp.length(v1) > 0.3:
                v1 = wp.normalize(v1) * 0.3
            grad += 3.0 * weight(q) * c * (e1 - eps)
            div += 3.0 * weight(q) * wp.dot(c, v1 - vs)
        value += (
            rho(g, g.phi[n])
            / (3.0 * eps)
            * (wp.dot(vs - g.u[n], grad) - (1.0 - g.eps[n]) * div)
        )
    g.pressure_next[n] = value


@wp.kernel(enable_backward=False)
def pressure_filter(g: Grid):
    n = wp.tid()
    if g.wall[n] != 0:
        return
    p = coord(g, n)
    value = float(0.0)
    for q in range(g.q):
        nb = neighbor(g, p + lattice(q))
        other = g.pressure_next[n]
        if nb >= 0:
            other = g.pressure_next[nb]
        value += weight(q) * other
    g.pressure[n] = value
