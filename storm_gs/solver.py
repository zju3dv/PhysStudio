"""Staggered-grid moist vortices with terrain contact and transported dust.

Cyclogenesis supplies the moist-buoyancy, eddy-mixing, and latent-heat model.
These local showcases use a controlled rotating inflow and warm-cloud initial
conditions. The pressure solve, advection, condensation, and dust transport are
computed in three dimensions with Warp.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import warp as wp


@wp.struct
class Parameters:
    origin: wp.vec3
    dx: float
    nx: int
    ny: int
    nz: int
    height: float
    radius: float
    top_radius: float
    swirl: float
    updraft: float
    inflow: float
    drive: float
    humidity: float
    temperature: float
    dust_rate: float
    cloud: float
    lean: float
    eddy: float
    moisture_core: float
    moisture_rate: float


@wp.func
def sample(a: wp.array3d(dtype=float), p: wp.vec3):
    q = wp.vec3(
        wp.clamp(p[0], 0.0, float(a.shape[0] - 1) - 0.001),
        wp.clamp(p[1], 0.0, float(a.shape[1] - 1) - 0.001),
        wp.clamp(p[2], 0.0, float(a.shape[2] - 1) - 0.001),
    )
    i, j, k = int(wp.floor(q[0])), int(wp.floor(q[1])), int(wp.floor(q[2]))
    f = q - wp.vec3(float(i), float(j), float(k))
    return wp.lerp(
        wp.lerp(
            wp.lerp(a[i, j, k], a[i + 1, j, k], f[0]),
            wp.lerp(a[i, j + 1, k], a[i + 1, j + 1, k], f[0]),
            f[1],
        ),
        wp.lerp(
            wp.lerp(a[i, j, k + 1], a[i + 1, j, k + 1], f[0]),
            wp.lerp(a[i, j + 1, k + 1], a[i + 1, j + 1, k + 1], f[0]),
            f[1],
        ),
        f[2],
    )


@wp.func
def sample4(a: wp.array3d(dtype=wp.vec4), p: wp.vec3):
    q = wp.vec3(
        wp.clamp(p[0], 0.0, float(a.shape[0] - 1) - 0.001),
        wp.clamp(p[1], 0.0, float(a.shape[1] - 1) - 0.001),
        wp.clamp(p[2], 0.0, float(a.shape[2] - 1) - 0.001),
    )
    i, j, k = int(wp.floor(q[0])), int(wp.floor(q[1])), int(wp.floor(q[2]))
    f = q - wp.vec3(float(i), float(j), float(k))
    return wp.lerp(
        wp.lerp(
            wp.lerp(a[i, j, k], a[i + 1, j, k], f[0]),
            wp.lerp(a[i, j + 1, k], a[i + 1, j + 1, k], f[0]),
            f[1],
        ),
        wp.lerp(
            wp.lerp(a[i, j, k + 1], a[i + 1, j, k + 1], f[0]),
            wp.lerp(a[i, j + 1, k + 1], a[i + 1, j + 1, k + 1], f[0]),
            f[1],
        ),
        f[2],
    )


@wp.func
def sample3(a: wp.array3d(dtype=wp.vec3), p: wp.vec3):
    q = wp.vec3(
        wp.clamp(p[0], 0.0, float(a.shape[0] - 1) - 0.001),
        wp.clamp(p[1], 0.0, float(a.shape[1] - 1) - 0.001),
        wp.clamp(p[2], 0.0, float(a.shape[2] - 1) - 0.001),
    )
    i, j, k = int(wp.floor(q[0])), int(wp.floor(q[1])), int(wp.floor(q[2]))
    f = q - wp.vec3(float(i), float(j), float(k))
    return wp.lerp(
        wp.lerp(
            wp.lerp(a[i, j, k], a[i + 1, j, k], f[0]),
            wp.lerp(a[i, j + 1, k], a[i + 1, j + 1, k], f[0]),
            f[1],
        ),
        wp.lerp(
            wp.lerp(a[i, j, k + 1], a[i + 1, j, k + 1], f[0]),
            wp.lerp(a[i, j + 1, k + 1], a[i + 1, j + 1, k + 1], f[0]),
            f[1],
        ),
        f[2],
    )


@wp.func
def velocity(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    p: wp.vec3,
):
    return wp.vec3(
        sample(u, p - wp.vec3(0.0, 0.5, 0.5)),
        sample(v, p - wp.vec3(0.5, 0.0, 0.5)),
        sample(w, p - wp.vec3(0.5, 0.5, 0.0)),
    )


@wp.func
def fluid(s: wp.array3d(dtype=int), i: int, j: int, k: int):
    result = 0
    if (
        i >= 0
        and j >= 0
        and k >= 0
        and i < s.shape[0]
        and j < s.shape[1]
        and k < s.shape[2]
    ):
        result = 1 - s[i, j, k]
    return result


@wp.func
def core(p: wp.vec3, t: float, c: Parameters):
    z = wp.clamp(p[2] / c.height, 0.0, 1.0)
    cx = c.lean * z * z + 0.22 * wp.sin(z * 5.0 + t * 0.55) * z
    cy = 0.3 * wp.sin(z * 4.0 - t * 0.38) * z
    rr = c.radius + (c.top_radius - c.radius) * wp.pow(z, 2.2)
    return wp.vec3(cx, cy, rr)


@wp.func
def driver(p: wp.vec3, t: float, c: Parameters):
    shape = core(p, t, c)
    x = p[0] - shape[0]
    y = p[1] - shape[1]
    r = wp.sqrt(x * x + y * y + 0.0001)
    rc = shape[2]
    z = wp.clamp(p[2] / c.height, 0.0, 1.0)
    turning = 2.0 * c.swirl * rc / (r * r + rc * rc)
    # An axisymmetric streamfunction supplies a paired updraft and return flow.
    # Its radial and vertical components obey continuity before projection.
    eb = wp.exp(-wp.max(p[2], 0.0) / (c.radius * 1.5))
    et = wp.exp(-wp.max(c.height - p[2], 0.0) / (c.height * 0.12))
    envelope = (1.0 - eb) * (1.0 - et)
    derivative = eb / (c.radius * 1.5) * (1.0 - et) - (1.0 - eb) * et / (
        c.height * 0.12
    )
    radius_derivative = (c.top_radius - c.radius) * 2.2 / c.height * wp.pow(z, 1.2)
    gaussian = wp.exp(-r * r / (rc * rc))
    radial = (
        -0.5
        * c.updraft
        * r
        * gaussian
        * (derivative + envelope * 2.0 * r * r * radius_derivative / (rc * rc * rc))
    )
    inflow = radial / r - c.inflow * wp.exp(-z * 12.0) / (r * r + rc * rc)
    rise = c.updraft * (1.0 - r * r / (rc * rc)) * gaussian * envelope
    fade = 1.0 - wp.smoothstep(0.84, 1.02, z)
    turbulence = wp.vec3(
        wp.sin(p[1] * 1.7 + p[2] * 1.1 - t * 1.3),
        wp.sin(p[0] * 1.4 - p[2] * 0.8 + t * 1.1),
        wp.sin(p[0] * 1.8 + p[1] * 1.3 + t * 0.7),
    )
    return wp.vec3(
        -turning * y + inflow * x, turning * x + inflow * y, rise
    ) * fade + turbulence * (0.18 * fade)


@wp.func
def saturation(T: float, p: float):
    tc = T - 273.15
    es = 611.2 * wp.exp(17.67 * tc / (tc + 243.5))
    return 0.622 * es / wp.max(10000.0, p - es)


@wp.func
def phase_change(a: wp.vec4, p: float, dt: float):
    T, qv, qc, qr = a[0], a[1], a[2], a[3]
    qs = saturation(T, p)
    derivative = (saturation(T + 0.05, p) - saturation(T - 0.05, p)) / 0.1
    delta = (qv - qs) / (1.0 + 2490.04 * derivative) * (1.0 - wp.exp(-dt / 0.12))
    delta = wp.clamp(delta, -qc, qv)
    T += 2490.04 * delta
    qv -= delta
    qc += delta
    rain = wp.max(qc - 0.0012, 0.0) * (1.0 - wp.exp(-dt * 0.18))
    qc -= rain
    qr += rain
    evaporation = wp.min(
        qr, wp.max(saturation(T, p) - qv, 0.0) * (1.0 - wp.exp(-dt * 0.7))
    )
    T -= 2490.04 * evaporation
    qv += evaporation
    qr -= evaporation
    return wp.vec4(T, qv, qc, qr)


@wp.kernel
def initialize(
    a: wp.array3d(dtype=wp.vec4),
    dust: wp.array3d(dtype=float),
    energy: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    c: Parameters,
):
    i, j, k = wp.tid()
    p = c.origin + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * c.dx
    shape = core(p, 0.0, c)
    r = wp.length(wp.vec2(p[0] - shape[0], p[1] - shape[1]))
    pressure = 101325.0 * wp.exp(-p[2] / 8400.0)
    ambient = c.temperature - 0.0065 * p[2]
    vertical = 1.0 - wp.smoothstep(0.88, 1.03, p[2] / c.height)
    cold = c.cloud * 1.5 * wp.exp(-r * r / (shape[2] * shape[2] * 1.5)) * vertical
    cap = (
        c.cloud
        * wp.exp(-wp.pow((p[2] / c.height - 0.82) / 0.12, 2.0))
        * wp.exp(-wp.pow(r / (c.top_radius * 1.8), 4.0))
    )
    moist = c.humidity
    if c.cloud > 0.0:
        moist += (
            (c.moisture_core - c.humidity)
            * wp.exp(-r * r / (shape[2] * shape[2] * 1.7))
            * vertical
        )
    a[i, j, k] = wp.vec4(
        ambient - cold, moist * saturation(ambient, pressure), 0.0012 * cap, 0.0
    )
    dust[i, j, k] = 0.0
    energy[i, j, k] = 0.12


@wp.kernel
def initialize_velocity(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    c: Parameters,
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j), float(k))
    if fluid(solid, i - 1, j, k) * fluid(solid, i, j, k) == 1:
        u[i, j, k] = driver(c.origin + (p + wp.vec3(0.0, 0.5, 0.5)) * c.dx, 0.0, c)[0]
    if fluid(solid, i, j - 1, k) * fluid(solid, i, j, k) == 1:
        v[i, j, k] = driver(c.origin + (p + wp.vec3(0.5, 0.0, 0.5)) * c.dx, 0.0, c)[1]
    if fluid(solid, i, j, k - 1) * fluid(solid, i, j, k) == 1:
        w[i, j, k] = driver(c.origin + (p + wp.vec3(0.5, 0.5, 0.0)) * c.dx, 0.0, c)[2]


@wp.kernel
def curl_field(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    omega: wp.array3d(dtype=wp.vec3),
    magnitude: wp.array3d(dtype=float),
    c: Parameters,
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    ex, ey, ez = wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)
    gx = (velocity(u, v, w, p + ex) - velocity(u, v, w, p - ex)) / (2.0 * c.dx)
    gy = (velocity(u, v, w, p + ey) - velocity(u, v, w, p - ey)) / (2.0 * c.dx)
    gz = (velocity(u, v, w, p + ez) - velocity(u, v, w, p - ez)) / (2.0 * c.dx)
    o = wp.vec3(gy[2] - gz[1], gz[0] - gx[2], gx[1] - gy[0])
    omega[i, j, k] = o
    magnitude[i, j, k] = wp.length(o)


@wp.func
def forces(
    p: wp.vec3,
    world: wp.vec3,
    vel: wp.vec3,
    a: wp.array3d(dtype=wp.vec4),
    omega: wp.array3d(dtype=wp.vec3),
    mag: wp.array3d(dtype=float),
    t: float,
    c: Parameters,
):
    q = p - wp.vec3(0.5)
    s = sample4(a, q)
    ambient = c.temperature - 0.0065 * world[2]
    q0 = c.humidity * saturation(ambient, 101325.0 * wp.exp(-world[2] / 8400.0))
    buoy = 9.81 * ((s[0] - ambient) / ambient + 0.61 * (s[1] - q0) - s[2] - s[3])
    gradient = wp.vec3(
        sample(mag, q + wp.vec3(1.0, 0.0, 0.0))
        - sample(mag, q - wp.vec3(1.0, 0.0, 0.0)),
        sample(mag, q + wp.vec3(0.0, 1.0, 0.0))
        - sample(mag, q - wp.vec3(0.0, 1.0, 0.0)),
        sample(mag, q + wp.vec3(0.0, 0.0, 1.0))
        - sample(mag, q - wp.vec3(0.0, 0.0, 1.0)),
    )
    confinement = 0.22 * c.dx * wp.cross(wp.normalize(gradient), sample3(omega, q))
    return (driver(world, t, c) - vel) * c.drive + wp.vec3(0.0, 0.0, buoy) + confinement


@wp.kernel
def advect_velocity(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    un: wp.array3d(dtype=float),
    vn: wp.array3d(dtype=float),
    wn: wp.array3d(dtype=float),
    a: wp.array3d(dtype=wp.vec4),
    energy: wp.array3d(dtype=float),
    omega: wp.array3d(dtype=wp.vec3),
    mag: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    c: Parameters,
    dt: float,
    t: float,
):
    i, j, k = wp.tid()
    base = wp.vec3(float(i), float(j), float(k))
    for axis in range(3):
        offset = wp.vec3(0.5)
        offset[axis] = 0.0
        p = base + offset
        current = velocity(u, v, w, p)
        midpoint = p - current * (0.5 * dt / c.dx)
        previous = velocity(u, v, w, p - velocity(u, v, w, midpoint) * (dt / c.dx))
        world = c.origin + p * c.dx
        acc = forces(p, world, current, a, omega, mag, t, c)
        nu = 0.2 * c.dx * wp.sqrt(wp.max(sample(energy, p - wp.vec3(0.5)), 0.0))
        lap = (
            velocity(u, v, w, p + wp.vec3(1.0, 0.0, 0.0))
            + velocity(u, v, w, p - wp.vec3(1.0, 0.0, 0.0))
            + velocity(u, v, w, p + wp.vec3(0.0, 1.0, 0.0))
            + velocity(u, v, w, p - wp.vec3(0.0, 1.0, 0.0))
            + velocity(u, v, w, p + wp.vec3(0.0, 0.0, 1.0))
            + velocity(u, v, w, p - wp.vec3(0.0, 0.0, 1.0))
            - 6.0 * current
        )
        value = previous[axis] + dt * (acc[axis] + nu * lap[axis] / (c.dx * c.dx))
        if axis == 0:
            if fluid(solid, i - 1, j, k) * fluid(solid, i, j, k) == 0:
                value = 0.0
            un[i, j, k] = value
        elif axis == 1:
            if fluid(solid, i, j - 1, k) * fluid(solid, i, j, k) == 0:
                value = 0.0
            vn[i, j, k] = value
        else:
            if fluid(solid, i, j, k - 1) * fluid(solid, i, j, k) == 0:
                value = 0.0
            wn[i, j, k] = value


@wp.kernel
def divergence(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    div: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    dx: float,
):
    i, j, k = wp.tid()
    div[i, j, k] = (
        (
            u[i + 1, j, k]
            - u[i, j, k]
            + v[i, j + 1, k]
            - v[i, j, k]
            + w[i, j, k + 1]
            - w[i, j, k]
        )
        / dx
        * float(1 - solid[i, j, k])
    )


@wp.kernel
def pressure_sor(
    p: wp.array3d(dtype=float),
    div: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    dx: float,
    color: int,
):
    i, j, k = wp.tid()
    if (i + j + k) % 2 == color and solid[i, j, k] == 0:
        total = float(0.0)
        count = int(0)
        for axis in range(3):
            for sign in range(2):
                q = wp.vec3i(i, j, k)
                q[axis] += 2 * sign - 1
                if fluid(solid, q[0], q[1], q[2]) == 1:
                    total += p[q[0], q[1], q[2]]
                    count += 1
        if count > 0:
            value = (total - div[i, j, k] * dx * dx) / float(count)
            p[i, j, k] += 1.6 * (value - p[i, j, k])


@wp.kernel
def project(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    p: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    dx: float,
):
    i, j, k = wp.tid()
    if fluid(solid, i - 1, j, k) * fluid(solid, i, j, k) == 1:
        u[i, j, k] -= (p[i, j, k] - p[i - 1, j, k]) / dx
    if fluid(solid, i, j - 1, k) * fluid(solid, i, j, k) == 1:
        v[i, j, k] -= (p[i, j, k] - p[i, j - 1, k]) / dx
    if fluid(solid, i, j, k - 1) * fluid(solid, i, j, k) == 1:
        w[i, j, k] -= (p[i, j, k] - p[i, j, k - 1]) / dx


@wp.kernel
def advect_dust(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    src: wp.array3d(dtype=float),
    dst: wp.array3d(dtype=float),
    c: Parameters,
    dt: float,
    settling: float,
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    current = velocity(u, v, w, p) - wp.vec3(0.0, 0.0, settling)
    mid = p - current * (0.5 * dt / c.dx)
    back = (
        p
        - (velocity(u, v, w, mid) - wp.vec3(0.0, 0.0, settling)) * (dt / c.dx)
        - wp.vec3(0.5)
    )
    dst[i, j, k] = sample(src, back)


@wp.kernel
def correct_dust(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    src: wp.array3d(dtype=float),
    forward: wp.array3d(dtype=float),
    dst: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    c: Parameters,
    dt: float,
    settling: float,
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    current = velocity(u, v, w, p) - wp.vec3(0.0, 0.0, settling)
    mid = p + current * (0.5 * dt / c.dx)
    reverse = (
        p
        + (velocity(u, v, w, mid) - wp.vec3(0.0, 0.0, settling)) * (dt / c.dx)
        - wp.vec3(0.5)
    )
    result = forward[i, j, k] + 0.5 * (src[i, j, k] - sample(forward, reverse))
    mid = p - current * (0.5 * dt / c.dx)
    back = (
        p
        - (velocity(u, v, w, mid) - wp.vec3(0.0, 0.0, settling)) * (dt / c.dx)
        - wp.vec3(0.5)
    )
    lo = float(1.0e20)
    hi = float(-1.0e20)
    cell = wp.vec3i(
        int(wp.floor(back[0])), int(wp.floor(back[1])), int(wp.floor(back[2]))
    )
    for x in range(2):
        for y in range(2):
            for z in range(2):
                value = src[
                    wp.clamp(cell[0] + x, 0, c.nx - 1),
                    wp.clamp(cell[1] + y, 0, c.ny - 1),
                    wp.clamp(cell[2] + z, 0, c.nz - 1),
                ]
                lo = wp.min(lo, value)
                hi = wp.max(hi, value)
    dst[i, j, k] = wp.clamp(result, lo, hi) * float(1 - solid[i, j, k])


@wp.kernel
def transport(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    a: wp.array3d(dtype=wp.vec4),
    an: wp.array3d(dtype=wp.vec4),
    dust: wp.array3d(dtype=float),
    dust_advected: wp.array3d(dtype=float),
    dn: wp.array3d(dtype=float),
    energy: wp.array3d(dtype=float),
    en: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    terrain: wp.array2d(dtype=float),
    c: Parameters,
    dt: float,
    t: float,
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    world = c.origin + p * c.dx
    vel = velocity(u, v, w, p)
    q = p - velocity(u, v, w, p - vel * (0.5 * dt / c.dx)) * (dt / c.dx) - wp.vec3(0.5)
    s = sample4(a, q)
    e = wp.max(sample(energy, q), 0.0001)
    nu = 0.2 * c.dx * wp.sqrt(e)
    mix = 3.0 * c.eddy * nu
    lap = wp.vec4(0.0)
    lapd = float(0.0)
    center = a[i, j, k]
    for axis in range(3):
        for sign in range(2):
            at = wp.vec3i(i, j, k)
            at[axis] += 2 * sign - 1
            if fluid(solid, at[0], at[1], at[2]) == 1:
                lap += a[at[0], at[1], at[2]] - center
                lapd += dust[at[0], at[1], at[2]] - dust[i, j, k]
    s += mix * dt / (c.dx * c.dx) * lap
    # Rain and dust settle relative to the staggered air velocity.
    s[3] = sample4(a, q + wp.vec3(0.0, 0.0, 5.0 * dt / c.dx))[3]
    d = dust_advected[i, j, k] + mix * dt / (c.dx * c.dx) * lapd
    zground = world[2] - terrain[i, j]
    shape = core(world, t, c)
    r = wp.length(wp.vec2(world[0] - shape[0], world[1] - shape[1]))
    pressure = 101325.0 * wp.exp(-world[2] / 8400.0)
    ambient = c.temperature - 0.0065 * world[2]
    cap = (
        c.cloud
        * wp.exp(-wp.pow((world[2] / c.height - 0.82) / 0.13, 2.0))
        * wp.exp(-wp.pow(r / (c.top_radius * 1.8), 4.0))
    )
    # Ambient upper cloud supply represents the parent storm outside this crop.
    feed = wp.min(dt * 0.8 * cap, 1.0)
    s[2] = wp.lerp(wp.max(s[2], 0.0), 0.0012 * cap, feed)
    moist = c.humidity
    vertical = 1.0 - wp.smoothstep(0.88, 1.03, world[2] / c.height)
    core_weight = float(0.0)
    if c.cloud > 0.0:
        core_weight = wp.exp(-r * r / (shape[2] * shape[2] * 1.7)) * vertical
        moist += (c.moisture_core - c.humidity) * core_weight
    s[1] = wp.lerp(
        wp.max(s[1], 0.0),
        moist * saturation(ambient, pressure),
        1.0 - wp.exp(-dt * (0.3 + c.moisture_rate * core_weight)),
    )
    # Cyclostrophic pressure cooling is restricted to the rotating moist core.
    cool = (
        c.cloud
        * c.swirl
        * c.swirl
        / (1004.0 * (1.0 + wp.pow(r / shape[2], 2.0)))
        * vertical
    )
    s[0] = wp.lerp(s[0], ambient - cool, wp.min(dt * 0.5, 1.0))
    s = phase_change(s, pressure, dt)
    source = (
        c.dust_rate
        * wp.exp(-wp.pow(r / (c.radius * 3.0 + 0.5), 2.0))
        * wp.exp(-wp.pow(zground / (c.dx * 2.0), 2.0))
        * wp.max(wp.length(wp.vec2(vel[0], vel[1])) - 2.5, 0.0)
    )
    d = wp.max(0.0, d + dt * source - dt * d * 0.025)
    # A one-equation mixing-length closure with production and dissipation.
    shear = wp.length(
        velocity(u, v, w, p + wp.vec3(0.0, 0.0, 1.0))
        - velocity(u, v, w, p - wp.vec3(0.0, 0.0, 1.0))
    ) / (2.0 * c.dx)
    e = wp.max(0.0001, e + dt * (nu * shear * shear - 0.2 * e * wp.sqrt(e) / c.dx))
    if solid[i, j, k] != 0:
        s = wp.vec4(ambient, c.humidity * saturation(ambient, pressure), 0.0, 0.0)
        d = 0.0
        e = 0.0001
    an[i, j, k] = s
    dn[i, j, k] = d
    en[i, j, k] = e


@dataclass
class Settings:
    shape: tuple = (64, 64, 96)
    dx: float = 0.32
    height: float = 25.0
    radius: float = 0.9
    top_radius: float = 4.0
    swirl: float = 19.0
    updraft: float = 6.5
    inflow: float = 2.5
    drive: float = 0.8
    humidity: float = 0.985
    temperature: float = 293.15
    dust_rate: float = 0.12
    cloud: float = 1.0
    lean: float = 1.0
    eddy: float = 0.12
    moisture_core: float = 1.12
    moisture_rate: float = 8.0
    pressure_iterations: int = 45


class Storm:
    def __init__(self, settings=None, terrain=None, solid_mask=None, device="cuda:0"):
        wp.init()
        self.device = device
        self.settings = settings or Settings()
        s = self.settings
        self.shape = tuple(s.shape)
        self.face_shape = tuple(v + 1 for v in self.shape)
        self.time = 0.0
        self.origin = np.array(
            [-self.shape[0] * s.dx * 0.5, -self.shape[1] * s.dx * 0.5, -2 * s.dx],
            dtype="f",
        )
        self.terrain = (
            np.zeros(self.shape[:2], dtype="f")
            if terrain is None
            else np.asarray(terrain, dtype="f")
        )
        z = self.origin[2] + (np.arange(self.shape[2]) + 0.5) * s.dx
        solid = (z[None, None, :] <= self.terrain[:, :, None]).astype("i")
        if solid_mask is not None:
            solid = np.maximum(solid, np.asarray(solid_mask, dtype="i"))
        c = Parameters()
        c.origin = wp.vec3(*self.origin)
        c.dx = s.dx
        c.nx, c.ny, c.nz = self.shape
        for key, value in asdict(s).items():
            if key not in ["shape", "dx", "pressure_iterations"]:
                setattr(c, key, value)
        self.c = c
        self.solid = wp.array(solid, dtype=int, device=device)
        self.floor = wp.array(self.terrain, dtype=float, device=device)
        self.u, self.v, self.w = [
            wp.zeros(self.face_shape, dtype=float, device=device) for _ in range(3)
        ]
        self.un, self.vn, self.wn = [
            wp.zeros(self.face_shape, dtype=float, device=device) for _ in range(3)
        ]
        self.a, self.an = [
            wp.zeros(self.shape, dtype=wp.vec4, device=device) for _ in range(2)
        ]
        self.dust, self.dn, self.energy, self.en, self.p, self.div, self.mag = [
            wp.zeros(self.shape, dtype=float, device=device) for _ in range(7)
        ]
        self.dust_forward, self.dust_advected = [
            wp.zeros(self.shape, dtype=float, device=device) for _ in range(2)
        ]
        self.omega = wp.zeros(self.shape, dtype=wp.vec3, device=device)
        wp.launch(
            initialize,
            self.shape,
            [self.a, self.dust, self.energy, self.solid, c],
            device=device,
        )
        wp.launch(
            initialize_velocity,
            self.face_shape,
            [self.u, self.v, self.w, self.solid, c],
            device=device,
        )
        wp.launch(
            pressure_sor,
            self.shape,
            [self.p, self.div, self.solid, s.dx, 0],
            device=device,
        )
        self.pressure_graph = None
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                for _ in range(s.pressure_iterations):
                    for color in range(2):
                        wp.launch(
                            pressure_sor,
                            self.shape,
                            [self.p, self.div, self.solid, s.dx, color],
                            device=device,
                        )
            self.pressure_graph = capture.graph
        self.project()
        self.p.zero_()

    def project(self):
        wp.launch(
            divergence,
            self.shape,
            [self.u, self.v, self.w, self.div, self.solid, self.settings.dx],
            device=self.device,
        )
        if self.pressure_graph is not None:
            wp.capture_launch(self.pressure_graph)
        else:
            for _ in range(self.settings.pressure_iterations):
                for color in range(2):
                    wp.launch(
                        pressure_sor,
                        self.shape,
                        [self.p, self.div, self.solid, self.settings.dx, color],
                        device=self.device,
                    )
        wp.launch(
            project,
            self.face_shape,
            [self.u, self.v, self.w, self.p, self.solid, self.settings.dx],
            device=self.device,
        )

    def step(self, dt):
        c = self.c
        wp.launch(
            curl_field,
            self.shape,
            [self.u, self.v, self.w, self.omega, self.mag, c],
            device=self.device,
        )
        wp.launch(
            advect_velocity,
            self.face_shape,
            [
                self.u,
                self.v,
                self.w,
                self.un,
                self.vn,
                self.wn,
                self.a,
                self.energy,
                self.omega,
                self.mag,
                self.solid,
                c,
                dt,
                self.time,
            ],
            device=self.device,
        )
        self.u, self.un = self.un, self.u
        self.v, self.vn = self.vn, self.v
        self.w, self.wn = self.wn, self.w
        self.project()
        wp.launch(
            advect_dust,
            self.shape,
            [self.u, self.v, self.w, self.dust, self.dust_forward, c, dt, 0.26],
            device=self.device,
        )
        wp.launch(
            correct_dust,
            self.shape,
            [
                self.u,
                self.v,
                self.w,
                self.dust,
                self.dust_forward,
                self.dust_advected,
                self.solid,
                c,
                dt,
                0.26,
            ],
            device=self.device,
        )
        wp.launch(
            transport,
            self.shape,
            [
                self.u,
                self.v,
                self.w,
                self.a,
                self.an,
                self.dust,
                self.dust_advected,
                self.dn,
                self.energy,
                self.en,
                self.solid,
                self.floor,
                c,
                dt,
                self.time,
            ],
            device=self.device,
        )
        self.a, self.an = self.an, self.a
        self.dust, self.dn = self.dn, self.dust
        self.energy, self.en = self.en, self.energy
        self.time += dt

    def diagnostics(self):
        a = self.a.numpy()
        d = self.dust.numpy()
        energy = self.energy.numpy()
        wp.launch(
            divergence,
            self.shape,
            [self.u, self.v, self.w, self.div, self.solid, self.settings.dx],
            device=self.device,
        )
        div = self.div.numpy()
        air = self.solid.numpy() == 0
        vmax = max(float(np.abs(v.numpy()).max()) for v in [self.u, self.v, self.w])
        assert (
            np.isfinite(a).all() and np.isfinite(d).all() and np.isfinite(energy).all()
        )
        assert np.isfinite(div).all() and np.isfinite(vmax)
        return dict(
            time=self.time,
            max_component_speed=vmax,
            divergence_rms=float(np.sqrt(np.mean(div[air] ** 2))),
            cloud_max=float(a[..., 2].max()),
            cloud_mass=float(a[..., 2][air].sum() * self.settings.dx**3 * 1.2),
            dust_max=float(d.max()),
            dust_mass=float(d.sum() * self.settings.dx**3),
            temperature_range=[
                float(a[..., 0][air].min()),
                float(a[..., 0][air].max()),
            ],
            energy_max=float(energy.max()),
            minimum_water=float(a[..., 1:].min()),
        )
