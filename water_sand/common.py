"""Lattices and shared state for the 3D mixture solver."""

import warp as wp


@wp.struct
class Grid:
    nx: int
    ny: int
    nz: int
    q: int
    rho_air: float
    nu_water: float
    nu_air: float
    gravity: float
    sigma: float
    interface: float
    mobility: float
    rho_s: float
    mu: float
    lam: float
    pvol: float
    retention: int
    absorb_rate: float
    cohesion: float
    friction: float
    cd: float
    drag_scale: float
    coupled: int
    conservative: int
    wall: wp.array(dtype=int)
    f: wp.array2d(dtype=float)
    fn: wp.array2d(dtype=float)
    h: wp.array2d(dtype=float)
    hn: wp.array2d(dtype=float)
    phi: wp.array(dtype=float)
    u: wp.array(dtype=wp.vec3)
    ut: wp.array(dtype=wp.vec3)
    force: wp.array(dtype=wp.vec3)
    pressure: wp.array(dtype=float)
    pressure_next: wp.array(dtype=float)
    viscosity: wp.array(dtype=float)
    gradphi: wp.array(dtype=wp.vec3)
    gradeps: wp.array(dtype=wp.vec3)
    source: wp.array(dtype=float)
    mass: wp.array(dtype=float)
    momentum: wp.array(dtype=wp.vec3)
    stressforce: wp.array(dtype=wp.vec3)
    vs: wp.array(dtype=wp.vec3)
    delta: wp.array(dtype=float)
    eps: wp.array(dtype=float)
    eps_old: wp.array(dtype=float)
    eps_eff: wp.array(dtype=float)
    bound: wp.array(dtype=float)
    capacity: wp.array(dtype=float)
    bound_next: wp.array(dtype=float)
    absorbed: wp.array(dtype=float)
    drag: wp.array(dtype=wp.vec3)
    totals: wp.array(dtype=wp.float64)
    control: wp.array(dtype=float)


@wp.struct
class Sand:
    x: wp.array(dtype=wp.vec3)
    v: wp.array(dtype=wp.vec3)
    c: wp.array(dtype=wp.mat33)
    fe: wp.array(dtype=wp.mat33)
    plastic: wp.array(dtype=float)
    hardening: wp.array(dtype=float)
    alpha: wp.array(dtype=float)
    water: wp.array(dtype=float)
    saturation: wp.array(dtype=float)


@wp.func
def coord(g: Grid, n: int):
    return wp.vec3i(n // (g.ny * g.nz), (n // g.nz) % g.ny, n % g.nz)


@wp.func
def index(g: Grid, p: wp.vec3i):
    return (p[0] * g.ny + p[1]) * g.nz + p[2]


@wp.func
def neighbor(g: Grid, p: wp.vec3i):
    result = int(-1)
    if (
        p[0] >= 0
        and p[1] >= 0
        and p[2] >= 0
        and p[0] < g.nx
        and p[1] < g.ny
        and p[2] < g.nz
    ):
        n = index(g, p)
        if g.wall[n] == 0:
            result = n
    return result


@wp.func
def lattice(q: int):
    return wp.vec3i(q // 9 - 1, (q // 3) % 3 - 1, q % 3 - 1)


@wp.func
def direction(q: int):
    c = lattice(q)
    return wp.vec3(float(c[0]), float(c[1]), float(c[2]))


@wp.func
def weight(q: int):
    c = lattice(q)
    w = float(1.0)
    for a in range(3):
        if c[a] == 0:
            w *= 2.0 / 3.0
        else:
            w *= 1.0 / 6.0
    return w


@wp.func
def phase_direction(q: int):
    c = wp.vec3i(0)
    if q > 0:
        axis = (q - 1) // 2
        c[axis] = 1 - 2 * ((q - 1) % 2)
    return c


@wp.func
def phase_weight(q: int):
    w = float(0.125)
    if q == 0:
        w = 0.25
    return w


@wp.func
def rho(g: Grid, phi: float):
    return g.rho_air + (1.0 - g.rho_air) * wp.clamp(phi, 0.0, 1.0)


@wp.func
def nu(g: Grid, phi: float):
    p = wp.clamp(phi, 0.0, 1.0)
    return (p * g.nu_water + (1.0 - p) * g.rho_air * g.nu_air) / rho(g, p)


@wp.func
def weights(x: float):
    return wp.vec3(
        0.5 * (1.5 - x) ** 2.0, 0.75 - (x - 1.0) ** 2.0, 0.5 * (x - 0.5) ** 2.0
    )


@wp.func
def inverse_moment(c: int, u: float):
    out = wp.vec3(1.0 - u * u, -2.0 * u, -1.0)
    if c == 1:
        out = wp.vec3(0.5 * u * (u + 1.0), u + 0.5, 0.5)
    elif c == -1:
        out = wp.vec3(0.5 * u * (u - 1.0), u - 0.5, 0.5)
    return out


@wp.func
def population(q: int, u: wp.vec3, force: wp.vec3, stress: wp.mat33):
    """Tensor-product inverse central moments, including sixth-order equilibrium."""
    c = lattice(q)
    x = inverse_moment(c[0], u[0])
    y = inverse_moment(c[1], u[1])
    z = inverse_moment(c[2], u[2])
    ex, ey, ez = x[0] + x[2] / 3.0, y[0] + y[2] / 3.0, z[0] + z[2] / 3.0
    f = ex * ey * ez
    f += 0.5 * (
        force[0] * x[1] * ey * ez
        + force[1] * ex * y[1] * ez
        + force[2] * ex * ey * z[1]
    )
    f += (
        stress[0, 0] * x[2] * y[0] * z[0]
        + stress[1, 1] * x[0] * y[2] * z[0]
        + stress[2, 2] * x[0] * y[0] * z[2]
    )
    f += (
        stress[0, 1] * x[1] * y[1] * z[0]
        + stress[0, 2] * x[1] * y[0] * z[1]
        + stress[1, 2] * x[0] * y[1] * z[1]
    )
    return f
