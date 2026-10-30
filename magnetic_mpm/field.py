"""Langevin magnetization and matrix-free Newton-PCG on a Cartesian grid.

Sun et al. 2021, equations 5, 13, 22--26. G is a centered difference and
GT its exact transpose. The induced potential has homogeneous Neumann BC;
the collocated operator's eight parity gauges are removed after Newton updates. Permanent magnets are
prescribed sources; only the *induced material* magnetization enters the
Poisson RHS. This retains the external field in a vacuum and accounts for
the permanent source magnetization separately from the unknown material M.
"""

import time
import numpy as np
import torch
import warp as wp

MU0 = 4.0 * np.pi * 1.0e-7
MOMENT = 1.09e-19
BETA = MU0 * MOMENT / (1.380649e-23 * 300.0)
MS = 3.60e17 * MOMENT / (7.8e-3**3)


@wp.func
def langevin(a: float):
    l = float(0.0)
    dl = float(0.0)
    if a < 0.1:
        a2 = a * a
        l = a * (1.0 / 3.0 + a2 * (-1.0 / 45.0 + a2 * (2.0 / 945.0 - a2 / 4725.0)))
        dl = 1.0 / 3.0 + a2 * (-1.0 / 15.0 + a2 * (2.0 / 189.0 - a2 / 675.0))
    else:
        e = wp.exp(-2.0 * a)
        l = (1.0 + e) / (1.0 - e) - 1.0 / a
        dl = 1.0 / (a * a) - 4.0 * e / ((1.0 - e) * (1.0 - e))
    return l, dl


@wp.func
def gradient(p: wp.array3d(dtype=float), i: int, j: int, k: int, n: int, ih: float):
    g = wp.vec3(0.0)
    if i > 0 and i < n - 1:
        g[0] = (p[i + 1, j, k] - p[i - 1, j, k]) * (0.5 * ih)
    if j > 0 and j < n - 1:
        g[1] = (p[i, j + 1, k] - p[i, j - 1, k]) * (0.5 * ih)
    if k > 0 and k < n - 1:
        g[2] = (p[i, j, k + 1] - p[i, j, k - 1]) * (0.5 * ih)
    return g


@wp.kernel(enable_backward=False)
def transpose_gradient(
    q: wp.array3d(dtype=wp.vec3), out: wp.array3d(dtype=float), n: int, ih: float
):
    i, j, k = wp.tid()
    val = float(0.0)
    if i < n - 2:
        val -= q[i + 1, j, k][0]
    if j < n - 2:
        val -= q[i, j + 1, k][1]
    if k < n - 2:
        val -= q[i, j, k + 1][2]
    if i > 1:
        val += q[i - 1, j, k][0]
    if j > 1:
        val += q[i, j - 1, k][1]
    if k > 1:
        val += q[i, j, k - 1][2]
    out[i, j, k] = val * (0.5 * ih)


@wp.kernel(enable_backward=False)
def constitutive(
    phi: wp.array3d(dtype=float),
    external: wp.array3d(dtype=wp.vec3),
    saturation: wp.array3d(dtype=float),
    beta: float,
    n: int,
    ih: float,
    h: wp.array3d(dtype=wp.vec3),
    m: wp.array3d(dtype=wp.vec3),
    jac: wp.array3d(dtype=wp.mat33),
    flux: wp.array3d(dtype=wp.vec3),
):
    i, j, k = wp.tid()
    g = gradient(phi, i, j, k, n, ih)
    hv = external[i, j, k] - g
    length = wp.length(hv)
    a = beta * length
    l, dl = langevin(a)
    tangent = saturation[i, j, k] * beta / 3.0
    mv = wp.vec3(0.0)
    d = wp.mat33(0.0)
    if a > 1.0e-6:
        unit = hv / length
        tangent = saturation[i, j, k] * l / length
        mv = saturation[i, j, k] * l * unit
        d = (saturation[i, j, k] * beta * dl - tangent) * wp.outer(unit, unit)
    else:
        mv = tangent * hv
    for axis in range(3):
        d[axis, axis] += 1.0 + tangent
    h[i, j, k] = hv
    m[i, j, k] = mv
    jac[i, j, k] = d
    flux[i, j, k] = g - mv


@wp.kernel(enable_backward=False)
def jac_gradient(
    phi: wp.array3d(dtype=float),
    jac: wp.array3d(dtype=wp.mat33),
    n: int,
    ih: float,
    flux: wp.array3d(dtype=wp.vec3),
):
    i, j, k = wp.tid()
    flux[i, j, k] = jac[i, j, k] * gradient(phi, i, j, k, n, ih)


@wp.kernel(enable_backward=False)
def diagonal(
    jac: wp.array3d(dtype=wp.mat33), n: int, ih: float, out: wp.array3d(dtype=float)
):
    i, j, k = wp.tid()
    val = float(0.0)
    if i < n - 2:
        val += jac[i + 1, j, k][0, 0]
    if j < n - 2:
        val += jac[i, j + 1, k][1, 1]
    if k < n - 2:
        val += jac[i, j, k + 1][2, 2]
    if i > 1:
        val += jac[i - 1, j, k][0, 0]
    if j > 1:
        val += jac[i, j - 1, k][1, 1]
    if k > 1:
        val += jac[i, j, k - 1][2, 2]
    out[i, j, k] = val * (0.25 * ih * ih)


@wp.kernel(enable_backward=False)
def source_field(
    out: wp.array3d(dtype=wp.vec3),
    origin: wp.vec3,
    dx: float,
    center: wp.vec3,
    radius: float,
    intensity: float,
    uniform: wp.vec3,
):
    i, j, k = wp.tid()
    r = origin + dx * wp.vec3(float(i), float(j), float(k)) - center
    dist = wp.length(r)
    hv = uniform
    if intensity != 0.0:
        # Exact H of a uniformly magnetized sphere, magnetic moment along z.
        if dist < radius:
            hv += wp.vec3(0.0, 0.0, -intensity / 3.0)
        else:
            u = r / dist
            hv += (
                (intensity / 3.0)
                * wp.pow(radius / dist, 3.0)
                * (3.0 * u[2] * u - wp.vec3(0.0, 0.0, 1.0))
            )
    out[i, j, k] = hv


@wp.kernel(enable_backward=False)
def stress(
    h: wp.array3d(dtype=wp.vec3),
    m: wp.array3d(dtype=wp.vec3),
    external: wp.array3d(dtype=wp.vec3),
    mu0: float,
    out: wp.array3d(dtype=wp.mat33),
):
    i, j, k = wp.tid()
    hv, mv, ev = h[i, j, k], m[i, j, k], external[i, j, k]
    # Analytic vacuum Maxwell stress has zero divergence away from a source.
    # Subtract it before differencing, giving exactly zero force for Ms=0.
    delta = hv - ev
    t = mu0 * (wp.outer(delta + mv, hv) + wp.outer(ev, delta))
    pressure = 0.5 * mu0 * wp.dot(delta, hv + ev)
    for axis in range(3):
        t[axis, axis] -= pressure
    out[i, j, k] = t


@wp.kernel(enable_backward=False)
def stress_divergence(
    t: wp.array3d(dtype=wp.mat33), n: int, ih: float, out: wp.array3d(dtype=wp.vec3)
):
    i, j, k = wp.tid()
    f = wp.vec3(0.0)
    if i > 0 and j > 0 and k > 0 and i < n - 1 and j < n - 1 and k < n - 1:
        tx = (t[i + 1, j, k] - t[i - 1, j, k]) * (0.5 * ih)
        ty = (t[i, j + 1, k] - t[i, j - 1, k]) * (0.5 * ih)
        tz = (t[i, j, k + 1] - t[i, j, k - 1]) * (0.5 * ih)
        for a in range(3):
            f[a] = tx[a, 0] + ty[a, 1] + tz[a, 2]
    out[i, j, k] = f


class Field:
    def __init__(self, n, dx, origin, device="cuda:0", tolerance=2.0e-4):
        wp.init()
        self.device = device
        if device.startswith("cuda"):
            wp.set_stream(
                wp.stream_from_torch(torch.cuda.current_stream(device)), device=device
            )
        self.n, self.dx, self.origin, self.tolerance = n, dx, origin, tolerance
        self.shape = (n, n, n)

        def buf(dtype):
            return wp.zeros(self.shape, dtype=dtype, device=device)

        self.phi, self.saturation, self.residual, self.diag, self.product = [
            buf(float) for _ in range(5)
        ]
        self.external, self.h, self.m, self.flux, self.temp, self.force_density = [
            buf(wp.vec3) for _ in range(6)
        ]
        self.jac, self.stress = [buf(wp.mat33) for _ in range(2)]
        self.pt = wp.to_torch(self.phi)
        self.rt = wp.to_torch(self.residual)
        self.dt = wp.to_torch(self.diag)
        self.at = wp.to_torch(self.product)
        self.last_stats = {}

    def launch(self, kernel, inputs):
        wp.launch(kernel, self.shape, inputs, device=self.device)

    def set_source(
        self,
        center=(0.0, 0.0, 1.0),
        radius=0.04,
        intensity=1.0e6,
        uniform=(0.0, 0.0, 0.0),
    ):
        self.launch(
            source_field,
            [
                self.external,
                wp.vec3(*self.origin),
                self.dx,
                wp.vec3(*center),
                radius,
                intensity,
                wp.vec3(*uniform),
            ],
        )

    def evaluate(self):
        self.launch(
            constitutive,
            [
                self.phi,
                self.external,
                self.saturation,
                BETA,
                self.n,
                1 / self.dx,
                self.h,
                self.m,
                self.jac,
                self.flux,
            ],
        )
        self.launch(transpose_gradient, [self.flux, self.residual, self.n, 1 / self.dx])
        return torch.linalg.vector_norm(self.rt).item()

    def matvec(self, p):
        self.launch(
            jac_gradient, [wp.from_torch(p), self.jac, self.n, 1 / self.dx, self.temp]
        )
        self.launch(transpose_gradient, [self.temp, self.product, self.n, 1 / self.dx])
        return self.at

    def solve(self, max_newton=12, max_cg=240):
        start = time.perf_counter()
        initial = self.evaluate()
        history = [initial]
        cg_counts = []
        # Scale convergence against the zero-potential RHS, not the previous
        # frame's nearly converged residual.
        old = self.pt.clone()
        self.pt.zero_()
        reference = self.evaluate()
        self.pt.copy_(old)
        residual = self.evaluate()
        target = max(reference * self.tolerance, 1.0e-4)
        for iteration in range(max_newton):
            if residual <= target:
                break
            self.launch(diagonal, [self.jac, self.n, 1 / self.dx, self.diag])
            r = -self.rt.clone()
            r -= r.mean()
            step = torch.zeros_like(r)
            z = r / self.dt
            p = z.clone()
            rz = (r * z).sum()
            # Inexact Newton, tighten each linear system near convergence.
            cg_target = max(target * 0.12, residual * 0.001)
            for ci in range(max_cg):
                ap = self.matvec(p)
                denom = (p * ap).sum()
                alpha = rz / denom.clamp_min(1.0e-30)
                step.add_(alpha * p)
                r.sub_(alpha * ap)
                if (ci + 1) % 8 == 0 and torch.linalg.vector_norm(r).item() < cg_target:
                    break
                z = r / self.dt
                rz_new = (r * z).sum()
                p = z + (rz_new / rz.clamp_min(1.0e-30)) * p
                rz = rz_new
            cg_counts.append(ci + 1)
            old = self.pt.clone()
            accepted = False
            for ls in range(14):
                self.pt.copy_(old + step * (0.5**ls))
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            block = self.pt[a::2, b::2, c::2]
                            block.sub_(block.mean())
                trial = self.evaluate()
                if trial < residual:
                    residual = trial
                    accepted = True
                    break
            if not accepted:
                self.pt.copy_(old)
                self.evaluate()
                break
            history.append(residual)
        self.launch(stress, [self.h, self.m, self.external, MU0, self.stress])
        self.launch(
            stress_divergence, [self.stress, self.n, 1 / self.dx, self.force_density]
        )
        self.last_stats = dict(
            residuals=history,
            reference=reference,
            relative_residual=residual / max(reference, 1.0e-20),
            converged=bool(residual <= target),
            cg_iterations=cg_counts,
            seconds=time.perf_counter() - start,
        )
        if not np.isfinite(residual):
            raise FloatingPointError("Non-finite magnetic residual")
        return self.last_stats
