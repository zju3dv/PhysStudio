"""GPU Neo-Hookean FEM, stress actuators and batched implicit solves.

Warp evaluates/assembles physical terms and applies the block sparse operator.
Torch's CUDA eigh performs the same 12x12 element PSD projection as fem.py.
No simulation-sized matrix is factorized on the CPU. A NumPy solve adapter is
provided for the small OSQP controller; solve_device keeps results on the GPU.
"""

import numpy as np
import torch
import warp as wp

from .mesh import Cage


@wp.kernel
def _elastic(
    x: wp.array(dtype=wp.vec3d),
    tets: wp.array(dtype=wp.vec4i),
    grad: wp.array2d(dtype=wp.vec3d),
    volume: wp.array(dtype=wp.float64),
    mu: wp.float64,
    lam: wp.float64,
    deformation: wp.array(dtype=wp.mat33d),
    force: wp.array(dtype=wp.vec3d),
    energy: wp.array(dtype=wp.float64),
    dets: wp.array(dtype=wp.float64),
):
    e = wp.tid()
    f = wp.mat33d(wp.float64(0.0))
    for a in range(4):
        f += wp.outer(x[tets[e][a]], grad[e, a])
    j = wp.determinant(f)
    deformation[e] = f
    dets[e] = j
    # An inverted state is rejected by the stepper, not silently accepted.
    logj = wp.log(wp.max(j, wp.float64(1.0e-20)))
    p = mu * f + (lam * logj - mu) * wp.transpose(wp.inverse(f))
    for a in range(4):
        wp.atomic_add(force, tets[e][a], -volume[e] * (p * grad[e, a]))
    ic = wp.float64(0.0)
    for i in range(3):
        for k in range(3):
            ic += f[i, k] * f[i, k]
    energy[e] = volume[e] * (
        mu * wp.float64(0.5) * (ic - wp.float64(3.0))
        - mu * logj
        + lam * wp.float64(0.5) * logj * logj
    )


@wp.kernel
def _tangent(
    f: wp.array(dtype=wp.mat33d),
    grad: wp.array2d(dtype=wp.vec3d),
    volume: wp.array(dtype=wp.float64),
    mu: wp.float64,
    lam: wp.float64,
    out: wp.array3d(dtype=wp.float64),
):
    e, a, b = wp.tid()
    fit = wp.transpose(wp.inverse(f[e]))
    qa, qb = fit * grad[e, a], fit * grad[e, b]
    d = wp.dot(grad[e, a], grad[e, b])
    coefficient = mu - lam * wp.log(wp.max(wp.determinant(f[e]), wp.float64(1.0e-20)))
    for i in range(3):
        for k in range(3):
            val = lam * qa[i] * qb[k] + coefficient * qb[i] * qa[k]
            if i == k:
                val += mu * d
            out[e, 3 * a + i, 3 * b + k] = volume[e] * val


@wp.kernel
def _assemble(
    elements: wp.array3d(dtype=wp.float64),
    slots: wp.array3d(dtype=wp.int32),
    blocks: wp.array(dtype=wp.mat33d),
):
    e, a, b = wp.tid()
    block = wp.mat33d(wp.float64(0.0))
    for i in range(3):
        for j in range(3):
            block[i, j] = elements[e, 3 * a + i, 3 * b + j]
    wp.atomic_add(blocks, slots[e, a, b], block)


@wp.kernel
def _activation(
    f: wp.array(dtype=wp.mat33d),
    rot: wp.array(dtype=wp.mat33d),
    tets: wp.array(dtype=wp.vec4i),
    grad: wp.array2d(dtype=wp.vec3d),
    volume: wp.array(dtype=wp.float64),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array2d(dtype=wp.float64),
    young: wp.float64,
    result: wp.array2d(dtype=wp.float64),
):
    e, m = wp.tid()
    d = rot[e] * directions[m]
    fit = wp.transpose(wp.inverse(f[e]))
    coefficient = -young * volume[e] * wp.determinant(f[e]) * weights[e, m]
    for a in range(4):
        fa = coefficient * wp.dot(d, fit * grad[e, a]) * d
        for i in range(3):
            wp.atomic_add(result, 3 * tets[e][a] + i, m, fa[i])


@wp.kernel
def _system(
    current: wp.array(dtype=wp.mat33d),
    rest: wp.array(dtype=wp.mat33d),
    rows: wp.array(dtype=wp.int32),
    columns: wp.array(dtype=wp.int32),
    mass: wp.array(dtype=wp.float64),
    dt: wp.float64,
    alpha: wp.float64,
    beta: wp.float64,
    blocks: wp.array(dtype=wp.mat33d),
    inverse_diagonal: wp.array(dtype=wp.mat33d),
):
    k = wp.tid()
    h = dt * dt * current[k] + dt * beta * rest[k]
    if rows[k] == columns[k]:
        diagonal = (wp.float64(1.0) + dt * alpha) * mass[rows[k]]
        for i in range(3):
            h[i, i] += diagonal
        inverse_diagonal[rows[k]] = wp.inverse(h)
    blocks[k] = h


@wp.kernel
def _matvec(
    offset: wp.array(dtype=wp.int32),
    columns: wp.array(dtype=wp.int32),
    blocks: wp.array(dtype=wp.mat33d),
    x: wp.array2d(dtype=wp.float64),
    y: wp.array2d(dtype=wp.float64),
):
    node, rhs = wp.tid()
    v = wp.vec3d(wp.float64(0.0))
    for k in range(offset[node], offset[node + 1]):
        col = columns[k]
        v += blocks[k] * wp.vec3d(
            x[3 * col, rhs], x[3 * col + 1, rhs], x[3 * col + 2, rhs]
        )
    for i in range(3):
        y[3 * node + i, rhs] = v[i]


@wp.kernel
def _precondition(
    inverse: wp.array(dtype=wp.mat33d),
    r: wp.array2d(dtype=wp.float64),
    z: wp.array2d(dtype=wp.float64),
):
    node, rhs = wp.tid()
    v = inverse[node] * wp.vec3d(
        r[3 * node, rhs], r[3 * node + 1, rhs], r[3 * node + 2, rhs]
    )
    for i in range(3):
        z[3 * node + i, rhs] = v[i]


@wp.kernel
def embed_points(
    x: wp.array(dtype=wp.vec3d),
    ids: wp.array(dtype=wp.vec4i),
    weights: wp.array(dtype=wp.vec4d),
    out: wp.array(dtype=wp.vec3d),
):
    k = wp.tid()
    p = wp.vec3d(wp.float64(0.0))
    for a in range(4):
        p += x[ids[k][a]] * weights[k][a]
    out[k] = p


class GPUFactor:
    """Batched block-Jacobi PCG. All right-hand sides share the FEM operator."""

    def __init__(self, owner, dt):
        self.owner = owner
        n = len(owner.cage.x)
        self.blocks = wp.empty(owner.nnz, dtype=wp.mat33d, device=owner.device)
        self.inverse = wp.empty(n, dtype=wp.mat33d, device=owner.device)
        wp.launch(
            _system,
            owner.nnz,
            [
                owner.k,
                owner.k0,
                owner.row,
                owner.col,
                owner.mass_wp,
                dt,
                owner.alpha,
                owner.beta,
                self.blocks,
                self.inverse,
            ],
            device=owner.device,
        )
        self.last_solve = {}

    def multiply(self, x):
        y = torch.empty_like(x)
        wp.launch(
            _matvec,
            (len(self.owner.cage.x), x.shape[1]),
            [
                self.owner.offset,
                self.owner.col,
                self.blocks,
                wp.from_torch(x),
                wp.from_torch(y),
            ],
            device=self.owner.device,
        )
        return y

    def solve_device(self, b, tol=2e-9, maxiter=240):
        b = torch.as_tensor(
            b, dtype=torch.float64, device=self.owner.torch_device
        ).contiguous()
        one = b.ndim == 1
        if one:
            b = b[:, None].contiguous()
        x = torch.zeros_like(b)
        r = b.clone()
        z = torch.empty_like(b)

        def precondition():
            wp.launch(
                _precondition,
                (len(self.owner.cage.x), b.shape[1]),
                [self.inverse, wp.from_torch(r), wp.from_torch(z)],
                device=self.owner.device,
            )

        precondition()
        p = z.clone()
        rz = (r * z).sum(0)
        norm = torch.linalg.vector_norm(b, dim=0).clamp_min(1e-20)
        for it in range(maxiter):
            ap = self.multiply(p)
            denom = (p * ap).sum(0)
            active = torch.linalg.vector_norm(r, dim=0) > tol * norm
            step = torch.where(
                active, rz / denom.clamp_min(1e-100), torch.zeros_like(rz)
            )
            x += p * step
            r -= ap * step
            if it % 12 == 11 and not bool(active.any()):
                break
            precondition()
            new = (r * z).sum(0)
            beta = torch.where(active, new / rz.clamp_min(1e-100), torch.zeros_like(rz))
            p = z + p * beta
            rz = new
        actual = torch.linalg.vector_norm(self.multiply(x) - b, dim=0) / norm
        residual = float(actual.max())
        self.last_solve = dict(
            iterations=it + 1,
            max_relative_residual=residual,
            right_hand_sides=b.shape[1],
        )
        if not np.isfinite(residual) or residual > max(tol * 20, 1e-7):
            raise RuntimeError(f"GPU PCG failed: {self.last_solve}")
        return x[:, 0] if one else x

    def solve(self, b):
        return self.solve_device(b).cpu().numpy()


class WarpFEM:
    def __init__(
        self,
        cage,
        young=3500.0,
        poisson=0.35,
        density=80.0,
        alpha=0.12,
        beta=0.002,
        device="cuda:0",
    ):
        wp.init()
        self.cage, self.young, self.alpha, self.beta = cage, young, alpha, beta
        self.device = wp.get_device(device)
        self.torch_device = device
        # Share the Torch current stream, avoiding hidden cross-stream races.
        wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream(device)))
        self.mu = young / (2 * (1 + poisson))
        self.lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        self.mass = np.zeros(len(cage.x))
        np.add.at(self.mass, cage.tets.ravel(), np.repeat(cage.vol * density / 4, 4))
        self.md = np.repeat(self.mass, 3)
        self.segments, self.directions, self.weights = cage.muscles()
        self.nmuscle = len(self.directions)

        def arr(x, dtype):
            return wp.array(x, dtype=dtype, device=device)

        self.tets = arr(cage.tets, wp.vec4i)
        self.grad = arr(cage.grad, wp.vec3d)
        self.volume = arr(cage.vol, wp.float64)
        self.mass_wp = arr(self.mass, wp.float64)
        self.dir_wp = arr(self.directions, wp.vec3d)
        self.weights_wp = arr(self.weights, wp.float64)
        pairs = np.stack(
            np.broadcast_arrays(cage.tets[:, :, None], cage.tets[:, None, :]), -1
        ).reshape(-1, 2)
        unique, slots = np.unique(pairs, axis=0, return_inverse=True)
        self.nnz = len(unique)
        self.slots = arr(slots.reshape(-1, 4, 4).astype(np.int32), wp.int32)
        self.row = arr(unique[:, 0].astype(np.int32), wp.int32)
        self.col = arr(unique[:, 1].astype(np.int32), wp.int32)
        self.offset = arr(
            np.r_[
                0, np.cumsum(np.bincount(unique[:, 0], minlength=len(cage.x)))
            ].astype(np.int32),
            wp.int32,
        )
        self.f = wp.empty(len(cage.tets), dtype=wp.mat33d, device=device)
        self.dets = wp.empty(len(cage.tets), dtype=wp.float64, device=device)
        self.energy = wp.empty(len(cage.tets), dtype=wp.float64, device=device)
        self.k = wp.empty(self.nnz, dtype=wp.mat33d, device=device)
        self.elastic_device(cage.x)
        self.k0 = wp.clone(self.k)

    def elastic_device(self, x, project=True):
        xt = torch.as_tensor(
            x, dtype=torch.float64, device=self.torch_device
        ).contiguous()
        force = torch.zeros_like(xt)
        wp.launch(
            _elastic,
            len(self.cage.tets),
            [
                wp.from_torch(xt, dtype=wp.vec3d),
                self.tets,
                self.grad,
                self.volume,
                self.mu,
                self.lam,
                self.f,
                wp.from_torch(force, dtype=wp.vec3d),
                self.energy,
                self.dets,
            ],
            device=self.device,
        )
        ke = torch.empty(
            (len(self.cage.tets), 12, 12), dtype=torch.float64, device=self.torch_device
        )
        wp.launch(
            _tangent,
            (len(self.cage.tets), 4, 4),
            [self.f, self.grad, self.volume, self.mu, self.lam, wp.from_torch(ke)],
            device=self.device,
        )
        if project:
            values, vectors = torch.linalg.eigh(ke)
            ke = (
                (vectors * values.clamp_min(0)[:, None, :]) @ vectors.transpose(1, 2)
            ).contiguous()
        self.k.zero_()
        wp.launch(
            _assemble,
            (len(self.cage.tets), 4, 4),
            [wp.from_torch(ke), self.slots, self.k],
            device=self.device,
        )
        return force, ke

    def activation_device(self):
        u, _, vh = torch.linalg.svd(wp.to_torch(self.f))
        rotation = (u @ vh).contiguous()
        a = torch.zeros(
            (len(self.cage.x) * 3, self.nmuscle),
            dtype=torch.float64,
            device=self.torch_device,
        )
        wp.launch(
            _activation,
            (len(self.cage.tets), self.nmuscle),
            [
                self.f,
                wp.from_torch(rotation, dtype=wp.mat33d),
                self.tets,
                self.grad,
                self.volume,
                self.dir_wp,
                self.weights_wp,
                self.young,
                wp.from_torch(a),
            ],
            device=self.device,
        )
        return a

    def linear_step_device(self, x, v, dt):
        force, _ = self.elastic_device(x)
        dets = wp.to_torch(self.dets)
        minj, maxj = float(dets.min()), float(dets.max())
        if minj <= 0:
            raise FloatingPointError(f"Inverted tetrahedron J={minj}")
        mass = wp.to_torch(self.mass_wp)
        force[:, 2] -= 9.81 * mass
        vt = torch.as_tensor(v, dtype=torch.float64, device=self.torch_device)
        rhs = (mass[:, None] * vt + dt * force).reshape(-1, 1)
        activation = dt * self.activation_device()
        factor = GPUFactor(self, dt)
        solved = factor.solve_device(torch.cat([rhs, activation], 1))
        return (
            factor,
            solved[:, 0],
            solved[:, 1:],
            dict(
                min_j=minj,
                max_j=maxj,
                elastic_energy=float(wp.to_torch(self.energy).sum()),
                linear_solve=factor.last_solve,
            ),
        )

    def linear_step(self, x, v, dt):
        factor, free, response, stats = self.linear_step_device(x, v, dt)
        return factor, free.cpu().numpy(), response.cpu().numpy(), stats


class WarpEmbedding:
    def __init__(self, cage, points, device="cuda:0"):
        ids, weights = cage.bind(points)
        self.device = device
        self.ids = wp.array(ids, dtype=wp.vec4i, device=device)
        self.weights = wp.array(weights, dtype=wp.vec4d, device=device)
        self.count = len(points)

    def __call__(self, x):
        xt = torch.as_tensor(x, dtype=torch.float64, device=self.device).contiguous()
        out = torch.empty((self.count, 3), dtype=torch.float64, device=self.device)
        wp.launch(
            embed_points,
            self.count,
            [
                wp.from_torch(xt, dtype=wp.vec3d),
                self.ids,
                self.weights,
                wp.from_torch(out, dtype=wp.vec3d),
            ],
            device=self.device,
        )
        return out
