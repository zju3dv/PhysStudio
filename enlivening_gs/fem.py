"""Neo-Hookean tetrahedral FEM with implicit stepping and muscle stress."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


class FEM:
    def __init__(
        self, cage, young=3500.0, poisson=0.35, density=80.0, alpha=0.12, beta=0.002
    ):
        self.cage = cage
        self.young = young
        self.mu = young / (2 * (1 + poisson))
        self.lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        self.mass = np.zeros(len(cage.x))
        np.add.at(self.mass, cage.tets.ravel(), np.repeat(cage.vol * density / 4, 4))
        self.md = np.repeat(self.mass, 3)
        self.dofs = (cage.tets[:, :, None] * 3 + np.arange(3)).reshape(-1, 12)
        self.rows = np.repeat(self.dofs, 12, axis=1).ravel()
        self.cols = np.tile(self.dofs, (1, 12)).ravel()
        self.segments, self.directions, self.weights = cage.muscles()
        self.nmuscle = len(self.directions)
        _, self.k0, _ = self.elastic(cage.x)
        self.damping = sparse.diags(alpha * self.md) + beta * self.k0

    def deformation(self, x):
        return np.einsum("eai,eaj->eij", x[self.cage.tets], self.cage.grad)

    def elastic(self, x, project=True):
        c = self.cage
        f = self.deformation(x)
        det = np.linalg.det(f)
        if det.min() <= 0:
            raise FloatingPointError(f"Inverted tetrahedron J={det.min()}")
        fit = np.linalg.inv(f).transpose(0, 2, 1)
        logj = np.log(det)
        stress = self.mu * f + (self.lam * logj - self.mu)[:, None, None] * fit
        ef = -c.vol[:, None, None] * np.einsum("eij,eaj->eai", stress, c.grad)
        force = np.zeros_like(x)
        np.add.at(force, c.tets.ravel(), ef.reshape(-1, 3))
        q = np.einsum("eij,eaj->eai", fit, c.grad)
        dot = np.einsum("eaj,ebj->eab", c.grad, c.grad)
        # Tangent K_ai,bk = d² energy / dx_ai dx_bk.
        k = self.mu * np.einsum("eab,ik->eaibk", dot, np.eye(3))
        k += self.lam * np.einsum("eai,ebk->eaibk", q, q)
        k += np.einsum("e,ebi,eak->eaibk", self.mu - self.lam * logj, q, q)
        k = (k * c.vol[:, None, None, None, None]).reshape(-1, 12, 12)
        if project:
            val, vec = np.linalg.eigh(k)
            k = (vec * np.maximum(val, 0)[:, None, :]) @ vec.transpose(0, 2, 1)
        kt = sparse.coo_matrix(
            (k.ravel(), (self.rows, self.cols)), shape=(len(x) * 3,) * 2
        ).tocsc()
        energy = float(
            np.sum(
                c.vol
                * (
                    self.mu / 2 * ((f * f).sum((1, 2)) - 3)
                    - self.mu * logj
                    + self.lam / 2 * logj**2
                )
            )
        )
        return (
            force,
            kt,
            dict(min_j=float(det.min()), max_j=float(det.max()), elastic_energy=energy),
        )

    def activation_matrix(self, x):
        c = self.cage
        f = self.deformation(x)
        # Rotate rest fiber stress using the local polar rotation (supp. Eq.20).
        u, _, vt = np.linalg.svd(f)
        r = u @ vt
        d = np.einsum("eij,mj->emi", r, self.directions)
        spatial_grad = np.einsum("eaj,eji->eai", c.grad, np.linalg.inv(f))
        # -V sigma gradN gives equal/opposite face forces and zero net torque.
        contraction = np.einsum("emi,eai->eam", d, spatial_grad)
        ef = (
            -self.young
            * (c.vol * np.linalg.det(f))[:, None, None, None]
            * self.weights[:, None, :, None]
        )
        ef = ef * contraction[:, :, :, None] * d[:, None, :, :]
        a = np.zeros((len(x), 3, self.nmuscle))
        np.add.at(
            a, c.tets.ravel(), ef.transpose(0, 1, 3, 2).reshape(-1, 3, self.nmuscle)
        )
        return a.reshape(-1, self.nmuscle)

    def linear_step(self, x, v, dt):
        force, k, stats = self.elastic(x)
        gravity = np.zeros_like(x)
        gravity[:, 2] = -9.81 * self.mass
        h = sparse.diags(self.md) + dt * self.damping + dt * dt * k
        factor = splu(h.tocsc())
        rhs = self.md * v.ravel() + dt * (force + gravity).ravel()
        free = factor.solve(rhs)
        a = dt * self.activation_matrix(x)
        response = factor.solve(a)
        return factor, free, response, stats
