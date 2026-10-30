"""Penalty prediction, contact-state QP and Coulomb LCP active-set correction."""

import osqp
import numpy as np
from scipy import sparse


def friction_project(t, radius):
    """Projection onto the four-direction Coulomb pyramid |tx|+|ty| <= mu*n."""
    if abs(t).sum() <= radius:
        return t
    u = np.sort(abs(t))[::-1]
    threshold = (u.sum() - radius) / 2 if u[0] - u[1] < radius else u[0] - radius
    return np.sign(t) * np.maximum(abs(t) - threshold, 0)


def forward_lcp_reference(velocity, delassus, gap, dt, mu, start=None, iterations=180):
    """Projected block pivots for Eq.9; no kinematic position projection."""
    nc = len(gap)
    impulse = np.zeros((nc, 3)) if start is None else start.copy().reshape(nc, 3)
    vel = velocity + delassus @ impulse.ravel()
    bias = gap / dt
    for it in range(iterations):
        change = 0.0
        for i in range(nc):
            idx = slice(3 * i, 3 * i + 3)
            w = delassus[idx, idx]
            old = impulse[i].copy()
            impulse[i, 2] = max(
                0, old[2] - (vel[3 * i + 2] + bias[i]) / max(w[2, 2], 1e-12)
            )
            vel += delassus[:, 3 * i + 2] * (impulse[i, 2] - old[2])
            tang = old[:2] - np.linalg.solve(
                w[:2, :2] + np.eye(2) * 1e-10, vel[3 * i : 3 * i + 2]
            )
            impulse[i, :2] = friction_project(tang, mu * impulse[i, 2])
            vel += delassus[:, 3 * i : 3 * i + 2] @ (impulse[i, :2] - old[:2])
            change = max(change, float(np.max(abs(impulse[i] - old))))
        if change < 1e-8:
            break
    return impulse, vel.reshape(nc, 3), it + 1


from .contact_kernels import forward_lcp


class ContactSet:
    def __init__(self, cage, means, axes, opacity):
        points = np.stack(
            [means]
            + [means + s * 3 * axes[:, :, i] for s in [-1, 1] for i in range(3)],
            1,
        )
        self.rest = points.reshape(-1, 3)
        self.binding = cage.matrix(self.rest)
        self.spacing = cage.h * 0.8

    def select(self, x, v, dt):
        pos = self.binding @ x
        vel = self.binding @ v
        eligible = np.where(
            (pos[:, 2] < 0.045) | ((pos[:, 2] + dt * vel[:, 2]) < 0.02)
        )[0]
        if not len(eligible):
            return (
                sparse.csr_matrix((0, len(x))),
                np.zeros((0, 3)),
                dict(min_support=float(pos[:, 2].min()), candidates=0),
            )
        # Keep the deepest embedding in each small world-space contact patch.
        # All embeddings are rechecked every step, including split endpoints.
        order = eligible[np.argsort(pos[eligible, 2])]
        bins = np.floor(pos[order, :2] / self.spacing).astype(int)
        _, first = np.unique(bins, axis=0, return_index=True)
        ids = order[first]
        if len(ids) > 96:
            ids = ids[np.argsort(pos[ids, 2])[:96]]
        return (
            self.binding[ids],
            pos[ids],
            dict(
                min_support=float(pos[:, 2].min()),
                candidates=len(eligible),
                contacts=len(ids),
            ),
        )


class HybridController:
    def __init__(self, mu=0.65, effort=0.003):
        self.mu, self.effort = mu, effort
        self.last = None

    def solve(
        self,
        factor,
        free,
        muscle_response,
        binding,
        points,
        dt,
        target_rows,
        target_vel,
        weights,
    ):
        nc = len(points)
        j = sparse.kron(binding, sparse.eye(3), format="csr")
        rf = factor.solve(j.T.toarray()) if nc else np.zeros((len(free), 0))
        response = np.concatenate([muscle_response, rf], 1)
        qfree = np.asarray(j @ free)
        qa = np.asarray(j @ muscle_response)
        w = np.asarray(j @ rf)
        gap = points[:, 2]
        b = (target_rows @ response) * weights[:, None]
        goal = (target_vel - target_rows @ free) * weights
        activation, force, stats = self.solve_reduced(qfree, qa, w, gap, dt, b, goal)
        velocity = free + muscle_response @ activation + rf @ force.ravel()
        return velocity.reshape(-1, 3), activation, force, stats

    def solve_device(
        self,
        factor,
        free,
        muscle_response,
        binding,
        points,
        dt,
        target_rows,
        target_vel,
        weights,
    ):
        import torch

        def t(x):
            return torch.as_tensor(x, dtype=torch.float64, device=free.device)

        nc = len(points)
        j = t(sparse.kron(binding, sparse.eye(3), format="csr").toarray())
        rf = (
            factor.solve_device(j.T.contiguous())
            if nc
            else free.new_zeros((len(free), 0))
        )
        response = torch.cat([muscle_response, rf], 1)
        tr = t(target_rows)
        b = (tr @ response) * t(weights)[:, None]
        goal = (t(target_vel) - tr @ free) * t(weights)
        # Only reduced control/contact matrices leave the GPU.
        activation, force, stats = self.solve_reduced(
            (j @ free).cpu().numpy(),
            (j @ muscle_response).cpu().numpy(),
            (j @ rf).cpu().numpy(),
            points[:, 2],
            dt,
            b.cpu().numpy(),
            goal.cpu().numpy(),
        )
        velocity = free + muscle_response @ t(activation) + rf @ t(force.ravel())
        return velocity.reshape(-1, 3), activation, force, stats

    def solve_reduced(self, qfree, qa, w, gap, dt, b, goal):
        m = qa.shape[1]
        nc = len(gap)
        if self.last is None:
            self.last = np.zeros(m)
        # Stage 1: paper Eq.10–11 regularized contact prediction with previous a.
        predict = (qfree + qa @ self.last).reshape(nc, 3)
        penalty = np.zeros((nc, 3))
        if nc:
            effective = 1 / np.maximum(np.diag(w).reshape(nc, 3)[:, 2], 1e-9)
            penalty[:, 2] = effective * np.maximum(-(predict[:, 2] + (gap / dt)), 0)
            for i in range(nc):
                penalty[i, :2] = friction_project(
                    -effective[i] * predict[i, :2], self.mu * penalty[i, 2]
                )
            # LCP polishing removes penalty compliance before choosing branches.
            impulse, predicted, lcp_iters = forward_lcp(
                qfree + qa @ self.last, w, gap, dt, self.mu, penalty
            )
            state = (impulse[:, 2] > 1e-7).astype(int)
            state[(state == 1) & (np.linalg.norm(predicted[:, :2], axis=1) > 0.015)] = 2
        else:
            impulse, predicted, state = np.zeros((0, 3)), predict, np.zeros(0, int)
            lcp_iters = 0
        reg = np.r_[np.full(m, self.effort), np.full(3 * nc, 1e-7)]
        p = b.T @ b + np.diag(reg)
        q = -b.T @ goal
        q[:m] -= self.effort * 0.25 * self.last
        lower_speed = -gap / dt
        trial = None
        statuses = []
        for pivot in range(7):
            rows = []
            lo = []
            hi = []

            def add(row, l=-np.inf, u=np.inf):
                rows.append(row)
                lo.append(l)
                hi.append(u)

            # Stress amplitudes are regularized; numerical limits are generous.
            for k in range(m):
                row = np.zeros(m + 3 * nc)
                row[k] = 1
                add(row, -3.0, 3.0)
            jr = np.concatenate([qa, w], 1)
            for i in range(nc):
                off = m + 3 * i
                er = np.eye(3, m + 3 * nc, k=off)
                if state[i] == 0:
                    for r in er:
                        add(r, 0, 0)
                    add(jr[3 * i + 2], lower_speed[i] - qfree[3 * i + 2])
                else:
                    add(er[2], 0)
                    add(
                        jr[3 * i + 2],
                        lower_speed[i] - qfree[3 * i + 2],
                        lower_speed[i] - qfree[3 * i + 2],
                    )
                    if state[i] == 1:
                        for k in range(2):
                            add(jr[3 * i + k], -qfree[3 * i + k], -qfree[3 * i + k])
                        for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                            add(sx * er[0] + sy * er[1] - self.mu * er[2], u=0)
                    else:
                        direction = np.zeros(2)
                        major = np.argmax(abs(predicted[i, :2]))
                        direction[major] = np.sign(predicted[i, major]) or 1
                        for k in range(2):
                            add(er[k] + self.mu * direction[k] * er[2], 0, 0)
                        add(
                            direction[0] * jr[3 * i] + direction[1] * jr[3 * i + 1],
                            -direction @ qfree[3 * i : 3 * i + 2],
                        )
            qp = osqp.OSQP()
            qp.setup(
                P=sparse.csc_matrix(np.triu(p)),
                q=q,
                A=sparse.csc_matrix(np.array(rows)),
                l=np.array(lo),
                u=np.array(hi),
                verbose=False,
                eps_abs=1e-6,
                eps_rel=1e-5,
                max_iter=6000,
                polishing=True,
                adaptive_rho=True,
            )
            qp.warm_start(x=np.r_[self.last, impulse.ravel()])
            solved = qp.solve()
            statuses.append(solved.info.status)
            if solved.info.status_val in (1, 2):
                trial = solved.x
                check, cv, _ = forward_lcp(
                    qfree + qa @ trial[:m],
                    w,
                    gap,
                    dt,
                    self.mu,
                    trial[m:].reshape(nc, 3),
                )
                # Redundant support points can have different forces but the
                # same velocity. Pivot based on the physical contact response.
                predicted_qp = (qfree + qa @ trial[:m] + w @ trial[m:]).reshape(nc, 3)
                if not nc or np.max(abs(cv - predicted_qp)) < 2e-3:
                    break
                impulse, predicted = check, cv
            else:
                # Pivot with a feasible forward LCP; vary the inverse-control
                # candidate instead of applying a direct target displacement.
                uncon = np.linalg.solve(p[:m, :m], -q[:m])
                uncon = np.clip(uncon, -3, 3)
                mix = [0.25, 0.5, 0.75, 1.0, 0.0, 0.1, 0.9][pivot]
                candidate = (1 - mix) * self.last + mix * uncon
                impulse, predicted, _ = forward_lcp(
                    qfree + qa @ candidate, w, gap, dt, self.mu
                )
            state = (impulse[:, 2] > 1e-7).astype(int)
            state[(state == 1) & (np.linalg.norm(predicted[:, :2], axis=1) > 0.008)] = 2
        fallback = trial is None
        if fallback:
            # Explicitly reported forward LCP fallback, never a kinematic move.
            activation = np.clip(np.linalg.solve(p[:m, :m], -q[:m]), -3, 3)
        else:
            activation = trial[:m]
        force, cv, iterations = forward_lcp(
            qfree + qa @ activation,
            w,
            gap,
            dt,
            self.mu,
            trial[m:].reshape(nc, 3) if trial is not None else impulse,
            iterations=300,
        )
        self.last = activation.copy()
        normal_res = cv[:, 2] + (gap / dt) if nc else np.zeros(0)
        stats = dict(
            qp_status=statuses[-1],
            pivots=pivot,
            forward_fallback=fallback,
            lcp_iterations=iterations,
            penalty_lcp_iterations=lcp_iters,
            normal_complementarity=float(
                np.max(abs(force[:, 2] * normal_res), initial=0)
            ),
            normal_velocity_violation=float(max(0, -np.min(normal_res, initial=0))),
            friction_violation=float(
                np.max(abs(force[:, :2]).sum(1) - self.mu * force[:, 2], initial=0)
            ),
            activation_max=float(abs(activation).max()),
            tracking_residual=float(
                np.linalg.norm(b @ np.r_[activation, force.ravel()] - goal)
            ),
        )
        return activation, force, stats


class WarpContactSet(ContactSet):
    """Evaluate every Gaussian support point on GPU; transfer selected patches."""

    def __init__(self, cage, means, axes, opacity, device="cuda:0"):
        super().__init__(cage, means, axes, opacity)
        from .warp_fem import WarpEmbedding

        self.embedding = WarpEmbedding(cage, self.rest, device)

    def select(self, x, v, dt):
        import torch

        pos = self.embedding(x)
        vel = self.embedding(v)
        eligible = torch.where(
            (pos[:, 2] < 0.045) | ((pos[:, 2] + dt * vel[:, 2]) < 0.02)
        )[0]
        minimum = float(pos[:, 2].min())
        if not len(eligible):
            return (
                sparse.csr_matrix((0, len(x))),
                np.zeros((0, 3)),
                dict(min_support=minimum, candidates=0, contacts=0),
            )
        order = eligible[torch.argsort(pos[eligible, 2], stable=True)]
        bins = torch.floor(pos[order, :2] / self.spacing).long()
        _, inv = torch.unique(bins, dim=0, return_inverse=True)
        first = torch.full(
            (int(inv.max()) + 1,), len(order), dtype=torch.long, device=pos.device
        )
        first.scatter_reduce_(
            0,
            inv,
            torch.arange(len(order), device=pos.device),
            reduce="amin",
            include_self=True,
        )
        ids = order[first]
        if len(ids) > 96:
            ids = ids[torch.argsort(pos[ids, 2])[:96]]
        return (
            self.binding[ids.cpu().numpy()],
            pos[ids].cpu().numpy(),
            dict(min_support=minimum, candidates=len(eligible), contacts=len(ids)),
        )
