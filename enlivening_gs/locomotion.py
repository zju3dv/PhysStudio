"""Trajectory objectives and a Warp FEM / reduced OSQP locomotion runner."""

from physstudio.paths import data_path
import argparse
import json
import signal
from pathlib import Path
import time

import numpy as np
import torch

from .mesh import Cage
from .warp_fem import WarpFEM
from .contact import HybridController, WarpContactSet


class Goals:
    def __init__(self, cage, mass, mode="hop"):
        self.rest = cage.x.copy()
        self.mass = mass / mass.sum()
        self.com = self.mass @ self.rest
        self.mode = mode
        x = self.rest
        # Global COM, lower/upper local COMs, and four lateral sections.
        regions = [
            np.ones(len(x), bool),
            x[:, 2] < np.quantile(x[:, 2], 0.23),
            x[:, 2] > np.quantile(x[:, 2], 0.7),
            x[:, 0] < self.com[0],
            x[:, 0] >= self.com[0],
            x[:, 1] < self.com[1],
            x[:, 1] >= self.com[1],
        ]
        means = []
        for mask in regions:
            w = self.mass * mask
            means.append(w / w.sum())
        self.region = np.array(means)
        self.rows = np.concatenate(
            [np.kron(self.region, np.eye(3)), np.eye(3 * len(x))], axis=0
        )
        self.weights = np.r_[
            np.repeat([12, 7, 5, 3, 3, 3, 3], 3).astype(float),
            np.full(3 * len(x), 8 / np.sqrt(len(x))),
        ]
        self.weights[2] = 20

    def pose(self, time):
        if self.mode == "sway":
            u = np.clip((time - 0.3) / 1.8, 0, 1)
            phase = np.sin(2 * np.pi * u) * np.sin(np.pi * u)
            pose = self.rest.copy()
            z = self.rest[:, 2] / self.rest[:, 2].max()
            pose[:, 0] += 0.09 * phase * z
            pose[:, 1] += 0.045 * np.sin(np.pi * u) * z
            return pose
        # One controlled crouch / take-off / flight / recovery sequence.
        # These are objectives only: no position or velocity is overwritten.
        scale = 1.0
        offset = np.zeros(3)
        yaw = 0.0
        lean = 0.0
        if 0.30 <= time < 0.75:
            u = (time - 0.30) / 0.45
            scale = 1 - 0.25 * (0.5 - 0.5 * np.cos(np.pi * u))
            lean = 0.14 * (0.5 - 0.5 * np.cos(np.pi * u))
        elif 0.75 <= time < 0.95:
            u = (time - 0.75) / 0.20
            scale = 0.75 + 0.42 * (0.5 - 0.5 * np.cos(np.pi * u))
            offset[2] = 0.11 * u * u
            offset[0] = 0.10 * u * u
            lean = 0.14 * (1 - u)
        elif 0.95 <= time < 1.47:
            u = time - 0.95
            scale = 1.17 - 0.17 * min(u / 0.22, 1)
            offset[2] = max(0.0, 0.11 + 2.1 * u - 0.5 * 9.81 * u * u)
            offset[0] = 0.10 + 0.4 * u
        elif time >= 1.47:
            offset[0] = 0.308
        if self.mode == "turn":
            yaw = 0.6 * np.clip((time - 0.7) / 0.75, 0, 1)
        c, s = np.cos(yaw), np.sin(yaw)
        r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pose = self.rest.copy()
        pose[:, 2] *= scale
        pose[:, 0] += lean * self.rest[:, 2] / self.rest[:, 2].max()
        pose[:, :2] = (pose[:, :2] - self.com[:2]) @ r[:2, :2].T + self.com[:2]
        return pose + offset

    def __call__(self, time, x):
        target = self.pose(time)
        derivative = (self.pose(time + 1e-4) - self.pose(max(time - 1e-4, 0))) / (
            2e-4 if time > 1e-4 else 1e-4
        )
        vel = self.region @ derivative + 10 * (self.region @ (target - x))
        regularized = derivative + 7 * (target - x)
        return (
            self.rows,
            np.r_[vel.ravel(), regularized.ravel()],
            self.weights,
            self.region[0] @ target,
        )


def run(args):
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    out = Path(args.output)
    if (out / "trajectory.npz").exists() or (out / "progress.jsonl").exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(data_path("enlivening_gs") / args.scene / "gaussians.npz")
    cage = Cage(data["means"], data["axes"], args.resolution)
    cage.save(out / "cage.npz")
    density = np.full(len(cage.tets), args.density)
    if args.mass_profile == "base-heavy":
        # An enclosing cage also spans air between flowers/leaves. Giving that
        # region solid-pot density makes a scanned plant artificially top-heavy.
        z = (cage.centers[:, 2] - cage.x[:, 2].min()) / np.ptp(cage.x[:, 2])
        density *= 0.025 + 0.975 / (1 + np.exp((z - 0.30) / 0.035))
    np.savez_compressed(out / "material.npz", density=density)
    physics = WarpFEM(cage, young=args.young, density=density)
    contact = WarpContactSet(cage, data["means"], data["axes"], data["opacity"])
    controller = HybridController(mu=args.friction, effort=args.effort)
    goals = Goals(cage, physics.mass, args.motion)
    x = torch.tensor(cage.x, device="cuda", dtype=torch.float64)
    v = torch.zeros_like(x)
    dt = 1 / args.hz
    states = [cage.x.copy()]
    velocities = [np.zeros_like(cage.x)]
    times = [0.0]
    activations = [np.zeros(physics.nmuscle)]
    targets = [goals.com.copy()]
    config = vars(args).copy()
    config.update(
        nodes=len(x),
        tetrahedra=len(cage.tets),
        muscles=physics.nmuscle,
        solver="Warp Neo-Hookean implicit FEM + CPU reduced OSQP/LCP",
        physical_dt=dt,
        source=str((data_path("enlivening_gs") / args.scene / "source.json").resolve()),
    )
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    start = time.perf_counter()
    log = []
    completed = False

    def advance(x, v, now, dt, level=0):
        previous = x.cpu().numpy()
        saved = None if controller.last is None else controller.last.copy()
        tic = time.perf_counter()
        factor, free, response, stats = physics.linear_step_device(x, v, dt)
        fem_ms = 1000 * (time.perf_counter() - tic)
        tic = time.perf_counter()
        binding, points, contact_stats = contact.select(x, v, dt)
        detection_ms = 1000 * (time.perf_counter() - tic)
        rows, desired, weights, target = goals(now + dt, previous)
        tic = time.perf_counter()
        vnew, a, impulse, control_stats = controller.solve_device(
            factor, free, response, binding, points, dt, rows, desired, weights
        )
        control_ms = 1000 * (time.perf_counter() - tic)
        xnew = x + dt * vnew
        xp = xnew.cpu().numpy()
        f = np.einsum("eai,eaj->eij", xp[cage.tets], cage.grad)
        next_j = float(np.linalg.det(f).min())
        support = float(contact.embedding(xnew)[:, 2].min())
        invalid = not np.isfinite(xp).all() or next_j <= 0.05 or support < -0.006
        if invalid:
            controller.last = saved
            if level >= 7:
                raise FloatingPointError(
                    f"Unresolved step t={now:.5f}, dt={dt}: J={next_j}, support={support}"
                )
            xa, va, _, _, sa = advance(x, v, now, dt / 2, level + 1)
            xb, vb, ab, tb, sb = advance(xa, va, now + dt / 2, dt / 2, level + 1)
            for key in ["fem_ms", "detection_ms", "control_ms", "adaptive_substeps"]:
                sb[key] += sa[key]
            sb["next_min_j"] = min(sa["next_min_j"], sb["next_min_j"])
            sb["min_support"] = min(sa["min_support"], sb["min_support"])
            return xb, vb, ab, tb, sb
        record = dict(
            next_min_j=next_j,
            max_speed=float(torch.linalg.vector_norm(vnew, dim=-1).max()),
            fem_ms=fem_ms,
            detection_ms=detection_ms,
            control_ms=control_ms,
            adaptive_substeps=1,
            **stats,
            **contact_stats,
            **control_stats,
        )
        record["min_support"] = min(record["min_support"], support)
        return xnew, vnew, a, target, record

    def checkpoint():
        tmp = out / "trajectory.partial.npz"
        np.savez_compressed(
            tmp,
            x=np.array(states),
            v=np.array(velocities),
            time=times,
            activation=activations,
            target=targets,
        )
        tmp.replace(out / "trajectory.npz")

    try:
        for step in range(round(args.seconds * args.hz)):
            now = step * dt
            x, v, a, target, step_stats = advance(x, v, now, dt)
            xp = x.cpu().numpy()
            com = physics.mass @ xp / physics.mass.sum()
            record = dict(
                step=step + 1,
                time=now + dt,
                com=com.tolist(),
                target=target.tolist(),
                elapsed=time.perf_counter() - start,
                **step_stats,
            )
            log.append(record)
            with (out / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if (step + 1) % max(1, args.hz // 30) == 0:
                states.append(xp)
                velocities.append(v.cpu().numpy())
                times.append(now + dt)
                activations.append(a)
                targets.append(target)
            if step % 15 == 0:
                checkpoint()
                print(
                    f"step={step + 1} t={now + dt:.3f} COM={com.round(3)} target={target.round(3)} "
                    f"J={record['next_min_j']:.3f} contacts={record['contacts']} QP={record['qp_status']} "
                    f"fallback={record['forward_fallback']} wall={record['elapsed']:.1f}s",
                    flush=True,
                )
        completed = True
    finally:
        checkpoint()
        summary = dict(
            completed=completed,
            steps=len(log),
            frames=len(states),
            elapsed=time.perf_counter() - start,
        )
        if log:
            summary.update(
                min_j=min(r["next_min_j"] for r in log),
                max_support_penetration=max(0, -min(r["min_support"] for r in log)),
                fallback_steps=sum(r["forward_fallback"] for r in log),
                max_normal_velocity_violation=max(
                    r["normal_velocity_violation"] for r in log
                ),
                maximum_height=max(r["com"][2] for r in log),
                final_displacement=(np.array(log[-1]["com"]) - goals.com).tolist(),
                com_rmse=float(
                    np.sqrt(
                        np.mean(
                            [
                                np.sum((np.array(r["com"]) - r["target"]) ** 2)
                                for r in log
                            ]
                        )
                    )
                ),
            )
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=["wolf", "bread", "vasedeck"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--motion", choices=["hop", "turn", "sway"], default="hop")
    p.add_argument("--seconds", type=float, default=2.4)
    p.add_argument("--hz", type=int, default=90)
    p.add_argument("--resolution", type=int, default=8)
    p.add_argument("--young", type=float, default=3500.0)
    p.add_argument("--density", type=float, default=80.0)
    p.add_argument(
        "--mass-profile", choices=["uniform", "base-heavy"], default="uniform"
    )
    p.add_argument("--friction", type=float, default=0.65)
    p.add_argument("--effort", type=float, default=0.3)
    run(p.parse_args())
