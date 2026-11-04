"""Closed-loop reference controller and reproducible physics recordings.

This controller is not an RL policy. It commands the same seven joint actions
available to a policy; cup and water motion still come from the simulation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .config import PouringConfig
from .env import PouringEnv
from .model import inverse_kinematics


def smooth(t):
    t = np.clip(t, 0., 1.)
    return t * t * (3. - 2. * t)


class ReferenceController:
    def __init__(self, env):
        self.env = env
        self.start = env.pos.detach().cpu().numpy().copy()
        self.q = env.q.detach().cpu().numpy().copy()
        self.angle = np.zeros(env.num_envs)
        self.returning = np.zeros(env.num_envs, dtype=bool)
        self.max_ik_error = 0.
        self.max_orientation_error = 0.

    def __call__(self):
        env = self.env
        target = env.target.detach().cpu().numpy()
        mass = env.mass_fractions().detach().cpu().numpy()
        ctrl = env.ctrl[:, :7].detach().cpu().numpy()
        t = env.steps.detach().cpu().numpy() * env.cfg.control_dt
        actions = []
        for e in range(env.num_envs):
            captured = mass[e, 1] + mass[e, 4]
            # Account for already airborne water before turning the cup back.
            anticipated = captured + mass[e, 2]
            threshold = float(env.goal[e]) - (.10 if env.cfg.task == "flower" else .03)
            if anticipated >= threshold or mass[e, 0] < .015:
                self.returning[e] = True
            rate = -.70 if self.returning[e] else .30
            if t[e] > 1.5:
                self.angle[e] = np.clip(self.angle[e] + rate * env.cfg.control_dt, 0., 1.9)
            rotation = Rotation.from_euler("x", -self.angle[e]).as_matrix()
            offset = [0., -.04, .308]
            if env.cfg.task == "flower":
                offset = [-.04, -.07, .268]
            elif env.cfg.task == "precision":
                offset = [0., -.07, .288]
            lip = target[e] + np.array(offset)
            desired = lip - rotation @ np.array([0., env.cfg.cup_radius, env.cfg.cup_height])
            if t[e] < 1.5:
                desired = self.start[e] * (1 - smooth(t[e] / 1.5)) + desired * smooth(t[e] / 1.5)
            self.q[e], ep, er = inverse_kinematics(env.model, desired, rotation, q0=self.q[e], iterations=35)
            self.max_ik_error = max(self.max_ik_error, ep)
            self.max_orientation_error = max(self.max_orientation_error, er)
            action = (self.q[e, :7] - ctrl[e]) / (env.cfg.action_speed * env.cfg.control_dt)
            actions.append(np.clip(action, -1., 1.))
        return torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=env.device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["basin", "flower", "precision"], default="basin")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--spacing", type=float, default=.006)
    p.add_argument("--seconds", type=float, default=12.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--frames", type=int, help="Saved states including frame zero, at unchanged physical time steps")
    args = p.parse_args()
    if args.frames is not None and args.frames < 2: p.error("--frames must be at least two")
    args.output.mkdir(parents=True, exist_ok=False)
    states = args.output / "states"; states.mkdir()
    cfg = PouringConfig(task=args.task, spacing=args.spacing, episode_seconds=args.seconds,
                        seed=args.seed, randomize=args.randomize, graph=not args.no_graph)
    (args.output / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    env = PouringEnv(cfg)
    controller = ReferenceController(env)
    (args.output / "model.xml").write_text(env.xml)
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")}
    (args.output / "sources.json").write_text(json.dumps(hashes, indent=2) + "\n")
    np.savez_compressed(states / "00000.npz", **env.snapshot())
    records = []; started = time.perf_counter()
    for i in range(min(cfg.max_steps, args.frames-1 if args.frames is not None else cfg.max_steps)):
        action = controller()
        obs, reward, terminated, truncated, info = env.step(action)
        snap = env.snapshot()
        snap["action"] = action[0].cpu().numpy(); snap["reward"] = float(reward[0])
        np.savez_compressed(states / f"{i+1:05d}.npz", **snap)
        row = dict(frame=i+1, time=snap["time"], mass=snap["mass_fractions"].tolist(), reward=float(reward[0]),
                   success=bool(info["success"][0]), failure=bool(info["failure"][0]),
                   cup_force=snap["cup_force"].tolist())
        records.append(row)
        if i % 25 == 0:
            print(json.dumps(row), flush=True)
        if args.preview and i % 25 == 0:
            from PIL import Image
            Image.fromarray(env.render()).save(args.output / f"preview_{i+1:05d}.png")
        if bool(terminated[0] | truncated[0]):
            break
    report = dict(frames=len(records)+1, prefix_only=not bool(terminated[0] | truncated[0]),
                  particles=env.g.n, water_ml=float(info["water_ml"][0]),
                  final=records[-1], elapsed_seconds=time.perf_counter()-started,
                  max_ik_position_error=controller.max_ik_error,
                  max_ik_rotation_error=controller.max_orientation_error,
                  controller="closed-loop reference, not trained RL", records=records)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
