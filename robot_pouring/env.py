"""GPU tensor environment and a Gymnasium single-environment adapter."""
from . import require_runtime
require_runtime()
from pathlib import Path
import time
import numpy as np
import torch
import warp as wp
import mujoco
import mujoco_warp as mjw
import gymnasium as gym
from gymnasium import spaces
from . import ROOT, CACHE_ROOT
from .config import PouringConfig
from .model import make_model, inverse_kinematics
from .fluid import ParticleWater, copy_pose, reset_particles, apply_wrench


class PouringEnv:
    """Batched GPU environment; explicit selective reset, no hidden autoreset.

    step(actions) returns obs, reward, terminated, truncated, info as tensors.
    Actions are seven normalized joint target velocities. Call reset(env_ids)
    for finished worlds before stepping again. Observations are privileged
    state summaries; no rendering or host copies occur in the physics loop.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, cfg=None):
        self.cfg = cfg or PouringConfig()
        c = self.cfg
        if c.backend != "mujoco_warp":
            raise ValueError("This implementation uses the mujoco_warp backend")
        wp.config.kernel_cache_dir = str(CACHE_ROOT / "warp")
        wp.init()
        self.device = torch.device(c.device)
        self.num_envs = c.num_envs
        self.model, self.xml = make_model(c)
        self.cup_id = self.model.body("cup").id
        self.robot_model = self.model
        self.initial_q, ep, er = inverse_kinematics(self.model, [.41, -.12, .32], np.eye(3))
        if ep > .001 or er > .01:
            raise RuntimeError(f"Reset IK failed: position={ep}, rotation={er}")
        with wp.ScopedDevice(c.device):
            self.wm = mjw.put_model(self.model)
            # CUDA 12.2 drivers support ordinary graph replay but not conditional
            # graph nodes. The upstream fixed-iteration path has the same solver.
            self.wm.opt.graph_conditional = False
            self.data = mjw.make_data(self.model, nworld=c.num_envs, nconmax=64, njmax=96)
            self.water = ParticleWater(c)
        self.g = self.water.g
        self.q = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.pos = wp.to_torch(self.g.cup_pos)
        self.rot = wp.to_torch(self.g.cup_rot)
        self.target = wp.to_torch(self.g.target)
        self.count = wp.to_torch(self.g.initial_count)
        self.metrics = wp.to_torch(self.g.metrics)
        self.mask = wp.to_torch(self.g.mask)
        self.rng = [np.random.default_rng(c.seed + 1009 * i) for i in range(c.num_envs)]
        self.steps = torch.zeros(c.num_envs, device=self.device, dtype=torch.long)
        self.done = torch.zeros(c.num_envs, device=self.device, dtype=torch.bool)
        self.hold = torch.zeros_like(self.steps)
        self.last_action = torch.zeros((c.num_envs, 7), device=self.device)
        self.cup_velocity = torch.zeros((c.num_envs, 3), device=self.device)
        self.previous_mass = torch.zeros((c.num_envs, 5), device=self.device)
        self.previous_distance = torch.zeros(c.num_envs, device=self.device)
        self.goal = torch.full((c.num_envs,), c.goal_fraction, device=self.device)
        self.lower = torch.tensor(self.model.jnt_range[:7, 0] + .01, dtype=torch.float32, device=self.device)
        self.upper = torch.tensor(self.model.jnt_range[:7, 1] - .01, dtype=torch.float32, device=self.device)
        self.graph = None
        self.renderer = None
        self.reset(seed=c.seed)
        # Compile once outside capture; reset all state after the warm-up.
        self._substep()
        wp.synchronize_device(c.device)
        self.reset(seed=c.seed)
        if c.graph and wp.get_device(c.device).is_cuda:
            with wp.ScopedDevice(c.device), wp.ScopedCapture() as capture:
                self._substep()
            self.graph = capture.graph
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(c.num_envs, 51), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(c.num_envs, 7), dtype=np.float32)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, shape=(51,), dtype=np.float32)
        self.single_action_space = spaces.Box(-1, 1, shape=(7,), dtype=np.float32)

    def _substep(self):
        with wp.ScopedDevice(self.cfg.device):
            wp.launch(apply_wrench, self.num_envs, [self.g, self.data.xfrc_applied, self.data.xipos,
                                                   self.cup_id, int(self.cfg.two_way)])
            mjw.step(self.wm, self.data)
            # mj_step integrates q after forward dynamics; refresh the boundary
            # pose so the fluid sees the new robot state in this same substep.
            mjw.kinematics(self.wm, self.data)
            mjw.com_pos(self.wm, self.data)
            wp.launch(copy_pose, self.num_envs, [self.g, self.data.xpos, self.data.xmat, self.cup_id])
            self.water.step()

    def reset(self, *, seed=None, env_ids=None, options=None):
        if seed is not None:
            self.rng = [np.random.default_rng(seed + 1009 * i) for i in range(self.num_envs)]
        ids = np.arange(self.num_envs) if env_ids is None else np.asarray(torch.as_tensor(env_ids).cpu(), dtype=int)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= self.num_envs):
            raise ValueError("env_ids must be unique valid world indices")
        idx = torch.as_tensor(ids, device=self.device)
        self.mask.zero_(); self.mask[idx] = True
        with wp.ScopedDevice(self.cfg.device):
            mjw.reset_data(self.wm, self.data, self.g.mask)
        q = np.tile(self.initial_q, (len(ids), 1))
        target = np.tile([self.cfg.target_x, self.cfg.target_y, .012], (len(ids), 1))
        fill = np.full(len(ids), self.cfg.fill_height)
        for j, e in enumerate(ids):
            if self.cfg.randomize:
                target[j, :2] += self.rng[e].uniform(-.035, .035, 2)
                fill[j] *= self.rng[e].uniform(.75, 1.0)
                q[j, :7] += self.rng[e].uniform(-.008, .008, 7)
        self.q[idx] = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        self.ctrl[idx, :7] = self.q[idx, :7]
        self.ctrl[idx, 7] = .025
        self.target[idx] = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        wp.to_torch(self.data.mocap_pos)[idx, 0] = self.target[idx]
        wp.to_torch(self.g.fill)[idx] = torch.as_tensor(fill, dtype=torch.float32, device=self.device)
        with wp.ScopedDevice(self.cfg.device):
            mjw.kinematics(self.wm, self.data)
            mjw.com_pos(self.wm, self.data)
            wp.launch(copy_pose, self.num_envs, [self.g, self.data.xpos, self.data.xmat, self.cup_id])
        self.count[idx] = 0
        wp.to_torch(self.g.failures)[idx] = 0
        wp.to_torch(self.g.absorption_attempts)[idx] = 0
        wp.to_torch(self.g.force)[idx] = 0
        wp.to_torch(self.g.torque)[idx] = 0
        wp.launch(reset_particles, self.water.count, [self.g], device=self.cfg.device)
        if self.cfg.goal_ml is not None:
            self.goal[idx] = self.cfg.goal_ml / (self.count[idx] * self.g.mass * 1000)
            if (self.goal[idx] > 1).any():
                raise ValueError("Requested goal_ml exceeds the initialized water volume")
        self.steps[idx] = 0; self.done[idx] = False; self.hold[idx] = 0
        self.last_action[idx] = 0; self.cup_velocity[idx] = 0
        self.water.measure()
        self.previous_mass[idx] = self.mass_fractions()[idx]
        self.previous_distance[idx] = self.alignment_distance()[idx]
        return self.observe(), self.info()

    def mass_fractions(self):
        return self.metrics[:, :5] / self.count.clamp_min(1)[:, None]

    def lip(self):
        direction = -self.rot[:, 2, :2]
        length = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
        fallback = torch.tensor([0., 1.], device=self.device).expand(self.num_envs, 2)
        direction = torch.where(length > 1e-4, direction / length.clamp_min(1e-4), fallback)
        local = torch.cat([self.cfg.cup_radius * direction,
                           torch.full((self.num_envs, 1), self.cfg.cup_height, device=self.device)], dim=1)
        return self.pos + (self.rot @ local[:, :, None]).squeeze(-1)

    def alignment_distance(self):
        return torch.linalg.vector_norm(self.lip()[:, :2] - self.target[:, :2], dim=1)

    def observe(self):
        goals = torch.stack([self.goal, torch.full_like(self.goal, self.cfg.spill_budget)], dim=1)
        obs = torch.cat([self.q[:, :7], self.qvel[:, :7] * .1, self.ctrl[:, :7] - self.q[:, :7],
                         self.target - self.pos, self.pos, self.rot[:, :, :2].reshape(self.num_envs, 6),
                         self.cup_velocity, self.mass_fractions(), goals,
                         (self.steps / self.cfg.max_steps)[:, None], self.last_action], dim=1)
        return obs

    def info(self):
        fractions = self.mass_fractions()
        return {"mass_fractions": fractions.clone(), "captured_fraction": (fractions[:, 1] + fractions[:, 4]).clone(),
                "spill_fraction": fractions[:, 3].clone(), "water_ml": (self.count * self.g.mass * 1000).clone(),
                "goal_ml": (self.goal * self.count * self.g.mass * 1000).clone(),
                "elapsed_seconds": self.steps * self.cfg.control_dt,
                "physics_failure": wp.to_torch(self.g.failures).clone() > 0}

    def step(self, actions):
        # The rigid backend is not differentiable. Accept policy tensors without
        # retaining their autograd graph across environment steps.
        action = torch.as_tensor(actions, dtype=torch.float32, device=self.device).detach()
        if action.shape != (self.num_envs, 7):
            raise ValueError(f"Expected action shape {(self.num_envs, 7)}, got {tuple(action.shape)}")
        if not torch.isfinite(action).all():
            raise ValueError("Actions must be finite")
        if self.done.any():
            raise RuntimeError("Reset finished env_ids before calling step again")
        action = action.clamp(-1., 1.)
        self.ctrl[:, :7] = torch.maximum(self.lower, torch.minimum(self.upper,
                            self.ctrl[:, :7] + self.cfg.action_speed * self.cfg.control_dt * action))
        before = self.pos.clone()
        for _ in range(self.cfg.substeps):
            if self.graph is None:
                self._substep()
            else:
                wp.capture_launch(self.graph)
        self.cup_velocity = (self.pos - before) / self.cfg.control_dt
        self.water.measure()
        self.steps += 1
        mass = self.mass_fractions()
        captured, prev = mass[:, 1] + mass[:, 4], self.previous_mass[:, 1] + self.previous_mass[:, 4]
        goal = self.goal
        distance = self.alignment_distance()
        reward = 25. * (captured.clamp_max(goal) - prev.clamp_max(goal))
        reward -= 60. * (mass[:, 3] - self.previous_mass[:, 3])
        if self.cfg.task == "flower":
            reward -= 30. * ((captured - goal - .04).clamp_min(0) - (prev - goal - .04).clamp_min(0))
        reward += 2. * (self.previous_distance - distance)
        reward -= .002 + .001 * (action ** 2).mean(1) + .001 * ((action - self.last_action) ** 2).mean(1)
        good = (captured >= goal - .03) & (mass[:, 3] <= self.cfg.spill_budget) & (mass[:, 2] < .025) & (self.rot[:, 2, 2] > .94)
        if self.cfg.task == "flower":
            good &= captured <= goal + .05
        self.hold = torch.where(good, self.hold + 1, 0)
        success = self.hold >= round(.4 / self.cfg.control_dt)
        failure = (mass[:, 3] > .10) | (wp.to_torch(self.g.failures) > 0)
        failure |= (torch.abs(self.qvel[:, :7]) > 15).any(1) | (~torch.isfinite(self.q)).any(1)
        failure |= (self.pos[:, 2] < .06) | (self.pos[:, 2] > 1.1)
        terminated = success | failure
        truncated = (self.steps >= self.cfg.max_steps) & ~terminated
        reward += 10. * success.float() - 5. * failure.float()
        self.done = terminated | truncated
        self.previous_mass = mass.clone(); self.previous_distance = distance.clone()
        self.last_action = action.clone()
        info = self.info(); info.update(success=success, failure=failure)
        # Overflow is reported, never silently treated as valid physics.
        overflow = wp.to_torch(self.data.nacon)[0] > self.data.naconmax
        overflow |= (wp.to_torch(self.data.nefc) > self.data.njmax).any()
        if overflow:
            raise RuntimeError("MuJoCo Warp contact/constraint capacity exceeded")
        return self.observe(), reward, terminated, truncated, info

    def snapshot(self, world=0):
        wp.synchronize_device(self.cfg.device)
        e, n = world, self.g.n
        offset = np.array([3 * (e % 16), 3 * (e // 16), 0.])
        return {"qpos": self.q[e].cpu().numpy().copy(), "qvel": self.qvel[e].cpu().numpy().copy(),
                "ctrl": self.ctrl[e].cpu().numpy().copy(),
                "x": self.g.x.numpy()[e*n:(e+1)*n] - offset,
                "v": self.g.v.numpy()[e*n:(e+1)*n], "status": self.g.status.numpy()[e*n:(e+1)*n],
                "absorbed_pos": self.g.absorbed_pos.numpy()[e*n:(e+1)*n],
                "cup_pos": self.pos[e].cpu().numpy().copy(), "cup_rot": self.rot[e].cpu().numpy().copy(),
                "target": self.target[e].cpu().numpy().copy(), "mass_fractions": self.mass_fractions()[e].cpu().numpy().copy(),
                "cup_force": self.g.force.numpy()[e].copy(), "cup_torque": self.g.torque.numpy()[e].copy(),
                "time": float(self.steps[e]) * self.cfg.control_dt}

    def render(self, world=0):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640, max_geom=self.g.n + 2000)
        snap = self.snapshot(world)
        data = mujoco.MjData(self.model)
        data.qpos[:] = snap["qpos"]; data.mocap_pos[0] = snap["target"]
        mujoco.mj_forward(self.model, data)
        camera = mujoco.MjvCamera(); camera.lookat[:] = [.4, 0, .25]
        camera.distance = 1.45; camera.azimuth = 135; camera.elevation = -25
        self.renderer.update_scene(data, camera)
        scene = self.renderer.scene
        for p in snap["x"][snap["status"] < 2]:
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.full(3, self.cfg.spacing * .55), p, np.eye(3).ravel(), np.array([.05, .55, .9, 1.]))
            scene.ngeom += 1
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close(); self.renderer = None


class GymPouringEnv(gym.Env):
    """Standard NumPy Gymnasium adapter for single-world RL libraries."""
    metadata = PouringEnv.metadata

    def __init__(self, task="basin", render_mode=None, **kwargs):
        self.render_mode = render_mode
        self.env = PouringEnv(PouringConfig(task=task, num_envs=1, **kwargs))
        self.observation_space = self.env.single_observation_space
        self.action_space = self.env.single_action_space

    @staticmethod
    def _info(info):
        return {k: v[0].detach().cpu().numpy() for k, v in info.items()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed, options=options)
        return obs[0].cpu().numpy(), self._info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(np.asarray(action)[None])
        return obs[0].cpu().numpy(), float(reward[0]), bool(terminated[0]), bool(truncated[0]), self._info(info)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


for _id, _task in [("PhysStudio-PourBasin-v0", "basin"), ("PhysStudio-WaterPlant-v0", "flower"),
                   ("PhysStudio-PrecisionPour-v0", "precision")]:
    if _id not in gym.registry:
        gym.register(_id, entry_point=GymPouringEnv, kwargs={"task": _task})
