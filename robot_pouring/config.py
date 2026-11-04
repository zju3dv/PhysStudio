"""SI-unit task and solver configuration."""
from dataclasses import dataclass, asdict


@dataclass
class PouringConfig:
    task: str = "basin"
    num_envs: int = 1
    device: str = "cuda:0"
    backend: str = "mujoco_warp"
    spacing: float = 0.008
    dt: float = 0.0025
    substeps: int = 16
    iterations: int = 4
    episode_seconds: float = 12.0
    action_speed: float = 0.9
    seed: int = 42
    randomize: bool = True
    graph: bool = True
    cup_radius: float = 0.044
    cup_height: float = 0.14
    cup_wall: float = 0.006
    fill_height: float = 0.095
    target_radius: float | None = None
    target_height: float | None = None
    target_x: float = 0.59
    target_y: float = 0.02
    goal_fraction: float | None = None
    goal_ml: float | None = None
    spill_budget: float = 0.02
    viscosity: float = 0.025
    density: float = 1000.0
    two_way: bool = True

    def __post_init__(self):
        defaults = {"basin": (.13, .085, .85), "flower": (.085, .10, .55), "precision": (.072, .11, .75)}
        if self.task not in defaults:
            raise ValueError(f"Unknown task: {self.task}")
        for name, value in zip(("target_radius", "target_height", "goal_fraction"), defaults[self.task]):
            if getattr(self, name) is None: setattr(self, name, value)
        if self.num_envs < 1 or self.num_envs > 256:
            raise ValueError("num_envs must be in [1, 256]")
        if not 0.004 <= self.spacing <= 0.014:
            raise ValueError("spacing must be in [0.004, 0.014] meters")
        if self.dt <= 0 or self.substeps < 1 or self.iterations < 1:
            raise ValueError("Positive timesteps and iteration counts are required")
        if not 0 < self.goal_fraction <= 1 or not 0 <= self.spill_budget < 1:
            raise ValueError("Invalid task water fractions")
        if self.goal_ml is not None and self.goal_ml <= 0:
            raise ValueError("goal_ml must be positive")

    @property
    def control_dt(self):
        return self.dt * self.substeps

    @property
    def max_steps(self):
        return round(self.episode_seconds / self.control_dt)

    def to_dict(self):
        return asdict(self)
