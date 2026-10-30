"""Mass-accounted continuous lava inflow through an elevated cross-section."""

import numpy as np
import torch
import warp as wp


class LavaInlet:
    def __init__(
        self,
        center,
        half_width,
        thickness,
        speed,
        spacing,
        temperature=1490.0,
        seed=17,
        direction=(0.0, -1.0, 0.0),
        profile="ellipse",
    ):
        if min(half_width, thickness, speed, spacing) <= 0:
            raise ValueError("Inlet dimensions, speed and spacing must be positive")
        if profile not in ["ellipse", "slot"]:
            raise ValueError("Unknown inlet cross-section profile")
        self.center = np.asarray(center, dtype="f")
        self.speed = float(speed)
        self.spacing = float(spacing)
        self.direction = np.asarray(direction, dtype="f")
        if np.linalg.norm(self.direction) < 1.0e-8 or abs(self.direction[2]) > 1.0e-6:
            raise ValueError("Use a nonzero horizontal inlet direction")
        self.direction /= np.linalg.norm(self.direction)
        lateral = np.cross(self.direction, [0.0, 0.0, 1.0])
        xx = np.arange(-half_width + spacing * 0.5, half_width, spacing)
        zz = np.arange(spacing * 0.5, thickness, spacing)
        xx, zz = np.meshgrid(xx, zz, indexing="ij")
        # A rounded cross section gives a thick core and uneven banks.
        valid = (xx / half_width) ** 2 + (
            (zz - thickness * 0.45) / (thickness * 0.65)
        ) ** 2 < 1.0
        if profile == "slot":
            height = thickness * (
                0.8
                + 0.12 * np.cos(3.5 * xx / half_width)
                + 0.08 * np.sin(8.0 * xx / half_width)
            )
            height *= np.clip((1.0 - np.abs(xx) / half_width) * 5.0, 0.0, 1.0) ** 0.25
            valid = zz < height
        self.section = (
            xx[valid, None] * lateral + zz[valid, None] * np.array([0.0, 0.0, 1.0])
        ).astype("f")
        if len(self.section) == 0:
            raise ValueError("The inlet cross section needs finer particle spacing")
        self.temperature = temperature
        self.distance = 0.0
        self.layers = 0
        self.injected = 0
        self.rng = np.random.default_rng(seed)

    @property
    def volume_rate(self):
        return len(self.section) * self.spacing**2 * self.speed

    def advance(self, mpm, dt):
        self.distance += self.speed * dt
        while self.distance >= self.spacing:
            self.distance -= self.spacing
            p = self.section + self.center
            p += self.direction * self.distance
            p += self.rng.uniform(-0.08, 0.08, p.shape) * self.spacing
            temp = self.temperature + 15 * np.sin(p[:, 0] * 8.0 + self.layers * 0.15)
            v = np.broadcast_to(self.direction * self.speed, p.shape).copy()
            begin = mpm.count
            mpm.append(p, temp, v)
            # Material coordinates continue upstream rather than repeating at
            # each injected slice. The shader is advected with these particles.
            reference = p.copy() - self.direction * self.layers * self.spacing
            wp.to_torch(mpm.reference)[begin : mpm.count] = torch.as_tensor(
                reference, device=mpm.device
            )
            self.layers += 1
            self.injected += len(p)

    def seed(self, mpm):
        self.advance(mpm, self.spacing / self.speed + 1.0e-8)
