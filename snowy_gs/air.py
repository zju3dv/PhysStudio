"""Sparse airborne aggregates: drag toward gusting wind, flutter, deposition."""

import numpy as np
import torch
import warp as wp
from .mpm import sample, normal
from .surface import density_at


@wp.kernel(enable_backward=False)
def advect(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    hit: wp.array(dtype=int),
    hp: wp.array(dtype=wp.vec3),
    hv: wp.array(dtype=wp.vec3),
    phi: wp.array3d(dtype=float),
    o: wp.vec3,
    dx: float,
    low: wp.vec3,
    high: wp.vec3,
    ground: wp.vec3,
    t: float,
    dt: float,
    step: int,
    wind: float,
    snow: wp.array3d(dtype=float),
    snow_dx: float,
):
    i = wp.tid()
    p = x[i]
    vel = v[i]
    phase = float(i) * 2.399963
    gust = 1.0 + 0.35 * wp.sin(0.8 * t) + 0.16 * wp.sin(t * 1.9)
    air = wp.vec3(
        wind * gust + 0.23 * wp.sin(p[1] * 0.65 + t * 0.7),
        wind * 0.25 + 0.18 * wp.cos(p[0] * 0.8 - t * 0.8),
        0.0,
    )
    terminal = 0.35 + 90.0 * radius[i]
    target = air + wp.vec3(
        0.12 * wp.sin(t * 3.0 + phase), 0.13 * wp.cos(t * 2.5 + phase), -terminal
    )
    vel += (target - vel) * (1.0 - wp.exp(-dt * 4.0))
    p += vel * dt
    for axis in range(2):
        if p[axis] < low[axis]:
            p[axis] += high[axis] - low[axis]
        if p[axis] > high[axis]:
            p[axis] -= high[axis] - low[axis]
    sd = sample(phi, o, dx, p)
    on_snow = density_at(snow, o, snow_dx, p) > 0.11
    floor = ground[0] * p[0] + ground[1] * p[1] + ground[2]
    if sd < radius[i] or p[2] < floor or on_snow:
        if on_snow:
            n = -normal(snow, o, snow_dx, p)
            if n[2] > 0.35:
                hit[i] = 1
                hp[i] = p + n * 0.012
                hv[i] = vel
        elif sd < radius[i]:
            n = normal(phi, o, dx, p)
            if n[2] > 0.35:
                hit[i] = 1
                hp[i] = p + n * (radius[i] - sd + 0.008)
                hv[i] = vel
        state = wp.rand_init(19 + i, step)
        p = wp.vec3(
            wp.randf(state, low[0], high[0]), wp.randf(state, low[1], high[1]), high[2]
        )
    x[i] = p
    v[i] = vel


class AirSnow:
    def __init__(self, mpm, geo, count, cameras, seed=19, wind=0.45):
        self.mpm = mpm
        self.wind = wind
        self.device = mpm.device
        self.time = 0.0
        self.steps = 0
        b = geo["bounds"]
        self.low = np.array([b[0] - 3, b[1] - 3, geo["origin"][2]], "f")
        self.high = np.array(
            [
                b[2] + 3,
                b[3] + 3,
                geo["origin"][2] + geo["phi"].shape[2] * geo["dx"] + 3,
            ],
            "f",
        )
        cams = np.asarray(cameras)
        self.low[:2] = np.minimum(self.low[:2], cams[:, :2].min(0) - 3)
        self.high[:2] = np.maximum(self.high[:2], cams[:, :2].max(0) + 3)
        rng = np.random.default_rng(seed)
        positions = rng.uniform(self.low, self.high, (count, 3)).astype("f")
        radius = np.clip(
            rng.lognormal(np.log(0.0033), 0.55, count), 0.0008, 0.009
        ).astype("f")
        velocity = np.tile(np.array([wind, wind * 0.25, -0.7], "f"), (count, 1))
        self.x = wp.array(positions, dtype=wp.vec3, device=self.device)
        self.v = wp.array(velocity, dtype=wp.vec3, device=self.device)
        self.r = wp.array(radius, dtype=float, device=self.device)
        self.hit = wp.zeros(count, dtype=int, device=self.device)
        self.hp = wp.zeros(count, dtype=wp.vec3, device=self.device)
        self.hv = wp.zeros(count, dtype=wp.vec3, device=self.device)
        self.plane = geo["plane"]
        self.total_hits = 0
        self.snow = wp.zeros((2, 2, 2), dtype=float, device=self.device)
        self.snow_dx = mpm.dx

    def link_surface(self, surface):
        self.snow = surface.density
        self.snow_dx = surface.dx

    def advance(self, duration):
        self.hit.zero_()
        for _ in range(4):
            wp.launch(
                advect,
                len(self.x),
                [
                    self.x,
                    self.v,
                    self.r,
                    self.hit,
                    self.hp,
                    self.hv,
                    self.mpm.phi,
                    wp.vec3(*self.mpm.origin),
                    self.mpm.dx,
                    wp.vec3(*self.low),
                    wp.vec3(*self.high),
                    wp.vec3(*self.plane),
                    self.time,
                    duration / 4,
                    self.steps,
                    self.wind,
                    self.snow,
                    self.snow_dx,
                ],
                device=self.device,
            )
            self.time += duration / 4
            self.steps += 1
        ids = torch.where(wp.to_torch(self.hit) > 0)[0]
        self.total_hits += len(ids)
        # Resolved MPM packets aggregate every 32nd visual impact. Their mass
        # is recorded explicitly; this is a controllable deposition source.
        ids = ids[ids % 32 == 0]
        if len(ids):
            p = wp.to_torch(self.hp)[ids]
            g = (p - p.new_tensor(self.mpm.origin)) / self.mpm.dx
            keep = ((g > 3) & (g < g.new_tensor(self.mpm.shape) - 4)).all(-1)
            ids = ids[keep]
            self.mpm.deposit(
                wp.to_torch(self.hp)[ids],
                wp.to_torch(self.hv)[ids] * 0.15,
                (self.mpm.dx * 0.5) ** 3 * 0.02,
            )
