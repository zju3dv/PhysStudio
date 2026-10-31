"""Inertial rain and leaf fragments driven by the projected atmospheric grid."""
import numpy as np
import torch
import warp as wp
from gsplat import rasterization
from .solver import Parameters, velocity, fluid


@wp.func
def floor_height(floor: wp.array2d(dtype=float), p: wp.vec3, c: Parameters):
    q = (p - c.origin) / c.dx - wp.vec3(0.5)
    x = wp.clamp(q[0], 0.0, float(floor.shape[0] - 1) - 0.001)
    y = wp.clamp(q[1], 0.0, float(floor.shape[1] - 1) - 0.001)
    i, j = int(x), int(y)
    return wp.lerp(
        wp.lerp(floor[i, j], floor[i + 1, j], x - float(i)),
        wp.lerp(floor[i, j + 1], floor[i + 1, j + 1], x - float(i)),
        y - float(j),
    )


@wp.kernel
def initialize_particles(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    age: wp.array(dtype=float),
    kind: wp.array(dtype=int),
    floor: wp.array2d(dtype=float),
    c: Parameters,
    seed: int,
    rain_fraction: float,
):
    i = wp.tid()
    state = wp.rand_init(seed, i)
    rain = int(wp.randf(state) < rain_fraction)
    kind[i] = rain
    angle = wp.randf(state) * 6.2831853
    radius = wp.sqrt(wp.randf(state)) * (c.radius * 3.0 + 0.8)
    p = wp.vec3(radius * wp.cos(angle), radius * wp.sin(angle), 0.0)
    if rain == 1:
        p[0] = wp.randf(state) * float(c.nx - 4) * c.dx + c.origin[0] + 2.0 * c.dx
        p[1] = wp.randf(state) * float(c.ny - 4) * c.dx + c.origin[1] + 2.0 * c.dx
        p[2] = wp.randf(state) * c.height
    else:
        p[2] = floor_height(floor, p, c) + 0.08 + wp.randf(state) * 0.5
    x[i] = p
    v[i] = wp.vec3(0.0)
    age[i] = wp.randf(state) * 2.0


@wp.kernel
def move_particles(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    age: wp.array(dtype=float),
    kind: wp.array(dtype=int),
    u: wp.array3d(dtype=float),
    vg: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    floor: wp.array2d(dtype=float),
    solid: wp.array3d(dtype=int),
    c: Parameters,
    dt: float,
    t: float,
):
    i = wp.tid()
    p = x[i]
    old = p
    q = (p - c.origin) / c.dx
    air = velocity(u, vg, w, q)
    random = wp.rand_init(935, i)
    response = 0.045 + wp.randf(random) * 0.15
    if kind[i] == 1:
        response = 0.63
    decay = wp.exp(-dt / response)
    vn = (
        air + (v[i] - air) * decay + wp.vec3(0.0, 0.0, -9.81 * response * (1.0 - decay))
    )
    p += vn * dt
    z = floor_height(floor, p, c)
    hit = p[2] < z + 0.035
    cell = wp.vec3i((p - c.origin) / c.dx)
    if not hit and fluid(solid, cell[0], cell[1], cell[2]) == 0:
        p = old
        vn *= 0.1
    if hit and kind[i] == 0:
        p[2] = z + 0.035
        vn[2] = wp.max(vn[2], 0.0)
        vn[0] *= 0.75
        vn[1] *= 0.75
    age[i] += dt
    outside = (
        p[0] < c.origin[0] + c.dx
        or p[1] < c.origin[1] + c.dx
        or p[0] > c.origin[0] + float(c.nx - 1) * c.dx
        or p[1] > c.origin[1] + float(c.ny - 1) * c.dx
        or p[2] > c.origin[2] + float(c.nz - 2) * c.dx
    )
    if outside or age[i] > 12.0 or (hit and kind[i] == 1):
        state = wp.rand_init(517 + int(t * 1000.0), i)
        angle = wp.randf(state) * 6.2831853
        radius = wp.sqrt(wp.randf(state)) * (c.radius * 3.0 + 0.8)
        p = wp.vec3(radius * wp.cos(angle), radius * wp.sin(angle), 0.0)
        p[2] = floor_height(floor, p, c) + 0.07
        if kind[i] == 1:
            p[0] = wp.randf(state) * float(c.nx - 6) * c.dx + c.origin[0] + 3.0 * c.dx
            p[1] = wp.randf(state) * float(c.ny - 6) * c.dx + c.origin[1] + 3.0 * c.dx
            p[2] = c.height * 0.95
        vn = velocity(u, vg, w, (p - c.origin) / c.dx)
        age[i] = 0.0
    x[i] = p
    v[i] = vn


class Particles:
    def __init__(self, sim, cfg, seed=31):
        self.sim, self.cfg = sim, cfg
        n = cfg["particles"]
        self.x = wp.zeros(n, dtype=wp.vec3, device=sim.device)
        self.v = wp.zeros_like(self.x)
        self.age = wp.zeros(n, dtype=float, device=sim.device)
        self.kind = wp.zeros(n, dtype=int, device=sim.device)
        wp.launch(
            initialize_particles,
            n,
            [
                self.x,
                self.v,
                self.age,
                self.kind,
                sim.floor,
                sim.c,
                seed,
                cfg["rain_fraction"],
            ],
            device=sim.device,
        )
        rng = np.random.default_rng(seed)
        self.axes = torch.tensor(
            rng.normal(size=(n, 3)), device="cuda", dtype=torch.float32
        )
        self.axes = torch.nn.functional.normalize(self.axes, dim=-1)
        self.sizes = torch.tensor(
            rng.uniform(0.007, 0.031, n), device="cuda", dtype=torch.float32
        )
        self.phase = torch.tensor(
            rng.uniform(0, 6.28, n), device="cuda", dtype=torch.float32
        )
        self.colors = torch.tensor(
            rng.uniform(0.65, 1.15, (n, 1)) * np.array(cfg["dust_color"]),
            device="cuda",
            dtype=torch.float32,
        )

    def step(self, dt):
        s = self.sim
        wp.launch(
            move_particles,
            len(self.x),
            [
                self.x,
                self.v,
                self.age,
                self.kind,
                s.u,
                s.v,
                s.w,
                s.floor,
                s.solid,
                s.c,
                dt,
                s.time,
            ],
            device=s.device,
        )

    @torch.no_grad()
    def render(self, cam, depth):
        wp.synchronize_device(self.sim.device)
        positions = wp.to_torch(self.x)
        velocity = wp.to_torch(self.v)
        rain = wp.to_torch(self.kind).bool()
        scale = self.cfg["scale"]
        means = positions / scale + positions.new_tensor(self.cfg["center"])
        phase = self.phase + self.sim.time * 4.2
        normal = torch.stack(
            [torch.cos(phase), torch.sin(phase), torch.sin(phase * 0.73)], -1
        )
        normal = torch.nn.functional.normalize(normal, dim=-1)
        eye = torch.eye(3, device=means.device)[None]
        size = torch.where(rain, self.sizes * 0.13, self.sizes)
        cov = size[:, None, None].square() * (
            eye - 0.93 * normal[:, :, None] * normal[:, None, :]
        )
        shutter = torch.where(rain, 0.022, 0.008)
        cov += (
            velocity[:, :, None]
            * velocity[:, None, :]
            * shutter[:, None, None].square()
            / 12
        )
        cov /= scale * scale
        colors = torch.where(
            rain[:, None], self.colors.new_tensor([0.63, 0.69, 0.73]), self.colors
        )
        opacity = torch.where(
            rain, torch.full_like(size, 0.18), torch.full_like(size, 0.62)
        )
        ij = (
            (positions - positions.new_tensor(self.sim.origin)) / self.sim.settings.dx
        ).long()
        floor = wp.to_torch(self.sim.floor)[
            ij[:, 0].clamp(0, self.sim.shape[0] - 1),
            ij[:, 1].clamp(0, self.sim.shape[1] - 1),
        ]
        airborne = ((positions[:, 2] - floor - 0.06) / 0.32).clamp(0, 1)
        opacity *= torch.where(rain, torch.ones_like(airborne), airborne)
        image, alpha, _ = rasterization(
            means=means,
            quats=None,
            scales=None,
            covars=cov,
            opacities=opacity,
            colors=colors,
            viewmats=cam.view[None],
            Ks=cam.K[None],
            width=cam.width,
            height=cam.height,
            near_plane=0.02,
            far_plane=1000,
            render_mode="RGB+ED",
            packed=False,
            rasterize_mode="antialiased",
        )
        image = image[0]
        alpha = alpha[0, ..., 0]
        visible = (image[..., 3] < depth) & (image[..., 3] > 0.01)
        return image[..., :3] * visible[..., None], alpha * visible, image[..., 3]
