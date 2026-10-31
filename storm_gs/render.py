"""Depth-bounded volumetric scattering over a calibrated 3DGS environment."""
import math
import numpy as np
import torch
import warp as wp
from .solver import Parameters, sample4


@wp.func
def in_grid(p: wp.vec3, c: Parameters):
    q = (p - c.origin) / c.dx
    return (
        q[0] >= 0.0
        and q[1] >= 0.0
        and q[2] >= 0.0
        and q[0] < float(c.nx)
        and q[1] < float(c.ny)
        and q[2] < float(c.nz)
    )


@wp.kernel
def extinction(
    a: wp.array3d(dtype=wp.vec4),
    dust: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    out: wp.array3d(dtype=wp.vec4),
    cloud_scale: float,
    dust_scale: float,
):
    i, j, k = wp.tid()
    factor = float(1 - solid[i, j, k])
    edge = wp.min(
        wp.min(float(i), float(a.shape[0] - 1 - i)),
        wp.min(float(j), float(a.shape[1] - 1 - j)),
    )
    edge = wp.min(edge, float(a.shape[2] - 1 - k))
    factor *= wp.smoothstep(0.0, 7.0, edge)
    out[i, j, k] = wp.vec4(
        wp.max(a[i, j, k][2], 0.0) * cloud_scale * factor,
        wp.max(dust[i, j, k], 0.0) * dust_scale * factor,
        1.0,
        0.0,
    )


@wp.kernel
def illumination(
    src: wp.array3d(dtype=wp.vec4),
    out: wp.array3d(dtype=wp.vec4),
    sun: wp.vec3,
    c: Parameters,
):
    i, j, k = wp.tid()
    q = wp.vec3(float(i), float(j), float(k))
    tau = float(0.0)
    ds = c.dx * 2.0
    for n in range(28):
        at = q + sun * (float(n) + 0.5) * 2.0
        if (
            at[0] >= 0.0
            and at[1] >= 0.0
            and at[2] >= 0.0
            and at[0] < float(c.nx - 1)
            and at[1] < float(c.ny - 1)
            and at[2] < float(c.nz - 1)
        ):
            density = sample4(src, at)
            tau += (density[0] + density[1]) * ds
    s = src[i, j, k]
    out[i, j, k] = wp.vec4(s[0], s[1], wp.exp(-tau * 0.65), 0.0)


@wp.func
def fbm(p: wp.vec3):
    return (
        wp.noise(wp.uint32(1927), p) * 0.64
        + wp.noise(wp.uint32(513), p * 2.07 + wp.vec3(5.2, 1.3, 2.1)) * 0.26
        + wp.noise(wp.uint32(71), p * 4.13) * 0.10
    )


@wp.func
def sky(ray: wp.vec3, t: float, storm: float):
    h = wp.clamp(ray[2], 0.0, 1.0)
    base = wp.lerp(wp.vec3(0.65, 0.68, 0.69), wp.vec3(0.24, 0.31, 0.39), wp.sqrt(h))
    if storm < 0.1:
        base = wp.lerp(
            wp.vec3(0.73, 0.69, 0.60), wp.vec3(0.26, 0.43, 0.62), h / (h + 0.35)
        )
    else:
        q = ray / wp.max(0.09, ray[2] + 0.18)
        n = fbm(q * 1.8 + wp.vec3(t * 0.025, -t * 0.018, 1.8))
        base *= wp.clamp(0.68 + n * 0.6, 0.35, 0.95)
    return base


@wp.kernel
def composite(
    rgb: wp.array2d(dtype=wp.vec3),
    alpha: wp.array2d(dtype=float),
    depth: wp.array2d(dtype=float),
    rays: wp.array2d(dtype=wp.vec3),
    field: wp.array3d(dtype=wp.vec4),
    particle_rgb: wp.array2d(dtype=wp.vec3),
    particle_alpha: wp.array2d(dtype=float),
    particle_depth: wp.array2d(dtype=float),
    out: wp.array2d(dtype=wp.vec3),
    volume_alpha: wp.array2d(dtype=float),
    origin: wp.vec3,
    c: Parameters,
    t: float,
    dust_color: wp.vec3,
    exposure: float,
    ocean: int,
    storm: float,
):
    y, x = wp.tid()
    ray = rays[y, x]
    surface = depth[y, x]
    sky_color = sky(ray, t, storm)
    background = sky_color
    background_distance = float(1000.0)
    if ocean == 1 and ray[2] < -0.0001:
        water_distance = (0.0 - origin[2]) / ray[2]
        if water_distance > 0.0:
            p = origin + ray * water_distance
            waves = (
                fbm(p * 0.9 + wp.vec3(t * 0.32, t * 0.1, 0.0)) * 0.85
                + fbm(p * 3.5 + wp.vec3(-t * 0.1, t * 0.25, 1.0)) * 0.25
            )
            water = wp.vec3(0.075, 0.14, 0.17) * (1.0 + waves * 0.20)
            fresnel = wp.pow(1.0 - wp.abs(ray[2]), 5.0)
            foam = wp.pow(wp.max(0.0, waves - 0.26), 2.0) * 0.4
            water = wp.lerp(water, sky_color, fresnel * 0.7) + wp.vec3(foam)
            haze = 1.0 - wp.exp(-water_distance * 0.002)
            background = wp.lerp(water, sky_color, haze)
            background_distance = water_distance
    a = alpha[y, x]
    bg = rgb[y, x] * exposure + (1.0 - a) * background
    limit = surface
    if a < 0.03:
        limit = background_distance
    elif a < 0.95:
        limit = wp.lerp(background_distance, wp.min(surface, background_distance), a)
    upper = c.origin + wp.vec3(float(c.nx), float(c.ny), float(c.nz)) * c.dx
    near = float(0.0)
    far = wp.min(limit, 1000.0)
    for axis in range(3):
        direction = ray[axis]
        if wp.abs(direction) < 0.000001:
            direction = 0.000001
        a0 = (c.origin[axis] - origin[axis]) / direction
        a1 = (upper[axis] - origin[axis]) / direction
        near = wp.max(near, wp.min(a0, a1))
        far = wp.min(far, wp.max(a0, a1))
    # The same lighting volume supplies contact shadows on reconstructed surfaces.
    point = origin + ray * surface
    if a > 0.5 and in_grid(point, c):
        light = sample4(field, (point - c.origin) / c.dx - wp.vec3(0.5))[2]
        bg *= 0.78 + 0.22 * light
    trans = float(1.0)
    radiance = wp.vec3(0.0)
    particle_done = bool(False)
    volume_trans = float(1.0)
    pd = particle_depth[y, x]
    pa = particle_alpha[y, x]
    step = c.dx * 0.68
    random = wp.rand_init(193, y * rgb.shape[1] + x)
    distance = near + step * wp.randf(random)
    iteration = int(0)
    while distance < far and trans > 0.008 and iteration < 350:
        if not particle_done and pd < distance and pd < limit and pa > 0.0:
            radiance += trans * particle_rgb[y, x]
            trans *= 1.0 - pa
            particle_done = True
        p = origin + ray * distance
        q = (p - c.origin) / c.dx - wp.vec3(0.5)
        value = sample4(field, q)
        cloud = value[0]
        dust = value[1]
        sigma = cloud + dust
        if sigma > 0.0008:
            radius = wp.sqrt(p[0] * p[0] + p[1] * p[1] + 0.05)
            angle = -t * wp.min(c.swirl / (radius + 2.0), 2.0)
            rotated = wp.vec3(
                wp.cos(angle) * p[0] - wp.sin(angle) * p[1],
                wp.sin(angle) * p[0] + wp.cos(angle) * p[1],
                p[2] - t * c.updraft * 0.5,
            )
            detail = wp.clamp(
                0.95 + 2.1 * fbm(rotated * (1.0 / c.dx) * 0.38), 0.15, 1.8
            )
            sigma *= detail
            ds = wp.min(step, far - distance)
            opacity = 1.0 - wp.exp(-sigma * ds)
            shade = 0.28 + 0.78 * value[2]
            cloud_color = wp.vec3(0.69, 0.73, 0.76)
            albedo = (cloud * cloud_color + dust * dust_color) / wp.max(
                cloud + dust, 0.000001
            )
            radiance += trans * opacity * albedo * shade
            trans *= 1.0 - opacity
            volume_trans *= 1.0 - opacity
        distance += step
        iteration += 1
    if not particle_done and pd < limit and pa > 0.0:
        radiance += trans * particle_rgb[y, x]
        trans *= 1.0 - pa
    out[y, x] = radiance + trans * bg
    volume_alpha[y, x] = 1.0 - volume_trans


class Renderer:
    def __init__(self, scene, cfg, simulation, particles=None):
        self.scene, self.cfg, self.sim = scene, cfg, simulation
        self.particles = particles
        self.field = wp.zeros(simulation.shape, dtype=wp.vec4, device=simulation.device)
        self.lit = wp.zeros_like(self.field)
        self.sun = wp.normalize(wp.vec3(-0.6, -0.4, 0.9))

    @torch.no_grad()
    def render(self, camera):
        sim = self.sim
        cfg = self.cfg
        image, alpha = self.scene.render(
            camera, mask=getattr(self.scene, "storm_mask", None)
        )
        rgb = image[..., :3].clamp(0, 1).contiguous()
        if cfg.get("replace_baked_sky", False):
            world = camera.world_points(image[..., 3])
            color = rgb / alpha.clamp_min(0.001)[..., None]
            brightness = color.min(-1).values
            elevated = ((world[..., 2] - 0.45) / 0.6).clamp(0, 1)
            sky_weight = ((brightness - 0.55) / 0.25).clamp(0, 1) * elevated
            sky_weight = sky_weight.square() * (3 - 2 * sky_weight)
            rgb = (rgb * (1 - sky_weight[..., None])).contiguous()
            alpha = (alpha * (1 - sky_weight)).contiguous()
        ones = torch.ones((camera.height, camera.width), device=rgb.device)
        raw = camera.world_points(ones) - camera.center
        distance_scale = torch.linalg.vector_norm(raw, dim=-1)
        rays = (raw / distance_scale[..., None]).contiguous()
        depth = (image[..., 3] * distance_scale * cfg["scale"]).contiguous()
        if self.particles is not None:
            surface_depth = torch.where(
                alpha > 0.03, image[..., 3], torch.full_like(alpha, 10000)
            )
            pr, pa, pd = self.particles.render(camera, surface_depth)
            pd = (pd * distance_scale * cfg["scale"]).contiguous()
        else:
            pr = torch.zeros_like(rgb)
            pa = torch.zeros_like(alpha)
            pd = torch.ones_like(alpha) * 10000
        origin = (camera.center - camera.center.new_tensor(cfg["center"])) * cfg[
            "scale"
        ]
        out = torch.empty_like(rgb)
        opacity = torch.empty_like(alpha)
        wp.launch(
            extinction,
            sim.shape,
            [
                sim.a,
                sim.dust,
                sim.solid,
                self.field,
                cfg["cloud_extinction"],
                cfg["dust_extinction"],
            ],
            device=sim.device,
        )
        wp.launch(
            illumination,
            sim.shape,
            [self.field, self.lit, self.sun, sim.c],
            device=sim.device,
        )
        wp.launch(
            composite,
            (camera.height, camera.width),
            [
                wp.from_torch(rgb, dtype=wp.vec3),
                wp.from_torch(alpha.contiguous()),
                wp.from_torch(depth),
                wp.from_torch(rays, dtype=wp.vec3),
                self.lit,
                wp.from_torch(pr.contiguous(), dtype=wp.vec3),
                wp.from_torch(pa.contiguous()),
                wp.from_torch(pd),
                wp.from_torch(out, dtype=wp.vec3),
                wp.from_torch(opacity),
                wp.vec3(*origin.tolist()),
                sim.c,
                sim.time,
                wp.vec3(*cfg["dust_color"]),
                cfg["exposure"],
                int(cfg["ocean"]),
                float(sim.settings.cloud),
            ],
            device=sim.device,
        )
        # Warp's solver stream must finish before Torch consumes the output.
        wp.synchronize_device(sim.device)
        horizon = torch.exp(-rays[..., 2].clamp_min(0) / 0.4)
        clean_sky = rgb.new_tensor([0.31, 0.49, 0.69]) + horizon[
            ..., None
        ] * rgb.new_tensor([0.38, 0.29, 0.16])
        original = rgb + (1 - alpha[..., None]) * clean_sky
        return out.clamp(0, 1), opacity, original.clamp(0, 1)
