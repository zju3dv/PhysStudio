"""Render reproducible rain, wet ground, reflections and impact spray on FieryGS."""

from physstudio.paths import data_path
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp
import torch
import torch.nn.functional as F
from gsplat import rasterization

from .scene import Scene, Camera, save_image
from .physics import Terrain, ShallowWater, Rain, ImpactWaves, sample_grid


def sample_image(image, grid):
    return F.grid_sample(
        image.permute(2, 0, 1)[None],
        grid[None],
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0].permute(1, 2, 0)


class Compositor:
    def __init__(
        self,
        camera,
        buffers,
        terrain,
        wetness=0.8,
        haze=0.008,
        metres_per_unit=2.5,
        surface_treatment="uniform",
        overcast=0.0,
        sky_overcast=0.0,
    ):
        self.camera, self.b, self.terrain = camera, buffers, terrain
        self.wetness, self.haze = wetness, haze
        self.metres_per_unit = metres_per_unit
        self.surface_treatment, self.overcast = surface_treatment, overcast
        self.sky_overcast = sky_overcast
        self.vegetation = buffers.get("vegetation", torch.zeros_like(buffers["depth"]))
        h, w = camera.height, camera.width
        yy, xx = torch.meshgrid(
            torch.arange(h, device=terrain.device) + 0.5,
            torch.arange(w, device=terrain.device) + 0.5,
            indexing="ij",
        )
        self.screen_grid = torch.stack((xx / w * 2 - 1, yy / h * 2 - 1), -1)
        self.xy = buffers["world"][..., :2].reshape(-1, 2)
        in_domain = (
            ((self.xy >= terrain.bounds[:2]) & (self.xy < terrain.bounds[2:]))
            .all(-1)
            .reshape(h, w)
        )
        relative = (
            buffers["world"][..., 2]
            - self.xy.reshape(h, w, 2) @ terrain.ground[:2]
            - terrain.ground[2]
        )
        self.relative = relative
        self.local_puddles = "reflection_stack" in buffers
        ground_range = 0.20 if self.local_puddles else 0.085
        self.ground_mask = (
            (relative.abs() < ground_range)
            & (buffers["normal"][..., 2] > 0.6)
            & (buffers["alpha"] > 0.8)
            & in_domain
        ).float()
        self.ground_mask = F.avg_pool2d(self.ground_mask[None, None], 3, 1, 1)[0, 0]
        self.ground_normal = F.normalize(
            torch.cat((-terrain.ground[:2], terrain.ground.new_ones(1))), dim=0
        )
        self.view_direction = F.normalize(camera.center - buffers["world"], dim=-1)
        self.rays = (
            camera.world_points(torch.ones_like(buffers["depth"])) - camera.center
        )
        self.camera_plane_height = (
            camera.center[2]
            - camera.center[:2] @ terrain.ground[:2]
            - terrain.ground[2]
        )
        self.ray_plane_slope = (
            self.rays[..., 2] - self.rays[..., :2] @ terrain.ground[:2]
        )
        self.weather_depth = buffers["depth"]
        self.reflection = F.avg_pool2d(
            buffers["reflection"].permute(2, 0, 1)[None], 3, 1, 1
        )[0].permute(1, 2, 0)
        if self.local_puddles:
            self.reflections = F.avg_pool2d(
                buffers["reflection_stack"].permute(0, 3, 1, 2), 3, 1, 1
            ).permute(0, 2, 3, 1)
        if sky_overcast:
            self.reflection = self.cloud_sky(
                self.reflection.pow(2.2), buffers["reflection_sky"]
            ).pow(1 / 2.2)
            if self.local_puddles:
                self.reflections = self.cloud_sky(
                    self.reflections.pow(2.2), buffers["reflection_sky_stack"]
                ).pow(1 / 2.2)

    def cloud_sky(self, linear, weight):
        blend = (weight * self.sky_overcast)[..., None]
        return linear * (1 - blend) + linear.new_tensor([0.26, 0.29, 0.32]) * blend

    def intersect_water(self, surface, depth_grid):
        """Intersect camera rays with the evolving height field before compositing."""
        h, w = self.camera.height, self.camera.width
        denominator = self.ray_plane_slope.clamp_max(-1e-6)
        # Bracket the height-field crossing; fixed-point updates can oscillate
        # on noisy 3DGS ground and create hard holes in shallow puddles.
        near = ((surface.max() - self.camera_plane_height) / denominator).clamp_min(0)
        far = ((surface.min() - self.camera_plane_height) / denominator).clamp_min(0)
        for _ in range(16):
            distance = (near + far) * 0.5
            hit = self.camera.center + self.rays * distance[..., None]
            xy = hit[..., :2].reshape(-1, 2)
            height = sample_grid(surface, xy, self.terrain.bounds).reshape(h, w)
            above = self.camera_plane_height + self.ray_plane_slope * distance > height
            near = torch.where(above, distance, near)
            far = torch.where(above, far, distance)
        height = sample_grid(surface, xy, self.terrain.bounds).reshape(h, w)
        depth = sample_grid(depth_grid, xy, self.terrain.bounds).reshape(h, w)
        residual = (
            hit[..., 2]
            - hit[..., :2] @ self.terrain.ground[:2]
            - self.terrain.ground[2]
            - height
        ).abs()
        domain = (
            ((xy >= self.terrain.bounds[:2]) & (xy < self.terrain.bounds[2:]))
            .all(-1)
            .reshape(h, w)
        )
        visible = (
            (distance > 0.02)
            & domain
            & (self.ray_plane_slope < 0)
            & (residual < 0.003)
            & (self.b["alpha"] > 0.8)
        )
        occlusion = ((self.b["depth"] - distance) / 0.006 + 0.5).clamp(0, 1)
        occlusion = occlusion.square() * (3 - 2 * occlusion)
        return xy, height, depth, visible.float() * occlusion, distance

    def water(self, water, waves, t, variant="full"):
        rgb = self.b["rgb"]
        h, w = rgb.shape[:2]
        if variant == "dry":
            return rgb, torch.zeros((h, w), device=rgb.device)
        surface, depth_grid = self.terrain.water_surface(water)
        perturbation = waves.height
        # The resolved water level supplies slow flow; wave detail supplies the
        # linearized, impact-driven ripple normal over that same world grid.
        eta = surface + perturbation
        xy = self.xy
        visibility = self.ground_mask
        if self.local_puddles:
            xy, surface_height, puddle_depth, visibility, water_distance = (
                self.intersect_water(eta, depth_grid)
            )
        else:
            puddle_depth = sample_grid(depth_grid, xy, self.terrain.bounds).reshape(
                h, w
            )
            surface_height = sample_grid(surface, xy, self.terrain.bounds).reshape(h, w)
        dx = self.terrain.dx
        gy, gx = torch.gradient(eta, spacing=(dx, dx))
        nx = -sample_grid(gx, xy, self.terrain.bounds).reshape(h, w)
        ny = -sample_grid(gy, xy, self.terrain.bounds).reshape(h, w)
        nx = nx.clamp(-0.22, 0.22)
        ny = ny.clamp(-0.22, 0.22)
        normal = F.normalize(
            torch.stack(
                (
                    nx - self.terrain.ground[0],
                    ny - self.terrain.ground[1],
                    torch.ones_like(nx),
                ),
                -1,
            ),
            dim=-1,
        )
        coverage = (puddle_depth / (0.004 if self.local_puddles else 0.008)).clamp(0, 1)
        coverage = coverage.square() * (3 - 2 * coverage) * visibility
        self.weather_depth = self.b["depth"]
        if self.local_puddles and variant == "full":
            self.weather_depth = torch.where(
                coverage > 0.5, water_distance, self.b["depth"]
            )
        linear = rgb.clamp_min(0).pow(2.2)
        luminance = (linear * linear.new_tensor([0.2126, 0.7152, 0.0722])).sum(
            -1, keepdim=True
        )
        # Static PGSR has baked illumination; this is a wet-weather color
        # treatment, not a recovered-albedo relighting model.
        treated = (0.94 * linear + 0.06 * luminance) * linear.new_tensor(
            [0.94, 0.98, 1.02]
        )
        treated *= (1 - self.wetness * (0.06 + 0.22 * self.ground_mask))[..., None]
        if self.surface_treatment == "vegetation":
            # Wet blades/leaves retain their hue but their diffuse response
            # darkens. The fixed 3D feature avoids a camera-dependent 2D mask.
            leaf_wetness = (
                self.wetness * self.vegetation * (0.38 + 0.24 * self.ground_mask)
            )
            treated *= (1 - leaf_wetness)[..., None]
            soil_wetness = (
                self.wetness * 0.20 * self.ground_mask * (1 - self.vegetation)
            )
            treated *= (1 - soil_wetness)[..., None]
        # Compress baked bright sunlight for a more coherent cloudy appearance.
        # This is a color treatment; it does not reconstruct environment light.
        cloud = 1 - self.overcast * (0.10 + 0.38 * luminance.clamp(0, 1))
        treated *= cloud
        if self.sky_overcast:
            treated = self.cloud_sky(treated, self.b["sky"])
        cos_theta = (normal * self.view_direction).sum(-1).abs().clamp(0, 1)
        fresnel = 0.02037 + 0.97963 * (1 - cos_theta).pow(5)
        disturbance = normal - self.ground_normal
        camera_normal = disturbance @ self.camera.view[:3, :3].T
        shift = (
            camera_normal[..., :2]
            * camera_normal.new_tensor(
                [self.camera.K[0, 0] / w, self.camera.K[1, 1] / h]
            )
            * 1.8
        )
        shift = shift.clamp(-0.025, 0.025)
        reflection = (
            sample_image(self.reflection, self.screen_grid + shift)
            .clamp_min(0)
            .pow(2.2)
        )
        if self.local_puddles:
            levels = self.b["reflection_levels"]
            spacing = levels[1] - levels[0]
            height = surface_height.clamp(float(levels[0]), float(levels[-1]))
            reflection = torch.zeros_like(reflection)
            for level, texture in zip(levels, self.reflections):
                weight = (1 - (height - level).abs() / spacing).clamp(0, 1)
                reflection += (
                    sample_image(texture, self.screen_grid + shift)
                    .clamp_min(0)
                    .pow(2.2)
                    * weight[..., None]
                )
        reflection = reflection * reflection.new_tensor([0.81, 0.88, 0.94])
        reflection *= 1 - self.overcast * 0.2
        refract_shift = shift * (puddle_depth / 0.025).clamp(0, 1)[..., None] * 0.12
        refraction = sample_image(treated, self.screen_grid - refract_shift)
        # Dielectric reflection/transmission are blended in linear radiance.
        liquid = refraction * (1 - fresnel[..., None]) + reflection * fresnel[..., None]
        result = treated * (1 - coverage[..., None]) + liquid * coverage[..., None]
        if variant == "streaks":
            result = treated
        distance = self.b["depth"] * self.metres_per_unit
        transmission = torch.exp(-distance * self.haze)
        airlight = result.new_tensor([0.27, 0.32, 0.37])
        result = result * transmission[..., None] + airlight * (
            1 - transmission[..., None]
        )
        return result.clamp(0, 1).pow(1 / 2.2), coverage

    def rain(self, image, rain, exposure, brightness=1.0):
        means, covariance, opacity = rain.streaks(exposure)
        camera = self.camera
        xyz = means @ camera.view[:3, :3].T + camera.view[:3, 3]
        z = xyz[:, 2]
        uv = torch.stack(
            (
                camera.K[0, 0] * xyz[:, 0] / z.clamp_min(0.01) + camera.K[0, 2],
                camera.K[1, 1] * xyz[:, 1] / z.clamp_min(0.01) + camera.K[1, 2],
            ),
            -1,
        )
        index = uv.long()
        valid = (
            (z > 0.08)
            & (index[:, 0] >= 0)
            & (index[:, 0] < camera.width)
            & (index[:, 1] >= 0)
            & (index[:, 1] < camera.height)
        )
        ids = torch.nonzero(valid).flatten()
        pixels = index[ids]
        background_z = self.weather_depth[pixels[:, 1], pixels[:, 0]]
        ids = ids[z[ids] < background_z + 0.015]
        if not len(ids):
            return image, dict(
                visible_segments=0, rendered_alpha_mean=0.0, depth_rejected_pixels=0
            )
        pixels = index[ids]
        local_rgb = image[pixels[:, 1], pixels[:, 0]]
        brightness_base = local_rgb.mean(-1)
        colors = (0.62 + 0.32 * brightness_base[:, None]) * image.new_tensor(
            [0.90, 0.96, 1.0]
        )
        # Near droplets are out of focus; world-space defocus adds isotropic
        # covariance before projection rather than blurring the whole image.
        distance = z[ids]
        coc_pixels = (0.45 * (1 / distance.clamp_min(0.1) - 1 / 3).abs()).clamp(max=1.2)
        coc_world = coc_pixels * distance / camera.K[0, 0]
        cov = (
            covariance[ids]
            + torch.eye(3, device=means.device) * coc_world[:, None, None].square()
        )
        color, alpha, _ = rasterization(
            means=means[ids],
            covars=cov,
            quats=None,
            scales=None,
            opacities=(opacity[ids] * brightness).clamp(max=0.8),
            colors=colors,
            viewmats=camera.view[None],
            Ks=camera.K[None],
            width=camera.width,
            height=camera.height,
            packed=False,
            near_plane=0.04,
            eps2d=0.15,
            render_mode="RGB+ED",
        )
        # Test every covered pixel as well as each segment's center. Splitting
        # the exposure prevents a long streak from leaking across silhouettes.
        visible = (color[0, ..., 3] <= self.weather_depth + 0.007)[..., None]
        a = alpha[0] * visible
        return color[0, ..., :3] * visible + image * (1 - a), dict(
            visible_segments=len(ids),
            rendered_alpha_mean=float(a.mean()),
            depth_rejected_pixels=int(((alpha[0] > 0.001) & ~visible).sum()),
        )


class Video:
    def __init__(self, path, width, height, fps):
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, image):
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())

    def close(self):
        self.process.stdin.close()
        if self.process.wait():
            raise RuntimeError("Video encoding failed")


def transition_camera(scene, index, end, progress, width, easing="linear"):
    if end is None:
        return scene.camera(index, width)
    if easing == "smoothstep":
        progress = progress * progress * (3 - 2 * progress)
    a, b = scene.camera(index, width), scene.camera(end, width)
    c2wa, c2wb = (
        torch.linalg.inv(a.view).cpu().numpy(),
        torch.linalg.inv(b.view).cpu().numpy(),
    )
    rotation = Slerp(
        [0, 1], Rotation.from_matrix(np.stack((c2wa[:3, :3], c2wb[:3, :3])))
    )([progress]).as_matrix()[0]
    c2w = np.eye(4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = c2wa[:3, 3] * (1 - progress) + c2wb[:3, 3] * progress
    a.view = torch.linalg.inv(
        torch.tensor(c2w, device=scene.device, dtype=torch.float32)
    )
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scene", type=Path, default=data_path("rainy_gs/fierygs/Playground")
    )
    p.add_argument(
        "--scene-config",
        type=Path,
        help="Scene coordinate calibration, ground support and simulation domain",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1008)
    p.add_argument("--camera", type=int, default=61)
    p.add_argument("--camera-end", type=int)
    p.add_argument(
        "--camera-easing", choices=("linear", "smoothstep"), default="linear"
    )
    p.add_argument("--particles", type=int, default=16000)
    p.add_argument("--rain-mm-hour", type=float, default=40.0)
    p.add_argument("--exposure", type=float, default=1 / 60)
    p.add_argument("--brightness", type=float, default=1.0)
    p.add_argument("--water-level", type=float, default=0.002)
    p.add_argument("--water-init", choices=("global", "basins"), default="global")
    p.add_argument(
        "--puddle-depth-mm",
        type=float,
        default=50.0,
        help="Maximum initial local puddle depth before flow relaxation",
    )
    p.add_argument(
        "--surface-water-mm",
        type=float,
        default=0.0,
        help="Initial retained storm water on exposed terrain before flow relaxation",
    )
    p.add_argument(
        "--grass-drag",
        type=float,
        default=0.0,
        help="Additional momentum damping on the fixed vegetation proxy, in inverse seconds",
    )
    p.add_argument("--wetness", type=float, default=0.8)
    p.add_argument("--haze", type=float, default=0.005)
    p.add_argument(
        "--surface-treatment", choices=("uniform", "vegetation"), default="uniform"
    )
    p.add_argument("--overcast", type=float, default=0.0)
    p.add_argument(
        "--sky-overcast",
        type=float,
        default=0.0,
        help="Blend distant elevated blue sky toward overcast radiance, including reflections",
    )
    p.add_argument("--metres-per-unit", type=float, default=2.5)
    p.add_argument("--wind-x", type=float, default=0.35, help="Wind velocity in m/s")
    p.add_argument("--wind-y", type=float, default=0.12, help="Wind velocity in m/s")
    p.add_argument("--warmup", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save-frames", action="store_true")
    args = p.parse_args()
    if args.frames < 1 or args.fps < 1 or args.width < 32 or args.width % 2:
        p.error("Use positive frames/fps and an even image width of at least 32.")
    if (
        args.particles < 1
        or args.metres_per_unit <= 0
        or args.exposure <= 0
        or args.rain_mm_hour < 0
    ):
        p.error(
            "Particles, metric scale and exposure must be positive; rain rate must be nonnegative."
        )
    if (
        not 0 <= args.overcast <= 1
        or not 0 <= args.wetness <= 1
        or not 0 <= args.sky_overcast <= 1
    ):
        p.error("Overcast and wetness must lie between zero and one.")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    scene_config = (
        json.loads(args.scene_config.read_text()) if args.scene_config else {}
    )
    start = time.monotonic()
    scene = Scene(args.scene, config=scene_config)
    terrain = Terrain(
        scene, bounds=tuple(scene_config.get("bounds", [-4.0, -4.0, 4.0, 4.0]))
    )
    initial_depth = None
    if args.water_init == "basins":
        if args.puddle_depth_mm <= 0:
            p.error("Puddle depth must be positive")
        initial_depth = terrain.initial_puddles(
            args.water_level, args.puddle_depth_mm * 0.001 / args.metres_per_unit
        )
        if args.surface_water_mm < 0:
            p.error("Surface water must be nonnegative")
        initial_depth = (
            torch.maximum(
                initial_depth,
                initial_depth.new_tensor(
                    args.surface_water_mm * 0.001 / args.metres_per_unit
                ),
            )
            * terrain.rain_source_mask
        )
    elif args.surface_water_mm != 0:
        p.error("Surface water requires --water-init basins")
    if args.grass_drag < 0:
        p.error("Grass drag must be nonnegative")
    damping = (
        1.5 + args.grass_drag * terrain.coarse_vegetation if args.grass_drag else 1.5
    )
    water = ShallowWater(
        terrain.coarse_bed,
        (float(terrain.bounds[2] - terrain.bounds[0])) / 128,
        level=args.water_level,
        gravity=9.81 / args.metres_per_unit,
        initial_depth=initial_depth,
        damping=damping,
    )
    waves = ImpactWaves(terrain, gravity=9.81 / args.metres_per_unit)
    rain = Rain(
        terrain,
        args.particles,
        args.seed,
        args.metres_per_unit,
        wind=(args.wind_x, args.wind_y),
    )
    config = {
        k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    config.update(
        gaussians=len(scene.means),
        ground=scene.ground.tolist(),
        source=scene_config.get(
            "source", f"Released Gaussian model and cameras: {args.scene.name}"
        ),
        scene_calibration=scene_config,
        source_ply_sha256=hashlib.sha256(scene.ply.read_bytes()).hexdigest(),
        code_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        physical_scale="Estimated scene-unit conversion, configurable; release does not provide a metric scale",
        initial_water="Pre-existing puddles; local basin initialization and flow relaxation"
        if water.local_puddles
        else "Pre-existing shallow puddles; rainfall source uses the stated mm/hour",
        reflection="Mirror-camera Gaussian rendering with ripple-normal perturbation",
        renderer_particles="Representative visual drops; particle budget and physical rainfall flux are separate controls",
        collision="Gaussian-derived top height field; water surface takes precedence over solid ground",
        vegetation="Fixed per-Gaussian DC-color proxy; not measured material labels",
        comparison="Our dry/streaks-only/full ablation, not official RainyGS outputs",
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    print(
        "Scene loaded",
        len(scene.means),
        "bed",
        float(terrain.bed.min()),
        float(terrain.bed.max()),
        "water wet fraction",
        float((water.u[0] > 0.001).float().mean()),
        flush=True,
    )
    source_rate = args.rain_mm_hour * 0.001 / 3600 / args.metres_per_unit
    _, depth = terrain.water_surface(water)
    for _ in range(round(args.warmup * 120)):
        if water.local_puddles:
            water.step(1 / 120)
            _, depth = terrain.water_surface(water)
        rain.step(1 / 120, waves, depth)
        waves.step(1 / 120, depth)
    camera = transition_camera(
        scene, args.camera, args.camera_end, 0.0, args.width, args.camera_easing
    )
    reflection_levels = (-0.04, 0.02, 0.08, 0.14) if water.local_puddles else (0.0,)
    buffers = scene.buffers(camera, reflection_levels=reflection_levels)
    compositor = Compositor(
        camera,
        buffers,
        terrain,
        args.wetness,
        args.haze,
        args.metres_per_unit,
        args.surface_treatment,
        args.overcast,
        args.sky_overcast,
    )
    video = Video(args.output / "rain_demo.mp4", camera.width, camera.height, args.fps)
    comparison = Video(
        args.output / "comparison.mp4", camera.width * 2, camera.height, args.fps
    )
    records, camera_records = [], []
    for frame in range(args.frames):
        tick = time.monotonic()
        if args.camera_end is not None and frame:
            camera = transition_camera(
                scene,
                args.camera,
                args.camera_end,
                frame / max(1, args.frames - 1),
                args.width,
                args.camera_easing,
            )
            buffers = scene.buffers(camera, reflection_levels=reflection_levels)
            compositor = Compositor(
                camera,
                buffers,
                terrain,
                args.wetness,
                args.haze,
                args.metres_per_unit,
                args.surface_treatment,
                args.overcast,
                args.sky_overcast,
            )
        water.step(1 / args.fps, source_rate * terrain.rain_source_mask)
        _, depth = terrain.water_surface(water)
        for _ in range(4):
            rain.step(1 / (args.fps * 4), waves, depth)
            waves.step(1 / (args.fps * 4), depth)
        wet, coverage = compositor.water(water, waves, rain.time)
        image, stats = compositor.rain(wet, rain, args.exposure, args.brightness)
        if not torch.isfinite(image).all():
            raise FloatingPointError(f"Nonfinite rendered image at frame {frame}")
        pixels = (image.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
        dry = (buffers["rgb"].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
        video.write(pixels)
        panel = Image.fromarray(np.concatenate((dry, pixels), 1))
        draw = ImageDraw.Draw(panel)
        for x, label in [(0, "ORIGINAL 3DGS"), (camera.width, "RAIN + WET SURFACES")]:
            draw.rectangle((x + 12, 12, x + 235, 43), fill=(16, 25, 38))
            draw.text((x + 22, 22), label, fill=(232, 241, 252))
        comparison.write(np.asarray(panel))
        if args.save_frames:
            (args.output / "frames").mkdir(exist_ok=True)
            Image.fromarray(pixels).save(args.output / "frames" / f"{frame:05d}.png")
        if frame in {0, args.frames // 2, args.frames - 1}:
            save_image(args.output / f"rain_{frame:05d}.png", image)
            save_image(args.output / f"dry_{frame:05d}.png", buffers["rgb"])
            save_image(args.output / f"water_mask_{frame:05d}.png", coverage)
            save_image(
                args.output / f"vegetation_{frame:05d}.png", compositor.vegetation
            )
            no_water, _ = compositor.water(water, waves, rain.time, variant="streaks")
            no_water, _ = compositor.rain(
                no_water, rain, args.exposure, args.brightness
            )
            save_image(args.output / f"streaks_{frame:05d}.png", no_water)
        record = dict(
            frame=frame,
            simulation_time=rain.time,
            impacts=rain.impacts,
            active_spray=len(rain.splash_pos),
            water_coverage=float(coverage.mean()),
            ripple_height_max=float(waves.height.abs().max()),
            seconds=time.monotonic() - tick,
            **stats,
        )
        foreground_grass = (compositor.vegetation > 0.65) & (
            torch.arange(camera.height, device=scene.device)[:, None]
            > camera.height * 0.8
        )
        if foreground_grass.any():
            record["foreground_grass_water_blend"] = float(
                coverage[foreground_grass].mean()
            )
            record["foreground_grass_puddle_fraction"] = float(
                (coverage[foreground_grass] > 0.25).float().mean()
            )
        records.append(record)
        camera_records.append(
            dict(
                frame=frame, view=camera.view.cpu().tolist(), K=camera.K.cpu().tolist()
            )
        )
        if frame % 15 == 0:
            print(json.dumps(record), flush=True)
            (args.output / "status.json").write_text(json.dumps(record, indent=2))
    video.close()
    comparison.close()
    (args.output / "camera_path.json").write_text(json.dumps(camera_records))
    torch.save(
        dict(
            bed=terrain.bed.cpu(),
            top=terrain.top.cpu(),
            bounds=terrain.bounds.cpu(),
            water=water.u.cpu(),
            water_bed=water.bed.cpu(),
            local_puddles=water.local_puddles,
            ripple=waves.height.cpu(),
            ripple_velocity=waves.velocity.cpu(),
            particles=rain.pos.cpu(),
            velocity=rain.vel.cpu(),
            diameter=rain.diameter.cpu(),
            splash_pos=rain.splash_pos.cpu(),
            splash_vel=rain.splash_vel.cpu(),
            splash_age=rain.splash_age.cpu(),
            splash_life=rain.splash_life.cpu(),
            generator_state=rain.generator.get_state().cpu(),
            simulation_time=rain.time,
        ),
        args.output / "simulation.pt",
    )
    summary = dict(
        frames=args.frames,
        fps=args.fps,
        elapsed_seconds=time.monotonic() - start,
        mean_frame_seconds=float(np.mean([r["seconds"] for r in records])),
        water=water.diagnostics(),
        impacts=rain.impacts,
        frames_finite=all(math.isfinite(r["ripple_height_max"]) for r in records),
        records=records,
    )
    (args.output / "status.json").write_text(
        json.dumps(dict(status="complete", frames=args.frames), indent=2)
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
