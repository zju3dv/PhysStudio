"""Simulation-driven anisotropic Gaussian surfaces and refractive water shading.

Sand density and the LBM phase field define oriented surface Gaussians. A
second, low-opacity Gaussian volume measures optical thickness. All geometry
buffers use gsplat; the pixel shader adds Fresnel reflection and refraction.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
import torch.nn.functional as F
from gsplat import rasterization
import warp as wp

from .render import Renderer as Fields, reconstruct, smooth_fields
from gaussian_splashing.render import blur, sample
from rainy_gs.scene import Camera
from rainy_gs.demo import Video


def surfaces(field, level, spacing, origin):
    """One surface Gaussian per crossed cell, centered on its edge crossings."""
    corners = np.array(
        [[i, j, k] for i in range(2) for j in range(2) for k in range(2)], np.int32
    )
    shape = np.array(field.shape) - 1
    values = [
        field[c[0] : c[0] + shape[0], c[1] : c[1] + shape[1], c[2] : c[2] + shape[2]]
        for c in corners
    ]
    low = np.minimum.reduce(values)
    high = np.maximum.reduce(values)
    cells = np.argwhere((low < level) & (high >= level))
    if len(cells) == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32), cells
    v = np.stack([a[tuple(cells.T)] for a in values], -1)
    offset = np.zeros((len(cells), 3), np.float32)
    count = np.zeros(len(cells), np.float32)
    for a in range(8):
        for b in range(a + 1, 8):
            if np.abs(corners[a] - corners[b]).sum() != 1:
                continue
            cross = (v[:, a] < level) != (v[:, b] < level)
            t = np.zeros(len(cells), np.float32)
            t[cross] = (level - v[cross, a]) / (v[cross, b] - v[cross, a])
            offset += cross[:, None] * (
                corners[a] + t[:, None] * (corners[b] - corners[a])
            )
            count += cross
    lattice = (cells + offset / count[:, None]).astype(np.float32)
    grad = np.stack(
        [
            ndimage.map_coordinates(g, lattice.T, order=1, mode="nearest")
            for g in np.gradient(field)
        ],
        -1,
    )
    normal = -grad / np.maximum(np.linalg.norm(grad, axis=-1, keepdims=True), 1e-9)
    return (
        (lattice * spacing + origin).astype(np.float32),
        normal.astype(np.float32),
        lattice,
    )


def environment(d):
    sky = d.new_tensor([0.16, 0.24, 0.32]) * (0.75 + 0.25 * d[..., 2:3].clamp_min(0))
    panel = torch.exp(
        -(((d[..., 0] + 0.32) / 0.52) ** 8) - ((d[..., 1] + 0.28) / 0.20) ** 8
    )
    panel *= ((d[..., 2] - 0.20) * 8).clamp(0, 1)
    strip = torch.exp(
        -(((d[..., 0] - 0.62) / 0.075) ** 2) - ((d[..., 1] - 0.20) / 0.45) ** 8
    )
    return (
        sky
        + panel[..., None] * d.new_tensor([1.8, 1.72, 1.57])
        + strip[..., None] * d.new_tensor([0.6, 0.9, 1.3])
    )


def orbit(width, height, progress, device):
    t = progress**2 * (3 - 2 * progress)
    az, el = np.deg2rad([-112 + 62 * t, 29 + 7 * np.sin(progress * np.pi)])
    radius = 6.9 - 0.4 * np.sin(progress * np.pi)
    target = np.array([1.8, 1.15, 0.4])
    eye = target + radius * np.array(
        [np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]
    )
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.stack([right, down, forward], -1)
    c2w[:3, 3] = eye
    focal = width / (2 * np.tan(np.deg2rad(43) * 0.5))
    cam = Camera(
        torch.tensor(np.linalg.inv(c2w), device=device),
        torch.tensor(
            [[focal, 0, width * 0.5], [0, focal, height * 0.5], [0, 0, 1]],
            dtype=torch.float32,
            device=device,
        ),
        width,
        height,
    )
    return cam, dict(eye=eye.tolist(), target=target.tolist(), focal=focal)


class GaussianRenderer:
    def __init__(self, cfg, rest, width, height, device="cuda:0"):
        self.cfg, self.device = cfg, device
        self.fields = Fields(cfg, rest, width, height)
        self.width, self.height = width, height
        self.dx = cfg["dx"]
        self.origin = -3 * self.dx
        self.extent = (np.array(cfg["shape"]) - 6) * self.dx
        self.eye3 = torch.eye(3, device=device)

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def make_camera(self, progress):
        return orbit(self.width, self.height, progress, self.device)

    def sand_features(self, color, normals, wet):
        return torch.cat([color, normals, wet[:, None]], -1)

    def water_transmission(
        self, cam, state, thickness, depth, background_depth, refracted
    ):
        attenuation = torch.exp(-thickness[..., None] * self.tensor([0.85, 0.19, 0.09]))
        return refracted * attenuation + self.tensor([0.025, 0.15, 0.18]) * (
            1 - attenuation
        )

    def shade_sand(self, cam, image, alpha):
        albedo = image[..., :3] / alpha.clamp_min(1e-6)[..., None]
        normal = F.normalize(image[..., 3:6], dim=-1)
        moisture = image[..., 6] / alpha.clamp_min(1e-6)
        world = cam.world_points(image[..., -1])
        view = F.normalize(cam.center - world, dim=-1)
        normal = torch.where((normal * view).sum(-1, keepdim=True) < 0, -normal, normal)
        light = F.normalize(self.tensor([-0.6, -0.4, 1.0]), dim=0)
        half = F.normalize(light + view, dim=-1)
        diffuse = 0.37 + 0.63 * (normal * light).sum(-1).clamp(0, 1)
        grain = torch.frac(
            torch.sin((world * 450).floor() @ self.tensor([127.1, 311.7, 74.7]))
            * 43758.5453
        ).abs()
        sand = (
            albedo
            * ((0.84 + 0.26 * grain) * (1 - 0.46 * moisture) * diffuse)[..., None]
        )
        sand += ((normal * half).sum(-1).clamp(0, 1) ** 65 * (0.02 + 0.07 * moisture))[
            ..., None
        ]
        return sand, normal

    def covariances(self, normals, spacing, tangent=0.72, normal=0.13):
        n = self.tensor(normals)
        return spacing**2 * (
            tangent**2 * self.eye3
            + (normal**2 - tangent**2) * n[..., :, None] * n[..., None, :]
        )

    def splat(self, cam, x, cov, features, opacity=0.995, depth=True):
        if isinstance(opacity, (float, int)):
            opacity = torch.full((len(x),), opacity, device=self.device)
        im, alpha, _ = rasterization(
            means=x,
            quats=None,
            scales=None,
            covars=cov,
            opacities=opacity,
            colors=features,
            viewmats=cam.view[None],
            Ks=cam.K[None],
            width=cam.width,
            height=cam.height,
            packed=False,
            render_mode="RGB+ED" if depth else "RGB",
            near_plane=0.02,
            far_plane=50.0,
        )
        return im[0], alpha[0, ..., 0]

    def sand_surface(self, state):
        f = self.fields
        f.den.zero_()
        f.wet.zero_()
        f.pigment.zero_()
        x = wp.array(state["positions"], dtype=wp.vec3)
        moisture = wp.array(state["moisture"], dtype=float)
        wp.launch(
            reconstruct,
            len(x),
            inputs=[
                x,
                moisture,
                f.colors,
                f.den,
                f.wet,
                f.pigment,
                1 / self.cfg["ppc"] ** 3,
            ],
        )
        for axis in range(3):
            wp.launch(
                smooth_fields,
                f.den.shape,
                inputs=[
                    f.den,
                    f.wet,
                    f.pigment,
                    f.smooth_den,
                    f.smooth_wet,
                    f.smooth_pigment,
                    axis,
                ],
            )
            f.den, f.smooth_den = f.smooth_den, f.den
            f.wet, f.smooth_wet = f.smooth_wet, f.wet
            f.pigment, f.smooth_pigment = f.smooth_pigment, f.pigment
        den, wet, colors = f.den.numpy(), f.wet.numpy(), f.pigment.numpy()
        x, n, lattice = surfaces(den, 0.38, self.dx * 0.5, self.origin)
        d = ndimage.map_coordinates(den, lattice.T, order=1).clip(0.05)
        wet = ndimage.map_coordinates(wet, lattice.T, order=1) / d
        color = np.stack(
            [
                ndimage.map_coordinates(colors[..., k], lattice.T, order=1) / d
                for k in range(3)
            ],
            -1,
        )
        return (
            self.tensor(x),
            self.covariances(n, self.dx * 0.5),
            self.tensor(n),
            self.tensor(color),
            self.tensor(wet).clamp(0, 1),
        )

    def background(self, cam, x, cov):
        rays = (
            cam.world_points(torch.ones((cam.height, cam.width), device=self.device))
            - cam.center
        )
        depth = ((-0.035 - cam.center[2]) / rays[..., 2]).clamp(0.01, 100.0)
        point = cam.center + rays * depth[..., None]
        distance = torch.linalg.norm(
            point[..., :2] - self.tensor(self.extent[:2] * 0.5), dim=-1
        )
        base = self.tensor([0.065, 0.085, 0.105]) + self.tensor(
            [0.05, 0.06, 0.07]
        ) * torch.exp(-distance[..., None] * 0.25)
        edge = torch.minimum(
            torch.minimum(point[..., 0] + 0.10, self.extent[0] + 0.10 - point[..., 0]),
            torch.minimum(point[..., 1] + 0.10, self.extent[1] + 0.10 - point[..., 1]),
        )
        tiles = torch.sin(point[..., 0] * 24) * torch.sin(point[..., 1] * 24)
        tile_edge = (
            torch.minimum(
                torch.remainder(point[..., :2] * 3.0, 1.0),
                1 - torch.remainder(point[..., :2] * 3.0, 1.0),
            )
            .min(-1)
            .values
        )
        grout = torch.exp(-(tile_edge**2) / 0.00018)
        tray = self.tensor([0.24, 0.29, 0.31]) * (
            1 + 0.025 * tiles[..., None] - 0.16 * grout[..., None]
        )
        tray = torch.where(
            (edge < 0.035)[..., None], self.tensor([0.09, 0.12, 0.14]), tray
        )
        base = torch.where((edge > 0)[..., None], tray, base)
        light = self.tensor([-0.6, -0.4, 1.0])
        projection = self.eye3 - light[:, None] * self.tensor([[0, 0, 1.0]])
        shadow = x @ projection.T
        shadow[:, 2] = -0.033
        shadow_cov = projection @ cov @ projection.T + self.eye3 * 0.00016
        _, alpha = self.splat(
            cam, shadow, shadow_cov, torch.zeros_like(shadow), 0.30, False
        )
        base *= 1 - blur(alpha, 11)[..., None] * 0.40
        return base, depth, rays

    @torch.no_grad()
    def draw(self, state, progress, diagnostics=False):
        cam, camera_record = self.make_camera(progress)
        x, cov, normals, color, wet = self.sand_surface(state)
        background, background_depth, rays = self.background(cam, x, cov)
        features = self.sand_features(color, normals, wet)
        sim, sa = self.splat(cam, x, cov, features)
        sand, normal = self.shade_sand(cam, sim, sa)
        sand_depth = sim[..., -1]
        light = F.normalize(self.tensor([-0.6, -0.4, 1.0]), dim=0)
        coverage = (sa / 0.88).clamp(0, 1)
        background = sand * coverage[..., None] + background * (1 - coverage[..., None])
        background_depth = torch.where(sa > 0.5, sand_depth, background_depth)

        phi = state["phi"].astype(np.float32)
        water_x, water_n, _ = surfaces(phi, 0.5, self.dx, self.origin)
        wx = self.tensor(water_x)
        wc = self.covariances(water_n, self.dx, 0.78, 0.13)
        wn = self.tensor(water_n)
        water, wa = self.splat(cam, wx, wc, wn)
        depth = water[..., 3]
        n = F.normalize(water[..., :3], dim=-1)
        valid = (wa > 0.08).float()
        n = F.normalize(
            blur(n * valid[..., None], 5) / blur(valid, 5).clamp_min(1e-6)[..., None],
            dim=-1,
        )
        world = cam.world_points(depth)
        view = F.normalize(cam.center - world, dim=-1)
        n = torch.where((n * view).sum(-1, keepdim=True) < 0, -n, n)
        nv = (n * view).sum(-1).clamp(0, 1)

        # Gaussian optical depth is additive in the low-opacity limit.
        occupancy = np.clip(phi, 0, 1) * np.clip(state["eps"], 0, 1)
        ids = np.argwhere(occupancy > 0.02)
        vx = self.tensor(ids * self.dx + self.origin)
        sigma = 0.65 * self.dx
        vc = (self.eye3 * sigma**2).expand(len(vx), -1, -1).contiguous()
        op = self.tensor(occupancy[tuple(ids.T)]) * 0.018
        _, va = self.splat(cam, vx, vc, torch.zeros_like(vx), op, False)
        thickness = (
            -torch.log1p(-va.clamp(max=0.999)) * self.dx / (0.018 * 2 * np.pi * 0.65**2)
        )
        maximum_path = (
            (background_depth - depth) * torch.linalg.norm(rays, dim=-1)
        ).clamp_min(0)
        thickness = torch.minimum(thickness, maximum_path)
        h, w = cam.height, cam.width
        yy, xx = torch.meshgrid(
            torch.arange(h, device=self.device) + 0.5,
            torch.arange(w, device=self.device) + 0.5,
            indexing="ij",
        )
        screen = torch.stack([xx / w * 2 - 1, yy / h * 2 - 1], -1)
        # Snell's law followed by three background-depth intersection updates.
        # The background buffers contain the Gaussian sand and ceramic support.
        eta = 1 / 1.333
        refract = (
            -eta * view
            + (eta * nv - torch.sqrt((1 - eta**2 * (1 - nv**2)).clamp_min(0)))[
                ..., None
            ]
            * n
        )
        refract_cam = refract @ cam.view[:3, :3].T
        point_cam = (world - cam.center) @ cam.view[:3, :3].T
        uv = screen
        for _ in range(3):
            hit_depth = sample(background_depth[..., None], uv)[..., 0]
            travel = (
                (hit_depth - point_cam[..., 2]) / refract_cam[..., 2].clamp_min(0.05)
            ).clamp(0, 3.0)
            hit = point_cam + travel[..., None] * refract_cam
            projected = hit[..., :2] / hit[..., 2:3].clamp_min(0.1)
            uv = torch.stack(
                [
                    2 * cam.K[0, 0] * projected[..., 0] / w,
                    2 * cam.K[1, 1] * projected[..., 1] / h,
                ],
                -1,
            ).clamp(-1, 1)
        # A single-view buffer has no disoccluded samples. Confine its lookup
        # footprint to a local neighborhood and fade it through thin films.
        uv = (
            screen
            + (uv - screen).clamp(-0.045, 0.045)
            * (thickness / 0.20).clamp(0, 1)[..., None]
        )
        uv = uv.clamp(-1, 1)
        refracted = sample(background, uv)
        refracted_depth = sample(background_depth[..., None], uv)[..., 0]
        refracted = torch.where(
            (refracted_depth < depth - 0.025)[..., None], background, refracted
        )
        body = self.water_transmission(
            cam, state, thickness, depth, background_depth, refracted
        )
        fresnel = 0.0204 + 0.9796 * (1 - nv) ** 5
        reflected = environment(2 * nv[..., None] * n - view)
        liquid = body * (1 - fresnel[..., None]) + reflected * fresnel[..., None]
        half = F.normalize(light + view, dim=-1)
        liquid += ((n * half).sum(-1).clamp(0, 1) ** 220)[..., None] * self.tensor(
            [0.8, 0.76, 0.65]
        )
        visible = torch.sigmoid((background_depth - depth - 0.004) * 250)
        coverage = (wa / 0.86).clamp(0, 1) * visible
        rgb = liquid * coverage[..., None] + background * (1 - coverage[..., None])
        rgb = (rgb * (2.51 * rgb + 0.03) / (rgb * (2.43 * rgb + 0.59) + 0.14)).clamp(
            0, 1
        ) ** (1 / 2.2)
        if not torch.isfinite(rgb).all():
            raise FloatingPointError("Non-finite Gaussian render")
        metrics = dict(
            sand_gaussians=len(x),
            water_surface_gaussians=len(wx),
            water_volume_gaussians=len(vx),
            covariance_min_eigenvalue=float(
                torch.stack(
                    [torch.linalg.eigvalsh(chunk).min() for chunk in cov.split(65536)]
                ).min()
            ),
            finite=True,
        )
        metrics.update(getattr(self, "extra_stats", {}))
        buffers = {}
        if diagnostics:
            buffers = dict(
                sand=background,
                normals=n * 0.5 + 0.5,
                water_alpha=coverage,
                thickness=thickness / 2,
                sand_normals=normal * 0.5 + 0.5,
            )
        return rgb, camera_record, metrics, buffers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a fresh render directory")
    (out / "frames").mkdir(parents=True)
    cfg = json.loads((args.input / "config.json").read_text())
    rest = np.load(args.input / "initial.npz")["positions"]
    paths = sorted((args.input / "states").glob("*.npz"))[:: args.stride]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError("No saved simulation frames")
    wp.init()
    wp.set_device("cuda:0")
    renderer = GaussianRenderer(cfg, rest, args.width, args.height)
    video = Video(out / "demo.mp4", args.width, args.height, cfg["fps"])
    records = []
    start = time.time()
    try:
        for index, path in enumerate(paths):
            with np.load(path) as data:
                state = dict(data)
            rgb, cam, stats, buffers = renderer.draw(
                state, int(path.stem) / max(cfg["frames"] - 1, 1), args.diagnostics
            )
            image = (rgb.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
            Image.fromarray(image).save(out / "frames" / f"{index:05d}.png")
            video.write(image)
            for name, buf in buffers.items():
                folder = out / name
                folder.mkdir(exist_ok=True)
                data = np.uint8(buf.clamp(0, 1).cpu().numpy() * 255 + 0.5)
                Image.fromarray(data).save(folder / f"{index:05d}.png")
            records.append(dict(source_frame=int(path.stem), camera=cam, **stats))
            if index % 10 == 0 or index == len(paths) - 1:
                print(
                    f"Gaussian render {index + 1}/{len(paths)} {time.time() - start:.1f}s {stats}",
                    flush=True,
                )
    finally:
        video.close()
    (out / "frames.json").write_text(json.dumps(records, indent=2))
    (out / "config.json").write_text(
        json.dumps(
            dict(
                source=str(args.input.resolve()),
                renderer="gsplat_surface_and_volume",
                source_sha256={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(__file__).parent.glob("*.py")
                },
                width=args.width,
                height=args.height,
                fps=cfg["fps"],
                frames=len(records),
                render_seconds=time.time() - start,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
