"""Render actual MPM trajectories with pretrained 3DGS assets and liquid splats."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from gsplat import rasterization
from gsplat.cuda._wrapper import spherical_harmonics
from gaussian_splashing.assets import Object, camera
from gaussian_splashing.render import Renderer, smooth_depth, blur, environment
from rainy_gs.demo import Video


def floor(cam):
    rays = (
        cam.world_points(torch.ones((cam.height, cam.width), device=cam.view.device))
        - cam.center
    )
    depth = (-cam.center[2] / rays[..., 2]).clamp(0.001, 100.0)
    x = cam.center + depth[..., None] * rays
    tiles = (torch.floor(x[..., 0] / 0.04) + torch.floor(x[..., 1] / 0.04)).remainder(2)
    col = x.new_tensor([0.19, 0.23, 0.27]) + tiles[..., None] * 0.035
    edge = (
        torch.minimum(
            torch.remainder(x[..., :2] / 0.04, 1.0),
            1 - torch.remainder(x[..., :2] / 0.04, 1.0),
        )
        .min(-1)
        .values
    )
    col *= 1 - 0.13 * torch.exp(-(edge[..., None] ** 2) / 0.0004)
    fade = torch.exp(-torch.linalg.norm(x[..., :2], dim=-1) * 1.5)[..., None]
    col = col * fade + x.new_tensor([0.35, 0.40, 0.45]) * (1 - fade)
    shadow = torch.exp(
        -((x[..., :2] - x.new_tensor([-0.08, 0.04])) ** 2).sum(-1) / 0.006
    )
    col *= 1 - 0.35 * shadow[..., None]
    return col, depth, rays


def magnet(bg, depth, rays, cam, center, radius):
    offset = cam.center - torch.as_tensor(center, device=bg.device)
    aa = (rays * rays).sum(-1)
    bb = (rays * offset).sum(-1)
    disc = bb * bb - aa * ((offset * offset).sum() - radius**2)
    t = (-bb - torch.sqrt(disc.clamp_min(0))) / aa
    visible = (disc >= 0) & (t > 0) & (t < depth)
    normal = F.normalize(offset + t[..., None] * rays, dim=-1)
    view = F.normalize(-rays, dim=-1)
    light = F.normalize(bg.new_tensor([-0.4, -0.6, 1.0]), dim=0)
    diffuse = 0.3 + 0.7 * (normal * light).sum(-1).clamp_min(0)
    upper = bg.new_tensor([0.72, 0.13, 0.07])
    lower = bg.new_tensor([0.10, 0.25, 0.47])
    base = torch.where((normal[..., 2] > 0)[..., None], upper, lower)
    nv = (normal * view).sum(-1).clamp_min(0)
    refl = 2 * nv[..., None] * normal - view
    shade = (
        base * diffuse[..., None]
        + environment(refl) * (0.12 + 0.25 * (1 - nv) ** 4)[..., None]
    )
    # A slim metallic seam identifies the prescribed permanent magnet.
    shade = torch.where(
        (normal[..., 2].abs() < 0.035)[..., None], shade * 0.3 + 0.35, shade
    )
    return torch.where(visible[..., None], shade, bg), torch.where(visible, t, depth)


class BoundGaussians:
    def __init__(self, obj, rest, center, dx):
        self.obj = obj
        device = obj.local.device
        means = obj.local.cpu().numpy() + center
        distances, indices = cKDTree(rest).query(means, k=8)
        weights = np.exp(-((distances / (dx * 1.1)) ** 2))
        weights /= weights.sum(-1, keepdims=True)
        self.ids = torch.as_tensor(indices, device=device)
        self.weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        self.offset = torch.as_tensor(
            means[:, None] - rest[indices], device=device, dtype=torch.float32
        )

    def deform(self, state):
        obj = self.obj
        device = obj.local.device
        x = torch.as_tensor(state["positions"], device=device)
        f = torch.as_tensor(state["deformation"], device=device)
        fs = f[self.ids]
        means = (
            (x[self.ids] + torch.einsum("nkij,nkj->nki", fs, self.offset))
            * self.weights[..., None]
        ).sum(1)
        deform = (fs * self.weights[..., None, None]).sum(1)
        covariance = deform @ obj.cov @ deform.transpose(-1, -2)
        u, _, vh = torch.linalg.svd(deform)
        u[:, :, -1] *= torch.linalg.det(u @ vh)[:, None]
        rotation = u @ vh
        return means, covariance, rotation

    def render(self, cam, state):
        obj = self.obj
        means, covariance, rotation = self.deform(state)
        direction = (
            torch.einsum(
                "ni,nij->nj", F.normalize(means - cam.center, dim=-1), rotation
            )
            @ obj.baserot
        )
        color = (spherical_harmonics(3, direction, obj.sh) + 0.5).clamp_min(0)
        im, alpha, _ = rasterization(
            means=means,
            quats=None,
            scales=None,
            covars=covariance,
            opacities=obj.opacity,
            colors=color,
            viewmats=cam.view[None],
            Ks=cam.K[None],
            width=cam.width,
            height=cam.height,
            packed=False,
            render_mode="RGB+ED",
            near_plane=0.005,
            far_plane=10.0,
        )
        return im[0], alpha[0, ..., 0]


class View:
    def __init__(self, config, rest, device):
        self.cfg = config
        self.device = device
        self.obj = Object(
            config["asset"]["name"], height=config["asset"]["height"], device=device
        )
        self.center = np.asarray(config["asset"]["center"], dtype="f")
        self.fluid = Renderer(
            config["dx"] / 2, ([-0.24, -0.24, 0], [0.24, 0.24, 0.3]), device
        )
        self.bound = None
        if config["mode"] == "solid":
            self.bound = BoundGaussians(self.obj, rest, self.center, config["dx"])

    @torch.no_grad()
    def draw(self, cam, state, magnet_center):
        bg, depth, rays = floor(cam)
        if self.bound is None:
            im, alpha = self.obj.render(cam, self.center, [0, 0, 0, 1])
        else:
            im, alpha = self.bound.render(cam, state)
        bg = im[..., :3] + bg * (1 - alpha[..., None])
        depth = torch.where(alpha > 0.5, im[..., 3], depth)
        bg, depth = magnet(bg, depth, rays, cam, magnet_center, self.cfg["radius"])
        if self.cfg["mode"] == "solid":
            return bg.clamp(0, 1)
        x = torch.as_tensor(state["positions"], device=self.device)
        cov = self.fluid.anisotropy(x)
        im, alpha = self.fluid.splat(cam, x, 0.94, covars=cov)
        fd = (
            smooth_depth(im[..., 3], alpha, self.cfg["dx"] / 2, passes=10)
            - self.cfg["dx"] * 0.18
        )
        world = cam.world_points(fd)
        du = torch.roll(world, -1, 1) - torch.roll(world, 1, 1)
        dv = torch.roll(world, -1, 0) - torch.roll(world, 1, 0)
        normal = F.normalize(torch.cross(du, dv, dim=-1), dim=-1)
        view = F.normalize(cam.center - world, dim=-1)
        normal = torch.where((normal * view).sum(-1, keepdim=True) < 0, -normal, normal)
        valid = (alpha > 0.2).float()
        normal = F.normalize(
            blur(normal * valid[..., None], 5)
            / blur(valid, 5).clamp_min(1.0e-6)[..., None],
            dim=-1,
        )
        nv = (normal * view).sum(-1).clamp(0, 1)
        reflected = 2 * nv[..., None] * normal - view
        fresnel = 0.14 + 0.86 * (1 - nv) ** 5
        liquid = (
            x.new_tensor([0.012, 0.017, 0.022])
            + environment(reflected) * fresnel[..., None]
        )
        light = F.normalize(x.new_tensor([-0.4, -0.55, 1.0]), dim=0)
        half = F.normalize(light + view, dim=-1)
        liquid += (normal * half).sum(-1).clamp_min(0).pow(140)[..., None] * 1.1
        visible = torch.sigmoid((depth - fd) * 2000)
        coverage = (alpha / 0.85).clamp(0, 1) * visible
        return (liquid * coverage[..., None] + bg * (1 - coverage[..., None])).clamp(
            0, 1
        )


def main(args):
    src = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    if (
        args.single is None
        and not json.loads((src / "summary.json").read_text())["complete"]
    ):
        raise ValueError("Full video rendering requires a completed trajectory")
    trajectory = np.load(src / "trajectory.npz")
    wp.init()
    wp.set_stream(
        wp.stream_from_torch(torch.cuda.current_stream(args.device)), device=args.device
    )
    renderer = View(cfg, trajectory["rest"], args.device)
    positions, velocities, deformation = [
        trajectory[k] for k in ["positions", "velocities", "deformation"]
    ]
    duration = float(trajectory["time"][-1])
    count = len(positions)
    if args.single is not None:
        indices = [min(args.single, count - 1)]
    else:
        indices = list(range(count))
    video = (
        None
        if args.single is not None
        else Video(out / "demo.mp4", args.width, args.height, args.fps)
    )
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    try:
        for i in indices:
            cam = camera(
                args.width,
                args.height,
                azimuth=-68 + args.orbit * i / max(count - 1, 1),
                elevation=29,
                radius=0.71,
                target=(0.0, 0.0, 0.075),
                device=args.device,
            )
            state = dict(
                positions=positions[i],
                velocities=velocities[i],
                deformation=deformation[i],
            )
            rgb = renderer.draw(cam, state, trajectory["magnet_center"][i])
            image = Image.fromarray(np.uint8(rgb.cpu().numpy() * 255 + 0.5))
            d = ImageDraw.Draw(image)
            d.rectangle((0, 0, args.width, 48), fill=(17, 25, 34))
            label = (
                "Magnetic liquid + 3DGS obstacle"
                if cfg["mode"] == "fluid"
                else "Magnetized 3DGS / covariance transport"
            )
            if cfg["intensity"] == 0:
                label = "No magnetic field / matched control"
            d.text(
                (18, 14),
                f"{label}   |   t = {trajectory['time'][i]:.3f} s",
                font=font,
                fill=(230, 237, 242),
            )
            if video is not None:
                video.write(np.array(image))
            if args.single is not None or i in [
                0,
                count // 4,
                count // 2,
                3 * count // 4,
                count - 1,
            ]:
                image.save(out / f"frame_{i:04d}.png")
            if i % 20 == 0:
                print(f"render {i}/{count}", flush=True)
    finally:
        if video is not None:
            video.close()
    (out / "render.json").write_text(
        json.dumps(
            dict(
                frames=len(indices),
                physical_duration=duration,
                video_fps=args.fps,
                playback_speed=duration / (max(1, count - 1) / args.fps),
                orbit_degrees=args.orbit,
                width=args.width,
                height=args.height,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--orbit", type=float, default=22.0)
    p.add_argument("--single", type=int)
    main(p.parse_args())
