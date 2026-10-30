"""Render saved FEM trajectories with transported, adaptively split 3DGS."""

from physstudio.paths import data_path
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from gsplat import rasterization
from gsplat.cuda._wrapper import spherical_harmonics

from .mesh import Cage
from .gaussians import GaussianBody
from gaussian_splashing.assets import camera
from rainy_gs.demo import Video


def stage(cam, center, height):
    """A procedural studio floor; separate from the captured Gaussian object."""
    yy, xx = torch.meshgrid(
        torch.arange(cam.height, device="cuda") + 0.5,
        torch.arange(cam.width, device="cuda") + 0.5,
        indexing="ij",
    )
    rays = torch.stack(
        [
            (xx - cam.K[0, 2]) / cam.K[0, 0],
            (yy - cam.K[1, 2]) / cam.K[1, 1],
            torch.ones_like(xx),
        ],
        -1,
    )
    c2w = torch.linalg.inv(cam.view)
    rays = rays @ c2w[:3, :3].T
    t = -cam.center[2] / rays[..., 2]
    points = cam.center + t[..., None] * rays
    tiles = (
        torch.floor(points[..., 0] / 0.45) + torch.floor(points[..., 1] / 0.45)
    ).remainder(2)
    colors = tiles[..., None] * torch.tensor(
        [0.08, 0.08, 0.075], device="cuda"
    ) + torch.tensor([0.76, 0.79, 0.79], device="cuda")
    fade = (1 - t.clamp(0, 30) / 35).clamp(0.55, 1)
    colors = colors * fade[..., None] + torch.tensor(
        [0.94, 0.96, 0.97], device="cuda"
    ) * (1 - fade[..., None])
    shadow = torch.exp(
        -((points[..., :2] - center[:2]) ** 2).sum(-1) / (0.13 + 0.12 * height)
    ) * max(0.1, 0.45 - 0.3 * height)
    colors *= 1 - shadow[..., None]
    colors = torch.where(
        (t > 0)[..., None], colors, torch.tensor([0.93, 0.95, 0.97], device="cuda")
    )
    return colors, t


@torch.no_grad()
def render_body(body, x, cam):
    means, cov, r, angle = body.deform(x)
    dirs = torch.nn.functional.normalize(means - cam.center.double(), dim=-1)
    dirs = torch.einsum("ni,nij->nj", dirs, r) @ body.source_rotation
    colors = (spherical_harmonics(3, dirs.float(), body.sh.float()) + 0.5).clamp_min(0)
    im, alpha, _ = rasterization(
        means=means.float(),
        quats=None,
        scales=None,
        covars=cov.float(),
        opacities=body.opacity.float(),
        colors=colors,
        viewmats=cam.view[None],
        Ks=cam.K[None],
        width=cam.width,
        height=cam.height,
        packed=False,
        render_mode="RGB+ED",
        near_plane=0.02,
        far_plane=30.0,
    )
    bg, floor_depth = stage(cam, means.mean(0), float(means[:, 2].min()))
    rgb = im[0, :, :, :3] + bg * (1 - alpha[0])
    return rgb.clamp(0, 1), angle


def main(args):
    src = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    data = np.load(data_path("enlivening_gs") / cfg["scene"] / "gaussians.npz")
    cage = Cage.load(src / "cage.npz")
    tr = np.load(src / "trajectory.npz")
    body = GaussianBody(cage, data)
    # Repeat physical states only to set a transparent playback speed.
    duration = float(tr["time"][-1])
    count = round(duration * args.fps / args.speed) + 1
    video = Video(out / "demo.mp4", args.width, args.height, args.fps)
    summaries = []
    try:
        for i in range(count):
            tm = min(duration, i * args.speed / args.fps)
            k = np.searchsorted(tr["time"], tm, side="right") - 1
            k = max(0, min(k, len(tr["time"]) - 1))
            k1 = min(k + 1, len(tr["time"]) - 1)
            fraction = (
                0
                if k == k1
                else (tm - tr["time"][k]) / (tr["time"][k1] - tr["time"][k])
            )
            x = (1 - fraction) * tr["x"][k] + fraction * tr["x"][k1]
            cam = camera(
                args.width,
                args.height,
                azimuth=-64 + args.orbit * i / max(count - 1, 1),
                elevation=20,
                radius=3.4,
                target=(0.12, 0, 0.62),
            )
            rgb, angle = render_body(body, x, cam)
            split = body.split(angle) if i and i % 6 == 0 and args.split else 0
            image = Image.fromarray(np.uint8(rgb.cpu().numpy() * 255 + 0.5))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, args.width, 44), fill=(20, 29, 39))
            draw.text(
                (18, 14),
                f"{cfg['scene'].upper()}  |  Muscle-driven FEM / 3DGS  |  t = {tm:.2f} s",
                fill=(240, 244, 246),
            )
            video.write(np.array(image))
            if i in [0, count // 4, count // 2, 3 * count // 4, count - 1]:
                image.save(out / f"frame_{i:04d}.png")
            if i % 15 == 0:
                print(
                    f"render {i}/{count}, gaussians={len(body.means)}, splits={body.total_split}",
                    flush=True,
                )
            summaries.append(
                dict(
                    frame=i,
                    time=tm,
                    gaussians=len(body.means),
                    split=split,
                    max_embedding_angle_degrees=float(angle.max() * 180 / np.pi),
                )
            )
    finally:
        video.close()
    (out / "render.json").write_text(
        json.dumps(
            dict(
                frames=count,
                physical_duration=duration,
                playback_speed=args.speed,
                adaptive_splits=body.total_split,
                frames_info=summaries,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--speed", type=float, default=0.5)
    p.add_argument("--orbit", type=float, default=24.0)
    p.add_argument("--split", action="store_true")
    main(p.parse_args())
