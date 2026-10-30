"""Reproducible Gaussian Splashing-inspired PBF demos on public 3DGS assets."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
import torch
import warp as wp
from .assets import Object, camera
from .physics import PBF
from .render import Renderer
from rainy_gs.scene import save_image
from rainy_gs.demo import Video


def lattice(lo, hi, s):
    return (
        np.stack(
            np.meshgrid(
                *[np.arange(a + s * 0.5, b, s) for a, b in zip(lo, hi)], indexing="ij"
            ),
            -1,
        )
        .reshape(-1, 3)
        .astype(np.float32)
    )


def configure(scene, spacing=None):
    name = "bread" if scene == "plunge" else "wolf"
    obj = Object(name, height=1.20 if scene == "plunge" else 1.30)
    s = spacing or (0.027 if scene == "liquify" else 0.039)
    sdf, origin, dx, filled = obj.collision(dx=0.014)
    center = obj.body_center.copy()
    bounds = ([-1.65, -1.15, 0.0], [1.65, 1.15, 3.5])
    if scene == "liquify":
        x = lattice(origin, origin + (np.array(sdf.shape) - 1) * dx, s)
        distance = ndimage.map_coordinates(
            sdf, ((x - origin) / dx).T, order=1, mode="constant", cval=10
        )
        x = x[distance < 0] + center
        contact = False
        mass = 0.0
        v = np.zeros_like(x)
    else:
        level = 0.52 if scene == "plunge" else 0.24
        x = lattice(
            [bounds[0][0] + s * 0.1, bounds[0][1] + s * 0.1, 0.015],
            [bounds[1][0] - s * 0.1, bounds[1][1], level],
            s,
        )
        if scene == "wave":
            block = lattice([-1.60, -1.10, level + 0.02], [-0.92, 1.10, 1.04], s)
            x = np.concatenate((x, block))
            center[0] = 0.36
        else:
            center[2] += 1.0
        distance = ndimage.map_coordinates(
            sdf, ((x - center - origin) / dx).T, order=1, mode="constant", cval=10
        )
        x = x[distance > s * 0.4]
        v = np.zeros_like(x)
        if scene == "wave":
            v[:, 0] = np.where(x[:, 0] < -0.90, 0.7, 0.0)
        contact = True
        mass = 280.0 * obj.volume if scene == "plunge" else 0.0
    rng = np.random.default_rng(42)
    x += rng.uniform(-0.012 * s, 0.012 * s, size=x.shape).astype(np.float32)
    sim = PBF(
        x,
        s,
        bounds,
        sdf,
        origin,
        dx,
        center,
        obj.collision_samples,
        body_mass=mass,
        contact=contact,
        velocities=v,
    )
    config = dict(
        scene=scene,
        model=name,
        spacing=s,
        bounds=bounds,
        particles=len(x),
        solid_gaussians=obj.count,
        source_ply=str(obj.path),
        source_sha256=hashlib.sha256(obj.path.read_bytes()).hexdigest(),
        body_mass_kg=mass,
        volume_m3=obj.volume,
        sdf_dx=dx,
        density_kg_m3=1000,
        dt=1 / 360,
        substeps=6,
        iterations=5,
        fluid_model="PBF compression constraints + tensile correction + XSPH",
        solid_model="SDF rigid body contact with equal-and-opposite corrections"
        if scene == "plunge"
        else "static SDF"
        if contact
        else "Gaussian occupancy converted to fluid",
        collision_proxy="voxelized splat occupancy, closed and filled",
        reference="https://gaussiansplashing.github.io/",
        implementation="Independent approximation; no official GSP weights or solver used",
    )
    return obj, sim, config


def title_frame(im, title, subtitle):
    im = Image.fromarray(np.uint8(np.clip(im, 0, 1) * 255 + 0.5))
    draw = ImageDraw.Draw(im)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        f = ImageFont.truetype(font, 22)
        sm = ImageFont.truetype(font, 13)
    except OSError:
        f = sm = None
    draw.rounded_rectangle((20, 18, im.width - 20, 87), radius=10, fill=(20, 29, 39))
    draw.text((36, 29), title, font=f, fill=(240, 247, 255))
    draw.text((36, 59), subtitle, font=sm, fill=(167, 197, 215))
    return np.asarray(im)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=["liquify", "wave", "plunge"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--spacing", type=float)
    p.add_argument("--simulate-only", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument(
        "--render-output",
        type=Path,
        help="Fresh presentation directory for a cached trajectory",
    )
    p.add_argument("--render-start", type=int, default=0)
    p.add_argument("--render-step", type=int, default=1)
    p.add_argument("--static-camera", action="store_true")
    p.add_argument("--azimuth-offset", type=float, default=0.0)
    args = p.parse_args()
    torch.set_num_threads(4)
    wp.init()
    wp.set_device("cuda:0")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    states = out / "states"
    states.mkdir(exist_ok=True)
    if args.render_only:
        cfg = json.loads((out / "config.json").read_text())
        obj = Object(cfg["model"], height=1.20 if args.scene == "plunge" else 1.30)
    else:
        if (out / "config.json").exists():
            raise FileExistsError("Choose a fresh simulation output directory")
        obj, sim, cfg = configure(args.scene, args.spacing)
        cfg.update(frames=args.frames, seed=42, video_fps=30, playback_speed=0.5)
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        print(json.dumps(cfg), flush=True)
        start = time.time()
        audit = []
        for frame in range(args.frames):
            release = (
                frame >= 24 if args.scene in ["liquify", "plunge"] else frame >= 12
            )
            substeps = cfg["substeps"]
            if release:
                for _ in range(substeps):
                    sim.step(
                        dt=1 / (60 * substeps),
                        iterations=cfg["iterations"],
                        moving_body=args.scene == "plunge",
                    )
            state = sim.snapshot()
            if not all(np.isfinite(v).all() for v in state.values()):
                raise FloatingPointError(f"Nonfinite state at frame {frame}")
            np.savez_compressed(states / f"{frame:05d}.npz", **state)
            x = state["positions"]
            speed = np.linalg.norm(state["velocities"], axis=-1)
            boundary = np.minimum(
                x - np.array(cfg["bounds"][0]), np.array(cfg["bounds"][1]) - x
            )
            a = dict(
                frame=frame,
                particles=len(x),
                substeps=substeps,
                max_speed=float(speed.max()),
                p99_speed=float(np.quantile(speed, 0.99)),
                max_density=float(state["density"].max()),
                mean_density=float(state["density"].mean()),
                min_boundary_distance=float(boundary.min()),
                body_center=state["body_center"].tolist(),
            )
            audit.append(a)
            if frame % 12 == 0 or frame == args.frames - 1:
                (out / "simulation_audit.json").write_text(json.dumps(audit, indent=2))
                print(
                    f"sim {frame + 1}/{args.frames} {time.time() - start:.1f}s speed={a['max_speed']:.3f} rho={a['max_density']:.3f} body={a['body_center']}",
                    flush=True,
                )
        cfg["simulation_seconds"] = time.time() - start
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        if args.simulate_only:
            return
    renderer = Renderer(cfg["spacing"], cfg["bounds"])
    paths = sorted(states.glob("*.npz"))[
        args.render_start : args.frames : args.render_step
    ]
    if not paths:
        raise ValueError("No cached states")
    if args.render_output:
        out = args.render_output
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Use a fresh render output")
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(
            json.dumps(
                dict(
                    cfg, trajectory=str(states.resolve()), render_arguments=vars(args)
                ),
                indent=2,
                default=str,
            )
        )
    suffix = (
        ""
        if args.render_step == 1 and args.render_start == 0
        else f"_preview_{args.render_start}_{args.render_step}"
    )
    video = Video(out / f"demo{suffix}.mp4", args.width, args.height, 30)
    titles = {
        "liquify": "EVERYTHING IS WATER / Plush toy liquefaction",
        "wave": "SPLASH LAB / Wave meets a Gaussian object",
        "plunge": "SPLASH LAB / Bread drop and buoyancy",
    }
    keyframes = []
    camera_path = []
    start = time.time()
    try:
        for ix, path in enumerate(paths):
            frame = int(path.stem)
            state = dict(np.load(path))
            progress = frame / max(cfg["frames"] - 1, 1)
            ease = progress * progress * (3 - 2 * progress)
            az = -58 + args.azimuth_offset + (0 if args.static_camera else 24 * ease)
            radius = 4.9 if args.scene == "liquify" else 5.4
            c = camera(
                args.width,
                args.height,
                azimuth=az,
                elevation=29.0,
                radius=radius,
                target=(0.0, 0.0, 0.40 if args.scene == "liquify" else 0.56),
            )
            object_alpha = 1.0
            fluid_alpha = 1.0
            if args.scene == "liquify":
                fluid_alpha = float(np.clip((frame - 17) / 7, 0, 1))
                object_alpha = 1 - fluid_alpha
            image, buffers = renderer.render(c, obj, state, object_alpha, fluid_alpha)
            rendered = image.cpu().numpy()
            subtitle = f"Public PhysGaussian asset  |  PBF + Gaussian liquid shading  |  {cfg['particles']:,} fluid particles"
            video.write(title_frame(rendered, titles[args.scene], subtitle))
            camera_path.append(
                dict(frame=frame, view=c.view.cpu().tolist(), K=c.K.cpu().tolist())
            )
            if ix % max(1, len(paths) // 8) == 0 or ix == len(paths) - 1:
                save_image(out / f"frame_{frame:05d}.png", rendered)
                keyframes.append((frame, rendered))
                print(
                    f"render {ix + 1}/{len(paths)} frame={frame} {time.time() - start:.1f}s",
                    flush=True,
                )
            if ix == len(paths) // 3 and "normals" in buffers:
                for name in ["normals", "alpha", "thickness"]:
                    val = buffers[name]
                    if val.ndim == 2:
                        val = val[..., None].expand(-1, -1, 3)
                    if name == "thickness":
                        val = val / 1.2
                    save_image(out / f"{name}{suffix}.png", val)
                diffuse, _ = renderer.render(
                    c, obj, state, object_alpha, fluid_alpha, specular=False
                )
                save_image(
                    out / f"specular_ablation{suffix}.png",
                    torch.cat((diffuse, image), 1),
                )
        (out / f"camera_path{suffix}.json").write_text(json.dumps(camera_path))
    finally:
        video.close()
    thumbs = []
    for frame, im in keyframes:
        pic = Image.fromarray(np.uint8(im.clip(0, 1) * 255)).resize(
            (320, round(320 * args.height / args.width))
        )
        draw = ImageDraw.Draw(pic)
        draw.text(
            (8, 8), f"FRAME {frame}", fill="white", stroke_width=2, stroke_fill="black"
        )
        thumbs.append(pic)
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (320 * cols, thumbs[0].height * rows), (20, 29, 39))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * 320, (i // cols) * im.height))
    sheet.save(out / f"contact_sheet{suffix}.jpg", quality=93)
    print(f"COMPLETE {out.resolve()}/demo{suffix}.mp4", flush=True)


if __name__ == "__main__":
    main()
