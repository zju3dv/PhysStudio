"""Simulate and render a scene-scale storm into a fresh output directory."""
import argparse
from dataclasses import asdict
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import warp as wp
from rainy_gs.scene import save_image
from rainy_gs.demo import Video
from .scenes import PRESETS, load_scene, geometry, camera
from .solver import Settings, Storm
from .render import Renderer
from .particles import Particles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=PRESETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.012)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if min(args.frames, args.width, args.fps, args.dt) <= 0 or args.warmup < 0:
        parser.error(
            "Frames, width, fps, and dt must be positive; warmup must be nonnegative."
        )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    start = time.monotonic()
    scene, cfg = load_scene(args.scene)
    if args.config:
        cfg.update(json.loads(args.config.read_text()))
    settings = Settings(**cfg["simulation"])
    geo = geometry(scene, cfg, settings)
    np.savez_compressed(args.output / "geometry.npz", **geo)
    sim = Storm(settings, geo["terrain"], geo["solid"])
    particles = Particles(sim, cfg)
    renderer = Renderer(scene, cfg, sim, particles)
    substeps = int(np.ceil(1.0 / (args.fps * args.dt)))
    dt = 1.0 / (args.fps * substeps)
    (args.output / "config.json").write_text(
        json.dumps(
            dict(
                scene=args.scene,
                preset=cfg,
                solver=asdict(settings),
                frames=args.frames,
                fps=args.fps,
                width=args.width,
                warmup=args.warmup,
                dt=args.dt,
                effective_dt=dt,
                substeps_per_frame=substeps,
                preview=args.preview,
            ),
            indent=2,
        )
    )
    source = args.output / "source"
    source.mkdir()
    shutil.copytree(Path(__file__).parent / "configs", source / "configs")
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copy2(p, source / p.name)
    diagnostics = []
    for step in range(int(np.ceil(args.warmup / dt))):
        sim.step(dt)
        particles.step(dt)
        if step % 50 == 0:
            record = sim.diagnostics()
            diagnostics.append(record)
            print("warmup", step, record, flush=True)
    frames = (
        [0, args.frames // 2, args.frames - 1]
        if args.preview
        else list(range(args.frames))
    )
    video = None
    comparison = None
    paths = []
    records = []
    for f in range(args.frames):
        for _ in range(substeps):
            sim.step(dt)
            particles.step(dt)
        if f not in frames:
            continue
        cam = camera(cfg, f / max(args.frames - 1, 1), args.width, scene)
        result, alpha, original = renderer.render(cam)
        save_image(args.output / f"storm_{f:05d}.png", result)
        if f in [0, args.frames // 2, args.frames - 1]:
            save_image(args.output / f"original_{f:05d}.png", original)
            save_image(args.output / f"opacity_{f:05d}.png", alpha)
            np.savez_compressed(
                args.output / f"state_{f:05d}.npz",
                air=sim.a.numpy(),
                dust=sim.dust.numpy(),
                u=sim.u.numpy(),
                v=sim.v.numpy(),
                w=sim.w.numpy(),
                particle_x=particles.x.numpy(),
                particle_v=particles.v.numpy(),
                time=sim.time,
            )
        if not args.preview:
            if video is None:
                video = Video(
                    args.output / "storm_demo.mp4", cam.width, cam.height, args.fps
                )
                comparison = Video(
                    args.output / "comparison.mp4", cam.width * 2, cam.height, args.fps
                )
            raw = np.uint8(result.cpu().numpy() * 255 + 0.5)
            dry = np.uint8(original.cpu().numpy() * 255 + 0.5)
            video.write(raw)
            comparison.write(np.concatenate([dry, raw], 1))
        paths.append(dict(frame=f, view=cam.view.tolist(), K=cam.K.tolist()))
        if f % 30 == 0 or f in [args.frames - 1, args.frames // 2]:
            record = sim.diagnostics()
            record.update(
                frame=f,
                elapsed=time.monotonic() - start,
                opacity_mean=float(alpha.mean()),
                opacity_max=float(alpha.max()),
            )
            records.append(record)
            print(record, flush=True)
            (args.output / "status.json").write_text(json.dumps(record, indent=2))
    if video:
        video.close()
        comparison.close()
    (args.output / "camera_path.json").write_text(json.dumps(paths))
    (args.output / "diagnostics.json").write_text(
        json.dumps(dict(warmup=diagnostics, frames=records), indent=2)
    )
    (args.output / "summary.json").write_text(
        json.dumps(
            dict(
                complete=True,
                preview=args.preview,
                elapsed=time.monotonic() - start,
                final=sim.diagnostics(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
