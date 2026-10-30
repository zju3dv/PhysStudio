"""Reproducible moving-camera winter scenes, with independent scene presets."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from rainy_gs.scene import Scene, save_image
from rainy_gs.demo import Video, transition_camera
from .geometry import PRESETS, build_geometry
from .mpm import SnowMPM
from .air import AirSnow
from .render import WinterRenderer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=PRESETS, default="playground")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup", type=float, default=0.25)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--save-frames", action="store_true")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--haze", type=float, default=0.012)
    p.add_argument("--flakes", type=int)
    p.add_argument("--wind", type=float, default=0.45)
    p.add_argument("--seed", type=int, default=19)
    args = p.parse_args()
    if args.frames < 1 or args.width % 2 or args.width < 64:
        p.error("Use positive frames and even width >=64")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = time.monotonic()
    cfg = PRESETS[args.scene].copy()
    scene = Scene(
        cfg["path"],
        config=json.loads(Path(cfg["config"]).read_text()) if cfg["config"] else None,
    )
    print("Building snow geometry", args.scene, flush=True)
    geo = build_geometry(scene, cfg, args.seed)
    print("MPM grid", geo["phi"].shape, "particles", len(geo["positions"]), flush=True)
    mpm = SnowMPM(
        geo["positions"], geo["phi"], geo["origin"], geo["dx"], geo["volume"], dt=0.0015
    )
    cameras = [
        scene.camera(i, args.width).center.cpu().numpy() * cfg["scale"]
        for i in cfg["cameras"]
    ]
    air = AirSnow(mpm, geo, args.flakes or cfg["flakes"], cameras, args.seed, args.wind)
    renderer = WinterRenderer(
        scene, mpm, air, geo, cfg["scale"], args.strength, args.haze
    )
    air.link_surface(renderer.surface)
    config = {
        **vars(args),
        **cfg,
        "output": str(args.output),
        "code_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in Path(__file__).parent.glob("*.py")
        },
        "gaussians": len(scene.means),
        "grid": list(geo["phi"].shape),
        "mpm_particles_initial": mpm.count,
        "young_pa": mpm.young,
        "density_kg_m3": mpm.density,
        "theta_c": mpm.theta_c,
        "theta_s": mpm.theta_s,
        "hardening": mpm.hardening,
        "friction": mpm.friction,
        "metres_per_unit": cfg["scale"],
    }
    config["plastic_volume_range"] = [0.6, 2.0]
    config["snow_renderer"] = "Warp ray-marched MPM mass isosurface"
    config["air_particles"] = len(air.x)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    np.savez_compressed(
        args.output / "geometry.npz",
        phi=geo["phi"],
        origin=geo["origin"],
        dx=geo["dx"],
        initial=geo["positions"],
    )
    for _ in range(round(args.warmup * 30)):
        mpm.advance(1 / 30)
    # Air warmup circulates initial particles out of solid volumes; deposit only
    # after this phase so initial uniform samples do not add spurious snow mass.
    for _ in range(45):
        old = mpm.count
        air.advance(1 / 30)
        if mpm.count > old:
            import warp as wp

            wp.to_torch(mpm.vol)[old : mpm.count] = 0
            mpm.count = old
            mpm.deposited = 0
    logs = []
    paths = []
    videos = []
    nframes = 3 if args.preview else args.frames
    for f in range(nframes):
        progress = f / max(nframes - 1, 1)
        cam = transition_camera(
            scene, *cfg["cameras"], progress, args.width, "smoothstep"
        )
        if not args.preview or f > 0:
            air.advance(1 / args.fps)
            mpm.advance(1 / args.fps)
        rgb = renderer.render(cam, air.time)
        if args.preview or f in [0, nframes // 2, nframes - 1] or args.save_frames:
            save_image(args.output / f"snow_{f:05d}.png", rgb)
        dry, _ = scene.render(cam, mode="RGB")
        if args.preview or f in [0, nframes // 2, nframes - 1]:
            save_image(args.output / f"dry_{f:05d}.png", dry)
            save_image(
                args.output / f"pack_{f:05d}.png",
                renderer.render(cam, air.time, flakes=False),
            )
        if not args.preview:
            if not videos:
                videos = [
                    Video(
                        args.output / "snow_demo.mp4", cam.width, cam.height, args.fps
                    ),
                    Video(
                        args.output / "comparison.mp4",
                        cam.width * 2,
                        cam.height,
                        args.fps,
                    ),
                ]
            img = (rgb.cpu().numpy() * 255 + 0.5).astype("uint8")
            original = (dry.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype("uint8")
            videos[0].write(img)
            videos[1].write(np.concatenate((original, img), axis=1))
        if f % 15 == 0 or f == nframes - 1:
            diag = mpm.diagnostics()
            diag.update(
                frame=f, air_hits=air.total_hits, elapsed=time.monotonic() - start
            )
            logs.append(diag)
            print(json.dumps(diag), flush=True)
            (args.output / "status.json").write_text(json.dumps(diag, indent=2))
        paths.append(
            dict(frame=f, view=cam.view.cpu().tolist(), K=cam.K.cpu().tolist())
        )
    for video in videos:
        video.close()
    np.savez_compressed(args.output / "snow_state.npz", **mpm.state())
    np.savez_compressed(
        args.output / "air_state.npz",
        x=air.x.numpy(),
        v=air.v.numpy(),
        r=air.r.numpy(),
        time=air.time,
        steps=air.steps,
    )
    (args.output / "diagnostics.json").write_text(json.dumps(logs, indent=2))
    (args.output / "camera_path.json").write_text(json.dumps(paths))
    (args.output / "summary.json").write_text(
        json.dumps(
            dict(
                complete=True,
                preview=args.preview,
                frames=nframes,
                elapsed=time.monotonic() - start,
                final=mpm.diagnostics(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
