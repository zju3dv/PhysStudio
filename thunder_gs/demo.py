"""Render saved discharges with moving 3DGS cameras and frozen replay."""
import argparse
import os
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import warp as wp
from PIL import Image
from rainy_gs.demo import Video
from .scenes import load, camera
from .render import Renderer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--simulation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()
    if min(args.frames, args.fps) < 1 or args.width < 32 or args.width % 2:
        p.error("Frames and fps must be positive; width must be even and at least 32.")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    wp.init()
    cfg = json.loads((args.simulation / "config.json").read_text())
    scene, current = load(cfg["scene"])
    # Rendering settings may evolve without changing the saved physical graph.
    cfg.update({k: current[k] for k in ("camera", "exposure", "fog", "terrain_cap")})
    geometry = dict(np.load(args.simulation / "geometry.npz"))
    events = [dict(np.load(f)) for f in sorted(args.simulation.glob("event_*.npz"))]
    if not events:
        raise ValueError("The simulation directory contains no discharge events.")
    renderer = Renderer(scene, cfg, geometry, events)
    indices = (
        sorted({0, args.frames - 1, *[f for f in (33, 110, 186) if f < args.frames]})
        if args.preview
        else list(range(args.frames))
    )
    diagnostics = []
    poses = []
    video = None
    comparison = None
    start = time.monotonic()
    for i in indices:
        cam = camera(scene, cfg, i / max(args.frames - 1, 1), args.width)
        out, clean, diag = renderer.render(cam, i / args.fps)
        image = (out.cpu().numpy() * 255).round().astype("uint8")
        original = (clean.cpu().numpy() * 255).round().astype("uint8")
        Image.fromarray(image).save(args.output / f"thunder_{i:05d}.png")
        if args.preview or i in (0, 33, 110, 186, 239):
            Image.fromarray(original).save(args.output / f"original_{i:05d}.png")
        if not args.preview:
            if video is None:
                video = Video(
                    args.output / "thunder_demo.mp4", cam.width, cam.height, args.fps
                )
                comparison = Video(
                    args.output / "comparison.mp4", cam.width * 2, cam.height, args.fps
                )
            video.write(image)
            comparison.write(np.concatenate((original, image), axis=1))
        diagnostics.append(diag)
        poses.append(dict(frame=i, view=cam.view.tolist(), K=cam.K.tolist()))
        if i % 20 == 0 or args.preview:
            print(dict(frame=i, elapsed=time.monotonic() - start, **diag), flush=True)
    if video:
        video.close()
        comparison.close()
    (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    (args.output / "camera_path.json").write_text(json.dumps(poses, indent=2))
    cfg.update(
        simulation=os.path.relpath(args.simulation.resolve(), args.output.resolve()),
        width=cam.width,
        height=cam.height,
        frames=len(indices),
        fps=args.fps,
        preview=args.preview,
        code_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    )
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    print("Completed", args.output, flush=True)


if __name__ == "__main__":
    main()
