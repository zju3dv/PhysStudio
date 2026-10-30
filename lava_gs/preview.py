"""Render cached simulation states with a chosen material and camera path."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import warp as wp
from rainy_gs.scene import save_image
from .scene import Scene
from rainy_gs.demo import transition_camera
from .mpm import LavaMPM
from .surface import LavaSurface
from .cameras import cascade_camera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, nargs="+", default=[30, 90, 180])
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--crust", type=float, default=2.3)
    p.add_argument("--surface-iso", type=float, default=0.20)
    p.add_argument("--exposure", type=float, default=0.85)
    p.add_argument("--camera-path", choices=["dataset", "cascade"], default="cascade")
    p.add_argument("--camera-config", type=Path, help="Camera settings JSON")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cfg = json.loads((args.run / "config.json").read_text())
    geo = dict(np.load(args.run / "geometry.npz"))
    states = [dict(np.load(args.run / "states" / f"{i:05d}.npz")) for i in args.frames]
    scene = Scene(cfg["path"], config=json.loads(Path(cfg["config"]).read_text()))
    if args.camera_config:
        scene.config["lava_camera"] = json.loads(args.camera_config.read_text())
    (args.output / "camera_settings.json").write_text(
        json.dumps(scene.config.get("lava_camera", {}), indent=2)
    )
    m = LavaMPM(
        geo["positions"],
        geo["temperature"],
        geo,
        capacity=max(len(s["x"]) for s in states),
    )
    renderer = LavaSurface(
        m, geo, cfg["scale"], args.crust, args.exposure, args.surface_iso
    )
    for frame, state in zip(args.frames, states):
        m.count = len(state["x"])
        m.time = float(state["time"])
        wp.to_torch(m.skin_age)[: m.count].copy_(
            torch.as_tensor(
                state.get("skin_age", m.time - state.get("birth", np.zeros(m.count))),
                device=m.device,
            )
        )
        for attr, key in [
            ("x", "x"),
            ("temp", "temperature"),
            ("reference", "reference"),
            ("birth", "birth"),
        ]:
            wp.to_torch(getattr(m, attr))[: m.count].copy_(
                torch.as_tensor(state[key], device=m.device)
            )
        progress = frame / max(cfg["frames"] - 1, 1)
        if args.camera_path == "cascade":
            cam = cascade_camera(
                progress, args.width, m.device, scene.config.get("lava_camera")
            )
        else:
            cam = transition_camera(
                scene, *cfg["cameras"], progress, args.width, "smoothstep"
            )
        source, _ = scene.render(cam)
        rgb = renderer.render(source, cam)
        save_image(args.output / f"lava_{frame:05d}.png", rgb)
        print(
            frame,
            "visible fraction",
            float((renderer.last_glow.sum(-1) > 1.0e-4).float().mean()),
            flush=True,
        )


if __name__ == "__main__":
    main()
