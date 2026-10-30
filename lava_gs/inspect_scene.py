"""Render a calibrated scene and inspect its terrain support before simulation."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rainy_gs.scene import save_image
from .scene import Scene
from .geometry import PRESETS
from .canyon import build_canyon
from .cameras import cascade_camera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True, choices=PRESETS)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--geometry", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--width", type=int, default=1024)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cfg = PRESETS[args.scene]
    scene = Scene(cfg["path"], config=json.loads(Path(cfg["config"]).read_text()))
    cam = scene.camera(0, args.width)
    if "lava_camera" in scene.config:
        cam = cascade_camera(0, args.width, scene.device, scene.config["lava_camera"])
    image, alpha = scene.render(cam)
    save_image(args.output / "scene.png", image[..., :3])
    np.savez_compressed(
        args.output / "projection.npz",
        world=cam.world_points(image[..., 3]).cpu().numpy(),
        alpha=alpha.cpu().numpy(),
        depth=image[..., 3].cpu().numpy(),
    )
    xyz = scene.means.cpu().numpy()
    mask = scene.opacity.cpu().numpy() > 0.65
    print(
        "Scene quantiles",
        np.quantile(xyz[mask], [0.005, 0.05, 0.5, 0.95, 0.995], axis=0),
        flush=True,
    )
    if args.geometry:
        geo = build_canyon(scene, cfg)
        np.savez_compressed(args.output / "geometry.npz", **geo)
        print(
            "Grid",
            geo["phi"].shape,
            "height",
            float(geo["terrain"].min()),
            float(geo["terrain"].max()),
            flush=True,
        )
        if not args.plot:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h = geo["terrain"] / cfg["scale"]
        extent = [
            geo["origin"][0] / cfg["scale"],
            (geo["origin"][0] + (h.shape[0] - 1) * cfg["dx"]) / cfg["scale"],
            geo["origin"][1] / cfg["scale"],
            (geo["origin"][1] + (h.shape[1] - 1) * cfg["dx"]) / cfg["scale"],
        ]
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(h.T, origin="lower", extent=extent, cmap="terrain")
        ax.set(
            xlabel="Scene X",
            ylabel="Scene Y",
            title=args.scene + " reconstructed surface height",
        )
        fig.colorbar(im, ax=ax, label="Scene Z")
        fig.savefig(args.output / "terrain.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
