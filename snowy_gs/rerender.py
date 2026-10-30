"""Inspect saved physics without rerunning the MPM trajectory."""

import argparse, json
from pathlib import Path
import numpy as np
import torch
import warp as wp
from rainy_gs.scene import Scene, save_image
from rainy_gs.demo import transition_camera
from .geometry import PRESETS
from .mpm import SnowMPM
from .air import AirSnow
from .render import WinterRenderer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=1280)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config = json.loads((a.run / "config.json").read_text())
    cfg = PRESETS[config["scene"]]
    scene = Scene(
        cfg["path"],
        config=json.loads(Path(cfg["config"]).read_text()) if cfg["config"] else None,
    )
    z = np.load(a.run / "geometry.npz")
    s = np.load(a.run / "snow_state.npz")
    geo = dict(
        phi=z["phi"],
        origin=z["origin"],
        dx=float(z["dx"]),
        positions=z["initial"],
        bounds=np.array(cfg["bounds"]) * cfg["scale"],
        spacing=float(z["dx"]) * 0.5,
        plane=scene.ground.cpu().numpy().copy(),
    )
    geo["plane"][2] *= cfg["scale"]
    geo["volume"] = geo["spacing"] ** 2 * min(0.045, geo["dx"] * 0.55)
    m = SnowMPM(geo["positions"], geo["phi"], geo["origin"], geo["dx"], geo["volume"])
    cameras = [
        scene.camera(i, a.width).center.cpu().numpy() * cfg["scale"]
        for i in cfg["cameras"]
    ]
    air = AirSnow(m, geo, cfg["flakes"], cameras)
    renderer = WinterRenderer(scene, m, air, geo, cfg["scale"], haze=0.006)
    m.count = len(s["x"])
    for k in ("x", "v", "fe", "ft", "jp", "vol"):
        wp.to_torch(getattr(m, k))[: m.count] = torch.tensor(s[k], device=scene.device)
    cam = transition_camera(scene, *cfg["cameras"], 0.5, a.width, "smoothstep")
    save_image(a.output / "winter.png", renderer.render(cam, 0, flakes=False))
    save_image(
        a.output / "coating_only.png", renderer.render(cam, 0, flakes=False, pack=False)
    )
    renderer.sh = scene.sh
    save_image(
        a.output / "covariance_baseline.png",
        renderer.render(cam, 0, flakes=False, pack=False),
    )
    raw, _ = scene.render(cam, mode="RGB")
    save_image(a.output / "original.png", raw)


if __name__ == "__main__":
    main()
