"""Re-render saved storm keyframes without changing their physical fields."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import warp as wp
from rainy_gs.scene import Camera, save_image
from .scenes import load_scene
from .solver import Settings, Storm
from .particles import Particles
from .render import Renderer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    config = json.loads((args.input / "config.json").read_text())
    cfg = config["preset"]
    if args.config:
        cfg.update(json.loads(args.config.read_text()))
    scene, _ = load_scene(config["scene"])
    geo = np.load(args.input / "geometry.npz")
    sim = Storm(Settings(**config["solver"]), geo["terrain"], geo["solid"])
    particles = Particles(sim, cfg)
    paths = {
        f["frame"]: f for f in json.loads((args.input / "camera_path.json").read_text())
    }
    for path in sorted(args.input.glob("state_*.npz")):
        frame = int(path.stem.split("_")[-1])
        state = np.load(path)
        for key, target, dtype in [
            ("air", sim.a, wp.vec4),
            ("dust", sim.dust, float),
            ("u", sim.u, float),
            ("v", sim.v, float),
            ("w", sim.w, float),
        ]:
            wp.copy(target, wp.array(state[key], dtype=dtype, device=sim.device))
        present = "particle_x" in state
        if present:
            wp.copy(
                particles.x,
                wp.array(state["particle_x"], dtype=wp.vec3, device=sim.device),
            )
            wp.copy(
                particles.v,
                wp.array(state["particle_v"], dtype=wp.vec3, device=sim.device),
            )
        sim.time = float(state["time"])
        renderer = Renderer(scene, cfg, sim, particles if present else None)
        saved = paths[frame]
        K = torch.tensor(saved["K"], device="cuda")
        width = config["width"]
        height = round(float(K[1, 2]) * 2)
        camera = Camera(torch.tensor(saved["view"], device="cuda"), K, width, height)
        rgb, alpha, clean = renderer.render(camera)
        save_image(args.output / f"storm_{frame:05d}.png", rgb)
        save_image(args.output / f"opacity_{frame:05d}.png", alpha)
        print(frame, float(alpha.mean()), flush=True)
    (args.output / "replay.json").write_text(
        json.dumps(dict(input=str(args.input), settings=cfg), indent=2)
    )


if __name__ == "__main__":
    main()
