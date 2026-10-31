"""Generate reusable electric discharge trees independently of rendering."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import warp as wp
from . import scenes
from .solver import Settings, discharge


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=scenes.PRESETS, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--events", type=int, default=3)
    p.add_argument("--max-nodes", type=int, default=1800)
    args = p.parse_args()
    if args.events < 1 or args.max_nodes < 2:
        p.error("Events must be positive and max-nodes must be at least two.")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    wp.init()
    scene, cfg = scenes.load(args.scene)
    geometry = scenes.domain(scene, cfg)
    np.savez_compressed(args.output / "geometry.npz", **geometry)
    cfg.update(
        scene=args.scene,
        gaussians=len(scene.means),
        events=args.events,
        code_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    )
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    for event in range(args.events):
        rho, seed = scenes.charges(cfg, geometry, event)
        settings = Settings(
            dx=cfg["dx"],
            eta=cfg["eta"],
            critical=cfg["critical"],
            internal=cfg["internal"],
            seed=cfg["seed"] + 19 * event,
            max_nodes=args.max_nodes,
        )
        result = discharge(rho, geometry["solid"], geometry["origin"], settings, seed)
        report = result.pop("report")
        report["event"] = event
        np.savez_compressed(args.output / f"event_{event:02d}.npz", **result)
        (args.output / f"event_{event:02d}.json").write_text(
            json.dumps(report, indent=2)
        )
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
