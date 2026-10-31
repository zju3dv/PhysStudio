"""Check saved keyframes by independently replaying the electrical graph."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import warp as wp
from rainy_gs.scene import Camera
from .scenes import load
from .render import Renderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[33, 110, 186])
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=("firewoods", "zohar", "train"),
        default=["firewoods", "zohar", "train"],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    wp.init()
    report = {}
    for name in args.scenes:
        folder = args.root / name
        if (folder / "render").is_dir():
            folder = folder / "render"
        cfg = json.loads((folder / "config.json").read_text())
        path = Path(cfg["simulation"])
        if not path.is_absolute():
            path = folder / path
        scene, _ = load(name)
        geometry = dict(np.load(path / "geometry.npz"))
        events = [dict(np.load(p)) for p in sorted(path.glob("event_*.npz"))]
        renderer = Renderer(scene, cfg, geometry, events)
        poses = {
            p["frame"]: p for p in json.loads((folder / "camera_path.json").read_text())
        }
        report[name] = []
        for frame in args.frames:
            p = poses[frame]
            cam = Camera(
                torch.tensor(p["view"], device="cuda:0", dtype=torch.float32),
                torch.tensor(p["K"], device="cuda:0", dtype=torch.float32),
                cfg["width"],
                cfg["height"],
            )
            out, _, _ = renderer.render(cam, frame / cfg["fps"])
            actual = (out.cpu().numpy() * 255).round().astype("uint8")
            expected = np.asarray(Image.open(folder / f"thunder_{frame:05d}.png"))
            error = np.abs(actual.astype(int) - expected.astype(int))
            row = dict(
                frame=frame,
                max_png_error=int(error.max()),
                mean_png_error=float(error.mean()),
                identical=bool(np.array_equal(actual, expected)),
            )
            assert error.max() <= 1 and error.mean() < 0.001, row
            report[name].append(row)
            print(name, row, flush=True)
        del renderer, scene, events, geometry
        gc.collect()
        torch.cuda.empty_cache()
    output = args.output or args.root / "replay_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
