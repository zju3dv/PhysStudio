"""Run, render, and audit the three water-sand demonstrations."""

import argparse
from pathlib import Path
import subprocess
import sys


PRESETS = {
    "castle": dict(
        resolution=96,
        steps_per_frame=40,
        settle=300,
        warmup=80,
        young=500000,
        cohesion=0.08,
    ),
    "sediment": dict(resolution=96, steps_per_frame=40, settle=0, warmup=80),
    "breach": dict(resolution=112, steps_per_frame=48, settle=400, warmup=100),
}

# Physical rates from the original validated trajectories, independent of
# output folders. Refined runs keep these rates while changing the grid.
REFERENCE_DT = {
    "castle": 0.00040245440701357935,
    "sediment": 0.00040245440701357935,
    "breach": 0.00034216802604759616,
}


def settings(scene, quality):
    values = dict(PRESETS[scene])
    if quality == "refined":
        dt = REFERENCE_DT[scene]
        values.update(
            resolution=128,
            frame_dt=dt * values.pop("steps_per_frame"),
            settle_seconds=dt * values.pop("settle"),
            absorption_per_second=0.003 / dt,
            drag_reference_dx=3.6 / (PRESETS[scene]["resolution"] - 6),
        )
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", choices=PRESETS, default=list(PRESETS))
    parser.add_argument("--quality", choices=["refined", "legacy"], default="refined")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()
    if args.frames < 1 or args.width < 32 or args.width % 2:
        parser.error("Use positive frames and an even width of at least 32")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Choose a fresh experiment directory")
    root.mkdir(parents=True, exist_ok=True)
    simulations, renders = [], []
    for scene in args.scenes:
        scene_root = root / scene if len(args.scenes) > 1 else root
        sim, render = scene_root / "simulation", scene_root / "gaussian"
        command = [
            sys.executable,
            "-u",
            "-m",
            "water_sand.run",
            "--scene",
            scene,
            "--frames",
            str(args.frames),
            "--output",
            str(sim),
        ]
        for key, value in settings(scene, args.quality).items():
            command += ["--" + key.replace("_", "-"), str(value)]
        subprocess.run(command, check=True)
        renderer = "render_granular" if args.quality == "refined" else "render_gaussian"
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "water_sand." + renderer,
                "--input",
                str(sim),
                "--output",
                str(render),
                "--width",
                str(args.width),
                "--height",
                str(round(args.width * 5 / 16) * 2),
            ],
            check=True,
        )
        simulations.append(str(sim))
        renders.append(str(render))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "water_sand.report",
            "--simulations",
            *simulations,
            "--renders",
            *renders,
            "--output",
            str(root / "showcase"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
