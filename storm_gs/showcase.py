"""Run the three storm scenes, validate outputs, and build a local video gallery."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw


TITLES = {
    "playground": "Playground / Rain-wrapped tornado",
    "lands_end": "Land's End / Coastal waterspout",
    "calico": "Calico / Dust devil",
}
from physstudio.paths import ROOT
from physstudio.__main__ import environment


def audit_scene(path):
    config = json.loads((path / "config.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    assert summary["complete"] and not summary["preview"]
    assert len(list(path.glob("storm_*.png"))) == config["frames"]
    movie_info = {}
    for name in ["storm_demo.mp4", "comparison.mp4"]:
        movie = path / name
        probe = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,r_frame_rate:format=duration",
                "-of",
                "json",
                str(movie),
            ],
            text=True,
        )
        probe = json.loads(probe)
        assert int(probe["streams"][0]["nb_frames"]) == config["frames"]
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "2",
                "-i",
                str(movie),
                "-f",
                "null",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        movie_info[name] = probe
    cameras = json.loads((path / "camera_path.json").read_text())
    assert len(cameras) == config["frames"]
    eyes = [np.linalg.inv(np.array(c["view"]))[:3, 3] for c in cameras]
    movement = float(np.linalg.norm(eyes[-1] - eyes[0]))
    assert movement > 0.1
    records = json.loads((path / "diagnostics.json").read_text())["frames"]
    assert all(
        np.isfinite(v)
        for r in records
        for v in [
            r["max_component_speed"],
            r["divergence_rms"],
            r["cloud_mass"],
            r["dust_mass"],
        ]
    )
    assert max(r["divergence_rms"] for r in records) < 0.005
    geometry = np.load(path / "geometry.npz")
    solid = geometry["solid"].astype(bool)
    wall_max = 0.0
    for filename in sorted(path.glob("state_*.npz")):
        state = np.load(filename)
        for key in ["air", "dust", "u", "v", "w", "particle_x", "particle_v"]:
            assert np.isfinite(state[key]).all(), (filename, key)
        assert state["air"][..., 1:].min() >= -1.0e-7
        assert state["dust"].min() >= 0.0
        for axis, key in enumerate(["u", "v", "w"]):
            face = state[key]
            mask = np.zeros(face.shape, dtype=bool)
            lo = [slice(0, n) for n in solid.shape]
            hi = lo.copy()
            hi[axis] = slice(1, solid.shape[axis] + 1)
            mask[tuple(lo)] |= solid
            mask[tuple(hi)] |= solid
            edge = [slice(None)] * 3
            edge[axis] = 0
            mask[tuple(edge)] = True
            edge[axis] = -1
            mask[tuple(edge)] = True
            wall_max = max(wall_max, float(np.abs(face[mask]).max()))
    assert wall_max == 0.0
    result = dict(
        passed=True,
        videos=movie_info,
        camera_travel_scene_units=movement,
        maximum_saved_wall_normal_speed=wall_max,
        frame_divergence_rms_range=[
            min(r["divergence_rms"] for r in records),
            max(r["divergence_rms"] for r in records),
        ],
        max_component_speed=max(r["max_component_speed"] for r in records),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((path / "source").glob("*.py"))
        },
    )
    (path / "audit.json").write_text(json.dumps(result, indent=2))
    return result


def gallery(output, names):
    cards = []
    sheet = Image.new("RGB", (1152, len(names) * 270), "#111827")
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(names):
        path = output / name
        config = json.loads((path / "config.json").read_text())
        keyframes = [0, config["frames"] // 2, config["frames"] - 1]
        draw.text((12, row * 270 + 6), TITLES[name], fill="white")
        for col, frame in enumerate(keyframes):
            with Image.open(path / f"storm_{frame:05d}.png") as image:
                image.thumbnail((384, 240))
                sheet.paste(image, (col * 384, row * 270 + 28))
        poster = f"{name}/storm_{keyframes[1]:05d}.png"
        cards.append(
            f"""<section><h2>{html.escape(TITLES[name])}</h2>
<video controls autoplay muted loop playsinline preload="metadata" poster="{poster}">
<source src="{name}/storm_demo.mp4" type="video/mp4"></video>
<p><a href="{name}/comparison.mp4">Before / after video</a> ·
<a href="{name}/audit.json">Simulation and video checks</a></p></section>"""
        )
    sheet.save(output / "contact_sheet.jpg", quality=93)
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>StormGS · Local atmospheric simulations</title>
<style>body{max-width:1280px;margin:48px auto;padding:0 24px;background:#0c111b;
color:#e4eaf3;font:16px/1.5 system-ui}h1{font-size:42px;letter-spacing:-1px}
h2{font-size:21px}section{padding:20px;margin:28px 0;background:#161e2b;border-radius:18px}
video{width:100%;border-radius:10px;background:#000}a{color:#8bc8ff}p{color:#adb9cc}</style>
<h1>StormGS</h1><p>Moving-camera 3DGS scenes with Warp atmospheric flow,
condensation, transported dust, and inertial rain/debris.</p>
<p><a href="https://computationalsciences.org/publications/amador-herrera-2024-cyclogenesis.html">
Cyclogenesis: Simulating Hurricanes and Tornadoes</a></p>"""
    (output / "index.html").write_text(document + "\n".join(cards) + "</html>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    names = list(TITLES)
    if not args.audit_only:
        output.mkdir(parents=True, exist_ok=False)
        env = environment(args.gpu)
        jobs = [
            (
                "numerical_validation",
                [
                    "-m",
                    "storm_gs.validate",
                    "--output",
                    str(output / "numerical_validation.json"),
                ],
            )
        ]
        for name in names:
            jobs.append(
                (
                    name,
                    [
                        "-m",
                        "storm_gs.demo",
                        "--scene",
                        name,
                        "--output",
                        str(output / name),
                        "--frames",
                        str(args.frames),
                        "--width",
                        str(args.width),
                        "--warmup",
                        "8" if name == "calico" else "5",
                        "--dt",
                        ".012",
                    ],
                )
            )
        for name, command in jobs:
            print("Starting", name, flush=True)
            with (output / f"{name}.log").open("w") as log:
                subprocess.run(
                    [sys.executable, *command],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print("Finished", name, flush=True)
    result = {name: audit_scene(output / name) for name in names}
    gallery(output, names)
    (output / "audit.json").write_text(json.dumps(result, indent=2))
    print("Gallery:", output / "index.html", flush=True)


if __name__ == "__main__":
    main()
