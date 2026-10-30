"""Build matched-camera controls for simulation and Gaussian rendering changes."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LABELS = [
    "Original simulation + renderer",
    "Original simulation + granular renderer",
    "Refined simulation + granular renderer",
]


def load(path):
    return json.loads(path.read_text())


def panels(folders, frames, output):
    records = [load(folder / "frames.json") for folder in folders]
    maps = [
        {r["source_frame"]: (i, r) for i, r in enumerate(group)} for group in records
    ]
    width, height = 960, 600
    canvas = Image.new("RGB", (width * 3, (height + 45) * len(frames)), "#101820")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 21)
    except OSError:
        font = ImageFont.load_default(size=21)
    for row, frame in enumerate(frames):
        camera = maps[0][frame][1]["camera"]
        for column, (folder, mapping) in enumerate(zip(folders, maps)):
            index, record = mapping[frame]
            for key in camera:
                np.testing.assert_allclose(
                    record["camera"][key], camera[key], atol=1e-6
                )
            image = Image.open(folder / "frames" / f"{index:05d}.png").convert("RGB")
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            x, y = column * width, row * (height + 45)
            draw.text(
                (x + 12, y + 11),
                f"{LABELS[column]} | frame {frame}",
                fill="#e3edf0",
                font=font,
            )
            canvas.paste(image, (x, y + 45))
    canvas.save(output, quality=96)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ["baseline", "render-only", "refined"]:
        p.add_argument("--" + flag, nargs="+", type=Path, required=True)
    p.add_argument(
        "--close",
        nargs=3,
        type=Path,
        required=True,
        metavar=("ORIGINAL", "RENDER_ONLY", "REFINED"),
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if len({len(args.baseline), len(args.render_only), len(args.refined)}) != 1:
        raise ValueError("Provide the same scene order for all three methods")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose a fresh comparison directory")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, cards = [], []
    frames = [0, 60, 179]
    for folders in zip(args.baseline, args.render_only, args.refined):
        configs = [load(folder / "config.json") for folder in folders]
        sources = [Path(c["source"]).resolve() for c in configs]
        assert sources[0] == sources[1], (
            "Rendering control must use the original trajectory"
        )
        sim = [load(source / "config.json") for source in sources]
        assert len({c["scene"] for c in sim}) == 1
        audits = [load(source / "audit.json") for source in sources]
        scene = sim[0]["scene"]
        pairs = []
        for frame in frames:
            times = [a[frame]["time"] for a in audits]
            assert abs(times[0] - times[2]) <= sim[2]["dt"] * 0.501
            pairs.append(
                dict(frame=frame, original_seconds=times[0], refined_seconds=times[2])
            )
        panels(folders, frames, args.output / f"{scene}.jpg")
        summaries.append(
            dict(
                scene=scene,
                matched_times=pairs,
                original_grid=sim[0]["shape"],
                refined_grid=sim[2]["shape"],
                original_dx=sim[0]["dx"],
                refined_dx=sim[2]["dx"],
                original_particles=sim[0]["particles"],
                refined_particles=sim[2]["particles"],
                input_render_directories=[str(f.resolve()) for f in folders],
            )
        )
        cards.append(
            f'<h2>{html.escape(scene.title())}</h2><a href="{scene}.jpg"><img src="{scene}.jpg"></a>'
        )
    panels(args.close, [0, 60, 179], args.output / "close.jpg")
    # Preserve a native-resolution near-camera pair for inspecting real grains.
    for name, folder in [("before", args.close[0]), ("after", args.close[2])]:
        Image.open(folder / "frames" / "00000.png").save(args.output / f"{name}.png")
    (args.output / "comparison.json").write_text(
        json.dumps(
            dict(
                passed=True,
                scenes=summaries,
                close_directories=[str(f.resolve()) for f in args.close],
            ),
            indent=2,
        )
    )
    (args.output / "index.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Water + Sand / Comparisons</title>
<style>body{background:#0a1118;color:#e8f4f5;font:16px system-ui;margin:30px auto;max-width:1800px;padding:0 24px}
h1{font-size:48px;letter-spacing:-.04em}p{color:#a5b9c3;line-height:1.7}img{width:100%;border-radius:12px}
a{color:#90dfcd}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}</style>
<h1>Water + Sand / Controlled comparisons</h1><p>Identical cameras and matched physical times.
Each row compares the original output, rendering changes alone, and the combined simulation and rendering changes.
Click any panel for full resolution.</p><div class="pair"><div><h2>Before</h2><img src="before.png"></div>
<div><h2>After</h2><img src="after.png"></div></div><h2>Near-camera sand detail</h2><a href="close.jpg"><img src="close.jpg"></a>"""
        + "".join(cards)
        + '<p><a href="comparison.json">Correspondence audit</a></p></html>'
    )
    print(json.dumps(dict(passed=True, output=str(args.output.resolve())), indent=2))


if __name__ == "__main__":
    main()
