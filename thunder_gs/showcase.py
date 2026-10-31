"""Generate, audit, and present the three electrical-storm demos."""

import argparse
from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from physstudio.paths import ROOT
from physstudio.__main__ import environment

ORDER = ("firewoods", "zohar", "train")


def audit_scene(folder):
    cfg = json.loads((folder / "config.json").read_text())
    assert not cfg.get("preview", False)
    info = {}
    for name in ("thunder_demo.mp4", "comparison.mp4"):
        path = folder / name
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_read_frames,r_frame_rate,duration",
                    "-of",
                    "json",
                    str(path),
                ]
            )
        )["streams"][0]
        assert int(probe["nb_read_frames"]) == cfg["frames"]
        assert probe["width"] == cfg["width"] * (2 if name == "comparison.mp4" else 1)
        assert probe["height"] == cfg["height"]
        assert Fraction(probe["r_frame_rate"]) == cfg["fps"]
        assert abs(float(probe["duration"]) - cfg["frames"] / cfg["fps"]) < 0.001
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "2",
                "-i",
                str(path),
                "-f",
                "null",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        info[name] = dict(**probe, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    frames = sorted(folder.glob("thunder_*.png"))
    assert len(frames) == cfg["frames"]
    diagnostics = json.loads((folder / "diagnostics.json").read_text())
    assert len(diagnostics) == cfg["frames"] and all(d["finite"] for d in diagnostics)
    poses = json.loads((folder / "camera_path.json").read_text())
    assert len(poses) == cfg["frames"]
    eyes = [np.linalg.inv(p["view"])[:3, 3] for p in poses]
    travel = float(np.linalg.norm(eyes[-1] - eyes[0]))
    if cfg["frames"] > 1:
        assert travel > 0.3
    return dict(videos=info, camera_translation=travel, finite_frames=len(diagnostics))


def assemble(root):
    cards, rows, report = [], [], {}
    for name in ORDER:
        folder = root / name / "render"
        cfg = json.loads((folder / "config.json").read_text())
        report[name] = audit_scene(folder)
        indices = [min(i, cfg["frames"] - 1) for i in (33, 110, 186)]
        row = Image.new("RGB", (1440, 310), (13, 19, 29))
        draw = ImageDraw.Draw(row)
        for j, index in enumerate(indices):
            with Image.open(folder / f"thunder_{index:05d}.png") as image:
                image.thumbnail((480, 280))
                row.paste(
                    image,
                    (
                        j * 480 + (480 - image.width) // 2,
                        28 + (280 - image.height) // 2,
                    ),
                )
            draw.text(
                (j * 480 + 12, 9),
                f"{name.title()} / frame {index}",
                fill=(214, 228, 248),
            )
        rows.append(row)
        shutil.copyfile(folder / f"thunder_{indices[0]:05d}.png", folder / "poster.png")
        cards.append(
            f"<article><h2>{html.escape(name.title())}</h2>"
            f'<video controls autoplay muted loop playsinline poster="{name}/render/poster.png" '
            f'src="{name}/render/thunder_demo.mp4"></video>'
            f'<p><a href="{name}/render/comparison.mp4">Original / lightning comparison</a>'
            f' · <a href="{name}/render/config.json">Configuration</a></p></article>'
        )
    sheet = Image.new("RGB", (1440, len(rows) * 310))
    for i, row in enumerate(rows):
        sheet.paste(row, (0, i * 310))
    sheet.save(root / "contact_sheet.jpg", quality=94)
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ThunderGS</title>
<style>body{margin:0;background:#090d14;color:#e7edf7;font:16px/1.6 system-ui}
main{max-width:1280px;margin:auto;padding:48px 24px}h1{font-size:52px;margin:0}
article{margin-top:36px;padding:22px;background:#111a28;border:1px solid #26334a;border-radius:18px}
video{width:100%;max-height:720px;border-radius:10px;background:#000}a{color:#9bbcff}</style>
<main><h1>ThunderGS</h1><p>Electric-field-driven lightning in Gaussian scenes.</p>"""
    page += "".join(cards)
    page += '<p><a href="contact_sheet.jpg">Strike contact sheet</a> · <a href="video_audit.json">Video audit</a></p></main></html>'
    (root / "index.html").write_text(page)
    (root / "video_audit.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    if not args.audit_only:
        if args.frames < 1 or args.width < 32 or args.width % 2:
            parser.error("Frames must be positive; width must be even and at least 32.")
        args.output.mkdir(parents=True, exist_ok=False)
        for name in ORDER:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "physstudio",
                    "run",
                    "thunder",
                    name,
                    "--gpu",
                    args.gpu,
                    "--output",
                    str(args.output / name),
                    "--frames",
                    str(args.frames),
                    "--width",
                    str(args.width),
                ],
                cwd=ROOT,
                env=environment(args.gpu),
                check=True,
            )
    print(json.dumps(assemble(args.output), indent=2))


if __name__ == "__main__":
    main()
