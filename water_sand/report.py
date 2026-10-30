"""Audit simulation/render correspondence and build a local video gallery."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont


TITLES = {
    "castle": "Sandcastle / erosion",
    "sediment": "Three pigments / sedimentation",
    "breach": "Sand dam / overtopping",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--simulations", nargs="+", type=Path, required=True)
    p.add_argument("--renders", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--comparisons", type=Path, help="Audited output of water_sand.compare"
    )
    args = p.parse_args()
    if len(args.simulations) != len(args.renders):
        raise ValueError("Provide one render folder per simulation")
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a fresh gallery directory")
    out.mkdir(parents=True, exist_ok=True)
    records, cards = [], []
    contact = Image.new("RGB", (1200, len(args.renders) * 282), "#101820")
    draw = ImageDraw.Draw(contact)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = (
        ImageFont.truetype(font_path, 17)
        if Path(font_path).exists()
        else ImageFont.load_default()
    )
    for row, (sim, render) in enumerate(zip(args.simulations, args.renders)):
        cfg = json.loads((sim / "config.json").read_text())
        audit = json.loads((sim / "audit.json").read_text())
        complete = json.loads((sim / "complete.json").read_text())
        rcfg = json.loads((render / "config.json").read_text())
        frames = json.loads((render / "frames.json").read_text())
        n = cfg["frames"]
        scene = cfg["scene"]
        assert len(audit) == n == complete["frames"] == len(frames) == rcfg["frames"]
        assert Path(rcfg["source"]).resolve() == sim.resolve()
        assert [f["source_frame"] for f in frames] == list(range(n))
        assert len(list((sim / "states").glob("*.npz"))) == n
        assert len(list((render / "frames").glob("*.png"))) == n
        assert all(
            a["finite"] and a["particles_inside"] and a["min_det"] > 0 for a in audit
        )
        assert all(a["finite"] and a["covariance_min_eigenvalue"] > 0 for a in frames)
        water_error = max(abs(a["water_relative_error"]) for a in audit)
        sand_error = max(abs(a["sand_mass_relative_error"]) for a in audit)
        assert water_error < 1e-4 and sand_error < 2e-6
        video = render / "demo.mp4"
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
                    "stream=width,height,nb_read_frames,duration,avg_frame_rate",
                    "-of",
                    "json",
                    str(video),
                ],
                text=True,
            )
        )["streams"][0]
        assert int(probe["nb_read_frames"]) == n
        assert (
            int(probe["width"]) == rcfg["width"]
            and int(probe["height"]) == rcfg["height"]
        )
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True
        )
        shutil.copy2(video, out / f"{scene}.mp4")
        shutil.copy2(render / "frames" / "00000.png", out / f"{scene}_poster.png")
        ids = [0, n // 8, n - 1]
        draw.text((12, row * 282 + 6), TITLES[scene], fill="#d7e9ee", font=font)
        for col, frame in enumerate(ids):
            image = (
                Image.open(render / "frames" / f"{frame:05d}.png")
                .convert("RGB")
                .resize((400, 250))
            )
            contact.paste(image, (col * 400, row * 282 + 32))
        record = dict(
            scene=scene,
            frames=n,
            physical_seconds=audit[-1]["time"],
            grid=cfg["shape"],
            dx_meters=cfg["dx"],
            sand_particles=cfg["particles"],
            output_step_counts=[
                a["step"] - (audit[i - 1]["step"] if i else 0)
                for i, a in enumerate(audit)
            ],
            renderer=rcfg.get("renderer", "surface_gaussian"),
            max_water_relative_error=water_error,
            max_sand_mass_relative_error=sand_error,
            phase_range=[
                min(a["phi_min"] for a in audit),
                max(a["phi_max"] for a in audit),
            ],
            simulation=str(sim.resolve()),
            render=str(render.resolve()),
            video=probe,
            sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
            passed=True,
        )
        records.append(record)
        appearance = (
            "material-attached Gaussian grains"
            if rcfg.get("renderer") == "granular"
            else "anisotropic Gaussian splatting"
        )
        details = f"{rcfg['width']} × {rcfg['height']} · {rcfg['fps']} fps · {n / rcfg['fps']:.2f} s"
        cards.append(f'''<article><div class="label">{html.escape(TITLES[scene])}</div>
<video src="{scene}.mp4" poster="{scene}_poster.png" autoplay loop muted playsinline controls></video>
<p>3D LBM–MPM · {appearance} · 62° camera orbit<br>{details}</p></article>''')
    contact.save(out / "contact_sheet.jpg", quality=95)
    (out / "summary.json").write_text(
        json.dumps(dict(passed=True, scenes=records), indent=2)
    )
    comparison_link = ""
    if args.comparisons:
        comparison = json.loads((args.comparisons / "comparison.json").read_text())
        assert comparison["passed"]
        shutil.copytree(args.comparisons, out / "comparisons")
        comparison_link = '<p><a href="comparisons/index.html">Before / rendering only / simulation + rendering</a></p>'
    (out / "index.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Water + Sand</title>
<style>*{box-sizing:border-box}body{margin:0;background:#0a1118;color:#e8f4f5;font:16px system-ui}
main{max-width:1380px;margin:60px auto;padding:0 32px}.tag{letter-spacing:.25em;color:#78cbbb;font-size:12px}
h1{font-size:clamp(42px,7vw,88px);letter-spacing:-.06em;margin:12px 0}header p{color:#a5b9c3;max-width:820px;line-height:1.7}
.line{height:1px;background:linear-gradient(90deg,#74ddc3,#d8ad68,transparent);margin:32px 0}
article{background:#121e28;border:1px solid #29404c;border-radius:16px;overflow:hidden;margin:24px 0}
.label{padding:18px 24px;font-size:22px;font-weight:600}video{width:100%;display:block;background:#0a1118}
article p{color:#9bb7c2;padding:0 24px 8px;font-size:13px}a{color:#90dfcd}footer{padding:24px 0;color:#91a5b1}
</style><main><header><span class="tag">PHYSICS / GAUSSIAN RENDERING</span><h1>Water + Sand</h1>
<p>Air–water LBM meets frictional MPM sand. Simulated wetting, erosion, collapse, and sedimentation,
viewed through Gaussian surfaces and refractive water.</p>
<p>Inspired by <a href="https://lwkobe.github.io/papers/XWYDL2026/">Volume-Preserving LBM-MPM Coupling
for Air-Water-Sand Mixtures</a>. Three-dimensional Warp implementation and procedural Gaussian materials.</p>
</header>"""
        + comparison_link
        + """<div class="line"></div>"""
        + "".join(cards)
        + """
<footer>
<a href="contact_sheet.jpg">Contact sheet</a> · <a href="summary.json">Numerical and video audit</a></footer></main></html>"""
    )
    print(
        json.dumps(
            dict(
                gallery=str((out / "index.html").resolve()), passed=True, scenes=records
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
