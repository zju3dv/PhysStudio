"""Run a named scene into a fresh directory or audit a completed snow demo."""

import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from .geometry import PRESETS


def audit(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    assert summary["complete"] and not summary["preview"]
    state = np.load(path / "snow_state.npz")
    geo = np.load(path / "geometry.npz")
    assert all(np.isfinite(state[k]).all() for k in state.files)
    initial = config["mpm_particles_initial"]
    deposited = summary["final"]["deposited"]
    assert len(state["x"]) == initial + deposited
    sv = np.linalg.svd(state["fe"], compute_uv=False)
    assert (
        sv.min() >= 1 - config["theta_c"] - 2.0e-5
        and sv.max() <= 1 + config["theta_s"] + 2.0e-5
    )
    assert state["jp"].min() >= 0.6 - 1.0e-6 and state["jp"].max() <= 2.0 + 1.0e-6
    mass = state["vol"].astype("d").sum() * config["density_kg_m3"]
    packet_volume = (float(geo["dx"]) * 0.5) ** 3 * 0.02
    expected = (
        state["vol"][:initial].astype("d").sum() * config["density_kg_m3"]
        + deposited * packet_volume * config["density_kg_m3"]
    )
    relative = abs(mass - expected) / expected
    assert relative < 1.0e-6
    coords = (state["x"] - geo["origin"]) / float(geo["dx"])
    assert (coords > 1.5).all() and (coords < np.array(geo["phi"].shape) - 2.5).all()
    sd = map_coordinates(geo["phi"], coords.T, order=1, mode="nearest")
    penetration = float(max(0.0, -sd.min()))
    assert penetration < 0.002
    video = {}
    for filename in ["snow_demo.mp4", "comparison.mp4"]:
        value = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=width,height,nb_frames,avg_frame_rate,duration",
                    "-of",
                    "json",
                    str(path / filename),
                ]
            )
        )["streams"][0]
        assert int(value["nb_frames"]) == config["frames"]
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path / filename), "-f", "null", "-"],
            check=True,
            capture_output=True,
        )
        video[filename] = value
    result = dict(
        passed=True,
        particles=len(state["x"]),
        deposited=deposited,
        mass_kg=mass,
        mass_ledger_relative_error=relative,
        collision_proxy_penetration_m=penetration,
        elastic_stretch_min=float(sv.min()),
        elastic_stretch_max=float(sv.max()),
        speed_p95=float(np.quantile(np.linalg.norm(state["v"], axis=-1), 0.95)),
        video=video,
    )
    (path / "audit.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--scene", choices=PRESETS, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--gpu", default="1")
    r.add_argument("--width", type=int, default=1280)
    r.add_argument("--frames", type=int, default=240)
    r.add_argument("--preview", action="store_true")
    q = sub.add_parser("audit")
    q.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    if a.command == "audit":
        print(json.dumps(audit(a.run), indent=2))
        return
    if a.output.exists():
        p.error("Choose a fresh output directory")
    cmd = [
        sys.executable,
        "-m",
        "snowy_gs.demo",
        "--scene",
        a.scene,
        "--output",
        str(a.output),
        "--width",
        str(a.width),
        "--frames",
        str(a.frames),
        "--warmup",
        "2",
        "--haze",
        ".003",
    ]
    if a.preview:
        cmd.append("--preview")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = a.gpu
    subprocess.run(cmd, env=env, check=True)
    if not a.preview:
        print(json.dumps(audit(a.output), indent=2))


if __name__ == "__main__":
    main()
