"""Reproduce a selected magnetic-MPM / 3DGS demonstration and audit its replay."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

PRESETS = {
    "ferrofluid": dict(
        mode="fluid",
        asset="bread",
        n=64,
        frames=180,
        dt=0.00005,
        substeps=100,
        field_every=40,
        bulk=5000000,
        viscosity=80,
        intensity=1000000,
    ),
    "ferrofluid_control": dict(
        mode="fluid",
        asset="bread",
        n=64,
        frames=180,
        dt=0.00005,
        substeps=100,
        field_every=40,
        bulk=5000000,
        viscosity=80,
        intensity=0,
    ),
    "magnetic_bread": dict(
        mode="solid",
        asset="bread",
        n=64,
        frames=180,
        dt=0.00005,
        substeps=100,
        field_every=40,
        bulk=1000000,
        shear=12000,
        viscosity=60,
        intensity=1000000,
        clamp_height=0.035,
        magnet_shift=[-0.07, 0, 0],
    ),
    "magnetic_wolf": dict(
        mode="solid",
        asset="wolf",
        n=64,
        frames=180,
        dt=0.00008,
        substeps=60,
        field_every=25,
        bulk=1000000,
        shear=20000,
        viscosity=60,
        intensity=700000,
        clamp_height=0.035,
        magnet_shift=[-0.07, 0, -0.01],
    ),
}


def audit(root):
    root = Path(root)
    cfg = json.loads((root / "config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"], "Incomplete simulation"
    trajectory = np.load(root / "trajectory.npz")
    positions = trajectory["positions"]
    f = trajectory["deformation"]
    velocity = trajectory["velocities"]
    assert len(positions) == cfg["frames"] + 1
    assert (
        np.isfinite(positions).all()
        and np.isfinite(f).all()
        and np.isfinite(velocity).all()
    )
    j = np.linalg.det(f)
    assert j.min() > 0
    magnet = trajectory["magnet_center"]
    penetration = cfg["radius"] - np.linalg.norm(positions - magnet[:, None], axis=-1)
    initial_clearance = float(-penetration[0].max())
    max_penetration = float(max(0.0, penetration.max()))
    assert initial_clearance >= 0, "Initial intersection with permanent magnet"
    assert positions[..., 2].min() > -cfg["dx"]
    assert max_penetration < cfg["dx"], "Contact penetration exceeds a grid cell"
    field = json.loads((root / "field_solves.json").read_text())
    assert all(s["converged"] for s in field)
    result = dict(
        complete=True,
        frames=cfg["frames"],
        physical_duration=float(trajectory["time"][-1]),
        particle_count=positions.shape[1],
        gaussian_count=cfg["asset"]["gaussians"],
        minimum_j=float(j.min()),
        maximum_j=float(j.max()),
        j_percentiles=np.quantile(j, [0.001, 0.01, 0.5, 0.99, 0.999]).tolist(),
        initial_magnet_clearance_m=initial_clearance,
        maximum_magnet_penetration_m=max_penetration,
        minimum_particle_height_m=float(positions[..., 2].min()),
        final_height_p99_m=float(np.quantile(positions[-1, :, 2], 0.99)),
        center_displacement_m=(positions[-1].mean(0) - positions[0].mean(0)).tolist(),
        max_field_relative_residual=max(s["relative_residual"] for s in field),
        field_solves=len(field),
        median_field_solve_seconds=float(np.median([s["seconds"] for s in field])),
        simulation_and_save_seconds=summary["seconds"],
        asset_sha256=cfg["asset"]["sha256"],
    )
    video = root / "render/demo.mp4"
    if video.exists():
        info = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_read_frames,r_frame_rate",
                    "-of",
                    "json",
                    str(video),
                ],
                text=True,
            )
        )
        assert int(info["streams"][0]["nb_read_frames"]) == len(positions)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        result["video"] = info["streams"][0]
    (root / "audit.json").write_text(json.dumps(result, indent=2))
    return result


def compare(on, off):
    on, off = Path(on), Path(off)
    ca, cb = [json.loads((p / "config.json").read_text()) for p in (on, off)]
    for key in [
        "mode",
        "asset",
        "n",
        "frames",
        "dt",
        "substeps",
        "field_every",
        "bulk",
        "viscosity",
        "shear",
        "susceptibility",
        "radius",
        "clamp_height",
        "magnet_shift",
    ]:
        assert ca[key] == cb[key], "Mismatched control parameter: " + key
    assert cb["intensity"] == 0
    a, b = [np.load(p / "trajectory.npz") for p in (on, off)]
    assert np.array_equal(a["rest"], b["rest"]) and np.allclose(a["time"], b["time"])
    assert np.array_equal(a["magnet_center"], b["magnet_center"])
    xa, xb = a["positions"], b["positions"]
    result = dict(
        magnetic_run=str(on),
        control_run=str(off),
        physical_time=float(a["time"][-1]),
        magnetic_p99_height_m=float(np.quantile(xa[-1, :, 2], 0.99)),
        control_p99_height_m=float(np.quantile(xb[-1, :, 2], 0.99)),
        magnetic_mass_fraction_above_8cm=float(np.mean(xa[-1, :, 2] > 0.08)),
        control_mass_fraction_above_8cm=float(np.mean(xb[-1, :, 2] > 0.08)),
        matched_initial_particles=True,
        matched_material_and_timestep=True,
    )
    result["p99_lift_difference_m"] = (
        result["magnetic_p99_height_m"] - result["control_p99_height_m"]
    )
    difference = xa[-1] - xb[-1]
    upper = a["rest"][:, 2] >= np.quantile(a["rest"][:, 2], 0.75)
    result["center_on_minus_off_m"] = difference.mean(0).tolist()
    result["upper_quartile_on_minus_off_m"] = difference[upper].mean(0).tolist()
    result["particle_displacement_difference_rms_m"] = float(
        np.sqrt((difference * difference).sum(-1).mean())
    )
    (on / "control_comparison.json").write_text(json.dumps(result, indent=2))
    return result


def run(args):
    root = Path(args.output)
    if root.exists():
        raise FileExistsError("Choose a fresh output directory: " + str(root))
    config = PRESETS[args.case].copy()
    cmd = [sys.executable, "-u", "-m", "magnetic_mpm.demo", "--output", str(root)]
    for key, value in config.items():
        cmd += ["--" + key.replace("_", "-")]
        cmd += list(map(str, value)) if isinstance(value, list) else [str(value)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    code_root = Path(__file__).parent
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in code_root.glob("*.py")
    }
    subprocess.run(cmd, env=env, check=True)
    (root / "source_manifest.json").write_text(
        json.dumps(dict(preset=args.case, command=cmd, sha256=hashes), indent=2)
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "magnetic_mpm.render",
            "--input",
            str(root),
            "--output",
            str(root / "render"),
        ],
        env=env,
        check=True,
    )
    print(json.dumps(audit(root), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--case", choices=PRESETS, required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--gpu", type=int, default=0)
    a = sub.add_parser("audit")
    a.add_argument("--input", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--magnetic", required=True)
    c.add_argument("--control", required=True)
    args = p.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "audit":
        print(json.dumps(audit(args.input), indent=2))
    else:
        print(json.dumps(compare(args.magnetic, args.control), indent=2))
