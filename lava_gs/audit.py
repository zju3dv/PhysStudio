"""Audit a cached scene demo and re-render key frames for occlusion checks."""

import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
import warp as wp
from PIL import Image
from scipy.ndimage import map_coordinates
from rainy_gs.scene import Camera
from .scene import Scene
from .mpm import LavaMPM
from .surface import LavaSurface


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    run = args.run
    cfg = json.loads((run / "config.json").read_text())
    assert json.loads((run / "summary.json").read_text())["complete"]
    geo = dict(np.load(run / "geometry.npz"))
    files = sorted((run / "states").glob("*.npz"))
    assert len(files) == cfg["frames"]
    report = {
        "frames": len(files),
        "particles": cfg["particles"],
        "scene": cfg["scene"],
    }
    min_sd = float("inf")
    low_j = float("inf")
    high_j = 0.0
    means = []
    speeds = []
    counts = []
    inflow = cfg.get("inlet_volume_rate_m3_s", 0.0) > 0
    for path in files:
        with np.load(path) as state:
            x = state["x"]
            j = state["j"]
            t = state["temperature"]
            v = state["v"]
            assert all(np.isfinite(state[key]).all() for key in state.files)
            if "skin_age" in state:
                assert state["skin_age"].min() >= 0
                assert (
                    state["skin_age"] <= state["time"] - state["birth"] + 1.0e-5
                ).all()
            counts.append(len(x))
            if inflow:
                expected = cfg["particles"] * (
                    1
                    + int(
                        np.floor(
                            float(state["time"]) * cfg["inlet_speed"] / (cfg["dx"] / 2)
                            + 1.0e-5
                        )
                    )
                )
                assert len(x) == expected, (path, len(x), expected)
                assert state["birth"].min() >= 0
                assert state["birth"].max() <= state["time"] + 1.0e-5
            else:
                assert len(x) == cfg["particles"]
            sd = map_coordinates(
                geo["phi"],
                ((x - geo["origin"]) / float(geo["dx"])).T,
                order=1,
                mode="nearest",
            )
            min_sd = min(min_sd, float(sd.min()))
            low_j = min(low_j, float(j.min()))
            high_j = max(high_j, float(j.max()))
            means.append(float(t.mean()))
            speeds.append(float(np.linalg.norm(v, axis=1).mean()))
    assert min_sd >= -0.001 and low_j > 0.3 and high_j < 2.5
    if not inflow:
        assert np.diff(means).max() < 0.01
    initial = np.load(files[0])["x"]
    final = np.load(files[-1])["x"]
    report.update(
        min_collider_distance_m=min_sd,
        j_range=[low_j, high_j],
        mean_temperature_initial_final_k=[means[0], means[-1]],
        mean_speed_initial_final_m_s=[speeds[0], speeds[-1]],
        particles_initial_final=[counts[0], counts[-1]],
        inflow_ledger_passed=bool(inflow),
        mean_particle_displacement_m=float(
            np.linalg.norm(final[: len(initial)] - initial, axis=1).mean()
        ),
        initial_cohort_mean_drop_m=float(
            (initial[:, 2] - final[: len(initial), 2]).mean()
        ),
    )
    torch.set_num_threads(4)
    scene_config = run / "scene_config.json"
    if not scene_config.is_file():
        scene_config = Path(cfg["config"])
    scene = Scene(cfg["path"], config=json.loads(scene_config.read_text()))
    m = LavaMPM(geo["positions"], geo["temperature"], geo, capacity=max(counts))
    render = LavaSurface(
        m,
        geo,
        cfg["scale"],
        cfg["crust"],
        cfg["exposure"],
        cfg.get("surface_iso", 0.20),
    )
    cameras = json.loads((run / "camera_path.json").read_text())
    images = []
    for i in [0, len(files) // 2, len(files) - 1]:
        state = dict(np.load(files[i]))
        m.count = len(state["x"])
        m.time = float(state["time"])
        wp.to_torch(m.skin_age)[: m.count].copy_(
            torch.as_tensor(
                state.get("skin_age", m.time - state.get("birth", np.zeros(m.count))),
                device=m.device,
            )
        )
        for attr, key in [
            ("x", "x"),
            ("temp", "temperature"),
            ("reference", "reference"),
            ("birth", "birth"),
            ("v", "v"),
            ("fe", "fe"),
            ("j", "j"),
        ]:
            wp.to_torch(getattr(m, attr))[: m.count].copy_(
                torch.as_tensor(state.get(key, np.zeros(m.count)), device=m.device)
            )
        saved = np.array(Image.open(run / f"lava_{i:05d}.png"))
        cam = Camera(
            torch.tensor(cameras[i]["view"], device=m.device),
            torch.tensor(cameras[i]["K"], device=m.device),
            saved.shape[1],
            saved.shape[0],
        )
        source, _ = scene.render(cam)
        out = render.render(source, cam)
        actual = (out.cpu().numpy() * 255 + 0.5).astype("uint8")
        error = np.abs(actual.astype("f") - saved.astype("f"))
        visible = render.last_glow.sum(-1) > 1.0e-4
        violations = int(
            (
                (render.last_depth > source[..., 3] * cfg["scale"] + 1.0e-4) & visible
            ).sum()
        )
        images.append(
            dict(
                frame=i,
                occlusion_violation_pixels=violations,
                replay_mean_error_8bit=float(error.mean()),
                replay_max_error_8bit=float(error.max()),
            )
        )
        assert violations == 0 and error.mean() < 0.2
    report["keyframe_checks"] = images
    m.transfer(0.0)
    expected_mass = m.count * m.volume * m.rho
    relative_mass_error = abs(
        float(m.mass.numpy().sum(dtype=np.float64)) / expected_mass - 1
    )
    report["final_mass_kg"] = expected_mass
    report["relative_particle_grid_mass_error"] = relative_mass_error
    assert relative_mass_error < 2.0e-6
    report["videos"] = {}
    for name in ["lava_demo.mp4", "comparison.mp4"]:
        info = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_frames,r_frame_rate,duration",
                    "-of",
                    "json",
                    str(run / name),
                ]
            )
        )
        stream = info["streams"][0]
        assert int(stream["nb_frames"]) == cfg["frames"]
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(run / name), "-f", "null", "-"],
            check=True,
        )
        report["videos"][name] = stream
    report["passed"] = True
    (run / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
