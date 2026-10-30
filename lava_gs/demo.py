"""Thermomechanical lava in a static Gaussian scene, with moving-camera renders."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import time
import numpy as np
import torch
import warp as wp
from physstudio.paths import ROOT
from scipy.ndimage import map_coordinates
from rainy_gs.scene import save_image
from .scene import Scene
from rainy_gs.demo import Video, transition_camera
from .geometry import PRESETS
from .mpm import LavaMPM
from .surface import LavaSurface
from .inlet import LavaInlet
from .canyon import build_canyon, height_at
from .cameras import cascade_camera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=PRESETS, default="zohar")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--dx", type=float, default=0.16)
    p.add_argument("--inlet-speed", type=float, default=1.5)
    p.add_argument("--inlet-width", type=float, default=1.5)
    p.add_argument("--inlet-thickness", type=float, default=0.65)
    p.add_argument("--eta", type=float, default=900.0)
    p.add_argument("--yield-hot", type=float, default=700.0)
    p.add_argument("--bulk", type=float, default=1.2e6)
    p.add_argument("--volume-relaxation", type=float, default=0.02)
    p.add_argument("--inlet-x", type=float, default=-1.25)
    p.add_argument("--inlet-y", type=float, default=-0.25)
    p.add_argument(
        "--inlet-heading",
        type=float,
        default=0.0,
        help="Horizontal direction in degrees from +X",
    )
    p.add_argument("--inlet-profile", choices=["ellipse", "slot"], default="ellipse")
    p.add_argument(
        "--inlet-conform",
        action="store_true",
        help="Fit the inlet bottom to the local rock height",
    )
    p.add_argument("--friction", type=float, default=0.06)
    p.add_argument("--exchange", type=float, default=20.0)
    p.add_argument("--crust", type=float, default=1.0)
    p.add_argument("--surface-iso", type=float, default=0.20)
    p.add_argument("--exposure", type=float, default=0.55)
    p.add_argument("--camera-pitch", type=float, default=8.0)
    p.add_argument("--camera-end", type=float, default=0.35)
    p.add_argument("--camera-path", choices=["dataset", "cascade"], default="dataset")
    p.add_argument("--camera-config", type=Path, help="Camera settings JSON")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--save-frames", action="store_true")
    p.add_argument("--cache-states", action="store_true")
    p.add_argument("--replay", type=Path)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--scene", choices=PRESETS, default="zohar")
    selected, _ = selector.parse_known_args()
    scene_defaults = json.loads(
        Path(PRESETS[selected.scene]["config"]).read_text()
    ).get("lava_defaults", {})
    p.set_defaults(**scene_defaults)
    args = p.parse_args()
    if args.frames < 1 or args.width < 64 or args.width % 2:
        p.error("Use positive frames and even width >=64")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = time.monotonic()
    cfg = PRESETS[args.scene].copy()
    cfg["dx"] = args.dx
    replay_config = None
    if args.replay:
        replay_config = json.loads((args.replay / "config.json").read_text())
        if replay_config["scene"] != args.scene:
            raise ValueError("Replay scene differs from requested scene")
        for name in ["path", "config", "scale", "cameras", "bounds", "dx", "height"]:
            cfg[name] = replay_config[name]
        for name in [
            "eta",
            "yield_hot",
            "friction",
            "exchange",
            "volume_relaxation",
            "inlet_speed",
            "inlet_width",
            "inlet_thickness",
            "inlet_x",
            "inlet_y",
            "inlet_heading",
            "inlet_profile",
            "inlet_conform",
            "warmup",
            "seed",
        ]:
            if name in replay_config:
                setattr(args, name, replay_config[name])
        args.bulk = replay_config["bulk_pa"]
    scene = Scene(cfg["path"], config=json.loads(Path(cfg["config"]).read_text()))
    if args.camera_config:
        scene.config["lava_camera"] = json.loads(args.camera_config.read_text())
    if args.replay:
        geo = dict(np.load(args.replay / "geometry.npz"))
        cache = sorted((args.replay / "states").glob("*.npz"))
        if len(cache) < args.frames:
            raise ValueError("Replay has fewer states than requested frames")
        state = dict(np.load(cache[0]))
        geo["positions"] = state["x"]
        geo["temperature"] = state["temperature"]
        geo["velocity"] = state["v"]
    else:
        print("Building scene collision volume", flush=True)
        geo = build_canyon(scene, cfg, args.seed)
    emitter = None
    capacity = len(geo["positions"])
    if cfg.get("collider") == "terrain" and not args.replay:
        center = np.array([args.inlet_x, args.inlet_y, 0.0]) * cfg["scale"]
        angle = np.deg2rad(args.inlet_heading)
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        lateral = np.cross(direction, [0.0, 0.0, 1.0])
        xy = (
            center[:2]
            + np.linspace(-args.inlet_width / 2, args.inlet_width / 2, 20)[:, None]
            * lateral[:2]
        )
        center[2] = height_at(geo, xy).max() + cfg["dx"] * 0.55
        emitter = LavaInlet(
            center,
            args.inlet_width / 2,
            args.inlet_thickness,
            args.inlet_speed,
            cfg["dx"] / 2,
            temperature=1490.0,
            seed=args.seed,
            direction=direction,
            profile=args.inlet_profile,
        )
        if args.inlet_conform:
            positions = center + emitter.section
            emitter.section[:, 2] += (
                height_at(geo, positions[:, :2]) + cfg["dx"] * 0.55 - center[2]
            )
        capacity = (
            int(
                emitter.volume_rate
                * (args.frames / args.fps + args.warmup + 1)
                / (cfg["dx"] / 2) ** 3
            )
            + 1024
        )
        geo["inlet_center"] = center
        geo["inlet_volume_rate"] = emitter.volume_rate
        geo["inlet_section"] = emitter.section
        geo["inlet_direction"] = direction
        print(
            "Elevated inlet", center, "m, flow", emitter.volume_rate, "m3/s", flush=True
        )
    elif args.replay:
        capacity = max(capacity, len(np.load(cache[args.frames - 1])["x"]))
    print("Grid", geo["phi"].shape, "particles", len(geo["positions"]), flush=True)
    m = LavaMPM(
        geo["positions"],
        geo["temperature"],
        geo,
        geo["velocity"],
        capacity=capacity,
        bulk=args.bulk,
        volume_relaxation=args.volume_relaxation,
        eta=args.eta,
        yield_hot=args.yield_hot,
        friction=args.friction,
        exchange=args.exchange,
    )
    if emitter is not None:
        emitter.seed(m)
    renderer = LavaSurface(
        m, geo, cfg["scale"], args.crust, args.exposure, args.surface_iso
    )
    config = {
        **vars(args),
        **cfg,
        "output": str(args.output),
        "replay": str(args.replay) if args.replay else None,
        "camera_config": str(args.camera_config) if args.camera_config else None,
        "camera_settings": scene.config.get("lava_camera", {}),
        "code_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in Path(__file__).parent.glob("*.py")
        },
        "gaussians": len(scene.means),
        "grid": list(geo["phi"].shape),
        "particles": replay_config["particles"] if replay_config else m.count,
        "replay_source_config": replay_config,
        "density_kg_m3": m.rho,
        "bulk_pa": m.bulk,
        "shear_pa": m.mu,
        "initial_source": (
            replay_config["initial_source"]
            if replay_config
            else "elevated continuous inlet"
            if emitter
            else "procedural finite ribbon"
        ),
        "inlet_volume_rate_m3_s": (
            replay_config.get("inlet_volume_rate_m3_s", 0.0)
            if replay_config
            else emitter.volume_rate
            if emitter
            else 0.0
        ),
        "capacity": capacity,
        "thermal_boundary_time_multiplier": args.exchange,
        "solver": "quadratic MLS/APIC thermoviscoplastic MPM",
    }
    config["dependency_sha256"] = {
        str(f): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [
            ROOT / "rainy_gs/scene.py",
            ROOT / "rainy_gs/demo.py",
            ROOT / "snowy_gs/mpm.py",
            ROOT / "snowy_gs/surface.py",
            Path(cfg["config"]),
            Path(cfg["path"]) / "cameras.json",
        ]
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    (args.output / "scene_config.json").write_text(json.dumps(scene.config, indent=2))
    (args.output / "source").mkdir()
    for f in Path(__file__).parent.glob("*.py"):
        shutil.copy2(f, args.output / "source" / f.name)
    np.savez_compressed(args.output / "geometry.npz", **geo)
    if args.cache_states:
        (args.output / "states").mkdir()
    if not args.replay:
        for _ in range(round(args.warmup * args.fps)):
            m.advance(1 / args.fps, emitter=emitter)
    videos = []
    logs = []
    camera_path = []
    try:
        for f in range(args.frames):
            if args.replay:
                state = dict(np.load(cache[f]))
                m.time = float(state["time"])
                m.count = len(state["x"])
                wp.to_torch(m.birth)[: m.count].copy_(
                    torch.as_tensor(
                        state.get("birth", np.zeros(m.count)), device=m.device
                    )
                )
                wp.to_torch(m.skin_age)[: m.count].copy_(
                    torch.as_tensor(
                        state.get(
                            "skin_age", m.time - state.get("birth", np.zeros(m.count))
                        ),
                        device=m.device,
                    )
                )
                for attr, key in [
                    ("x", "x"),
                    ("v", "v"),
                    ("fe", "fe"),
                    ("j", "j"),
                    ("temp", "temperature"),
                    ("reference", "reference"),
                ]:
                    wp.to_torch(getattr(m, attr))[: m.count].copy_(
                        torch.as_tensor(state[key], device=m.device)
                    )
            elif f > 0:
                m.advance(1 / args.fps, emitter=emitter)
            cam = transition_camera(
                scene,
                *cfg["cameras"],
                args.camera_end * f / max(args.frames - 1, 1),
                args.width,
                "smoothstep",
            )
            if args.camera_path == "cascade":
                cam = cascade_camera(
                    f / max(args.frames - 1, 1),
                    args.width,
                    m.device,
                    scene.config.get("lava_camera"),
                )
            if args.camera_pitch:
                c2w = torch.linalg.inv(cam.view)
                theta = np.deg2rad(args.camera_pitch)
                rotation = c2w.new_tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, np.cos(theta), np.sin(theta)],
                        [0.0, -np.sin(theta), np.cos(theta)],
                    ]
                )
                c2w[:3, :3] = c2w[:3, :3] @ rotation
                cam.view = torch.linalg.inv(c2w)
            src, alpha = scene.render(cam)
            rgb = renderer.render(src, cam)
            key = f in [0, args.frames // 2, args.frames - 1]
            if key or args.save_frames:
                save_image(args.output / f"lava_{f:05d}.png", rgb)
            if key:
                save_image(args.output / f"original_{f:05d}.png", src[..., :3])
            if args.cache_states:
                np.savez_compressed(
                    args.output / "states" / f"{f:05d}.npz", **m.state()
                )
            if not args.no_video:
                if not videos:
                    videos = [
                        Video(
                            args.output / "lava_demo.mp4",
                            cam.width,
                            cam.height,
                            args.fps,
                        ),
                        Video(
                            args.output / "comparison.mp4",
                            2 * cam.width,
                            cam.height,
                            args.fps,
                        ),
                    ]
                image = (rgb.cpu().numpy() * 255 + 0.5).astype("uint8")
                original = (src[..., :3].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(
                    "uint8"
                )
                videos[0].write(image)
                videos[1].write(np.concatenate((original, image), axis=1))
            if f % 15 == 0 or key:
                diag = m.diagnostics()
                state = m.state()
                sd = map_coordinates(
                    geo["phi"],
                    ((state["x"] - geo["origin"]) / m.dx).T,
                    order=1,
                    mode="nearest",
                )
                diag.update(
                    frame=f,
                    min_collider_distance_m=float(sd.min()),
                    visible_lava_fraction=float(
                        (renderer.last_glow.sum(-1) > 1.0e-4).float().mean()
                    ),
                    elapsed=time.monotonic() - start,
                )
                logs.append(diag)
                print(json.dumps(diag), flush=True)
                (args.output / "status.json").write_text(json.dumps(diag, indent=2))
            camera_path.append(
                dict(frame=f, view=cam.view.cpu().tolist(), K=cam.K.cpu().tolist())
            )
    finally:
        for v in videos:
            v.close()
    np.savez_compressed(args.output / "final_state.npz", **m.state())
    (args.output / "diagnostics.json").write_text(json.dumps(logs, indent=2))
    (args.output / "camera_path.json").write_text(json.dumps(camera_path))
    (args.output / "summary.json").write_text(
        json.dumps(
            dict(
                complete=True,
                frames=args.frames,
                elapsed=time.monotonic() - start,
                final=logs[-1],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
