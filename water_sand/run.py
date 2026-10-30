"""Run and cache a complete 3D water-sand mixture trajectory."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import warp as wp
from .scenes import initial
from .solver import Mixture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scene", choices=["breach", "castle", "sediment"], default="breach"
    )
    p.add_argument("--mode", choices=["fluid", "sand", "coupled"], default="coupled")
    p.add_argument("--resolution", type=int, default=96)
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--steps-per-frame", type=int, default=32)
    p.add_argument("--frame-dt", type=float, help="Physical seconds per output frame")
    p.add_argument("--ppc", type=int, default=2)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--settle", type=int, default=100)
    p.add_argument("--settle-seconds", type=float, help="Physical settling duration")
    p.add_argument(
        "--absorption-per-second",
        type=float,
        help="Water uptake rate in inverse seconds",
    )
    p.add_argument(
        "--drag-reference-dx",
        type=float,
        help="Physical drag representative length in meters",
    )
    p.add_argument("--young", type=float, help="Override Young modulus in pascals")
    p.add_argument(
        "--cohesion", type=float, help="Override the dimensionless cohesion scale"
    )
    p.add_argument("--no-retention", action="store_true")
    p.add_argument("--reference-phase", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    states = out / "states"
    states.mkdir()
    cfg, wall, phi, u, x = initial(args.scene, args.resolution, args.ppc, args.mode)
    if args.young is not None:
        cfg["young"] = args.young
    if args.cohesion is not None:
        cfg["cohesion"] = args.cohesion
    cfg.update(
        frames=args.frames,
        steps_per_frame=args.steps_per_frame,
        warmup=args.warmup,
        settle=args.settle,
        retention=not args.no_retention,
        conservative=not args.reference_phase,
    )
    if args.frame_dt is not None:
        cfg["frame_dt"] = args.frame_dt
    if args.settle_seconds is not None:
        cfg["settle_seconds"] = args.settle_seconds
    if args.absorption_per_second is not None:
        cfg["absorption_per_second"] = args.absorption_per_second
    if args.drag_reference_dx is not None:
        cfg["drag_reference_dx"] = args.drag_reference_dx
    cfg["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    wp.init()
    wp.set_device("cuda:0")
    start = time.time()
    sim = Mixture(cfg, wall, phi, u, x)
    cfg.update(dt=sim.dt, velocity_scale=sim.velocity_scale, particles=len(x))
    targets = [
        round((frame + 1) * args.frame_dt / sim.dt)
        if args.frame_dt is not None
        else (frame + 1) * args.steps_per_frame
        for frame in range(args.frames)
    ]
    counts = np.diff([0] + targets).tolist()
    if not counts or min(counts) < 1:
        raise ValueError("Output interval must include at least one simulation step")
    cfg.update(
        timing_mode="physical_interval"
        if args.frame_dt is not None
        else "fixed_substeps",
        output_step_counts=counts,
    )
    if args.frame_dt is not None:
        cfg["steps_per_frame"] = None
    np.savez_compressed(
        out / "initial.npz", positions=sim.s.x.numpy(), wall=wall.reshape(cfg["shape"])
    )
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    audit = []
    print("initialized", json.dumps(sim.initial), flush=True)
    for frame in range(args.frames):
        count = counts[frame]
        sim.step(count)
        d = sim.diagnostics()
        d.update(frame=frame, wall_seconds=time.time() - start)
        audit.append(d)
        (out / "audit.json").write_text(json.dumps(audit, indent=2))
        print(json.dumps(d), flush=True)
        if (
            not d["finite"]
            or not d["particles_inside"]
            or d["fluid_speed_max"] > 0.5
            or d["sand_speed_max"] > 0.5
        ):
            raise FloatingPointError(
                "Trajectory failed physical state audit; inspect audit.json"
            )
        if not args.audit_only:
            np.savez_compressed(states / f"{frame:05d}.npz", **sim.snapshot())
    (out / "complete.json").write_text(
        json.dumps(
            dict(frames=args.frames, seconds=time.time() - start, final=audit[-1]),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
