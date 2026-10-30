"""Run a reproducible magnetic liquid / pretrained Gaussian solid experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
from scipy.ndimage import map_coordinates
from gaussian_splashing.assets import Object
from .mpm import MPM
from .field import MS


def magnet_motion(t, mode, shift=(0.0, 0.0, 0.0)):
    if mode == "solid":
        center = np.array(
            [0.14 + 0.025 * np.sin(2.0 * t), -0.018, 0.14 + 0.06 * np.sin(1.6 * t)],
            dtype="f",
        )
    else:
        u = np.clip((t - 0.12) / 0.65, 0, 1)
        smooth = u * u * (3 - 2 * u)
        center = np.array(
            [
                0.035 + 0.05 * smooth,
                -0.025 + 0.035 * np.sin(smooth * np.pi),
                0.108 + 0.075 * smooth,
            ],
            dtype="f",
        )
    return center + np.asarray(shift, dtype="f")


def initialize(args, dx):
    obj = Object(
        args.asset, height=0.135 if args.mode == "fluid" else 0.17, device=args.device
    )
    center = np.array([-0.105, 0.055, -obj.lower[2] + 0.002], dtype="f")
    if args.mode == "solid":
        center = np.array([-0.025, 0.0, -obj.lower[2] + 0.012], dtype="f")
    sdf, sdf_origin, sdf_dx, filled = obj.collision(dx * 0.5)
    sdf_origin += center
    if args.mode == "fluid":
        xx = np.arange(-0.10, 0.17, dx / 2)
        yy = np.arange(-0.15, 0.12, dx / 2)
        zz = np.arange(dx * 1.25, 0.054, dx / 2)
        positions = np.stack(np.meshgrid(xx, yy, zz, indexing="ij"), -1).reshape(-1, 3)
        keep = ((positions[:, :2] - [0.025, -0.015]) ** 2).sum(-1) < 0.106**2
        sd = map_coordinates(
            sdf,
            ((positions - sdf_origin) / sdf_dx).T,
            order=1,
            mode="constant",
            cval=1.0,
        )
        positions = positions[keep & (sd > dx * 0.7)]
    else:
        positions = filled + center
        # Uniform cell-center volume samples of the closed Gaussian occupancy.
    metadata = dict(
        name=args.asset,
        path=str(obj.path.resolve()),
        sha256=hashlib.sha256(obj.path.read_bytes()).hexdigest(),
        source="https://github.com/XPandora/PhysGaussian",
        height=0.135 if args.mode == "fluid" else 0.17,
        center=center.tolist(),
        gaussians=obj.count,
    )
    return positions.astype("f"), (sdf, sdf_origin, sdf_dx), metadata


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a fresh output directory: " + str(output))
    output.mkdir(parents=True)
    dx = 0.48 / args.n
    positions, collider, asset = initialize(args, dx)
    sim = MPM(
        positions,
        n=args.n,
        dt=args.dt,
        device=args.device,
        material=args.mode,
        bulk=args.bulk,
        viscosity=args.viscosity,
        shear=args.shear,
        ms=MS * args.susceptibility,
        clamp_height=args.clamp_height,
    )
    if args.mode == "fluid":
        sim.set_collider(*collider)
    config = dict(
        vars(args),
        asset=asset,
        dx=dx,
        particles=len(positions),
        saturation_A_m=MS * args.susceptibility,
        physical_seconds_per_output=args.dt * args.substeps,
        magnetic_solver="Langevin + Newton-PCG",
        magnetic_force="div(Einstein-Laub Maxwell stress), cell-integrated",
        mechanical_solver="Warp MLS-MPM",
        field_boundary="homogeneous Neumann induced potential",
        rendering="pretrained Gaussian object plus simulated fluid splats"
        if args.mode == "fluid"
        else "pretrained object Gaussians bound to MPM particles",
    )
    (output / "config.json").write_text(json.dumps(config, indent=2))
    initial = sim.state()
    states = [initial]
    magnets = [magnet_motion(0, args.mode, args.magnet_shift)]
    times = [0.0]
    stats = [sim.diagnostics(initial)]
    field_stats = []
    requested_stop = [False]

    def stop(*_):
        requested_stop[0] = True

    signal.signal(signal.SIGTERM, stop)
    start = time.perf_counter()
    error = None
    try:
        with (output / "progress.jsonl").open("w", buffering=1) as log:
            for frame in range(args.frames):
                for sub in range(args.substeps):
                    center = magnet_motion(sim.time, args.mode, args.magnet_shift)
                    next_center = magnet_motion(
                        sim.time + args.dt, args.mode, args.magnet_shift
                    )
                    velocity = (next_center - center) / args.dt
                    strength = args.intensity * min(1.0, sim.time / 0.08)
                    update = sim.steps % args.field_every == 0
                    sim.step(
                        center, args.radius, strength, velocity, update_field=update
                    )
                    if update:
                        field_stats.append(dict(time=sim.time, **sim.field.last_stats))
                state = sim.state()
                diag = sim.diagnostics(state)
                if diag["speed_max"] * args.dt / dx > 0.45:
                    raise FloatingPointError("MPM CFL exceeded: " + str(diag))
                states.append(state)
                magnets.append(magnet_motion(sim.time, args.mode, args.magnet_shift))
                times.append(sim.time)
                stats.append(diag)
                log.write(json.dumps(dict(frame=frame + 1, **diag)) + "\n")
                if (frame + 1) % 5 == 0:
                    print(
                        json.dumps(
                            dict(
                                frame=frame + 1,
                                wall_seconds=time.perf_counter() - start,
                                **diag,
                            )
                        ),
                        flush=True,
                    )
                if requested_stop[0]:
                    break
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        np.savez_compressed(
            output / "trajectory.npz",
            positions=np.stack([s["positions"] for s in states]),
            velocities=np.stack([s["velocities"] for s in states]),
            deformation=np.stack([s["deformation"] for s in states]),
            magnet_center=np.stack(magnets),
            time=np.asarray(times),
            rest=positions,
        )
        (output / "field_solves.json").write_text(json.dumps(field_stats, indent=2))
        report = dict(
            complete=len(states) == args.frames + 1,
            error=error,
            frames=len(states) - 1,
            seconds=time.perf_counter() - start,
            initial=stats[0],
            final=stats[-1],
            max_speed=max(s["speed_max"] for s in stats),
            min_j=min(s["min_j"] for s in stats),
            maximum_field_relative_residual=max(
                (s["relative_residual"] for s in field_stats), default=0
            ),
            all_field_solves_converged=all(s["converged"] for s in field_stats),
        )
        (output / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["fluid", "solid"], default="fluid")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--substeps", type=int, default=24)
    p.add_argument("--dt", type=float, default=0.00025)
    p.add_argument("--field-every", type=int, default=8)
    p.add_argument("--bulk", type=float, default=2.0e5)
    p.add_argument("--viscosity", type=float, default=12.0)
    p.add_argument("--shear", type=float, default=2.0e4)
    p.add_argument("--susceptibility", type=float, default=1.0)
    p.add_argument("--radius", type=float, default=0.032)
    p.add_argument("--intensity", type=float, default=1.0e6)
    p.add_argument("--asset", choices=["bread", "wolf"], default="bread")
    p.add_argument("--clamp-height", type=float, default=-1.0)
    p.add_argument("--magnet-shift", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    run(p.parse_args())
