"""Run dry-down and a fully hydrated control from the same plant rig."""
import argparse
import json
from pathlib import Path
import time
import shutil
import hashlib

import numpy as np
import warp as wp

from .assets import ROOT, SCENES
from .hydraulics import WaterModel
from .newton_solver import PlantSolver


def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    asset_dir = Path(args.asset_root) / args.scene
    rig = dict(np.load(asset_dir / 'rig.npz'))
    np.savez_compressed(out / 'rig.npz', **rig)
    for name in ['gaussians.npz', 'source.json']:
        shutil.copy2(asset_dir / name, out / name)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))
    source_dir = out / 'source'
    source_dir.mkdir()
    checksums = {}
    for name in ['assets.py', 'rig.py', 'hydraulics.py', 'newton_solver.py', 'simulate.py', 'render.py']:
        path = Path(__file__).parent / name
        shutil.copy2(path, source_dir / name)
        checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / 'source_hashes.json').write_text(json.dumps(checksums, indent=2))
    water = WaterModel(rig, hydraulic_radius_ratio=args.hydraulic_radius_ratio)
    duration = water.drydown_time(args.retained)
    times = np.linspace(0, duration, args.frames)
    hydration = water.evaluate(times)
    if args.hydrated:
        hydration[:] = water.initial
    solver = PlantSolver(rig, args.scene, water.initial, iterations=args.iterations, substeps=args.substeps, dt=args.dt)
    start = time.time()
    for step in range(args.warmup):
        solver.advance()
        if step % 30 == 0:
            print(f'{args.scene}: hydrated relaxation {step + 1}/{args.warmup}', flush=True)
    poses, gaps = [], []
    for frame, theta in enumerate(hydration):
        solver.set_water(theta, water.initial)
        solver.advance()
        pose = solver.poses()
        if not np.isfinite(pose).all():
            raise RuntimeError(f'Nonfinite state at {frame}')
        poses.append(pose)
        error = solver.joint_error(pose)
        displacement = np.linalg.norm(pose[:, :3] - solver.centers, axis=1).max()
        gaps.append(float(error.max()))
        if frame % 15 == 0 or frame == args.frames - 1:
            print(f'{args.scene}: {frame + 1}/{args.frames}, water={theta.sum()/water.initial.sum():.3f}, max joint gap={error.max():.6f} m, displacement={displacement:.3f} m, elapsed={time.time()-start:.1f}s', flush=True)
    np.savez_compressed(out / 'trajectory.npz', poses=np.array(poses), water=hydration, initial=water.initial, seconds=times,
                        mechanical_seconds=(np.arange(args.frames) + args.warmup + 3) * solver.substeps * solver.dt)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))
    report = dict(frames=args.frames, duration_seconds=duration, max_joint_gap_m=max(gaps),
                  last_joint_gap_m=gaps[-1], initial_water=float(water.initial.sum()),
                  final_water=float(hydration[-1].sum()), finite=True,
                  hydraulic_radius_ratio=args.hydraulic_radius_ratio,
                  max_relative_water_spread=float(np.ptp(hydration / water.initial, axis=1).max()),
                  backend='Newton official VBD / ROD joints',
                  wall_seconds=time.time()-start)
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(report, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scene', choices=SCENES, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--frames', type=int, default=240)
    p.add_argument('--iterations', type=int, default=160)
    p.add_argument('--substeps', type=int, default=12)
    p.add_argument('--dt', type=float, default=.001)
    p.add_argument('--warmup', type=int, default=120)
    p.add_argument('--retained', type=float, default=.16)
    p.add_argument('--hydrated', action='store_true')
    p.add_argument('--asset-root', default=str(ROOT))
    p.add_argument('--hydraulic-radius-ratio', type=float, default=.08)
    main(p.parse_args())
