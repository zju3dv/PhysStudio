"""Compare a dry-down trajectory against its constant-water control."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(dry, control):
    dry, control = Path(dry), Path(control)
    a = json.loads((dry / 'config.json').read_text())
    b = json.loads((control / 'config.json').read_text())
    for key in ['scene', 'frames', 'iterations', 'substeps', 'dt', 'warmup', 'retained']:
        assert a[key] == b[key], key
    assert not a['hydrated'] and b['hydrated']
    ra, rb = np.load(dry / 'rig.npz'), np.load(control / 'rig.npz')
    for key in ra.files:
        assert np.array_equal(ra[key], rb[key]), key
    ta = np.load(dry / 'trajectory.npz')
    tb = np.load(control / 'trajectory.npz')
    assert np.isfinite(ta['poses']).all() and np.isfinite(tb['poses']).all()
    assert np.allclose(tb['water'], tb['initial'][None])
    centers = (ra['head'] + ra['tail']) / 2
    dry_positions, healthy_positions = ta['poses'][..., :3], tb['poses'][..., :3]
    healthy_motion = np.linalg.norm(healthy_positions - healthy_positions[:1], axis=-1)
    excess = np.linalg.norm(dry_positions[-1] - healthy_positions[-1], axis=-1)
    return dict(scene=a['scene'], matched_settings=True, fixed_water_control=True,
        frames=a['frames'], healthy_max_settling_m=float(np.linalg.norm(healthy_positions[0] - centers, axis=-1).max()),
        healthy_max_motion_after_settling_m=float(healthy_motion.max()),
        dry_vs_healthy_final_max_displacement_m=float(excess.max()),
        dry_vs_healthy_final_rms_displacement_m=float(np.sqrt(np.mean(excess**2))),
        retained_water=float(ta['water'][-1].sum() / ta['initial'].sum()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry', required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = compare(args.dry, args.control)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
