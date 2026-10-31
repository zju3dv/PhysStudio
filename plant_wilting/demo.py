"""Prepare, simulate, and render the three Gaussian plant wilting presets."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from physstudio.paths import ROOT
from .runtime import solver_environment

PRESETS = {
    'potted_plant': dict(frames=180, retained=.32),
    'ficus': dict(frames=180, retained=.32),
    'alocasia': dict(frames=120, retained=.16),
}


def main(args):
    solver_env = solver_environment()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    logs = output / 'logs'
    logs.mkdir()
    commands = []

    def run(module, options, solver=False):
        command = [sys.executable, '-u', '-m', 'plant_wilting.' + module] + list(map(str, options))
        commands.append(command)
        print(shlex.join(command), flush=True)
        log = logs / f'{len(commands):02d}_{module}.log'
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, env=solver_env if solver else os.environ.copy(),
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{module} failed; see {log}')

    settings = PRESETS[args.scene]
    prepared = output / 'assets'
    simulation = output / 'simulation'
    run('rig', ['--scene', args.scene, '--output-root', prepared])
    common = ['--scene', args.scene, '--asset-root', prepared,
              '--frames', settings['frames'] if args.frames is None else args.frames,
              '--retained', settings['retained'], '--warmup', args.warmup,
              '--iterations', 160, '--substeps', 12, '--dt', .001,
              '--hydraulic-radius-ratio', args.hydraulic_radius_ratio]
    run('simulate', common + ['--output', simulation], solver=True)
    if args.controls:
        run('simulate', common + ['--hydrated', '--output', output / 'control'], solver=True)
        run('compare_control', ['--dry', simulation, '--control', output / 'control',
                                '--output', output / 'control_comparison.json'])
    run('render_water', ['--input', simulation, '--output', output / 'render',
                         '--frames', args.frames or 240, '--width', args.width or 1920,
                         '--orbit', 70, '--covariance', 'rod', '--binding', 'material'])
    run('validate', ['--gpu-binding', '--trajectories', simulation,
                     '--output', output / 'validation.json'])
    (output / 'commands.json').write_text(json.dumps(commands, indent=2))
    print(f'Video: {output / "render/water_demo.mp4"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=PRESETS, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--warmup', type=int, default=120)
    parser.add_argument('--controls', action='store_true')
    parser.add_argument('--hydraulic-radius-ratio', type=float, default=.08)
    args = parser.parse_args()
    if args.frames is not None and args.frames < 1:
        parser.error('--frames must be positive')
    if args.width is not None and (args.width < 32 or args.width % 2):
        parser.error('--width must be even and at least 32')
    main(args)
