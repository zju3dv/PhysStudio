"""Keep Newton's solver dependencies separate from the shared rendering runtime."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from physstudio.paths import ROOT

NEWTON_REVISION = 'e6068cfc1143fe6a62e43ac6cc24361e36618fc0'
WARP_VERSION = '1.17.0'


def runtime_path():
    return Path(os.environ.get('PHYSSTUDIO_PLANT_RUNTIME', ROOT / '.cache/plant_wilting_runtime')).expanduser().resolve()


def solver_environment():
    target = runtime_path()
    if not all((target / name / '__init__.py').exists() for name in ['newton', 'warp']):
        raise RuntimeError('Install the plant solver with: python -m plant_wilting.runtime')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(target), str(ROOT), env.get('PYTHONPATH', '')])
    env['WARP_CACHE_PATH'] = str(ROOT / '.cache/warp_plant')
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('OPENBLAS_NUM_THREADS', '4')
    return env


def main(args):
    target = runtime_path()
    identity = dict(newton_revision=NEWTON_REVISION, warp_version=WARP_VERSION)
    marker = target / 'runtime.json'
    if marker.exists() and json.loads(marker.read_text()) == identity:
        solver_environment()
        print(f'Plant solver ready: {target}')
        return
    command = [sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(target),
               '--cache-dir', str(ROOT / '.cache/pip'), '-r',
               str(Path(__file__).with_name('requirements-solver.txt'))]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if target.exists():
        raise FileExistsError(f'Choose a fresh PHYSSTUDIO_PLANT_RUNTIME directory: {target}')
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, '-c',
                    'import newton, warp; assert warp.__version__ == "1.17.0"; '
                    'from newton.solvers import SolverVBD; print("Newton / Warp runtime ready")'],
                   env=solver_environment(), check=True)
    marker.write_text(json.dumps(identity, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
