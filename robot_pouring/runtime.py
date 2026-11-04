"""Install pinned robot dependencies without changing the shared environment."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from . import RUNTIME, CACHE_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    requirements = Path(__file__).with_name('requirements-solver.txt')
    identity = {'requirements': requirements.read_text()}
    marker = RUNTIME / 'runtime.json'
    command = [sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(RUNTIME),
               '--cache-dir', str(CACHE_ROOT / 'pip'), '-r', str(requirements)]
    print(shlex.join(command), flush=True)
    if args.dry_run: return
    if not marker.exists():
        if RUNTIME.exists():
            raise FileExistsError('Choose a fresh PHYSSTUDIO_ROBOT_RUNTIME directory: ' + str(RUNTIME))
        subprocess.run(command, check=True)
    elif json.loads(marker.read_text()) != identity:
        raise ValueError('Runtime requirements changed; choose a fresh PHYSSTUDIO_ROBOT_RUNTIME')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(RUNTIME) + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run([sys.executable, '-c',
                    'import mujoco, mujoco_warp, warp, gymnasium; '
                    'assert mujoco.__version__ == "3.3.7"; '
                    'assert warp.__version__ == "1.10.1"; '
                    'assert gymnasium.__version__ == "1.2.3"; '
                    'print("Robot runtime imports passed")'], env=env, check=True)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    print('Robot runtime ready:', RUNTIME, flush=True)


if __name__ == '__main__': main()
