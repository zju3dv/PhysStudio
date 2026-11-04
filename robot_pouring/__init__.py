"""Batched robot/liquid tasks with an isolated MuJoCo Warp runtime."""
import os
from pathlib import Path
import sys
from physstudio.paths import ROOT

CACHE_ROOT = Path(os.environ.get('PHYSSTUDIO_ROBOT_CACHE', ROOT / '.cache/robot_pouring')).expanduser().resolve()
RUNTIME = Path(os.environ.get('PHYSSTUDIO_ROBOT_RUNTIME', CACHE_ROOT / 'runtime')).expanduser().resolve()
if RUNTIME.is_dir():
    sys.path.insert(0, str(RUNTIME))
os.environ.setdefault('MUJOCO_GL', 'egl')


def require_runtime():
    import warp
    if warp.__version__ != '1.10.1':
        raise RuntimeError('Robot Pouring requires Warp 1.10.1. Run python -m robot_pouring.runtime, '
                           'then start a fresh Python process before importing robot_pouring.env.')
