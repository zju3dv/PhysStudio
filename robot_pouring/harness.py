"""Record, audit, reconstruct, render and encode robot pouring demos."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from physstudio.paths import ROOT
from . import CACHE_ROOT

SCENES = ('basin', 'precision', 'flower')


def execute(command, env, log):
    command = list(map(str, command))
    print(shlex.join(command), flush=True)
    with log.open('a') as stream:
        stream.write(shlex.join(command) + '\n'); stream.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', choices=SCENES, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['all', 'simulate', 'validate', 'render', 'encode'], default='all')
    p.add_argument('--simulation', type=Path, help='Reuse a recorded episode for validate, render or encode')
    p.add_argument('--gpu', default='0')
    p.add_argument('--frames', type=int, help='Saved states including frame zero; full particle resolution and normal time steps')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--samples', type=int, default=32)
    p.add_argument('--blender', default=os.environ.get('BLENDER', 'blender'))
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, help='Exclusive final render frame')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--font-dir', type=Path)
    a = p.parse_args()
    if a.frames is not None and a.frames < 2: p.error('--frames must be at least two')
    if a.width < 32 or a.width % 2 or a.samples < 1: p.error('Use even width >= 32 and positive samples')
    if a.simulation and a.stage in ['all', 'simulate']: p.error('--simulation is for validate/render/encode')
    if a.frames is not None and a.stage not in ['all', 'simulate']: p.error('--frames applies to simulation')
    if (a.resume or a.start or a.end is not None) and a.stage != 'render': p.error('Frame ranges and --resume apply to render')
    blender = shutil.which(os.path.expanduser(a.blender))
    if a.stage in ['all', 'render'] and blender is None: p.error('Set BLENDER to a Blender 4.2 executable')
    out = a.output.expanduser().resolve()
    if a.stage == 'all' and out.exists(): p.error('Choose a fresh output directory')
    sim = a.simulation.expanduser().resolve() if a.simulation else out / 'simulation'
    mesh, frames = out / 'mesh', out / 'render'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=a.gpu, PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    for key, folder in [('XDG_CACHE_HOME', 'blender'), ('CUDA_CACHE_PATH', 'cuda'), ('OPTIX_CACHE_PATH', 'optix')]:
        env.setdefault(key, str(CACHE_ROOT / folder))
    out.mkdir(parents=True, exist_ok=True)
    stages = ['simulate', 'validate', 'render', 'encode'] if a.stage == 'all' else [a.stage]
    for stage in stages:
        if stage == 'simulate':
            cmd = [sys.executable, '-m', 'robot_pouring.demo', '--task', a.scene, '--output', sim]
            if a.frames is not None: cmd += ['--frames', a.frames]
            execute(cmd, env, out / 'simulate.log')
        config = json.loads((sim / 'config.json').read_text())
        report = json.loads((sim / 'report.json').read_text())
        if config['task'] != a.scene: raise ValueError('Recording does not match --scene')
        if len(list((sim / 'states').glob('*.npz'))) != report['frames']: raise ValueError('Incomplete recording')
        if stage == 'validate':
            cmd = [sys.executable, '-m', 'robot_pouring.audit', sim]
            if report.get('prefix_only', False): cmd += ['--allow-incomplete']
            execute(cmd, env, out / 'validate.log')
        elif stage == 'render':
            end = report['frames'] if a.end is None else a.end
            if not 0 <= a.start < end <= report['frames']: raise ValueError('Invalid render range')
            settings = dict(simulation=str(sim), config_sha256=hashlib.sha256((sim/'config.json').read_bytes()).hexdigest(),
                            report_sha256=hashlib.sha256((sim/'report.json').read_bytes()).hexdigest(),
                            surface_sha256=hashlib.sha256(Path(__file__).with_name('surface.py').read_bytes()).hexdigest())
            marker = out / 'mesh_settings.json'
            if mesh.exists():
                if not a.resume or not marker.exists() or json.loads(marker.read_text()) != settings:
                    raise ValueError('Use a fresh output or --resume with matching mesh settings')
                if not json.loads((mesh/'audit.json').read_text())['passed']: raise ValueError('Mesh audit failed')
                if len(list(mesh.glob('[0-9]*.npz'))) != report['frames']: raise ValueError('Incomplete mesh reconstruction; use fresh output')
            else:
                execute([sys.executable, '-m', 'robot_pouring.surface', '--input', sim, '--output', mesh], env, out/'surface.log')
                marker.write_text(json.dumps(settings, indent=2)+'\n')
            cmd = [blender, '-b', '--factory-startup', '-t', '6', '--python-exit-code', '1', '--python',
                   ROOT/'robot_pouring/render_blender.py', '--', '--input', mesh, '--output', frames,
                   '--width', a.width, '--height', round(a.width*9/32)*2, '--samples', a.samples,
                   '--indices', ','.join(map(str, range(a.start, end)))]
            if a.resume: cmd += ['--resume']
            execute(cmd, env, out/'render.log')
        elif stage == 'encode':
            cmd = [sys.executable, '-m', 'robot_pouring.finish_media', '--inputs', sim,
                   '--frames-dir', frames, '--output', out/'media']
            if a.font_dir: cmd += ['--font-dir', a.font_dir.expanduser().resolve()]
            execute(cmd, env, out/'encode.log')
    print(json.dumps(dict(output=str(out), stage=a.stage, frames=report['frames']), indent=2))


if __name__ == '__main__': main()
