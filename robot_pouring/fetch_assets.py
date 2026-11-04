"""Fetch hash-pinned official Panda assets and the showcase fonts."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from physstudio.paths import data_path


def fetch(record, destination):
    target = destination / record['file']
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError('Asset path escapes its destination')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Existing asset hash mismatch: ' + str(target))
        return
    temp = target.with_name(target.name + '.part')
    subprocess.run(['curl', '-fsSL', '--retry', '5', '--retry-all-errors',
                    '--connect-timeout', '15', record['url'], '-o', str(temp)], check=True)
    if hashlib.sha256(temp.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Downloaded asset hash mismatch: ' + record['file'])
    temp.replace(target)


def main():
    reference = Path(__file__).with_name('references')
    for manifest, folder in [('panda_assets.json', 'franka_emika_panda'), ('fonts.json', 'fonts')]:
        metadata = json.loads((reference / manifest).read_text())
        destination = data_path('robot_pouring', folder)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda record: fetch(record, destination), metadata['files']))
        (destination / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print('Verified', len(metadata['files']), 'assets in', destination, flush=True)


if __name__ == '__main__': main()
