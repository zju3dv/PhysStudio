"""Download the pinned DiffWind plant Gaussians and source segmentation settings."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from physstudio.paths import data_path

SCENES = ('potted_plant', 'ficus', 'alocasia')
REVISION = 'a47530ec2fafd6f647c87a8f8e086b1905dbfe52'
BASE = f'https://raw.githubusercontent.com/zju3dv/DiffWind/{REVISION}'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fetch(scene):
    expected = json.loads((Path(__file__).parent / 'references/assets.json').read_text())[scene]
    output = data_path('plant_wilting', 'source', scene)
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, relative in [('model.ply', f'assets/{scene}/model.ply'),
                           ('config.json', f'configs/{scene}.json')]:
        target = output / name
        url = f'{BASE}/{relative}'
        sha = expected[name]['sha256']
        if target.exists():
            if digest(target) != sha:
                raise ValueError(f'Checksum mismatch: {target}')
        else:
            temporary = target.with_suffix(target.suffix + '.part')
            print(f'Downloading {scene}/{name}', flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'PhysStudio'})
            try:
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        stream.write(block)
                if digest(temporary) != sha:
                    raise ValueError(f'Checksum mismatch for {url}')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        records[name] = dict(url=url, sha256=sha, bytes=target.stat().st_size)
    (output / 'download.json').write_text(json.dumps(dict(scene=scene, revision=REVISION, files=records), indent=2))
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=SCENES, nargs='+', default=list(SCENES))
    for scene in parser.parse_args().scene:
        fetch(scene)
