"""Download an openly licensed SuperSplat scene LOD from its public viewer."""

import argparse
import concurrent.futures
import hashlib
import json
import re
import urllib.request
from pathlib import Path


def fetch(url):
    for attempt in range(3):
        try:
            return urllib.request.urlopen(url, timeout=45).read()
        except Exception:
            if attempt == 2:
                raise


def download(scene_hash, lod, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    html = fetch("https://superspl.at/s?id=" + scene_hash).decode()
    bootstrap = json.loads(
        re.search(
            r'<script type="application/json" id="sse-bootstrap">(.*?)</script>', html
        ).group(1)
    )
    base = bootstrap["contentUrl"].rsplit("/", 1)[0] + "/"
    manifest = json.loads(fetch(bootstrap["contentUrl"]))
    (out / "viewer.json").write_text(json.dumps(bootstrap, indent=2))
    (out / "lod-meta.json").write_text(json.dumps(manifest))
    selected = [f for f in manifest["filenames"] if f.startswith(str(lod) + "_")]
    if not selected:
        raise ValueError("Requested LOD has no blocks in the source manifest")
    # The converter obtains the common attribute layout from the first LOD 0 block.
    initializer = next(f for f in manifest["filenames"] if f.startswith("0_0/"))
    auxiliary = [initializer] if initializer not in selected else []
    jobs = []
    for f in selected + auxiliary:
        meta = json.loads(fetch(base + f))
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        (out / f).write_text(json.dumps(meta))
        for item in meta.values():
            if isinstance(item, dict):
                for rel in item.get("files", []):
                    jobs.append(str(Path(f).parent / rel))

    def get(rel):
        path = out / rel
        if not path.exists():
            path.write_bytes(fetch(base + rel))
        return dict(
            path=rel,
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    files = list(concurrent.futures.ThreadPoolExecutor(6).map(get, jobs))
    info = dict(
        hash=scene_hash,
        url="https://superspl.at/scene/" + scene_hash,
        lod=lod,
        parts=selected,
        auxiliary_parts=auxiliary,
        files=files,
    )
    (out / "download.json").write_text(json.dumps(info, indent=2))
    print(
        json.dumps(
            dict(
                parts=selected, files=len(files), bytes=sum(f["bytes"] for f in files)
            ),
            indent=2,
        )
    )

    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hash", required=True)
    p.add_argument("--lod", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    download(args.hash, args.lod, args.output)


if __name__ == "__main__":
    main()
