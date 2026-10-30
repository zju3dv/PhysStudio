"""Download verified pretrained object PLYs released by PhysGaussian."""

from physstudio.paths import data_path
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

from rainy_gs.fetch import DriveZip

MODELS = {
    "bread": ("1e2_9OxfVtLSJTKl_y3TPaYQGjnvr3koT", 88582246, "tear_bread"),
    "wolf": ("1FhB_1LFaC2ax71gbySn6hVR88_IiK0eU", 209032720, "wolf"),
    "vasedeck": ("1AZh-JCNTwq9Za0rxFpDEBk-9tN6o_izs", 198747593, "vasedeck"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    fid, size, config = MODELS[args.model]
    out = data_path("gaussian_splashing") / args.model
    out.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
    remote = DriveZip(url, size)
    remote.cache.append((size - 65536, remote.fetch(size - 65536, size)))
    with zipfile.ZipFile(remote) as z:
        infos = z.infolist()
    inv = [
        dict(
            name=i.filename, bytes=i.file_size, compressed=i.compress_size, crc32=i.CRC
        )
        for i in infos
        if not i.is_dir()
    ]
    (out / "inventory.json").write_text(json.dumps(inv, indent=2))
    if args.list:
        print(json.dumps(inv, indent=2))
        return
    selected = [
        i
        for i in infos
        if not i.filename.startswith("__MACOSX/")
        and (
            i.filename.endswith(
                ("cameras.json", "cfg_args", "/iteration_30000/point_cloud.ply")
            )
        )
    ]
    manifest = dict(
        repository="https://github.com/XPandora/PhysGaussian",
        source=url,
        archive_bytes=size,
        files={},
    )
    for i in selected:
        name, result = remote.extract_member(i, out)
        manifest["files"][name] = dict(
            bytes=i.file_size,
            crc32=i.CRC,
            sha256=hashlib.sha256((out / name).read_bytes()).hexdigest(),
        )
        (out / "source.json").write_text(json.dumps(manifest, indent=2))
        print(name, result, flush=True)
    config_url = f"https://raw.githubusercontent.com/XPandora/PhysGaussian/main/config/{config}_config.json"
    (out / "source_config.json").write_bytes(
        urllib.request.urlopen(config_url, timeout=60).read()
    )


if __name__ == "__main__":
    main()
