"""Fetch selected, CRC-verified FieryGS assets from the official Drive ZIP."""

from physstudio.paths import data_path
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

from physstudio.remote_zip import RemoteZip

SCENES = {
    "playground": ("1NFXzr0YU3q1_ezjWVEgTmd-R8JX4J0lL", 1049696735),
    "firewoods": ("1EMWrcSLbeLjTWyGUh_cThYGvrrmMrvC0", 1313590023),
}


class DriveZip(RemoteZip):
    def fetch(self, start, end):
        end = min(end, self.size)
        if start == end:
            return b""
        for a, b in self.cache:
            if a <= start and end <= a + len(b):
                return b[start - a : end - a]
        if end - start > 8 * 1024 * 1024:
            blocks = []
            for offset in range(start, end, 8 * 1024 * 1024):
                blocks.append(self.fetch(offset, min(end, offset + 8 * 1024 * 1024)))
                print(
                    f"Downloaded {sum(map(len, blocks)) / 1e6:.1f}/{(end - start) / 1e6:.1f} MB",
                    flush=True,
                )
            return b"".join(blocks)
        for attempt in range(4):
            try:
                request = urllib.request.Request(
                    self.url + f"&range={start}-{end - 1}",
                    headers={"Range": f"bytes={start}-{end - 1}"},
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{end - 1}/{self.size}"
                    ):
                        raise IOError("Incorrect remote ZIP range")
                    data = response.read()
                if len(data) != end - start:
                    raise IOError("Incomplete ZIP range")
                return data
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=SCENES, default="playground")
    p.add_argument("--output", type=Path, default=data_path("rainy_gs/fierygs"))
    p.add_argument("--list", action="store_true")
    p.add_argument("--member", action="append", default=[])
    args = p.parse_args()
    file_id, size = SCENES[args.scene]
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    remote = DriveZip(url, size)
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = [
        dict(name=i.filename, size=i.file_size, compressed=i.compress_size, crc=i.CRC)
        for i in infos
        if not i.is_dir()
    ]
    (args.output / f"{args.scene}_inventory.json").write_text(
        json.dumps(inventory, indent=2)
    )
    if args.list:
        print(json.dumps(inventory, indent=2))
        return
    selected = [
        i
        for i in infos
        if not i.is_dir()
        and not i.filename.startswith("__MACOSX/")
        and (
            i.filename in args.member
            if args.member
            else (
                i.filename.endswith(
                    ("cameras.json", "cfg_args", "voxel_grid.npy", "material_grid.npy")
                )
                or i.filename.endswith(
                    (
                        "/iteration_30000/point_cloud.ply",
                        "/iteration_30000/scene_point_cloud.ply",
                    )
                )
            )
        )
    ]
    if not selected:
        raise ValueError("No selected files; inspect --list and select --member paths")
    manifest_path = args.output / f"{args.scene}_download.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else dict(
            source="https://github.com/PKU-VCL-Geometry/FieryGS#data",
            url=url,
            zip_size=size,
            files={},
        )
    )
    for info in selected:
        name, result = remote.extract_member(info, args.output)
        path = args.output / name
        manifest["files"][name] = dict(
            bytes=info.file_size,
            crc32=info.CRC,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(json.dumps(dict(file=name, result=result)), flush=True)


if __name__ == "__main__":
    main()
