"""Selectively fetch a released outdoor 3DGS model and its cameras."""

from physstudio.paths import data_path
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

from physstudio.remote_zip import RemoteZip

URL = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip"
TRAIN_COMMUNITY_URL = "https://huggingface.co/datasets/Voxel51/gaussian_splatting/resolve/main/FO_dataset/train/point_cloud/iteration_30000/point_cloud.ply"
TRAIN_COMMUNITY_SHA256 = (
    "a4c1906ce0256f5cb2255aa078d18b468dc01d8604f6eda1aaf22b235147a7a1"
)


class ModelZip(RemoteZip):
    def fetch(self, start, end):
        end = min(end, self.size)
        if end - start <= 16 * 1024 * 1024:
            return super().fetch(start, end)
        blocks = []
        for offset in range(start, end, 16 * 1024 * 1024):
            blocks.append(super().fetch(offset, min(offset + 16 * 1024 * 1024, end)))
            print(
                f"Downloaded {sum(map(len, blocks)) / 1e6:.1f}/{(end - start) / 1e6:.1f} MB",
                flush=True,
            )
        return b"".join(blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        choices=("train", "truck", "garden", "bicycle", "stump", "treehill", "flowers"),
        required=True,
    )
    parser.add_argument(
        "--provider", choices=("graphdeco", "voxel51"), default="graphdeco"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.provider == "voxel51" and args.scene != "train":
        parser.error("The Voxel51 adapter currently supports train")
    args.output = args.output or data_path("rainy_gs") / args.provider
    with urllib.request.urlopen(
        urllib.request.Request(URL, headers={"Range": "bytes=0-0"}), timeout=40
    ) as response:
        if response.status != 206:
            raise IOError("Model archive must support selective HTTP range downloads")
        size = int(response.headers["Content-Range"].split("/")[-1])
    remote = ModelZip(URL, size)
    remote.cache.append((size - 65536, remote.fetch(size - 65536, size)))
    with zipfile.ZipFile(remote) as archive:
        files = archive.infolist()
    names = (
        ("cameras.json", "cfg_args")
        if args.provider == "voxel51"
        else ("cameras.json", "cfg_args", "point_cloud/iteration_30000/point_cloud.ply")
    )
    selected = {f"{args.scene}/{name}" for name in names}
    manifest = dict(
        source="https://github.com/graphdeco-inria/gaussian-splatting",
        url=URL,
        zip_size=size,
        files={},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.scene}_archive_members.json").write_text(
        json.dumps(
            [
                dict(
                    name=i.filename,
                    bytes=i.file_size,
                    compressed=i.compress_size,
                    crc32=i.CRC,
                    header_offset=i.header_offset,
                )
                for i in files
                if i.filename in selected
            ],
            indent=2,
        )
    )
    for info in files:
        if info.filename not in selected:
            continue
        name, result = remote.extract_member(info, args.output)
        manifest["files"][name] = dict(
            bytes=info.file_size,
            crc32=info.CRC,
            sha256=hashlib.sha256((args.output / name).read_bytes()).hexdigest(),
        )
        (args.output / f"{args.scene}_download.json").write_text(
            json.dumps(manifest, indent=2)
        )
        print(json.dumps(dict(file=name, result=result)), flush=True)
    if set(manifest["files"]) != selected:
        raise ValueError("Incomplete scene in model archive")
    if args.provider == "voxel51":
        target = args.output / "train/point_cloud/iteration_30000/point_cloud.ply"
        target.parent.mkdir(parents=True, exist_ok=True)
        cached = (
            target.exists()
            and hashlib.sha256(target.read_bytes()).hexdigest()
            == TRAIN_COMMUNITY_SHA256
        )
        if not cached:
            temporary = target.with_suffix(".ply.partial")
            with (
                urllib.request.urlopen(TRAIN_COMMUNITY_URL, timeout=90) as response,
                temporary.open("wb") as stream,
            ):
                while True:
                    block = response.read(4 * 1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    print(f"Community model: {stream.tell() / 1e6:.1f} MB", flush=True)
            if (
                hashlib.sha256(temporary.read_bytes()).hexdigest()
                != TRAIN_COMMUNITY_SHA256
            ):
                raise ValueError(
                    "Community Train model does not match the validated SHA256"
                )
            temporary.replace(target)
        manifest["model_source"] = TRAIN_COMMUNITY_URL
        manifest["model_note"] = (
            "Voxel51 public model, distinct from the Graphdeco model; Graphdeco cameras"
        )
        manifest["files"][str(target.relative_to(args.output))] = dict(
            bytes=target.stat().st_size, sha256=TRAIN_COMMUNITY_SHA256
        )
        (args.output / f"{args.scene}_download.json").write_text(
            json.dumps(manifest, indent=2)
        )


if __name__ == "__main__":
    main()
