"""Install the calibrated cameras and attribution for a released lava scene."""

import argparse
from pathlib import Path
import shutil
from physstudio.paths import data_path


def prepare(scene):
    root = data_path("lava_gs", scene)
    root.mkdir(parents=True, exist_ok=True)
    configs = Path(__file__).parent / "configs"
    for suffix, destination in [
        ("cameras", "cameras.json"),
        ("attribution", "ATTRIBUTION.json"),
    ]:
        shutil.copy2(configs / f"{scene}_{suffix}.json", root / destination)
    print("Prepared", scene, "at", root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", choices=["zohar", "lands_end", "calico"], required=True
    )
    prepare(parser.parse_args().scene)


if __name__ == "__main__":
    main()
