"""Portable code, asset, and cache locations."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_path(*parts):
    return (
        Path(os.environ.get("PHYSSTUDIO_DATA_ROOT", str(ROOT / "data")))
        .expanduser()
        .resolve()
        .joinpath(*parts)
    )
