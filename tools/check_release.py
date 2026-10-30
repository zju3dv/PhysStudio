"""Check source portability, documentation links, and packaged video integrity."""

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".cache", ".venv", "data", "outputs", "__pycache__", "build", "dist"}


def main():
    checked = 0
    for p in ROOT.rglob("*"):
        if not p.is_file() or SKIP.intersection(p.relative_to(ROOT).parts):
            continue
        if p.suffix not in {".py", ".md", ".json", ".toml", ".txt", ".svg", ".html"}:
            continue
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"[\u4e00-\u9fff]", text), str(p)
        if p.suffix == ".py":
            ast.parse(text, filename=str(p))
            if p.name != "check_release.py":
                assert "/mnt/" not in text and "experiments." not in text, str(p)
        if p.suffix in {".md", ".html"}:
            links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
            links += re.findall(r'(?:href|src)="([^"{}]+)"', text)
            for link in links:
                if link.startswith(("http:", "https:", "mailto:", "#")):
                    continue
                target = unquote(link.split("#", 1)[0])
                assert (p.parent / target).exists(), (str(p), target)
        checked += 1
    videos = json.loads((ROOT / "media/manifest.json").read_text())
    assert len(videos) == 15
    counts = {}
    for entry in videos:
        p = ROOT / entry["video"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == entry["sha256"], str(p)
        info = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=width,height,nb_frames",
                    "-of",
                    "json",
                    str(p),
                ],
                stderr=subprocess.PIPE,
            )
        )["streams"][0]
        assert int(info["nb_frames"]) == int(entry["nb_frames"]), str(p)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-threads", "2", "-i", str(p), "-f", "null", "-"],
            check=True,
            capture_output=True,
        )
        assert p.with_suffix(".gif").stat().st_size > 0
        assert p.with_suffix(".jpg").stat().st_size > 0
        counts[entry["method"]] = counts.get(entry["method"], 0) + 1
    assert len(counts) == 5 and all(n == 3 for n in counts.values())
    print(
        json.dumps(
            dict(passed=True, text_files=checked, videos=len(videos), methods=counts),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
