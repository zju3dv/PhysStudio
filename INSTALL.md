# Installation

Tested with Linux, Python 3.10, PyTorch 2.4.0, CUDA 11.8, Warp 1.6.2,
gsplat 1.5.3, and an NVIDIA RTX 3090. A CUDA toolkit and a C++ compiler are
needed for gsplat's CUDA extension. Install `ffmpeg` and `ffprobe` on your system.

From this repository:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Set `CUDA_HOME` if the CUDA toolkit is outside the system search path.
The first render compiles the CUDA extension. The demo launcher stores build
caches in `.cache/` and outputs in `outputs/`.

```bash
python -m physstudio fetch bread
python -m physstudio run enlivening bread --gpu 0
```

The methods share Gaussian loading, camera, video, and download helpers;
install them together. Each method has its own simulation code and presets.
Run commands from the repository root. Choose a fresh `--output` for each run.

```bash
# Print the exact commands without running them.
python -m physstudio run snow playground --dry-run
# A short end-to-end run.
python -m physstudio run splashing plunge --frames 2 --width 320 \
  --output outputs/check_plunge --gpu 0
# Store downloaded assets outside the checkout.
export PHYSSTUDIO_DATA_ROOT=/path/to/assets
```

`--frames` controls saved simulation steps for splashing and magnetism, video
frames for rain, snow and lava, and the approximate half-speed video length for
locomotion (the initial state adds one frame).

See [VALIDATION.md](VALIDATION.md) for completed checks and validation commands.

Lava scene preparation also uses Node.js 22+ and the pinned PlayCanvas converter; see [LavaGS](lava_gs/README.md).
