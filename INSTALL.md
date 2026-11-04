# Installation

Tested with Linux, Python 3.10, PyTorch 2.4.0, CUDA 11.8, Warp 1.6.2,
gsplat 1.5.3, and an NVIDIA RTX 3090. A CUDA toolkit and a C++ compiler are
needed for gsplat's CUDA extension. Install `ffmpeg`, `ffprobe`, and the DejaVu
Sans fonts on your system.

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
frames for rain, snow, lava, storms, lightning, and water–sand, and the approximate half-speed video length for
locomotion (the initial state adds one frame).

For Plant Wilting previews, `--frames` sets both simulation and video frames.

See [VALIDATION.md](VALIDATION.md) for completed checks and validation commands.

Lava scene preparation also uses Node.js 22+ and the pinned PlayCanvas converter; see [LavaGS](lava_gs/README.md).

Water + Sand uses procedural scenes and the same installed dependencies:

```bash
python -m physstudio run water_sand castle --gpu 0
```

Plant Wilting installs its pinned Newton / Warp solver into a project-local directory:

```bash
python -m plant_wilting.runtime
python -m physstudio fetch potted_plant
python -m physstudio run wilting potted_plant --gpu 0
```

Its solver runs in a separate process; the shared rendering environment stays at
Warp 1.6.2. See [Plant Wilting](plant_wilting/README.md) for all three scenes.

StormGS uses the shared environment and the existing scene downloaders:

```bash
python -m physstudio fetch playground
python -m physstudio run storm playground --gpu 0
```

See [StormGS](storm_gs/README.md) for the coastal and desert scenes.

ThunderGS uses the same environment and scene downloaders:

```bash
python -m physstudio fetch firewoods
python -m physstudio run thunder firewoods --gpu 0
```

See [ThunderGS](thunder_gs/README.md) for Zohar, Train, cached discharges, and replay.

Capillary Contact uses NumPy/SciPy for simulation and Blender 4.2.9 for rendering:

```bash
python -m pip install -r surface_tension/requirements.txt
export BLENDER=/path/to/blender
python -m physstudio run capillary paperclips --gpu 0
```

See [Capillary Contact](surface_tension/README.md) for the orchard assets,
CPU rendering, and all three presets. `--frames` controls both saved states and video frames.

Ice Melting uses Warp 1.6.2 and trimesh for simulation and Blender 4.2.9 for rendering:

```bash
python -m pip install -r ice_melting/requirements.txt
export BLENDER=/path/to/blender
python -m physstudio run ice cubes --gpu 0
```

See [Ice Melting](ice_melting/README.md) for the arch, Bunny download, and replay.
`--frames` selects the number of saved states and video frames, with at least two;
short runs retain the normal time interval between frames.

Fire-X GS uses the shared Warp / gsplat environment and Matplotlib for its
diagnostic reports:

```bash
python -m firex_gs.fetch --scene all
python -m physstudio run firex firewoods --gpu 0
```

See [Fire-X GS](firex_gs/README.md) for Playground, Bonsai, frozen replay, and
matched no-water controls. Use `--frames 2 --width 320` for a short pipeline check.

Multi-fluid uses Warp 1.6.2 for simulation and Blender 4.2.9 for rendering:

```bash
python -m pip install -r implicit_mixture/requirements.txt
export BLENDER=/path/to/blender
python -m physstudio run mixture tea --gpu 0
```

See [Multi-fluid](implicit_mixture/README.md) for density inversion, colliding
liquids, and replay. All three scenes are procedural. `--frames` sets both saved
states and video frames, with at least two, at the preset 24 fps.

Ice Solidification uses Warp 1.6.2 for heat and liquid motion, with Blender 4.2.9
for rendering:

```bash
python -m pip install -r ice_solidification/requirements.txt
export BLENDER=/path/to/blender
python -m physstudio run freezing cascade --gpu 0
```

See [Ice Solidification](ice_solidification/README.md) for the cold-finger and
star-casting scenes. All geometry is procedural. Short `--frames` runs retain
the preset simulation interval; use at least two frames.

Tree & Root Growth uses Warp 1.6.2 for coupled root/shoot development and Blender 4.2.9
for rendering:

```bash
export BLENDER=/path/to/blender
python -m physstudio run rhizomorph oak_taproot --gpu 0
```

See [Tree & Root Growth](rhizomorph/README.md) for Maple, Birch, resource controls, and
replay. All geometry is procedural. `--frames` controls video sampling over the
full development trajectory; the method harness exposes `--steps` for shorter
simulations.

Windy Trees uses the shared Warp 1.6.2 environment, Blender 4.2.9 with
Cycles/OptiX, and FFmpeg:

```bash
export BLENDER=/path/to/blender
python -m physstudio run windy_trees oak_taproot --gpu 0
```

The pipeline generates a complete Rhizomorph tree before simulating wind. No
asset download or additional Python dependencies are needed. `--frames` shortens
the wind trajectory at the normal time step, with at least two frames; the tree
still completes 180 development steps. See [Windy Trees](windy_trees/README.md)
for the three scenes, numerical checks, and frozen-state replay.

Magnetic Rigid Bodies uses the shared Warp 1.6.2, NumPy/SciPy, and Pillow
packages, Blender 4.2.9 with Cycles/OptiX, and FFmpeg:

```bash
export BLENDER=/path/to/blender
python -m physstudio run magnetic_rigid compass_garden --gpu 0
```

All three scenes are procedural. `--frames` shortens the recording at the normal
30 fps and 120 Hz base physics rate, with at least two saved states. The full
body/sample resolution and camera/actuator timeline are retained. See
[Magnetic Rigid Bodies](magnetic_rigid/README.md) for the crane, globe, numerical
checks, and saved-state replay.

Thin-Film Bubbles uses the shared Warp, NumPy/SciPy, trimesh and Pillow packages,
Blender 4.2.9 with Cycles/OptiX, and FFmpeg:

```bash
export BLENDER=/path/to/blender
python -m physstudio run thin_film giant_recoil --gpu 0
```

The three presets are procedural. `--frames` records a shorter prefix at full
particle resolution and unchanged physics/camera timing, with at least two
saved states. The pipeline includes MLS surface reconstruction, mesh checks,
and 30 fps encoding. [Run guide and replay](melp_thin_film/README.md).

Robot Pouring uses an isolated MuJoCo Warp / Warp 1.10.1 runtime and the shared
Torch, NumPy, SciPy and Pillow packages:

```bash
python -m pip install -r robot_pouring/requirements.txt
python -m robot_pouring.runtime
python -m physstudio fetch robot_pouring
export BLENDER=/path/to/blender
python -m physstudio run robot_pouring basin --gpu 0
```

The installer writes to `.cache/robot_pouring/runtime/`; other methods retain
Warp 1.6.2. Use a fresh Python process when switching Warp runtimes. The batched
Torch and Gymnasium APIs, PPO commands, environment variables and staged replay
are in [Robot Pouring](robot_pouring/README.md). `--frames` saves a shorter prefix
at full 2,272-particle resolution and unchanged 25 Hz controls, including frame
zero. The three videos use the reference controller.
