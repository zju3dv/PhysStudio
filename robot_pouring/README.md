# Robot Pouring

GPU liquid-manipulation environments for a Franka Panda, using MuJoCo Warp for
articulated dynamics and an independent Warp position-based fluid solver.
Three tasks share the same batched Torch and Gymnasium interfaces: **Basin
Transfer**, **Precision Pour**, and **Gentle Watering**.

[Video gallery](../media/index.html#robot-pouring) · [Model and solver](METHOD.md) ·
[Validation](VALIDATION.md)

## Install and run

Start with the [PhysStudio environment](../INSTALL.md), then install the robot
runtime into its own cache directory. This keeps the other methods on Warp 1.6.2.

```bash
python -m pip install -r robot_pouring/requirements.txt
python -m robot_pouring.runtime
python -m physstudio fetch robot_pouring
export BLENDER=/path/to/blender

python -m physstudio run robot_pouring basin --gpu 0
python -m physstudio run robot_pouring precision --gpu 0
python -m physstudio run robot_pouring flower --gpu 0
```

The runtime pins MuJoCo 3.3.7, MuJoCo Warp 0.0.2, Warp 1.10.1 and Gymnasium
1.2.3 under `.cache/robot_pouring/runtime/`. Use `PHYSSTUDIO_ROBOT_RUNTIME` and
`PHYSSTUDIO_ROBOT_CACHE` to relocate dependencies and compiled kernels.
Start a fresh Python process before importing the environment if another Warp
version was already imported. Training uses the same isolated runtime.

The downloader verifies SHA256 hashes for the official Menagerie Panda model
and Google Sans fonts. Assets live under `data/robot_pouring/` or
`PHYSSTUDIO_DATA_ROOT/robot_pouring/`; no download occurs inside `reset()` or
`step()`. Model attribution, asset pins and font licensing are in
[references](references/sources.json). Blender 4.2.9 with Cycles/OptiX, FFmpeg
and ffprobe provide the optional presentation pipeline.

```bash
# Three saved states at full 2,272-particle showcase resolution.
python -m physstudio run robot_pouring basin --frames 3 --width 320 \
  --output outputs/robot_check --gpu 0
# Print the command without simulating.
python -m physstudio run robot_pouring flower --dry-run
```

`--frames` limits the saved prefix including frame zero; it retains the 400 Hz
physics and 25 Hz controls. It does not speed up the controller or reduce
particle count. Full demonstrations stop when the task completes. Outputs are
`simulation/`, `mesh/`, `render/`, and `media/<scene>.mp4`, with telemetry and a
local HTML gallery. Use a fresh output directory for each changed run.

## Torch interface

```python
import torch
from robot_pouring.config import PouringConfig
from robot_pouring.env import PouringEnv

env = PouringEnv(PouringConfig(task="flower", num_envs=16, spacing=.011,
                              goal_ml=220.0, seed=7))
obs, info = env.reset(seed=7)
action = torch.zeros((16, 7), device=env.device)
obs, reward, terminated, truncated, info = env.step(action)
finished = torch.where(terminated | truncated)[0]
if len(finished):
    obs, info = env.reset(env_ids=finished)
env.close()
```

`obs` is `(num_envs, 51)`, actions are `(num_envs, 7)`, and rewards and completion
flags are `(num_envs,)`. All are Torch tensors on the simulation device.
`step()` never silently resets. Preserve terminal observations for value
bootstrapping, then reset only finished worlds. `terminated` denotes success or
failure; `truncated` denotes the time limit. Calling `step()` with any unreset
finished world raises an error.

Actions in `[-1, 1]` command seven joint target velocities, with maximum target
speed `0.9 rad/s`. The target is integrated at 25 Hz and clamped to joint limits.
MuJoCo's position actuators, arm gravity compensation, joint inertia, contact,
and water reaction loads determine the actual motion. Physics runs at 400 Hz.

The 51 observation values, in order, are:

| Slice | State |
| --- | --- |
| 0:7 | Arm joint positions, rad |
| 7:14 | Arm joint velocities, scaled by 0.1 |
| 14:21 | Joint target minus position |
| 21:24 | Target center minus cup origin, m |
| 24:27 | Cup origin in the workspace, m |
| 27:33 | First two columns of cup rotation, flattened by rows |
| 33:36 | Cup translational velocity, m/s |
| 36:41 | Source, target, flight, spilled, absorbed water fractions |
| 41:43 | Desired fraction and spill budget |
| 43:44 | Elapsed fraction of the episode |
| 44:51 | Previous normalized action |

These are privileged simulator observations, suitable for an initial state-based
policy. They are not a camera-only or directly deployable real-robot sensor set.

## Gymnasium interface

```python
import gymnasium as gym
import robot_pouring.env  # registers three environments

env = gym.make("PhysStudio-WaterPlant-v0", spacing=.011, goal_ml=220.,
               render_mode="rgb_array")
obs, info = env.reset(seed=7)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
frame = env.render()  # optional diagnostic particle view
env.close()
```

The other IDs are `PhysStudio-PourBasin-v0` and `PhysStudio-PrecisionPour-v0`.
This single-world adapter returns NumPy observations and Python scalar rewards
and flags. The native Torch API avoids these per-step array transfers.

## Tasks and scoring

| Task | Vessel inner radius | Desired fraction | Additional condition |
| --- | --- | --- | --- |
| Basin | 130 mm | 85% | Return cup upright |
| Plant | 85 mm | 55% | At most 5 percentage points above target |
| Precision | 72 mm | 75% | Return cup upright |

`goal_ml` optionally requests a fixed dose; the fraction is recomputed from the
actual initial water volume at each reset. Otherwise `goal_fraction` applies.
The default spill budget is 2% of initial water. Success also requires less than
2.5% of water in flight and a 0.4-second hold with the cup upright. More than 10%
spillage or invalid dynamics ends the episode as a failure.

Reward measures incremental captured water, penalizes newly spilled water and
plant overwatering, and adds small alignment, action-change and time costs.
Keeping already delivered water does not repeatedly earn delivery reward.
`info` includes water and target volume in mL, captured/spilled fractions, physics
failure, success and elapsed time. `METHOD.md` specifies the exact formulas.

Reset randomization changes the target x/y by up to 35 mm, fill height by
75–100%, and initial joints by up to 0.008 rad. Each world has an independent
seeded generator. Joint actuators and all fluid parameters remain configurable.

## Train and evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python -m robot_pouring.train --task basin \
  --num-envs 16 --updates 100 --output outputs/pouring_ppo
CUDA_VISIBLE_DEVICES=0 python -m robot_pouring.evaluate \
  --checkpoint outputs/pouring_ppo/checkpoint.pt \
  --episodes 10 --output outputs/pouring_evaluation.json
CUDA_VISIBLE_DEVICES=0 python -m robot_pouring.validate \
  --output outputs/pouring_checks.json
```

The trainer uses 11 mm water spacing by default. The showcase controller uses
6 mm spacing. Resolution changes the fluid approximation and discretized water
volume; policy transfer between resolutions has not been established. PPO
handles termination/truncation separately, selective resets and checkpointing.
The included short integration run updates and reloads a policy successfully,
but has not learned the task. Showcase videos use a feedback reference controller
that sends ordinary joint actions through the same environment API.

## Record and replay

```bash
# Save physics only, without rendering.
CUDA_VISIBLE_DEVICES=0 python -m robot_pouring.demo --task flower \
  --output outputs/flower_recording
python -m robot_pouring.audit outputs/flower_recording

# Reconstruct and render a completed recording.
python -m robot_pouring.harness --scene flower --stage render \
  --simulation outputs/flower_recording --output outputs/flower_replay --gpu 0
# Resume matching PNGs; --start/--end select global frame indices.
python -m robot_pouring.harness --scene flower --stage render \
  --simulation outputs/flower_recording --output outputs/flower_replay --gpu 0 --resume
python -m robot_pouring.harness --scene flower --stage encode \
  --simulation outputs/flower_recording --output outputs/flower_replay
```

Rendering resumes only with matching source/settings; changed dimensions,
samples or inputs require a fresh output. An interrupted mesh reconstruction
uses a fresh replay directory. An interrupted simulation uses a fresh recording.
`audit --allow-incomplete` verifies a short prefix without requiring task success;
ordinary `audit` also requires success. Source hashes are checked against the
installed files, or a retained snapshot supplied with `--source-dir`.

`finish_media.py` preserves recorded telemetry and uses Google Sans 700/400,
translucent gradients and scene titles. Its `--font-dir` option accepts a custom
directory containing the same two named font files. Font binaries and original
Panda meshes are downloaded separately, not bundled in the Python package.

## Scope

The tasks start with a rigidly mounted pitcher. Water reacts on the cup, and
robot self-contact, table/vessel contact, fluid boundaries and particle mass
accounting are active. Soil is a localized, capacity-limited absorption model;
foliage is visual geometry. [METHOD.md](METHOD.md) describes the physical
approximation, observation limitations and collision coverage.

Original environment, controller and fluid code are licensed under
[Apache 2.0](../LICENSE). Downloaded Panda assets retain their
[Apache-2.0 notice](LICENSE.reference); Google Sans retains the
[SIL Open Font License](references/GoogleSans-OFL.txt).
