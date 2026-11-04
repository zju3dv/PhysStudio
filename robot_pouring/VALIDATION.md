# Validation

The preserved implementation uses MuJoCo 3.3.7, MuJoCo Warp 0.0.2, Warp 1.10.1,
Torch 2.4.0+cu118 and Gymnasium 1.2.3 on an RTX 3090. See the root
[installation guide](../INSTALL.md) for runtime isolation.

## Physical and RL checks

The development implementation passes 14 checks: nonzero joint commands against
CPU MuJoCo, static water/load balance, opposite boundary impulses and torque,
particle mass accounting, selective reset isolation, exact seeded reset,
overlapping-world neighbor exclusion, joint action response, soil capacity,
fixed milliliter goals, invalid actions, truncation/reset semantics, detached
policy actions and the official Gymnasium checker.

All three full-resolution reference-controller recordings pass per-frame audits:

| Task | Captured | Volume | Spilled | Success time |
| --- | --- | --- | --- | --- |
| Basin Transfer | 94.102% | 461.808 mL | 0 | 9.40 s |
| Precision Pour | 88.644% | 435.024 mL | 0 | 9.24 s |
| Gentle Watering | 53.917% | 264.600 mL | 0 | 8.48 s |

Each recording starts with 2,272 particles representing 490.752 mL. The plant
receives water within tolerance of its 269.914 mL goal. Basin and precision
use minimum-delivery goals. The controller returns the cup upright in all cases.
The maximum mass partition error is 5.97e-8; minimum solid gaps are within
8 micrometers of particle radius.

A 3,072-transition PPO integration run completes 12 updates and 8 episodes,
with finite losses/gradients and an exact checkpoint reload. Evaluation on two
unseen seeds succeeds in 0/2 episodes. These are integration checks; the videos
show a reference controller, not a successfully trained policy.

The 32-world configuration with 333 particles/world measures 420.35 aggregate
transitions/s (76.13 ms per vector step) on one RTX 3090. It includes observations,
rewards and action validation, and excludes rendering/policy optimization.
This is one shared-host measurement at 11 mm spacing, not a scaling study or
a throughput claim for the 6 mm showcase configuration.

## Public package checks

Public-package checks are recorded in the root [validation guide](../VALIDATION.md).

The public package passes all 14 physical/API checks in its freshly installed
isolated runtime. Each scene completes a three-frame public-CLI pipeline at
2,272 particles, including reconstruction, rendering, telemetry and full video
decoding. Particle positions/velocities and robot positions match the recorded
prefixes exactly; maximum joint-velocity difference is 2.99e-8 rad/s.
Nine frozen 1280 x 720 renders differ by at most one 8-bit channel level
(mean below 0.000047). Matching render resumes preserve PNGs; changed settings
are rejected. A 128-transition, two-update public PPO check has finite losses,
parameter change L2 0.22081 and exact checkpoint reload; no episode completes
within this short integration check.

The wheel includes all 26 method files and required notices, and installed-wheel
model/environment imports pass. The previous 315 package files remain identical
apart from the separately updated shared launcher. Existing 57 command plans
are preserved; the three new presets bring the total to 60.
Run numerical and API checks with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m robot_pouring.validate --output outputs/robot_checks.json
```

The three published videos are byte-identical to the reviewed Google Sans
versions: 1280 x 720, 25 fps, 236/232/213 frames. Their initial state frame adds
0.04 s to the simulated success times above. The runtime, downloaded assets,
short public pipelines, saved-state rendering and wheel contents are checked
separately from long-horizon policy convergence.
