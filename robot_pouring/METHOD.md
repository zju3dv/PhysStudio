# Robot and liquid model

## Components and references

- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp): official
  GPU articulated rigid-body dynamics, actuators, joint limits and contacts.
  Validated runtime: MuJoCo 3.3.7, MuJoCo Warp 0.0.2, Warp 1.10.1.
- [MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html):
  batched worlds, external body wrenches, graph capture and capacities.
- [Franka Emika Panda model](https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/franka_emika_panda):
  official Menagerie MJX model, inertial parameters and meshes, Apache-2.0.
  The downloaded XML is unmodified. `model.py` creates a task-specific model
  in memory with a pitcher, target, table and gravity compensation.
- [Macklin and Mueller, Position Based Fluids, 2013](https://mmacklin.com/pbf_sig_preprint.pdf):
  independent Warp implementation of density projection, pressure-gradient
  corrections, artificial pressure and XSPH viscosity.
- [MuJoCo fluid-force documentation](https://mujoco.readthedocs.io/en/stable/computation/fluid.html):
  its built-in surrounding-fluid drag is not a free-surface liquid solver.
  This environment therefore evolves a separate water state.

Newton was considered as an integration layer. This version uses its underlying
MuJoCo Warp rigid-body backend directly; it does not depend on Newton or claim
to implement a Newton solver. All original task and fluid code is independent.

## Robot, control and collision

The arm has seven controlled joints and two finger joints. Original Panda
position actuators and force limits are retained. Arm and empty-cup gravity
compensation is enabled; water loads remain external forces. The pitcher is
fixed to the hand, so grasp acquisition and grasp failure are outside this task.
Actions integrate position targets at 25 Hz; physics uses sixteen 2.5 ms substeps.
The target speed is limited to 0.9 rad/s and commands are clamped to joint ranges.

Robot self-contact and contact with the table and vessels use MuJoCo collision
meshes/primitives. The cup is excluded from its own hand, fingers and last link.
Vessels are open: one circular base and 32 wall boxes for rigid contact. Fluid
contact uses the corresponding smooth circular shell and bottom signed distance
functions. This introduces a small polygon-versus-circle discrepancy in rigid
contact; the rendered circular geometry follows the fluid boundary. Water is
coupled to the cup and collides with the receiving vessel, soil/stem and table.
Water-to-other-arm-link contact is not implemented in this first version.

## Water solve

Equal-mass particles have mass `rho0 * dx^3`, with `rho0 = 1000 kg/m^3`.
The initial cylindrical lattice is clipped to the cup interior. The interaction
radius is `h = 2.4 dx`; density weights use poly6 normalized by its discrete sum
on an infinite uniform lattice at spacing `dx`.

Each substep predicts positions under gravity, projects solid boundaries, then
performs four PBF density iterations. The inequality `rho_i/rho0 <= 1` avoids
tensile negative pressure near a free surface. Lambda uses the sum of squared
constraint gradients and regularization `0.01/h^2`. Pressure corrections use
the spiky gradient and a small artificial-pressure term. A short-range symmetric
separation correction below `0.65 dx` prevents particle pairing at boundaries;
the total correction is limited to `0.35 dx` per iteration. The neighbor grid
is rebuilt every iteration. An XSPH velocity smoothing coefficient of 0.025
is applied after position-derived velocities. Floor slip has local dissipation.

This is a graphics-oriented liquid approximation with controllable resolution,
not a calibrated Navier–Stokes/CFD model. Missing boundary neighbors and finite
iterations introduce free-surface and volume-discretization error. The particle
mass ledger remains exact. Air, calibrated surface tension, turbulence and
sub-particle splashing are not resolved. The diagnostic renderer displays
particles; the presentation renderer reconstructs a Gaussian isosurface.

## Coupling and soil

For each cup boundary projection `delta_x`, the water receives impulse
`m * delta_x / dt`. Its opposite is accumulated as a cup force
`-m * delta_x / dt^2`, with a torque about the cup origin. The external moment
is shifted to the MuJoCo inertial center before application. The robot receives
the accumulated wrench at the next substep: this is explicit staggered
two-way coupling, not a monolithic implicit solve. Water weight and impacts
therefore affect joint motion. Other scene boundaries are static.

Plant soil absorbs a particle only upon reaching the interior soil surface.
The absorbed label is permanent, and its impact location is retained for spatial
wetting visualization. Capacity is 75% of initial water; after this capacity,
water remains above the soil and can overflow the pot. The central stem is a
static cylinder. Foliage is visual, and the soil is not a Darcy/Richards model.

## Mass ledger and reward

Each initially active particle belongs to exactly one class: source cup,
receiving vessel, flight, irreversibly spilled, or absorbed soil. Unfilled
slots are inactive and are excluded from denominators. Spilled water keeps
moving but can never earn target credit again. Classification is idempotent.

Let `c` be captured fraction (target + absorbed), `s` spilled fraction, `g` goal,
and `d` horizontal distance between the lowest cup rim point and target center.
Per-step reward is:

```
25 * delta(min(c,g)) - 60 * delta(s) + 2 * (d_previous-d)
- 0.002 - 0.001 * mean(action^2) - 0.001 * mean(delta_action^2)
```

For plants, add `-30 * delta(max(c-g-0.04,0))`. Success gives +10; failure gives
-5. Success requires `c >= g-0.03`, spill <= configured budget, flight < 0.025,
and cup world-up alignment > 0.94 for 0.4 s. Plants also require `c <= g+0.05`.
Failure means >10% spilled, invalid or excessive particle/joint velocities,
non-finite robot state, or cup height outside [0.06,1.1] m. Contact/constraint
capacity overflow raises an error instead of silently continuing bad physics.

The state-based policy gets joint state, cup/target state, water fractions,
goal/budget, time and previous action. This observation is a compact privileged
summary and does not fully expose internal fluid velocities; recurrence or
richer fluid features may help long-horizon policies.

## Parallelism and reproducibility

MuJoCo Warp batches robot worlds. Water uses a shared spatial hash with separated
world origins plus an explicit world-id check on every neighbor interaction.
Reset clears selected robot/fluid state, forces, counters and task history.
Other worlds retain their state. Random generators are independent per world.
GPU floating-point atomic accumulation is not guaranteed bit-identical over
long trajectories or across different batch layouts.

CUDA 12.2 drivers lack conditional graph nodes. The implementation selects
MuJoCo Warp's supported fixed-iteration Newton path (`graph_conditional=False`)
and captures ordinary CUDA graphs. There are 12 Newton iterations with six
line-search iterations. This changes execution scheduling, not the physical
constraints. The method runtime is isolated from the shared Warp 1.6.2 environment.

PPO uses a squashed Gaussian policy, clipped ratios, GAE and separate termination
and truncation bootstrapping. The included small training run verifies updates,
resets and checkpoint reload. It is not evidence of policy convergence. Videos
are explicitly labeled as reference-controller rollouts.
