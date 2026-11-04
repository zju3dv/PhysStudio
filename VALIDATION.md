# Release validation

Validated from the PhysStudio checkout on Linux / RTX 3090 with the environment
listed in [INSTALL.md](INSTALL.md).

| Area | Completed checks |
| :--- | :--- |
| Rain | Shallow-water and particle checks; GPU occlusion; simulation and rendering |
| Splashing | Density relaxation and contact balance; simulation and rendering |
| Locomotion | FEM assembly against the NumPy reference; contact, affine binding, Gaussian splitting; simulation and rendering |
| Magnetism | Nonlinear field solve and discrete operators; Gaussian binding; simulation, rendering, trajectory audit |
| Snow | Free fall, plasticity, friction, deposition; depth rendering; simulation, rendering, mass/contact audit |
| Lava | Free fall, mass and heat balance, rheology, impact; all three scene simulations, rendering, and replay audits |
| Water + Sand | All three pipelines; conservation, porous equilibrium, resolution-scaled drag, grain attachment, and same-state rendering |
| Plant Wilting | Public-asset hashes, isolated Newton installation, all three pipelines, transport and binding checks |
| Storms | Moisture and latent heat, pressure projection, wall contact, dust transport, occlusion; all three pipelines and frozen-state rendering |
| Lightning | Poisson accuracy, gauge invariance, conductor enhancement, charge balance, occlusion; all three pipelines and frozen-state rendering |
| Capillary Contact | Energy/contact derivatives, surface-current operators, solid gaps, volume; all three simulation/render pipelines |
| Ice Melting | Heat and latent energy, rigid ice, liquid density, wetting kernels; all three simulation/render pipelines |
| Fire suppression | Reaction and water-energy balance, gas/solid boundaries, wood conduction, carbonization; all three pipelines and frozen-state rendering |
| Rhizomorph | All three growth/render pipelines; soil conservation, nutrient tropism, root/rock clearance, allocation, and frozen-state rendering |
| Windy Trees | All three growth/wind/render pipelines; implicit mechanics, damping, air feedback, dense paths, and frozen-state rendering |
| Magnetic Rigid Bodies | Three public pipelines, magnetic/contact checks, full audits and frozen-state rendering |
| Thin-Film Bubbles | Three public pipelines, conservative transfers, implicit operators, continuous contact, full audits and frozen-state rendering |
| Robot Pouring | CPU MuJoCo parity, liquid/load balance, independent resets, Gymnasium API, three public pipelines and frozen-state rendering |
| Launcher | All 60 presets; existing 57 command plans unchanged by the Robot Pouring addition |
| Downloads | ZIP extraction, CRC-checked cache, destination confinement; SuperSplat manifest, conversion, and camera setup |
| Media | All 61 MP4s decoded; preview files and hashes verified; MeGAS full-frame looping preview |
| Package | Wheel includes all method modules, reference attribution, and Plant Wilting, Ice Melting, and Fire-X GS runtime/download metadata |
| Source | Python syntax, English text, portable paths, local documentation links |

End-to-end checks render two video frames; Rhizomorph runs its complete 180-step
development trajectory. The Playground
rain showcase was also regenerated for its full 240 frames from the release
preset. The other showcase videos are the completed runs collected from the
original experiments. Sixteen core physics and rendering modules retain the
same Python syntax trees after import-path normalization and formatting. Five
LavaGS core modules also match their experiment versions after import-path
normalization. The packaged Calico demo uses the final 80-degree orbit.

Run the packaging check from the repository root:

```bash
python tools/check_release.py
```

The method READMEs list their numerical and GPU validation commands.

Thirteen Water + Sand solver, rendering, and validation modules match the
development version after import-path normalization and formatting. Shared
camera, video, and sampling helpers also match. Each of its three presets
completed a two-frame simulation, Gaussian render, and video audit at the full
128 × 84 × 75 simulation resolution. Same-state rendering was compared at
1280 × 800. The showcase contains the completed 180-frame trajectories.

Seven Plant Wilting core modules match the development version after import-path
normalization. The release workflow checks all three two-frame pipelines at
640 × 360, using the pinned public models and Newton installation. The Ficus
renderer reproduces five full-resolution showcase frames pixel for pixel.
The three plant showcase videos contain the complete reviewed trajectories.

Four StormGS core modules match the development version after import-path
normalization and formatting. Each of its three presets completed a two-frame
simulation, Gaussian render, and video audit. Nine frozen showcase keyframes
reproduce the development renders pixel for pixel at native resolution. The
three showcase videos contain 240 frames each. The wheel includes the StormGS
modules, scene calibrations, run guide, and reference metadata.

ThunderGS preserves the development solver and renderer after formatting.
All three presets regenerate their three discharges and complete a two-frame
Gaussian render, video audit, and gallery build. Nine full-resolution frozen
keyframes reproduce the development images pixel for pixel. The three released
videos contain 240 frames at 30 fps; the wheel includes all ThunderGS modules,
scene calibrations, run guide, and paper/asset references.

Capillary Contact keeps ten simulation and numerical-check modules byte-identical
to the accepted development version. Each exported preset completed a two-frame
simulation at the full 96 x 77 resolution, its numerical and trajectory checks,
a Cycles render, and video decoding. The complete showcase contains the accepted
180-frame orchard and 240-frame stirred cluster and paperclip trajectories.
All nine published MP4/GIF/JPG files match the approved originals; every GIF
frame and all three videos decode. Fresh Poly Haven downloads match the orchard
source assets. The wheel includes the solver, renderer, checks, run guide, and
paper/asset references.

Ice Melting keeps five core modules byte-identical to the development version;
its renderer matches after root-path normalization. All three presets completed
two-frame pipelines at the full particle resolution, including numerical checks,
mesh reconstruction, Cycles rendering, and video encoding. Cubes and arch run
without downloaded models. A fresh Bunny download matches its pinned SHA-256.
Two frozen Bunny frames were compared at 1280 x 800 with a maximum channel
difference of one 8-bit intensity level. The released videos match the three
approved 240-frame originals, including Bunny's updated wetting film; every GIF
frame decodes. The wheel includes the solver, renderer, checks, run guide, and
source metadata.

Fire-X GS preserves seven core physics and execution modules after import-path
normalization and formatting. All three presets completed two-frame pipelines
at their full simulation grid resolutions, including numerical checks, Gaussian
rendering, material and carbonization audits, and video/report generation.
Freshly prepared Playground and Bonsai material arrays match the development
assets exactly. Nine frozen frames reproduce the development images pixel for
pixel at 1280 x 800. The released videos match the accepted 420-frame Firewoods
and 450-frame Playground and Bonsai originals; every GIF frame decodes. The
wheel includes the modules, scene calibrations, run guide, CIE table, and
reference attribution.

Multi-fluid preserves the validated solver, scene, transport, surface, and
rendering implementation after portable imports and preset selection. All three
presets completed two-frame pipelines at their full particle resolution through
the shared launcher, including numerical checks, simulation audits, Cycles
rendering, and video decoding. Frozen tea and collision frames agree with the
development renders within one 8-bit channel level at 1280 x 800. The three
showcase MP4s match the approved files; every GIF frame decodes. The wheel
includes all modules, the control preset, run guide, and paper references.

Ice Solidification preserves twelve core modules after portable import and
root-path normalization. All three presets completed two-frame pipelines at
their full particle resolutions through the shared launcher, including heat
and contact audits, surface reconstruction, Cycles rendering, and MP4 decoding.
The packaged latent-heat, closed-exchange, isothermal, and Stefan-front checks
pass. Six frozen-state frames reproduce the reviewed images within one 8-bit
channel level at 1280 x 800. The three MP4s match the accepted originals;
every GIF frame decodes. The wheel includes all modules, cooling presets,
run guide, and paper references.

Rhizomorph preserves the development solver, presets, and checks after formatting.
All three presets completed 180 growth steps through the shared launcher,
including resource/trajectory audits, two-frame Cycles renders, and video decoding.
Transport checks pass for equilibrium, water/nutrient conservation, sealed cells,
and nonnegativity; all 256 paired root tips respond to the reversed nutrient
field. Three frozen frames reproduce the development images within one 8-bit
channel level at 960 x 1080. The released MP4s match the accepted 240-frame
originals, every GIF frame decodes, and the wheel includes the modules, run guide,
method notes, reference metadata, and attribution.

Windy Trees preserves seven core solver, scene, asset, simulation, flow and
rendering modules after import/root-path normalization. All three presets
completed 180-step source growth and two-frame wind/render/encode pipelines
through the shared launcher at the normal 15,732-air-particle resolution.
Independent matrix-free/dense mechanics and analytic damped-oscillator checks
pass, as do fixed-root, segment-length, tree/air force balance, recorded-flow
endpoints, and matched crown-feedback checks. Regenerated trees retain the
showcase topology; small floating-point differences remain in the GPU states.
Nine frozen frames reproduce the development renders within one 8-bit channel
level at 1280 x 720. The three 301-frame showcase MP4s match the accepted files,
all GIF frames decode, and the wheel includes the modules, dense recording
preset, run guide, method/flow notes, references and shared rendering helpers.

Magnetic Rigid Bodies preserves six implementation/check modules byte for byte;
its renderer differs only in the portable repository root. All three presets
completed two-frame pipelines through the shared launcher at their full body
and magnetic-sample resolutions, including physical audits, Cycles rendering,
and verified MP4 encoding. Six numerical checks and all three complete
showcase audits pass. Regenerated crane/globe prefixes match the accepted
states exactly; compass quaternion and angular-velocity differences are below
1.2e-13 and 3.8e-11 respectively. Nine frozen frames agree at 1280 x 720 within
two 8-bit channel levels (mean below 0.018). The three showcase MP4s match the
accepted originals, and all GIF frames decode. Existing 51 CLI plans, media
records, and showcase order are preserved. See the
[method validation](magnetic_rigid/VALIDATION.md) for measured results.

Thin-Film Bubbles preserves eleven implementation/check modules byte for byte.
All three presets completed three-frame public-CLI pipelines at full particle
resolution, including recording and mesh audits, Cycles renders and verified
encoding. Analytic geometry, conservative scatter, compatible pressure/CG,
force balance and continuous/unilateral contact checks pass, as do all three
complete recording audits. State-prefix differences are below 4.1e-13 m and
1.2e-10 m/s; nine frozen 1280 x 720 renders differ by at most one 8-bit channel
level (mean below 0.000054). Resume preserves PNGs and rejects changed settings.
The accepted MP4s are unchanged, all GIF frames decode, and the wheel includes
the solver, reconstruction, renderer, harness, checks, guides and references.
Existing 54 command plans, 55 media records and 21 showcase sections are
preserved. See [method validation](melp_thin_film/VALIDATION.md) for measurements
and the intact-film collision scope.

Robot Pouring passes 14 physical/API checks in its isolated MuJoCo Warp runtime.
All three public pipelines complete three saved frames at 2,272 particles,
including fluid surfaces, Blender rendering and encoded telemetry. Their
particle and joint-position prefixes match the accepted recordings exactly;
joint velocities differ by at most 2.99e-8 rad/s. Nine full-resolution frozen
renders differ by at most one 8-bit channel level (mean below 0.000047).
Render resumes preserve PNGs and reject changed settings. A 128-transition
PPO check updates and reloads a policy; task convergence is not established.
The wheel includes all 26 method files, notices and asset pins; installed-wheel
imports pass, and the existing 315 package files remain unchanged except for
the separately updated launcher. The three showcase MP4s retain the reviewed
Google Sans typography and exact video hashes. [Method validation](robot_pouring/VALIDATION.md).
