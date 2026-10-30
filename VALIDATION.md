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
| Launcher | All 18 presets; existing 15 command plans unchanged |
| Downloads | ZIP extraction, CRC-checked cache, destination confinement; SuperSplat manifest, conversion, and camera setup |
| Media | All 19 MP4s decoded; preview files and hashes verified; MeGAS full-frame looping preview |
| Package | Wheel built with LavaGS modules and scene configurations |
| Source | Python syntax, English text, portable paths, local documentation links |

Each end-to-end pipeline check used two simulation/video steps. The Playground
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
