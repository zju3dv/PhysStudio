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
| Launcher | All 15 presets and all five native CLI help commands |
| Downloads | ZIP extraction, CRC-checked cache, destination confinement |
| Media | All 15 MP4s decoded; preview files and hashes verified |
| Source | Python syntax, English text, portable paths, local documentation links |

Each end-to-end pipeline check used two simulation/video steps. The Playground
rain showcase was also regenerated for its full 240 frames from the release
preset. The other showcase videos are the completed runs collected from the
original experiments. Sixteen core physics and rendering modules retain the
same Python syntax trees after import-path normalization and formatting.

Run the packaging check from the repository root:

```bash
python tools/check_release.py
```

The five method READMEs list their numerical and GPU validation commands.
