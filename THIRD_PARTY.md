# Credits and third-party licenses

Original PhysStudio code is licensed under [Apache 2.0](LICENSE); see [NOTICE](NOTICE).
Third-party components and their derived code retain their upstream licenses
and copyright notices. External assets and official showcase media retain
their upstream terms and are not relicensed by PhysStudio's code license.

The Python distribution records Apache-2.0, MIT, BSD-2-Clause, and BSD-3-Clause
in its license expression to account for the original code and bundled
third-party components. The retained notices are listed below.

| Component | Use in PhysStudio | Retained license |
| :--- | :--- | :--- |
| LBM-MPM-Air-Water-Sand | Water + Sand constitutive and coupling routines | [MIT](water_sand/LICENSE.reference) |
| plant-water-diffusion | Plant Wilting water-transport reference | [MIT](plant_wilting/LICENSE.reference) |
| FieryGS | Fire-X GS CIE matching table | [Apache 2.0](firex_gs/LICENSE.reference) |
| PlenOctree spherical harmonics | Fire-X GS SH evaluator | [BSD 2-Clause notice in source](firex_gs/reference_sh.py) |
| Rhizomorph | Tree & Root Growth reference, reused by Windy Trees | [BSD 3-Clause](rhizomorph/LICENSE.reference) |

PhysStudio contains physics-based implementations and demo workflows inspired
by the papers linked in each method's README.

Runtime dependencies are installed separately: [PyTorch](https://pytorch.org/),
[NVIDIA Warp](https://github.com/NVIDIA/warp),
[gsplat](https://github.com/nerfstudio-project/gsplat),
[OSQP](https://github.com/osqp/osqp-python), NumPy, SciPy, Pillow, and plyfile.
Video encoding uses [FFmpeg](https://ffmpeg.org/).

Pretrained scene and object models come from FieryGS, Graphdeco, Voxel51,
PhysGaussian, and the SuperSplat authors credited in the LavaGS presets; [ASSETS.md](ASSETS.md) links each download and source.
Models, camera archives, paper PDFs, and third-party binary packages are not
included in this repository. The implementation demos are rendered by PhysStudio using those publicly released assets.

The MeGAS showcase uses the official Bidirectional Phase-Change Editing video from the [MeGAS project page](https://zju3dv.github.io/MeGAS/). The README plays a full-frame animated loop, and the video gallery autoplays the original MP4. The source URL and hash are recorded in `media/manifest.json`. [MeGAS code](https://github.com/zju3dv/MeGAS).

Water + Sand adapts constitutive and coupling routines from the
[released 2D LBM–MPM implementation](https://github.com/DacianShaw/LBM-MPM-Air-Water-Sand)
by Xiaoyu Xiao; its [MIT notice](water_sand/LICENSE.reference) is included.
[OpenHOMELBM](https://github.com/kuiwuchn/OpenHOMELBM) provides the 3D Warp LBM
organization reference. Pinned revisions are in
[water_sand/references/sources.json](water_sand/references/sources.json).

Plant Wilting uses the water-transport formulation from
[Filippo Maggioli and collaborators](https://github.com/filthynobleman/plant-water-diffusion),
with its [MIT notice](plant_wilting/LICENSE.reference), and
[Newton VBD rods](https://github.com/newton-physics/newton) as the GPU mechanical backend.
Plant Gaussians are downloaded from [DiffWind](https://github.com/zju3dv/DiffWind).
[Pinned sources](plant_wilting/references/sources.json) and
[asset hashes](plant_wilting/references/assets.json) accompany the workflow.

StormGS implements the moist-flow model from
[Cyclogenesis: Simulating Hurricanes and Tornadoes](https://doi.org/10.1145/3658149).
The authors' [CycloCore notebook](https://github.com/AgrosAmad/CycloCore)
(MPL 2.0) supplies reference equations; the Warp solver is implemented here.
The scene assets are shared with RainyGS and LavaGS.
[Pinned sources](storm_gs/references/sources.json).

ThunderGS implements the electrical-discharge model from
[Thunderstruck: Visually Simulating Electrical Storms](https://jonathank.de/research/paper/Herrera2025_thunderstruck.pdf),
with a neutral floating-channel correction and Gaussian-scene lighting.
Firewoods and Train reuse the RainyGS assets; Zohar is the CC BY 4.0 scene
by barak9177 used in LavaGS. [Sources](thunder_gs/references/sources.json).

Capillary Contact follows the surface-energy and rigid-contact formulation of
[Solid-Fluid Interaction with Surface-Tension-Dominant Contact](https://arxiv.org/abs/2105.08471)
(SIGGRAPH 2021), with a height-field fluid adapter and transported surface currents.
[Blender](https://www.blender.org/) supplies Cycles rendering; the orchard models
come from [Poly Haven](https://polyhaven.com/) under CC0.
[Sources](surface_tension/references/sources.json).

Ice Melting combines particle enthalpy transport, position-based fluids, rigid
shape matching, and ice/water surface forces from the visual ice-melting
literature. [Paper and asset references](ice_melting/references/sources.json)
record the source links and pinned Stanford Bunny download. Blender supplies
Cycles rendering; the other scene geometry is procedural.

Fire-X GS uses the thermochemical formulation of
[Fire-X](https://helgewrede.github.io/firex/) and the Gaussian surface and fire
rendering design of [FieryGS](https://pku-vcl-geometry.github.io/FieryGS/).
The CIE matching table is included with FieryGS's
[Apache-2.0 license](firex_gs/LICENSE.reference); the SH reference evaluator
retains its PlenOctree Authors BSD notice. The Warp solver and water coupling
are implemented here. [Pinned sources and asset links](firex_gs/references/sources.json).

Multi-fluid implements the phase-velocity mapping and conservative composition
transport from [An Implicitly Stable Mixture Model for Dynamic Multi-fluid
Simulations](https://doi.org/10.1145/3610548.3618215) in Warp. The authors'
[Taichi implementation](https://github.com/sakamotoyan/TiSPH_multiphase) provides
reference conventions for discrete viscosity and pressure iteration.
Blender supplies Cycles rendering. [Pinned references](implicit_mixture/references/sources.json).

Ice Solidification follows the enthalpy and SPH heat-transfer formulation of
[Solidification using smoothed particle hydrodynamics](https://doi.org/10.1016/j.jcp.2004.11.039),
with implicit heat exchange, position-based liquid motion, and continuous phase
optics. The solver runs in Warp and Blender supplies Cycles rendering.
[Paper references](ice_solidification/references/sources.json).

Rhizomorph follows the soil-resource coupling and plant-signal architecture of
[Rhizomorph: The Coordinated Function of Shoots and Roots](https://doi.org/10.1145/3592145).
The authors' [official implementation](https://github.com/edisonlee0212/rhizomorph)
provides reference growth and soil-transport logic. The Warp kernels and Python
graph implementation are developed here; Blender supplies Cycles rendering.
[Pinned reference](rhizomorph/references/sources.json) ·
[Reference BSD-3-Clause license](rhizomorph/LICENSE.reference).

Windy Trees follows the short-term wind/tree interaction in
[Windy Trees: Computing Stress Response for Developmental Tree Models](https://doi.org/10.1145/2661229.2661252)
by Sören Pirk, Till Niese, Torsten Hädrich, Bedrich Benes, and Oliver Deussen.
The Warp articulated solver, SPH coupling, and leaf hinges are implemented here.
Procedural trees reuse Rhizomorph; Blender rendering reuses the packaged
Rhizomorph and Ice Melting helpers. Rhizomorph's
[reference attribution](rhizomorph/LICENSE.reference) remains included.
The [official video](https://www.youtube.com/watch?v=o6EKTlEV44E) provides the
visual reference for neutral airflow filaments; the three showcase videos are
rendered from PhysStudio simulations. [Sources](windy_trees/references/sources.json).

Magnetic Rigid Bodies implements the equations in
[Strongly Coupled Simulation of Magnetic Rigid Bodies](https://doi.org/10.1111/cgf.15185)
independently in Python and NVIDIA Warp. This code is covered
by PhysStudio's Apache 2.0 license; the authors' STARK code is neither included
nor required. NumPy/SciPy supplies the host Newton solve, and Blender supplies
Cycles rendering. All scene assets are procedural. [Paper and rendering sources](magnetic_rigid/references/sources.json).

Thin-Film Bubbles follows [A Moving Eulerian-Lagrangian Particle Method for
Thin Film and Foam Simulation](https://doi.org/10.1145/3528223.3530174) by
Yitong Deng, Mengdi Wang, Xiangxin Kong, Shiying Xiong, Zangyueyang Xian and
Bo Zhu. The Python/Warp solver is independently implemented and covered by
PhysStudio's Apache 2.0 license. The [authors' C++ implementation](https://github.com/wang-mengdi/melp-thin-film)
was consulted for algorithm research; its code is neither bundled nor required.
Blender supplies Cycles rendering, and all scene assets are procedural.
[Paper, pinned reference and rendering metadata](melp_thin_film/references/sources.json).

Robot Pouring uses the official [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)
rigid-body backend and the [Franka Panda model](https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/franka_emika_panda)
from MuJoCo Menagerie, retaining its [Apache-2.0 license](robot_pouring/LICENSE.reference).
The Warp liquid solver independently implements position-based fluid concepts
from [Macklin and Mueller](https://mmacklin.com/pbf_sig_preprint.pdf). Task logic,
cup coupling and RL interfaces are original PhysStudio code. Blender supplies
Cycles rendering. The videos use Google Sans by the Google Sans Project Authors;
font downloads retain the [SIL Open Font License](robot_pouring/references/GoogleSans-OFL.txt).
Robot meshes and font binaries are downloaded separately and are not bundled.
[Source metadata](robot_pouring/references/sources.json).
