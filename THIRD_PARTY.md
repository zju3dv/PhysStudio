# Credits

PhysStudio contains our independent implementations and demo workflows inspired
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

The MeGAS showcase uses the official Kitchen video from the [MeGAS project page](https://zju3dv.github.io/MeGAS/). The README plays a full-frame animated loop, and the video gallery autoplays the original MP4. The source URL and hash are recorded in `media/manifest.json`. [MeGAS code](https://github.com/zju3dv/MeGAS).
