"""Three calibrated scene assets, ground conductors, and storm camera paths."""
from physstudio.paths import data_path
import copy
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, median_filter
import torch
from lava_gs.scene import Scene
from lava_gs.cameras import cascade_camera
from lava_gs.cameras import look_at
from rainy_gs.demo import transition_camera

PRESETS = {
    "firewoods": dict(
        path="rainy_gs/fierygs/firewoods_sand_dark",
        config="configs/firewoods.json",
        source="https://github.com/PKU-VCL-Geometry/FieryGS#data",
        origin=[-4.32, -4.32, -2.88],
        shape=[72, 72, 96],
        dx=0.12,
        cloud=[-0.3, -0.2, 3.1],
        cloud_radius=[1.7, 1.5, 0.8],
        upper_offset=[1.0, 0.5, 3.2],
        terrain_cap=-0.7,
        seed_min=0.4,
        seed_max=2.7,
        exposure=0.34,
        fog=0.035,
        camera=dict(dataset=[94, 105], target=[0.0, 0.0, -0.5], fov=66),
        critical=0.16,
        internal=0.045,
        eta=2.3,
        charge=2.0,
        seed=128,
        title="Firewoods · Close strike",
    ),
    "train": dict(
        path="rainy_gs/voxel51/train",
        config="configs/train.json",
        source="https://huggingface.co/datasets/Voxel51/gaussian_splatting/tree/main/FO_dataset/train",
        origin=[-5.04, -3.36, -1.12],
        shape=[72, 80, 96],
        dx=0.14,
        cloud=[-1.0, 0.4, 6.4],
        cloud_radius=[2.0, 1.8, 0.9],
        upper_offset=[1.2, 0.3, 3.4],
        terrain_cap=2.1,
        seed_min=3.0,
        seed_max=5.8,
        exposure=0.33,
        fog=0.022,
        camera=dict(dataset=[20, 300], pitch=5.0, fov=55),
        critical=0.19,
        internal=0.05,
        eta=2.5,
        charge=2.0,
        seed=211,
        title="Train · Conductive roof",
    ),
    "zohar": dict(
        path="lava_gs/zohar",
        config="configs/zohar.json",
        source="https://superspl.at/scene/7aa4d3d4",
        origin=[-2.4, -1.2, -1.4],
        shape=[72, 88, 112],
        dx=0.075,
        cloud=[0.15, 1.8, 4.4],
        cloud_radius=[1.2, 1.4, 0.65],
        upper_offset=[0.5, 0.5, 2.0],
        terrain_cap=2.5,
        seed_min=2.6,
        seed_max=4.0,
        exposure=0.42,
        fog=0.045,
        camera=dict(
            eye=[-0.3, -3.6, 0.9],
            offset=[0.85, 0.4, 0.25],
            target=[0.0, 1.2, 1.3],
            fov=63,
        ),
        critical=0.12,
        internal=0.025,
        eta=2.4,
        charge=2.5,
        seed=367,
        title="Zohar · Canyon discharge",
    ),
}

PRESETS = {name: PRESETS[name] for name in ("firewoods", "zohar", "train")}


def load(name):
    cfg = copy.deepcopy(PRESETS[name])
    config = json.loads((Path(__file__).parent / cfg["config"]).read_text())
    config.update(background=[0.0, 0.0, 0.0], far_plane=150.0)
    scene = Scene(data_path(cfg["path"]), config=config)
    scene.thunder_mask = None
    if name == "zohar":
        scene.thunder_mask = (scene.means[:, 2] < 3.0) & (
            scene.scales.max(-1).values < 1.5
        )
    return scene, cfg


def camera(scene, cfg, progress, width):
    c = cfg["camera"]
    if "dataset" in c:
        cam = transition_camera(scene, *c["dataset"], progress, width, "smoothstep")
        eye = cam.center.cpu().numpy()
        if "target" in c:
            target = np.array(c["target"])
        else:
            forward = torch.linalg.inv(cam.view)[:3, 2].cpu().numpy()
            forward[2] += np.tan(np.deg2rad(c.get("pitch", 0.0)))
            target = eye + forward * 5.0
        pose = look_at(eye, target, np.array([0.0, 0.0, 1.0]))
        cam.view = torch.tensor(
            np.linalg.inv(pose), dtype=torch.float32, device=scene.device
        )
        cam.K[0, 0] = cam.K[1, 1] = cam.height / (2 * np.tan(np.deg2rad(c["fov"]) / 2))
        return cam
    return cascade_camera(progress, width, settings=c)


def domain(scene, cfg):
    shape = np.array(cfg["shape"])
    dx = cfg["dx"]
    origin = np.array(cfg["origin"])
    points = scene.means.cpu().numpy()
    opacity = scene.opacity.cpu().numpy()
    sizes = scene.scales.max(-1).values.cpu().numpy()
    cells = np.floor((points - origin) / dx).astype(int)
    valid = (
        (cells >= 0).all(-1)
        & (cells < shape).all(-1)
        & (opacity > 0.65)
        & (sizes < dx * 2.0)
        & (points[:, 2] < cfg["terrain_cap"])
    )
    cells = cells[valid]
    z = points[valid, 2]
    roof = np.full(tuple(shape[:2]), -np.inf)
    np.maximum.at(roof, (cells[:, 0], cells[:, 1]), z)
    missing = ~np.isfinite(roof)
    if missing.all():
        raise ValueError("No scene surface inside discharge domain")
    nearest = distance_transform_edt(
        missing, return_distances=False, return_indices=True
    )
    roof[missing] = roof[tuple(nearest[:, missing])]
    roof = median_filter(roof, size=3)
    roof = gaussian_filter(roof, 0.45)
    coords = np.stack(
        np.meshgrid(
            *[origin[d] + (np.arange(shape[d]) + 0.5) * dx for d in range(3)],
            indexing="ij"
        ),
        -1,
    )
    solid = coords[..., 2] <= roof[..., None]
    solid[:, :, 0] = True
    return dict(coords=coords, solid=solid, roof=roof, origin=origin)


def charges(cfg, geometry, event=0):
    p = geometry["coords"]
    cloud = np.array(cfg["cloud"], dtype=float)
    cloud += np.array(
        [0.32 * np.sin(event * 2.1), 0.25 * np.cos(event * 1.8), 0.08 * event]
    )
    radius = np.array(cfg["cloud_radius"])
    upper = cloud + cfg["upper_offset"]
    negative = np.exp(-0.5 * np.sum(((p - cloud) / radius) ** 2, -1))
    positive = np.exp(
        -0.5 * np.sum(((p - upper) / (radius * np.array([1.35, 1.2, 0.85]))) ** 2, -1)
    )
    negative[geometry["solid"]] = 0
    positive[geometry["solid"]] = 0
    # The atmosphere can carry net charge; the grounded Earth supplies its
    # induced counterpart. The upper reservoir only partially screens it.
    rho = cfg["charge"] * (
        -negative + 0.40 * positive * negative.sum() / max(positive.sum(), 1e-10)
    )
    seed = (
        (p[..., 2] > cfg["seed_min"])
        & (p[..., 2] < cfg["seed_max"])
        & (np.linalg.norm((p[..., :2] - cloud[:2]) / radius[:2], axis=-1) < 0.65)
    )
    return rho.astype("f"), seed
