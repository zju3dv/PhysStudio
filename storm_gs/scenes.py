"""Calibrated Gaussian assets, local storm domains, and moving camera paths."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree
import torch
from lava_gs.scene import Scene
from lava_gs.cameras import cascade_camera
from lava_gs.cameras import look_at
from rainy_gs.demo import transition_camera
from .solver import Settings
from physstudio.paths import data_path


PRESETS = {
    "playground": dict(
        path="rainy_gs/fierygs/Playground",
        config=None,
        scale=2.5,
        center=[0.8, -1.3, -0.500],
        terrain="plane",
        ocean=False,
        exposure=0.65,
        dust_color=[0.50, 0.43, 0.34],
        camera=dict(dataset=[72, 90], target=[0.6, -0.7, 0.55], fov=51),
        simulation=dict(
            shape=(80, 80, 88),
            dx=0.14,
            height=8.5,
            radius=0.38,
            top_radius=1.9,
            swirl=18.0,
            updraft=9.0,
            inflow=2.0,
            drive=2.5,
            cloud=1.2,
            dust_rate=0.0012,
            humidity=0.91,
            lean=0.65,
            pressure_iterations=80,
        ),
        cloud_extinction=1200.0,
        dust_extinction=650.0,
        particles=18000,
        rain_fraction=0.65,
        replace_baked_sky=True,
        source="https://pku-vcl-geometry.github.io/FieryGS/",
    ),
    "lands_end": dict(
        path="lava_gs/lands_end",
        config="configs/lands_end.json",
        scale=1.4,
        center=[-18, -9, -0.45],
        terrain="coast",
        ocean=True,
        exposure=0.66,
        dust_color=[0.60, 0.66, 0.67],
        camera=dict(
            eye=[-3, -34, 14], offset=[7.5, 2.0, 1.2], target=[-4, 1, 8.5], fov=56
        ),
        simulation=dict(
            shape=(80, 80, 104),
            dx=0.38,
            height=28.0,
            radius=1.15,
            top_radius=3.8,
            swirl=28.0,
            updraft=10.0,
            inflow=2.3,
            drive=2.5,
            cloud=1.4,
            dust_rate=0.002,
            humidity=0.91,
            lean=2.5,
            pressure_iterations=80,
        ),
        cloud_extinction=1200.0,
        dust_extinction=450.0,
        particles=20000,
        rain_fraction=0.8,
        source="https://superspl.at/scene/9e2051b8",
    ),
    "calico": dict(
        path="lava_gs/calico",
        config="configs/calico.json",
        scale=3.0,
        center=[-0.30, 1.0, 1.15],
        terrain="rock",
        ocean=False,
        exposure=0.90,
        dust_color=[0.66, 0.47, 0.28],
        camera=dict(
            eye=[-0.9, -4.4, 3.45],
            offset=[1.6, 0.2, 0.28],
            target=[-0.1, 1.1, 3.1],
            fov=50,
        ),
        simulation=dict(
            shape=(72, 72, 128),
            dx=0.19,
            height=20.5,
            radius=0.48,
            top_radius=1.15,
            swirl=18.0,
            updraft=18.0,
            inflow=1.5,
            drive=3.0,
            cloud=0.0,
            dust_rate=0.002,
            humidity=0.30,
            temperature=301.15,
            lean=1.8,
            pressure_iterations=80,
        ),
        cloud_extinction=0.0,
        dust_extinction=400.0,
        particles=22000,
        rain_fraction=0.0,
        source="https://superspl.at/scene/5d517144",
    ),
}


def load_scene(name):
    cfg = PRESETS[name].copy()
    config = (
        json.loads((Path(__file__).parent / cfg["config"]).read_text())
        if cfg["config"]
        else {}
    )
    config.pop("sky", None)
    config["background"] = [0.0, 0.0, 0.0]
    scene = Scene(data_path(cfg["path"]), config=config)
    if name == "playground":
        # Remove the unbounded sky splats before adding a world-space sky.
        scene.storm_mask = (scene.means[:, 2] < 8.0) & (
            scene.scales.max(-1).values < 1.5
        )
    else:
        scene.storm_mask = None
    return scene, cfg


def camera(cfg, progress, width, scene=None):
    if "dataset" in cfg["camera"]:
        cam = transition_camera(
            scene, *cfg["camera"]["dataset"], progress, width, "smoothstep"
        )
        if "target" in cfg["camera"]:
            eye = cam.center.cpu().numpy()
            target = np.array(cfg["camera"]["target"], dtype=float)
            cam.view = torch.tensor(
                np.linalg.inv(look_at(eye, target, np.array([0.0, 0.0, 1.0]))),
                device=scene.device,
                dtype=torch.float32,
            )
            focal = cam.height / (2 * np.tan(np.deg2rad(cfg["camera"]["fov"]) / 2))
            cam.K[0, 0] = focal
            cam.K[1, 1] = focal
        return cam
    return cascade_camera(progress, width, settings=cfg["camera"])


def geometry(scene, cfg, settings):
    shape = np.array(settings.shape)
    dx = settings.dx
    origin = np.array([-shape[0] * dx * 0.5, -shape[1] * dx * 0.5, -2 * dx])
    scale = cfg["scale"]
    center = np.array(cfg["center"])
    pts = scene.means.cpu().numpy()
    opacity = scene.opacity.cpu().numpy()
    normals = scene.normals.cpu().numpy()
    sizes = scene.scales.cpu().numpy() * scale
    local = (pts - center) * scale
    valid = (
        (local[:, 0] > origin[0] - dx)
        & (local[:, 0] < origin[0] + shape[0] * dx + dx)
        & (local[:, 1] > origin[1] - dx)
        & (local[:, 1] < origin[1] + shape[1] * dx + dx)
        & (opacity > 0.6)
        & (sizes.max(-1) < dx * 3)
    )
    xy = np.stack(
        np.meshgrid(
            origin[0] + (np.arange(shape[0]) + 0.5) * dx,
            origin[1] + (np.arange(shape[1]) + 0.5) * dx,
            indexing="ij",
        ),
        -1,
    )
    worldxy = xy / scale + center[:2]
    plane = scene.ground.cpu().numpy()
    floor = (worldxy @ plane[:2] + plane[2] - center[2]) * scale
    if cfg["terrain"] in ["rock", "coast"]:
        upward = (
            valid
            & (normals[:, 2] > 0.45)
            & (local[:, 2] > -2.0)
            & (local[:, 2] < settings.height * 0.6)
        )
        cells = np.floor((local[upward, :2] - origin[:2]) / dx).astype(int)
        inside = (cells >= 0).all(-1) & (cells < shape[:2]).all(-1)
        cells = cells[inside]
        zz = local[upward, 2][inside]
        floor = np.full(tuple(shape[:2]), -np.inf, dtype="f")
        if len(cells):
            np.maximum.at(floor, (cells[:, 0], cells[:, 1]), zz)
        missing = ~np.isfinite(floor)
        if cfg["terrain"] == "coast":
            floor[missing] = 0.0
        else:
            nearest = distance_transform_edt(
                missing, return_distances=False, return_indices=True
            )
            floor[missing] = floor[tuple(nearest[:, missing])]
        floor = gaussian_filter(floor, 1.1)
    coords = np.stack(
        np.meshgrid(
            origin[0] + (np.arange(shape[0]) + 0.5) * dx,
            origin[1] + (np.arange(shape[1]) + 0.5) * dx,
            origin[2] + (np.arange(shape[2]) + 0.5) * dx,
            indexing="ij",
        ),
        -1,
    )
    solid = coords[..., 2] <= floor[..., None]
    if cfg["terrain"] == "plane":
        shell = (
            valid
            & (local[:, 2] > 0.3)
            & (local[:, 2] < settings.height)
            & (local[:, 2] > origin[2])
        )
        objects = local[shell]
        if len(objects):
            _, ids = np.unique(
                np.floor(objects / (dx * 0.4)).astype(int), axis=0, return_index=True
            )
            distance = (
                cKDTree(objects[ids])
                .query(coords.reshape(-1, 3), workers=3)[0]
                .reshape(tuple(shape))
            )
            solid |= distance < dx * 0.48
    return dict(
        terrain=floor.astype("f"),
        solid=solid.astype("i"),
        origin=origin.astype("f"),
        dx=dx,
        center=center.astype("f"),
        scale=scale,
    )
