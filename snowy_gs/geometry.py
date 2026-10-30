"""Metre-scale scene collision volume, snow exposure and initial powder layer."""

from physstudio.paths import ROOT, data_path
import numpy as np
from scipy.ndimage import (
    median_filter,
    gaussian_filter,
    map_coordinates,
    distance_transform_edt,
)
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F


PRESETS = {
    "playground": dict(
        path=str(data_path("rainy_gs/fierygs/Playground")),
        config=None,
        scale=2.5,
        cameras=[51, 85],
        bounds=[-5.0, -4.0, 5.0, 4.0],
        top=2.8,
        dx=0.20,
        flakes=110000,
    ),
    "firewoods": dict(
        path=str(data_path("rainy_gs/fierygs/firewoods_sand_dark")),
        config=str(ROOT / "rainy_gs/configs/firewoods.json"),
        scale=0.7,
        cameras=[94, 102],
        bounds=[-3.5, -3.5, 3.5, 3.5],
        top=0.9,
        dx=0.055,
        flakes=18000,
    ),
    "train": dict(
        path=str(data_path("rainy_gs/voxel51/train")),
        config=str(ROOT / "rainy_gs/configs/train.json"),
        scale=2.5,
        cameras=[300, 0],
        bounds=[-5.0, -5.0, 5.0, 5.0],
        top=2.8,
        dx=0.20,
        flakes=110000,
    ),
}


def build_geometry(scene, preset, seed=19):
    scale = preset["scale"]
    dx = preset["dx"]
    b = np.array(preset["bounds"]) * scale
    xyz = scene.means.cpu().numpy() * scale
    normals = scene.normals.cpu().numpy()
    opacity = scene.opacity.cpu().numpy()
    sizes = scene.scales.cpu().numpy() * scale
    plane = scene.ground.cpu().numpy().copy()
    plane[2] *= scale
    floor = lambda xy: xy[..., 0] * plane[0] + xy[..., 1] * plane[1] + plane[2]
    zmin = floor(b[:2]) - 4 * dx
    zmax = preset["top"] * scale
    origin = np.array([b[0] - 4 * dx, b[1] - 4 * dx, zmin], dtype="f")
    shape = (
        np.ceil((np.array([b[2] + 4 * dx, b[3] + 4 * dx, zmax]) - origin) / dx).astype(
            int
        )
        + 1
    )
    valid = (
        (xyz[:, :2] > b[:2] - dx).all(-1)
        & (xyz[:, :2] < b[2:] + dx).all(-1)
        & (xyz[:, 2] > zmin)
        & (xyz[:, 2] < zmax - 4 * dx)
        & (opacity > 0.55)
        & (sizes.max(-1) < dx * 3)
    )
    elevated = xyz[:, 2] - floor(xyz[:, :2]) > dx * 0.8
    points = xyz[valid & elevated]
    # Occupancy from dense reconstructed surfaces, with a finite shell around
    # thin/open geometry. Unlike a single height field, gaps beneath slides
    # and branches remain free space in this 3D contact volume.
    cell = np.floor(points / (dx * 0.2)).astype(np.int32)
    _, ids = np.unique(cell, axis=0, return_index=True)
    points = points[ids]
    tree = cKDTree(points)
    coords = np.stack(
        np.meshgrid(
            *[origin[a] + np.arange(shape[a]) * dx for a in range(3)], indexing="ij"
        ),
        -1,
    )
    d = tree.query(coords.reshape(-1, 3), workers=4)[0].reshape(tuple(shape))
    phi = np.minimum(d - dx * 0.38, coords[..., 2] - floor(coords[..., :2])).astype("f")
    # Directional top exposure sampled from upward-facing Gaussian surfaces.
    spacing = dx * 0.5
    nx, ny = np.ceil((b[2:] - b[:2]) / spacing).astype(int)
    xy = np.stack(
        np.meshgrid(
            b[0] + (np.arange(nx) + 0.5) * spacing,
            b[1] + (np.arange(ny) + 0.5) * spacing,
            indexing="ij",
        ),
        -1,
    )
    top = floor(xy).astype("f")
    upward = valid & (normals[:, 2] > 0.45)
    ij = np.floor((xyz[upward, :2] - b[:2]) / spacing).astype(int)
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    np.maximum.at(top, (ij[inside, 0], ij[inside, 1]), xyz[upward, 2][inside])
    top = median_filter(top, size=3)
    dzx, dzy = np.gradient(top, spacing)
    slope = np.sqrt(dzx**2 + dzy**2)
    # Jitter the regular quadrature so its render does not form a square grid.
    rng = np.random.default_rng(seed)
    positions = np.concatenate((xy, top[..., None]), axis=-1).reshape(-1, 3)
    positions[:, :2] += rng.uniform(-0.35, 0.35, (len(positions), 2)) * spacing
    positions[:, 2] += dx * 0.65 + 0.015
    valid_seed = (slope.reshape(-1) < 1.4) & (positions[:, 2] < zmax - 4 * dx)
    positions = positions[valid_seed]
    # The mechanical foreground contains rigid supports. Foliage in the
    # background retains a lighter appearance coating, avoiding solid slabs
    # initialized across disconnected reconstructed leaves.
    near = cKDTree(xyz[valid]).query(positions, workers=4)[1]
    vegetation = scene.vegetation.cpu().numpy()[valid][near]
    height = positions[:, 2] - floor(positions[:, :2])
    positions = positions[~((vegetation > 0.28) & (height > 0.7))]
    # Seed outside the reconstructed shell, including sloping surfaces.
    for _ in range(3):
        q = (positions - origin) / dx
        sd = map_coordinates(phi, q.T, order=1, mode="nearest")
        positions[:, 2] += np.maximum(0.012 - sd, 0)
    return dict(
        phi=phi,
        origin=origin,
        dx=dx,
        positions=positions.astype("f"),
        plane=plane,
        bounds=b,
        top=top,
        spacing=spacing,
        volume=spacing**2 * min(0.045, dx * 0.55),
        collision_points=len(points),
    )


def field_noise(p):
    return (
        0.5
        + 0.22
        * torch.sin(p[..., 0] * 1.7 + p[..., 1] * 0.8 + torch.sin(p[..., 2] * 2.3))
        + 0.14 * torch.sin(p[..., 1] * 4.3 - p[..., 0] * 2.4)
        + 0.08 * torch.sin(p[..., 0] * 13.4 + p[..., 1] * 9.7)
    ).clamp(0, 1)


def winter_material(scene, scale, strength=1.0):
    """World-locked surface coating; the geometric blanket is simulated by MPM."""
    p = scene.means * scale
    n = scene.normals
    dc = (scene.sh[:, 0] * 0.2820947918 + 0.5).clamp(0, 1)
    lum = (dc * dc.new_tensor([0.2126, 0.7152, 0.0722])).sum(-1)
    relative = (scene.means[:, 2] - scene.ground_height(scene.means[:, :2])) * scale
    slope = ((n[:, 2] - 0.2) / 0.65).clamp(0, 1).pow(0.8)
    # Grass has thin blade normals; support proximity promotes a continuous
    # blanket instead of leaving a green foreground bounded by an ROI.
    support = torch.exp(-((relative - 0.03).clamp_min(0) / 0.22).square())
    support = torch.maximum(
        support,
        scene.vegetation * torch.exp(-((relative - 0.03).clamp_min(0) / 0.65).square()),
    )
    ground = torch.maximum(slope, support * (0.75 + 0.25 * scene.vegetation))
    noise = field_noise(p)
    confidence = ((scene.opacity - 0.3) / 0.55).clamp(0, 1)
    small = (0.28 / (scene.scales.max(-1).values * scale).clamp_min(0.28)).pow(2)
    # Broad low-opacity reconstruction splats must not become white mist.
    confidence = torch.maximum(confidence, support * 0.97)
    coverage = (
        (ground * (0.80 + 0.25 * noise) * strength).clamp(0, 0.98) * small * confidence
    )
    coverage = torch.maximum(coverage, support * 0.97 * strength)
    coverage *= 1 - 0.40 * scene.vegetation * ((relative - 2.0) / 3.0).clamp(0, 1)
    sky = ((dc[:, 2] - dc[:, 0] - 0.07) / 0.12).clamp(0, 1) * (
        (relative - 2.0) / 3.0
    ).clamp(0, 1)
    coverage *= 1 - sky
    snow_light = (0.47 + 0.45 * lum.clamp_min(0.01).pow(0.32)) * (0.90 + 0.10 * n[:, 2])
    snow = snow_light[:, None] * dc.new_tensor([0.95, 0.98, 1.0])
    # Cool overcast materials, retaining recognisable wood, metal and foliage.
    cold = dc * 0.72 + lum[:, None] * 0.28
    cold = cold * dc.new_tensor([0.91, 0.96, 1.0]) + dc.new_tensor(
        [0.008, 0.012, 0.018]
    )
    cold = (
        cold * (1 - sky[:, None] * 0.9)
        + dc.new_tensor([0.66, 0.71, 0.76]) * sky[:, None] * 0.9
    )
    sh = scene.sh.clone()
    sh[:, 0] = (
        cold * (1 - coverage[:, None]) + snow * coverage[:, None] - 0.5
    ) / 0.2820947918
    sh[:, 1:] *= (0.9 * (1 - coverage))[:, None, None]
    return sh, coverage
