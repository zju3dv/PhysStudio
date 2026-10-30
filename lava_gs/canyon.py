"""Reconstructed canyon terrain and an elevated, continuously fed lava source."""

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    median_filter,
    map_coordinates,
)


def build_canyon(scene, cfg, seed=17):
    scale, dx = cfg["scale"], cfg["dx"]
    bounds = np.array(cfg["bounds"]) * scale
    xyz = scene.means.cpu().numpy() * scale
    opacity = scene.opacity.cpu().numpy()
    sizes = scene.scales.cpu().numpy() * scale
    z_bounds = np.array(cfg.get("terrain_z_bounds", [-1.4, 2.4])) * scale
    xy0 = bounds[:2] - 4 * dx
    shape = np.ceil((bounds[2:] + 4 * dx - xy0) / dx).astype(int) + 1
    valid = (
        (xyz[:, :2] >= xy0).all(-1)
        & (xyz[:, :2] < xy0 + (shape - 1) * dx).all(-1)
        & (xyz[:, 2] > z_bounds[0])
        & (xyz[:, 2] < z_bounds[1])
        & (opacity > 0.65)
        & (sizes.max(-1) < dx * 2.0)
    )
    points = xyz[valid]
    ij = np.floor((points[:, :2] - xy0) / dx).astype(int)
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    order = np.argsort(flat)
    f = flat[order]
    z = points[order, 2]
    starts = np.r_[0, np.where(np.diff(f))[0] + 1]
    ends = np.r_[starts[1:], len(f)]
    terrain = np.full(tuple(shape), np.nan, dtype="f")
    count = np.zeros(tuple(shape), dtype="i")
    for a, b in zip(starts, ends):
        coord = np.unravel_index(f[a], tuple(shape))
        count[coord] = b - a
        terrain[coord] = np.quantile(z[a:b], 0.90)
    missing = ~np.isfinite(terrain)
    ids = distance_transform_edt(missing, return_distances=False, return_indices=True)
    terrain[missing] = terrain[tuple(ids[:, missing])]
    terrain = gaussian_filter(median_filter(terrain, size=3), 0.6)
    zmin = float(terrain.min() - 5 * dx)
    zmax = float(terrain.max() + 6 * dx)
    nz = int(np.ceil((zmax - zmin) / dx)) + 1
    solid = (zmin + np.arange(nz)[None, None, :] * dx) < terrain[..., None]
    phi = (distance_transform_edt(~solid) - distance_transform_edt(solid)) * dx
    origin = np.array([*xy0, zmin], dtype="f")
    geo = dict(
        phi=phi.astype("f"),
        terrain=terrain,
        terrain_count=count,
        origin=origin,
        dx=dx,
        plane=np.array([0, 0, float(terrain.min())], dtype="f"),
        bounds=bounds,
        positions=np.empty((0, 3), dtype="f"),
        temperature=np.empty(0, dtype="f"),
        velocity=np.empty((0, 3), dtype="f"),
        collision_points=len(points),
        seed=seed,
    )
    return geo


def height_at(geo, xy):
    return map_coordinates(
        geo["terrain"],
        ((np.asarray(xy) - geo["origin"][:2]) / geo["dx"]).T,
        order=1,
        mode="nearest",
    )
