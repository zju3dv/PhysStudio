"""3DGS winter appearance, MPM surface illumination and depth-aware snowfall."""

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import torch
from .geometry import winter_material
from .surface import SnowSurface


class WinterRenderer:
    def __init__(self, scene, mpm, air, geo, scale, strength=1.0, haze=0.012):
        self.scene, self.mpm, self.air, self.geo = scene, mpm, air, geo
        self.scale = scale
        self.haze = haze
        dev = scene.device
        self.sh, self.coverage = winter_material(scene, scale, strength)
        positions = mpm.state()["x"]
        # Reuse low-frequency illumination recorded in the source reconstruction
        # for contact contrast. This is appearance transfer, not recovered light.
        valid = (scene.opacity > 0.55) & (scene.scales.max(-1).values * scale < 0.4)
        source_pos = (scene.means[valid] * scale).cpu().numpy()
        dc = (scene.sh[valid, 0] * 0.2820947918 + 0.5).clamp(0, 1).cpu().numpy()
        distance, ids = cKDTree(source_pos).query(positions, k=8, workers=4)
        weight = 1 / (distance + 0.03) ** 2
        weight /= weight.sum(-1, keepdims=True)
        local = (dc[ids] * weight[..., None]).sum(1)
        luminance = (local * np.array([0.2126, 0.7152, 0.0722])).sum(-1)
        shade = 0.46 + 0.44 * np.clip(luminance, 0.025, 1) ** 0.3
        # Nearby overhanging structure supplies soft ambient contact shading.
        occ = np.zeros(len(positions))
        for h in (0.10, 0.22, 0.40, 0.70):
            q = positions.copy()
            q[:, 2] += h
            sd = map_coordinates(
                geo["phi"], ((q - geo["origin"]) / geo["dx"]).T, order=1, mode="nearest"
            )
            occ += np.exp(-np.maximum(sd, 0) / 0.12) / 4
        shade *= 1 - 0.20 * occ
        self.pack_light = torch.full((mpm.capacity,), 0.83, device=dev)
        self.pack_light[: len(shade)] = torch.tensor(
            shade, dtype=torch.float32, device=dev
        )
        self.surface = SnowSurface(mpm, geo, self.pack_light)

    @torch.no_grad()
    def render(self, camera, time, flakes=True, pack=True):
        original = self.scene.sh
        try:
            self.scene.sh = self.sh
            rgbd, _ = self.scene.render(camera)
        finally:
            self.scene.sh = original
        if pack:
            return self.surface.render(
                rgbd, camera, self.air, self.scale, flakes, self.haze
            )
        return rgbd[..., :3].clamp(0, 1)
