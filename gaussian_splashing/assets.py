"""Pretrained Gaussian objects, collision sampling, and orbit cameras."""

from physstudio.paths import data_path
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData
from scipy import ndimage
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.cuda._wrapper import spherical_harmonics
from rainy_gs.scene import Camera, quaternion_matrix


class Object:
    def __init__(self, name, height=1.25, device="cuda:0"):
        self.name, self.device = name, device
        self.path = next(data_path("gaussian_splashing", name).rglob("point_cloud.ply"))
        v = PlyData.read(str(self.path))["vertex"]

        def a(keys):
            return np.stack([v[k] for k in keys], -1).astype(np.float32)

        xyz = a(["x", "y", "z"])
        op = 1 / (1 + np.exp(-np.clip(a(["opacity"])[:, 0], -40, 40)))
        scales = np.exp(a([f"scale_{i}" for i in range(3)]))
        q = a([f"rot_{i}" for i in range(4)])
        dc = a([f"f_dc_{i}" for i in range(3)])[:, None]
        sh = np.concatenate(
            (
                dc,
                a([f"f_rest_{i}" for i in range(45)])
                .reshape(-1, 3, 15)
                .transpose(0, 2, 1),
            ),
            1,
        )
        keep = (
            (op > 0.035)
            & np.isfinite(xyz).all(-1)
            & (scales.max(-1) < np.max(np.ptp(xyz, axis=0)) * 0.15)
        )
        xyz, op, scales, q, sh = [z[keep] for z in (xyz, op, scales, q, sh)]
        self.source_rotation = (
            Rotation.from_euler("y", 25, degrees=True).as_matrix()
            if name == "bread"
            else np.eye(3)
        )
        xyz = xyz @ self.source_rotation.T
        lower, upper = np.quantile(xyz, [0.002, 0.998], axis=0)
        self.scale = height / (upper[2] - lower[2])
        self.source_center = (lower + upper) * 0.5
        xyz = (xyz - self.source_center) * self.scale
        keep = (np.abs(xyz) < (upper - lower) * self.scale * 0.56).all(-1)
        xyz, op, scales, q, sh = [z[keep] for z in (xyz, op, scales, q, sh)]

        def t(a):
            return torch.as_tensor(a, dtype=torch.float32, device=device)

        self.local = t(xyz)
        self.opacity = t(op)
        self.scales = t(scales * self.scale)
        self.source_quats = t(q)
        self.sh = t(sh)
        self.baserot = t(self.source_rotation)
        self.cov = quaternion_matrix(self.source_quats)
        self.cov = self.baserot @ self.cov
        self.cov = (self.cov * self.scales[:, None, :] ** 2) @ self.cov.transpose(
            -1, -2
        )
        self.lower = xyz.min(0)
        self.upper = xyz.max(0)
        self.count = len(xyz)
        self.body_center = np.array([0.0, 0.0, -self.lower[2] + 0.025], np.float32)

    def collision(self, dx=0.018):
        points = self.local.cpu().numpy()
        origin = points.min(0) - dx * 6
        dims = np.ceil((points.max(0) - origin) / dx).astype(int) + 7
        occ = np.zeros(dims, bool)
        ijk = np.floor((points - origin) / dx).astype(int)
        occ[tuple(ijk.T)] = True
        # Seal splat sampling gaps before filling the enclosed interior.
        occ = ndimage.binary_closing(occ, iterations=2)
        occ = ndimage.binary_fill_holes(occ)
        occ = ndimage.binary_dilation(occ, iterations=1)
        sdf = (
            ndimage.distance_transform_edt(~occ) - ndimage.distance_transform_edt(occ)
        ) * dx
        filled = (np.argwhere(occ) + 0.5) * dx + origin
        # Samples on the actual occupancy, not a solid object bounding box.
        samples = filled[:: max(1, len(filled) // 3000)]
        self.volume = occ.sum() * dx**3
        self.collision_samples = samples.astype(np.float32)
        return sdf.astype(np.float32), origin.astype(np.float32), dx, filled

    @torch.no_grad()
    def render(self, camera, center, quat, opacity=1.0):
        center = torch.as_tensor(center, device=self.device, dtype=torch.float32)
        # Simulator quaternions use xyzw; gsplat uses wxyz.
        q = torch.as_tensor(quat, device=self.device, dtype=torch.float32)[
            [3, 0, 1, 2]
        ][None]
        rot = quaternion_matrix(q)[0]
        means = self.local @ rot.T + center
        cov = rot @ self.cov @ rot.T
        directions = F.normalize(means - camera.center, dim=-1) @ rot @ self.baserot
        colors = (spherical_harmonics(3, directions, self.sh) + 0.5).clamp_min(0.0)
        image, alpha, _ = rasterization(
            means=means,
            quats=None,
            scales=None,
            covars=cov,
            opacities=self.opacity * opacity,
            colors=colors,
            viewmats=camera.view[None],
            Ks=camera.K[None],
            width=camera.width,
            height=camera.height,
            render_mode="RGB+ED",
            packed=False,
            near_plane=0.02,
            far_plane=50.0,
        )
        return image[0], alpha[0, ..., 0]


def camera(
    width=960,
    height=640,
    azimuth=-58.0,
    elevation=26.0,
    radius=5.8,
    target=(0.0, 0.0, 0.5),
    device="cuda:0",
):
    az, el = np.deg2rad([azimuth, elevation])
    center = np.array(target) + radius * np.array(
        [np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]
    )
    forward = np.array(target) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.stack((right, down, forward), -1)
    c2w[:3, 3] = center
    focal = width * 0.94
    return Camera(
        torch.tensor(np.linalg.inv(c2w), device=device),
        torch.tensor(
            [[focal, 0, width * 0.5], [0, focal, height * 0.5], [0, 0, 1]],
            dtype=torch.float32,
            device=device,
        ),
        width,
        height,
    )
