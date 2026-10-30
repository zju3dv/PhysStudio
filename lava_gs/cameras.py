"""Camera paths that include the elevated inlet and downstream accumulation."""

import numpy as np
import torch
from rainy_gs.scene import Camera


def look_at(eye, target, up):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack((right, down, forward), 1)
    c2w[:3, 3] = eye
    return c2w


def cascade_camera(progress, width, device="cuda:0", settings=None):
    settings = settings or {}
    u = progress * progress * (3.0 - 2.0 * progress)
    target = np.array(settings.get("target", [-0.55, 0.10, 0.32]), dtype=float)
    target += u * np.array(settings.get("target_offset", [0.0, 0.0, 0.0]))
    if "orbit" in settings:
        orbit = settings["orbit"]
        angle = np.deg2rad(
            (1.0 - u) * orbit["azimuth_start"] + u * orbit["azimuth_end"]
        )
        radius = (1.0 - u) * orbit["radius_start"] + u * orbit["radius_end"]
        elevation = (1.0 - u) * orbit["height_start"] + u * orbit["height_end"]
        eye = target + np.array(
            [radius * np.cos(angle), radius * np.sin(angle), elevation]
        )
    else:
        start = np.array(settings.get("eye", [0.00, -3.30, 1.04]))
        eye = start + u * np.array(settings.get("offset", [0.18, 0.12, 0.06]))
    inverse = look_at(eye, target, np.array([0.0, 0.0, 1.0]))
    height = int(width * 0.625) // 2 * 2
    focal = height / (2 * np.tan(np.deg2rad(settings.get("fov", 43.0)) / 2))
    intrinsics = [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]
    return Camera(
        torch.tensor(np.linalg.inv(inverse), dtype=torch.float32, device=device),
        torch.tensor(intrinsics, dtype=torch.float32, device=device),
        width,
        height,
    )
