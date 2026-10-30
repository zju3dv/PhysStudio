"""Plane depth fitting, cut regularization and plane-guided inpainting."""

import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares


def fit_depth_plane(points, rays, camera_center, depths):
    """Paper Eq.18–19, robust initialization followed by depth-space fitting."""
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), bool)
    for _ in range(240):
        tri = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        n /= max(np.linalg.norm(n), 1e-12)
        mask = np.abs((points - tri[0]) @ n) < 0.04
        if mask.sum() > best.sum():
            best = mask
    center = points[best].mean(0)
    _, _, vt = np.linalg.svd(points[best] - center, full_matrices=False)
    n = vt[-1]
    if n @ (camera_center - center) < 0:
        n = -n

    def residual(q):
        n1 = q[:3] / np.linalg.norm(q[:3])
        pred = -(camera_center @ n1 + q[3]) / (rays[best] @ n1)
        return pred - depths[best]

    q = least_squares(residual, np.r_[n, -n @ center], loss="soft_l1", f_scale=0.03).x
    return q[:3] / np.linalg.norm(q[:3]), q[3], best


def fit_vase_plane(path, data):
    import torch
    from gsplat import rasterization
    from rainy_gs.scene import Camera, save_image

    c = json.loads((path.parents[2] / "cameras.json").read_text())[0]
    width, height = 800, 600
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = c["rotation"], c["position"]

    def t(x):
        return torch.as_tensor(x, dtype=torch.float32, device="cuda")

    k = np.array(
        [
            [c["fx"] * width / c["width"], 0, width / 2],
            [0, c["fy"] * height / c["height"], height / 2],
            [0, 0, 1],
        ]
    )
    cam = Camera(t(np.linalg.inv(c2w)), t(k), width, height)
    im, al, _ = rasterization(
        means=t(data["means"]),
        quats=None,
        scales=None,
        covars=t(data["axes"] @ data["axes"].transpose(0, 2, 1)),
        opacities=t(data["opacity"]),
        colors=t(data["sh"]),
        viewmats=cam.view[None],
        Ks=cam.K[None],
        width=width,
        height=height,
        packed=False,
        render_mode="RGB+ED",
        sh_degree=3,
    )
    xyz = cam.world_points(im[0, ..., 3]).cpu().numpy()
    yy, xx = np.mgrid[:height, :width]
    # Explicit user-style planar ROI: visible wooden deck below the vase.
    mask = (yy > height * 0.87) & (al[0, ..., 0].cpu().numpy() > 0.99)
    mask &= (xx % 4 == 0) & (yy % 4 == 0)
    rays = (
        np.stack(
            [(xx - k[0, 2]) / k[0, 0], (yy - k[1, 2]) / k[1, 1], np.ones_like(xx)], -1
        )
        @ c2w[:3, :3].T
    )
    n, d, inliers = fit_depth_plane(
        xyz[mask], rays[mask], c2w[:3, 3], im[0, ..., 3].cpu().numpy()[mask]
    )
    out = Path("outputs/enlivening_gs/inspection")
    out.mkdir(parents=True, exist_ok=True)
    save_image(out / "vase_source.png", im[0, ..., :3] + 1 - al[0])
    return dict(
        normal=n.tolist(),
        offset=float(d),
        depth_samples=int(mask.sum()),
        inlier_fraction=float(inliers.mean()),
        roi="bottom 13 percent of camera 0",
    )


def segmentation_loss(means, axes, normal, offset):
    """Differentiable paper Eq.20–21 on intersecting splats only (Torch)."""
    import torch

    reach = (3 * torch.einsum("i,nij->nj", normal, axes)).abs().amax(-1)
    return torch.relu(reach - (means @ normal + offset).abs()).square().mean()


def unproject_patch(rgb, mask, view, intrinsic, normal, offset):
    """Lift an inpainted RGB patch to the fitted plane (supplement Sec.1.3)."""
    yy, xx = np.where(mask)
    c2w = np.linalg.inv(view)
    rays = (
        np.stack([xx + 0.5, yy + 0.5, np.ones_like(xx)], -1)
        @ np.linalg.inv(intrinsic).T
        @ c2w[:3, :3].T
    )
    distance = -(c2w[:3, 3] @ normal + offset) / (rays @ normal)
    valid = distance > 0
    xyz = c2w[:3, 3] + distance[:, None] * rays
    return xyz[valid], rgb[yy[valid], xx[valid]]
