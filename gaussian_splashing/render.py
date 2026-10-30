"""Gaussian fluid buffers, depth-volume normals, Fresnel and Beer transmission.

Surface shading is evaluated per pixel after Gaussian splatting. This uses the
paper's depth-volume normal alternative, rather than retraining GaussianShader
materials. Fluid thickness uses low-opacity splat optical depth.
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization
from rainy_gs.scene import quaternion_matrix
from .anisotropy import Anisotropy


def sample(im, grid):
    return F.grid_sample(
        im.permute(2, 0, 1)[None],
        grid[None],
        align_corners=False,
        padding_mode="border",
    )[0].permute(1, 2, 0)


def blur(im, k=5):
    if im.ndim == 2:
        return F.avg_pool2d(im[None, None], k, 1, k // 2)[0, 0]
    return F.avg_pool2d(im.permute(2, 0, 1)[None], k, 1, k // 2)[0].permute(1, 2, 0)


def smooth_depth(depth, alpha, spacing, passes=9):
    valid = (alpha > 0.08).float()
    d = depth
    for it in range(passes):
        total = torch.zeros_like(d)
        norm = torch.zeros_like(d)
        stride = 1 if it < 2 else 2 if it < 5 else 3
        for dy, dx in (
            (0, 0),
            (-stride, 0),
            (stride, 0),
            (0, -stride),
            (0, stride),
            (-stride, -stride),
            (stride, stride),
            (-stride, stride),
            (stride, -stride),
        ):
            other = torch.roll(d, (dy, dx), (0, 1))
            v = torch.roll(valid, (dy, dx), (0, 1))
            w = (
                v
                * torch.exp(-(((other - d) / (spacing * 3.5)) ** 2))
                * (0.65 if dx or dy else 1.0)
            )
            total += other * w
            norm += w
        d = total / norm.clamp_min(1e-6)
    return d


def environment(direction):
    z = direction[..., 2:3]
    sky = direction.new_tensor([0.42, 0.59, 0.73]) * (0.65 + 0.35 * z.clamp(0, 1))
    ground = direction.new_tensor([0.24, 0.25, 0.27])
    color = ground * (1 - (z * 3 + 0.5).clamp(0, 1)) + sky * (z * 3 + 0.5).clamp(0, 1)
    # Broad studio panels create coherent moving highlights on the liquid.
    panel = torch.exp(
        -(((direction[..., 0] - 0.12) / 0.35) ** 8)
        - ((direction[..., 1] + 0.40) / 0.16) ** 8
    )
    panel *= torch.sigmoid((direction[..., 2] - 0.35) * 30)
    strip = torch.exp(
        -(((direction[..., 0] + 0.65) / 0.07) ** 2)
        - ((direction[..., 1] - 0.15) / 0.48) ** 8
    )
    return color + panel[..., None] * 2.5 + strip[..., None] * 1.1


class Renderer:
    def __init__(self, spacing, bounds, device="cuda:0"):
        self.spacing = spacing
        self.bounds = bounds
        self.device = device
        self.anisotropy = Anisotropy(spacing, device)

    def splat(
        self, camera, x, opacity, scales=None, colors=None, mode="RGB+ED", covars=None
    ):
        n = len(x)
        q = torch.zeros((n, 4), device=self.device)
        q[:, 0] = 1.0
        if scales is None:
            scales = torch.full((n, 3), self.spacing * 0.68, device=self.device)
        if colors is None:
            colors = torch.zeros((n, 3), device=self.device)
        if not isinstance(opacity, torch.Tensor):
            opacity = torch.full((n,), opacity, device=self.device)
        im, alpha, _ = rasterization(
            means=x,
            quats=q if covars is None else None,
            scales=scales if covars is None else None,
            covars=covars,
            opacities=opacity,
            colors=colors,
            viewmats=camera.view[None],
            Ks=camera.K[None],
            width=camera.width,
            height=camera.height,
            render_mode=mode,
            packed=False,
            near_plane=0.02,
            far_plane=50.0,
        )
        return im[0], alpha[0, ..., 0]

    def background(self, camera, obj, center, quat, object_alpha=1.0):
        h, w = camera.height, camera.width
        ones = torch.ones((h, w), device=self.device)
        rays = camera.world_points(ones) - camera.center
        depth = (-camera.center[2] / rays[..., 2]).clamp(0.01, 100.0)
        pos = camera.center + rays * depth[..., None]
        grid = ((torch.floor(pos[..., 0] * 3) + torch.floor(pos[..., 1] * 3)) % 2)[
            ..., None
        ]
        base = grid * pos.new_tensor([0.31, 0.34, 0.37]) + (1 - grid) * pos.new_tensor(
            [0.39, 0.42, 0.44]
        )
        # Fine tile grout provides a cue for refraction, without high contrast.
        edge = (
            torch.minimum(
                torch.remainder(pos[..., :2] * 3, 1.0),
                1 - torch.remainder(pos[..., :2] * 3, 1.0),
            )
            .min(-1)
            .values
        )
        base *= 1 - 0.10 * torch.exp(-(edge[..., None] ** 2) / 0.0003)
        dist = torch.linalg.norm(pos[..., :2], dim=-1)
        base = (
            base * (1 - (dist / 15).clamp(0, 0.6)[..., None])
            + pos.new_tensor([0.51, 0.55, 0.59]) * (dist / 15).clamp(0, 0.6)[..., None]
        )
        lo, hi = self.bounds
        bx = torch.minimum((pos[..., 0] - lo[0]).abs(), (pos[..., 0] - hi[0]).abs())
        by = torch.minimum((pos[..., 1] - lo[1]).abs(), (pos[..., 1] - hi[1]).abs())
        inx = (pos[..., 0] > lo[0] - 0.015) & (pos[..., 0] < hi[0] + 0.015)
        iny = (pos[..., 1] > lo[1] - 0.015) & (pos[..., 1] < hi[1] + 0.015)
        rim = ((bx < 0.012) & iny) | ((by < 0.012) & inx)
        base = torch.where(rim[..., None], base * 0.45, base)
        if object_alpha > 0:
            # Directional soft contact shadow, using the actual Gaussian object.
            q = torch.as_tensor(quat, dtype=torch.float32, device=self.device)[
                [3, 0, 1, 2]
            ][None]
            rot = quaternion_matrix(q)[0]
            x = obj.local @ rot.T + torch.as_tensor(center, device=self.device)
            light = x.new_tensor([-0.5, -0.35, 1.0])
            projection = torch.eye(3, device=self.device) - light[
                :, None
            ] * x.new_tensor([[0.0, 0.0, 1.0]])
            shadow = x @ projection.T
            shadow[:, 2] = 0.002
            # Floor shadow resolves silhouette; opacity softens high splat density.
            cov = (
                projection @ rot @ obj.cov @ rot.T @ projection.T
                + torch.eye(3, device=self.device)[None] * 0.00008
            )
            sh, sa, _ = rasterization(
                means=shadow,
                quats=None,
                scales=None,
                covars=cov,
                opacities=obj.opacity * 0.16,
                colors=torch.zeros((len(x), 3), device=self.device),
                viewmats=camera.view[None],
                Ks=camera.K[None],
                width=w,
                height=h,
                packed=False,
                render_mode="RGB",
            )
            base *= 1 - blur(sa[0, ..., 0], 9)[..., None] * 0.40 * object_alpha
            im, alpha = obj.render(camera, center, quat, object_alpha)
            base = im[..., :3] + base * (1 - alpha[..., None])
            depth = torch.where(alpha > 0.5, im[..., 3], depth)
        return base, depth

    @torch.no_grad()
    def render(
        self, camera, obj, state, object_alpha=1.0, fluid_alpha=1.0, specular=True
    ):
        bg, bgdepth = self.background(
            camera, obj, state["body_center"], state["body_quat"], object_alpha
        )
        if fluid_alpha <= 0:
            return bg, dict(background=bg)
        x = torch.as_tensor(state["positions"], device=self.device)
        v = torch.as_tensor(state["velocities"], device=self.device)
        covars = self.anisotropy(x)
        im, alpha = self.splat(camera, x, 0.93, covars=covars)
        depth = smooth_depth(im[..., 3], alpha, self.spacing) - self.spacing * 0.42
        # Low-opacity pass approximates additive Gaussian optical thickness.
        _, thin = self.splat(camera, x, 0.08, mode="RGB", covars=covars)
        thick = -torch.log((1 - thin).clamp_min(0.0001)) * self.spacing * 1.25 / 0.08
        world = camera.world_points(depth)
        dx = torch.roll(world, -1, 1) - torch.roll(world, 1, 1)
        dy = torch.roll(world, -1, 0) - torch.roll(world, 1, 0)
        n = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        view = F.normalize(camera.center - world, dim=-1)
        n = torch.where((n * view).sum(-1, keepdim=True) < 0, -n, n)
        # Mild normal filtering suppresses individual particle facets.
        valid = (alpha > 0.15).float()
        n = F.normalize(
            blur(n * valid[..., None], 5) / blur(valid, 5).clamp_min(1e-5)[..., None],
            dim=-1,
        )
        nv = (n * view).sum(-1).clamp(0, 1)
        refl = 2 * nv[..., None] * n - view
        env = environment(refl)
        h, w = camera.height, camera.width
        yy, xx = torch.meshgrid(
            torch.arange(h, device=self.device) + 0.5,
            torch.arange(w, device=self.device) + 0.5,
            indexing="ij",
        )
        screen = torch.stack((xx / w * 2 - 1, yy / h * 2 - 1), -1)
        ncam = n @ camera.view[:3, :3].T
        # Refraction is thickness-aware and depth-tested against foreground solids.
        offset = (
            ncam[..., :2]
            * (thick.clamp(max=0.8) / depth.clamp_min(0.2))[..., None]
            * 0.18
        )
        refracted = sample(bg, (screen + offset).clamp(-1, 1))
        refractdepth = sample(bgdepth[..., None], (screen + offset).clamp(-1, 1))[
            ..., 0
        ]
        refracted = torch.where((refractdepth < depth - 0.03)[..., None], bg, refracted)
        absorb = torch.exp(-thick[..., None] * x.new_tensor([1.5, 0.43, 0.20]))
        body = refracted * absorb + x.new_tensor([0.045, 0.24, 0.29]) * (1 - absorb)
        fresnel = 0.0204 + 0.9796 * (1 - nv) ** 5
        if not specular:
            fresnel = fresnel * 0
        water = body * (1 - fresnel[..., None]) + env * fresnel[..., None]
        light = F.normalize(x.new_tensor([-0.5, -0.4, 1.0]), dim=0)
        half = F.normalize(light + view, dim=-1)
        highlight = (n * half).sum(-1).clamp(0, 1) ** 180
        if specular:
            water += highlight[..., None] * 0.72
        # Keep world particle visibility and background depth coupled.
        visible = torch.sigmoid((bgdepth - depth - 0.005) * 200)
        coverage = (alpha / 0.80).clamp(0, 1) * visible * fluid_alpha
        color = water * coverage[..., None] + bg * (1 - coverage[..., None])
        # Aerated fast/sparse particles become a separate foam/spray splat layer.
        density = torch.as_tensor(state["density"], device=self.device)
        speed = torch.linalg.norm(v, dim=-1)
        foam = ((speed - 1.1) / 2).clamp(0, 0.6) * ((0.90 - density) / 0.35).clamp(0, 1)
        active = foam > 0.025
        if active.any():
            fim, fa = self.splat(
                camera,
                x[active],
                foam[active] * 0.25,
                scales=torch.full(
                    (int(active.sum()), 3), self.spacing * 0.38, device=self.device
                ),
                colors=torch.full((int(active.sum()), 3), 0.92, device=self.device),
            )
            fvis = (fim[..., 3] < bgdepth) & (fim[..., 3] < depth + self.spacing * 1.8)
            color = (
                color * (1 - (fa * fvis)[..., None]) + fim[..., :3] * fvis[..., None]
            )
        return color.clamp(0, 1), dict(
            background=bg,
            depth=depth,
            normals=n * 0.5 + 0.5,
            thickness=thick,
            alpha=coverage,
        )
