"""Gaussian scene rendering with per-asset far-plane and background settings."""

import torch
from gsplat import rasterization
from rainy_gs.scene import Scene as BaseScene


class Scene(BaseScene):
    @torch.no_grad()
    def render(self, camera, colors=None, mask=None, mode="RGB+ED", view=None):
        if not any(k in self.config for k in ["far_plane", "background", "sky"]):
            return super().render(camera, colors, mask, mode, view)
        idx = slice(None) if mask is None else mask
        color = self.sh if colors is None else colors
        render_view = (camera.view if view is None else view) @ self.world_transform
        sky = None
        if "sky" in self.config and colors is None:
            settings = self.config["sky"]
            ray = (
                camera.world_points(
                    torch.ones((camera.height, camera.width), device=self.device)
                )
                - camera.center
            )
            ray = torch.nn.functional.normalize(ray, dim=-1)
            horizon = torch.exp(-ray[..., 2].clamp_min(0) / 0.25)
            top = ray.new_tensor(settings.get("zenith", [0.28, 0.52, 0.82]))
            bottom = ray.new_tensor(settings.get("horizon", [0.82, 0.81, 0.77]))
            sky = top + horizon[..., None] * (bottom - top)
            sun = torch.nn.functional.normalize(
                ray.new_tensor(settings.get("sun_direction", [-0.6, 1.0, 0.05])), dim=0
            )
            halo = (ray @ sun).clamp(0, 1).pow(160.0)
            sky = (sky + halo[..., None] * ray.new_tensor([0.22, 0.15, 0.07])).clamp(
                0, 1
            )
        background = (
            [0.0, 0.0, 0.0]
            if sky is not None
            else self.config.get("background", [0.55, 0.62, 0.67])
        )
        image, alpha, _ = rasterization(
            means=self.render_means[idx],
            quats=self.quats[idx],
            scales=self.scales[idx],
            opacities=self.opacity[idx],
            colors=color[idx],
            sh_degree=3 if colors is None else None,
            viewmats=render_view[None],
            Ks=camera.K[None],
            width=camera.width,
            height=camera.height,
            packed=False,
            near_plane=0.02,
            far_plane=self.config.get("far_plane", 100.0),
            backgrounds=torch.tensor([background], device=self.device)
            if colors is None
            else None,
            render_mode=mode,
        )
        image, alpha = image[0], alpha[0, ..., 0]
        if sky is not None:
            image[..., :3] += (1 - alpha[..., None]) * sky
            if image.shape[-1] == 4:
                image[..., 3] = torch.where(
                    alpha > 0.01, image[..., 3], self.config.get("far_plane", 100.0)
                )
        return image, alpha
