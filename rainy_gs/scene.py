"""Released FieryGS cameras/PGSR splats; RGB-depth and planar-reflection buffers."""

from physstudio.paths import data_path
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
import torch
import torch.nn.functional as F
from gsplat import rasterization


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, torch.Tensor):
        image = image.detach().clamp(0, 1).cpu().numpy()
    Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255 + 0.5)).save(path)


def quaternion_matrix(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(-1, 3, 3)


@dataclass
class Camera:
    view: torch.Tensor
    K: torch.Tensor
    width: int
    height: int

    @property
    def center(self):
        return torch.linalg.inv(self.view)[:3, 3]

    def world_points(self, depth):
        yy, xx = torch.meshgrid(
            torch.arange(self.height, device=depth.device) + 0.5,
            torch.arange(self.width, device=depth.device) + 0.5,
            indexing="ij",
        )
        rays = torch.stack(
            (
                (xx - self.K[0, 2]) / self.K[0, 0],
                (yy - self.K[1, 2]) / self.K[1, 1],
                torch.ones_like(xx),
            ),
            -1,
        )
        xyz = rays * depth[..., None]
        inverse = torch.linalg.inv(self.view)
        return xyz @ inverse[:3, :3].T + inverse[:3, 3]


class Scene:
    def __init__(self, path, device="cuda:0", config=None):
        self.path, self.device = Path(path), device
        self.config = config or {}
        self.ply = next(self.path.glob("point_cloud/iteration_30000/*point_cloud.ply"))
        vertex = PlyData.read(str(self.ply))["vertex"]

        def tensor(keys):
            return torch.tensor(
                np.stack([vertex[k] for k in keys], -1).copy(),
                dtype=torch.float32,
                device=device,
            )

        self.means = tensor(["x", "y", "z"])
        self.scales = (
            tensor(["scale_" + str(i) for i in range(3)]).exp().clamp_min(1e-7)
        )
        self.quats = F.normalize(tensor(["rot_" + str(i) for i in range(4)]), dim=-1)
        self.opacity = tensor(["opacity"])[:, 0].sigmoid()
        dc = tensor(["f_dc_" + str(i) for i in range(3)])[:, None]
        # A fixed world-space vegetation proxy, derived from the DC color.
        # It is an appearance cue, not an externally provided material label.
        base_color = (dc[:, 0] * 0.28209479177387814 + 0.5).clamp(0, 1)
        red, green, blue = base_color.unbind(-1)
        excess_green = (green - 0.5 * (red + blue)) / (base_color.sum(-1) + 1e-4)
        vegetation = ((excess_green - 0.055) / 0.075).clamp(0, 1)
        vegetation *= ((green - torch.maximum(red, blue)) / 0.025).clamp(0, 1)
        self.vegetation = vegetation.square() * (3 - 2 * vegetation)
        rest = (
            tensor(["f_rest_" + str(i) for i in range(45)])
            .reshape(-1, 3, 15)
            .transpose(1, 2)
        )
        self.sh = torch.cat((dc, rest), 1)
        rotation = quaternion_matrix(self.quats)
        self.normals = rotation.gather(
            2, self.scales.argmin(-1)[:, None, None].expand(-1, 3, 1)
        )[:, :, 0]
        # Keep source splats and SH directions in their original coordinate
        # frame. Only simulation geometry/cameras use the calibrated frame.
        self.render_means = self.means
        self.world_transform = torch.tensor(
            self.config.get("world_transform", np.eye(4).tolist()),
            device=device,
            dtype=torch.float32,
        )
        rot = self.world_transform[:3, :3]
        if not (
            torch.allclose(rot.T @ rot, torch.eye(3, device=device), atol=1e-5)
            and abs(float(torch.det(rot)) - 1.0) < 1e-5
            and torch.allclose(
                self.world_transform[3], rot.new_tensor([0.0, 0.0, 0.0, 1.0])
            )
        ):
            raise ValueError(
                "world_transform must be a rigid proper rotation and translation"
            )
        if "world_transform" in self.config:
            self.means = self.means @ rot.T + self.world_transform[:3, 3]
            self.normals = self.normals @ rot.T
        self.normals *= torch.where(self.normals[:, 2:3] < 0, -1.0, 1.0)
        self.cameras = json.loads((self.path / "cameras.json").read_text())
        self.ground = (
            torch.tensor(
                self.config["ground_plane"], device=device, dtype=torch.float32
            )
            if "ground_plane" in self.config
            else self.fit_ground()
        )

    def camera(self, index, width=1008):
        value = self.cameras[index]
        c2w = torch.eye(4, device=self.device)
        c2w[:3, :3] = torch.tensor(value["rotation"], device=self.device)
        c2w[:3, 3] = torch.tensor(value["position"], device=self.device)
        c2w = self.world_transform @ c2w
        height = round(value["height"] * width / value["width"] / 2) * 2
        k = torch.tensor(
            [
                [value["fx"] * width / value["width"], 0, width / 2],
                [0, value["fy"] * height / value["height"], height / 2],
                [0, 0, 1.0],
            ],
            device=self.device,
        )
        return Camera(torch.linalg.inv(c2w), k, width, height)

    def fit_ground(self):
        # Fit the low horizontal support near the playground, not the global
        # covariance of vegetation, cameras and distant buildings.
        fit = self.config.get("ground_fit", {})
        bounds = self.means.new_tensor(
            fit.get("bounds", [-3.5, -3.5, -0.75, 3.5, 3.5, -0.25])
        )
        mask = (
            ((self.means > bounds[:3]) & (self.means < bounds[3:])).all(-1)
            & (self.normals[:, 2] > fit.get("normal_z", 0.94))
            & (self.opacity > 0.5)
        )
        points = self.means[mask][::4].double()
        if len(points) < 100:
            raise ValueError("Need a scene-specific ground initialization")
        design = torch.cat((points[:, :2], torch.ones_like(points[:, :1])), -1)
        coef = torch.linalg.lstsq(design, points[:, 2]).solution
        for _ in range(6):
            residual = (design @ coef - points[:, 2]).abs()
            weight = (0.015 / residual.clamp_min(0.015)).square()
            coef = torch.linalg.lstsq(
                design * weight[:, None], points[:, 2] * weight
            ).solution
        return coef.float()

    def ground_height(self, xy):
        return xy @ self.ground[:2] + self.ground[2]

    def sky_weight(self, rgb, world, direction):
        height = world[..., 2] - self.ground_height(world[..., :2])
        red, green, blue = rgb.unbind(-1)
        chroma = ((blue - red - 0.08) / 0.12).clamp(0, 1) * (
            (blue - green - 0.04) / 0.08
        ).clamp(0, 1)
        chroma *= ((blue - 0.35) / 0.15).clamp(0, 1)
        elevated = torch.maximum(
            ((height - 1.0) / 2.0).clamp(0, 1),
            ((direction[..., 2] - 0.01) / 0.06).clamp(0, 1),
        )
        return elevated * chroma

    @torch.no_grad()
    def render(self, camera, colors=None, mask=None, mode="RGB+ED", view=None):
        idx = slice(None) if mask is None else mask
        color = self.sh if colors is None else colors
        render_view = (camera.view if view is None else view) @ self.world_transform
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
            far_plane=100.0,
            backgrounds=torch.tensor([[0.55, 0.62, 0.67]], device=self.device)
            if colors is None
            else None,
            render_mode=mode,
        )
        return image[0], alpha[0, ..., 0]

    @torch.no_grad()
    def buffers(self, camera, reflection_levels=(0.0,)):
        rgbd, alpha = self.render(camera)
        rgb, depth = rgbd[..., :3], rgbd[..., 3]
        normal, _ = self.render(camera, colors=self.normals, mode="RGB")
        vegetation, _ = self.render(camera, colors=self.vegetation[:, None], mode="RGB")
        normal = F.normalize(normal, dim=-1)
        world = camera.world_points(depth)
        direction = F.normalize(
            camera.world_points(torch.ones_like(depth)) - camera.center, dim=-1
        )
        # Reflect the camera through the fitted support plane. This includes
        # objects outside the original image, unlike a screen-only reflection.
        n = F.normalize(torch.cat((-self.ground[:2], self.ground.new_ones(1))), dim=0)
        mirrors, mirror_skies = [], []
        for level in reflection_levels:
            d = -(self.ground[2] + level) / torch.sqrt(
                1 + self.ground[:2].square().sum()
            )
            reflection = torch.eye(4, device=self.device)
            reflection[:3, :3] -= 2 * n[:, None] * n[None]
            reflection[:3, 3] = -2 * d * n
            margin = 0.035 if len(reflection_levels) == 1 else 0.008
            above = (
                self.means[:, 2] - self.ground_height(self.means[:, :2])
                > level + margin
            )
            mirror_view = camera.view @ reflection
            mirrored, _ = self.render(
                camera, mask=above, mode="RGB+ED", view=mirror_view
            )
            mirror_camera = Camera(mirror_view, camera.K, camera.width, camera.height)
            mirror_world = mirror_camera.world_points(mirrored[..., 3])
            mirror_direction = F.normalize(
                mirror_camera.world_points(torch.ones_like(depth))
                - mirror_camera.center,
                dim=-1,
            )
            mirror_skies.append(
                self.sky_weight(mirrored[..., :3], mirror_world, mirror_direction)
            )
            mirrors.append(mirrored[..., :3].clamp(0, 1))
        result = dict(
            rgb=rgb.clamp(0, 1),
            depth=depth,
            alpha=alpha,
            world=world,
            normal=normal,
            reflection=mirrors[0],
            vegetation=vegetation[..., 0].clamp(0, 1),
            sky=self.sky_weight(rgb, world, direction),
            reflection_sky=mirror_skies[0],
        )
        if len(mirrors) > 1:
            result.update(
                reflection_stack=torch.stack(mirrors),
                reflection_sky_stack=torch.stack(mirror_skies),
                reflection_levels=torch.tensor(reflection_levels, device=self.device),
            )
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scene", type=Path, default=data_path("rainy_gs/fierygs/Playground")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=1008)
    p.add_argument("--camera", type=int, default=61)
    p.add_argument("--gallery", action="store_true")
    p.add_argument("--scene-config", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config = json.loads(args.scene_config.read_text()) if args.scene_config else None
    start = time.monotonic()
    scene = Scene(args.scene, config=config)
    print(
        "Gaussians",
        len(scene.means),
        "ground z=ax+by+c",
        scene.ground.tolist(),
        flush=True,
    )
    camera = scene.camera(args.camera, args.width)
    buffers = scene.buffers(camera)
    for name in ("rgb", "reflection"):
        save_image(args.output / (name + ".png"), buffers[name])
    save_image(args.output / "normal.png", buffers["normal"] * 0.5 + 0.5)
    save_image(args.output / "depth.png", (buffers["depth"] / 20).clamp(0, 1))
    torch.save({k: v.cpu() for k, v in buffers.items()}, args.output / "buffers.pt")
    torch.save(
        dict(
            view=camera.view.cpu(),
            K=camera.K.cpu(),
            width=camera.width,
            height=camera.height,
            ground=scene.ground.cpu(),
        ),
        args.output / "camera.pt",
    )
    if args.gallery:
        indices = np.linspace(0, len(scene.cameras) - 1, 12).astype(int).tolist()
        tiles = []
        for index in indices:
            cam = scene.camera(index, 504)
            rgb, _ = scene.render(cam, mode="RGB")
            tile = Image.fromarray(
                (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            )
            ImageDraw.Draw(tile).text((8, 8), str(index), fill="white")
            tiles.append(tile)
        gallery = Image.new("RGB", (504 * 3, tiles[0].height * 4))
        for i, tile in enumerate(tiles):
            gallery.paste(tile, ((i % 3) * 504, (i // 3) * tile.height))
        gallery.save(args.output / "gallery.jpg")
    (args.output / "scene.json").write_text(
        json.dumps(
            dict(
                gaussians=len(scene.means),
                camera=args.camera,
                ground=scene.ground.tolist(),
                elapsed_seconds=time.monotonic() - start,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
