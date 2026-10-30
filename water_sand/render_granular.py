"""Material-attached Gaussian grains, reduced filtering, and wet granular shading."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage
from PIL import Image
import torch
import torch.nn.functional as F
import warp as wp

from .common import weights
from .render_gaussian import GaussianRenderer, surfaces
from rainy_gs.scene import Camera
from rainy_gs.demo import Video


@wp.kernel(enable_backward=False)
def deposit(
    x: wp.array(dtype=wp.vec3),
    water: wp.array(dtype=float),
    color: wp.array(dtype=wp.vec3),
    den: wp.array3d(dtype=float),
    wet: wp.array3d(dtype=float),
    pigment: wp.array3d(dtype=wp.vec3),
    volume: float,
):
    p = wp.tid()
    point = x[p] * 2.0
    base = wp.vec3i(
        int(wp.floor(point[0] - 0.5)),
        int(wp.floor(point[1] - 0.5)),
        int(wp.floor(point[2] - 0.5)),
    )
    f = point - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    wx, wy, wz = weights(f[0]), weights(f[1]), weights(f[2])
    for a in range(3):
        for b in range(3):
            for c in range(3):
                i, j, k = base[0] + a, base[1] + b, base[2] + c
                if (
                    i >= 0
                    and j >= 0
                    and k >= 0
                    and i < den.shape[0]
                    and j < den.shape[1]
                    and k < den.shape[2]
                ):
                    w = wx[a] * wy[b] * wz[c] * 8.0 * volume
                    wp.atomic_add(den, i, j, k, w)
                    wp.atomic_add(wet, i, j, k, w * water[p] / (volume * 0.293))
                    wp.atomic_add(pigment, i, j, k, w * color[p])


@wp.kernel(enable_backward=False)
def filter_fields(
    den: wp.array3d(dtype=float),
    wet: wp.array3d(dtype=float),
    col: wp.array3d(dtype=wp.vec3),
    od: wp.array3d(dtype=float),
    ow: wp.array3d(dtype=float),
    oc: wp.array3d(dtype=wp.vec3),
    axis: int,
    sigma: float,
):
    i, j, k = wp.tid()
    total, d, w = float(0.0), float(0.0), float(0.0)
    c = wp.vec3(0.0)
    for offset in range(-3, 4):
        p = wp.vec3i(i, j, k)
        p[axis] += offset
        if (
            p[0] >= 0
            and p[1] >= 0
            and p[2] >= 0
            and p[0] < den.shape[0]
            and p[1] < den.shape[1]
            and p[2] < den.shape[2]
        ):
            weight = wp.exp(-float(offset * offset) / (2.0 * sigma * sigma))
            total += weight
            d += weight * den[p[0], p[1], p[2]]
            w += weight * wet[p[0], p[1], p[2]]
            c += weight * col[p[0], p[1], p[2]]
    od[i, j, k] = d / total
    ow[i, j, k] = w / total
    oc[i, j, k] = c / total


def random_values(ids, count, salt):
    bits = ids.astype(np.uint32)[:, None] * np.uint32(747796405)
    bits = (
        bits
        + np.arange(count, dtype=np.uint32)[None] * np.uint32(2891336453)
        + np.uint32(salt)
    )
    bits = (bits ^ (bits >> 16)) * np.uint32(2246822519)
    bits = (bits ^ (bits >> 13)) * np.uint32(3266489917)
    bits ^= bits >> 16
    return (bits.astype(np.float64) / 4294967296.0).astype(np.float32)


def close_camera(width, height, device, scene="castle"):
    target = (
        np.array([2.12, 0.63, 0.65])
        if scene == "castle"
        else np.array([1.5, 0.85, 0.45])
    )
    eye = target + np.array([-1.45, -1.65, 0.85])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.stack([right, down, forward], -1)
    c2w[:3, 3] = eye
    focal = width / (2 * np.tan(np.deg2rad(43) * 0.5))
    cam = Camera(
        torch.tensor(np.linalg.inv(c2w), device=device),
        torch.tensor(
            [[focal, 0, width * 0.5], [0, focal, height * 0.5], [0, 0, 1]],
            dtype=torch.float32,
            device=device,
        ),
        width,
        height,
    )
    return cam, dict(eye=eye.tolist(), target=target.tolist(), focal=focal)


class GranularRenderer(GaussianRenderer):
    def __init__(
        self,
        cfg,
        rest,
        width,
        height,
        device="cuda:0",
        sigma=1.40,
        grains=20,
        suspension=0.10,
    ):
        super().__init__(cfg, rest, width, height, device)
        self.smoothing, self.grains = sigma, grains
        self.suspension = suspension
        if cfg["scene"] != "sediment":
            self.fields.colors.assign(
                np.tile(np.array([0.71, 0.43, 0.16], np.float32), (len(rest), 1))
            )
        self.base_count = 0
        self.cloud_start = 0
        self.dilute = None

    def splat(self, cam, x, cov, features, opacity=0.995, depth=True):
        if features.shape[-1] == 7:
            count = self.base_count
            base, ba = super().splat(
                cam,
                x[:count],
                cov[:count],
                features[:count],
                self.sand_opacity[:count],
                depth,
            )
            if len(x) == count:
                return base, ba
            direction = F.normalize(cam.center - x[count:], dim=-1)
            facing = ((features[count:, 3:6] * direction).sum(-1) * 5).clamp(0, 1)
            facing[self.cloud_start - count :] = 1.0
            # Cull individual grains before rasterization. Testing the blended
            # grain depth would let occluded rear grains hide visible front grains.
            camera_x = x[count:] @ cam.view[:3, :3].T + cam.view[:3, 3]
            projected = camera_x @ cam.K.T
            uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-6)
            uv = uv / uv.new_tensor([cam.width, cam.height]) * 2 - 1
            samples = F.grid_sample(
                torch.stack([base[..., -1], ba])[None],
                uv[None, None],
                align_corners=False,
                padding_mode="border",
            )[0, :, 0]
            visible = torch.where(
                samples[1] > 0.1,
                torch.sigmoid((samples[0] + self.dx * 0.45 - camera_x[:, 2]) * 500),
                1.0,
            )
            grain, ga = super().splat(
                cam,
                x[count:],
                cov[count:],
                features[count:],
                self.sand_opacity[count:] * facing * visible,
                depth,
            )
            out = base * (1 - ga[..., None]) + grain
            combined_alpha = ba + (1 - ba) * ga
            out[..., -1] = (
                base[..., -1] * ba * (1 - ga) + grain[..., -1] * ga
            ) / combined_alpha.clamp_min(1e-6)
            return out, combined_alpha
        return super().splat(cam, x, cov, features, opacity, depth)

    def background(self, cam, x, cov):
        return super().background(cam, x[: self.base_count], cov[: self.base_count])

    def sand_surface(self, state):
        f = self.fields
        f.den.zero_()
        f.wet.zero_()
        f.pigment.zero_()
        p = wp.array(state["positions"], dtype=wp.vec3)
        water = wp.array(state["moisture"], dtype=float)
        wp.launch(
            deposit,
            len(p),
            inputs=[
                p,
                water,
                f.colors,
                f.den,
                f.wet,
                f.pigment,
                1 / self.cfg["ppc"] ** 3,
            ],
        )
        for axis in range(3):
            wp.launch(
                filter_fields,
                f.den.shape,
                inputs=[
                    f.den,
                    f.wet,
                    f.pigment,
                    f.smooth_den,
                    f.smooth_wet,
                    f.smooth_pigment,
                    axis,
                    self.smoothing,
                ],
            )
            f.den, f.smooth_den = f.smooth_den, f.den
            f.wet, f.smooth_wet = f.smooth_wet, f.wet
            f.pigment, f.smooth_pigment = f.smooth_pigment, f.pigment
        den, wet, colors = f.den.numpy(), f.wet.numpy(), f.pigment.numpy()
        cell = self.dx * 0.5
        x, n, lattice = surfaces(den, 0.38, cell, self.origin)
        self.base_count = len(x)

        def interpolate(a, coordinates):
            if a.ndim == 4:
                return np.stack(
                    [
                        ndimage.map_coordinates(
                            a[..., k], coordinates.T, order=1, mode="nearest"
                        )
                        for k in range(3)
                    ],
                    -1,
                )
            return ndimage.map_coordinates(a, coordinates.T, order=1, mode="nearest")

        d = interpolate(den, lattice).clip(0.05)
        w = (interpolate(wet, lattice) / d).clip(0, 1)
        c = interpolate(colors, lattice) / d[:, None]
        light = np.array([-0.6, -0.4, 1.0], np.float32)
        light /= np.linalg.norm(light)
        trans = np.ones(len(x), np.float32)
        for step in range(1, 13):
            q = lattice + n * 1.7 + light * step * 2.0
            trans *= np.exp(-np.clip(interpolate(den, q) - 0.15, 0, 1) * 0.40)
        c *= (0.72 + 0.28 * trans)[:, None]
        base_cov = self.covariances(n, cell, 0.72, 0.13)
        opacity = np.full(len(x), 0.995, np.float32)
        grain_count = 0
        cloud_count = 0
        self.dilute = None
        keys = []
        if self.grains:
            grad = np.stack(np.gradient(den, cell), -1)
            parent_lattice = state["positions"] * 2
            pd = interpolate(den, parent_lattice)
            pg = interpolate(grad, parent_lattice)
            norm = np.linalg.norm(pg, axis=-1)
            spacing = self.dx / self.cfg["ppc"]
            distance = (pd - 0.38) / np.maximum(norm, 1e-7)
            active = (np.abs(distance) < spacing * 2.0) & (norm > 0.5) & (pd > 0.1)
            ids = np.flatnonzero(active)
            pn = -pg[ids] / norm[ids, None]
            center = (state["positions"][ids] - 3) * self.dx + pn * distance[ids, None]
            # Elastic polar rotation orients each patch; no elastic stretch is
            # interpreted as a total plastic deformation.
            u, _, vt = np.linalg.svd(state["deformation"][ids])
            rotation = u @ vt
            count = self.grains
            local = (
                np.stack(
                    [random_values(ids, count, s) - 0.5 for s in [19, 419, 919]], -1
                )
                * spacing
                * 1.45
            )
            offset = np.einsum("nij,nkj->nki", rotation, local)
            offset -= np.sum(offset * pn[:, None], axis=-1, keepdims=True) * pn[:, None]
            gx = (center[:, None] + offset).reshape(-1, 3)
            for _ in range(3):
                gl = (gx - self.origin) / cell
                gg = interpolate(grad, gl)
                gnlen = np.linalg.norm(gg, axis=-1).clip(1e-6)
                gn = -gg / gnlen[:, None]
                gd = interpolate(den, gl)
                correction = ((gd - 0.38) / gnlen).clip(-spacing * 0.7, spacing * 0.7)
                gx += gn * correction[:, None]
            radius = (0.0009 + 0.0009 * random_values(ids, count, 123)).reshape(-1)
            residual = np.abs(interpolate(den, (gx - self.origin) / cell) - 0.38)
            gx += gn * (radius * 0.60)[:, None]
            gl = (gx - self.origin) / cell
            density = interpolate(den, gl).clip(0.05)
            gc = interpolate(colors, gl) / density[:, None]
            gw = (interpolate(wet, gl) / density).clip(0, 1)
            variation = random_values(ids, count, 1619).reshape(-1)
            gc *= (0.66 + 0.60 * variation)[:, None]
            quartz = variation > 0.94
            gc[quartz] = 0.35 * gc[quartz] + 0.65 * np.array(
                [0.73, 0.65, 0.48], np.float32
            )
            gc[variation < 0.035] *= 0.45
            tangent = np.stack(
                [random_values(ids, count, s) - 0.5 for s in [621, 921, 1221]], -1
            ).reshape(-1, 3)
            tangent -= np.sum(tangent * gn, axis=-1, keepdims=True) * gn
            shading_n = gn + tangent * (1.0 - 0.45 * gw[:, None])
            shading_n /= np.linalg.norm(shading_n, axis=-1, keepdims=True).clip(1e-6)
            fade = np.clip(
                (spacing * 2.0 - np.abs(distance[ids])) / (spacing * 0.45), 0, 1
            )
            go = np.repeat(fade, count) * 0.93 * np.clip(1 - residual / 0.20, 0, 1)
            gcov = (
                self.covariances(gn, 1.0, 1.0, 0.75)
                * (self.tensor(radius * 0.48) ** 2)[:, None, None]
            )
            base_cov = torch.cat([base_cov, gcov])
            x = np.concatenate([x, gx])
            n = np.concatenate([n, shading_n])
            c = np.concatenate([c, gc])
            w = np.concatenate([w, gw])
            opacity = np.concatenate([opacity, go])
            grain_count = len(gx)
            keys.append(
                (
                    ids[:, None].astype(np.int64) * count + np.arange(count)[None]
                ).reshape(-1)
            )
            self.cloud_start = len(x)
            # Material points below the isosurface threshold still carry sand.
            # Retain them as deterministic moving grain clusters, rather than
            # dropping detached or suspended material during surface extraction.
            cloud_ids = np.flatnonzero(pd < 0.22)
            if len(cloud_ids):
                fade = np.clip((0.22 - pd[cloud_ids]) / 0.14, 0, 1)
                local = np.stack(
                    [
                        random_values(cloud_ids, count, s) - 0.5
                        for s in [4191, 5191, 6191]
                    ],
                    -1,
                )
                spread = spacing * (0.30 / np.maximum(pd[cloud_ids], 0.035)) ** (1 / 3)
                cx = ((state["positions"][cloud_ids] - 3) * self.dx)[
                    :, None
                ] + local * spread[:, None, None]
                cx = cx.reshape(-1, 3)
                cn = local.reshape(-1, 3)
                cn /= np.linalg.norm(cn, axis=-1, keepdims=True).clip(1e-6)
                cw = (
                    state["moisture"][cloud_ids] / (0.293 / self.cfg["ppc"] ** 3)
                ).clip(0, 1)
                cc = np.repeat(f.colors.numpy()[cloud_ids], count, axis=0)
                cc *= 0.65 + 0.55 * random_values(cloud_ids, count, 9011).reshape(-1, 1)
                cr = (0.0009 + 0.0009 * random_values(cloud_ids, count, 123)).reshape(
                    -1
                )
                ccov = self.eye3[None] * (self.tensor(cr * 0.48) ** 2)[:, None, None]
                base_cov = torch.cat([base_cov, ccov])
                x = np.concatenate([x, cx])
                n = np.concatenate([n, cn])
                c = np.concatenate([c, cc])
                w = np.concatenate([w, np.repeat(cw, count)])
                opacity = np.concatenate([opacity, np.repeat(fade * 0.90, count)])
                cloud_count = len(cx)
                keys.append(
                    (
                        (1 << 48)
                        + cloud_ids[:, None].astype(np.int64) * count
                        + np.arange(count)[None]
                    ).reshape(-1)
                )
                ph = ndimage.map_coordinates(
                    state["phi"], state["positions"][cloud_ids].T, order=1
                ).clip(0, 1)
                self.dilute = dict(
                    x=(state["positions"][cloud_ids] - 3) * self.dx,
                    fade=fade * ph,
                    colors=f.colors.numpy()[cloud_ids],
                    spread=spread,
                )
        else:
            self.cloud_start = len(x)
        self.sand_opacity = self.tensor(opacity)
        self.grain_keys = np.concatenate(keys) if keys else np.empty(0, np.int64)
        self.extra_stats = dict(
            surface_gaussians=self.base_count,
            grain_gaussians=grain_count + cloud_count,
            detached_grain_gaussians=cloud_count,
            smoothing_sigma_meters=float(self.smoothing * cell),
            grain_radius_meters=[0.0009, 0.0018],
        )
        return self.tensor(x), base_cov, self.tensor(n), self.tensor(c), self.tensor(w)

    def shade_sand(self, cam, image, alpha):
        normal = F.normalize(image[..., 3:6], dim=-1)
        moisture = (image[..., 6] / alpha.clamp_min(1e-6)).clamp(0, 1)
        albedo = image[..., :3] / alpha.clamp_min(1e-6)[..., None]
        world = cam.world_points(image[..., -1])
        view = F.normalize(cam.center - world, dim=-1)
        normal = torch.where((normal * view).sum(-1, keepdim=True) < 0, -normal, normal)
        # Explicit Gaussian microgeometry supplies the normal variation.
        # Differentiating an interpolated material G-buffer as a height field
        # introduced contour-like bands at curved coarse splats.
        light = F.normalize(self.tensor([-0.6, -0.4, 1.0]), dim=0)
        half = F.normalize(light + view, dim=-1)
        nl = (normal * light).sum(-1).clamp(0, 1)
        nv = (normal * view).sum(-1).clamp(0.01, 1)
        nh = (normal * half).sum(-1).clamp(0, 1)
        vh = (view * half).sum(-1).clamp(0, 1)
        roughness = 0.82 - 0.48 * moisture
        a2 = roughness**4
        distribution = a2 / (np.pi * (nh * nh * (a2 - 1) + 1) ** 2).clamp_min(1e-5)
        k = (roughness + 1) ** 2 / 8
        geometry = nl / (nl * (1 - k) + k) * nv / (nv * (1 - k) + k)
        fresnel = 0.035 + 0.965 * (1 - vh) ** 5
        spec = distribution * geometry * fresnel / (4 * nl * nv).clamp_min(0.01)
        sand = albedo * ((1 - 0.46 * moisture) * (0.33 + 0.67 * nl))[..., None]
        sand += (spec * nl * 0.65)[..., None] * self.tensor([1.0, 0.96, 0.88])
        return sand, normal

    def water_transmission(
        self, cam, state, thickness, depth, background_depth, refracted
    ):
        body = super().water_transmission(
            cam, state, thickness, depth, background_depth, refracted
        )
        if self.dilute is None or self.suspension == 0:
            return body
        d = self.dilute
        selected = d["fade"] > 0.03
        if not selected.any():
            return body
        sigma = self.tensor(d["spread"][selected] * 1.30)
        cov = self.eye3[None] * sigma[:, None, None] ** 2
        volume = (self.dx / self.cfg["ppc"]) ** 3
        # Geometric extinction of coarse sand, integrated over a Gaussian cloud.
        # Grains are also rendered explicitly. This residual optical weight
        # controls unresolved suspended material without obscuring those grains.
        peak_tau = (
            self.suspension
            * self.tensor(d["fade"][selected])
            * volume
            * 3
            / (4 * 0.0015)
            / (2 * np.pi * sigma**2)
        )
        opacity = (1 - torch.exp(-peak_tau)).clamp(max=0.8)
        color = self.tensor(d["colors"][selected]) * 0.52
        centers = self.tensor(d["x"][selected])
        camera_x = centers @ cam.view[:3, :3].T + cam.view[:3, 3]
        projected = camera_x @ cam.K.T
        uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-6)
        uv = uv / uv.new_tensor([cam.width, cam.height]) * 2 - 1
        bounds = F.grid_sample(
            torch.stack([depth, background_depth])[None],
            uv[None, None],
            align_corners=False,
            padding_mode="border",
        )[0, :, 0]
        visible = torch.sigmoid((camera_x[:, 2] - bounds[0]) * 150) * torch.sigmoid(
            (bounds[1] - self.dx * 0.5 - camera_x[:, 2]) * 150
        )
        im, alpha = super().splat(cam, centers, cov, color, opacity * visible, True)
        return body * (1 - alpha[..., None]) + im[..., :3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--profile", choices=["legacy", "granular"], default="granular")
    parser.add_argument("--sigma", type=float, default=1.40)
    parser.add_argument("--grains", type=int, default=20)
    parser.add_argument(
        "--suspension",
        type=float,
        default=0.10,
        help="Residual suspended-material optical weight",
    )
    parser.add_argument("--camera", choices=["orbit", "close"], default="orbit")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a fresh render directory")
    (out / "frames").mkdir(parents=True)
    cfg = json.loads((args.input / "config.json").read_text())
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    rest = np.load(args.input / "initial.npz")["positions"]
    paths = (
        [args.input / "states" / f"{frame:05d}.npz" for frame in args.frames]
        if args.frames
        else sorted((args.input / "states").glob("*.npz"))[:: args.stride]
    )
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError("No simulation frames")
    wp.init()
    wp.set_device("cuda:0")
    renderer = (
        GranularRenderer(
            cfg,
            rest,
            args.width,
            args.height,
            sigma=args.sigma,
            grains=args.grains,
            suspension=args.suspension,
        )
        if args.profile == "granular"
        else GaussianRenderer(cfg, rest, args.width, args.height)
    )
    if args.camera == "close":
        renderer.make_camera = lambda progress: close_camera(
            args.width, args.height, "cuda:0", cfg["scene"]
        )
    video = Video(out / "demo.mp4", args.width, args.height, cfg["fps"])
    records = []
    start = time.time()
    try:
        for index, path in enumerate(paths):
            with np.load(path) as data:
                state = dict(data)
            rgb, camera, stats, buffers = renderer.draw(
                state, int(path.stem) / max(cfg["frames"] - 1, 1), args.diagnostics
            )
            image = (rgb.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
            Image.fromarray(image).save(out / "frames" / f"{index:05d}.png")
            video.write(image)
            records.append(dict(source_frame=int(path.stem), camera=camera, **stats))
            for name, buf in buffers.items():
                folder = out / name
                folder.mkdir(exist_ok=True)
                Image.fromarray(
                    np.uint8(buf.clamp(0, 1).cpu().numpy() * 255 + 0.5)
                ).save(folder / f"{index:05d}.png")
            if index % 10 == 0 or index == len(paths) - 1:
                print(
                    f"{args.profile} render {index + 1}/{len(paths)} {time.time() - start:.1f}s {stats}",
                    flush=True,
                )
    finally:
        video.close()
    (out / "frames.json").write_text(json.dumps(records, indent=2))
    (out / "config.json").write_text(
        json.dumps(
            dict(
                source=str(args.input.resolve()),
                renderer=args.profile,
                width=args.width,
                height=args.height,
                fps=cfg["fps"],
                frames=len(records),
                parameters=dict(
                    sigma=args.sigma,
                    grains=args.grains,
                    suspension=args.suspension,
                    camera=args.camera,
                ),
                source_sha256=source_hashes,
                render_seconds=time.time() - start,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
