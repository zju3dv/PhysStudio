"""Depth-tested lightning emission and local relighting of a 3DGS scene."""
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp


@wp.func
def fbm(p: wp.vec3):
    return (
        0.57 * wp.noise(wp.uint32(91), p)
        + 0.27 * wp.noise(wp.uint32(183), p * 2.03)
        + 0.16 * wp.noise(wp.uint32(57), p * 4.11)
    )


@wp.func
def sky_color(ray: wp.vec3, t: float, flash: float):
    h = wp.max(ray[2], 0.0)
    p = ray / wp.max(0.12, ray[2] + 0.22) * 2.1 + wp.vec3(t * 0.025, -t * 0.012, 4.1)
    n = fbm(p)
    fine = fbm(p * 2.2)
    shade = wp.clamp(0.47 + n * 0.85 + fine * 0.18, 0.12, 0.86)
    horizon = wp.exp(-h * 5.0)
    ambient = wp.lerp(
        wp.vec3(0.018, 0.028, 0.046), wp.vec3(0.045, 0.060, 0.077), horizon
    )
    return ambient * shade + flash * wp.vec3(0.30, 0.35, 0.48) * wp.clamp(
        0.45 + n, 0.1, 0.95
    )


@wp.func
def surface_height(
    p: wp.vec3, roof: wp.array2d(dtype=float), origin: wp.vec3, dx: float
):
    i = int(wp.floor((p[0] - origin[0]) / dx))
    j = int(wp.floor((p[1] - origin[1]) / dx))
    result = float(-10000.0)
    if i >= 0 and j >= 0 and i < roof.shape[0] and j < roof.shape[1]:
        result = roof[i, j]
    return result


@wp.kernel
def backdrop(
    rgb: wp.array2d(dtype=wp.vec3),
    alpha: wp.array2d(dtype=float),
    depth: wp.array2d(dtype=float),
    rays: wp.array2d(dtype=wp.vec3),
    normals: wp.array2d(dtype=wp.vec3),
    out: wp.array2d(dtype=wp.vec3),
    light: wp.array(dtype=wp.vec3),
    power: wp.array(dtype=float),
    roof: wp.array2d(dtype=float),
    origin: wp.vec3,
    dx: float,
    eye: wp.vec3,
    t: float,
    exposure: float,
    fog: float,
    flash: float,
):
    y, x = wp.tid()
    ray = rays[y, x]
    z = depth[y, x]
    a = alpha[y, x]
    sky = sky_color(ray, t, flash)
    surface = eye + ray * z
    color = rgb[y, x] / wp.max(a, 0.001)
    linear = wp.vec3(
        wp.pow(wp.max(color[0], 0.0), 2.2),
        wp.pow(wp.max(color[1], 0.0), 2.2),
        wp.pow(wp.max(color[2], 0.0), 2.2),
    )
    illumination = float(0.0)
    near_distance = float(1.0e10)
    nearest = wp.vec3(0.0)
    for n in range(light.shape[0]):
        delta = light[n] - surface
        d2 = wp.dot(delta, delta)
        lambert = 0.25 + 0.75 * wp.abs(wp.dot(normals[y, x], wp.normalize(delta)))
        illumination += power[n] * lambert / (d2 + 0.35)
        if d2 < near_distance:
            near_distance = d2
            nearest = light[n]
    visibility = float(1.0)
    if a > 0.5 and flash > 0.0001 and light.shape[0] > 0:
        for s in range(1, 20):
            p = wp.lerp(surface, nearest, float(s) / 20.0)
            clearance = p[2] + dx * 0.85 - surface_height(p, roof, origin, dx)
            visibility = wp.min(
                visibility, 0.30 + 0.70 * wp.smoothstep(-dx * 0.8, dx * 1.8, clearance)
            )
    reflected = wp.cw_mul(linear, wp.vec3(0.74, 0.84, 1.0)) * exposure * exposure
    reflected += wp.cw_mul(linear, wp.vec3(0.75, 0.84, 1.0)) * illumination * visibility
    distance = wp.min(z, 120.0)
    haze = 1.0 - wp.exp(-fog * distance)
    reflected = wp.lerp(
        reflected, sky_color(wp.vec3(ray[0], ray[1], 0.04), t, flash * 0.3), haze
    )
    out[y, x] = a * reflected + (1.0 - a) * sky


@wp.func
def project(p: wp.vec3, view: wp.mat44, fx: float, fy: float, cx: float, cy: float):
    c = wp.transform_point(view, p)
    return wp.vec3(
        fx * c[0] / wp.max(c[2], 0.001) + cx, fy * c[1] / wp.max(c[2], 0.001) + cy, c[2]
    )


@wp.kernel
def raster_channels(
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    strength: wp.array(dtype=float),
    radius: wp.array(dtype=float),
    surface_z: wp.array2d(dtype=float),
    emission: wp.array2d(dtype=wp.vec3),
    view: wp.mat44,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    fog: float,
):
    n, lane = wp.tid()
    p = project(a[n], view, fx, fy, cx, cy)
    q = project(b[n], view, fx, fy, cx, cy)
    if p[2] > 0.02 and q[2] > 0.02 and strength[n] > 0.00001:
        sigma = wp.clamp(fx * radius[n] / (0.5 * (p[2] + q[2])), 0.48, 2.4)
        halo = sigma * 3.5
        pad = halo * 3.0
        xmin = wp.max(0, int(wp.floor(wp.min(p[0], q[0]) - pad)))
        xmax = wp.min(emission.shape[1] - 1, int(wp.ceil(wp.max(p[0], q[0]) + pad)))
        ymin = wp.max(0, int(wp.floor(wp.min(p[1], q[1]) - pad)))
        ymax = wp.min(emission.shape[0] - 1, int(wp.ceil(wp.max(p[1], q[1]) + pad)))
        width = xmax - xmin + 1
        count = width * (ymax - ymin + 1)
        d = wp.vec2(q[0] - p[0], q[1] - p[1])
        den = wp.dot(d, d) + 0.00001
        pixel = lane
        while pixel < count and width > 0:
            x = xmin + pixel % width
            y = ymin + pixel / width
            u = wp.clamp(
                wp.dot(wp.vec2(float(x) + 0.5 - p[0], float(y) + 0.5 - p[1]), d) / den,
                0.0,
                1.0,
            )
            near = wp.vec2(p[0], p[1]) + u * d
            dd = wp.vec2(float(x) + 0.5, float(y) + 0.5) - near
            d2 = wp.dot(dd, dd)
            z = 1.0 / ((1.0 - u) / p[2] + u / q[2])
            if z < surface_z[y, x] + 0.018:
                core = wp.exp(-0.5 * d2 / (sigma * sigma))
                corona = wp.exp(-0.5 * d2 / (halo * halo)) * 0.13
                c = core * wp.vec3(0.85, 0.91, 1.0) + corona * wp.vec3(0.24, 0.38, 1.0)
                # The segment endpoint taper avoids bright beads at junctions.
                edge = wp.clamp(wp.sqrt(den) / wp.max(2.0 * sigma, 1.0), 0.25, 1.0)
                wp.atomic_add(
                    emission, y, x, c * (strength[n] * edge * wp.exp(-fog * z * 0.5))
                )
            pixel += 128


@wp.kernel
def rain_segments(
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    power: wp.array(dtype=float),
    radius: wp.array(dtype=float),
    origin: wp.vec3,
    extent: wp.vec3,
    t: float,
    flash: float,
):
    n = wp.tid()
    rng = wp.rand_init(721, n)
    xx = wp.randf(rng)
    yy = wp.randf(rng)
    zz = wp.randf(rng)
    speed = 3.2 + wp.randf(rng) * 2.4
    p = origin + wp.vec3(
        wp.frac(xx + t * 0.027) * extent[0],
        yy * extent[1],
        wp.frac(zz - t * speed / extent[2]) * extent[2],
    )
    a[n] = p
    b[n] = p + wp.vec3(-0.006, 0.003, 0.04 + speed * 0.008)
    power[n] = 0.007 + flash * 0.022
    radius[n] = 0.00035


def pulse(t, start):
    dt = t - start
    peak = 0.0
    for delay, scale in ((0.0, 1.0), (0.13, 0.65), (0.31, 0.9), (0.51, 0.38)):
        x = dt - delay
        if 0 <= x < 0.22:
            peak += scale * np.exp(-x / 0.044)
    return float(peak)


class Channel:
    def __init__(self, data, cfg):
        x = data["positions"]
        parents = data["parents"]
        flow = data["flow"]
        main = data["main_path"]
        reference = max(float(flow.max()), 1e-8)
        relative = np.clip((flow / reference) ** 0.55, 0.004, 1.0)
        relative[main] = 1.0
        rng = np.random.default_rng(391)
        aa = []
        bb = []
        strength = []
        width = []
        birth = []
        dx = float(data["dx"])
        for i in range(1, len(x)):
            if parents[i] < 0:
                continue
            p = x[parents[i]]
            q = x[i]
            direction = q - p
            # Subcell detail preserves the simulated graph and its endpoints.
            offsets = rng.normal(size=(5, 3)) * dx * 0.065
            offsets[0] = offsets[-1] = 0
            points = p + np.linspace(0, 1, 5)[:, None] * direction + offsets
            for j in range(4):
                middle = 0.5 * (points[j] + points[j + 1])
                density = np.exp(
                    -0.5
                    * np.sum(
                        (
                            (middle - np.array(cfg["cloud"]))
                            / np.array(cfg["cloud_radius"])
                        )
                        ** 2
                    )
                )
                attenuation = np.exp(-10.0 * density)
                aa.append(points[j])
                bb.append(points[j + 1])
                strength.append(relative[i] * attenuation)
                width.append(
                    dx * (0.014 if main[i] else 0.007) * (0.7 + 0.3 * relative[i])
                )
                birth.append(data["born"][i] / max(data["born"].max(), 1))
        self.a = np.asarray(aa, dtype="f")
        self.b = np.asarray(bb, dtype="f")
        self.strength = np.asarray(strength, dtype="f")
        self.radius = np.asarray(width, dtype="f")
        self.birth = np.array(birth)
        ids = np.where(main)[0]
        if len(ids) < 4:
            ids = np.argsort(flow)[-30:]
        ids = ids[np.linspace(0, len(ids) - 1, min(24, len(ids))).astype(int)]
        self.lights = x[ids].astype("f")


class Renderer:
    def __init__(self, scene, cfg, geometry, events):
        self.scene = scene
        self.cfg = cfg
        self.geometry = geometry
        self.channels = [Channel(e, cfg) for e in events]
        self.roof = wp.array(geometry["roof"].astype("f"), dtype=float, device="cuda:0")
        self.rain = [
            wp.empty(1400, dtype=wp.vec3),
            wp.empty(1400, dtype=wp.vec3),
            wp.empty(1400, dtype=float),
            wp.empty(1400, dtype=float),
        ]

    @torch.no_grad()
    def render(self, cam, t, starts=(1.1, 3.65, 6.2)):
        scene = self.scene
        cfg = self.cfg
        rgbd, alpha = scene.render(cam, mask=scene.thunder_mask)
        rgb = rgbd[..., :3].clamp(0, 1)
        clean = (
            rgb + (1 - alpha[..., None]) * rgb.new_tensor([0.46, 0.57, 0.70])
        ).clamp(0, 1)
        raw = cam.world_points(torch.ones_like(alpha)) - cam.center
        ray_scale = torch.linalg.vector_norm(raw, dim=-1)
        rays = raw / ray_scale[..., None]
        world = cam.world_points(rgbd[..., 3])
        color = rgb / alpha.clamp_min(0.001)[..., None]
        sky = scene.sky_weight(color, world, rays)
        if cfg["scene"] == "train":
            red, green, blue = color.unbind(-1)
            sky = ((blue - red - 0.08) / 0.10).clamp(0, 1) * (
                (blue - green + 0.06) / 0.05
            ).clamp(0, 1)
            distant = ((rgbd[..., 3] - 3.1) / 0.35).clamp(0, 1)
            sky = torch.maximum(
                sky, ((color.min(-1).values - 0.75) / 0.15).clamp(0, 1) * distant
            )
            sky *= ((blue - 0.30) / 0.25).clamp(0, 1)
            foreground = (
                (rgbd[..., 3] < 3.5)
                & (world[..., 2] < 1.12)
                & ((blue - red) < 0.22)
                & ~((blue > 0.85) & (green > 0.85))
            )
            sky *= ~foreground
            sky *= ((rays[..., 2] - 0.065) / 0.055).clamp(0, 1)
            # This calibrated camera corridor has no geometry in the zenith.
            sky = torch.maximum(sky, ((rays[..., 2] - 0.45) / 0.06).clamp(0, 1))
        alpha = alpha * (1 - sky)
        rgb = rgb * (1 - sky[..., None])
        surface_z = torch.where(
            alpha > 0.35, rgbd[..., 3], torch.full_like(alpha, 1000.0)
        ).contiguous()
        depth = surface_z * ray_scale
        normals = F.normalize(
            torch.cross(
                torch.roll(world, -1, 0) - world,
                torch.roll(world, -1, 1) - world,
                dim=-1,
            ),
            dim=-1,
        )
        normals = torch.nan_to_num(normals)
        a = []
        b = []
        strength = []
        radius = []
        lights = []
        powers = []
        flash = 0.0
        for channel, start in zip(self.channels, starts):
            peak = pulse(t, start)
            flash += peak
            leader = (t - start + 0.18) / 0.18
            if -0.18 <= t - start < 0:
                weights = channel.strength * 0.035 * (channel.birth < leader)
            else:
                weights = channel.strength * peak * 2.8
            if weights.max() > 0:
                a.append(channel.a)
                b.append(channel.b)
                strength.append(weights)
                radius.append(channel.radius)
            if peak > 0:
                lights.append(channel.lights)
                powers.append(
                    np.full(
                        len(channel.lights),
                        peak * 10.0 / len(channel.lights),
                        dtype="f",
                    )
                )
        out = torch.zeros_like(rgb)
        emission = torch.zeros_like(rgb)

        def array(items, dtype):
            arr = (
                np.concatenate(items)
                if items
                else np.empty((0, 3) if dtype == wp.vec3 else (0,), dtype="f")
            )
            return wp.array(np.asarray(arr, dtype="f"), dtype=dtype, device="cuda:0")

        light = array(lights, wp.vec3)
        power = array(powers, float)
        eye = wp.vec3(*cam.center.tolist())
        origin = wp.vec3(*self.geometry["origin"])
        torch.cuda.synchronize()
        wp.launch(
            backdrop,
            (cam.height, cam.width),
            [
                wp.from_torch(rgb.contiguous(), dtype=wp.vec3),
                wp.from_torch(alpha.contiguous()),
                wp.from_torch(depth.contiguous()),
                wp.from_torch(rays.contiguous(), dtype=wp.vec3),
                wp.from_torch(normals.contiguous(), dtype=wp.vec3),
                wp.from_torch(out, dtype=wp.vec3),
                light,
                power,
                self.roof,
                origin,
                cfg["dx"],
                eye,
                t,
                cfg["exposure"],
                cfg["fog"],
                flash,
            ],
            device="cuda:0",
        )
        view = wp.mat44(*cam.view.cpu().numpy().reshape(-1).tolist())
        intrinsics = [
            float(cam.K[0, 0]),
            float(cam.K[1, 1]),
            float(cam.K[0, 2]),
            float(cam.K[1, 2]),
        ]
        if a:
            wa, wb, ws, wr = (
                array(a, wp.vec3),
                array(b, wp.vec3),
                array(strength, float),
                array(radius, float),
            )
            wp.launch(
                raster_channels,
                (len(wa), 128),
                [
                    wa,
                    wb,
                    ws,
                    wr,
                    wp.from_torch(surface_z),
                    wp.from_torch(emission, dtype=wp.vec3),
                    view,
                    *intrinsics,
                    cfg["fog"],
                ],
                device="cuda:0",
            )
        extent = wp.vec3(*(np.array(cfg["shape"]) * cfg["dx"]))
        wp.launch(
            rain_segments, 1400, [*self.rain, origin, extent, t, flash], device="cuda:0"
        )
        wp.launch(
            raster_channels,
            (1400, 128),
            [
                *self.rain,
                wp.from_torch(surface_z),
                wp.from_torch(emission, dtype=wp.vec3),
                view,
                *intrinsics,
                cfg["fog"],
            ],
            device="cuda:0",
        )
        wp.synchronize_device("cuda:0")
        # A multiscale camera point-spread function is applied after occlusion.
        signal = emission.permute(2, 0, 1)[None]
        bloom = torch.zeros_like(signal)
        for factor, weight in ((4, 0.32), (12, 0.25), (32, 0.16)):
            size = (max(1, cam.height // factor), max(1, cam.width // factor))
            small = F.interpolate(signal, size=size, mode="area")
            small = F.avg_pool2d(small, 5, stride=1, padding=2)
            bloom += weight * F.interpolate(
                small,
                size=(cam.height, cam.width),
                mode="bilinear",
                align_corners=False,
            )
        radiance = out + emission + bloom[0].permute(1, 2, 0)
        mapped = (
            (
                radiance
                / (1 + torch.maximum(radiance - 1, torch.zeros_like(radiance)) * 0.4)
            )
            .clamp(0, 1)
            .pow(1 / 2.2)
        )
        return (
            mapped,
            clean,
            dict(
                time=t,
                flash=flash,
                channel_segments=sum(len(v) for v in a),
                finite=bool(torch.isfinite(mapped).all()),
            ),
        )
