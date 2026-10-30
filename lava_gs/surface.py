"""World-space lava isosurface, advected crust detail and thermal emission."""

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp
from snowy_gs.mpm import weights, sample, normal
from snowy_gs.surface import density_at


@wp.kernel(enable_backward=False)
def splat(
    x: wp.array(dtype=wp.vec3),
    temperature: wp.array(dtype=float),
    reference: wp.array(dtype=wp.vec3),
    skin_age: wp.array(dtype=float),
    birth: wp.array(dtype=float),
    fields: wp.array4d(dtype=float),
    o: wp.vec3,
    dx: float,
    volume: float,
):
    p = wp.tid()
    g = (x[p] - o) / dx
    b = wp.vec3i(
        int(wp.floor(g[0] - 0.5)), int(wp.floor(g[1] - 0.5)), int(wp.floor(g[2] - 0.5))
    )
    f = g - wp.vec3(float(b[0]), float(b[1]), float(b[2]))
    wx, wy, wz = weights(f[0]), weights(f[1]), weights(f[2])
    for a in range(3):
        for d in range(3):
            for e in range(3):
                i, j, k = b[0] + a, b[1] + d, b[2] + e
                if (
                    i >= 0
                    and j >= 0
                    and k >= 0
                    and i < fields.shape[1]
                    and j < fields.shape[2]
                    and k < fields.shape[3]
                ):
                    w = wx[a] * wy[d] * wz[e] * volume / (dx * dx * dx)
                    wp.atomic_add(fields, 0, i, j, k, w)
                    wp.atomic_add(fields, 1, i, j, k, w * temperature[p])
                    for c in range(3):
                        wp.atomic_add(fields, c + 2, i, j, k, w * reference[p][c])
                    wp.atomic_add(fields, 5, i, j, k, w * skin_age[p])
                    wp.atomic_add(fields, 6, i, j, k, w * birth[p])


@wp.func
def noise(p: wp.vec3):
    base = wp.vec3(wp.floor(p[0]), wp.floor(p[1]), wp.floor(p[2]))
    f = p - base
    f = wp.cw_mul(wp.cw_mul(f, f), wp.vec3(3.0) - 2.0 * f)
    value = float(0.0)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                c = base + wp.vec3(float(i), float(j), float(k))
                raw = wp.sin(wp.dot(c, wp.vec3(127.1, 311.7, 74.7))) * 43758.5453
                w = wp.vec3(1.0) - f
                if i == 1:
                    w[0] = f[0]
                if j == 1:
                    w[1] = f[1]
                if k == 1:
                    w[2] = f[2]
                value += w[0] * w[1] * w[2] * (raw - wp.floor(raw))
    return value


@wp.func
def relief(p: wp.vec3):
    return 0.55 * noise(p * 4.0) + 0.30 * noise(p * 11.0) + 0.15 * noise(p * 29.0)


@wp.func
def surface_value(
    den: wp.array3d(dtype=float),
    rx: wp.array3d(dtype=float),
    ry: wp.array3d(dtype=float),
    rz: wp.array3d(dtype=float),
    o: wp.vec3,
    dx: float,
    p: wp.vec3,
    iso: float,
):
    d = density_at(den, o, dx, p)
    if d < 0.05:
        return d - iso
    ref = wp.vec3(
        sample(rx, o, dx, p), sample(ry, o, dx, p), sample(rz, o, dx, p)
    ) / wp.max(d, 1.0e-6)
    return d - iso * (0.55 + 1.0 * relief(ref))


@wp.func
def emission(t: float):
    # Three visible-wavelength Planck samples, normalized at 1450 K.
    # This is an RGB approximation, not spectral colorimetry.
    t = wp.clamp(t, 650.0, 1800.0)
    red = wp.exp(15.2655 - 22135.0 / t)
    green = 2.30 * wp.exp(15.2655 - 26160.0 / t)
    blue = 6.29 * wp.exp(15.2655 - 31973.0 / t)
    return 0.55 * wp.vec3(red, green, blue)


@wp.kernel(enable_backward=False)
def raymarch(
    den: wp.array3d(dtype=float),
    heat: wp.array3d(dtype=float),
    rx: wp.array3d(dtype=float),
    ry: wp.array3d(dtype=float),
    rz: wp.array3d(dtype=float),
    skin_age: wp.array3d(dtype=float),
    birth: wp.array3d(dtype=float),
    coarse: wp.array3d(dtype=float),
    o: wp.vec3,
    dx: float,
    coarse_dx: float,
    camera: wp.mat44,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    scale: float,
    source: wp.array3d(dtype=float),
    result: wp.array3d(dtype=float),
    glow: wp.array3d(dtype=float),
    depth: wp.array2d(dtype=float),
    iso: float,
    crust: float,
    time: float,
):
    iy, ix = wp.tid()
    center = wp.vec3(camera[0, 3], camera[1, 3], camera[2, 3]) * scale
    vr = wp.vec3((float(ix) + 0.5 - cx) / fx, (float(iy) + 0.5 - cy) / fy, 1.0)
    ray = wp.vec3(
        camera[0, 0] * vr[0] + camera[0, 1] * vr[1] + camera[0, 2],
        camera[1, 0] * vr[0] + camera[1, 1] * vr[1] + camera[1, 2],
        camera[2, 0] * vr[0] + camera[2, 1] * vr[1] + camera[2, 2],
    )
    high = o + dx * wp.vec3(
        float(den.shape[0] - 1), float(den.shape[1] - 1), float(den.shape[2] - 1)
    )
    near = float(0.02)
    far = source[iy, ix, 3] * scale
    for a in range(3):
        inverse = 1.0 / (ray[a] + 1.0e-10)
        t1 = (o[a] - center[a]) * inverse
        t2 = (high[a] - center[a]) * inverse
        near = wp.max(near, wp.min(t1, t2))
        far = wp.min(far, wp.max(t1, t2))
    t = near
    last = near
    found = int(0)
    for step in range(1600):
        if t >= far:
            break
        point = center + t * ray
        if surface_value(den, rx, ry, rz, o, dx, point, iso) > 0.0:
            found = 1
            break
        last = t
        delta = dx * 0.40 / wp.length(ray)
        if density_at(coarse, o, coarse_dx, point) < 1.0e-7:
            delta = coarse_dx * 0.7 / wp.length(ray)
        t += delta
    rgb = wp.vec3(source[iy, ix, 0], source[iy, ix, 1], source[iy, ix, 2])
    light = wp.vec3(0.0)
    z = source[iy, ix, 3] * scale
    if found == 1:
        for it in range(7):
            mid = (last + t) * 0.5
            if surface_value(den, rx, ry, rz, o, dx, center + mid * ray, iso) > 0.0:
                t = mid
            else:
                last = mid
        p = center + t * ray
        d = wp.max(sample(den, o, dx, p), 1.0e-6)
        temp = sample(heat, o, dx, p) / d
        ref = (
            wp.vec3(sample(rx, o, dx, p), sample(ry, o, dx, p), sample(rz, o, dx, p))
            / d
        )
        n = -normal(den, o, dx, p)
        if wp.dot(n, ray) > 0.0:
            n = -n
        material_age = wp.max(0.0, time - sample(birth, o, dx, p) / d)
        surface_age = wp.max(0.0, sample(skin_age, o, dx, p) / d)
        age = 0.75 * material_age + 0.25 * surface_age
        patch = (
            0.55 * noise(ref * 3.1)
            + 0.30 * noise(ref * 10.3)
            + 0.15 * noise(ref * 31.0)
        )
        fine = noise(ref * 14.0)
        # A sub-grid radiatively cooled skin represents the unresolved outer
        # millimetres. Open patches expose the hotter simulated bulk material.
        skin = wp.max(
            750.0, temp - 180.0 * wp.sqrt(age + 0.08) * crust - (patch - 0.5) * 190.0
        )
        hot = wp.clamp((temp - 1100.0) / 350.0, 0.0, 1.0)
        folds = 0.5 + 0.5 * wp.sin(
            ref[1] * 44.0 + 7.0 * noise(ref * 2.0) + 2.0 * wp.sin(ref[0] * 14.0)
        )
        fold_light = 0.65 + 0.35 * folds
        furrow = (
            (1.0 - wp.smoothstep(0.02, 0.10, folds))
            * wp.smoothstep(0.45, 0.65, patch)
            * 0.24
        )
        opening = wp.smoothstep(0.64 - 0.08 * hot, 0.77, patch) * hot * 0.78
        renewal = wp.clamp(1.0 - surface_age / wp.max(material_age, 0.01), 0.0, 1.0)
        opening = wp.min(
            0.9, opening + 0.25 * renewal * wp.smoothstep(0.50, 0.72, patch)
        )
        fresh = wp.smoothstep(1240.0, 1460.0, skin)
        coverage = wp.clamp(
            wp.max(furrow, opening) + fresh * (0.35 + 0.45 * patch), 0.0, 1.0
        )
        effective_temperature = skin + (temp - skin) * coverage
        light = emission(effective_temperature) * fold_light
        rough = (0.35 + 1.3 * fine) * (0.5 + noise(ref * 125.0))
        sun = wp.normalize(wp.vec3(-0.4, -0.8, 1.0))
        e = float(0.008)
        bump = wp.vec3(
            relief(ref + wp.vec3(e, 0.0, 0.0)) - relief(ref - wp.vec3(e, 0.0, 0.0)),
            relief(ref + wp.vec3(0.0, e, 0.0)) - relief(ref - wp.vec3(0.0, e, 0.0)),
            0.0,
        ) / (2.0 * e)
        n = wp.normalize(n - 0.075 * bump)
        diffuse = 0.25 + 0.75 * wp.max(0.0, wp.dot(n, sun))
        basalt = wp.vec3(0.026, 0.026, 0.029) * (diffuse * rough * fold_light)
        view = wp.normalize(-ray)
        halfway = wp.normalize(sun + view)
        sheen = 0.004 * wp.pow(wp.max(0.0, wp.dot(n, halfway)), 24.0)
        rgb = basalt + wp.vec3(sheen) + light
        z = t
    for c in range(3):
        result[iy, ix, c] = rgb[c]
        glow[iy, ix, c] = light[c]
    depth[iy, ix] = z


def srgb_to_linear(v):
    return torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(v):
    return torch.where(
        v <= 0.0031308, 12.92 * v, 1.055 * v.clamp_min(0).pow(1 / 2.4) - 0.055
    )


def aces(v):
    return ((v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14)).clamp(0, 1)


class LavaSurface:
    def __init__(self, mpm, geo, scale, crust=1.0, exposure=0.55, surface_iso=0.20):
        self.mpm, self.geo, self.scale = mpm, geo, scale
        self.dx = mpm.dx * 0.5
        self.shape = tuple((np.array(mpm.shape) - 1) * 2 + 1)
        self.fields = torch.zeros((7, *self.shape), device=mpm.device)
        self.crust, self.exposure = crust, exposure
        self.surface_iso = surface_iso
        k = torch.tensor([0.19, 0.62, 0.19], device=mpm.device)
        self.kernels = [
            k.reshape(1, 1, 3, 1, 1).repeat(7, 1, 1, 1, 1),
            k.reshape(1, 1, 1, 3, 1).repeat(7, 1, 1, 1, 1),
            k.reshape(1, 1, 1, 1, 3).repeat(7, 1, 1, 1, 1),
        ]

    def reconstruct(self):
        m = self.mpm
        self.fields.zero_()
        wp.launch(
            splat,
            m.count,
            [
                m.x,
                m.temp,
                m.reference,
                m.skin_age,
                m.birth,
                wp.from_torch(self.fields),
                wp.vec3(*m.origin),
                self.dx,
                m.volume,
            ],
            device=m.device,
        )
        f = self.fields[None]
        for i, k in enumerate(self.kernels):
            padding = [0, 0, 0]
            padding[i] = 1
            f = F.conv3d(f, k, padding=tuple(padding), groups=7)
        self.fields = f[0].contiguous()
        self.coarse = F.max_pool3d(self.fields[0][None, None], 5, 2, 2)[
            0, 0
        ].contiguous()
        # Coarse 3D occupancy gives contact darkening and nearby warm spill.
        heat = (self.fields[1] / self.fields[0].clamp_min(1.0e-5) - 1050).clamp(
            0, 500
        ) / 500
        heat = heat * self.fields[0].clamp(0, 1)
        near_field = torch.stack((self.fields[0].clamp(0, 1), heat))[None]
        near_field = F.avg_pool3d(near_field, 4, 4)
        self.near_field = F.avg_pool3d(near_field, 5, 1, 2)

    @torch.no_grad()
    def render(self, source, camera):
        m = self.mpm
        self.reconstruct()
        out = torch.empty((*source.shape[:2], 3), device=source.device)
        glow = torch.zeros_like(out)
        depth = torch.empty(source.shape[:2], device=source.device)
        p = camera.world_points(source[..., 3]) * self.scale
        uv = (
            2
            * (p - p.new_tensor(m.origin) - self.dx * 1.5)
            / (p.new_tensor(self.near_field.shape[2:]) - 1)
            / (4 * self.dx)
            - 1
        )
        uv = uv[..., [2, 1, 0]][None, None]
        sampled = F.grid_sample(
            self.near_field, uv, align_corners=True, padding_mode="zeros"
        )[0, :, 0]
        occlusion = sampled[0].clamp(0, 1) * 0.5
        spill = sampled[1] * 0.035
        src = source.clone()
        linear = srgb_to_linear(src[..., :3].clamp(0, 1))
        linear *= linear.new_tensor([0.78, 0.85, 1.0]) * self.exposure
        linear *= 1 - occlusion[..., None]
        linear += spill[..., None] * linear.new_tensor([1.0, 0.15, 0.012])
        src[..., :3] = linear
        intr = camera.K.cpu().numpy()
        inverse = torch.linalg.inv(camera.view).cpu().numpy()
        wp.launch(
            raymarch,
            source.shape[:2],
            [
                *[wp.from_torch(a) for a in self.fields],
                wp.from_torch(self.coarse),
                wp.vec3(*m.origin),
                self.dx,
                m.dx,
                wp.mat44(inverse),
                float(intr[0, 0]),
                float(intr[1, 1]),
                float(intr[0, 2]),
                float(intr[1, 2]),
                self.scale,
                wp.from_torch(src.contiguous()),
                wp.from_torch(out),
                wp.from_torch(glow),
                wp.from_torch(depth),
                self.surface_iso,
                self.crust,
                float(m.time),
            ],
            device=m.device,
        )
        # Bloom is computed from visible emission only; opaque objects occlude
        # the lava before this optical point-spread approximation is applied.
        light = glow.permute(2, 0, 1)[None]
        bloom = torch.zeros_like(light)
        for down, weight in [(4, 0.01), (12, 0.012), (32, 0.006)]:
            small = F.interpolate(light, scale_factor=1 / down, mode="area")
            small = F.avg_pool2d(small, 7, 1, 3)
            bloom += (
                F.interpolate(
                    small, size=out.shape[:2], mode="bilinear", align_corners=False
                )
                * weight
            )
        out += bloom[0].permute(1, 2, 0)
        final = linear_to_srgb(aces(out)).clamp(0, 1)
        self.last_depth = depth
        self.last_glow = glow
        return final
