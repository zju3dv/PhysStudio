"""Continuous MPM mass isosurface, ray rendered with per-pixel GS occlusion.

Quadratic particle reconstruction on a grid at half the simulation spacing.
All samples are world-space and independent of the camera. Warp ray marching
avoids a mesh dependency and preserves sharp snow/object silhouettes.
"""

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp
from .mpm import weights, sample, normal


@wp.func
def density_at(a: wp.array3d(dtype=float), o: wp.vec3, dx: float, p: wp.vec3):
    g = (p - o) / dx
    value = float(0.0)
    if (
        g[0] >= 0.0
        and g[1] >= 0.0
        and g[2] >= 0.0
        and g[0] < float(a.shape[0] - 1)
        and g[1] < float(a.shape[1] - 1)
        and g[2] < float(a.shape[2] - 1)
    ):
        value = sample(a, o, dx, p)
    return value


@wp.kernel(enable_backward=False)
def reconstruct(
    x: wp.array(dtype=wp.vec3),
    vol: wp.array(dtype=float),
    light: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    illum: wp.array3d(dtype=float),
    o: wp.vec3,
    dx: float,
    phi: wp.array3d(dtype=float),
    sim_dx: float,
    plane: wp.vec3,
):
    p = wp.tid()
    point = x[p]
    relative = point[2] - plane[0] * point[0] - plane[1] * point[1] - plane[2]
    if relative > sim_dx * 1.5 and sample(phi, o, sim_dx, point) < sim_dx * 0.6:
        # Remove the known collision shell margin from the visible surface.
        point -= normal(phi, o, sim_dx, point) * (sim_dx * 0.30)
    g = (point - o) / dx
    b = wp.vec3i(
        int(wp.floor(g[0] - 0.5)), int(wp.floor(g[1] - 0.5)), int(wp.floor(g[2] - 0.5))
    )
    f = g - wp.vec3(float(b[0]), float(b[1]), float(b[2]))
    wx, wy, wz = weights(f[0]), weights(f[1]), weights(f[2])
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ix, iy, iz = b[0] + i, b[1] + j, b[2] + k
                if (
                    ix >= 0
                    and iy >= 0
                    and iz >= 0
                    and ix < density.shape[0]
                    and iy < density.shape[1]
                    and iz < density.shape[2]
                ):
                    w = wx[i] * wy[j] * wz[k] * vol[p] / (dx * dx * dx)
                    wp.atomic_add(density, ix, iy, iz, w)
                    wp.atomic_add(illum, ix, iy, iz, w * light[p])


@wp.kernel(enable_backward=False)
def trace(
    density: wp.array3d(dtype=float),
    illum: wp.array3d(dtype=float),
    coarse: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
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
    depth: wp.array2d(dtype=float),
    plane: wp.vec3,
    bounds: wp.vec4,
    iso: float,
):
    iy, ix = wp.tid()
    center = wp.vec3(camera[0, 3], camera[1, 3], camera[2, 3]) * scale
    viewray = wp.vec3((float(ix) + 0.5 - cx) / fx, (float(iy) + 0.5 - cy) / fy, 1.0)
    ray = wp.vec3(
        camera[0, 0] * viewray[0] + camera[0, 1] * viewray[1] + camera[0, 2],
        camera[1, 0] * viewray[0] + camera[1, 1] * viewray[1] + camera[1, 2],
        camera[2, 0] * viewray[0] + camera[2, 1] * viewray[1] + camera[2, 2],
    )
    high = o + dx * wp.vec3(
        float(density.shape[0] - 1),
        float(density.shape[1] - 1),
        float(density.shape[2] - 1),
    )
    near = float(0.0)
    far = source[iy, ix, 3] * scale + 0.035
    for a in range(3):
        inverse = 1.0 / (ray[a] + 1.0e-10)
        t1 = (o[a] - center[a]) * inverse
        t2 = (high[a] - center[a]) * inverse
        near = wp.max(near, wp.min(t1, t2))
        far = wp.min(far, wp.max(t1, t2))
    t = near
    last = near
    found = int(0)
    for step in range(1500):
        if t >= far:
            break
        point = center + t * ray
        d = density_at(density, o, dx, point)
        if d > iso:
            found = 1
            break
        last = t
        occupied = density_at(coarse, o, coarse_dx, point)
        delta = dx * 0.35 / wp.length(ray)
        if occupied < 1.0e-10:
            delta = coarse_dx * 0.7 / wp.length(ray)
        t += delta
    rgb = wp.vec3(source[iy, ix, 0], source[iy, ix, 1], source[iy, ix, 2])
    z = source[iy, ix, 3] * scale
    if found == 1:
        for it in range(6):
            mid = (last + t) * 0.5
            if density_at(density, o, dx, center + mid * ray) > iso:
                t = mid
            else:
                last = mid
        point = center + t * ray
        n = -normal(density, o, dx, point)
        if wp.dot(n, ray) > 0.0:
            n = -n
        d = sample(density, o, dx, point)
        lighting = sample(illum, o, dx, point) / wp.max(d, 1.0e-5)
        # Broad sky light and weak cool directional light, plus granular detail.
        sun = wp.normalize(wp.vec3(-0.5, -0.3, 1.0))
        shadow = float(1.0)
        for si in range(1, 9):
            q = point + n * (dx * 1.2) + sun * (float(si) * dx * 1.5)
            if density_at(density, o, dx, q) > iso * 1.1:
                shadow = 0.4
        diffuse = 0.69 + 0.31 * wp.max(0.0, wp.dot(n, sun)) * shadow
        detail = (
            1.0
            + 0.006 * wp.sin(point[0] * 53.0 + wp.sin(point[1] * 47.0))
            + 0.004 * wp.sin(point[1] * 79.0 + point[2] * 36.0)
        )
        snow = wp.vec3(0.96, 0.985, 1.0) * (lighting * diffuse * detail)
        edge = wp.min(
            wp.min(point[0] - bounds[0], bounds[2] - point[0]),
            wp.min(point[1] - bounds[1], bounds[3] - point[1]),
        )
        blend = wp.clamp(edge / (coarse_dx * 3.0), 0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        rgb = rgb * (1.0 - blend) + snow * blend
        if blend > 0.5:
            z = t
    result[iy, ix, 0] = rgb[0]
    result[iy, ix, 1] = rgb[1]
    result[iy, ix, 2] = rgb[2]
    depth[iy, ix] = z


@wp.kernel(enable_backward=False)
def draw_flakes(
    cp: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    depth: wp.array2d(dtype=float),
    tau: wp.array2d(dtype=float),
    fx: float,
    fy: float,
    cx: float,
    cy: float,
):
    p = wp.tid()
    pos = cp[p]
    if pos[2] > 0.18 and pos[2] < 22.0:
        px = pos[0] / pos[2] * fx + cx
        py = pos[1] / pos[2] * fy + cy
        sigma = fx * radius[p] / pos[2]
        # Thin lens defocus in pixels, and an analytic shutter ellipse.
        defocus = fx * 0.0009 * wp.max(0.0, 1.0 / pos[2] - 1.0 / 7.0)
        s2 = sigma * sigma + defocus * defocus + 0.15
        motion = wp.vec2(
            (vel[p][0] * pos[2] - pos[0] * vel[p][2]) / (pos[2] * pos[2]) * fx / 90.0,
            (vel[p][1] * pos[2] - pos[1] * vel[p][2]) / (pos[2] * pos[2]) * fy / 90.0,
        )
        a = s2 + motion[0] * motion[0] / 12.0
        b = motion[0] * motion[1] / 12.0
        c = s2 + motion[1] * motion[1] / 12.0
        det = a * c - b * b
        extent = wp.min(45.0, 3.0 * wp.sqrt(wp.max(a, c)))
        lo_x = wp.max(0, int(wp.floor(px - extent)))
        hi_x = wp.min(depth.shape[1] - 1, int(wp.ceil(px + extent)))
        lo_y = wp.max(0, int(wp.floor(py - extent)))
        hi_y = wp.min(depth.shape[0] - 1, int(wp.ceil(py + extent)))
        opacity = 0.86 * sigma * sigma / wp.sqrt(det)
        opacity *= wp.clamp((pos[2] - 0.18) / 0.3, 0.0, 1.0) * wp.clamp(
            (22.0 - pos[2]) / 9.0, 0.0, 1.0
        )
        opacity *= wp.clamp(sigma / 0.5, 0.0, 1.0)
        opacity *= wp.exp(-0.012 * pos[2])
        for y in range(lo_y, hi_y + 1):
            for x in range(lo_x, hi_x + 1):
                if pos[2] < depth[y, x] + 0.002:
                    dx = float(x) + 0.5 - px
                    dy = float(y) + 0.5 - py
                    q = (c * dx * dx - 2.0 * b * dx * dy + a * dy * dy) / det
                    if q < 9.0:
                        alpha = wp.min(0.95, opacity * wp.exp(-0.5 * q))
                        wp.atomic_add(tau, y, x, -wp.log(1.0 - alpha))


class SnowSurface:
    def __init__(self, mpm, geo, light):
        self.mpm, self.geo = mpm, geo
        self.dx = mpm.dx * 0.5
        self.shape = tuple((np.array(mpm.shape) - 1) * 2 + 1)
        self.density = wp.zeros(self.shape, dtype=float, device=mpm.device)
        self.illum = wp.zeros(self.shape, dtype=float, device=mpm.device)
        self.coarse = wp.zeros(mpm.shape, dtype=float, device=mpm.device)
        self.coarse_light = wp.zeros(mpm.shape, dtype=float, device=mpm.device)
        self.light = light
        k = torch.tensor([0.22, 0.56, 0.22], device=mpm.device)
        self.kernels = [
            k.reshape(1, 1, 3, 1, 1),
            k.reshape(1, 1, 1, 3, 1),
            k.reshape(1, 1, 1, 1, 3),
        ]

    def smooth(self, array):
        v = wp.to_torch(array)[None, None]
        for i, k in enumerate(self.kernels):
            padding = [0, 0, 0]
            padding[i] = 1
            v = F.conv3d(v, k, padding=tuple(padding))
        wp.to_torch(array).copy_(v[0, 0])

    def render(self, source, camera, air, scale, flakes=True, haze=0.004):
        m = self.mpm
        self.density.zero_()
        self.illum.zero_()
        wp.launch(
            reconstruct,
            m.count,
            [
                m.x,
                m.vol,
                wp.from_torch(self.light),
                self.density,
                self.illum,
                wp.vec3(*m.origin),
                self.dx,
                m.phi,
                m.dx,
                wp.vec3(*self.geo["plane"]),
            ],
            device=m.device,
        )
        self.smooth(self.density)
        self.smooth(self.illum)
        self.coarse.zero_()
        self.coarse_light.zero_()
        wp.launch(
            reconstruct,
            m.count,
            [
                m.x,
                m.vol,
                wp.from_torch(self.light),
                self.coarse,
                self.coarse_light,
                wp.vec3(*m.origin),
                m.dx,
                m.phi,
                m.dx,
                wp.vec3(*self.geo["plane"]),
            ],
            device=m.device,
        )
        out = torch.empty((*source.shape[:2], 3), device=source.device)
        depth = torch.empty(source.shape[:2], device=source.device)
        inverse = torch.linalg.inv(camera.view).cpu().numpy()
        intr = camera.K.cpu().numpy()
        wp.launch(
            trace,
            source.shape[:2],
            [
                self.density,
                self.illum,
                self.coarse,
                m.phi,
                wp.vec3(*m.origin),
                self.dx,
                m.dx,
                wp.mat44(inverse),
                float(intr[0, 0]),
                float(intr[1, 1]),
                float(intr[0, 2]),
                float(intr[1, 2]),
                scale,
                wp.from_torch(source.contiguous()),
                wp.from_torch(out),
                wp.from_torch(depth),
                wp.vec3(*self.geo["plane"]),
                wp.vec4(*self.geo["bounds"]),
                0.11,
            ],
            device=m.device,
        )
        fog = (1 - torch.exp(-haze * depth)).clamp(0, 0.32)
        out = (
            out * (1 - fog[..., None])
            + out.new_tensor([0.72, 0.77, 0.83]) * fog[..., None]
        )
        if flakes:
            pos = (
                wp.to_torch(air.x) @ camera.view[:3, :3].T + camera.view[:3, 3] * scale
            )
            vel = wp.to_torch(air.v) @ camera.view[:3, :3].T
            tau = torch.zeros_like(depth)
            wp.launch(
                draw_flakes,
                len(pos),
                [
                    wp.from_torch(pos.contiguous(), dtype=wp.vec3),
                    wp.from_torch(vel.contiguous(), dtype=wp.vec3),
                    air.r,
                    wp.from_torch(depth),
                    wp.from_torch(tau),
                    float(intr[0, 0]),
                    float(intr[1, 1]),
                    float(intr[0, 2]),
                    float(intr[1, 2]),
                ],
                device=m.device,
            )
            alpha = 1 - torch.exp(-tau)
            out = (
                out * (1 - alpha[..., None])
                + out.new_tensor([0.88, 0.91, 0.95]) * alpha[..., None]
            )
        return out.clamp(0, 1)
