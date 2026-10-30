"""World-space water refraction, granular surfaces, and cinematic orbit views."""

import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import warp as wp
from .common import weights
from rainy_gs.demo import Video


@wp.kernel(enable_backward=False)
def reconstruct(
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
    b = wp.vec3i(
        int(wp.floor(point[0] - 0.5)),
        int(wp.floor(point[1] - 0.5)),
        int(wp.floor(point[2] - 0.5)),
    )
    f = point - wp.vec3(float(b[0]), float(b[1]), float(b[2]))
    wx, wy, wz = weights(f[0]), weights(f[1]), weights(f[2])
    for a in range(3):
        for c in range(3):
            for d in range(3):
                i, j, k = b[0] + a, b[1] + c, b[2] + d
                if (
                    i >= 0
                    and j >= 0
                    and k >= 0
                    and i < den.shape[0]
                    and j < den.shape[1]
                    and k < den.shape[2]
                ):
                    w = wx[a] * wy[c] * wz[d] * 8.0 * volume
                    wp.atomic_add(den, i, j, k, w)
                    wp.atomic_add(wet, i, j, k, w * water[p] / (volume * 0.293))
                    wp.atomic_add(pigment, i, j, k, w * color[p])


@wp.kernel(enable_backward=False)
def smooth_fields(
    den: wp.array3d(dtype=float),
    wet: wp.array3d(dtype=float),
    col: wp.array3d(dtype=wp.vec3),
    out_den: wp.array3d(dtype=float),
    out_wet: wp.array3d(dtype=float),
    out_col: wp.array3d(dtype=wp.vec3),
    axis: int,
):
    i, j, k = wp.tid()
    total, d, w = float(0.0), float(0.0), float(0.0)
    c = wp.vec3(0.0)
    for offset in range(-4, 5):
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
            weight = wp.exp(-float(offset * offset) / (2.0 * 1.5 * 1.5))
            total += weight
            d += weight * den[p[0], p[1], p[2]]
            w += weight * wet[p[0], p[1], p[2]]
            c += weight * col[p[0], p[1], p[2]]
    out_den[i, j, k] = d / total
    out_wet[i, j, k] = w / total
    out_col[i, j, k] = c / total


@wp.func
def sample(a: wp.array3d(dtype=float), o: wp.vec3, dx: float, p: wp.vec3):
    f = (p - o) / dx
    value = float(0.0)
    if (
        f[0] >= 0.0
        and f[1] >= 0.0
        and f[2] >= 0.0
        and f[0] < float(a.shape[0] - 1)
        and f[1] < float(a.shape[1] - 1)
        and f[2] < float(a.shape[2] - 1)
    ):
        b = wp.vec3i(int(f[0]), int(f[1]), int(f[2]))
        f -= wp.vec3(float(b[0]), float(b[1]), float(b[2]))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    w = wp.vec3(1.0) - f
                    if i == 1:
                        w[0] = f[0]
                    if j == 1:
                        w[1] = f[1]
                    if k == 1:
                        w[2] = f[2]
                    value += w[0] * w[1] * w[2] * a[b[0] + i, b[1] + j, b[2] + k]
    return value


@wp.func
def sample_color(a: wp.array3d(dtype=wp.vec3), o: wp.vec3, dx: float, p: wp.vec3):
    f = (p - o) / dx
    value = wp.vec3(0.0)
    if (
        f[0] >= 0.0
        and f[1] >= 0.0
        and f[2] >= 0.0
        and f[0] < float(a.shape[0] - 1)
        and f[1] < float(a.shape[1] - 1)
        and f[2] < float(a.shape[2] - 1)
    ):
        b = wp.vec3i(int(f[0]), int(f[1]), int(f[2]))
        f -= wp.vec3(float(b[0]), float(b[1]), float(b[2]))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    w = wp.vec3(1.0) - f
                    if i == 1:
                        w[0] = f[0]
                    if j == 1:
                        w[1] = f[1]
                    if k == 1:
                        w[2] = f[2]
                    value += w[0] * w[1] * w[2] * a[b[0] + i, b[1] + j, b[2] + k]
    return value


@wp.func
def normal(a: wp.array3d(dtype=float), o: wp.vec3, dx: float, p: wp.vec3):
    h = dx * 0.65
    n = wp.vec3(
        sample(a, o, dx, p - wp.vec3(h, 0.0, 0.0))
        - sample(a, o, dx, p + wp.vec3(h, 0.0, 0.0)),
        sample(a, o, dx, p - wp.vec3(0.0, h, 0.0))
        - sample(a, o, dx, p + wp.vec3(0.0, h, 0.0)),
        sample(a, o, dx, p - wp.vec3(0.0, 0.0, h))
        - sample(a, o, dx, p + wp.vec3(0.0, 0.0, h)),
    )
    return n / (wp.length(n) + 1.0e-12)


@wp.func
def environment(d: wp.vec3):
    sky = wp.vec3(0.14, 0.22, 0.30) * (0.75 + 0.25 * wp.max(d[2], 0.0))
    panel = wp.exp(
        -wp.pow((d[0] + 0.32) / 0.52, 8.0) - wp.pow((d[1] + 0.28) / 0.20, 8.0)
    )
    panel *= wp.clamp((d[2] - 0.20) * 8.0, 0.0, 1.0)
    strip = wp.exp(
        -wp.pow((d[0] - 0.62) / 0.075, 2.0) - wp.pow((d[1] - 0.20) / 0.45, 8.0)
    )
    return sky + wp.vec3(1.8, 1.72, 1.57) * panel + wp.vec3(0.6, 0.9, 1.3) * strip


@wp.func
def hash_noise(p: wp.vec3):
    n = wp.sin(wp.dot(p, wp.vec3(127.1, 311.7, 74.7))) * 43758.5453
    return n - wp.floor(n)


@wp.func
def background(ro: wp.vec3, rd: wp.vec3, extent: wp.vec3):
    t = (-0.035 - ro[2]) / (rd[2] + 1.0e-12)
    col = environment(rd) * 0.6
    if t > 0.0:
        p = ro + t * rd
        center = wp.vec3(extent[0] * 0.5, extent[1] * 0.5, 0.0)
        dist = wp.length(p - center)
        col = wp.vec3(0.065, 0.085, 0.105) + wp.vec3(0.05, 0.06, 0.07) * wp.exp(
            -dist * 0.25
        )
        tray = (
            p[0] > -0.10
            and p[0] < extent[0] + 0.10
            and p[1] > -0.10
            and p[1] < extent[1] + 0.10
        )
        if tray:
            # A ceramic basin provides stable refractive and shadow cues.
            tiles = wp.sin(p[0] * 24.0) * wp.sin(p[1] * 24.0)
            col = wp.vec3(0.24, 0.29, 0.31) * (1.0 + 0.025 * tiles)
            edge = wp.min(
                wp.min(p[0] + 0.10, extent[0] + 0.10 - p[0]),
                wp.min(p[1] + 0.10, extent[1] + 0.10 - p[1]),
            )
            if edge < 0.035:
                col = wp.vec3(0.09, 0.12, 0.14)
    return col


@wp.func
def sand_shade(
    den: wp.array3d(dtype=float),
    wet: wp.array3d(dtype=float),
    pigment: wp.array3d(dtype=wp.vec3),
    o: wp.vec3,
    dx: float,
    p: wp.vec3,
    rd: wp.vec3,
):
    n = normal(den, o, dx, p)
    if wp.dot(n, rd) > 0.0:
        n = -n
    light = wp.normalize(wp.vec3(-0.6, -0.4, 1.0))
    shadow = float(1.0)
    for i in range(1, 13):
        q = p + n * dx * 1.8 + light * float(i) * dx * 2.5
        if sample(den, o, dx, q) > 0.38:
            shadow = 0.30
    density = wp.max(sample(den, o, dx, p), 0.05)
    moisture = wp.clamp(sample(wet, o, dx, p) / density, 0.0, 1.0)
    base = sample_color(pigment, o, dx, p) / density
    noise = hash_noise(
        wp.vec3(wp.floor(p[0] * 450.0), wp.floor(p[1] * 450.0), wp.floor(p[2] * 450.0))
    )
    grain = 0.84 + 0.26 * noise
    base *= grain * (1.0 - 0.46 * moisture)
    ao = float(0.0)
    for i in range(1, 5):
        ao += wp.clamp(
            sample(den, o, dx, p + n * dx * float(i) * 1.5), 0.0, 1.0
        ) / float(i)
    diffuse = (0.42 + 0.58 * wp.max(wp.dot(n, light), 0.0) * shadow) * wp.exp(
        -ao * 0.38
    )
    view = -rd
    half = wp.normalize(light + view)
    glint = wp.pow(wp.max(wp.dot(n, half), 0.0), 65.0) * (0.02 + 0.07 * moisture)
    return base * diffuse + wp.vec3(glint)


@wp.func
def floor_shadow(
    den: wp.array3d(dtype=float), o: wp.vec3, dx: float, ro: wp.vec3, rd: wp.vec3
):
    value = float(1.0)
    t = (-0.035 - ro[2]) / (rd[2] + 1.0e-12)
    if t > 0.0:
        point = ro + t * rd
        light = wp.normalize(wp.vec3(-0.6, -0.4, 1.0))
        for i in range(1, 33):
            q = point + light * float(i) * dx * 2.5
            density = sample(den, o, dx, q)
            value *= wp.exp(-wp.clamp(density, 0.0, 1.0) * 0.16)
    return wp.max(value, 0.40)


@wp.func
def march(
    phi: wp.array3d(dtype=float),
    den: wp.array3d(dtype=float),
    o: wp.vec3,
    dx: float,
    ro: wp.vec3,
    rd: wp.vec3,
    water_enabled: int,
):
    hi = o + dx * wp.vec3(
        float(phi.shape[0] - 1), float(phi.shape[1] - 1), float(phi.shape[2] - 1)
    )
    near, far = float(0.0), float(30.0)
    for axis in range(3):
        t0 = (o[axis] - ro[axis]) / (rd[axis] + 1.0e-12)
        t1 = (hi[axis] - ro[axis]) / (rd[axis] + 1.0e-12)
        near = wp.max(near, wp.min(t0, t1))
        far = wp.min(far, wp.max(t0, t1))
    t = near
    result = wp.vec4(-1.0, 0.0, 0.0, 0.0)
    thickness, mud = float(0.0), float(0.0)
    for step in range(900):
        if t > far:
            break
        p = ro + t * rd
        d = sample(den, o, dx * 0.5, p)
        w = sample(phi, o, dx, p)
        hit = int(0)
        if d > 0.38:
            hit = 2
        elif water_enabled == 1 and w > 0.5:
            hit = 1
        if hit > 0:
            lo = wp.max(near, t - dx * 0.45)
            high = t
            for it in range(5):
                middle = (lo + high) * 0.5
                inside = sample(den, o, dx * 0.5, ro + middle * rd) > 0.38
                if hit == 1:
                    inside = sample(phi, o, dx, ro + middle * rd) > 0.5
                if inside:
                    high = middle
                else:
                    lo = middle
            result = wp.vec4((lo + high) * 0.5, float(hit), thickness, mud)
            break
        thickness += wp.clamp(w, 0.0, 1.0) * dx * 0.45
        mud += wp.clamp(d, 0.0, 0.38) * dx * 0.45
        t += dx * 0.45
    if result[0] < 0.0:
        result = wp.vec4(-1.0, 0.0, thickness, mud)
    return result


@wp.kernel(enable_backward=False)
def trace(
    phi: wp.array3d(dtype=float),
    den: wp.array3d(dtype=float),
    wet: wp.array3d(dtype=float),
    pigment: wp.array3d(dtype=wp.vec3),
    o: wp.vec3,
    dx: float,
    eye: wp.vec3,
    right: wp.vec3,
    up: wp.vec3,
    forward: wp.vec3,
    focal: float,
    extent: wp.vec3,
    result: wp.array3d(dtype=float),
):
    iy, ix = wp.tid()
    height, width = result.shape[0], result.shape[1]
    sx = (float(ix) + 0.5 - float(width) * 0.5) / focal
    sy = (float(height) * 0.5 - float(iy) - 0.5) / focal
    ray = wp.normalize(forward + sx * right + sy * up)
    hit = march(phi, den, o, dx, eye, ray, 1)
    col = background(eye, ray, extent) * floor_shadow(den, o, dx * 0.5, eye, ray)
    if hit[1] == 2.0:
        point = eye + hit[0] * ray
        col = sand_shade(den, wet, pigment, o, dx * 0.5, point, ray)
    elif hit[1] == 1.0:
        point = eye + hit[0] * ray
        n = normal(phi, o, dx, point)
        if wp.dot(n, ray) > 0.0:
            n = -n
        cosine = wp.clamp(-wp.dot(n, ray), 0.0, 1.0)
        eta = 1.0 / 1.333
        refract = wp.normalize(
            eta * ray
            + (
                eta * cosine
                - wp.sqrt(wp.max(0.0, 1.0 - eta * eta * (1.0 - cosine * cosine)))
            )
            * n
        )
        start = point + refract * dx * 0.08
        back = march(phi, den, o, dx, start, refract, 0)
        body = background(start, refract, extent) * floor_shadow(
            den, o, dx * 0.5, start, refract
        )
        if back[1] == 2.0:
            body = sand_shade(
                den, wet, pigment, o, dx * 0.5, start + back[0] * refract, refract
            )
        thickness = back[2]
        mud = back[3]
        absorb = wp.vec3(
            wp.exp(-thickness * 0.85 - mud * 10.0),
            wp.exp(-thickness * 0.19 - mud * 13.0),
            wp.exp(-thickness * 0.09 - mud * 18.0),
        )
        scatter = wp.vec3(0.025, 0.15, 0.18) * (
            1.0 - wp.clamp(mud * 4.0, 0.0, 1.0)
        ) + wp.vec3(0.34, 0.21, 0.075) * wp.clamp(mud * 4.0, 0.0, 1.0)
        body = wp.cw_mul(body, absorb) + wp.cw_mul(scatter, wp.vec3(1.0) - absorb)
        reflected = ray - 2.0 * wp.dot(ray, n) * n
        fresnel = 0.0204 + 0.9796 * wp.pow(1.0 - cosine, 5.0)
        col = (1.0 - fresnel) * body + fresnel * environment(reflected)
        light = wp.normalize(wp.vec3(-0.6, -0.4, 1.0))
        half = wp.normalize(light - ray)
        col += wp.vec3(0.8, 0.76, 0.65) * wp.pow(wp.max(wp.dot(n, half), 0.0), 220.0)
    # A filmic shoulder keeps reflections detailed without neon-looking water.
    col = wp.cw_div(
        wp.cw_mul(col, 2.51 * col + wp.vec3(0.03)),
        wp.cw_mul(col, 2.43 * col + wp.vec3(0.59)) + wp.vec3(0.14),
    )
    for a in range(3):
        result[iy, ix, a] = wp.pow(wp.clamp(col[a], 0.0, 1.0), 1.0 / 2.2)


class Renderer:
    def __init__(self, cfg, rest, width=1280, height=800):
        self.cfg, self.width, self.height = cfg, width, height
        shape = tuple(cfg["shape"])
        fine = tuple(n * 2 for n in shape)
        self.phi = wp.zeros(shape, dtype=float)
        self.den = wp.zeros(fine, dtype=float)
        self.wet = wp.zeros(fine, dtype=float)
        self.pigment = wp.zeros(fine, dtype=wp.vec3)
        self.smooth_den = wp.zeros(fine, dtype=float)
        self.smooth_wet = wp.zeros(fine, dtype=float)
        self.smooth_pigment = wp.zeros(fine, dtype=wp.vec3)
        self.result = wp.zeros((height, width, 3), dtype=float)
        points = (rest - 3) * cfg["dx"]
        color = np.tile(np.array([0.71, 0.43, 0.16], np.float32), (len(rest), 1))
        if cfg["scene"] == "sediment":
            for lo, hi, c in [
                (-1, 1.3, [0.78, 0.48, 0.16]),
                (1.3, 2.3, [0.49, 0.20, 0.075]),
                (2.3, 5, [0.83, 0.68, 0.38]),
            ]:
                color[(points[:, 0] >= lo) & (points[:, 0] < hi)] = c
        else:
            color *= (1.0 + 0.10 * np.sin(points[:, 2] * 35.0))[:, None]
        self.colors = wp.array(color, dtype=wp.vec3)

    def draw(self, state, progress):
        self.phi.assign(state["phi"].astype(np.float32))
        self.den.zero_()
        self.wet.zero_()
        self.pigment.zero_()
        x = wp.array(state["positions"], dtype=wp.vec3)
        water = wp.array(state["moisture"], dtype=float)
        wp.launch(
            reconstruct,
            len(x),
            inputs=[
                x,
                water,
                self.colors,
                self.den,
                self.wet,
                self.pigment,
                1.0 / self.cfg["ppc"] ** 3,
            ],
        )
        for axis in range(3):
            wp.launch(
                smooth_fields,
                self.den.shape,
                inputs=[
                    self.den,
                    self.wet,
                    self.pigment,
                    self.smooth_den,
                    self.smooth_wet,
                    self.smooth_pigment,
                    axis,
                ],
            )
            self.den, self.smooth_den = self.smooth_den, self.den
            self.wet, self.smooth_wet = self.smooth_wet, self.wet
            self.pigment, self.smooth_pigment = self.smooth_pigment, self.pigment
        t = progress * progress * (3.0 - 2.0 * progress)
        angle = np.deg2rad(-112 + 62 * t)
        elev = np.deg2rad(29 + 7 * np.sin(progress * np.pi))
        radius = 6.9 - 0.4 * np.sin(progress * np.pi)
        target = np.array([1.80, 1.15, 0.40])
        eye = target + radius * np.array(
            [np.cos(angle) * np.cos(elev), np.sin(angle) * np.cos(elev), np.sin(elev)]
        )
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        dx = self.cfg["dx"]
        extent = (np.array(self.cfg["shape"]) - 6) * dx
        focal = self.width / (2 * np.tan(np.deg2rad(43) * 0.5))
        wp.launch(
            trace,
            (self.height, self.width),
            inputs=[
                self.phi,
                self.den,
                self.wet,
                self.pigment,
                wp.vec3(-3 * dx),
                dx,
                wp.vec3(*eye),
                wp.vec3(*right),
                wp.vec3(*up),
                wp.vec3(*forward),
                focal,
                wp.vec3(*extent),
                self.result,
            ],
        )
        return self.result.numpy(), dict(
            eye=eye.tolist(), target=target.tolist(), focal=focal
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a fresh presentation directory")
    (out / "frames").mkdir(parents=True)
    cfg = json.loads((args.input / "config.json").read_text())
    rest = np.load(args.input / "initial.npz")["positions"]
    paths = sorted((args.input / "states").glob("*.npz"))[:: args.stride]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError("No saved simulation frames")
    wp.init()
    wp.set_device("cuda:0")
    renderer = Renderer(cfg, rest, args.width, args.height)
    video = Video(out / "demo.mp4", args.width, args.height, cfg["fps"])
    cameras = []
    start = time.time()
    try:
        for ix, path in enumerate(paths):
            state = dict(np.load(path))
            progress = int(path.stem) / max(cfg["frames"] - 1, 1)
            rgb, cam = renderer.draw(state, progress)
            image = np.uint8(np.clip(rgb, 0, 1) * 255 + 0.5)
            Image.fromarray(image).save(out / "frames" / f"{ix:05d}.png")
            video.write(image)
            cameras.append(cam)
            if ix % 10 == 0 or ix == len(paths) - 1:
                print(
                    f"render {ix + 1}/{len(paths)} {time.time() - start:.1f}s",
                    flush=True,
                )
    finally:
        video.close()
    (out / "camera_path.json").write_text(json.dumps(cameras, indent=2))
    (out / "config.json").write_text(
        json.dumps(
            dict(
                cfg,
                source=str(args.input.resolve()),
                render_seconds=time.time() - start,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
