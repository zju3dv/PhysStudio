"""Numerical checks for moisture transfer, projection, and renderer ordering."""
import argparse, json
from pathlib import Path
import numpy as np
import warp as wp
from .solver import (
    Storm,
    Settings,
    phase_change,
    divergence,
    fluid,
    advect_dust,
    correct_dust,
)
from .render import composite


@wp.kernel
def microphysics(a: wp.array(dtype=wp.vec4), out: wp.array(dtype=wp.vec4), dt: float):
    i = wp.tid()
    out[i] = phase_change(a[i], 101325.0, dt)


@wp.kernel
def perturbed_velocity(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
):
    i, j, k = wp.tid()
    if fluid(solid, i - 1, j, k) * fluid(solid, i, j, k) == 1:
        u[i, j, k] = wp.sin(float(i) * 0.73 + float(j) * 0.25)
    if fluid(solid, i, j - 1, k) * fluid(solid, i, j, k) == 1:
        v[i, j, k] = wp.cos(float(i) * 0.31 - float(j) * 0.6 + float(k) * 0.2)
    if fluid(solid, i, j, k - 1) * fluid(solid, i, j, k) == 1:
        w[i, j, k] = wp.sin(float(k) * 0.52 + float(j) * 0.41)


def run(device="cuda:0"):
    rng = np.random.default_rng(17)
    a = np.zeros((4096, 4), dtype="f")
    a[:, 0] = rng.uniform(283, 307, len(a))
    a[:, 1] = rng.uniform(0.002, 0.035, len(a))
    a[:, 2:] = rng.uniform(0, 0.004, (len(a), 2))
    source = wp.array(a, dtype=wp.vec4, device=device)
    out = wp.empty_like(source)
    wp.launch(microphysics, len(a), [source, out, 0.4], device=device)
    b = out.numpy()
    water = float(np.max(np.abs(a[:, 1:].sum(-1) - b[:, 1:].sum(-1))))
    enthalpy = float(
        np.max(np.abs((a[:, 0] + 2490.04 * a[:, 1]) - (b[:, 0] + 2490.04 * b[:, 1])))
    )
    assert water < 1e-7 and enthalpy < 1e-4
    assert b[:, 1:].min() >= -1e-8
    s = Storm(
        Settings(shape=(20, 20, 24), dx=0.6, height=11.0, pressure_iterations=150),
        device=device,
    )
    s.u.zero_()
    s.v.zero_()
    s.w.zero_()
    s.p.zero_()
    wp.launch(perturbed_velocity, s.face_shape, [s.u, s.v, s.w, s.solid], device=device)
    wp.launch(
        divergence,
        s.shape,
        [s.u, s.v, s.w, s.div, s.solid, s.settings.dx],
        device=device,
    )
    before = float(np.sqrt(np.mean(s.div.numpy() ** 2)))
    s.project()
    wp.launch(
        divergence,
        s.shape,
        [s.u, s.v, s.w, s.div, s.solid, s.settings.dx],
        device=device,
    )
    after = float(np.sqrt(np.mean(s.div.numpy() ** 2)))
    assert after / before < 0.01, (before, after)
    for _ in range(20):
        s.step(0.01)
    diagnostics = s.diagnostics()
    assert diagnostics["minimum_water"] >= -1e-7
    solid = s.solid.numpy()
    wall_max = 0.0
    for axis, component in enumerate([s.u, s.v, s.w]):
        face = component.numpy()
        mask = np.zeros(s.face_shape, dtype=bool)
        lo = [slice(0, n) for n in s.shape]
        hi = lo.copy()
        hi[axis] = slice(1, s.shape[axis] + 1)
        mask[tuple(lo)] |= solid.astype(bool)
        mask[tuple(hi)] |= solid.astype(bool)
        edge = [slice(None)] * 3
        edge[axis] = 0
        mask[tuple(edge)] = True
        edge[axis] = -1
        mask[tuple(edge)] = True
        wall_max = max(wall_max, float(np.abs(face[mask]).max()))
    assert wall_max == 0.0, wall_max
    # A translated Gaussian tests reduced diffusion without creating extrema.
    t = Storm(
        Settings(shape=(40, 8, 8), dx=0.5, height=3.0, pressure_iterations=2),
        device=device,
    )
    t.u.fill_(1.0)
    t.v.zero_()
    t.w.zero_()
    t.solid.zero_()
    xx = np.arange(40, dtype="f")[:, None, None] + 0.5
    pulse = np.broadcast_to(np.exp(-(((xx - 12.0) / 2.5) ** 2)), t.shape).copy()
    first = wp.array(pulse, dtype=float, device=device)
    limited = wp.array(pulse, dtype=float, device=device)
    forward = wp.empty_like(first)
    corrected = wp.empty_like(first)
    simple = wp.empty_like(first)
    for _ in range(20):
        wp.launch(
            advect_dust,
            t.shape,
            [t.u, t.v, t.w, first, simple, t.c, 0.1, 0.0],
            device=device,
        )
        wp.launch(
            advect_dust,
            t.shape,
            [t.u, t.v, t.w, limited, forward, t.c, 0.1, 0.0],
            device=device,
        )
        wp.launch(
            correct_dust,
            t.shape,
            [t.u, t.v, t.w, limited, forward, corrected, t.solid, t.c, 0.1, 0.0],
            device=device,
        )
        first, simple = simple, first
        limited, corrected = corrected, limited
    target = np.broadcast_to(np.exp(-(((xx - 16.0) / 2.5) ** 2)), t.shape)
    first_error = float(np.sqrt(np.mean((first.numpy() - target) ** 2)))
    limited_np = limited.numpy()
    limited_error = float(np.sqrt(np.mean((limited_np - target) ** 2)))
    assert limited_error < first_error * 0.7, (limited_error, first_error)
    assert limited_np.min() >= 0 and limited_np.max() <= pulse.max() + 1e-6
    # An opaque surface in front of the entire volume must receive no fog.
    shape = (4, 4)
    rgb = np.ones((*shape, 3), dtype="f") * 0.4
    alpha = np.ones(shape, dtype="f")
    depth = np.ones(shape, dtype="f")
    rays = np.zeros((*shape, 3), dtype="f")
    rays[:, :, 2] = -1
    field = wp.array(np.ones((*s.shape, 4), dtype="f"), dtype=wp.vec4, device=device)
    image = wp.zeros(shape, dtype=wp.vec3, device=device)
    opacity = wp.zeros(shape, dtype=float, device=device)
    zero = wp.zeros(shape, dtype=float, device=device)
    zero_color = wp.zeros(shape, dtype=wp.vec3, device=device)
    wp.launch(
        composite,
        shape,
        [
            wp.array(rgb, dtype=wp.vec3, device=device),
            wp.array(alpha, dtype=float, device=device),
            wp.array(depth, dtype=float, device=device),
            wp.array(rays, dtype=wp.vec3, device=device),
            field,
            zero_color,
            zero,
            zero,
            image,
            opacity,
            wp.vec3(0.0, 0.0, -10.0),
            s.c,
            0.0,
            wp.vec3(0.5),
            1.0,
            0,
            1.0,
        ],
        device=device,
    )
    assert float(opacity.numpy().max()) == 0
    assert np.max(np.abs(image.numpy() - rgb)) < 1e-6
    return dict(
        passed=True,
        water_transfer_max_error=water,
        latent_enthalpy_scaled_error=enthalpy,
        projection_divergence_before=before,
        projection_divergence_after=after,
        projection_ratio=after / before,
        solid_normal_speed=wall_max,
        foreground_occlusion_passed=True,
        dust_advection_rmse=dict(first_order=first_error, maccormack=limited_error),
        short_rollout=diagnostics,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = run()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
