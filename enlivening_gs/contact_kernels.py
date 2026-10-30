"""Compiled small-contact LCP sweep, sharing the reference pyramid friction."""

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _lcp(
    q: wp.array(dtype=wp.float64),
    w: wp.array2d(dtype=wp.float64),
    gap: wp.array(dtype=wp.float64),
    dt: wp.float64,
    mu: wp.float64,
    impulse: wp.array(dtype=wp.float64),
    velocity: wp.array(dtype=wp.float64),
    n: int,
    maxiter: int,
    used: wp.array(dtype=wp.int32),
):
    for j in range(3 * n):
        val = q[j]
        for k in range(3 * n):
            val += w[j, k] * impulse[k]
        velocity[j] = val
    for it in range(maxiter):
        change = wp.float64(0.0)
        for i in range(n):
            off = 3 * i
            oldx, oldy, oldz = impulse[off], impulse[off + 1], impulse[off + 2]
            normal = wp.max(
                wp.float64(0.0),
                oldz
                - (velocity[off + 2] + gap[i] / dt)
                / wp.max(w[off + 2, off + 2], wp.float64(1e-12)),
            )
            dz = normal - oldz
            for j in range(3 * n):
                velocity[j] += w[j, off + 2] * dz
            a = w[off, off] + wp.float64(1e-10)
            b = w[off, off + 1]
            c = w[off + 1, off]
            d = w[off + 1, off + 1] + wp.float64(1e-10)
            det = a * d - b * c
            tx = oldx - (d * velocity[off] - b * velocity[off + 1]) / det
            ty = oldy - (-c * velocity[off] + a * velocity[off + 1]) / det
            radius = mu * normal
            if wp.abs(tx) + wp.abs(ty) > radius:
                hi = wp.max(wp.abs(tx), wp.abs(ty))
                lo = wp.min(wp.abs(tx), wp.abs(ty))
                threshold = hi - radius
                if hi - lo < radius:
                    threshold = (hi + lo - radius) * wp.float64(0.5)
                tx = wp.sign(tx) * wp.max(wp.abs(tx) - threshold, wp.float64(0.0))
                ty = wp.sign(ty) * wp.max(wp.abs(ty) - threshold, wp.float64(0.0))
            dx, dy = tx - oldx, ty - oldy
            for j in range(3 * n):
                velocity[j] += w[j, off] * dx + w[j, off + 1] * dy
            impulse[off] = tx
            impulse[off + 1] = ty
            impulse[off + 2] = normal
            change = wp.max(change, wp.max(wp.abs(dx), wp.max(wp.abs(dy), wp.abs(dz))))
        used[0] = it + 1
        if change < wp.float64(1e-8):
            break


def forward_lcp(velocity, delassus, gap, dt, mu, start=None, iterations=180):
    n = len(gap)
    if not n:
        return np.zeros((0, 3)), np.zeros((0, 3)), 0

    def a(x):
        return wp.array(
            np.ascontiguousarray(x, dtype=np.float64), dtype=wp.float64, device="cpu"
        )

    impulse = a(np.zeros(3 * n) if start is None else start.ravel())
    out = wp.empty(3 * n, dtype=wp.float64, device="cpu")
    used = wp.zeros(1, dtype=wp.int32, device="cpu")
    wp.launch(
        _lcp,
        1,
        [a(velocity), a(delassus), a(gap), dt, mu, impulse, out, n, iterations, used],
        device="cpu",
    )
    return (
        impulse.numpy().reshape(n, 3),
        out.numpy().reshape(n, 3),
        int(used.numpy()[0]),
    )
