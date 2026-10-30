"""Local particle covariance for anisotropic liquid Gaussian reconstruction."""

import torch
import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def moments(
    x: wp.array(dtype=wp.vec3),
    grid: wp.uint64,
    radius: float,
    cov: wp.array(dtype=wp.mat33),
    counts: wp.array(dtype=float),
):
    i = wp.tid()
    mean = wp.vec3(0.0)
    moment = wp.mat33(0.0)
    total = float(0.0)
    count = float(0.0)
    for j in wp.hash_grid_query(grid, x[i], radius):
        r = x[j] - x[i]
        q = wp.max(0.0, 1.0 - wp.dot(r, r) / (radius * radius))
        weight = q * q * q
        mean += weight * r
        moment += weight * wp.outer(r, r)
        total += weight
        if weight > 0.05:
            count += 1.0
    mean = mean / wp.max(total, 1e-6)
    cov[i] = moment / wp.max(total, 1e-6) - wp.outer(mean, mean)
    counts[i] = count


class Anisotropy:
    def __init__(self, spacing, device="cuda:0"):
        self.spacing = spacing
        self.device = device
        self.grid = wp.HashGrid(128, 128, 128, device=device)

    @torch.no_grad()
    def __call__(self, x):
        positions = wp.from_torch(x.contiguous(), dtype=wp.vec3)
        covariance = wp.zeros(len(x), dtype=wp.mat33, device=self.device)
        counts = wp.zeros(len(x), dtype=float, device=self.device)
        radius = self.spacing * 2.5
        self.grid.build(positions, radius)
        wp.launch(
            moments,
            len(x),
            inputs=[positions, self.grid.id, radius, covariance, counts],
            device=self.device,
        )
        cov = wp.to_torch(covariance)
        values, vectors = torch.linalg.eigh(cov)
        values = values.clamp_min(self.spacing**2 * 0.04)
        lengths = values.sqrt()
        # Preserve kernel volume while stretching tangentially to sheets.
        lengths = lengths / lengths.log().mean(-1, keepdim=True).exp()
        lengths = lengths.clamp(0.55, 1.65)
        lengths = lengths / lengths.log().mean(-1, keepdim=True).exp()
        sparse = wp.to_torch(counts) < 8
        lengths[sparse] = 1.0
        sigma = lengths * self.spacing * 0.74
        return (vectors * sigma[:, None, :] ** 2) @ vectors.transpose(-1, -2)
