"""Bidirectional local embedding, covariance transport and moment splitting."""

import numpy as np
import torch
from .warp_fem import WarpEmbedding


def polar(f):
    u, _, vh = torch.linalg.svd(f)
    sign = torch.linalg.det(u @ vh)
    u = u.clone()
    u[:, :, -1] *= sign[:, None]
    return u @ vh


def split_moments(means, axes, opacity):
    """Half-Gaussian moments; the printed reciprocal kappa is corrected.

    For sigma along the cut normal: displacement=sigma*sqrt(2/pi), and the
    child variance is sigma²*(1-2/pi). Weighted moments 0,1,2 are preserved.
    """
    scales = torch.linalg.vector_norm(axes, dim=1)
    major = scales.argmax(-1)
    axis = axes[torch.arange(len(axes), device=axes.device), :, major]
    offset = axis * np.sqrt(2 / np.pi)
    child_axes = axes.clone()
    child_axes[torch.arange(len(axes), device=axes.device), :, major] *= np.sqrt(
        1 - 2 / np.pi
    )
    return (
        torch.cat([means - offset, means + offset]),
        torch.cat([child_axes, child_axes]),
        opacity.repeat(2) * 0.5,
    )


class GaussianBody:
    def __init__(self, cage, data, device="cuda:0"):
        self.cage, self.device = cage, device

        def t(x):
            return torch.as_tensor(x, dtype=torch.float64, device=device)

        self.means = t(data["means"])
        self.axes = t(data["axes"])
        self.opacity = t(data["opacity"])
        self.sh = t(data["sh"])
        self.source_rotation = t(data["source_rotation"])
        self.generation = torch.zeros(len(self.means), dtype=torch.long, device=device)
        self.total_split = 0
        self.rebind()

    def rebind(self):
        rest = torch.stack(
            [self.means]
            + [
                self.means + s * 3 * self.axes[:, :, i]
                for s in [-1, 1]
                for i in range(3)
            ],
            1,
        )
        self.embedding = WarpEmbedding(
            self.cage, rest.reshape(-1, 3).cpu().numpy(), self.device
        )
        # Some support-plane splats are extremely flat; inverse is regularized
        # only for rotation estimation, not for the transported covariance.
        self.inverse = torch.linalg.pinv(3 * self.axes, rtol=1e-9)

    def deform(self, x):
        points = self.embedding(x).reshape(-1, 7, 3)
        means = points[:, 0]
        dp = (points[:, 4:7] - means[:, None]).transpose(1, 2)
        dm = (means[:, None] - points[:, 1:4]).transpose(1, 2)
        fp = dp @ self.inverse
        fm = dm @ self.inverse
        rp, rm = polar(fp), polar(fm)
        trace = (rp * rm).sum((1, 2))
        angle = torch.acos(((trace - 1) / 2).clamp(-1, 1))
        covariance = dp @ dp.transpose(1, 2) / 9
        covariance += torch.eye(3, device=self.device) * np.exp(-20)
        return means, covariance, rp, angle

    def split(self, angle, threshold_degrees=15, max_new=1500):
        ids = torch.where(
            (angle > np.deg2rad(threshold_degrees))
            & (self.generation < 3)
            & (self.opacity > 0.05)
        )[0]
        if not len(ids):
            return 0
        ids = ids[torch.argsort(angle[ids], descending=True)[:max_new]]
        means, axes, opacity = split_moments(
            self.means[ids], self.axes[ids], self.opacity[ids]
        )
        # Bind prospective children before changing the object.
        probe = torch.stack(
            [means]
            + [means + s * 3 * axes[:, :, i] for s in [-1, 1] for i in range(3)],
            1,
        )
        try:
            self.cage.bind(probe.reshape(-1, 3).cpu().numpy())
        except ValueError:
            return 0
        keep = torch.ones(len(self.means), dtype=torch.bool, device=self.device)
        keep[ids] = False
        self.means = torch.cat([self.means[keep], means])
        self.axes = torch.cat([self.axes[keep], axes])
        self.opacity = torch.cat([self.opacity[keep], opacity])
        self.sh = torch.cat([self.sh[keep], self.sh[ids].repeat(2, 1, 1)])
        self.generation = torch.cat(
            [self.generation[keep], (self.generation[ids] + 1).repeat(2)]
        )
        self.total_split += len(ids)
        self.rebind()
        return len(ids)
