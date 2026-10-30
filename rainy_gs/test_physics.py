"""CPU regression checks for the numerical and world-space rain contract."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

import torch

from .physics import (
    ShallowWater,
    ImpactWaves,
    Rain,
    Terrain,
    sample_grid,
    depression_fill,
)
from .scene import Camera


def flat_terrain(n=32):
    terrain = SimpleNamespace(
        device="cpu",
        bounds=torch.tensor([-1.0, -1.0, 1.0, 1.0]),
        resolution=n,
        dx=2 / n,
        ground=torch.zeros(3),
        bed=torch.zeros(n, n),
        top=torch.zeros(n, n),
        obstacles=torch.zeros(n, n, dtype=torch.bool),
    )
    terrain.height = lambda xy: sample_grid(terrain.top, xy, terrain.bounds)
    return terrain


class PhysicsChecks(unittest.TestCase):
    def test_spatial_drag_slows_flow_without_removing_water(self):
        bed = torch.zeros(24, 24, dtype=torch.float64)
        low = ShallowWater(bed, 0.1, level=0.02, damping=0.0)
        high = ShallowWater(bed, 0.1, level=0.02, damping=torch.full_like(bed, 6.0))
        low.u[1] = low.u[0] * 0.2
        high.u[1] = high.u[0] * 0.2
        for _ in range(10):
            low.step(0.01)
            high.step(0.01)
        self.assertLess(
            float(high.u[1:].square().sum()), float(low.u[1:].square().sum())
        )
        self.assertLess(abs(high.diagnostics()["mass_residual"]), 1e-12)

    def test_disconnected_basins_have_different_spill_levels(self):
        bed = np.zeros((9, 15), dtype=np.float64)
        bed[2:7, 1:6] = 0.06
        bed[3:6, 2:5] = 0.03
        bed[2:7, 9:14] = 0.16
        bed[3:6, 10:13] = 0.12
        original = bed.copy()
        filled = depression_fill(bed)
        self.assertAlmostEqual(filled[4, 3], 0.06)
        self.assertAlmostEqual(filled[4, 11], 0.16)
        self.assertTrue(np.array_equal(bed, original))
        water = ShallowWater(
            torch.from_numpy(bed),
            0.1,
            level=-0.008,
            initial_depth=torch.from_numpy(filled - bed),
        )
        start = water.u.clone()
        for _ in range(30):
            water.step(0.02)
        self.assertLess(float((water.u - start).abs().max()), 1e-12)
        self.assertLess(abs(water.diagnostics()["mass_residual"]), 1e-12)

    def test_elevated_lawn_puddle_survives_surface_reconstruction(self):
        terrain = flat_terrain(32)
        terrain.bed[:] = 0.05
        water = ShallowWater(
            torch.full((16, 16), 0.05),
            0.125,
            level=-0.008,
            initial_depth=torch.full((16, 16), 0.012),
        )
        eta, depth = Terrain.water_surface(terrain, water)
        self.assertTrue(torch.allclose(eta, torch.full_like(eta, 0.062), atol=1e-7))
        self.assertTrue(torch.allclose(depth, torch.full_like(depth, 0.012), atol=1e-7))
        dry = ShallowWater(
            torch.full((16, 16), 0.05),
            0.125,
            level=0.2,
            initial_depth=torch.zeros(16, 16),
        )
        _, dry_depth = Terrain.water_surface(terrain, dry)
        self.assertEqual(float(dry_depth.max()), 0.0)

    def test_lake_at_rest_over_irregular_bed(self):
        x = torch.linspace(-1, 1, 32, dtype=torch.float64)
        yy, xx = torch.meshgrid(x, x, indexing="ij")
        bed = 0.025 * torch.cos(xx * 3) * torch.cos(yy * 2)
        water = ShallowWater(bed, 2 / 32, level=0.018)
        start = water.u.clone()
        for _ in range(40):
            water.step(0.025)
        self.assertLess(float((water.u - start).abs().max()), 1e-12)
        self.assertLess(abs(water.diagnostics()["mass_residual"]), 1e-12)

    def test_wet_dry_dam_break_positive_and_conservative(self):
        bed = torch.zeros((32, 32), dtype=torch.float64)
        water = ShallowWater(bed, 1 / 32, level=0, damping=0)
        water.u[0, :, :12] = 0.08
        water.initial_mass = float(water.u[0].sum()) * water.dx**2
        for _ in range(60):
            water.step(0.01)
        self.assertGreater(float(water.u[0, :, 16:].sum()), 0)
        self.assertGreaterEqual(float(water.u[0].min()), 0)
        self.assertLess(abs(water.diagnostics()["mass_residual"]), 1e-12)

    def test_rainfall_volume_balance(self):
        water = ShallowWater(torch.zeros(24, 24, dtype=torch.float64), 0.1, level=0.001)
        for _ in range(30):
            water.step(0.1, rain_rate=40 * 0.001 / 3600)
        report = water.diagnostics()
        self.assertAlmostEqual(
            report["rainfall_volume"], 3 * 40 * 0.001 / 3600 * 2.4**2, places=12
        )
        self.assertLess(abs(report["mass_residual"]), 1e-12)

    def test_surface_interpolation_preserves_flat_level(self):
        terrain = flat_terrain()
        terrain.bed = (
            torch.randn(32, 32, generator=torch.Generator().manual_seed(2)) * 0.001
        )
        coarse = torch.randn(16, 16, generator=torch.Generator().manual_seed(3)) * 0.001
        water = ShallowWater(coarse, 0.125, level=0.01)
        eta, depth = Terrain.water_surface(terrain, water)
        self.assertLess(float((eta - 0.01).abs().max()), 1e-8)
        self.assertLess(float((depth + terrain.bed - 0.01).abs().max()), 1e-8)

    def test_particle_replay_and_collisions(self):
        terrain = flat_terrain()
        a, b = Rain(terrain, 400, seed=17), Rain(terrain, 400, seed=17)
        initial = a.pos.clone()
        for _ in range(240):
            a.step(1 / 120)
            b.step(1 / 120)
            self.assertGreaterEqual(float(a.pos[:, 2].min()), 0)
        self.assertTrue(torch.equal(a.pos, b.pos))
        self.assertTrue(torch.equal(a.splash_pos, b.splash_pos))
        self.assertFalse(torch.equal(initial, a.pos))
        self.assertGreater(a.impacts, 0)

    def test_exposure_covariance_and_time_direction(self):
        rain = Rain(flat_terrain(), 16)
        exposure = 0.02
        means, covariance, _ = rain.streaks(exposure, segments=3)
        center = means.reshape(16, 3, 3).mean(1)
        self.assertTrue(
            torch.allclose(center, rain.pos - rain.vel * exposure / 2, atol=1e-6)
        )
        delta = means.reshape(16, 3, 3) - center[:, None]
        mixture = covariance.reshape(16, 3, 3, 3).mean(1) + (
            delta[..., None] * delta[..., None, :]
        ).mean(1)
        _, full, _ = rain.streaks(exposure, segments=1)
        self.assertTrue(torch.allclose(mixture, full, atol=1e-8))
        self.assertGreater(float(torch.linalg.eigvalsh(covariance).min()), 0)

    def test_impact_creates_finite_decaying_ripple(self):
        terrain = flat_terrain(48)
        waves = ImpactWaves(terrain)
        waves.impulse(torch.tensor([[0.0, 0.0]]), torch.tensor([0.02]))
        # A centered impulse has zero net displacement on this closed patch.
        self.assertLess(abs(float(waves.velocity.sum())), 1e-8)
        maxima = []
        for _ in range(300):
            waves.step(1 / 120, torch.full_like(terrain.bed, 0.015))
            maxima.append(float(waves.height.abs().max()))
        self.assertTrue(torch.isfinite(waves.height).all())
        self.assertGreater(max(maxima), 0)
        self.assertLess(maxima[-1], max(maxima) * 0.3)

    def test_camera_depth_roundtrip(self):
        view = torch.eye(4)
        view[:3, 3] = torch.tensor([1.0, -2.0, 0.5])
        k = torch.tensor([[50.0, 0.0, 4.0], [0.0, 55.0, 3.0], [0.0, 0.0, 1.0]])
        camera = Camera(view, k, 8, 6)
        depth = torch.full((6, 8), 3.0)
        xyz = camera.world_points(depth) @ view[:3, :3].T + view[:3, 3]
        uv = xyz @ k.T
        uv = uv[..., :2] / uv[..., 2:]
        yy, xx = torch.meshgrid(
            torch.arange(6) + 0.5, torch.arange(8) + 0.5, indexing="ij"
        )
        self.assertTrue(torch.allclose(uv, torch.stack((xx, yy), -1), atol=1e-5))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
