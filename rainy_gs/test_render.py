"""Small CUDA integration checks for foreground/background rain occlusion."""

from types import SimpleNamespace
import unittest
import torch

from .scene import Camera, Scene
from .demo import Compositor


class RainOcclusionChecks(unittest.TestCase):
    def test_rigid_calibration_preserves_source_sh_render(self):
        scene = Scene.__new__(Scene)
        scene.device = "cuda"
        scene.render_means = torch.tensor(
            [[0.1, 0.05, 2.0], [-0.15, -0.1, 2.5]], device="cuda"
        )
        scene.quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device="cuda")
        scene.scales = torch.full((2, 3), 0.07, device="cuda")
        scene.opacity = torch.full((2,), 0.9, device="cuda")
        scene.sh = torch.arange(96, device="cuda").reshape(2, 16, 3).float() / 300
        scene.world_transform = torch.eye(4, device="cuda")
        k = torch.tensor(
            [[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]], device="cuda"
        )
        camera = Camera(torch.eye(4, device="cuda"), k, 32, 32)
        reference, _ = scene.render(camera)
        transform = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.3],
                [0.0, 0.0, -1.0, 0.4],
                [0.0, 1.0, 0.0, -0.7],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device="cuda",
        )
        scene.world_transform = transform
        calibrated = Camera(torch.linalg.inv(transform), k, 32, 32)
        result, _ = scene.render(calibrated)
        self.assertTrue(torch.allclose(result, reference, atol=1e-5))
        original_world = camera.world_points(reference[..., 3])
        calibrated_world = calibrated.world_points(result[..., 3])
        self.assertTrue(
            torch.allclose(
                calibrated_world,
                original_world @ transform[:3, :3].T + transform[:3, 3],
                atol=1e-5,
            )
        )

    def test_elevated_water_intersection_and_object_occlusion(self):
        view = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device="cuda"))
        view[2, 3] = 1.0
        camera = Camera(
            view,
            torch.tensor(
                [[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]], device="cuda"
            ),
            32,
            32,
        )
        depth = torch.ones(32, 32, device="cuda")
        depth[:, 16:] = 0.5
        normal = torch.zeros(32, 32, 3, device="cuda")
        normal[..., 2] = 1
        terrain = SimpleNamespace(
            device="cuda",
            bounds=torch.tensor([-2.0, -2.0, 2.0, 2.0], device="cuda"),
            ground=torch.zeros(3, device="cuda"),
        )
        rgb = torch.full_like(normal, 0.2)
        buffers = dict(
            rgb=rgb,
            depth=depth,
            normal=normal,
            alpha=torch.ones_like(depth),
            world=camera.world_points(depth),
            reflection=rgb,
        )
        compositor = Compositor(camera, buffers, terrain)
        _, elevation, _, visible, distance = compositor.intersect_water(
            torch.full((16, 16), 0.1, device="cuda"),
            torch.full((16, 16), 0.1, device="cuda"),
        )
        self.assertTrue(
            torch.allclose(distance, torch.full_like(distance, 0.9), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(elevation, torch.full_like(elevation, 0.1), atol=1e-6)
        )
        self.assertTrue((visible[:, :16] == 1).all())
        self.assertTrue((visible[:, 16:] == 0).all())
        grid_x = -2 + (torch.arange(128, device="cuda") + 0.5) * 4 / 128
        sloped = (0.1 - 2 * grid_x)[None].expand(128, -1)
        _, _, _, visible, distance = compositor.intersect_water(
            sloped, torch.full_like(sloped, 0.1)
        )
        ray_x = (torch.arange(16, device="cuda") + 0.5 - 16) / 32
        analytic = 0.9 / (1 - 2 * ray_x)
        self.assertTrue(
            torch.allclose(distance[:, :16], analytic[None].expand(32, -1), atol=1e-4)
        )
        self.assertTrue((visible[:, :16] == 1).all())

    def compositor(self, depth):
        camera = Camera(
            torch.eye(4, device="cuda"),
            torch.tensor(
                [[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]], device="cuda"
            ),
            32,
            32,
        )
        normal = torch.zeros(32, 32, 3, device="cuda")
        normal[..., 2] = 1
        rgb = torch.full_like(normal, 0.2)
        terrain = SimpleNamespace(
            device="cuda",
            bounds=torch.tensor([-4.0, -4.0, 4.0, 4.0], device="cuda"),
            ground=torch.zeros(3, device="cuda"),
        )
        buffers = dict(
            rgb=rgb,
            depth=depth,
            normal=normal,
            alpha=torch.ones_like(depth),
            world=camera.world_points(depth),
            reflection=rgb,
        )
        return Compositor(camera, buffers, terrain), rgb

    def particle(self, z):
        def streaks(exposure):
            return (
                torch.tensor([[-0.015, 0.0, z]], device="cuda"),
                torch.eye(3, device="cuda")[None] * 0.05**2,
                torch.tensor([0.7], device="cuda"),
            )

        return SimpleNamespace(streaks=streaks)

    def test_hidden_particle_does_not_change_image(self):
        compositor, image = self.compositor(torch.ones(32, 32, device="cuda"))
        result, record = compositor.rain(image, self.particle(2.0), 1 / 60)
        self.assertTrue(torch.equal(result, image))
        self.assertEqual(record["visible_segments"], 0)

    def test_pixel_occlusion_at_silhouette(self):
        depth = torch.full((32, 32), 2.0, device="cuda")
        depth[:, 16:] = 0.5
        compositor, image = self.compositor(depth)
        result, record = compositor.rain(image, self.particle(1.0), 1 / 60)
        self.assertGreater(float((result[:, :16] - image[:, :16]).abs().max()), 0.02)
        self.assertTrue(torch.equal(result[:, 16:], image[:, 16:]))
        self.assertGreater(record["depth_rejected_pixels"], 0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
