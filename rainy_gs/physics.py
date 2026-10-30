"""Conservative terrain water, persistent wind-driven drops and impact waves."""

import math
import heapq

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F


def sample_grid(field, xy, bounds):
    grid = (xy - bounds[:2]) / (bounds[2:] - bounds[:2]) * 2 - 1
    return F.grid_sample(
        field[None, None],
        grid[None, None],
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).reshape(-1)


def depression_fill(bed):
    """Four-neighbour priority flood: return each cell's spill elevation.

    The outer boundary drains. This only identifies depressions; the original
    bed stays unchanged and the shallow-water solver evolves the added water.
    """
    bed = np.asarray(bed)
    filled = bed.copy()
    height, width = bed.shape
    visited = np.zeros_like(bed, dtype=bool)
    queue = []
    for y in range(height):
        for x in range(width):
            if y in (0, height - 1) or x in (0, width - 1):
                heapq.heappush(queue, (float(bed[y, x]), y, x))
                visited[y, x] = True
    while queue:
        elevation, y, x = heapq.heappop(queue)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < height and 0 <= xx < width and not visited[yy, xx]:
                visited[yy, xx] = True
                level = max(elevation, float(bed[yy, xx]))
                filled[yy, xx] = level
                heapq.heappush(queue, (level, yy, xx))
    return filled


class ShallowWater:
    """Rusanov flux with hydrostatic reconstruction and matching bed sources.

    Reflecting edges, CFL substeps, positive depth, and no artificial height
    floor. Momentum damping acts on velocity, not on water mass.
    """

    def __init__(
        self, bed, dx, level=0.01, gravity=9.81, damping=1.5, initial_depth=None
    ):
        self.bed, self.dx, self.gravity, self.damping = bed, dx, gravity, damping
        self.level = level
        self.local_puddles = initial_depth is not None
        depth = (
            (level - bed).clamp_min(0)
            if initial_depth is None
            else initial_depth.to(bed)
        )
        if (
            depth.shape != bed.shape
            or not torch.isfinite(depth).all()
            or (depth < 0).any()
        ):
            raise ValueError(
                "Initial water depth must match the bed and be finite and nonnegative"
            )
        self.u = torch.stack((depth, torch.zeros_like(bed), torch.zeros_like(bed)))
        self.added = 0.0
        self.steps = 0
        self.minimum_depth = float(self.u[0].min())
        self.initial_mass = float(self.u[0].sum()) * dx * dx

    def faces(self, axis):
        u, b = self.u, self.bed
        dim = 2 if axis == 1 else 1
        if axis == 1:
            left, right = u[:, :, :1].clone(), u[:, :, -1:].clone()
            left[1] *= -1
            right[1] *= -1
            pad = torch.cat((left, u, right), 2)
            bp = torch.cat((b[:, :1], b, b[:, -1:]), 1)
            a, c, ba, bc = pad[:, :, :-1], pad[:, :, 1:], bp[:, :-1], bp[:, 1:]
        else:
            top, bottom = u[:, :1].clone(), u[:, -1:].clone()
            top[2] *= -1
            bottom[2] *= -1
            pad = torch.cat((top, u, bottom), 1)
            bp = torch.cat((b[:1], b, b[-1:]), 0)
            a, c, ba, bc = pad[:, :-1], pad[:, 1:], bp[:-1], bp[1:]
        topography = torch.maximum(ba, bc)
        ha, hc = (
            (a[0] + ba - topography).clamp_min(0),
            (c[0] + bc - topography).clamp_min(0),
        )
        va = torch.where(a[:1] > 1e-8, a[1:] / a[:1].clamp_min(1e-8), 0.0)
        vc = torch.where(c[:1] > 1e-8, c[1:] / c[:1].clamp_min(1e-8), 0.0)
        ua, uc = torch.cat((ha[None], va * ha)), torch.cat((hc[None], vc * hc))
        fa, fc = ua * va[axis - 1 : axis], uc * vc[axis - 1 : axis]
        fa[axis] += 0.5 * self.gravity * ha.square()
        fc[axis] += 0.5 * self.gravity * hc.square()
        speed = torch.maximum(
            va[axis - 1].abs() + torch.sqrt(self.gravity * ha),
            vc[axis - 1].abs() + torch.sqrt(self.gravity * hc),
        )
        flux = 0.5 * (fa + fc) - 0.5 * speed[None] * (uc - ua)
        flux_a, flux_c = flux.clone(), flux.clone()
        flux_a[axis] += 0.5 * self.gravity * (a[0].square() - ha.square())
        flux_c[axis] += 0.5 * self.gravity * (c[0].square() - hc.square())
        return flux_c.narrow(dim, 0, u.shape[dim]), flux_a.narrow(dim, 1, u.shape[dim])

    def step(self, dt, rain_rate=0.0):
        remaining = dt
        source = torch.as_tensor(
            rain_rate, dtype=self.u.dtype, device=self.u.device
        ).expand_as(self.bed)
        while remaining > 1e-9:
            h = self.u[0]
            speed = torch.where(
                h[None] > 1e-8, self.u[1:].abs() / h.clamp_min(1e-8), 0.0
            )
            max_speed = float((speed.sum(0) + 2 * torch.sqrt(self.gravity * h)).max())
            substep = min(remaining, 0.4 * self.dx / max(max_speed, 1e-8))
            fl, fr = self.faces(1)
            ft, fb = self.faces(2)
            candidate = self.u + substep / self.dx * (fl - fr + ft - fb)
            minimum = float(candidate[0].min())
            if minimum < -1e-7 or not torch.isfinite(candidate).all():
                raise FloatingPointError(
                    f"Invalid shallow-water update: minimum depth {minimum}"
                )
            candidate[0].clamp_(min=0)
            candidate[0] += substep * source
            self.added += substep * float(source.sum()) * self.dx**2
            attenuation = (
                torch.exp(-self.damping * substep)
                if isinstance(self.damping, torch.Tensor)
                else math.exp(-self.damping * substep)
            )
            candidate[1:] *= attenuation
            candidate[1:, candidate[0] < 1e-8] = 0
            self.u = candidate
            remaining -= substep
            self.steps += 1
            self.minimum_depth = min(self.minimum_depth, minimum)

    def diagnostics(self):
        mass = float(self.u[0].sum()) * self.dx**2
        return dict(
            initial_volume=self.initial_mass,
            rainfall_volume=self.added,
            current_volume=mass,
            mass_residual=mass - self.initial_mass - self.added,
            minimum_depth=self.minimum_depth,
            steps=self.steps,
        )


class Terrain:
    def __init__(self, scene, bounds=(-4.0, -4.0, 4.0, 4.0), resolution=384):
        self.device = scene.device
        self.bounds = torch.tensor(bounds, device=scene.device)
        self.resolution = resolution
        self.dx = (bounds[2] - bounds[0]) / resolution
        xy = scene.means[:, :2].cpu().numpy()
        z = scene.means[:, 2].cpu().numpy()
        ground = scene.ground.cpu().numpy()
        relative = z - xy @ ground[:2] - ground[2]
        normal = scene.normals.cpu().numpy()
        opacity = scene.opacity.cpu().numpy()
        cell = np.floor((xy - np.array(bounds[:2])) / self.dx).astype(np.int64)
        valid = (cell >= 0).all(-1) & (cell < resolution).all(-1) & (opacity > 0.25)
        floor = valid & (abs(relative) < 0.10) & (normal[:, 2] > 0.85)
        counts = np.zeros((resolution, resolution), np.float64)
        total = np.zeros_like(counts)
        np.add.at(counts, (cell[floor, 1], cell[floor, 0]), 1)
        np.add.at(total, (cell[floor, 1], cell[floor, 0]), relative[floor])
        mask = counts > 0
        floor_height = np.divide(total, counts, out=np.zeros_like(total), where=mask)
        _, indices = ndimage.distance_transform_edt(~mask, return_indices=True)
        floor_height = ndimage.gaussian_filter(floor_height[tuple(indices)], 1.25)
        # Ordinary 3DGS splat centers can lie below the visible ground. Fuse
        # ground-facing rendered depths from several cameras for those assets.
        support_cameras = getattr(scene, "config", {}).get("depth_ground_cameras", [])
        self.depth_support_cells = 0
        if support_cameras:
            support_sum = np.zeros_like(total)
            support_count = np.zeros_like(counts)
            for index in support_cameras:
                camera = scene.camera(index, 640)
                rgbd, alpha = scene.render(camera)
                normal_image, _ = scene.render(camera, colors=scene.normals, mode="RGB")
                normal_image = F.normalize(normal_image, dim=-1)
                world = camera.world_points(rgbd[..., 3])
                relative_height = world[..., 2] - scene.ground_height(world[..., :2])
                direction = F.normalize(world - camera.center, dim=-1)
                keep = (
                    (alpha > 0.98)
                    & (normal_image[..., 2] > 0.65)
                    & (direction[..., 2] < -0.03)
                    & (relative_height > -0.15)
                    & (relative_height < 0.28)
                )
                xyz = world[keep].cpu().numpy()
                cells = np.floor((xyz[:, :2] - np.array(bounds[:2])) / self.dx).astype(
                    np.int64
                )
                inside = (cells >= 0).all(-1) & (cells < resolution).all(-1)
                cells, xyz = cells[inside], xyz[inside]
                height = xyz[:, 2] - xyz[:, :2] @ ground[:2] - ground[2]
                np.add.at(support_count, (cells[:, 1], cells[:, 0]), 1)
                np.add.at(support_sum, (cells[:, 1], cells[:, 0]), height)
            count_smooth = ndimage.gaussian_filter(support_count, 1.25)
            sum_smooth = ndimage.gaussian_filter(support_sum, 1.25)
            supported = count_smooth > 0.25
            floor_height = np.where(
                supported, sum_smooth / count_smooth.clip(1e-8), floor_height
            )
            self.depth_support_cells = int(supported.sum())
        vegetation_sum = np.zeros_like(total)
        vegetation = scene.vegetation.cpu().numpy()
        np.add.at(vegetation_sum, (cell[floor, 1], cell[floor, 0]), vegetation[floor])
        vegetation_grid = np.divide(
            vegetation_sum, counts, out=np.zeros_like(total), where=mask
        )
        vegetation_grid = ndimage.gaussian_filter(vegetation_grid[tuple(indices)], 1.0)
        self.coarse_vegetation = F.interpolate(
            torch.tensor(vegetation_grid, device=scene.device, dtype=torch.float32)[
                None, None
            ],
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        top = floor_height.copy()
        obstacle = valid & (relative > 0.04) & (relative < 4.0)
        upper = np.full_like(top, -0.5)
        np.maximum.at(upper, (cell[obstacle, 1], cell[obstacle, 0]), relative[obstacle])
        upper = ndimage.maximum_filter(upper, size=3)
        top = np.maximum(top, upper)
        self.bed = torch.tensor(floor_height, device=scene.device, dtype=torch.float32)
        self.top = torch.tensor(top, device=scene.device, dtype=torch.float32)
        self.ground = scene.ground
        self.obstacles = self.top - self.bed > 0.10
        # Coarse water excludes vertical structures; high beds are hydrostatic
        # barriers, while the finer field resolves impact ripples.
        coarse = F.interpolate(
            self.bed[None, None], size=(128, 128), mode="bilinear", align_corners=False
        )[0, 0]
        barrier = (
            F.adaptive_max_pool2d(self.obstacles.float()[None, None], (128, 128))[0, 0]
            > 0.5
        )
        self.rain_source_mask = ~barrier
        self.coarse_bed = torch.where(
            barrier, torch.maximum(coarse, coarse.new_tensor(0.3)), coarse
        )

    def height(self, xy):
        return (
            sample_grid(self.top, xy, self.bounds)
            + xy @ self.ground[:2]
            + self.ground[2]
        )

    def initial_puddles(self, level, max_depth):
        spill = depression_fill(self.coarse_bed.cpu().numpy())
        spill = torch.as_tensor(spill, device=self.device, dtype=self.coarse_bed.dtype)
        local = (spill - self.coarse_bed).clamp(0, max_depth)
        connected = (level - self.coarse_bed).clamp_min(0)
        return torch.maximum(local, connected) * self.rain_source_mask

    def water_surface(self, water):
        if water.local_puddles:
            # Each disconnected puddle can stand at its own elevation. Dry
            # cells must not drag elevated lawn puddles to the global datum.
            wet = (water.u[0] > 1e-5).to(water.u.dtype)
            numerator = F.avg_pool2d(
                ((water.bed + water.u[0]) * wet)[None, None], 3, 1, 1
            )
            denominator = F.avg_pool2d(wet[None, None], 3, 1, 1)
            elevation = numerator / denominator.clamp_min(1e-8)
            eta = F.interpolate(
                elevation, size=self.bed.shape, mode="bilinear", align_corners=False
            )[0, 0]
            support = F.interpolate(
                wet[None, None],
                size=self.bed.shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            depth = (eta - self.bed).clamp_min(0) * support * (~self.obstacles)
            eta = torch.where(support > 0, eta, self.bed)
            return eta, depth
        # Interpolate free-surface elevation, not depth plus a different bed.
        # Dry obstacle cells have no water surface; extend the initial level
        # there for interpolation, then remove the solid cells from the mask.
        eta = torch.where(water.u[0] > 5e-4, water.bed + water.u[0], water.level)
        eta = F.interpolate(
            eta[None, None], size=self.bed.shape, mode="bilinear", align_corners=False
        )[0, 0]
        depth = (eta - self.bed).clamp_min(0) * (~self.obstacles)
        return eta, depth


class ImpactWaves:
    def __init__(self, terrain, gravity=3.924):
        self.terrain = terrain
        self.height = torch.zeros_like(terrain.bed)
        self.velocity = torch.zeros_like(self.height)
        self.gravity = gravity
        self.count = 0
        x = torch.arange(-3, 4, device=terrain.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(x, x, indexing="ij")
        kernel = torch.exp(-(xx.square() + yy.square()) / 2)
        kernel /= kernel.sum()
        self.kernel = kernel[None, None]

    def impulse(self, xy, strength):
        if len(xy) == 0:
            return
        n = self.terrain.resolution
        cell = ((xy - self.terrain.bounds[:2]) / self.terrain.dx).long()
        valid = (cell >= 0).all(-1) & (cell < n).all(-1)
        cell, strength = cell[valid], strength[valid]
        impulses = torch.zeros_like(self.height)
        impulses.flatten().scatter_add_(0, cell[:, 1] * n + cell[:, 0], strength)
        # Zero-integral impact packet displaces water instead of inventing it.
        smooth = F.conv2d(impulses[None, None], self.kernel, padding=3)[0, 0]
        wide = F.avg_pool2d(smooth[None, None], 7, 1, 3)[0, 0]
        self.velocity += smooth - wide
        self.count += len(cell)

    def step(self, dt, depth):
        speed = torch.sqrt(self.gravity * depth.clamp(0.004, 0.025))
        steps = max(1, math.ceil(dt * float(speed.max()) / (self.terrain.dx * 0.45)))
        for _ in range(steps):
            step = dt / steps
            h = F.pad(self.height[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
            lap = (
                h[1:-1, :-2]
                + h[1:-1, 2:]
                + h[:-2, 1:-1]
                + h[2:, 1:-1]
                - 4 * self.height
            ) / self.terrain.dx**2
            self.velocity = (self.velocity + step * speed.square() * lap) * math.exp(
                -step * 2.8
            )
            self.height += self.velocity * step
            wet = (depth > 0.001) & ~self.terrain.obstacles
            self.height *= wet
            self.velocity *= wet


class Rain:
    def __init__(
        self, terrain, count=16000, seed=7, metres_per_unit=2.5, wind=(0.35, 0.12)
    ):
        self.terrain, self.device = terrain, terrain.device
        self.scale, self.count = metres_per_unit, count
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.wind = torch.tensor(wind, device=self.device) / metres_per_unit
        self.pos = torch.empty((count, 3), device=self.device)
        self.diameter = (
            0.35 + self.rand(count).exponential_(1.4, generator=self.generator)
        ).clamp(0.35, 3.5)
        # Terminal-speed fit in m/s, diameter in mm. Scene scale is explicit.
        self.terminal = (9.65 - 10.3 * torch.exp(-0.6 * self.diameter)).clamp_min(
            0.6
        ) / metres_per_unit
        self.vel = torch.zeros_like(self.pos)
        self.vel[:, :2] = self.wind
        self.vel[:, 2] = -self.terminal
        self.respawn(torch.arange(count, device=self.device), fill=True)
        self.previous = self.pos.clone()
        self.splash_pos = self.pos[:0].clone()
        self.splash_vel = self.pos[:0].clone()
        self.splash_age = self.diameter[:0].clone()
        self.splash_life = self.diameter[:0].clone()
        self.time = 0.0
        self.impacts = 0
        self.maximum_penetration = 0.0

    def rand(self, *shape):
        return torch.rand(shape, generator=self.generator, device=self.device)

    def respawn(self, indices, fill=False):
        count = len(indices)
        xy = self.terrain.bounds[:2] + self.rand(count, 2) * (
            self.terrain.bounds[2:] - self.terrain.bounds[:2]
        )
        self.pos[indices, :2] = xy
        floor = self.terrain.height(xy)
        ceiling = xy @ self.terrain.ground[:2] + self.terrain.ground[2] + 4.2
        self.pos[indices, 2] = (
            floor + 0.03 + self.rand(count) * (ceiling - floor - 0.03)
            if fill
            else ceiling + self.rand(count) * 0.3
        )

    def step(self, dt, waves=None, water_depth=None):
        self.previous = self.pos.clone()
        self.time += dt
        target = self.wind + torch.stack(
            (
                0.08 * torch.sin(self.pos[:, 1] * 0.9 + self.time * 0.7),
                0.05 * torch.sin(self.pos[:, 0] * 1.1 - self.time * 0.8),
            ),
            -1,
        )
        self.vel[:, :2] += (target - self.vel[:, :2]) * (1 - math.exp(-dt * 2))
        self.vel[:, 2] += (-self.terminal - self.vel[:, 2]) * (1 - math.exp(-dt * 4))
        proposed = self.pos + self.vel * dt
        height = self.terrain.height(proposed[:, :2])
        if water_depth is not None:
            floor = sample_grid(
                self.terrain.bed + water_depth, proposed[:, :2], self.terrain.bounds
            )
            floor += proposed[:, :2] @ self.terrain.ground[:2] + self.terrain.ground[2]
            height = torch.maximum(height, floor)
        hit = proposed[:, 2] <= height
        inside = (
            (proposed[:, :2] >= self.terrain.bounds[:2])
            & (proposed[:, :2] < self.terrain.bounds[2:])
        ).all(-1)
        ids = torch.nonzero(hit & inside).flatten()
        self.pos = proposed
        if len(ids):
            impact = proposed[ids].clone()
            impact[:, 2] = height[ids] + 0.001
            self.impacts += len(ids)
            if waves is not None:
                water_hit = torch.ones(len(ids), device=self.device, dtype=torch.bool)
                if water_depth is not None:
                    water_hit = (
                        sample_grid(water_depth, impact[:, :2], self.terrain.bounds)
                        > 0.001
                    ) & ((height[ids] - floor[ids]).abs() < 0.005)
                waves.impulse(
                    impact[water_hit, :2],
                    0.04 * (self.diameter[ids[water_hit]] / 1.5).square().clamp(max=3),
                )
            picked = self.rand(len(ids)) < 0.27
            impact = impact[picked]
            if len(impact):
                impact = impact.repeat_interleave(4, 0)
                angle = self.rand(len(impact)) * 2 * math.pi
                radial = 0.08 + self.rand(len(impact)) * 0.22
                vel = torch.stack(
                    (
                        angle.cos() * radial,
                        angle.sin() * radial,
                        0.18 + self.rand(len(impact)) * 0.35,
                    ),
                    -1,
                )
                self.splash_pos = torch.cat((self.splash_pos, impact))
                self.splash_vel = torch.cat((self.splash_vel, vel))
                self.splash_age = torch.cat(
                    (self.splash_age, torch.zeros(len(impact), device=self.device))
                )
                self.splash_life = torch.cat(
                    (self.splash_life, 0.10 + self.rand(len(impact)) * 0.2)
                )
        # Discard overshoots before rendering. The horizontal travel per
        # substep stays smaller than a collision height-field cell.
        self.respawn(torch.nonzero(hit | ~inside).flatten())
        if len(self.splash_pos):
            self.splash_age += dt
            self.splash_vel[:, 2] -= dt * 9.81 / self.scale
            self.splash_pos += self.splash_vel * dt
            splash_surface = self.terrain.height(self.splash_pos[:, :2])
            if water_depth is not None:
                floor = sample_grid(
                    self.terrain.bed + water_depth,
                    self.splash_pos[:, :2],
                    self.terrain.bounds,
                )
                floor += (
                    self.splash_pos[:, :2] @ self.terrain.ground[:2]
                    + self.terrain.ground[2]
                )
                splash_surface = torch.maximum(splash_surface, floor)
            alive = (self.splash_age < self.splash_life) & (
                self.splash_pos[:, 2] >= splash_surface
            )
            self.splash_pos, self.splash_vel = (
                self.splash_pos[alive],
                self.splash_vel[alive],
            )
            self.splash_age, self.splash_life = (
                self.splash_age[alive],
                self.splash_life[alive],
            )

    def streaks(self, exposure=1 / 60, radius_scale=2.0, segments=3):
        radius = self.diameter * 0.0005 / self.scale * radius_scale
        eye = torch.eye(3, device=self.device)
        offsets = (
            (torch.arange(segments, device=self.device) + 0.5) * exposure / segments
        )
        means = (
            self.pos[:, None] - self.vel[:, None] * offsets[None, :, None]
        ).reshape(-1, 3)
        cov = radius[:, None, None].square() * eye + self.vel[:, :, None] * self.vel[
            :, None, :
        ] * (exposure**2 / (12 * segments**2))
        cov = cov.repeat_interleave(segments, 0)
        opacity = (
            (0.10 + 0.09 * (self.diameter / 1.5))
            .clamp(max=0.32)
            .repeat_interleave(segments)
        )
        if len(self.splash_pos):
            means = torch.cat((means, self.splash_pos))
            splash_cov = eye[None] * 0.00055**2 + self.splash_vel[
                :, :, None
            ] * self.splash_vel[:, None, :] * (exposure**2 / 12)
            cov = torch.cat((cov, splash_cov))
            opacity = torch.cat(
                (opacity, 0.36 * (1 - self.splash_age / self.splash_life))
            )
        return means, cov, opacity
