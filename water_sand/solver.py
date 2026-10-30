"""Warp orchestration, unit conversion, snapshots, and physical diagnostics."""

import math
import numpy as np
import warp as wp
from .common import Grid, Sand
from . import fluid, sand


class Mixture:
    def __init__(self, cfg, wall, phi, velocity, positions, device="cuda:0"):
        self.cfg, self.device, self.steps = cfg, device, 0
        g = Grid()
        g.nx, g.ny, g.nz = cfg["shape"]
        g.q = 27
        self.n = g.nx * g.ny * g.nz
        speed = (
            math.sqrt(cfg["gravity"] * cfg["reference_length"] * cfg["dx"] * 4)
            / cfg["lattice_velocity"]
        )
        self.velocity_scale = speed
        self.dt = cfg["dx"] / speed
        if "settle_seconds" in cfg:
            cfg["settle"] = round(cfg["settle_seconds"] / self.dt)
        g.rho_air = 1.25 / 1000.0
        g.nu_water, g.nu_air = (
            1.0e-4 / (cfg["dx"] * speed),
            1.0e-2 / (cfg["dx"] * speed),
        )
        g.gravity = cfg["gravity"] * cfg["dx"] / speed**2
        g.sigma = 0.072 / (speed**2 * cfg["dx"] * 1000.0)
        g.interface, g.mobility, g.cd = 5.0, 0.1, 0.1
        g.drag_scale = cfg["dx"] / cfg.get("drag_reference_dx", cfg["dx"])
        g.rho_s = cfg["rho_s"] / 1000.0
        g.mu = cfg["young"] / (2 * 1.3) / (1000 * speed**2)
        g.lam = cfg["young"] * 0.3 / (1.3 * 0.4) / (1000 * speed**2)
        g.pvol = 1.0 / cfg["ppc"] ** 3
        g.retention = int(cfg["retention"] and cfg["mode"] == "coupled")
        g.absorb_rate, g.cohesion, g.friction = 0.003, cfg["cohesion"], 0.75
        if "absorption_per_second" in cfg:
            g.absorb_rate = cfg["absorption_per_second"] * self.dt
        g.coupled = int(cfg["mode"] == "coupled")
        g.conservative = int(cfg.get("conservative", True))
        g.wall = wp.array(wall, dtype=int, device=device)
        for name in ["f", "fn"]:
            setattr(g, name, wp.zeros((27, self.n), dtype=float, device=device))
        for name in ["h", "hn"]:
            setattr(g, name, wp.zeros((7, self.n), dtype=float, device=device))
        for name in [
            "phi",
            "pressure",
            "pressure_next",
            "viscosity",
            "source",
            "mass",
            "delta",
            "eps",
            "eps_old",
            "eps_eff",
            "bound",
            "capacity",
            "bound_next",
            "absorbed",
        ]:
            setattr(g, name, wp.zeros(self.n, dtype=float, device=device))
        for name in [
            "u",
            "ut",
            "force",
            "gradphi",
            "gradeps",
            "momentum",
            "stressforce",
            "vs",
            "drag",
        ]:
            setattr(g, name, wp.zeros(self.n, dtype=wp.vec3, device=device))
        g.phi.assign(phi)
        g.u.assign(velocity / speed)
        g.totals = wp.zeros(3, dtype=wp.float64, device=device)
        g.control = wp.ones(1, dtype=float, device=device)
        s = Sand()
        self.np = len(positions)
        for name in ["x", "v"]:
            setattr(s, name, wp.zeros(self.np, dtype=wp.vec3, device=device))
        s.x.assign(positions)
        s.fe = wp.array(
            np.tile(np.eye(3, dtype=np.float32), (self.np, 1, 1)),
            dtype=wp.mat33,
            device=device,
        )
        s.c = wp.zeros(self.np, dtype=wp.mat33, device=device)
        for name in ["plastic", "hardening", "water", "saturation"]:
            setattr(s, name, wp.zeros(self.np, dtype=float, device=device))
        s.alpha = wp.full(self.np, 0.267765, dtype=float, device=device)
        self.g, self.s, self.graphs = g, s, {}
        self.launch(fluid.initialize)
        if cfg["mode"] != "sand":
            for _ in range(cfg["warmup"]):
                self.launch(fluid.gradients_forces)
                g.hn.zero_()
                self.launch(fluid.collide_phase, 1)
                self.launch(fluid.macros, 1)
            g.totals.zero_()
        if cfg["mode"] != "fluid":
            coupled, retention = g.coupled, g.retention
            g.coupled, g.retention = 0, 0
            for _ in range(cfg["settle"]):
                self.sand_step()
            g.coupled, g.retention = coupled, retention
            absorption = g.absorb_rate
            g.absorb_rate = 0.0
            self.sand_step()
            g.absorb_rate = absorption
        self.launch(fluid.initialize_pore_water)
        self.launch(fluid.gradients_forces)
        wp.launch(fluid.volume_control, 1, inputs=[g, 1], device=device)
        self.initial = self.diagnostics()

    def launch(self, kernel, *args, particles=False):
        inputs = [self.g, self.s] if particles else [self.g]
        wp.launch(
            kernel,
            self.np if particles else self.n,
            inputs=inputs + list(args),
            device=self.device,
        )

    def sand_step(self):
        self.launch(sand.clear_grid)
        self.launch(sand.p2g, particles=True)
        self.launch(sand.grid_update)
        self.launch(sand.g2p, particles=True)

    def step_once(self):
        if self.cfg["mode"] != "fluid":
            self.sand_step()
        if self.cfg["mode"] != "sand":
            self.launch(fluid.refresh_phase)
            self.launch(fluid.gradients_forces)
            wp.launch(fluid.volume_control, 1, inputs=[self.g, 0], device=self.device)
            self.launch(fluid.collide_flow)
            self.g.hn.zero_()
            self.launch(fluid.collide_phase, 0)
            self.launch(fluid.macros, 0)
            self.launch(fluid.pressure_update)
            self.launch(fluid.pressure_filter)

    def step(self, count):
        if count not in self.graphs:
            # First execute once to compile kernels before capture.
            self.step_once()
            with wp.ScopedCapture(device=self.device) as capture:
                for _ in range(count):
                    self.step_once()
            self.graphs[count] = capture.graph
            self.steps += 1
            if count > 1:
                for _ in range(count - 1):
                    self.step_once()
            self.steps += count - 1
        else:
            wp.capture_launch(self.graphs[count])
            self.steps += count

    def snapshot(self):
        shape = self.cfg["shape"]
        return dict(
            phi=self.g.phi.numpy().reshape(shape),
            eps=self.g.eps_eff.numpy().reshape(shape),
            sand_density=self.g.delta.numpy().reshape(shape),
            bound_grid=self.g.bound_next.numpy().reshape(shape),
            positions=self.s.x.numpy(),
            velocities=self.s.v.numpy(),
            moisture=self.s.water.numpy(),
            saturation=self.s.saturation.numpy(),
            deformation=self.s.fe.numpy(),
        )

    def diagnostics(self):
        g, s = self.g, self.s
        phi, eps, wall = g.phi.numpy(), g.eps_eff.numpy(), g.wall.numpy()
        x, v, fe, water = s.x.numpy(), s.v.numpy(), s.fe.numpy(), s.water.numpy()
        uv, mass = g.u.numpy(), g.mass.numpy()
        valid = wall == 0
        volume = float(
            np.sum(phi[valid].astype(np.float64) * eps[valid])
            + np.sum(water, dtype=np.float64)
        )
        vref = float(g.totals.numpy()[0])
        return dict(
            step=self.steps,
            time=self.steps * self.dt,
            finite=bool(
                all(np.isfinite(a).all() for a in [phi, eps, x, v, fe, uv, water])
            ),
            water_volume_cells=volume,
            water_relative_error=(volume - vref) / max(vref, 1.0e-12),
            phi_min=float(phi.min()),
            phi_max=float(phi.max()),
            eps_min=float(eps.min()),
            fluid_speed_max=float(np.linalg.norm(uv, axis=1).max()),
            sand_speed_max=float(np.linalg.norm(v, axis=1).max()),
            particles_inside=bool(
                np.all((x >= 1) & (x < np.array(self.cfg["shape"]) - 2))
            ),
            min_det=float(np.linalg.det(fe).min()),
            sand_mass_relative_error=float(
                mass.sum(dtype=np.float64) / (self.np * g.pvol * g.rho_s) - 1
            ),
            bound_water=float(water.sum(dtype=np.float64)),
            source_ratio=float(g.control.numpy()[0]),
        )
