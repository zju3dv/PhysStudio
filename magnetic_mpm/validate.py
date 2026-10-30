"""Independent constitutive, adjoint, Newton-Jacobian and field checks."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import warp as wp
from scipy.optimize import brentq
from .field import (
    Field,
    MS,
    BETA,
    jac_gradient,
    transpose_gradient,
    gradient,
    constitutive,
)
from .mpm import MPM


def numpy_m(h, ms):
    s = np.linalg.norm(h, axis=-1)
    a = BETA * s
    safe = np.maximum(a, 1.0e-20)
    l = np.where(
        a < 0.1,
        a * (1 / 3 - a * a / 45 + 2 * a**4 / 945 - a**6 / 4725),
        1 / np.tanh(safe) - 1 / safe,
    )
    return h * (ms * l / np.maximum(s, 1.0e-20))[..., None]


def run(device, n):
    torch.manual_seed(82)
    field = Field(n, 0.6 / n, (-0.3, -0.3, -0.3), device, tolerance=1.0e-5)
    strengths = np.geomspace(1.0e-7, 1.0e9, n).astype("f")
    sweep = np.zeros((n, n, n, 3), dtype="f")
    sweep[..., 2] = strengths[:, None, None]
    wp.to_torch(field.external).copy_(torch.tensor(sweep, device=device))
    field.saturation.fill_(MS)
    field.evaluate()
    sweep_reference = numpy_m(sweep.astype("d"), np.full((n, n, n), MS))
    sweep_error = np.max(np.abs(field.m.numpy() - sweep_reference)) / MS
    high_field_saturation = float(field.m.numpy()[-1, 0, 0, 2] / MS)
    xyz = np.stack(
        np.meshgrid(*([np.arange(n) * field.dx - 0.3] * 3), indexing="ij"), -1
    )
    inside = np.linalg.norm(xyz, axis=-1) < 0.075
    ms = np.where(inside, MS, 0.0).astype("f")
    wp.to_torch(field.saturation).copy_(torch.tensor(ms, device=device))
    field.set_source(intensity=0, uniform=(0, 0, 30000))
    # Constitutive law and positive Jacobian, including zero and saturation.
    field.evaluate()
    hv = field.h.numpy()
    mv = field.m.numpy()
    jac = field.jac.numpy()
    reference = numpy_m(hv, ms)
    law_error = np.linalg.norm(mv - reference) / np.linalg.norm(reference)
    eig_min = np.linalg.eigvalsh(jac.reshape(-1, 3, 3)).min()
    p = torch.randn(field.shape, device=device)
    p -= p.mean()
    q = torch.randn(field.shape, device=device)
    q -= q.mean()
    ap = field.matvec(p).clone()
    aq = field.matvec(q).clone()
    adjoint_error = abs((p * aq).sum().item() - (q * ap).sum().item()) / max(
        abs((p * aq).sum().item()), 1.0
    )
    r0 = field.rt.clone()
    epsilon = 0.05
    field.pt.copy_(epsilon * p)
    field.evaluate()
    rp = field.rt.clone()
    field.pt.copy_(-epsilon * p)
    field.evaluate()
    rm = field.rt.clone()
    derivative_error = (
        torch.linalg.vector_norm((rp - rm) / (2 * epsilon) - ap)
        / torch.linalg.vector_norm(ap)
    ).item()
    field.pt.zero_()
    sphere = field.solve(max_cg=400)

    # Nonlinear magnetized sphere: H_in = H_applied - M(H_in)/3.
    def scalar_m(h):
        return float(numpy_m(np.array([0.0, 0.0, h]), MS)[2])

    analytical_h = brentq(lambda h: h + scalar_m(h) / 3 - 30000, 1.0, 30000.0)
    core = np.linalg.norm(xyz, axis=-1) < 0.04
    core_h = field.h.numpy()[core].mean(0)
    field_error = abs(core_h[2] - analytical_h) / analytical_h
    saturation_ratio = np.max(np.linalg.norm(field.m.numpy(), axis=-1) / (ms + 1.0e-20))
    force = field.force_density.numpy() * inside[..., None] * field.dx**3
    net_force = np.linalg.norm(force.sum((0, 1, 2)))
    abs_force = np.linalg.norm(force, axis=-1).sum()
    # Vacuum must retain external field and have exactly zero mechanical force.
    field.phi.zero_()
    field.saturation.zero_()
    field.set_source(center=(0, 0, 0.17), radius=0.03, intensity=1.0e6)
    vacuum = field.solve()
    vacuum_force = float(np.abs(field.force_density.numpy()).max())
    # Zero applied field with magnetic material must also produce zero force.
    wp.to_torch(field.saturation).copy_(torch.tensor(ms, device=device))
    field.set_source(intensity=0)
    field.phi.zero_()
    zero = field.solve()
    zero_force = float(np.abs(field.force_density.numpy()).max())
    report = dict(
        n=n,
        magnetization_relative_error=float(law_error),
        jacobian_min_eigenvalue=float(eig_min),
        operator_symmetry_error=float(adjoint_error),
        jacobian_fd_relative_error=float(derivative_error),
        sphere=dict(
            solver=sphere,
            analytical_h=analytical_h,
            computed_core_h=core_h.tolist(),
            relative_error=float(field_error),
            saturation_ratio=float(saturation_ratio),
            net_force_N=float(net_force),
            net_to_absolute_force=float(net_force / max(abs_force, 1.0e-20)),
        ),
        vacuum_force_max=vacuum_force,
        zero_field_force_max=zero_force,
    )
    report.update(
        langevin_sweep_absolute_error_over_Ms=float(sweep_error),
        high_field_saturation_ratio=high_field_saturation,
    )
    # Unforced uniform motion: mass, magnetic micro-particle count and momentum.
    points = (
        np.stack(np.meshgrid(*([np.arange(7) * 0.005 - 0.015] * 3), indexing="ij"), -1)
        .reshape(-1, 3)
        .astype("f")
    )
    points[:, 2] += 0.19
    sim = MPM(points, n=48, gravity=0.0, device=device)
    sim.v.fill_(wp.vec3(0.12, -0.03, 0.04))
    sim.transfer()
    mass = float(sim.mass.numpy().sum())
    expected = len(points) * sim.volume * sim.density
    magnetic_count = float(sim.field.saturation.numpy().sum() * sim.dx**3)
    expected_count = len(points) * sim.volume * MS
    initial_v = sim.v.numpy().mean(0)
    for step in range(8):
        sim.step((0, 0, 1), 0.0, 0.0, update_field=step == 0)
    final = sim.state()
    velocity_error = float(np.max(np.abs(final["velocities"] - initial_v)))
    translation_error = float(
        np.max(np.abs(final["positions"] - points - sim.time * initial_v))
    )
    report["mpm"] = dict(
        grid_mass_relative_error=abs(mass - expected) / expected,
        microscopic_moment_relative_error=abs(magnetic_count - expected_count)
        / expected_count,
        translation_velocity_error=velocity_error,
        translation_position_error=translation_error,
    )
    print(json.dumps(report, indent=2), flush=True)
    assert law_error < 2.0e-5 and eig_min >= 1.0 - 1.0e-5
    assert adjoint_error < 5.0e-5 and derivative_error < 0.02
    assert sphere["converged"] and field_error < 0.08
    assert net_force / max(abs_force, 1.0e-20) < 0.01
    assert saturation_ratio <= 1.0001 and vacuum_force == 0 and zero_force == 0
    assert sweep_error < 2.0e-5 and 0.999 < high_field_saturation <= 1.000001
    assert report["mpm"]["grid_mass_relative_error"] < 2.0e-6
    assert report["mpm"]["microscopic_moment_relative_error"] < 2.0e-6
    assert velocity_error < 1.0e-4 and translation_error < 1.0e-6
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--output", default="outputs/magnetic_mpm/validation.json")
    args = parser.parse_args()
    report = run(args.device, args.n)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
