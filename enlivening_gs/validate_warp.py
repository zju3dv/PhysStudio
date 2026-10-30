"""CPU/GPU numerical checks, force adjoint check and warmed timings."""

from physstudio.paths import data_path
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import warp as wp

from .mesh import Cage
from .fem import FEM
from .warp_fem import WarpFEM, WarpEmbedding, _elastic


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def median_time(fn, repeats=3):
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    return float(np.median(values))


def check(scene, resolution):
    data = np.load(data_path("enlivening_gs") / scene / "gaussians.npz")
    cage = Cage(data["means"], data["axes"], resolution)
    print(scene, "nodes", len(cage.x), "tets", len(cage.tets), flush=True)
    cpu = FEM(cage)
    gpu = WarpFEM(cage)
    rng = np.random.default_rng(19)
    x = cage.x.copy()
    # Nonuniform deformation exercises the element Hessian and muscle rotation.
    x[:, 0] += 0.04 * np.sin(x[:, 2] * 2.3)
    x[:, 2] *= 0.94
    v = rng.normal(size=x.shape) * 0.02
    force, k, stats = cpu.elastic(x)
    force_g, ke = gpu.elastic_device(x)
    from scipy import sparse

    kg = sparse.coo_matrix(
        (ke.cpu().numpy().ravel(), (cpu.rows, cpu.cols)), shape=k.shape
    ).tocsc()
    a = cpu.activation_matrix(x)
    ag = gpu.activation_device().cpu().numpy()
    direction = rng.normal(size=x.shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    _, _, ep = cpu.elastic(x + eps * direction)
    _, _, em = cpu.elastic(x - eps * direction)
    fd = (ep["elastic_energy"] - em["elastic_energy"]) / (2 * eps)
    force_fd = abs(fd + np.sum(force * direction)) / max(abs(fd), 1e-12)
    fp, _, _ = cpu.elastic(x + eps * direction)
    fm, _, _ = cpu.elastic(x - eps * direction)
    # Compare the unprojected tangent to the derivative of the force.
    _, kraw, _ = cpu.elastic(x, project=False)
    tangent_fd = relative(-(fp - fm).ravel() / (2 * eps), kraw @ direction.ravel())
    # AD through the actual Warp elastic energy, independently of the Hessian.
    wx = wp.array(x, dtype=wp.vec3d, device="cuda:0", requires_grad=True)
    wf = wp.zeros(len(x), dtype=wp.vec3d, device="cuda:0", requires_grad=True)
    we = wp.zeros(len(cage.tets), dtype=wp.float64, device="cuda:0", requires_grad=True)
    deformation = wp.empty(len(cage.tets), dtype=wp.mat33d, device="cuda:0")
    dets = wp.empty(len(cage.tets), dtype=wp.float64, device="cuda:0")
    with wp.Tape() as tape:
        wp.launch(
            _elastic,
            len(cage.tets),
            [
                wx,
                gpu.tets,
                gpu.grad,
                gpu.volume,
                gpu.mu,
                gpu.lam,
                deformation,
                wf,
                we,
                dets,
            ],
            device="cuda:0",
        )
    tape.backward(
        grads={we: wp.ones(len(cage.tets), dtype=wp.float64, device="cuda:0")}
    )
    ad_error = relative(wx.grad.numpy(), -force)
    cf, cv, ca, _ = cpu.linear_step(x, v, 1 / 120)
    gf, gv, ga, gs = gpu.linear_step(x, v, 1 / 120)
    points = np.stack(
        [data["means"]]
        + [
            data["means"] + sign * 3 * data["axes"][:, :, i]
            for sign in [-1, 1]
            for i in range(3)
        ],
        1,
    ).reshape(-1, 3)
    binding = cage.matrix(points)
    emb = WarpEmbedding(cage, points)
    ep = emb(x).cpu().numpy()
    result = dict(
        scene=scene,
        nodes=len(x),
        tetrahedra=len(cage.tets),
        gaussians=len(data["means"]),
        embedding_points=len(points),
        force_relative_error=relative(force_g.cpu().numpy(), force),
        tangent_relative_error=relative(kg.toarray(), k.toarray()),
        muscle_relative_error=relative(ag, a),
        force_energy_fd_error=float(force_fd),
        tangent_fd_error=tangent_fd,
        warp_energy_ad_error=ad_error,
        free_velocity_relative_error=relative(gv, cv),
        activation_response_relative_error=relative(ga, ca),
        embedding_max_absolute_error=float(abs(ep - binding @ x).max()),
        gpu_solver=gs["linear_solve"],
    )
    for key in [
        "force_relative_error",
        "tangent_relative_error",
        "muscle_relative_error",
        "force_energy_fd_error",
        "tangent_fd_error",
        "warp_energy_ad_error",
        "free_velocity_relative_error",
        "activation_response_relative_error",
    ]:
        if result[key] > 2e-6:
            raise AssertionError((key, result[key]))
    if result["embedding_max_absolute_error"] > 1e-9:
        raise AssertionError(result)
    result["cpu_step_ms"] = 1000 * median_time(lambda: cpu.linear_step(x, v, 1 / 120))
    result["gpu_step_with_host_adapter_ms"] = 1000 * median_time(
        lambda: gpu.linear_step(x, v, 1 / 120)
    )
    xt = torch.tensor(x, device="cuda", dtype=torch.float64)
    vt = torch.tensor(v, device="cuda", dtype=torch.float64)
    result["gpu_step_device_ms"] = 1000 * median_time(
        lambda: gpu.linear_step_device(xt, vt, 1 / 120)
    )
    result["cpu_embedding_ms"] = 1000 * median_time(lambda: binding @ x)
    result["gpu_embedding_ms"] = 1000 * median_time(lambda: emb(xt))
    result["step_speedup"] = (
        result["cpu_step_ms"] / result["gpu_step_with_host_adapter_ms"]
    )
    result["embedding_speedup"] = (
        result["cpu_embedding_ms"] / result["gpu_embedding_ms"]
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=["wolf", "bread", "vasedeck"])
    p.add_argument("--resolution", type=int, default=7)
    p.add_argument("--output", default="outputs/enlivening_gs/warp_validation_v01.json")
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for scene in args.scenes:
        results.append(check(scene, args.resolution))
        out.write_text(
            json.dumps(
                dict(
                    warp_version=wp.__version__,
                    gpu=torch.cuda.get_device_name(),
                    resolution=args.resolution,
                    results=results,
                ),
                indent=2,
            )
        )
