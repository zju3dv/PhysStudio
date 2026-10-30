"""Analytic depth and overlapping near/far snowflake occlusion checks."""

import argparse, json
from pathlib import Path
import numpy as np
import torch
import warp as wp
from .surface import draw_flakes, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    wp.init()
    wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()), device="cuda:0")
    depth = torch.full((32, 32), 2.0, device="cuda")
    results = {}
    maps = []
    for label, positions in [
        ("front", [[0, 0, 1]]),
        ("back", [[0, 0, 3]]),
        ("overlap", [[0, 0, 1], [0, 0, 3]]),
    ]:
        pos = wp.array(np.array(positions, "f"), dtype=wp.vec3, device="cuda:0")
        vel = wp.zeros(len(positions), dtype=wp.vec3, device="cuda:0")
        r = wp.array(np.full(len(positions), 0.01, "f"), dtype=float, device="cuda:0")
        tau = torch.zeros_like(depth)
        wp.launch(
            draw_flakes,
            len(positions),
            [
                pos,
                vel,
                r,
                wp.from_torch(depth),
                wp.from_torch(tau),
                100.0,
                100.0,
                16.0,
                16.0,
            ],
            device="cuda:0",
        )
        results[label] = float(tau.sum())
        maps.append(tau)
    assert results["front"] > 0 and results["back"] == 0
    assert torch.equal(maps[0], maps[2])
    rho = torch.zeros((32, 32, 32), device="cuda")
    rho[:, :, 20:25] = 1
    illum = rho * 0.85
    coarse = torch.ones((16, 16, 16), device="cuda")
    phi = coarse * 10
    source = torch.tensor([[[0.3, 0.3, 0.3, 4.0]]], device="cuda")
    out = torch.empty((1, 1, 3), device="cuda")
    z = torch.empty((1, 1), device="cuda")
    wp.launch(
        trace,
        (1, 1),
        [
            wp.from_torch(rho),
            wp.from_torch(illum),
            wp.from_torch(coarse),
            wp.from_torch(phi),
            wp.vec3(-0.5, -0.5, 1.0),
            0.05,
            0.1,
            wp.mat44(np.eye(4, dtype="f")),
            100.0,
            100.0,
            0.5,
            0.5,
            1.0,
            wp.from_torch(source),
            wp.from_torch(out),
            wp.from_torch(z),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec4(-0.5, -0.5, 1.0, 1.0),
            0.11,
        ],
        device="cuda:0",
    )
    results["surface_depth"] = float(z[0, 0])
    results["expected_depth"] = 1.0 + (19.0 + 0.11) * 0.05
    assert abs(results["surface_depth"] - results["expected_depth"]) < 0.0005
    results["passed"] = True
    (a.output / "validation.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
