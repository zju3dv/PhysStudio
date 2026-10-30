"""Compare castle stability with the coupled model's dry constitutive law."""

import argparse
import json
from pathlib import Path
import numpy as np
import warp as wp
from .scenes import initial
from .solver import Mixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=2.9)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh control report")
    wp.init()
    wp.set_device("cuda:0")
    report = {}
    for name, young, cohesion in [
        ("original", 140000.0, 0.01),
        ("stronger", 500000.0, 0.08),
    ]:
        cfg, wall, phi, u, x = initial("castle", args.resolution, 2, "coupled")
        cfg.update(
            warmup=0, settle=200, young=young, cohesion=cohesion, conservative=True
        )
        phi[:] = 0
        u[:] = 0
        sim = Mixture(cfg, wall, phi, u, x)
        sim.g.coupled = 0
        # Keep retention enabled for the same dry limit of the cohesion curve.
        sim.cfg = dict(cfg, mode="sand")

        def shape():
            x = (sim.s.x.numpy() - 3) * cfg["dx"]
            return dict(
                height_p99=float(np.quantile(x[:, 2], 0.99)),
                center_height=float(x[:, 2].mean()),
                radial=float(np.linalg.norm(x[:, :2] - [2.4, 1.15], axis=-1).mean()),
            )

        before = shape()
        remaining = round(args.seconds / sim.dt)
        while remaining:
            count = min(40, remaining)
            sim.step(count)
            remaining -= count
        report[name] = dict(
            config=cfg, before=before, after=shape(), diagnostics=sim.diagnostics()
        )
        print(name, json.dumps(report[name]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
