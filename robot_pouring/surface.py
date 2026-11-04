"""Reconstruct liquid surfaces and export recorded robot geometry for Blender."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import mujoco
import warp as wp
from . import ROOT, CACHE_ROOT
from .config import PouringConfig
from .model import make_model


@wp.kernel(enable_backward=False)
def deposit(x: wp.array(dtype=wp.vec3), status: wp.array(dtype=int), field: wp.array3d(dtype=float),
            origin: wp.vec3, ds: float, dx: float):
    p = wp.tid()
    if status[p] >= 2:
        return
    point = x[p]
    base = wp.vec3i(int(wp.floor((point[0]-origin[0])/ds)), int(wp.floor((point[1]-origin[1])/ds)), int(wp.floor((point[2]-origin[2])/ds)))
    reach = int(wp.ceil(2.1 * dx / ds))
    for i in range(-reach, reach+1):
        for j in range(-reach, reach+1):
            for k in range(-reach, reach+1):
                a = base + wp.vec3i(i, j, k)
                if a[0] >= 0 and a[1] >= 0 and a[2] >= 0 and a[0] < field.shape[0] and a[1] < field.shape[1] and a[2] < field.shape[2]:
                    q = (origin + ds * wp.vec3(float(a[0]), float(a[1]), float(a[2])) - point) / (0.65 * dx)
                    value = wp.exp(-0.5 * wp.dot(q, q)) / (15.74960995 * 0.65 * 0.65 * 0.65)
                    wp.atomic_add(field, a[0], a[1], a[2], value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--indices", help="Comma-separated frame indices; default all")
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    cfg = PouringConfig(**json.loads((args.input / "config.json").read_text()))
    files = sorted((args.input / "states").glob("*.npz"))
    indices = [int(i) for i in args.indices.split(",")] if args.indices else list(range(len(files)))
    if not indices or min(indices) < 0 or max(indices) >= len(files):
        raise ValueError(f"Frame indices must be in [0, {len(files)-1}]")
    wp.config.kernel_cache_dir = str(CACHE_ROOT / "warp")
    wp.init()
    minimum = np.full(3, np.inf); maximum = -minimum
    for f in files:
        with np.load(f) as d:
            points = d["x"][d["status"] < 2]
            minimum = np.minimum(minimum, points.min(0)); maximum = np.maximum(maximum, points.max(0))
    origin = minimum - 2.5 * cfg.spacing; upper = maximum + 2.5 * cfg.spacing
    ds = .5 * cfg.spacing
    shape = tuple((np.ceil((upper-origin)/ds).astype(int)+1).tolist())
    if np.prod(shape) > 50_000_000:
        raise ValueError(f"Fluid bounds unexpectedly large: {shape}")
    field = wp.zeros(shape, dtype=float, device=cfg.device)
    mc = wp.MarchingCubes(*shape, max_verts=1000000, max_tris=2000000, device=cfg.device)
    model, _ = make_model(cfg); md = mujoco.MjData(model)
    geom_ids = np.where((model.geom_group == 2) & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH))[0]
    meshes, geoms = {}, []
    for g in geom_ids:
        mid = int(model.geom_dataid[g]); key = f"mesh_{mid}"
        if key + "_vertices" not in meshes:
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fadr, fnum = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            meshes[key + "_vertices"] = model.mesh_vert[adr:adr+num].copy()
            meshes[key + "_faces"] = model.mesh_face[fadr:fadr+fnum].copy()
        material = model.geom_matid[g]
        color = model.mat_rgba[material] if material >= 0 else model.geom_rgba[g]
        geoms.append(dict(id=int(g), mesh=key, rgba=color.tolist(), body=model.body(int(model.geom_bodyid[g])).name))
    np.savez_compressed(args.output / "robot_meshes.npz", **meshes)
    meta = dict(config=cfg.to_dict(), frames=len(files), indices=indices, geoms=geoms, grid_shape=shape,
                spacing=ds, isovalue=.45, kernel_sigma=.65*cfg.spacing)
    (args.output / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    started = time.perf_counter(); records = []
    for i in indices:
        with np.load(files[i]) as d:
            x = wp.array(d["x"], dtype=wp.vec3, device=cfg.device)
            status = wp.array(d["status"], dtype=int, device=cfg.device)
            field.zero_()
            wp.launch(deposit, len(x), [x, status, field, wp.vec3(*origin), ds, cfg.spacing], device=cfg.device)
            mc.surface(field, .45)
            vertices = mc.verts.numpy() * ds + origin
            faces = mc.indices.numpy().reshape(-1, 3)[:, [0, 2, 1]]
            assert np.isfinite(vertices).all() and len(faces) > 0
            md.qpos[:] = d["qpos"]; md.mocap_pos[0] = d["target"]
            mujoco.mj_forward(model, md)
            np.savez_compressed(args.output / f"{i:05d}.npz", vertices=vertices.astype(np.float32), faces=faces,
                geom_pos=md.geom_xpos[geom_ids], geom_mat=md.geom_xmat[geom_ids].reshape(-1, 3, 3),
                cup_pos=d["cup_pos"], cup_rot=d["cup_rot"], target=d["target"],
                absorbed=d["absorbed_pos"][d["status"] == 2],
                mass_fractions=d["mass_fractions"], time=d["time"])
            records.append(dict(frame=i, vertices=len(vertices), triangles=len(faces)))
            if i % 25 == 0:
                print(json.dumps(records[-1] | {"elapsed": time.perf_counter()-started}), flush=True)
    (args.output / "audit.json").write_text(json.dumps(dict(passed=True, records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
