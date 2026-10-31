"""Extract a rooted branch/venation graph from connected Gaussian geometry."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .assets import ROOT, SCENES, prepare


def make_rig(scene, spacing=.065, output_root=ROOT):
    output_root = Path(output_root)
    d = prepare(scene, output_root)
    xyz = d['means']
    floor = dict(potted_plant=.59, ficus=.425, alocasia=.615)[scene]
    # Pot/soil membership is separate from the root boundary condition.
    # Hanging leaf tips remain deformable even below the soil surface height.
    static = d['static'].copy()
    pot_fragments = np.zeros(len(xyz), bool)
    if scene == 'potted_plant':
        # The released foliage group also contains small pot-surface fragments.
        # Identify those by proximity to the known pot inside its footprint,
        # rather than fixing every point below a horizontal height cutoff.
        below = np.flatnonzero(~static & (xyz[:, 2] < floor))
        distance = cKDTree(xyz[static]).query(xyz[below])[0]
        pot_fragments[below] = (distance < .04) & (np.linalg.norm(xyz[below, :2], axis=1) < .28)
        static |= pot_fragments
    cage_samples = ~static & (xyz[:, 2] >= floor)
    moving = xyz[cage_samples]
    voxel = .014
    _, inv = np.unique(np.floor(moving / voxel).astype(int), axis=0, return_inverse=True)
    n = int(inv.max()) + 1
    pts = np.stack([np.bincount(inv, weights=moving[:, k], minlength=n) for k in range(3)], axis=1)
    pts /= np.bincount(inv)[:, None]
    tree = cKDTree(pts)
    ds, ids = tree.query(pts, k=14)
    rows = np.broadcast_to(np.arange(n)[:, None], ids.shape)
    keep = (ds > 0) & (ds < .055)
    a, b, lengths = rows[keep], ids[keep], ds[keep]
    graph = coo_matrix((lengths, (a, b)), shape=(n, n)).tocsr()
    graph = graph.maximum(graph.T)
    components, labels = connected_components(graph)
    # Bridge disconnected scan fragments through their nearest observed samples.
    bridges = []
    while components > 1:
        main = np.argmax(np.bincount(labels))
        ii = np.flatnonzero(labels == main)
        other = np.flatnonzero(labels != main)
        distance, nn = cKDTree(pts[ii]).query(pts[other])
        for label in np.unique(labels[other]):
            subset = np.flatnonzero(labels[other] == label)
            k = subset[np.argmin(distance[subset])]
            u, v, length = int(ii[nn[k]]), int(other[k]), float(distance[k])
            bridges.append([u, v, length])
            a = np.r_[a, u, v]; b = np.r_[b, v, u]; lengths = np.r_[lengths, length, length]
        graph = coo_matrix((lengths, (a, b)), shape=(n, n)).tocsr()
        graph = graph.maximum(graph.T)
        components, labels = connected_components(graph)
    root = int(np.argmin((pts[:, 2] - floor) ** 2 + .3 * np.sum(pts[:, :2] ** 2, axis=1)))
    distance, pred = dijkstra(graph, indices=root, return_predecessors=True)
    bins = np.floor(distance / spacing).astype(int)
    voxel_node = np.empty(n, int)
    centers, members = [], []
    for level in range(bins.max() + 1):
        ii = np.flatnonzero(bins == level)
        count, labels = connected_components(graph[ii][:, ii])
        for label in range(count):
            part = ii[labels == label]
            voxel_node[part] = len(centers)
            centers.append(pts[part].mean(0))
            members.append(part)
    centers = np.array(centers)
    parents = np.full(len(centers), -1, int)
    for i, part in enumerate(members):
        k = part[np.argmin(distance[part])]
        if k == root:
            continue
        j = pred[k]
        while voxel_node[j] == i and j != root:
            j = pred[j]
        parents[i] = voxel_node[j]
    root_node = voxel_node[root]
    assert parents[root_node] == -1
    # A short fixed root rod anchors the vascular system to the soil.
    base = centers[root_node].copy(); base[2] = floor - .035
    centers = np.vstack([base, centers])
    parents = np.r_[-1, np.where(parents < 0, 0, parents + 1)]
    # Contract artificial hairpins created where geodesic bands split a broad leaf.
    for i in range(1, len(centers)):
        while parents[i] > 0:
            p = parents[i]; gp = parents[p]
            incoming = centers[p] - centers[gp]
            outgoing = centers[i] - centers[p]
            cosine = np.dot(incoming, outgoing) / (np.linalg.norm(incoming) * np.linalg.norm(outgoing))
            if cosine > -.55:
                break
            parents[i] = gp
    heads = centers[np.maximum(parents, 0)].copy()
    heads[0] = base - [0, 0, .035]
    lengths = np.linalg.norm(centers - heads, axis=1)
    assert lengths.min() > 1e-6
    nearest = voxel_node[inv] + 1
    node = np.zeros(len(xyz), int); node[cage_samples] = nearest
    # Sparse hanging tips extend their observed leaf instead of introducing
    # disconnected tiny rods below the soil boundary.
    lower = ~static & ~cage_samples
    node[lower] = nearest[cKDTree(moving).query(xyz[lower])[1]]
    sample_count = np.bincount(nearest, minlength=len(centers))
    # Local PCA distinguishes planar leaf laminae from thin stems.
    leaf = np.zeros(len(centers), bool)
    widths = np.full(len(centers), .006)
    for i in range(1, len(centers)):
        samples = moving[nearest == i]
        if len(samples) < 6:
            continue
        val = np.linalg.eigvalsh(np.cov(samples.T))
        widths[i] = np.sqrt(max(val[-2], 1e-8))
        leaf[i] = widths[i] > .013
    radius = np.where(leaf, .003, .0065)
    radius *= np.clip((1.9 - centers[:, 2]) / 1.1, .8, 1.3)
    radius[0] = .015
    if scene == 'ficus':
        radius[~leaf] *= 1.7
    # Carry the downstream leaf area through the stem cross section (pipe model).
    # Local planar samples near a junction must not turn the main trunk into a leaf vein.
    leaf_area = 5 * widths * lengths * leaf
    downstream_area = leaf_area.copy()
    for i in range(len(parents) - 1, 0, -1):
        downstream_area[parents[i]] += downstream_area[i]
    fraction = downstream_area / max(downstream_area[0], 1e-8)
    base_radius = dict(potted_plant=.018, ficus=.022, alocasia=.017)[scene]
    radius = np.maximum(radius, base_radius * np.sqrt(fraction))
    leaf &= fraction < .12
    centers = centers.astype(np.float32)
    result = dict(tail=centers, head=heads.astype(np.float32), parent=parents.astype(np.int32),
                  radius=radius.astype(np.float32), leaf=leaf, length=lengths.astype(np.float32),
                  node=node.astype(np.int32), static=static, sample_count=sample_count,
                  leaf_area=leaf_area.astype(np.float32), downstream_area=downstream_area.astype(np.float32),
                  root_height=np.array(floor))
    result['hydraulic_leaf'] = leaf.copy()
    np.savez_compressed(output_root / scene / 'rig.npz', **result)
    report = dict(nodes=len(centers), leaf_nodes=int(leaf.sum()),
                  roots=int((parents < 0).sum()), bridge_count=len(bridges),
                  max_scan_bridge=max([x[2] for x in bridges], default=0), spacing=spacing,
                  moving_below_soil=int(((xyz[:, 2] < floor) & ~static).sum()),
                  recovered_pot_fragments=int(pot_fragments.sum()),
                  static_policy='released pot/soil membership; fixed root rod')
    (output_root / scene / 'rig.json').write_text(json.dumps(report, indent=2))
    print(scene, report, flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scene', choices=SCENES, nargs='+', default=list(SCENES))
    p.add_argument('--spacing', type=float, default=.065)
    p.add_argument('--output-root', default=str(ROOT))
    args = p.parse_args()
    for scene in args.scene:
        make_rig(scene, args.spacing, args.output_root)
