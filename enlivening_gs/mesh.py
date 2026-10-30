"""Conforming voxel tetrahedra, exact barycentric binding and muscle paths."""

from itertools import permutations
import numpy as np
from scipy import ndimage, sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra

PERMS = np.array(list(permutations(range(3))))


class Cage:
    def __init__(self, means, axes, resolution=8):
        self.h = np.ptp(means[:, 2]) / resolution
        points = np.concatenate(
            [means] + [means + s * 3 * axes[:, :, i] for s in [-1, 1] for i in range(3)]
        )
        self.origin = np.floor(points.min(0) / self.h) * self.h - self.h * 0.01
        ijk = np.floor((points - self.origin) / self.h).astype(int)
        dims = ijk.max(0) + 1
        occ = np.zeros(tuple(dims + 4), bool)
        occ[tuple((ijk + 2).T)] = True
        closed = ndimage.binary_fill_holes(ndimage.binary_closing(occ, iterations=1))
        occ |= closed
        cells = np.argwhere(occ) - 2
        self.cell_map = {tuple(c): i for i, c in enumerate(cells)}
        vertex_map, vertices, tets = {}, [], []
        for cell in cells:
            for order in PERMS:
                offsets = np.zeros((4, 3), int)
                for j in range(3):
                    offsets[j + 1] = offsets[j]
                    offsets[j + 1, order[j]] += 1
                tet = []
                for p in cell + offsets:
                    key = tuple(p)
                    if key not in vertex_map:
                        vertex_map[key] = len(vertices)
                        vertices.append(p)
                    tet.append(vertex_map[key])
                tets.append(tet)
        self.x = np.array(vertices) * self.h + self.origin
        self.tets = np.array(tets, dtype=np.int32)
        dm = (self.x[self.tets[:, 1:]] - self.x[self.tets[:, :1]]).transpose(0, 2, 1)
        self.inv = np.linalg.inv(dm)
        self.vol = np.abs(np.linalg.det(dm)) / 6
        self.grad = np.concatenate([-self.inv.sum(1)[:, None, :], self.inv], 1)
        self.centers = self.x[self.tets].mean(1)
        self.tree = cKDTree(self.centers)

    def bind(self, points):
        z = (points - self.origin) / self.h
        cell = np.floor(z).astype(int)
        frac = z - cell
        order = np.argsort(-frac, axis=1)
        rank = (order[:, None, :] == PERMS[None]).all(-1).argmax(-1)
        ci = np.array([self.cell_map.get(tuple(c), -1) for c in cell])
        tid = ci * 6 + rank
        outside = ci < 0
        if outside.any():
            # New split endpoints can leave the original cell by a few ulps.
            # Search nearby tets; explicitly reject true extrapolation.
            _, candidates = self.tree.query(points[outside], k=min(96, len(self.tets)))
            p0 = self.x[self.tets[candidates, 0]]
            w123 = np.einsum(
                "nkij,nkj->nki", self.inv[candidates], points[outside, None] - p0
            )
            w = np.concatenate([1 - w123.sum(-1, keepdims=True), w123], -1)
            choice = w.min(-1).argmax(-1)
            best = w[np.arange(len(choice)), choice]
            if best.min() < -1e-5:
                raise ValueError(
                    f"Embedding outside cage: minimum weight {best.min():.4g}"
                )
            tid[outside] = candidates[np.arange(len(choice)), choice]
        v123 = np.einsum(
            "nij,nj->ni", self.inv[tid], points - self.x[self.tets[tid, 0]]
        )
        weights = np.concatenate([1 - v123.sum(-1, keepdims=True), v123], -1)
        ids = self.tets[tid]
        return ids, weights

    def matrix(self, points):
        ids, w = self.bind(points)
        return sparse.csr_matrix(
            (w.ravel(), (np.repeat(np.arange(len(points)), 4), ids.ravel())),
            shape=(len(points), len(self.x)),
        )

    def muscles(self, levels=5):
        """Four longitudinal fibers plus radial and circumferential segments.

        Users may supply other polygonal paths as in the paper. Our default
        uses centers/radii of axial sections, then mesh-geodesic stress weights.
        """
        x = self.x
        zmin, zmax = np.quantile(x[:, 2], [0.12, 0.88])
        zlevels = np.linspace(zmin, zmax, levels)
        rings, centers = [], []
        for z in zlevels:
            sel = x[np.abs(x[:, 2] - z) < self.h * 1.2]
            c = np.array([*sel[:, :2].mean(0), z])
            rad = np.maximum(np.ptp(sel[:, :2], axis=0) * 0.27, self.h * 0.6)
            ring = np.array(
                [
                    c + [rad[0] * np.cos(t), rad[1] * np.sin(t), 0]
                    for t in np.arange(4) * np.pi / 2
                ]
            )
            rings.append(ring)
            centers.append(c)
        segments = []
        for i in range(levels - 1):
            segments.extend([[rings[i][j], rings[i + 1][j]] for j in range(4)])
        for i in range(levels):
            segments.extend([[centers[i], rings[i][j]] for j in range(4)])
            segments.extend([[rings[i][j], rings[i][(j + 1) % 4]] for j in range(4)])
        segments = np.array(segments)
        direction = segments[:, 1] - segments[:, 0]
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        # Connectivity through shared faces, Euclidean edge length on dual mesh.
        faces = np.concatenate(
            [
                np.sort(self.tets[:, idx], axis=1)
                for idx in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            ]
        )
        owners = np.tile(np.arange(len(self.tets)), 4)
        sort = np.lexsort(faces.T[::-1])
        equal = (faces[sort][1:] == faces[sort][:-1]).all(-1)
        ii, jj = owners[sort][:-1][equal], owners[sort][1:][equal]
        dist = np.linalg.norm(self.centers[ii] - self.centers[jj], axis=1)
        graph = sparse.coo_matrix(
            (np.r_[dist, dist], (np.r_[ii, jj], np.r_[jj, ii])),
            shape=(len(self.tets),) * 2,
        ).tocsr()
        seeds = self.tree.query(segments.mean(1))[1]
        geodesic = dijkstra(graph, directed=False, indices=seeds)
        weights = np.exp(-((geodesic / (np.ptp(x[:, 2]) * 0.22)) ** 2)).T
        return segments, direction, weights

    def save(self, path):
        np.savez_compressed(
            path,
            x=self.x,
            tets=self.tets,
            inv=self.inv,
            vol=self.vol,
            grad=self.grad,
            h=self.h,
            origin=self.origin,
            cells=np.array(list(self.cell_map)),
        )

    @classmethod
    def load(cls, path):
        z = np.load(path)
        obj = cls.__new__(cls)
        for k in ["x", "tets", "inv", "vol", "grad", "h", "origin"]:
            setattr(obj, k, z[k])
        obj.cell_map = {tuple(c): i for i, c in enumerate(z["cells"])}
        obj.centers = obj.x[obj.tets].mean(1)
        obj.tree = cKDTree(obj.centers)
        return obj
