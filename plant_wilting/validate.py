"""Numerical checks for transport, Gaussian binding, and recorded trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from .assets import ROOT, SCENES
from .hydraulics import WaterModel


def water_checks():
    rig = dict(radius=np.array([.006, .003, .0015, .001]),
               length=np.array([.07, .08, .06, .04]),
               parent=np.array([-1, 0, 1, 1]), leaf=np.array([False, False, True, True]))
    model = WaterModel(rig)
    t = np.array([0., .001, .1, 1., 100.])
    analytic = model.evaluate(t)
    reference = np.array([expm(model.matrix * s) @ model.initial for s in t])
    error = np.max(np.abs(analytic - reference)) / np.max(model.initial)
    assert error < 2e-8, error
    conservation = WaterModel(rig, loss_rate=0.)
    residual = np.max(np.abs(conservation.matrix.sum(0)))
    assert residual < 1e-7, residual
    values = model.evaluate(np.linspace(0, model.drydown_time(), 100))
    endpoint_error = abs(values[-1].sum() / model.initial.sum() - .16)
    assert endpoint_error < 1e-8
    assert (np.diff(values.sum(1)) <= 1e-8).all()
    assert np.all(values >= 0)
    budget = np.max(np.abs(model.matrix.sum(0) + model.loss))
    assert budget < 1e-7
    return dict(exponential_relative_error=float(error),
                closed_system_column_sum_error=float(residual),
                loss_budget_error=float(budget), retained_target_error=float(endpoint_error),
                positive=True, total_water_monotonic=True)


def binding_checks(scene, rig_path=None):
    import torch
    from scipy.spatial.transform import Rotation
    from .render import GaussianPlant
    plant = GaussianPlant(scene, rig_path=rig_path)
    centers = plant.centers.cpu().numpy()
    pose = np.c_[centers, np.tile([0., 0., 0., 1.], (len(centers), 1))]
    with torch.no_grad():
        means, cov, rotation, gradient = plant.deform(pose)
        identity_error = float((means - plant.means).abs().max())
        jacobian_error = float((gradient - torch.eye(3, device=gradient.device)).abs().max())
        eig = torch.linalg.eigvalsh(cov)
        assert identity_error < 3e-6 and jacobian_error < 2e-5
        assert float(eig.min()) > 0
        rng = np.random.default_rng(7)
        pose[:, 3:] = Rotation.from_rotvec(rng.normal(size=(len(pose), 3)) * .08).as_quat()
        pose[:, :3] += rng.normal(size=(len(pose), 3)) * .003
        _, _, _, mapped_jacobian = plant.deform(pose)
        _, rotated_covariance, orientation, _ = plant.deform(pose)
        covariance_eigenvalues = torch.linalg.eigvalsh(rotated_covariance)
        initial_eigenvalues = torch.linalg.eigvalsh(plant.axes @ plant.axes.transpose(-1, -2) + torch.eye(3, device=eig.device) * 1e-10)
        covariance_spectrum_error = float((covariance_eigenvalues - initial_eigenvalues).abs().max())
        assert covariance_spectrum_error < 1e-6
        orthogonality_error = float((orientation @ orientation.transpose(-1, -2) - torch.eye(3, device=eig.device)).abs().max())
        assert orthogonality_error < 1e-5
        moving = np.flatnonzero(~plant.static.cpu().numpy())
        selected = moving[np.linspace(0, len(moving) - 1, 40).astype(int)]
        ids = plant.ids.cpu().numpy()[selected]
        points = plant.means.cpu().numpy()[selected].astype(float)
        rotations = Rotation.from_quat(pose[:, 3:]).as_matrix()[ids]

        def mapping(x):
            delta = x[:, None] - centers[ids]
            weight = (np.sum(delta**2, axis=-1) + .035**2)**-2
            weight *= plant.material_gate.cpu().numpy()[selected]
            if plant.binding in ('spatial', 'material'):
                distance = np.sqrt(np.sum(delta**2, axis=-1) + 1e-12)
                ratio = distance / (distance[:, -1:] + 1e-6)
                weight *= (1 - ratio)**4 * (1 + 4 * ratio)
            weight /= weight.sum(1, keepdims=True)
            transformed = np.einsum('nkij,nkj->nki', rotations, delta) + pose[ids, :3]
            return np.sum(weight[..., None] * transformed, axis=1)

        numerical = []
        for axis in np.eye(3) * 1e-5:
            numerical.append((mapping(points + axis) - mapping(points - axis)) / 2e-5)
        numerical = np.stack(numerical, axis=-1)
        fd_error = float(np.max(np.abs(numerical - mapped_jacobian[selected].cpu().numpy())))
        assert fd_error < 2e-4, fd_error
    return dict(identity_position_error=identity_error, identity_jacobian_error=jacobian_error,
                covariance_min_eigenvalue=float(eig.min()), finite_difference_jacobian_error=fd_error,
                covariance_spectrum_error=covariance_spectrum_error, rotation_orthogonality_error=orthogonality_error)


def trajectory_checks(folder):
    from scipy.spatial.transform import Rotation
    cfg = json.loads((folder / 'config.json').read_text())
    rig = np.load(folder / 'rig.npz')
    tr = np.load(folder / 'trajectory.npz')
    pose = tr['poses']; centers = .5 * (rig['head'] + rig['tail'])
    assert np.isfinite(pose).all()
    root_motion = np.linalg.norm(pose[:, 0, :3] - centers[0], axis=-1).max()
    assert root_motion < 1e-6, root_motion
    qerr = np.abs(np.linalg.norm(pose[..., 3:], axis=-1) - 1).max()
    assert qerr < 1e-4
    gaps = []
    for state in pose:
        rot = Rotation.from_quat(state[:, 3:]).as_matrix()
        head = state[:, :3] + np.einsum('nij,nj->ni', rot, rig['head'] - centers)
        tail = state[:, :3] + np.einsum('nij,nj->ni', rot, rig['tail'] - centers)
        gaps.append(np.linalg.norm(head[1:] - tail[rig['parent'][1:]], axis=1).max())
    assert max(gaps) < .005, max(gaps)
    water = tr['water']
    assert (water >= 0).all() and (np.diff(water.sum(1)) <= 1e-7).all()
    return dict(frames=len(pose), root_motion_m=float(root_motion),
                quaternion_norm_error=float(qerr), max_joint_gap_m=float(max(gaps)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu-binding', action='store_true')
    p.add_argument('--trajectories', nargs='*', default=[])
    p.add_argument('--output', default='outputs/plant_wilting/validation.json')
    args = p.parse_args()
    report = dict(water=water_checks())
    if args.gpu_binding:
        if args.trajectories:
            report['binding'] = {s: binding_checks(json.loads((Path(s) / 'config.json').read_text())['scene'],
                                                   Path(s) / 'rig.npz') for s in args.trajectories}
        else:
            report['binding'] = {s: binding_checks(s) for s in SCENES}
    report['trajectories'] = {s: trajectory_checks(Path(s)) for s in args.trajectories}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
