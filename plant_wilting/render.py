"""Material Gaussian skinning with the full spatial deformation Jacobian."""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization
from gsplat.cuda._wrapper import spherical_harmonics
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from .assets import ROOT
from gaussian_splashing.assets import camera
from rainy_gs.demo import Video


class GaussianPlant:
    def __init__(self, scene, device='cuda', rig_path=None, binding='material', covariance_mode='rod'):
        asset_path = Path(rig_path).with_name('gaussians.npz') if rig_path else ROOT / scene / 'gaussians.npz'
        if not asset_path.exists():
            asset_path = ROOT / scene / 'gaussians.npz'
        data = np.load(asset_path)
        rig = np.load(rig_path or ROOT / scene / 'rig.npz')
        self.covariance_mode = covariance_mode
        centers = (rig['head'] + rig['tail']) * .5
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        self.means, self.axes, self.sh, self.opacity = [t(data[k]) for k in ['means', 'axes', 'sh', 'opacity']]
        self.static = torch.as_tensor(rig['static'], device=device)
        self.centers = t(centers)
        adjacent = [{i} for i in range(len(centers))]
        for i, p in enumerate(rig['parent']):
            if p >= 0:
                adjacent[i].add(int(p)); adjacent[p].add(i)
        candidates = np.zeros((len(centers), 8), int)
        for i in range(len(centers)):
            ring = set(adjacent[i])
            for j in list(ring):
                ring.update(adjacent[j])
            ring = np.array(sorted(ring))
            ring = ring[np.argsort(np.linalg.norm(centers[ring] - centers[i], axis=1))[:8]]
            candidates[i] = np.resize(ring, 8)
        if binding in ('spatial', 'material'):
            # A compact kernel vanishes smoothly at the neighbor boundary.
            # This avoids jumps caused by changing topological influence sets.
            _, ids = cKDTree(centers).query(data['means'], k=min(16, len(centers)))
        else:
            ids = candidates[rig['node']]
        self.binding = binding
        self.ids = torch.as_tensor(ids, dtype=torch.long, device=device)
        delta = self.means[:, None] - self.centers[self.ids]
        d2 = (delta * delta).sum(-1) + .035**2
        raw = d2.pow(-2)
        raw_grad = raw[..., None] * (-4 * delta / d2[..., None])
        self.material_gate = torch.ones_like(raw)
        if binding == 'material':
            # Nearby leaves can belong to distant branches. Rest-material
            # membership suppresses such influences before spatial blending.
            children = np.flatnonzero(rig['parent'] >= 0)
            parents = rig['parent'][children]
            lengths = .5 * (rig['length'][children] + rig['length'][parents])
            graph = coo_matrix((lengths, (children, parents)), shape=(len(centers), len(centers))).tocsr()
            distances = shortest_path(graph, directed=False)
            geodesic = distances[rig['node'][:, None], ids]
            euclidean = np.linalg.norm(centers[rig['node'][:, None]] - centers[ids], axis=-1)
            gate = np.exp(-.5 * (np.maximum(geodesic - euclidean, 0) / .06)**2)
            self.material_gate = t(gate)
            raw *= self.material_gate
            raw_grad *= self.material_gate[..., None]
        if binding in ('spatial', 'material'):
            distance = torch.sqrt((delta * delta).sum(-1) + 1e-12)
            support = distance[:, -1:] + 1e-6
            ratio = distance / support
            phi = (1 - ratio).pow(4) * (1 + 4 * ratio)
            grad_ratio = delta / (distance * support)[..., None]
            grad_ratio -= (distance / support**2)[..., None] * (delta[:, -1:] / distance[:, -1:, None])
            grad_phi = (-20 * ratio * (1 - ratio).pow(3))[..., None] * grad_ratio
            raw_grad = raw_grad * phi[..., None] + raw[..., None] * grad_phi
            raw = raw * phi
        total = raw.sum(-1, keepdim=True)
        self.weight = raw / total
        self.grad = (raw_grad - self.weight[..., None] * raw_grad.sum(1, keepdim=True)) / total[..., None]

    def deform(self, pose):
        device = self.means.device
        rot = torch.as_tensor(Rotation.from_quat(pose[:, 3:]).as_matrix(), dtype=torch.float32, device=device)
        position = torch.as_tensor(pose[:, :3], dtype=torch.float32, device=device)
        rotation = rot[self.ids]
        offset = self.means[:, None] - self.centers[self.ids]
        transformed = torch.einsum('nkij,nkj->nki', rotation, offset) + position[self.ids]
        means = (self.weight[..., None] * transformed).sum(1)
        gradient = (self.weight[..., None, None] * rotation).sum(1)
        rod_rotation = gradient.clone()
        gradient += torch.einsum('nki,nkj->nij', transformed - means[:, None], self.grad)
        means = torch.where(self.static[:, None], self.means, means)
        eye = torch.eye(3, device=device)
        gradient = torch.where(self.static[:, None, None], eye, gradient)
        # The rod model bends locally without predicting a volumetric leaf strain.
        # Cross-branch skinning derivatives can stretch a splat by >20x; they
        # describe the interpolant, not the elastic strain of a leaf lamina.
        rotation_map = rod_rotation if self.covariance_mode == 'rod' else gradient
        rotation_map = torch.where(self.static[:, None, None], eye, rotation_map)
        u, _, vh = torch.linalg.svd(rotation_map)
        det = torch.linalg.det(u @ vh)
        u[:, :, -1] *= det[:, None]
        orientation = u @ vh
        axes = (orientation if self.covariance_mode == 'rod' else gradient) @ self.axes
        covariance = axes @ axes.transpose(-1, -2) + eye * 1e-10
        return means, covariance, orientation, gradient


def backdrop(cam):
    device = cam.view.device
    y, x = torch.meshgrid(torch.arange(cam.height, device=device) + .5, torch.arange(cam.width, device=device) + .5, indexing='ij')
    rays = torch.stack([(x - cam.K[0, 2]) / cam.K[0, 0], (y - cam.K[1, 2]) / cam.K[1, 1], torch.ones_like(x)], -1)
    c2w = torch.linalg.inv(cam.view)
    ray = rays @ c2w[:3, :3].T
    hit = cam.center - ray * (cam.center[2] / ray[..., 2])[..., None]
    shadow = torch.exp(-((hit[..., :2] / .34)**2).sum(-1)) * .14
    vertical = (y / cam.height)[..., None]
    bg = torch.tensor([.94, .953, .93], device=device) * (1 - vertical) + torch.tensor([.83, .86, .82], device=device) * vertical
    return bg - shadow[..., None]


@torch.no_grad()
def frame_image(plant, pose, cam):
    means, covariance, rot, gradient = plant.deform(pose)
    direction = torch.nn.functional.normalize(means - cam.center, dim=-1)
    direction = torch.einsum('ni,nij->nj', direction, rot)
    colors = (spherical_harmonics(3, direction, plant.sh) + .5).clamp_min(0)
    im, alpha, _ = rasterization(means=means, quats=None, scales=None, covars=covariance,
        opacities=plant.opacity, colors=colors, viewmats=cam.view[None], Ks=cam.K[None],
        width=cam.width, height=cam.height, packed=False, near_plane=.02, far_plane=30.)
    rgb = (im[0] + (1 - alpha[0]) * backdrop(cam)).clamp(0, 1)
    return Image.fromarray((rgb.cpu().numpy() * 255 + .5).astype(np.uint8))


def annotate(image, scene, retained, progress):
    draw = ImageDraw.Draw(image)
    fontpath = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title = ImageFont.truetype(fontpath, max(18, image.width // 38))
    small = ImageFont.truetype(fontpath, max(12, image.width // 70))
    margin = image.width // 25
    draw.text((margin, margin), scene.replace('_', ' ').upper(), font=title, fill='#253a2e')
    draw.text((margin, margin + title.size + 9), 'WATER TRANSPORT  /  PLANT WILTING', font=small, fill='#516252')
    y = image.height - margin - 18
    draw.text((margin, y - 24), f'Water retained   {retained * 100:.0f}%', font=small, fill='#365547')
    draw.rounded_rectangle((margin, y, margin + image.width * .22, y + 5), radius=2, fill='#becabd')
    draw.rounded_rectangle((margin, y, margin + max(3, image.width * .22 * retained), y + 5), radius=2, fill='#3f8972')
    draw.text((image.width - margin - 110, y - 10), f'{progress * 100:.0f}% dry-down', font=small, fill='#516252')
    return image


def resample_trajectory(poses, water, count):
    """Interpolate recorded rigid frames for smooth video playback."""
    if count is None or count == len(poses):
        return poses, water
    source_t = np.arange(len(poses), dtype=float)
    target_t = np.linspace(0, len(poses) - 1, count)
    output = np.empty((count, poses.shape[1], 7), dtype=np.float32)
    for i in range(poses.shape[1]):
        for k in range(3):
            output[:, i, k] = np.interp(target_t, source_t, poses[:, i, k])
        output[:, i, 3:] = Slerp(source_t, Rotation.from_quat(poses[:, i, 3:]))(target_t).as_quat()
    humidity = np.stack([np.interp(target_t, source_t, water[:, i]) for i in range(water.shape[1])], axis=1)
    return output, humidity


def main(args):
    src, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    source_text = Path(__file__).read_bytes()
    (out / 'render_source.py').write_bytes(source_text)
    cfg = json.loads((src / 'config.json').read_text())
    scene = cfg['scene']; plant = GaussianPlant(scene, rig_path=src / 'rig.npz', binding=args.binding,
                                               covariance_mode=args.covariance)
    tr = np.load(src / 'trajectory.npz')
    poses, water = resample_trajectory(tr['poses'], tr['water'], args.frames)
    ids = np.unique(np.linspace(0, len(poses) - 1, 5).astype(int)) if args.preview else np.arange(len(poses))
    video = None if args.preview else Video(out / 'demo.mp4', args.width, args.height, args.fps)
    samples = []
    try:
        for k in ids:
            progress = float(k / max(1, len(poses) - 1))
            azimuth = dict(potted_plant=5., ficus=-65., alocasia=-25.)[scene]
            cam = camera(args.width, args.height, azimuth=azimuth + args.orbit * progress,
                         elevation=16, radius=dict(potted_plant=3.15, ficus=2.75, alocasia=2.45)[scene] * max(1., args.width / args.height), target=(0, 0, .83))
            im = frame_image(plant, poses[k], cam)
            im = annotate(im, scene, water[k].sum() / tr['initial'].sum(), progress)
            if video:
                video.write(np.array(im))
            if k in np.unique(np.linspace(0, len(poses) - 1, 5).astype(int)):
                im.save(out / f'frame_{k:04d}.png')
                samples.append(im.copy())
            if k % 15 == 0:
                print(f'{scene} render {k + 1}/{len(poses)}', flush=True)
    finally:
        if video:
            video.close()
    sheet = Image.new('RGB', (args.width * len(samples), args.height))
    for k, im in enumerate(samples):
        sheet.paste(im, (k * args.width, 0))
    sheet.save(out / 'timeline.jpg')
    (out / 'render.json').write_text(json.dumps(dict(scene=scene, trajectory=str(src.resolve()),
        simulation_frames=len(tr['poses']), rendered_frames=len(ids), fps=args.fps,
        renderer_sha256=hashlib.sha256(source_text).hexdigest(),
        width=args.width, height=args.height, orbit_degrees=args.orbit, binding=args.binding,
        covariance_transport=args.covariance,
        interpolation='linear translation and quaternion slerp'), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--width', type=int, default=960); p.add_argument('--height', type=int, default=800)
    p.add_argument('--frames', type=int); p.add_argument('--fps', type=int, default=30); p.add_argument('--orbit', type=float, default=70.)
    p.add_argument('--preview', action='store_true')
    p.add_argument('--binding', choices=['spatial', 'material', 'topological'], default='material')
    p.add_argument('--covariance', choices=['rod', 'jacobian'], default='rod')
    main(p.parse_args())
