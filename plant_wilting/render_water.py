"""Render paired Gaussian appearance and measured hydration from saved trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from gsplat import rasterization
from gsplat.cuda._wrapper import spherical_harmonics

from .render import GaussianPlant, backdrop, camera, resample_trajectory, Video


COLOR_POSITIONS = np.array([0., .25, .5, .75, 1.])
COLOR_RGB = np.array([[193, 57, 47], [234, 151, 55], [183, 207, 125],
                      [48, 175, 197], [43, 77, 168]], dtype=float) / 255
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
SIZE = (1920, 1080)


def palette_numpy(values):
    values = np.asarray(values)
    return np.stack([np.interp(values, COLOR_POSITIONS, COLOR_RGB[:, k]) for k in range(3)], axis=-1)


def palette_tensor(values):
    rgb = torch.as_tensor(COLOR_RGB, device=values.device, dtype=values.dtype)
    coordinate = values.clamp(0, 1) * 4
    index = coordinate.long().clamp(max=3)
    fraction = coordinate - index
    return rgb[index] * (1 - fraction[..., None]) + rgb[index + 1] * fraction[..., None]


def map_water(plant, relative):
    values = torch.as_tensor(relative, dtype=plant.means.dtype, device=plant.means.device)
    return (plant.weight * values[plant.ids]).sum(1)


@torch.no_grad()
def render_pair(plant, pose, relative, cam):
    means, covariance, rotation, _ = plant.deform(pose)
    direction = torch.nn.functional.normalize(means - cam.center, dim=-1)
    direction = torch.einsum('ni,nij->nj', direction, rotation)
    appearance = (spherical_harmonics(3, direction, plant.sh) + .5).clamp_min(0)
    scalar = map_water(plant, relative)
    heat = palette_tensor(scalar)
    luminance = (appearance * appearance.new_tensor([.2126, .7152, .0722])).sum(-1).clamp(0, 1)
    # Preserve leaf veins and depth cues in brightness; hue encodes hydration.
    heat = heat * (.65 + .35 * torch.sqrt(luminance))[:, None]
    # Keep the pot in its original appearance: its water content is not simulated.
    heat = torch.where(plant.static[:, None], appearance, heat)
    colors = torch.cat([appearance, heat], dim=-1)
    image, alpha, _ = rasterization(means=means, quats=None, scales=None, covars=covariance,
        opacities=plant.opacity, colors=colors, viewmats=cam.view[None], Ks=cam.K[None],
        width=cam.width, height=cam.height, packed=False, near_plane=.02, far_plane=30.)
    background = backdrop(cam)
    rgb = (image[0] + (1 - alpha[0]) * torch.cat([background, background], dim=-1)).clamp(0, 1)
    result = (rgb.cpu().numpy() * 255 + .5).astype(np.uint8)
    return Image.fromarray(result[..., :3]), Image.fromarray(result[..., 3:])


class WaterDashboard:
    def __init__(self, scene, seconds, water, initial):
        self.scene = scene
        self.seconds = np.asarray(seconds)
        self.total = np.asarray(water).sum(1)
        self.initial = float(np.sum(initial))
        self.fraction = self.total / self.initial
        self.local_quantiles = np.quantile(np.asarray(water) / np.asarray(initial)[None], [.1, .9], axis=1)
        self.fonts = {n: ImageFont.truetype(FONT, n) for n in [15, 17, 18, 20, 24, 32, 44]}
        self.base = Image.new('RGB', SIZE, '#edf2ec')
        draw = ImageDraw.Draw(self.base)
        draw.text((58, 30), scene.replace('_', ' ').upper(), font=self.fonts[32], fill='#203b31')
        draw.text((59, 72), 'WATER LOSS  /  MATERIAL SOFTENING  /  PLANT WILTING', font=self.fonts[17], fill='#647969')
        draw.line((958, 130, 958, 835), fill='#d3ddd0', width=2)
        draw.rounded_rectangle((42, 882, 1878, 1050), radius=18, fill='#f9faf7')
        self.chart_box = (100, 929, 1130, 1013)
        x0, y0, x1, y1 = self.chart_box
        for level in [0, .5, 1]:
            y = int(y1 - level * (y1 - y0))
            draw.line((x0, y, x1, y), fill='#dce4d8', width=1)
            draw.text((60, y - 9), f'{level*100:.0f}', font=self.fonts[15], fill='#69806c')
        draw.text((100, 895), 'TOTAL WATER (%)  /  SHADED: LOCAL 10-90% RANGE', font=self.fonts[17], fill='#466653')
        draw.text((100, 1024), '0 min', font=self.fonts[15], fill='#69806c')
        duration = float(self.seconds[-1])
        draw.text((1045, 1024), f'{duration / 60:.2f} min', font=self.fonts[15], fill='#69806c')
        draw.text((460, 1024), 'SIMULATED DRY-DOWN TIME', font=self.fonts[15], fill='#69806c')
        self.curve = np.stack([x0 + self.seconds / max(duration, 1e-9) * (x1 - x0),
                               y1 - self.fraction * (y1 - y0)], axis=1)
        self.curve_points = [tuple(p) for p in self.curve]
        self.local_y = y1 - self.local_quantiles * (y1 - y0)
        draw.line(self.curve_points, fill='#d5e2dd', width=2)
        # Fixed limits across all frames and scenes; no per-frame contrast normalization.
        bar = (palette_numpy(np.linspace(1, 0, 440))[:, None] * 255 + .5).astype(np.uint8)
        self.legend = Image.fromarray(np.repeat(bar, 20, axis=1))

    def frame(self, appearance, heat, index):
        output = self.base.copy()
        output.paste(appearance, (80, 115))
        output.paste(heat, (1040, 115))
        draw = ImageDraw.Draw(output)
        draw.text((108, 130), '3D GAUSSIAN APPEARANCE', font=self.fonts[18], fill='#466653')
        draw.text((1068, 130), 'LOCAL WATER / INITIAL', font=self.fonts[18], fill='#466653')
        output.paste(self.legend, (1810, 270))
        for value in [1., .75, .5, .25, 0.]:
            y = int(270 + (1 - value) * 439)
            draw.line((1806, y, 1834, y), fill='#405c4e', width=1)
            draw.text((1842, y - 9), f'{value*100:.0f}%', font=self.fonts[17], fill='#466653')
        draw.text((1788, 233), 'WATER', font=self.fonts[15], fill='#466653')
        seconds = float(self.seconds[index])
        minute = int(seconds // 60)
        second = int(seconds % 60)
        draw.text((1512, 38), f'SIMULATED TIME   {minute:02d}:{second:02d}', font=self.fonts[20], fill='#466653')
        draw.text((1513, 72), 'Time-compressed playback', font=self.fonts[17], fill='#69806c')
        if index > 0:
            xs = self.curve[:index + 1, 0]
            lower = list(zip(xs, self.local_y[0, :index + 1]))
            upper = list(zip(xs[::-1], self.local_y[1, :index + 1][::-1]))
            draw.polygon(lower + upper, fill='#c8dcd6')
            draw.line(self.curve_points[:index + 1], fill='#347fa1', width=4)
        x, y = self.curve[index]
        draw.line((x, 927, x, 1015), fill='#a8bfbe', width=1)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill='#2c83a7', outline='#f9faf7', width=2)
        fraction = float(np.clip(self.fraction[index], 0, 1))
        lost = max(0., self.initial - float(self.total[index]))
        shade = tuple((palette_numpy(fraction) * 255 + .5).astype(int).tolist())
        draw.text((1220, 904), 'WATER RETAINED', font=self.fonts[17], fill='#69806c')
        draw.ellipse((1187, 948, 1205, 966), fill=shade)
        draw.text((1216, 934), f'{fraction * 100:.1f}%', font=self.fonts[44], fill='#285f73')
        draw.text((1220, 996), f'{self.total[index]:.2f} / {self.initial:.2f} g', font=self.fonts[20], fill='#466653')
        draw.text((1544, 904), 'WATER LOST', font=self.fonts[17], fill='#69806c')
        draw.text((1540, 934), f'{lost:.2f} g', font=self.fonts[44], fill='#9b6946')
        draw.text((1544, 996), f'{(1 - fraction) * 100:.1f}% of initial', font=self.fonts[20], fill='#69806c')
        return output


def main(args):
    source, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((source / 'config.json').read_text())
    scene = cfg['scene']
    tr = np.load(source / 'trajectory.npz')
    poses, water = resample_trajectory(tr['poses'], tr['water'], args.frames)
    seconds = np.interp(np.linspace(0, len(tr['poses']) - 1, len(poses)),
                        np.arange(len(tr['poses'])), tr['seconds'])
    initial = tr['initial']
    if not np.isfinite(water).all() or np.any(initial <= 0) or np.any(water < 0):
        raise ValueError('Hydration data must be finite and nonnegative with positive initial mass.')
    relative = water / initial[None]
    if relative.max() > 1 + 1e-6:
        raise ValueError('This fixed 0-100% palette cannot show water above the initial value.')
    plant = GaussianPlant(scene, rig_path=source / 'rig.npz', binding=args.binding, covariance_mode=args.covariance)
    dashboard = WaterDashboard(scene, seconds, water, initial)
    selected = np.unique(np.linspace(0, len(poses) - 1, 5).astype(int))
    indices = selected if args.preview else np.arange(len(poses))
    size = (args.width, round(args.width * 9 / 32) * 2)
    panel = max(16, round(760 * args.width / SIZE[0]))
    video = None if args.preview else Video(output / 'water_demo.mp4', *size, args.fps)
    samples = []
    scalar_errors = []
    try:
        for index in indices:
            progress = float(index / max(1, len(poses) - 1))
            azimuth = dict(potted_plant=5., ficus=-65., alocasia=-25.)[scene] + args.orbit * progress
            cam = camera(panel, panel, azimuth=azimuth, elevation=16,
                radius=dict(potted_plant=3.15, ficus=2.75, alocasia=2.45)[scene], target=(0, 0, .83))
            left, right = render_pair(plant, poses[index], relative[index], cam)
            if panel != 760:
                left = left.resize((760, 760), Image.Resampling.LANCZOS)
                right = right.resize((760, 760), Image.Resampling.LANCZOS)
            frame = dashboard.frame(left, right, index)
            if size != SIZE:
                frame = frame.resize(size, Image.Resampling.LANCZOS)
            if video:
                video.write(np.asarray(frame))
            if index in selected:
                frame.save(output / f'frame_{index:04d}.png')
                samples.append(frame.resize((960, 540), Image.Resampling.LANCZOS))
                with torch.no_grad():
                    mapped = map_water(plant, relative[index])
                    error = max(float(relative[index].min() - mapped.min()),
                                float(mapped.max() - relative[index].max()), 0.)
                    scalar_errors.append(error)
                    assert error < 1e-6, error
            if index % 30 == 0:
                print(f'{scene}: water visualization {index + 1}/{len(poses)}', flush=True)
    finally:
        if video:
            video.close()
    sheet = Image.new('RGB', (960, 540 * len(samples)))
    for index, frame in enumerate(samples):
        sheet.paste(frame, (0, 540 * index))
    sheet.save(output / 'timeline.jpg')
    snapshots = {}
    for path in [Path(__file__), Path(__file__).with_name('render.py')]:
        data = path.read_bytes()
        (output / path.name).write_bytes(data)
        snapshots[path.name] = hashlib.sha256(data).hexdigest()
    metadata = dict(scene=scene, source=str(source.resolve()), width=size[0], height=size[1],
        frames=len(indices), fps=args.fps, orbit_degrees=args.orbit, preview=args.preview,
        covariance_transport=args.covariance,
        binding=args.binding,
        source_trajectory_sha256=hashlib.sha256((source / 'trajectory.npz').read_bytes()).hexdigest(),
        scalar='per-segment water mass divided by its initial mass, interpolated by Gaussian skinning weights',
        heatmap_shading='hydration hue with brightness modulated by the original Gaussian luminance',
        color_range=[0, 1], color_positions=COLOR_POSITIONS.tolist(), color_rgb=COLOR_RGB.tolist(),
        max_node_relative_water_spread=float(np.ptp(relative, axis=1).max()),
        scalar_interpolation_max_bound_error=max(scalar_errors),
        simulated_duration_seconds=float(seconds[-1]),
        initial_water_g=float(initial.sum()), final_water_g=float(water[-1].sum()),
        source_hashes=snapshots)
    (output / 'water_render.json').write_text(json.dumps(metadata, indent=2))
    np.savez_compressed(output / 'water_plot.npz', seconds=seconds, total_water_g=water.sum(1),
        relative_water=relative, initial_water_g=initial)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, default=240)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--orbit', type=float, default=70.)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--covariance', choices=['rod', 'jacobian'], default='rod')
    parser.add_argument('--binding', choices=['spatial', 'material', 'topological'], default='material')
    main(parser.parse_args())
