"""Spectral water transport following the authors' released implementation."""
import numpy as np
from scipy.linalg import eigh
from scipy.special import expit
from scipy.optimize import brentq


PRESSURE_CONSTANT = 11.538249539485484


class WaterModel:
    def __init__(self, rig, loss_rate=.006, initial_water=8., blocked=(), hydraulic_radius_ratio=.08):
        # The reference uses centimetres, grams, and seconds.
        r = np.asarray(rig['radius'], dtype=float) * 100
        length = np.asarray(rig['length'], dtype=float) * 100
        volume = np.pi * r * r * length
        self.volume = volume
        self.initial = volume / volume.sum() * initial_water
        if not 0 < hydraulic_radius_ratio <= 1:
            raise ValueError('The effective hydraulic radius ratio must be in (0, 1].')
        self.loss = loss_rate * r * length * rig.get('hydraulic_leaf', rig['leaf'])
        # The load-bearing rod is not one hollow water pipe. Keep tissue storage
        # and mechanical radii fixed while calibrating effective transport area.
        # Ratio 1 reproduces the original whole-cylinder transport convention.
        self.hydraulic_radius_ratio = hydraulic_radius_ratio
        hydraulic_r = r * hydraulic_radius_ratio
        conductance = 1e6 * np.pi * hydraulic_r**4 / (8 * .9 * length)
        lap = np.zeros((len(r), len(r)))
        for i, parent in enumerate(rig['parent']):
            if parent < 0 or i in blocked:
                continue
            g = .5 * (conductance[i] + conductance[parent])
            lap[i, parent] += g; lap[parent, i] += g
            lap[i, i] -= g; lap[parent, parent] -= g
        self.matrix = PRESSURE_CONSTANT * lap / volume[None, :] - np.diag(self.loss)
        self.sqrt_volume = np.sqrt(volume)
        symmetric = PRESSURE_CONSTANT * lap / np.sqrt(volume[:, None] * volume[None, :]) - np.diag(self.loss)
        self.eigenvalues, self.eigenvectors = eigh(symmetric)
        self.coefficients = self.eigenvectors.T @ (self.initial / self.sqrt_volume)

    def evaluate(self, seconds):
        seconds = np.atleast_1d(seconds)
        values = (self.eigenvectors @ (np.exp(self.eigenvalues[:, None] * seconds[None, :]) * self.coefficients[:, None])).T
        water = values * self.sqrt_volume
        if water.min() < -1e-8:
            raise RuntimeError(f'Negative water content: {water.min()}')
        return np.maximum(water, 0)

    def drydown_time(self, retained=.16):
        if not 0 < retained < 1:
            raise ValueError('Retained water must be in (0, 1).')
        # Multiple decay modes matter once local hydraulic gradients develop.
        def residual(t):
            return float(self.evaluate(t).sum() / self.initial.sum() - retained)
        upper = 1.
        while residual(upper) > 0 and upper < 1e12:
            upper *= 2
        if residual(upper) > 0:
            raise ValueError('The requested water level is unreachable with this transport graph.')
        return float(brentq(residual, 0., upper))


def modulus(relative_water, leaf, scene):
    """Logistic turgor mapping, calibrated in relative water coordinates."""
    activation = expit(22 * (relative_water - .65))
    dry = np.where(leaf, 8e4, 6e5)
    healthy = np.where(leaf, 2e9, 5e9)
    if scene == 'ficus':
        dry = np.where(leaf, 8e4, 2e8)
    return dry + (healthy - dry) * activation
