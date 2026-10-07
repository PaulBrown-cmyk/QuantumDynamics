#!/usr/bin/env python3
"""Regression checks for Wigner phase-space diagnostics."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ShowDyn import _trapezoid, compute_wigner


def normalized_gaussian(x, center, sigma, wave_number):
    psi = np.exp(-((x - center) ** 2) / (2.0 * sigma**2) + 1j * wave_number * x)
    return psi / np.sqrt(_trapezoid(np.abs(psi) ** 2, x))


nx = 256
dx = 0.05
x = (np.arange(nx) - nx // 2) * dx
sigma = 0.5
k0 = 3.0
psi = normalized_gaussian(x, center=-1.0, sigma=sigma, wave_number=k0)

xw, k, wigner = compute_wigner(x, psi, max_points=None)
rho = _trapezoid(wigner, k, axis=1)
momentum_density = _trapezoid(wigner, xw, axis=0)
dk = k[1] - k[0]

assert abs(k[np.argmax(momentum_density)] - k0) < 2.0 * dk
assert np.max(np.abs(rho - np.abs(psi) ** 2)) < 1.0e-10
assert abs(float(_trapezoid(rho, xw)) - 1.0) < 1.0e-10
assert np.min(wigner) > -1.0e-10

# Resampling keeps the integrated population while reducing animation cost.
x_small, k_small, wigner_small = compute_wigner(x, psi, max_points=64)
rho_small = _trapezoid(wigner_small, k_small, axis=1)
assert x_small.size == 64
assert abs(float(_trapezoid(rho_small, x_small)) - 1.0) < 5.0e-4

# A separated coherent superposition has interference fringes and negative W.
left = normalized_gaussian(x, center=-1.2, sigma=0.35, wave_number=0.0)
right = normalized_gaussian(x, center=1.2, sigma=0.35, wave_number=0.0)
cat = left + right
cat /= np.sqrt(_trapezoid(np.abs(cat) ** 2, x))
_, _, wigner_cat = compute_wigner(x, cat, max_points=None)
assert np.min(wigner_cat) < -0.05

print("PASS Wigner marginals, resampling, and negativity")
