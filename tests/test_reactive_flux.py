#!/usr/bin/env python3
"""Regression checks for reactive current and energy-resolved transmission."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ShowDyn import TrajData  # noqa: E402
from reactive_flux import (  # noqa: E402
    AMU_TO_ELECTRON_MASS,
    ANGSTROM_TO_BOHR,
    FS_TO_AU,
    energy_resolved_transmission,
    reactive_flux_diagnostics,
    surface_probability_current,
)


mass_amu = 1.00784
mass = mass_amu * AMU_TO_ELECTRON_MASS
nx = 256
x = np.linspace(-8.0, 8.0, nx, endpoint=False)
x_bohr = x * ANGSTROM_TO_BOHR
dx_bohr = x_bohr[1] - x_bohr[0]
wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx_bohr)

# A normalized plane wave has analytic current k/(m L).
k_index = 8
k0 = wave_numbers[k_index]
length_bohr = nx * dx_bohr
plane_bohr = np.exp(1j * k0 * x_bohr) / np.sqrt(length_bohr)
plane_angstrom = plane_bohr * np.sqrt(ANGSTROM_TO_BOHR)
plane_current = surface_probability_current(
    x, plane_angstrom[None, None, :], mass_amu, 0.0
)[0, 0]
expected_current = k0 / (mass * length_bohr) * FS_TO_AU
assert np.isclose(plane_current, expected_current, rtol=2.0e-12, atol=1.0e-14)

# Exact free-particle propagation of a right-moving localized packet.
sigma_bohr = 0.55 * ANGSTROM_TO_BOHR
center_bohr = -4.0 * ANGSTROM_TO_BOHR
k_packet = 8.0 / ANGSTROM_TO_BOHR
psi0_bohr = np.exp(
    -0.5 * ((x_bohr - center_bohr) / sigma_bohr) ** 2 + 1j * k_packet * x_bohr
)
psi0_bohr /= np.sqrt(np.sum(np.abs(psi0_bohr) ** 2) * dx_bohr)
coefficients = np.fft.fft(psi0_bohr)
time = np.linspace(0.0, 160.0, 321)
trajectory_bohr = np.empty((time.size, nx), dtype=complex)
for index, time_fs in enumerate(time):
    phase = np.exp(-0.5j * wave_numbers**2 * (time_fs * FS_TO_AU) / mass)
    trajectory_bohr[index] = np.fft.ifft(coefficients * phase)
trajectory = trajectory_bohr * np.sqrt(ANGSTROM_TO_BOHR)
density = np.abs(trajectory) ** 2
data = TrajData(
    x=x,
    t=time,
    psi=trajectory,
    psi1=None,
    psi2=None,
    rho=density,
    rho1=None,
    rho2=None,
    V=None,
    V11=None,
    V22=None,
    V12=None,
    V_lower=None,
    V_upper=None,
    Vadiab=None,
    Vdiab=None,
)
flux = reactive_flux_diagnostics(data, mass_amu, 0.0, 40.0, 120.0)
assert float(flux["cumulative_forward_flux"][-1]) > 0.98
assert float(flux["cumulative_backward_flux"][-1]) < 1.0e-3
assert float(flux["integrated_flux_side_error"]) < 4.0e-3
assert float(flux["continuity_rms_fs-1"]) < 2.0e-4

spectrum = energy_resolved_transmission(
    x,
    time,
    trajectory[None, ...],
    mass_amu,
    -2.0,
    2.0,
    window="none",
    incident_threshold=1.0e-3,
)
assert np.isfinite(spectrum["integrated_transmission"])
assert abs(float(spectrum["integrated_transmission"]) - 1.0) < 0.08
valid = np.asarray(spectrum["valid_transmission"], dtype=bool)
transmission = np.asarray(spectrum["transmission_probability"])
assert np.nanmedian(transmission[valid]) > 0.85
assert np.nanmedian(transmission[valid]) < 1.15

print("PASS reactive flux continuity and energy-resolved transmission")
