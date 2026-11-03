#!/usr/bin/env python3
"""Regression checks for canonical and stochastic thermal initial states."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantum_fdt_density import (  # noqa: E402
    Model,
    build_hamiltonian,
    canonical_populations,
    prepare_initial_density_energy,
    thermal_pure_density,
)


energies = np.array([0.0, 0.001, 0.003])
low = canonical_populations(energies, 1.0)
assert low[0] > 1.0 - 1.0e-12
high = canonical_populations(energies, 1.0e12)
assert np.max(np.abs(high - 1.0 / energies.size)) < 1.0e-8

rho_sampled, diagnostics = thermal_pure_density(
    energies, temperature_k=300.0, samples=12000, seed=90210
)
exact = np.diag(canonical_populations(energies, 300.0))
assert np.isclose(np.trace(rho_sampled), 1.0, atol=2.0e-14)
assert np.linalg.eigvalsh(rho_sampled)[0] > -2.0e-14
assert np.linalg.norm(rho_sampled - exact, ord="fro") < 0.025
assert diagnostics["thermal_effective_sample_size"] > 0.5 * diagnostics["thermal_samples"]
assert diagnostics["thermal_trace_estimator_error"] < 0.03
assert diagnostics["thermal_population_error"] < 0.015

model = Model(nx=8)
x, hamiltonian = build_hamiltonian(model)
system_energies, eigenvectors = np.linalg.eigh(hamiltonian)
rho_gibbs, gibbs_diagnostics = prepare_initial_density_energy(
    "gibbs", model, x, system_energies, eigenvectors, 100.0
)
expected = canonical_populations(system_energies, 100.0)
assert np.allclose(rho_gibbs, np.diag(expected), atol=1.0e-15)
assert gibbs_diagnostics["thermal_trace_estimator_error"] == 0.0
assert gibbs_diagnostics["thermal_density_frobenius_error"] == 0.0

rho_wavepacket, _ = prepare_initial_density_energy(
    "wavepacket", model, x, system_energies, eigenvectors, 100.0
)
assert np.isclose(np.trace(rho_wavepacket), 1.0, atol=2.0e-14)
assert np.linalg.matrix_rank(rho_wavepacket, tol=1.0e-10) == 1

print("PASS canonical and stochastic thermal initial states")
