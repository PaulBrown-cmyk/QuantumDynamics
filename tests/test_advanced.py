#!/usr/bin/env python3
"""Physics regressions for quantum-FDT density and 2D wavepacket solvers."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multidimensional_transfer import Model2D, SplitOperator2D  # noqa: E402
from quantum_fdt_density import (  # noqa: E402
    KB_HARTREE_PER_K,
    davies_rate_matrix,
    propagate_density,
)


energies = np.array([0.0, 0.01])
coupling = np.array([[0.1, 1.0], [1.0, -0.2]])
temperature = 300.0
generator, escape, dephasing = davies_rate_matrix(
    energies, coupling, temperature, gamma_fs_inv=0.05, cutoff_fs_inv=0.2
)
beta = 1.0 / (KB_HARTREE_PER_K * temperature)
assert np.isclose(generator[1, 0] / generator[0, 1], np.exp(-beta * 0.01), rtol=1e-13)
gibbs = np.exp(-beta * energies)
gibbs /= gibbs.sum()
assert np.max(np.abs(generator @ gibbs)) < 1e-20

initial = np.array([1.0 + 0.0j, 0.0 + 0.0j])
rho = propagate_density(
    energies, np.eye(2), initial, generator, escape, dephasing,
    np.linspace(0.0, 1.0e6, 20),
)
for state in rho:
    assert np.isclose(np.trace(state), 1.0, atol=2e-14)
    assert np.max(np.abs(state - state.conj().T)) < 2e-14
    assert np.linalg.eigvalsh(state)[0] > -2e-14
assert np.allclose(np.real(np.diag(rho[-1])), gibbs, atol=2e-5)

solver = SplitOperator2D(Model2D(nx=20, ny=12))
initial_norm = sum(solver.populations())
for _ in range(20):
    solver.step(0.01)
final_norm = sum(solver.populations())
assert abs(final_norm - initial_norm) < 2e-13

print("PASS quantum KMS density dynamics and 2D norm")
