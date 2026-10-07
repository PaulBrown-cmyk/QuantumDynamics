#!/usr/bin/env python3
"""Strong-coupling non-Markovian HEOM regressions."""

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heom_nonmarkovian import HEOMConfig, ScaledHEOM, drude_matsubara
from quantum_fdt_density import (
    CMINV_TO_HARTREE,
    FS_TO_AU,
    KB_HARTREE_PER_K,
)

hamiltonian = np.array([[0.0, 0.003], [0.003, 0.002]], dtype=complex)
coupling = np.diag([-1.0, 1.0])
initial = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
times = np.linspace(0.0, 2000.0, 41)

# Zero reorganization energy must recover exact unitary dynamics and zero ADOs.
zero_config = HEOMConfig(
    temperature_k=300.0,
    reorganization_cminv=0.0,
    cutoff_fs_inv=0.03,
    matsubara_terms=1,
    hierarchy_depth=2,
    system_states=2,
    rtol=1.0e-10,
    atol=1.0e-12,
)
zero_solver = ScaledHEOM(hamiltonian, coupling, zero_config)
unitary_density, zero_ado_norm = zero_solver.solve(initial, times[:10])
exact = []
for time in times[:10]:
    propagator = expm(-1.0j * hamiltonian * time)
    exact.append(propagator @ initial @ propagator.conj().T)
assert np.max(np.abs(unitary_density - np.asarray(exact))) < 3.0e-9
assert np.max(zero_ado_norm) == 0.0

# Bath decomposition plus residual terminator must preserve exact real integral.
beta = 1.0 / (KB_HARTREE_PER_K * 300.0)
reorganization = 600.0 * CMINV_TO_HARTREE
cutoff = 0.03 / FS_TO_AU
rates, coefficients, tail = drude_matsubara(reorganization, cutoff, beta, 1)
target = 2.0 * reorganization / (beta * cutoff)
assert np.isclose(np.sum(coefficients.real / rates) + tail, target, rtol=2.0e-15)

# Strong-coupling hierarchy: invariants, positive root state, active memory ADOs.
strong_config = HEOMConfig(
    temperature_k=300.0,
    reorganization_cminv=600.0,
    cutoff_fs_inv=0.03,
    matsubara_terms=1,
    hierarchy_depth=3,
    system_states=2,
    rtol=1.0e-9,
    atol=1.0e-11,
)
strong_solver = ScaledHEOM(hamiltonian, coupling, strong_config)
root, ado_norm = strong_solver.solve(initial, times)
assert strong_solver.nado == 10
assert np.max(np.abs(np.trace(root, axis1=1, axis2=2) - 1.0)) < 2.0e-12
assert max(np.max(np.abs(state - state.conj().T)) for state in root) < 2.0e-12
assert min(np.linalg.eigvalsh(state)[0] for state in root) > -2.0e-10
assert np.max(ado_norm) > 0.1

# Removing hierarchy memory must measurably change strong-coupling dynamics.
memoryless_config = HEOMConfig(
    temperature_k=300.0,
    reorganization_cminv=600.0,
    cutoff_fs_inv=0.03,
    matsubara_terms=1,
    hierarchy_depth=0,
    system_states=2,
    rtol=1.0e-9,
    atol=1.0e-11,
)
memoryless, _ = ScaledHEOM(hamiltonian, coupling, memoryless_config).solve(
    initial, times
)
assert np.max(np.abs(root - memoryless)) > 0.05

print("PASS strong-coupling non-Markovian HEOM")
