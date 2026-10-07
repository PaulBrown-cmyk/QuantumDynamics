#!/usr/bin/env python3
"""Regressions for correlated HEOM, spin bath, 3D transfer, and PES import."""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heom_nonmarkovian import HEOMConfig, ScaledHEOM
from multidimensional_transfer_3d import Model3D, SplitOperator3D
from potential_data import DiabaticPES3D
from quantum_fdt_density import ANGSTROM_TO_BOHR, CMINV_TO_HARTREE
from spin_bath_non_gaussian import (
    ExactSpinBath,
    SpinBathConfig,
    trace_distance,
)

# Stationary HEOM solve produces nonzero correlated ADOs and remains stationary.
hamiltonian = np.array([[0.0, 0.003], [0.003, 0.002]], dtype=complex)
coupling = np.diag([-1.0, 1.0])
heom = ScaledHEOM(
    hamiltonian,
    coupling,
    HEOMConfig(
        temperature_k=300.0,
        reorganization_cminv=400.0,
        cutoff_fs_inv=0.04,
        matsubara_terms=1,
        hierarchy_depth=2,
        system_states=2,
        rtol=1.0e-10,
        atol=1.0e-12,
    ),
)
equilibrium, residual = heom.equilibrium_hierarchy()
assert residual < 2.0e-11
assert np.isclose(np.trace(equilibrium[0]), 1.0, atol=2.0e-12)
assert np.linalg.norm(equilibrium[1:]) > 1.0e-3
stationary, _ = heom.solve_hierarchy(equilibrium, np.array([0.0, 500.0]))
assert np.max(abs(stationary[-1] - stationary[0])) < 2.0e-9
projector = np.diag([1.0, 0.0])
prepared = heom.prepare_correlated(equilibrium, projector)
assert np.isclose(np.trace(prepared[0]), 1.0, atol=2.0e-12)
assert np.linalg.norm(prepared[1:]) > 1.0e-4

# Exact finite spin bath preserves physical density and exhibits memory backflow.
spin_bath = ExactSpinBath(
    SpinBathConfig(
        bath_frequencies_cminv=(80.0, 130.0, 210.0),
        bath_couplings_cminv=(55.0, 45.0, 35.0),
        temperature_k=250.0,
    )
)
times = np.linspace(0.0, 900.0, 301)
reactant = np.diag([1.0, 0.0]).astype(complex)
product = np.diag([0.0, 1.0]).astype(complex)
rho_reactant = spin_bath.evolve(reactant, times)
rho_product = spin_bath.evolve(product, times)
assert np.max(abs(np.trace(rho_reactant, axis1=1, axis2=2) - 1.0)) < 2.0e-13
assert min(np.linalg.eigvalsh(item)[0] for item in rho_reactant) > -2.0e-13
distance = trace_distance(rho_reactant, rho_product)
assert np.sum(np.maximum(np.diff(distance), 0.0)) > 1.0e-3

# 3D propagation conserves norm. External physical-unit PES reproduces analytic grid.
model = Model3D(nx=10, ny=8, nz=6)
analytic = SplitOperator3D(model)
initial_norm = sum(analytic.populations())
for _ in range(8):
    analytic.step(0.01)
assert abs(sum(analytic.populations()) - initial_norm) < 3.0e-13

reference = SplitOperator3D(model)
pes = DiabaticPES3D(
    reference.x / ANGSTROM_TO_BOHR,
    reference.y / ANGSTROM_TO_BOHR,
    reference.z / ANGSTROM_TO_BOHR,
    reference.v11 / CMINV_TO_HARTREE,
    reference.v22 / CMINV_TO_HARTREE,
    reference.v12 / CMINV_TO_HARTREE,
)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "pes.h5"
    pes.to_hdf5(path, source="regression fixture")
    loaded = DiabaticPES3D.from_hdf5(path)
    external = SplitOperator3D(model, loaded)
    assert np.max(abs(external.v11 - reference.v11)) < 2.0e-15
    assert np.max(abs(external.v22 - reference.v22)) < 2.0e-15
    assert np.max(abs(external.v12 - reference.v12)) < 2.0e-15
    external.step(0.01)
    assert abs(sum(external.populations()) - 1.0) < 3.0e-13
    with h5py.File(path, "r+") as handle:
        handle["x_angstrom"].attrs["units"] = "bohr"
    try:
        DiabaticPES3D.from_hdf5(path)
    except ValueError as error:
        assert "units" in str(error)
    else:
        raise AssertionError("PES importer accepted incorrect coordinate units")

print("PASS correlated HEOM, non-Gaussian spin bath, 3D transfer, and PES import")
