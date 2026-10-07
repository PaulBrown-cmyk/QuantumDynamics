#!/usr/bin/env python3
"""Strong-coupling non-Markovian H-atom transfer with scaled HEOM.

The bath is harmonic and Gaussian, has a Drude-Lorentz spectral density, and
couples linearly to a dimensionless transfer coordinate.  Its finite-temperature
correlation function is expanded into one Drude pole plus Matsubara poles.  A
scaled hierarchy of auxiliary density operators (ADOs) retains bath memory and
system-bath correlations beyond Born, Markov, and secular approximations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import solve_ivp

from quantum_fdt_density import (
    ANGSTROM_TO_BOHR,
    CMINV_TO_HARTREE,
    FS_TO_AU,
    KB_HARTREE_PER_K,
    Model,
    build_hamiltonian,
    initial_state,
)


@dataclass(frozen=True)
class HEOMConfig:
    temperature_k: float = 300.0
    reorganization_cminv: float = 500.0
    cutoff_fs_inv: float = 0.04
    matsubara_terms: int = 2
    hierarchy_depth: int = 4
    system_states: int = 10
    rtol: float = 2.0e-8
    atol: float = 2.0e-10


def drude_matsubara(
    reorganization_hartree: float,
    cutoff_au: float,
    beta_au: float,
    matsubara_terms: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return exponential rates, coefficients, and white-tail correction.

    Uses J(w)=2*lambda*gamma*w/(w^2+gamma^2) and
    C(t)=sum_k c_k exp(-nu_k*t)+2*delta*DiracDelta(t).
    """
    if reorganization_hartree < 0.0:
        raise ValueError("reorganization energy must be nonnegative")
    if cutoff_au <= 0.0 or beta_au <= 0.0 or matsubara_terms < 0:
        raise ValueError("invalid Drude-Matsubara parameters")
    gamma = float(cutoff_au)
    beta = float(beta_au)
    rates = [gamma]
    angle = 0.5 * beta * gamma
    sine = np.sin(angle)
    if abs(sine) < 1.0e-12:
        raise ValueError("Drude pole coincides with a Matsubara singularity")
    coefficients = [reorganization_hartree * gamma * (1.0 / np.tan(angle) - 1.0j)]
    for index in range(1, matsubara_terms + 1):
        nu = 2.0 * np.pi * index / beta
        denominator = nu * nu - gamma * gamma
        if abs(denominator) < 1.0e-14 * max(nu * nu, gamma * gamma):
            raise ValueError("Matsubara pole is degenerate with Drude cutoff")
        coefficient = 4.0 * reorganization_hartree * gamma * nu / (beta * denominator)
        rates.append(nu)
        coefficients.append(complex(coefficient))
    rates_array = np.asarray(rates, dtype=float)
    coefficients_array = np.asarray(coefficients, dtype=complex)
    exact_real_integral = 2.0 * reorganization_hartree / (beta * gamma)
    retained_real_integral = float(np.sum(np.real(coefficients_array) / rates_array))
    tail = exact_real_integral - retained_real_integral
    return rates_array, coefficients_array, tail


def hierarchy_indices(nterms: int, depth: int) -> list[tuple[int, ...]]:
    if nterms < 1 or depth < 0:
        raise ValueError("invalid hierarchy shape")
    indices = [
        entry
        for entry in product(range(depth + 1), repeat=nterms)
        if sum(entry) <= depth
    ]
    indices.sort(key=lambda entry: (sum(entry), entry))
    return indices


class ScaledHEOM:
    """Scaled finite-temperature HEOM for one Drude-Lorentz bath."""

    def __init__(
        self,
        hamiltonian: np.ndarray,
        coupling: np.ndarray,
        config: HEOMConfig,
    ) -> None:
        self.hamiltonian = np.asarray(hamiltonian, dtype=complex)
        self.coupling = np.asarray(coupling, dtype=complex)
        if (
            self.hamiltonian.ndim != 2
            or self.hamiltonian.shape[0] != self.hamiltonian.shape[1]
        ):
            raise ValueError("Hamiltonian must be square")
        if self.coupling.shape != self.hamiltonian.shape:
            raise ValueError("coupling operator shape mismatch")
        if config.temperature_k <= 0.0 or config.cutoff_fs_inv <= 0.0:
            raise ValueError("temperature and cutoff must be positive")
        self.config = config
        beta = 1.0 / (KB_HARTREE_PER_K * config.temperature_k)
        self.rates, self.coefficients, self.tail = drude_matsubara(
            config.reorganization_cminv * CMINV_TO_HARTREE,
            config.cutoff_fs_inv / FS_TO_AU,
            beta,
            config.matsubara_terms,
        )
        self.indices = hierarchy_indices(self.rates.size, config.hierarchy_depth)
        self.index_of = {entry: index for index, entry in enumerate(self.indices)}
        self.nado = len(self.indices)
        self.dimension = self.hamiltonian.shape[0]
        self.decay = np.asarray(
            [np.dot(entry, self.rates) for entry in self.indices], dtype=float
        )
        self.up = np.full((self.nado, self.rates.size), -1, dtype=int)
        self.down = np.full_like(self.up, -1)
        for ado_index, entry in enumerate(self.indices):
            for term in range(self.rates.size):
                raised = list(entry)
                raised[term] += 1
                self.up[ado_index, term] = self.index_of.get(tuple(raised), -1)
                if entry[term] > 0:
                    lowered = list(entry)
                    lowered[term] -= 1
                    self.down[ado_index, term] = self.index_of[tuple(lowered)]

    @staticmethod
    def _commutator(operator: np.ndarray, density: np.ndarray) -> np.ndarray:
        return operator @ density - density @ operator

    def derivative(self, _time: float, flat_state: np.ndarray) -> np.ndarray:
        ado = flat_state.reshape((self.nado, self.dimension, self.dimension))
        result = np.empty_like(ado)
        q = self.coupling
        for ado_index, entry in enumerate(self.indices):
            density = ado[ado_index]
            derivative = -1.0j * self._commutator(self.hamiltonian, density)
            derivative -= self.decay[ado_index] * density
            if self.tail != 0.0:
                derivative -= self.tail * self._commutator(
                    q, self._commutator(q, density)
                )
            for term, occupation in enumerate(entry):
                magnitude = abs(self.coefficients[term])
                upper_index = self.up[ado_index, term]
                if upper_index >= 0 and magnitude > 0.0:
                    factor = np.sqrt((occupation + 1) * magnitude)
                    derivative -= 1.0j * factor * self._commutator(q, ado[upper_index])
                lower_index = self.down[ado_index, term]
                if lower_index >= 0 and magnitude > 0.0:
                    lower = ado[lower_index]
                    coefficient = self.coefficients[term]
                    factor = np.sqrt(occupation / magnitude)
                    derivative -= (
                        1.0j
                        * factor
                        * (
                            coefficient * (q @ lower)
                            - coefficient.conjugate() * (lower @ q)
                        )
                    )
            result[ado_index] = derivative
        return result.reshape(-1)

    def solve(
        self, initial_density: np.ndarray, times_au: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        initial_density = np.asarray(initial_density, dtype=complex)
        times_au = np.asarray(times_au, dtype=float)
        if initial_density.shape != (self.dimension, self.dimension):
            raise ValueError("initial density shape mismatch")
        if times_au.ndim != 1 or times_au.size < 2 or np.any(np.diff(times_au) <= 0.0):
            raise ValueError("times must be a strictly increasing vector")
        initial = np.zeros((self.nado, self.dimension, self.dimension), dtype=complex)
        initial[0] = initial_density
        solution = solve_ivp(
            self.derivative,
            (float(times_au[0]), float(times_au[-1])),
            initial.reshape(-1),
            t_eval=times_au,
            method="DOP853",
            rtol=self.config.rtol,
            atol=self.config.atol,
        )
        if not solution.success:
            raise RuntimeError(f"HEOM integration failed: {solution.message}")
        hierarchy = solution.y.T.reshape(
            (times_au.size, self.nado, self.dimension, self.dimension)
        )
        roots = hierarchy[:, 0]
        ado_norms = np.linalg.norm(hierarchy[:, 1:].reshape(times_au.size, -1), axis=1)
        return roots, ado_norms


def simulate_model(
    model: Model,
    config: HEOMConfig,
    duration_fs: float,
    frames: int,
) -> dict[str, np.ndarray | float | int]:
    if duration_fs <= 0.0 or frames < 2:
        raise ValueError("duration and frames must be positive")
    x, full_hamiltonian = build_hamiltonian(model)
    energies, eigenvectors = np.linalg.eigh(full_hamiltonian)
    nstates = min(config.system_states, energies.size)
    if nstates < 2:
        raise ValueError("at least two system states are required")
    basis = eigenvectors[:, :nstates]
    hamiltonian = np.diag(energies[:nstates] - energies[0])
    coordinate_grid = x / max(float(np.max(np.abs(x))), np.finfo(float).eps)
    coordinate_full = np.diag(np.tile(coordinate_grid, 2))
    coupling = basis.conj().T @ coordinate_full @ basis
    wavefunction = initial_state(model, x)
    projected = basis.conj().T @ wavefunction
    retained_norm = float(np.vdot(projected, projected).real)
    if retained_norm <= 1.0e-14:
        raise ValueError("truncated system basis excludes initial state")
    projected /= np.sqrt(retained_norm)
    initial_density = np.outer(projected, projected.conj())
    solver = ScaledHEOM(hamiltonian, coupling, config)
    times_fs = np.linspace(0.0, duration_fs, frames)
    root_energy, ado_norm = solver.solve(initial_density, times_fs * FS_TO_AU)

    nx = model.nx
    dx = float(x[1] - x[0])
    density1 = np.empty((frames, nx))
    density2 = np.empty((frames, nx))
    pop1 = np.empty(frames)
    pop2 = np.empty(frames)
    trace_error = 0.0
    hermiticity_error = 0.0
    min_eigenvalue = np.inf
    for index, root in enumerate(root_energy):
        trace_error = max(trace_error, abs(np.trace(root) - 1.0))
        hermiticity_error = max(
            hermiticity_error, float(np.max(np.abs(root - root.conj().T)))
        )
        physical_root = 0.5 * (root + root.conj().T)
        min_eigenvalue = min(
            min_eigenvalue, float(np.linalg.eigvalsh(physical_root)[0])
        )
        grid_density = basis @ physical_root @ basis.conj().T
        density1[index] = np.real(np.diag(grid_density)[:nx]) / dx
        density2[index] = np.real(np.diag(grid_density)[nx:]) / dx
        pop1[index] = np.sum(density1[index]) * dx
        pop2[index] = np.sum(density2[index]) * dx

    positive_gaps = np.diff(energies[:nstates])
    reference_gap = float(np.median(positive_gaps[positive_gaps > 0.0]))
    coupling_ratio = (
        config.reorganization_cminv * CMINV_TO_HARTREE / reference_gap
        if reference_gap > 0.0
        else np.inf
    )
    return {
        "x_angstrom": x / ANGSTROM_TO_BOHR,
        "time_fs": times_fs,
        "energy_cminv": energies[:nstates] / CMINV_TO_HARTREE,
        "rho_energy": root_energy,
        "density1_angstrom_inv": density1 * ANGSTROM_TO_BOHR,
        "density2_angstrom_inv": density2 * ANGSTROM_TO_BOHR,
        "pop1": pop1,
        "pop2": pop2,
        "ado_norm": ado_norm,
        "bath_rates_fs_inv": solver.rates * FS_TO_AU,
        "bath_coefficients_au2": solver.coefficients,
        "tail_correction_au": solver.tail,
        "ado_count": solver.nado,
        "retained_initial_norm": retained_norm,
        "max_trace_error": float(trace_error),
        "max_hermiticity_error": float(hermiticity_error),
        "min_density_eigenvalue": float(min_eigenvalue),
        "max_auxiliary_norm": float(np.max(ado_norm)),
        "reorganization_to_gap_ratio": float(coupling_ratio),
    }


def write_output(
    path: Path, result: dict[str, np.ndarray | float | int], config: HEOMConfig
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["method"] = "scaled Drude-Lorentz HEOM with Matsubara expansion"
        handle.attrs["temperature_K"] = config.temperature_k
        handle.attrs["reorganization_cm-1"] = config.reorganization_cminv
        handle.attrs["cutoff_fs-1"] = config.cutoff_fs_inv
        handle.attrs["matsubara_terms"] = config.matsubara_terms
        handle.attrs["hierarchy_depth"] = config.hierarchy_depth
        handle.attrs["system_states"] = config.system_states
        for name, value in result.items():
            if name == "rho_energy":
                array = np.asarray(value)
                handle.create_dataset("rho_energy_real", data=array.real)
                handle.create_dataset("rho_energy_imag", data=array.imag)
            elif name == "bath_coefficients_au2":
                array = np.asarray(value)
                handle.create_dataset("bath_coefficients_real_au2", data=array.real)
                handle.create_dataset("bath_coefficients_imag_au2", data=array.imag)
            elif np.ndim(value) == 0:
                handle.attrs[name] = value
            else:
                handle.create_dataset(name, data=value)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", default="heom_nonmarkovian.h5")
    result.add_argument("--nx", type=int, default=12)
    result.add_argument("--xmin", type=float, default=-3.0)
    result.add_argument("--xmax", type=float, default=3.0)
    result.add_argument("--mass", type=float, default=1.00784)
    result.add_argument("--coupling", type=float, default=150.0, help="cm^-1")
    result.add_argument("--temperature", type=float, default=300.0, help="K")
    result.add_argument("--reorganization", type=float, default=500.0, help="cm^-1")
    result.add_argument("--cutoff", type=float, default=0.04, help="fs^-1")
    result.add_argument("--matsubara", type=int, default=2)
    result.add_argument("--depth", type=int, default=4)
    result.add_argument("--states", type=int, default=10)
    result.add_argument("--duration", type=float, default=100.0, help="fs")
    result.add_argument("--frames", type=int, default=101)
    result.add_argument("--rtol", type=float, default=2.0e-8)
    result.add_argument("--atol", type=float, default=2.0e-10)
    return result


def main() -> None:
    args = parser().parse_args()
    model = Model(
        nx=args.nx,
        xmin_angstrom=args.xmin,
        xmax_angstrom=args.xmax,
        mass_amu=args.mass,
        coupling_cminv=args.coupling,
    )
    config = HEOMConfig(
        temperature_k=args.temperature,
        reorganization_cminv=args.reorganization,
        cutoff_fs_inv=args.cutoff,
        matsubara_terms=args.matsubara,
        hierarchy_depth=args.depth,
        system_states=args.states,
        rtol=args.rtol,
        atol=args.atol,
    )
    result = simulate_model(model, config, args.duration, args.frames)
    write_output(Path(args.output), result, config)
    print("Strong-coupling non-Markovian HEOM complete")
    for name in (
        "ado_count",
        "retained_initial_norm",
        "max_trace_error",
        "max_hermiticity_error",
        "min_density_eigenvalue",
        "max_auxiliary_norm",
        "reorganization_to_gap_ratio",
    ):
        print(f"{name}: {result[name]}")


if __name__ == "__main__":
    main()
