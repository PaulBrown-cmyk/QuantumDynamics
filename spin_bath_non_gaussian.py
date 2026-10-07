#!/usr/bin/env python3
"""Exact finite non-Gaussian spin-bath dynamics for reactant/product states."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from quantum_fdt_density import CMINV_TO_HARTREE, FS_TO_AU, KB_HARTREE_PER_K

IDENTITY2 = np.eye(2, dtype=complex)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Z = np.diag([1.0, -1.0]).astype(complex)


@dataclass(frozen=True)
class SpinBathConfig:
    system_bias_cminv: float = 250.0
    system_coupling_cminv: float = 120.0
    bath_frequencies_cminv: tuple[float, ...] = (90.0, 130.0, 180.0, 240.0)
    bath_couplings_cminv: tuple[float, ...] = (45.0, 38.0, 32.0, 28.0)
    temperature_k: float = 300.0


def embedded_operator(operator: np.ndarray, site: int, sites: int) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]])
    for index in range(sites):
        result = np.kron(result, operator if index == site else IDENTITY2)
    return result


class ExactSpinBath:
    """Exact diagonalization of a two-state system coupled to bath spins.

    Hamiltonian is ``Hs + Hb + sigma_z sum_j g_j sigma_x,j``. Finite spins
    generate non-Gaussian bath statistics and retain exact finite-size memory.
    """

    def __init__(self, config: SpinBathConfig):
        if config.temperature_k <= 0.0:
            raise ValueError("temperature must be positive")
        frequencies = np.asarray(config.bath_frequencies_cminv, dtype=float)
        couplings = np.asarray(config.bath_couplings_cminv, dtype=float)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("at least one bath spin is required")
        if frequencies.shape != couplings.shape:
            raise ValueError("bath frequency and coupling counts differ")
        if np.any(frequencies <= 0.0) or np.any(couplings < 0.0):
            raise ValueError(
                "bath frequencies must be positive and couplings nonnegative"
            )
        if frequencies.size > 10:
            raise ValueError("exact spin bath is limited to ten bath spins")
        self.config = config
        self.nbath = frequencies.size
        self.bath_dimension = 2**self.nbath
        bias = config.system_bias_cminv * CMINV_TO_HARTREE
        tunneling = config.system_coupling_cminv * CMINV_TO_HARTREE
        self.system_hamiltonian = 0.5 * bias * SIGMA_Z + tunneling * SIGMA_X
        bath_hamiltonian = np.zeros(
            (self.bath_dimension, self.bath_dimension), dtype=complex
        )
        bath_coupling = np.zeros_like(bath_hamiltonian)
        for site, (frequency, coupling) in enumerate(
            zip(frequencies, couplings, strict=True)
        ):
            bath_hamiltonian += (
                0.5
                * frequency
                * CMINV_TO_HARTREE
                * embedded_operator(SIGMA_Z, site, self.nbath)
            )
            bath_coupling += (
                coupling
                * CMINV_TO_HARTREE
                * embedded_operator(SIGMA_X, site, self.nbath)
            )
        self.bath_hamiltonian = bath_hamiltonian
        self.hamiltonian = (
            np.kron(self.system_hamiltonian, np.eye(self.bath_dimension))
            + np.kron(IDENTITY2, bath_hamiltonian)
            + np.kron(SIGMA_Z, bath_coupling)
        )
        self.energies, self.eigenvectors = np.linalg.eigh(self.hamiltonian)
        bath_energies, bath_vectors = np.linalg.eigh(bath_hamiltonian)
        beta = 1.0 / (KB_HARTREE_PER_K * config.temperature_k)
        weights = np.exp(-beta * (bath_energies - bath_energies.min()))
        weights /= weights.sum()
        self.bath_density = (bath_vectors * weights) @ bath_vectors.conj().T

    def evolve(
        self, initial_system_density: np.ndarray, times_fs: np.ndarray
    ) -> np.ndarray:
        density = np.asarray(initial_system_density, dtype=complex)
        times = np.asarray(times_fs, dtype=float)
        if density.shape != (2, 2):
            raise ValueError("initial system density must be 2x2")
        if not np.allclose(density, density.conj().T, atol=1.0e-12):
            raise ValueError("initial system density must be Hermitian")
        if not np.isclose(np.trace(density), 1.0, atol=1.0e-12):
            raise ValueError("initial system density must have unit trace")
        if times.ndim != 1 or times.size < 1 or np.any(times < 0.0):
            raise ValueError("times must be a nonnegative vector")
        total_initial = np.kron(density, self.bath_density)
        energy_density = self.eigenvectors.conj().T @ total_initial @ self.eigenvectors
        reduced = np.empty((times.size, 2, 2), dtype=complex)
        for index, time_fs in enumerate(times):
            phase = np.exp(-1.0j * self.energies * time_fs * FS_TO_AU)
            total = (
                self.eigenvectors
                @ (phase[:, None] * energy_density * phase.conj()[None, :])
                @ self.eigenvectors.conj().T
            )
            tensor = total.reshape((2, self.bath_dimension, 2, self.bath_dimension))
            reduced[index] = np.einsum("abcb->ac", tensor)
        return reduced


def trace_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = np.asarray(first) - np.asarray(second)
    return np.asarray(
        [
            0.5 * np.sum(abs(np.linalg.eigvalsh(0.5 * (item + item.conj().T))))
            for item in difference
        ]
    )


def run(args: argparse.Namespace) -> dict[str, float]:
    if args.spins < 1 or args.duration < 0.0 or args.frames < 1:
        raise ValueError(
            "spins and frames must be positive; duration must be nonnegative"
        )
    frequency = tuple(np.linspace(args.frequency_min, args.frequency_max, args.spins))
    coupling = tuple(args.bath_coupling / np.sqrt(args.spins) * np.ones(args.spins))
    config = SpinBathConfig(
        system_bias_cminv=args.bias,
        system_coupling_cminv=args.tunneling,
        bath_frequencies_cminv=frequency,
        bath_couplings_cminv=coupling,
        temperature_k=args.temperature,
    )
    solver = ExactSpinBath(config)
    times = np.linspace(0.0, args.duration, args.frames)
    reactant = np.diag([1.0, 0.0]).astype(complex)
    product = np.diag([0.0, 1.0]).astype(complex)
    density = solver.evolve(reactant, times)
    alternative = solver.evolve(product, times)
    distance = trace_distance(density, alternative)
    increments = np.diff(distance)
    backflow = float(np.sum(increments[increments > 0.0]))
    trace_error = float(np.max(abs(np.trace(density, axis1=1, axis2=2) - 1.0)))
    min_eigenvalue = float(min(np.linalg.eigvalsh(item)[0] for item in density))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.attrs["method"] = "exact finite non-Gaussian spin bath"
        handle.attrs["temperature_K"] = config.temperature_k
        handle.attrs["bath_spins"] = solver.nbath
        handle.attrs["max_trace_error"] = trace_error
        handle.attrs["min_density_eigenvalue"] = min_eigenvalue
        handle.attrs["information_backflow"] = backflow
        handle.create_dataset("time_fs", data=times)
        handle.create_dataset("population_reactant", data=density[:, 0, 0].real)
        handle.create_dataset("population_product", data=density[:, 1, 1].real)
        handle.create_dataset("trace_distance", data=distance)
        handle.create_dataset("rho_real", data=density.real)
        handle.create_dataset("rho_imag", data=density.imag)
        handle.create_dataset("bath_frequency_cm-1", data=frequency)
        handle.create_dataset("bath_coupling_cm-1", data=coupling)
    summary = {
        "max_trace_error": trace_error,
        "min_density_eigenvalue": min_eigenvalue,
        "information_backflow": backflow,
        "final_product_population": float(density[-1, 1, 1].real),
    }
    print("Exact non-Gaussian spin-bath dynamics complete")
    for key, value in summary.items():
        print(f"{key}: {value:.10g}")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", default="spin_bath.h5")
    result.add_argument("--spins", type=int, default=5)
    result.add_argument("--bias", type=float, default=250.0, help="cm^-1")
    result.add_argument("--tunneling", type=float, default=120.0, help="cm^-1")
    result.add_argument("--frequency-min", type=float, default=80.0, help="cm^-1")
    result.add_argument("--frequency-max", type=float, default=260.0, help="cm^-1")
    result.add_argument("--bath-coupling", type=float, default=80.0, help="cm^-1 total")
    result.add_argument("--temperature", type=float, default=300.0, help="K")
    result.add_argument("--duration", type=float, default=500.0, help="fs")
    result.add_argument("--frames", type=int, default=501)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
