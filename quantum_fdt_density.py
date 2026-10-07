#!/usr/bin/env python3
"""Finite-grid density-matrix H-atom transfer with quantum detailed balance.

Uses a completely positive secular Davies generator. Drude-Ohmic upward and
downward rates obey the KMS relation at every resolved Bohr frequency. This is
the controlled weak-coupling/Markovian quantum-FDT solver; it does not pretend
that classical stochastic trajectories reproduce a quantum bath.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import expm


ANGSTROM_TO_BOHR = 1.8897261254578281
FS_TO_AU = 41.341374575751
CMINV_TO_HARTREE = 1.0 / 219474.6313705
AMU_TO_ELECTRON_MASS = 1822.888486209
KB_HARTREE_PER_K = 3.166811563e-6


@dataclass(frozen=True)
class Model:
    nx: int = 24
    xmin_angstrom: float = -3.0
    xmax_angstrom: float = 3.0
    mass_amu: float = 1.00784
    k1_cminv_angstrom2: float = 1500.0
    k2_cminv_angstrom2: float = 1500.0
    x1_angstrom: float = -1.0
    x2_angstrom: float = 1.0
    v1_shift_cminv: float = 0.0
    v2_shift_cminv: float = -200.0
    coupling_cminv: float = 150.0
    coupling_width_angstrom: float = 0.4
    x0_angstrom: float = -1.0
    p0_angstrom_inv: float = 0.0
    sigma0_angstrom: float = 0.3


def build_hamiltonian(model: Model) -> tuple[np.ndarray, np.ndarray]:
    if model.nx < 4 or model.xmax_angstrom <= model.xmin_angstrom:
        raise ValueError("invalid grid")
    xmin = model.xmin_angstrom * ANGSTROM_TO_BOHR
    xmax = model.xmax_angstrom * ANGSTROM_TO_BOHR
    dx = (xmax - xmin) / model.nx
    x = xmin + (np.arange(model.nx) + 0.5) * dx
    mass = model.mass_amu * AMU_TO_ELECTRON_MASS

    kinetic = np.eye(model.nx) / (mass * dx * dx)
    off = -0.5 / (mass * dx * dx)
    for i in range(model.nx):
        kinetic[i, (i - 1) % model.nx] = off
        kinetic[i, (i + 1) % model.nx] = off

    a2b = ANGSTROM_TO_BOHR
    k1 = model.k1_cminv_angstrom2 * CMINV_TO_HARTREE / a2b**2
    k2 = model.k2_cminv_angstrom2 * CMINV_TO_HARTREE / a2b**2
    x1 = model.x1_angstrom * a2b
    x2 = model.x2_angstrom * a2b
    width = model.coupling_width_angstrom * a2b
    v11 = 0.5 * k1 * (x - x1) ** 2 + model.v1_shift_cminv * CMINV_TO_HARTREE
    v22 = 0.5 * k2 * (x - x2) ** 2 + model.v2_shift_cminv * CMINV_TO_HARTREE
    v12 = model.coupling_cminv * CMINV_TO_HARTREE * np.exp(
        -0.5 * ((x - 0.5 * (x1 + x2)) / width) ** 2
    )
    zero = np.zeros_like(kinetic)
    hamiltonian = np.block(
        [[kinetic + np.diag(v11), zero + np.diag(v12)],
         [zero + np.diag(v12), kinetic + np.diag(v22)]]
    )
    return x, hamiltonian


def initial_state(model: Model, x_bohr: np.ndarray) -> np.ndarray:
    x0 = model.x0_angstrom * ANGSTROM_TO_BOHR
    sigma = model.sigma0_angstrom * ANGSTROM_TO_BOHR
    p0 = model.p0_angstrom_inv / ANGSTROM_TO_BOHR
    psi1 = np.exp(-0.5 * ((x_bohr - x0) / sigma) ** 2 + 1j * p0 * x_bohr)
    dx = float(x_bohr[1] - x_bohr[0])
    psi1 /= np.sqrt(np.vdot(psi1, psi1).real * dx)
    # Euclidean coefficients make trace(rho)=1; coordinate densities recover /dx.
    state = np.concatenate((psi1 * np.sqrt(dx), np.zeros_like(psi1)))
    return state


def davies_rate_matrix(
    energies: np.ndarray,
    coupling_energy_basis: np.ndarray,
    temperature_k: float,
    gamma_fs_inv: float,
    cutoff_fs_inv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return population generator, escape rates, and pure-dephasing rates."""
    if temperature_k <= 0.0 or gamma_fs_inv < 0.0 or cutoff_fs_inv <= 0.0:
        raise ValueError("invalid quantum-bath parameters")
    beta = 1.0 / (KB_HARTREE_PER_K * temperature_k)
    gamma = gamma_fs_inv / FS_TO_AU
    cutoff = cutoff_fs_inv / FS_TO_AU
    nstate = energies.size
    generator = np.zeros((nstate, nstate), dtype=float)

    for low in range(nstate - 1):
        for high in range(low + 1, nstate):
            omega = float(energies[high] - energies[low])
            if omega <= 64.0 * np.finfo(float).eps:
                continue
            bose = 0.0 if beta * omega > 700.0 else 1.0 / np.expm1(beta * omega)
            drude_ohmic = (omega / cutoff) / (1.0 + (omega / cutoff) ** 2)
            strength = gamma * drude_ohmic * abs(coupling_energy_basis[low, high]) ** 2
            down = strength * (bose + 1.0)
            up = strength * bose
            generator[low, high] += down
            generator[high, low] += up

    for source in range(nstate):
        generator[source, source] = -np.sum(generator[:, source])
    escape = -np.diag(generator).copy()

    # Zero-frequency limit of symmetrized Drude-Ohmic quantum noise.
    diagonal = np.real(np.diag(coupling_energy_basis))
    zero_noise = 2.0 * gamma / (beta * cutoff)
    dephasing = 0.5 * zero_noise * (diagonal[:, None] - diagonal[None, :]) ** 2
    return generator, escape, dephasing


def propagate_density(
    energies: np.ndarray,
    eigenvectors: np.ndarray,
    initial: np.ndarray,
    generator: np.ndarray,
    escape: np.ndarray,
    dephasing: np.ndarray,
    times_au: np.ndarray,
) -> np.ndarray:
    rho0 = np.outer(initial, initial.conj())
    rho0_e = eigenvectors.conj().T @ rho0 @ eigenvectors
    populations0 = np.real(np.diag(rho0_e))
    delta = energies[:, None] - energies[None, :]
    damping = 0.5 * (escape[:, None] + escape[None, :]) + dephasing
    result = np.empty((times_au.size, energies.size, energies.size), dtype=complex)
    for index, time in enumerate(times_au):
        populations = expm(generator * time) @ populations0
        rho_e = rho0_e * np.exp((-1j * delta - damping) * time)
        np.fill_diagonal(rho_e, populations)
        result[index] = rho_e
    return result


def run(args: argparse.Namespace) -> dict[str, float]:
    model = Model(
        nx=args.nx,
        xmin_angstrom=args.xmin,
        xmax_angstrom=args.xmax,
        mass_amu=args.mass,
        coupling_cminv=args.coupling,
    )
    x, hamiltonian = build_hamiltonian(model)
    initial = initial_state(model, x)
    energies, eigenvectors = np.linalg.eigh(hamiltonian)
    x_center = 0.5 * (x[0] + x[-1])
    x_scale = max(np.ptp(x), np.finfo(float).eps)
    coordinate = np.diag(np.tile((x - x_center) / x_scale, 2))
    coupling_e = eigenvectors.conj().T @ coordinate @ eigenvectors
    generator, escape, dephasing = davies_rate_matrix(
        energies, coupling_e, args.temperature, args.gamma, args.cutoff
    )
    times_fs = np.linspace(0.0, args.duration, args.frames)
    rho_energy = propagate_density(
        energies, eigenvectors, initial, generator, escape, dephasing,
        times_fs * FS_TO_AU,
    )

    nx = model.nx
    dx = float(x[1] - x[0])
    pop1 = np.empty(args.frames)
    pop2 = np.empty(args.frames)
    density1 = np.empty((args.frames, nx))
    density2 = np.empty((args.frames, nx))
    min_eigenvalue = np.inf
    max_trace_error = 0.0
    for index, rho_e in enumerate(rho_energy):
        rho = eigenvectors @ rho_e @ eigenvectors.conj().T
        rho = 0.5 * (rho + rho.conj().T)
        trace = np.trace(rho).real
        max_trace_error = max(max_trace_error, abs(trace - 1.0))
        min_eigenvalue = min(min_eigenvalue, float(np.linalg.eigvalsh(rho)[0]))
        density1[index] = np.real(np.diag(rho)[:nx]) / dx
        density2[index] = np.real(np.diag(rho)[nx:]) / dx
        pop1[index] = np.sum(density1[index]) * dx
        pop2[index] = np.sum(density2[index]) * dx

    beta = 1.0 / (KB_HARTREE_PER_K * args.temperature)
    gibbs = np.exp(-beta * (energies - energies[0]))
    gibbs /= gibbs.sum()
    kms_residual = float(np.max(np.abs(generator @ gibbs)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.attrs["method"] = "secular Davies quantum detailed-balance master equation"
        handle.attrs["temperature_K"] = args.temperature
        handle.attrs["gamma_fs-1"] = args.gamma
        handle.attrs["cutoff_fs-1"] = args.cutoff
        handle.attrs["kms_stationarity_residual_au-1"] = kms_residual
        handle.create_dataset("x_angstrom", data=x / ANGSTROM_TO_BOHR)
        handle.create_dataset("time_fs", data=times_fs)
        handle.create_dataset("energy_cminv", data=energies / CMINV_TO_HARTREE)
        handle.create_dataset("rho_energy_real", data=rho_energy.real)
        handle.create_dataset("rho_energy_imag", data=rho_energy.imag)
        handle.create_dataset("density1_angstrom-1", data=density1 * ANGSTROM_TO_BOHR)
        handle.create_dataset("density2_angstrom-1", data=density2 * ANGSTROM_TO_BOHR)
        handle.create_dataset("pop1", data=pop1)
        handle.create_dataset("pop2", data=pop2)

    summary = {
        "max_trace_error": max_trace_error,
        "min_density_eigenvalue": min_eigenvalue,
        "kms_residual_au-1": kms_residual,
        "final_product_population": float(pop2[-1]),
    }
    print("Quantum-FDT density solver complete")
    for key, value in summary.items():
        print(f"{key}: {value:.10g}")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", default="quantum_density.h5")
    result.add_argument("--nx", type=int, default=24)
    result.add_argument("--xmin", type=float, default=-3.0)
    result.add_argument("--xmax", type=float, default=3.0)
    result.add_argument("--mass", type=float, default=1.00784)
    result.add_argument("--coupling", type=float, default=150.0, help="cm^-1")
    result.add_argument("--temperature", type=float, default=100.0, help="K")
    result.add_argument("--gamma", type=float, default=0.02, help="fs^-1")
    result.add_argument("--cutoff", type=float, default=0.2, help="fs^-1")
    result.add_argument("--duration", type=float, default=200.0, help="fs")
    result.add_argument("--frames", type=int, default=101)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
