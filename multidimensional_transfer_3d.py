#!/usr/bin/env python3
"""Three-coordinate, two-state FFT split-operator H-atom transfer solver."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from potential_data import DiabaticPES3D
from quantum_fdt_density import (
    AMU_TO_ELECTRON_MASS,
    ANGSTROM_TO_BOHR,
    CMINV_TO_HARTREE,
    FS_TO_AU,
)


@dataclass(frozen=True)
class Model3D:
    nx: int = 48
    ny: int = 20
    nz: int = 16
    xmin: float = -4.0
    xmax: float = 4.0
    ymin: float = -1.8
    ymax: float = 1.8
    zmin: float = -1.5
    zmax: float = 1.5
    mass_x_amu: float = 1.00784
    mass_y_amu: float = 12.0
    mass_z_amu: float = 16.0
    kx_cminv_a2: float = 1800.0
    ky_cminv_a2: float = 700.0
    kz_cminv_a2: float = 500.0
    x1: float = -1.0
    x2: float = 1.0
    y1: float = -0.25
    y2: float = 0.25
    z1: float = -0.15
    z2: float = 0.15
    xy_path_coupling: float = 0.20
    xz_path_coupling: float = -0.15
    bias_cminv: float = -300.0
    diabatic_coupling_cminv: float = 150.0
    coupling_sigma_x: float = 0.45
    coupling_sigma_y: float = 0.75
    coupling_sigma_z: float = 0.70
    x0: float = -1.0
    y0: float = -0.25
    z0: float = -0.15
    sigma_x: float = 0.30
    sigma_y: float = 0.35
    sigma_z: float = 0.35


class SplitOperator3D:
    def __init__(self, model: Model3D, pes: DiabaticPES3D | None = None):
        self.model = model
        if min(model.nx, model.ny, model.nz) < 4:
            raise ValueError("nx, ny, and nz must be at least four")
        if not (
            model.xmin < model.xmax
            and model.ymin < model.ymax
            and model.zmin < model.zmax
        ):
            raise ValueError("coordinate bounds must be strictly increasing")
        if min(model.mass_x_amu, model.mass_y_amu, model.mass_z_amu) <= 0.0:
            raise ValueError("coordinate masses must be positive")
        self.dx = (model.xmax - model.xmin) * ANGSTROM_TO_BOHR / model.nx
        self.dy = (model.ymax - model.ymin) * ANGSTROM_TO_BOHR / model.ny
        self.dz = (model.zmax - model.zmin) * ANGSTROM_TO_BOHR / model.nz
        self.x = model.xmin * ANGSTROM_TO_BOHR + (np.arange(model.nx) + 0.5) * self.dx
        self.y = model.ymin * ANGSTROM_TO_BOHR + (np.arange(model.ny) + 0.5) * self.dy
        self.z = model.zmin * ANGSTROM_TO_BOHR + (np.arange(model.nz) + 0.5) * self.dz
        self.zz, self.yy, self.xx = np.meshgrid(self.z, self.y, self.x, indexing="ij")
        kx = 2.0 * np.pi * np.fft.fftfreq(model.nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(model.ny, d=self.dy)
        kz = 2.0 * np.pi * np.fft.fftfreq(model.nz, d=self.dz)
        kzz, kyy, kxx = np.meshgrid(kz, ky, kx, indexing="ij")
        self.kinetic = (
            kxx**2 / (2.0 * model.mass_x_amu * AMU_TO_ELECTRON_MASS)
            + kyy**2 / (2.0 * model.mass_y_amu * AMU_TO_ELECTRON_MASS)
            + kzz**2 / (2.0 * model.mass_z_amu * AMU_TO_ELECTRON_MASS)
        )
        self.v11, self.v22, self.v12 = (
            self._analytic_potentials()
            if pes is None
            else pes.interpolate_atomic(self.x, self.y, self.z)
        )
        self.psi1, self.psi2 = self._initial_state()

    def _analytic_potentials(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = self.model
        conversion = ANGSTROM_TO_BOHR
        kx = model.kx_cminv_a2 * CMINV_TO_HARTREE / conversion**2
        ky = model.ky_cminv_a2 * CMINV_TO_HARTREE / conversion**2
        kz = model.kz_cminv_a2 * CMINV_TO_HARTREE / conversion**2
        x1, x2, y1, y2, z1, z2 = (
            value * conversion
            for value in (model.x1, model.x2, model.y1, model.y2, model.z1, model.z2)
        )
        qy1 = self.yy - y1 - model.xy_path_coupling * (self.xx - x1)
        qy2 = self.yy - y2 + model.xy_path_coupling * (self.xx - x2)
        qz1 = self.zz - z1 - model.xz_path_coupling * (self.xx - x1)
        qz2 = self.zz - z2 + model.xz_path_coupling * (self.xx - x2)
        v11 = 0.5 * kx * (self.xx - x1) ** 2 + 0.5 * ky * qy1**2 + 0.5 * kz * qz1**2
        v22 = (
            0.5 * kx * (self.xx - x2) ** 2
            + 0.5 * ky * qy2**2
            + 0.5 * kz * qz2**2
            + model.bias_cminv * CMINV_TO_HARTREE
        )
        sx = model.coupling_sigma_x * conversion
        sy = model.coupling_sigma_y * conversion
        sz = model.coupling_sigma_z * conversion
        v12 = (
            model.diabatic_coupling_cminv
            * CMINV_TO_HARTREE
            * np.exp(
                -0.5 * (self.xx / sx) ** 2
                - 0.5 * (self.yy / sy) ** 2
                - 0.5 * (self.zz / sz) ** 2
            )
        )
        return v11, v22, v12.astype(complex)

    def _initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        model = self.model
        conversion = ANGSTROM_TO_BOHR
        psi1 = np.exp(
            -0.5
            * ((self.xx - model.x0 * conversion) / (model.sigma_x * conversion)) ** 2
            - 0.5
            * ((self.yy - model.y0 * conversion) / (model.sigma_y * conversion)) ** 2
            - 0.5
            * ((self.zz - model.z0 * conversion) / (model.sigma_z * conversion)) ** 2
        ).astype(complex)
        psi1 /= np.sqrt(np.sum(abs(psi1) ** 2) * self.volume_element)
        return psi1, np.zeros_like(psi1)

    @property
    def volume_element(self) -> float:
        return self.dx * self.dy * self.dz

    def _potential_half_step(self, dt_au: float) -> None:
        trace = 0.5 * (self.v11 + self.v22)
        diagonal = 0.5 * (self.v11 - self.v22)
        frequency = np.sqrt(diagonal**2 + abs(self.v12) ** 2)
        angle = 0.5 * dt_au * frequency
        sine = np.empty_like(frequency)
        small = frequency < 1.0e-14
        sine[small] = 0.5 * dt_au
        sine[~small] = np.sin(angle[~small]) / frequency[~small]
        cosine = np.cos(angle)
        phase = np.exp(-0.5j * dt_au * trace)
        state1 = self.psi1.copy()
        state2 = self.psi2.copy()
        self.psi1 = phase * (
            (cosine - 1.0j * sine * diagonal) * state1 - 1.0j * sine * self.v12 * state2
        )
        self.psi2 = phase * (
            -1.0j * sine * self.v12.conj() * state1
            + (cosine + 1.0j * sine * diagonal) * state2
        )

    def step(self, dt_fs: float) -> None:
        if dt_fs <= 0.0:
            raise ValueError("time step must be positive")
        dt_au = dt_fs * FS_TO_AU
        self._potential_half_step(dt_au)
        kinetic_phase = np.exp(-1.0j * dt_au * self.kinetic)
        self.psi1 = np.fft.ifftn(np.fft.fftn(self.psi1) * kinetic_phase)
        self.psi2 = np.fft.ifftn(np.fft.fftn(self.psi2) * kinetic_phase)
        self._potential_half_step(dt_au)

    def populations(self) -> tuple[float, float]:
        return (
            float(np.sum(abs(self.psi1) ** 2) * self.volume_element),
            float(np.sum(abs(self.psi2) ** 2) * self.volume_element),
        )


def run(args: argparse.Namespace) -> dict[str, float]:
    model = Model3D(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        zmin=args.zmin,
        zmax=args.zmax,
        mass_x_amu=args.mass_x,
        mass_y_amu=args.mass_y,
        mass_z_amu=args.mass_z,
    )
    pes = DiabaticPES3D.from_hdf5(args.pes) if args.pes else None
    solver = SplitOperator3D(model, pes)
    if args.steps < 0 or args.save_every <= 0 or args.dt <= 0.0:
        raise ValueError("invalid propagation controls")
    saved_steps = list(range(0, args.steps + 1, args.save_every))
    if saved_steps[-1] != args.steps:
        saved_steps.append(args.steps)
    populations = np.empty((len(saved_steps), 2))
    populations[0] = solver.populations()
    frame = 1
    for step in range(1, args.steps + 1):
        solver.step(args.dt)
        if step in saved_steps[1:]:
            populations[frame] = solver.populations()
            frame += 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.attrs["method"] = "3D two-state FFT split operator"
        handle.attrs["potential_source"] = str(args.pes or "analytic")
        handle.attrs["dt_fs"] = args.dt
        handle.create_dataset("x_angstrom", data=solver.x / ANGSTROM_TO_BOHR)
        handle.create_dataset("y_angstrom", data=solver.y / ANGSTROM_TO_BOHR)
        handle.create_dataset("z_angstrom", data=solver.z / ANGSTROM_TO_BOHR)
        handle.create_dataset("time_fs", data=np.asarray(saved_steps) * args.dt)
        handle.create_dataset("pop1", data=populations[:, 0])
        handle.create_dataset("pop2", data=populations[:, 1])
        handle.create_dataset("V11_cm-1", data=solver.v11 / CMINV_TO_HARTREE)
        handle.create_dataset("V22_cm-1", data=solver.v22 / CMINV_TO_HARTREE)
        handle.create_dataset("V12_real_cm-1", data=solver.v12.real / CMINV_TO_HARTREE)
        handle.create_dataset("V12_imag_cm-1", data=solver.v12.imag / CMINV_TO_HARTREE)
    norm = populations.sum(axis=1)
    summary = {
        "max_norm_drift": float(np.max(abs(norm - norm[0]))),
        "final_product_population": float(populations[-1, 1]),
    }
    print("3D transfer solver complete")
    for key, value in summary.items():
        print(f"{key}: {value:.10g}")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", default="transfer_3d.h5")
    result.add_argument("--pes", help="unit-annotated 3D diabatic PES HDF5")
    result.add_argument("--nx", type=int, default=48)
    result.add_argument("--ny", type=int, default=20)
    result.add_argument("--nz", type=int, default=16)
    result.add_argument("--xmin", type=float, default=-4.0, help="angstrom")
    result.add_argument("--xmax", type=float, default=4.0, help="angstrom")
    result.add_argument("--ymin", type=float, default=-1.8, help="angstrom")
    result.add_argument("--ymax", type=float, default=1.8, help="angstrom")
    result.add_argument("--zmin", type=float, default=-1.5, help="angstrom")
    result.add_argument("--zmax", type=float, default=1.5, help="angstrom")
    result.add_argument("--mass-x", type=float, default=1.00784, help="amu")
    result.add_argument("--mass-y", type=float, default=12.0, help="amu")
    result.add_argument("--mass-z", type=float, default=16.0, help="amu")
    result.add_argument("--dt", type=float, default=0.01, help="fs")
    result.add_argument("--steps", type=int, default=500)
    result.add_argument("--save-every", type=int, default=10)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
