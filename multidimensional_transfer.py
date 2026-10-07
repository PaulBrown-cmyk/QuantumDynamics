#!/usr/bin/env python3
"""Two-coordinate, two-diabatic-state split-operator transfer solver."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from quantum_fdt_density import (
    AMU_TO_ELECTRON_MASS,
    ANGSTROM_TO_BOHR,
    CMINV_TO_HARTREE,
    FS_TO_AU,
)


@dataclass(frozen=True)
class Model2D:
    nx: int = 96
    ny: int = 48
    xmin: float = -4.0
    xmax: float = 4.0
    ymin: float = -2.0
    ymax: float = 2.0
    mass_x_amu: float = 1.00784
    mass_y_amu: float = 12.0
    kx_cminv_a2: float = 1800.0
    ky_cminv_a2: float = 700.0
    x1: float = -1.0
    x2: float = 1.0
    y1: float = -0.25
    y2: float = 0.25
    path_coupling: float = 0.25
    bias_cminv: float = -300.0
    diabatic_coupling_cminv: float = 150.0
    coupling_sigma_x: float = 0.4
    coupling_sigma_y: float = 0.7
    x0: float = -1.0
    y0: float = -0.25
    px0_a_inv: float = 0.0
    py0_a_inv: float = 0.0
    sigma_x: float = 0.3
    sigma_y: float = 0.35


class SplitOperator2D:
    def __init__(self, model: Model2D):
        self.model = model
        if min(model.nx, model.ny) < 4:
            raise ValueError("nx and ny must be at least four")
        self.dx = (model.xmax - model.xmin) * ANGSTROM_TO_BOHR / model.nx
        self.dy = (model.ymax - model.ymin) * ANGSTROM_TO_BOHR / model.ny
        self.x = model.xmin * ANGSTROM_TO_BOHR + (np.arange(model.nx) + 0.5) * self.dx
        self.y = model.ymin * ANGSTROM_TO_BOHR + (np.arange(model.ny) + 0.5) * self.dy
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.kx = 2.0 * np.pi * np.fft.fftfreq(model.nx, d=self.dx)
        self.ky = 2.0 * np.pi * np.fft.fftfreq(model.ny, d=self.dy)
        kxx, kyy = np.meshgrid(self.kx, self.ky)
        mx = model.mass_x_amu * AMU_TO_ELECTRON_MASS
        my = model.mass_y_amu * AMU_TO_ELECTRON_MASS
        self.kinetic = kxx**2 / (2.0 * mx) + kyy**2 / (2.0 * my)
        self.v11, self.v22, self.v12 = self._potentials()
        self.psi1, self.psi2 = self._initial_state()

    def _potentials(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = self.model
        a2b = ANGSTROM_TO_BOHR
        kx = m.kx_cminv_a2 * CMINV_TO_HARTREE / a2b**2
        ky = m.ky_cminv_a2 * CMINV_TO_HARTREE / a2b**2
        x1, x2, y1, y2 = (value * a2b for value in (m.x1, m.x2, m.y1, m.y2))
        q1 = self.yy - y1 - m.path_coupling * (self.xx - x1)
        q2 = self.yy - y2 + m.path_coupling * (self.xx - x2)
        v11 = 0.5 * kx * (self.xx - x1) ** 2 + 0.5 * ky * q1**2
        v22 = (
            0.5 * kx * (self.xx - x2) ** 2
            + 0.5 * ky * q2**2
            + m.bias_cminv * CMINV_TO_HARTREE
        )
        sx = m.coupling_sigma_x * a2b
        sy = m.coupling_sigma_y * a2b
        v12 = m.diabatic_coupling_cminv * CMINV_TO_HARTREE * np.exp(
            -0.5 * (self.xx / sx) ** 2 - 0.5 * (self.yy / sy) ** 2
        )
        return v11, v22, v12

    def _initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        m = self.model
        a2b = ANGSTROM_TO_BOHR
        phase = (
            (m.px0_a_inv / a2b) * self.xx
            + (m.py0_a_inv / a2b) * self.yy
        )
        psi1 = np.exp(
            -0.5 * ((self.xx - m.x0 * a2b) / (m.sigma_x * a2b)) ** 2
            -0.5 * ((self.yy - m.y0 * a2b) / (m.sigma_y * a2b)) ** 2
            + 1j * phase
        )
        psi2 = np.zeros_like(psi1)
        norm = np.sum(abs(psi1) ** 2) * self.dx * self.dy
        psi1 /= np.sqrt(norm)
        return psi1, psi2

    def _potential_half_step(self, dt_au: float) -> None:
        trace = 0.5 * (self.v11 + self.v22)
        dz = 0.5 * (self.v11 - self.v22)
        omega = np.sqrt(dz**2 + self.v12**2)
        phase = np.exp(-0.5j * dt_au * trace)
        angle = 0.5 * dt_au * omega
        sine = np.empty_like(omega)
        small = omega < 1.0e-14
        sine[small] = 0.5 * dt_au
        sine[~small] = np.sin(angle[~small]) / omega[~small]
        cosine = np.cos(angle)
        a = self.psi1.copy()
        b = self.psi2.copy()
        self.psi1 = phase * ((cosine - 1j * sine * dz) * a - 1j * sine * self.v12 * b)
        self.psi2 = phase * (-1j * sine * self.v12 * a + (cosine + 1j * sine * dz) * b)

    def step(self, dt_fs: float, absorber_width_a: float = 0.0,
             absorber_rate_fs_inv: float = 0.0, absorber_power: int = 4) -> None:
        dt = dt_fs * FS_TO_AU
        self._potential_half_step(dt)
        kinetic_phase = np.exp(-1j * dt * self.kinetic)
        self.psi1 = np.fft.ifft2(np.fft.fft2(self.psi1) * kinetic_phase)
        self.psi2 = np.fft.ifft2(np.fft.fft2(self.psi2) * kinetic_phase)
        self._potential_half_step(dt)
        if absorber_width_a > 0.0 and absorber_rate_fs_inv > 0.0:
            width = absorber_width_a * ANGSTROM_TO_BOHR
            xedge = np.maximum(
                self.model.xmin * ANGSTROM_TO_BOHR + width - self.xx,
                self.xx - (self.model.xmax * ANGSTROM_TO_BOHR - width),
            )
            yedge = np.maximum(
                self.model.ymin * ANGSTROM_TO_BOHR + width - self.yy,
                self.yy - (self.model.ymax * ANGSTROM_TO_BOHR - width),
            )
            scaled = np.maximum.reduce((xedge / width, yedge / width, np.zeros_like(xedge)))
            mask = np.exp(-absorber_rate_fs_inv * dt_fs * scaled**absorber_power)
            self.psi1 *= mask
            self.psi2 *= mask

    def populations(self) -> tuple[float, float]:
        factor = self.dx * self.dy
        return (
            float(np.sum(abs(self.psi1) ** 2) * factor),
            float(np.sum(abs(self.psi2) ** 2) * factor),
        )


def run(args: argparse.Namespace) -> dict[str, float]:
    model = Model2D(nx=args.nx, ny=args.ny)
    solver = SplitOperator2D(model)
    if args.steps < 0 or args.save_every <= 0 or args.dt <= 0.0:
        raise ValueError("invalid propagation controls")
    frames = args.steps // args.save_every + 1
    time = np.empty(frames)
    pop1 = np.empty(frames)
    pop2 = np.empty(frames)
    density1 = np.empty((frames, model.ny, model.nx))
    density2 = np.empty_like(density1)

    def save(index: int, step: int) -> None:
        time[index] = step * args.dt
        pop1[index], pop2[index] = solver.populations()
        scale = ANGSTROM_TO_BOHR**2
        density1[index] = abs(solver.psi1) ** 2 * scale
        density2[index] = abs(solver.psi2) ** 2 * scale

    save(0, 0)
    frame = 1
    for step in range(1, args.steps + 1):
        solver.step(args.dt, args.absorber_width, args.absorber_rate, args.absorber_power)
        if step % args.save_every == 0:
            save(frame, step)
            frame += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.attrs["method"] = "2D two-state split operator"
        handle.attrs["dt_fs"] = args.dt
        handle.create_dataset("x_angstrom", data=solver.x / ANGSTROM_TO_BOHR)
        handle.create_dataset("y_angstrom", data=solver.y / ANGSTROM_TO_BOHR)
        handle.create_dataset("time_fs", data=time)
        handle.create_dataset("pop1", data=pop1)
        handle.create_dataset("pop2", data=pop2)
        handle.create_dataset("density1_angstrom-2", data=density1)
        handle.create_dataset("density2_angstrom-2", data=density2)
        handle.create_dataset("V11_cm-1", data=solver.v11 / CMINV_TO_HARTREE)
        handle.create_dataset("V22_cm-1", data=solver.v22 / CMINV_TO_HARTREE)
        handle.create_dataset("V12_cm-1", data=solver.v12 / CMINV_TO_HARTREE)

    total = pop1 + pop2
    summary = {
        "max_norm_drift": float(np.max(np.abs(total - total[0]))),
        "final_product_population": float(pop2[-1]),
    }
    print("2D transfer solver complete")
    for key, value in summary.items():
        print(f"{key}: {value:.10g}")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", default="transfer_2d.h5")
    result.add_argument("--nx", type=int, default=96)
    result.add_argument("--ny", type=int, default=48)
    result.add_argument("--dt", type=float, default=0.01, help="fs")
    result.add_argument("--steps", type=int, default=1000)
    result.add_argument("--save-every", type=int, default=20)
    result.add_argument("--absorber-width", type=float, default=0.0, help="angstrom")
    result.add_argument("--absorber-rate", type=float, default=0.0, help="fs^-1")
    result.add_argument("--absorber-power", type=int, default=4)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
