"""Validated physical-unit HDF5 interface for three-dimensional diabatic PES data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from quantum_fdt_density import ANGSTROM_TO_BOHR, CMINV_TO_HARTREE


@dataclass(frozen=True)
class DiabaticPES3D:
    """Hermitian two-state diabatic potential on a regular physical-unit grid.

    Array order is ``(z, y, x)``. Axis datasets use angstrom; energy datasets
    use reciprocal centimetres. Complex coupling is stored as real/imaginary
    datasets so HDF5 files remain portable across producer codes.
    """

    x_angstrom: np.ndarray
    y_angstrom: np.ndarray
    z_angstrom: np.ndarray
    v11_cminv: np.ndarray
    v22_cminv: np.ndarray
    v12_cminv: np.ndarray

    def __post_init__(self) -> None:
        axes = (
            np.asarray(self.x_angstrom, dtype=float),
            np.asarray(self.y_angstrom, dtype=float),
            np.asarray(self.z_angstrom, dtype=float),
        )
        for name, axis in zip(("x", "y", "z"), axes, strict=True):
            if axis.ndim != 1 or axis.size < 2:
                raise ValueError(f"{name} axis must contain at least two points")
            if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
                raise ValueError(f"{name} axis must be finite and strictly increasing")
        shape = (axes[2].size, axes[1].size, axes[0].size)
        for name, surface in (
            ("V11", self.v11_cminv),
            ("V22", self.v22_cminv),
            ("V12", self.v12_cminv),
        ):
            array = np.asarray(surface)
            if array.shape != shape:
                raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")

    @classmethod
    def from_hdf5(cls, path: str | Path) -> DiabaticPES3D:
        with h5py.File(path, "r") as handle:
            required = (
                "x_angstrom",
                "y_angstrom",
                "z_angstrom",
                "V11_cm-1",
                "V22_cm-1",
                "V12_real_cm-1",
            )
            missing = [name for name in required if name not in handle]
            if missing:
                raise ValueError(f"PES file missing datasets: {', '.join(missing)}")

            def units(name: str) -> str:
                value = handle[name].attrs.get("units", "")
                return value.decode("utf-8") if isinstance(value, bytes) else str(value)

            for name in ("x_angstrom", "y_angstrom", "z_angstrom"):
                if units(name) != "angstrom":
                    raise ValueError(f"{name} units must be 'angstrom'")
            for name in ("V11_cm-1", "V22_cm-1", "V12_real_cm-1"):
                if units(name) != "cm^-1":
                    raise ValueError(f"{name} units must be 'cm^-1'")
            imaginary = np.zeros_like(handle["V12_real_cm-1"][...])
            if "V12_imag_cm-1" in handle:
                if units("V12_imag_cm-1") != "cm^-1":
                    raise ValueError("V12_imag_cm-1 units must be 'cm^-1'")
                imaginary = handle["V12_imag_cm-1"][...]
            return cls(
                handle["x_angstrom"][...],
                handle["y_angstrom"][...],
                handle["z_angstrom"][...],
                handle["V11_cm-1"][...],
                handle["V22_cm-1"][...],
                handle["V12_real_cm-1"][...] + 1.0j * imaginary,
            )

    def to_hdf5(self, path: str | Path, source: str = "external") -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, "w") as handle:
            handle.attrs["schema"] = "QuantumDynamics diabatic PES 3D v1"
            handle.attrs["source"] = source
            for name, data in (
                ("x_angstrom", self.x_angstrom),
                ("y_angstrom", self.y_angstrom),
                ("z_angstrom", self.z_angstrom),
            ):
                dataset = handle.create_dataset(name, data=data)
                dataset.attrs["units"] = "angstrom"
            for name, data in (
                ("V11_cm-1", self.v11_cminv),
                ("V22_cm-1", self.v22_cminv),
                ("V12_real_cm-1", np.real(self.v12_cminv)),
                ("V12_imag_cm-1", np.imag(self.v12_cminv)),
            ):
                dataset = handle.create_dataset(name, data=data)
                dataset.attrs["units"] = "cm^-1"

    def interpolate_atomic(
        self, x_bohr: np.ndarray, y_bohr: np.ndarray, z_bohr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate surfaces at atomic-unit coordinates; return hartree."""
        x = np.asarray(x_bohr, dtype=float) / ANGSTROM_TO_BOHR
        y = np.asarray(y_bohr, dtype=float) / ANGSTROM_TO_BOHR
        z = np.asarray(z_bohr, dtype=float) / ANGSTROM_TO_BOHR
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        points = np.column_stack((zz.ravel(), yy.ravel(), xx.ravel()))
        axes = (self.z_angstrom, self.y_angstrom, self.x_angstrom)

        def interpolate(surface: np.ndarray) -> np.ndarray:
            function = RegularGridInterpolator(axes, surface, bounds_error=True)
            return function(points).reshape(zz.shape) * CMINV_TO_HARTREE

        return (
            interpolate(np.asarray(self.v11_cminv)),
            interpolate(np.asarray(self.v22_cminv)),
            interpolate(np.asarray(self.v12_cminv)),
        )
