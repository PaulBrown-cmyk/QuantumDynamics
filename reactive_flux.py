#!/usr/bin/env python3
"""Reactive-flux and energy-resolved transmission analysis for 1D trajectories.

The tool evaluates state-resolved probability current from saved complex
wavefunctions, integrates net/forward/backward crossings, checks the spatial
continuity equation, estimates a user-selected rate plateau, and constructs a
wavepacket transmission spectrum from two flux surfaces.  Time-origin
flux-side/flux-flux arrays are trajectory correlations, not equilibrium Kubo
correlations; this distinction is recorded in every output file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from ShowDyn import TrajData, read_traj_auto


ANGSTROM_TO_BOHR = 1.8897261254578281
FS_TO_AU = 41.341374575751
AMU_TO_ELECTRON_MASS = 1822.888486209
HARTREE_TO_CMINV = 219474.6313705


def _trapezoid(values: np.ndarray, coordinate: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, coordinate))
    return float(np.trapz(values, coordinate))


def _uniform_spacing(axis: np.ndarray, name: str) -> float:
    axis = np.asarray(axis, dtype=float).reshape(-1)
    if axis.size < 4 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain at least four finite values")
    spacing = np.diff(axis)
    if np.any(spacing <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    mean = float(np.mean(spacing))
    if not np.allclose(spacing, mean, rtol=1.0e-8, atol=1.0e-12 * max(1.0, abs(mean))):
        raise ValueError(f"{name} must be uniformly spaced")
    return mean


def _components(data: TrajData) -> np.ndarray:
    if data.psi1 is not None and data.psi2 is not None:
        result = np.stack((data.psi1, data.psi2), axis=0)
    elif data.psi is not None:
        result = data.psi[None, ...]
    else:
        raise ValueError("complex wavefunctions are required for flux analysis")
    if result.shape[1:] != (data.t.size, data.x.size):
        raise ValueError("wavefunction, time, and coordinate shapes are inconsistent")
    if not np.all(np.isfinite(result)):
        raise ValueError("wavefunctions contain non-finite values")
    return np.asarray(result, dtype=complex)


def _surface_interpolate(values: np.ndarray, x: np.ndarray, surface: float) -> np.ndarray:
    if not np.isfinite(surface) or surface < x[0] or surface > x[-1]:
        raise ValueError("surface lies outside coordinate grid")
    upper = int(np.searchsorted(x, surface, side="right"))
    if upper == 0:
        return values[..., 0]
    if upper >= x.size:
        return values[..., -1]
    lower = upper - 1
    fraction = (surface - x[lower]) / (x[upper] - x[lower])
    return (1.0 - fraction) * values[..., lower] + fraction * values[..., upper]


def surface_wavefunction_and_derivative(
    x_angstrom: np.ndarray,
    wavefunctions_angstrom: np.ndarray,
    surface_angstrom: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return state/time wavefunction and derivative at a surface in atomic units."""
    x_angstrom = np.asarray(x_angstrom, dtype=float).reshape(-1)
    dx_angstrom = _uniform_spacing(x_angstrom, "coordinate grid")
    wavefunctions = np.asarray(wavefunctions_angstrom, dtype=complex)
    if wavefunctions.ndim != 3 or wavefunctions.shape[-1] != x_angstrom.size:
        raise ValueError("wavefunctions must have shape (state,time,x)")
    x_bohr = x_angstrom * ANGSTROM_TO_BOHR
    dx_bohr = dx_angstrom * ANGSTROM_TO_BOHR
    psi_bohr = wavefunctions / np.sqrt(ANGSTROM_TO_BOHR)
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(x_bohr.size, d=dx_bohr)
    derivative = np.fft.ifft(
        1j * wave_numbers[None, None, :] * np.fft.fft(psi_bohr, axis=-1),
        axis=-1,
    )
    return (
        _surface_interpolate(psi_bohr, x_bohr, surface_angstrom * ANGSTROM_TO_BOHR),
        _surface_interpolate(derivative, x_bohr, surface_angstrom * ANGSTROM_TO_BOHR),
    )


def surface_probability_current(
    x_angstrom: np.ndarray,
    wavefunctions_angstrom: np.ndarray,
    mass_amu: float,
    surface_angstrom: float,
) -> np.ndarray:
    """State-resolved surface current in probability per femtosecond."""
    if not np.isfinite(mass_amu) or mass_amu <= 0.0:
        raise ValueError("mass must be positive and finite")
    psi, derivative = surface_wavefunction_and_derivative(
        x_angstrom, wavefunctions_angstrom, surface_angstrom
    )
    current_au = np.imag(np.conj(psi) * derivative) / (
        mass_amu * AMU_TO_ELECTRON_MASS
    )
    return current_au * FS_TO_AU


def _cumulative_trapezoid(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    result[1:] = np.cumsum(0.5 * (values[:-1] + values[1:]) * np.diff(time))
    return result


def _trajectory_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Raw multiple-time-origin correlation C(lag)=mean(left(t) right(t+lag))."""
    count = left.size
    result = np.empty(count, dtype=float)
    for lag in range(count):
        result[lag] = float(np.mean(left[: count - lag] * right[lag:]))
    return result


def reactive_flux_diagnostics(
    data: TrajData,
    mass_amu: float,
    surface_angstrom: float = 0.0,
    plateau_start_fs: Optional[float] = None,
    plateau_end_fs: Optional[float] = None,
    absorber_width_angstrom: Optional[float] = None,
    absorber_rate_fs_inv: float = 0.0,
    absorber_power: int = 4,
) -> dict[str, np.ndarray | float]:
    """Compute currents, side populations, recrossing, continuity, and plateau rate."""
    dx = _uniform_spacing(data.x, "coordinate grid")
    _uniform_spacing(data.t, "time grid")
    wavefunctions = _components(data)
    state_current = surface_probability_current(
        data.x, wavefunctions, mass_amu, surface_angstrom
    )
    total_current = np.sum(state_current, axis=0)

    # Periodic FFT grids have a second side boundary at the box seam.  Retain its
    # current so the discrete side-population continuity test remains meaningful.
    boundary_current = np.sum(
        surface_probability_current(data.x, wavefunctions, mass_amu, float(data.x[0])),
        axis=0,
    )
    net_side_current = total_current - boundary_current
    mask = data.x >= surface_angstrom
    if not np.any(mask) or np.all(mask):
        raise ValueError("dividing surface must split coordinate grid")
    # Treat periodic grid values as cell centers.  Fractional weight for the cell
    # cut by the requested surface removes half-cell bias in flux/side continuity.
    side_weights = np.clip((data.x + 0.5 * dx - surface_angstrom) / dx, 0.0, 1.0)
    side_population = np.sum(data.rho * side_weights[None, :], axis=1) * dx
    total_norm = np.sum(data.rho, axis=1) * dx
    reactant_population = total_norm - side_population

    state_side_population = np.empty((state_current.shape[0], data.t.size))
    component_density = np.abs(wavefunctions) ** 2
    for state in range(state_current.shape[0]):
        state_side_population[state] = (
            np.sum(component_density[state] * side_weights[None, :], axis=1) * dx
        )

    cumulative_net = _cumulative_trapezoid(net_side_current, data.t)
    cumulative_forward = _cumulative_trapezoid(np.maximum(net_side_current, 0.0), data.t)
    cumulative_backward = _cumulative_trapezoid(np.maximum(-net_side_current, 0.0), data.t)
    predicted_side = side_population[0] + cumulative_net
    side_derivative = np.gradient(side_population, data.t, edge_order=2)
    continuity_residual = side_derivative - net_side_current
    norm_loss = total_norm[0] - total_norm

    left_absorber_rate = np.zeros(data.t.size)
    right_absorber_rate = np.zeros(data.t.size)
    if absorber_width_angstrom is not None or absorber_rate_fs_inv > 0.0:
        if (
            absorber_width_angstrom is None
            or absorber_width_angstrom <= 0.0
            or 2.0 * absorber_width_angstrom >= data.x[-1] - data.x[0] + dx
        ):
            raise ValueError("absorber width must be positive and smaller than half the box")
        if absorber_rate_fs_inv <= 0.0 or absorber_power < 1:
            raise ValueError("absorber rate and power must be positive")
        left_profile = np.clip(
            (data.x[0] + absorber_width_angstrom - data.x) / absorber_width_angstrom,
            0.0,
            1.0,
        ) ** absorber_power
        right_profile = np.clip(
            (data.x - (data.x[-1] - absorber_width_angstrom)) / absorber_width_angstrom,
            0.0,
            1.0,
        ) ** absorber_power
        # Absorber acts on amplitude as exp(-Gamma dt f), hence probability
        # loss rate is 2 Gamma integral(f rho dx).
        left_absorber_rate = (
            2.0 * absorber_rate_fs_inv * np.sum(data.rho * left_profile[None, :], axis=1) * dx
        )
        right_absorber_rate = (
            2.0 * absorber_rate_fs_inv * np.sum(data.rho * right_profile[None, :], axis=1) * dx
        )
    cumulative_left_absorber = _cumulative_trapezoid(left_absorber_rate, data.t)
    cumulative_right_absorber = _cumulative_trapezoid(right_absorber_rate, data.t)

    safe_reactant = np.maximum(reactant_population, 64.0 * np.finfo(float).eps)
    instantaneous_rate = net_side_current / safe_reactant
    start = float(data.t[int(0.8 * (data.t.size - 1))]) if plateau_start_fs is None else plateau_start_fs
    end = float(data.t[-1]) if plateau_end_fs is None else plateau_end_fs
    plateau_mask = (data.t >= start) & (data.t <= end)
    if np.count_nonzero(plateau_mask) < 2:
        raise ValueError("plateau interval contains fewer than two frames")
    plateau_values = instantaneous_rate[plateau_mask]
    plateau_mean = float(np.mean(plateau_values))
    plateau_sem = float(np.std(plateau_values, ddof=1) / np.sqrt(plateau_values.size))
    plateau_time = data.t[plateau_mask]
    plateau_slope = float(np.polyfit(plateau_time, plateau_values, 1)[0])
    plateau_span = max(float(end - start), np.finfo(float).eps)
    plateau_established = float(
        plateau_mean > 0.0
        and plateau_mean > 2.0 * plateau_sem
        and abs(plateau_slope) * plateau_span < 0.5 * abs(plateau_mean)
    )

    flux_side = _trajectory_correlation(net_side_current, side_population)
    flux_flux = _trajectory_correlation(net_side_current, net_side_current)
    forward_total = float(cumulative_forward[-1])
    recrossing_fraction = (
        float(cumulative_backward[-1] / forward_total) if forward_total > 0.0 else 0.0
    )
    return {
        "time_fs": data.t.copy(),
        "lag_fs": data.t - data.t[0],
        "state_current_fs-1": state_current,
        "total_current_fs-1": total_current,
        "periodic_boundary_current_fs-1": boundary_current,
        "net_side_current_fs-1": net_side_current,
        "side_population": side_population,
        "state_side_population": state_side_population,
        "reactant_population": reactant_population,
        "total_norm": total_norm,
        "absorbed_norm": norm_loss,
        "left_absorber_loss_rate_fs-1": left_absorber_rate,
        "right_absorber_loss_rate_fs-1": right_absorber_rate,
        "cumulative_left_absorber_yield": cumulative_left_absorber,
        "cumulative_right_absorber_yield": cumulative_right_absorber,
        "cumulative_net_flux": cumulative_net,
        "cumulative_forward_flux": cumulative_forward,
        "cumulative_backward_flux": cumulative_backward,
        "predicted_side_population": predicted_side,
        "continuity_residual_fs-1": continuity_residual,
        "instantaneous_rate_fs-1": instantaneous_rate,
        "trajectory_flux_side_correlation_fs-1": flux_side,
        "trajectory_flux_flux_correlation_fs-2": flux_flux,
        "plateau_start_fs": float(start),
        "plateau_end_fs": float(end),
        "plateau_rate_fs-1": plateau_mean,
        "plateau_rate_sem_fs-1": plateau_sem,
        "plateau_rate_slope_fs-2": plateau_slope,
        "plateau_established": plateau_established,
        "continuity_rms_fs-1": float(np.sqrt(np.mean(continuity_residual**2))),
        "continuity_max_fs-1": float(np.max(np.abs(continuity_residual))),
        "integrated_flux_side_error": float(np.max(np.abs(predicted_side - side_population))),
        "recrossing_fraction": recrossing_fraction,
        "final_absorbed_norm": float(norm_loss[-1]),
    }


def _time_window(size: int, kind: str) -> np.ndarray:
    if kind == "none":
        return np.ones(size)
    if kind == "hann":
        return np.hanning(size)
    if kind == "tukey":
        # Ten-percent cosine tapers preserve separated crossing amplitudes while
        # suppressing endpoint leakage; no SciPy dependency required.
        result = np.ones(size)
        edge = max(1, int(round(0.05 * (size - 1))))
        phase = np.linspace(0.0, np.pi, edge + 1)
        taper = 0.5 * (1.0 - np.cos(phase))
        result[: edge + 1] = taper
        result[-edge - 1 :] = taper[::-1]
        return result
    raise ValueError(f"unknown time window: {kind}")


def energy_resolved_transmission(
    x_angstrom: np.ndarray,
    time_fs: np.ndarray,
    wavefunctions_angstrom: np.ndarray,
    mass_amu: float,
    incident_surface_angstrom: float,
    product_surface_angstrom: float,
    window: str = "tukey",
    incident_threshold: float = 1.0e-3,
) -> dict[str, np.ndarray | float]:
    """Return flux spectra and their single-channel transmission ratio."""
    dt_fs = _uniform_spacing(time_fs, "time grid")
    if incident_surface_angstrom >= product_surface_angstrom:
        raise ValueError("incident surface must lie left of product surface")
    if not 0.0 < incident_threshold < 1.0:
        raise ValueError("incident threshold must lie between zero and one")
    mass = mass_amu * AMU_TO_ELECTRON_MASS
    dt_au = dt_fs * FS_TO_AU
    taper = _time_window(time_fs.size, window)[None, :]

    def spectrum_at(surface: float) -> np.ndarray:
        psi, derivative = surface_wavefunction_and_derivative(
            x_angstrom, wavefunctions_angstrom, surface
        )
        psi_energy = np.fft.ifft(psi * taper, axis=-1) * psi.shape[-1] * dt_au
        derivative_energy = (
            np.fft.ifft(derivative * taper, axis=-1) * derivative.shape[-1] * dt_au
        )
        return np.sum(np.imag(np.conj(psi_energy) * derivative_energy), axis=0) / mass

    frequencies = np.fft.fftfreq(time_fs.size, d=dt_au)
    energy_hartree = 2.0 * np.pi * frequencies
    positive = energy_hartree > 0.0
    energy_hartree = energy_hartree[positive]
    if energy_hartree.size < 2:
        raise ValueError("time grid is too short for an energy-resolved spectrum")
    incident_signed = spectrum_at(incident_surface_angstrom)[positive]
    product_signed = spectrum_at(product_surface_angstrom)[positive]
    incident = np.maximum(incident_signed, 0.0)
    product = np.maximum(product_signed, 0.0)
    threshold = incident_threshold * max(float(np.max(incident)), np.finfo(float).tiny)
    valid = incident > threshold
    transmission = np.full(incident.shape, np.nan)
    transmission[valid] = product[valid] / incident[valid]
    if np.count_nonzero(valid) < 2:
        integrated = np.nan
    else:
        integrated = float(
            _trapezoid(product[valid], energy_hartree[valid])
            / _trapezoid(incident[valid], energy_hartree[valid])
        )
    bound_violation = (
        float(max(0.0, np.nanmax(transmission[valid]) - 1.0))
        if np.any(valid)
        else np.nan
    )
    return {
        "energy_cminv": energy_hartree * HARTREE_TO_CMINV,
        "incident_flux_spectrum": incident,
        "product_flux_spectrum": product,
        "incident_signed_flux_spectrum": incident_signed,
        "product_signed_flux_spectrum": product_signed,
        "transmission_probability": transmission,
        "cumulative_reaction_probability": transmission.copy(),
        "valid_transmission": valid.astype(np.uint8),
        "integrated_transmission": integrated,
        "maximum_transmission_bound_violation": bound_violation,
        "energy_resolution_cminv": float(
            (energy_hartree[1] - energy_hartree[0]) * HARTREE_TO_CMINV
        ),
        "energy_nyquist_cminv": float(energy_hartree[-1] * HARTREE_TO_CMINV),
    }


def _write_outputs(
    output: Path,
    flux: dict[str, np.ndarray | float],
    spectrum: Optional[dict[str, np.ndarray | float]],
    args: argparse.Namespace,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.attrs["method"] = "wavepacket reactive-flux analysis"
        handle.attrs["correlation_scope"] = (
            "multiple-time-origin trajectory estimators; not equilibrium Kubo correlations"
        )
        handle.attrs["mass_amu"] = args.mass
        handle.attrs["dividing_surface_angstrom"] = args.surface
        handle.attrs["absorber_width_angstrom"] = (
            np.nan if args.absorber_width is None else args.absorber_width
        )
        handle.attrs["absorber_rate_fs-1"] = args.absorber_rate
        handle.attrs["absorber_power"] = args.absorber_power
        handle.attrs["plateau_rate_fs-1"] = flux["plateau_rate_fs-1"]
        handle.attrs["plateau_rate_sem_fs-1"] = flux["plateau_rate_sem_fs-1"]
        handle.attrs["recrossing_fraction"] = flux["recrossing_fraction"]
        handle.attrs["plateau_established"] = int(flux["plateau_established"])
        handle.attrs["continuity_rms_fs-1"] = flux["continuity_rms_fs-1"]
        handle.attrs["integrated_flux_side_error"] = flux["integrated_flux_side_error"]
        for key, value in flux.items():
            if isinstance(value, np.ndarray):
                handle.create_dataset(key, data=value)
            elif key not in handle.attrs:
                handle.attrs[key] = value
        if spectrum is not None:
            group = handle.create_group("energy_resolved")
            group.attrs["incident_surface_angstrom"] = args.incident_surface
            group.attrs["product_surface_angstrom"] = args.product_surface
            group.attrs["time_window"] = args.window
            group.attrs["incident_threshold"] = args.incident_threshold
            group.attrs["integrated_transmission"] = spectrum["integrated_transmission"]
            group.attrs["maximum_transmission_bound_violation"] = spectrum[
                "maximum_transmission_bound_violation"
            ]
            for key, value in spectrum.items():
                if isinstance(value, np.ndarray):
                    group.create_dataset(key, data=value)
                elif key not in group.attrs:
                    group.attrs[key] = value

    time_columns = np.column_stack(
        (
            flux["time_fs"],
            flux["side_population"],
            flux["reactant_population"],
            flux["net_side_current_fs-1"],
            flux["cumulative_net_flux"],
            flux["cumulative_forward_flux"],
            flux["cumulative_backward_flux"],
            flux["instantaneous_rate_fs-1"],
            flux["absorbed_norm"],
            flux["cumulative_left_absorber_yield"],
            flux["cumulative_right_absorber_yield"],
        )
    )
    np.savetxt(
        output.with_suffix(".time.csv"),
        time_columns,
        delimiter=",",
        header=(
            "time_fs,side_population,reactant_population,net_current_fs-1,"
            "cumulative_net,cumulative_forward,cumulative_backward,rate_fs-1,absorbed_norm,"
            "left_absorber_yield,right_absorber_yield"
        ),
        comments="",
    )
    if spectrum is not None:
        energy_columns = np.column_stack(
            (
                spectrum["energy_cminv"],
                spectrum["incident_flux_spectrum"],
                spectrum["product_flux_spectrum"],
                spectrum["transmission_probability"],
                spectrum["valid_transmission"],
            )
        )
        np.savetxt(
            output.with_suffix(".energy.csv"),
            energy_columns,
            delimiter=",",
            header="energy_cminv,incident_flux,product_flux,transmission,valid",
            comments="",
        )


def _plot_results(
    path: Path,
    flux: dict[str, np.ndarray | float],
    spectrum: Optional[dict[str, np.ndarray | float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = 3 if spectrum is not None else 2
    fig, axes = plt.subplots(rows, 1, figsize=(9, 3.2 * rows), constrained_layout=True)
    time = flux["time_fs"]
    axes[0].plot(time, flux["side_population"], label="direct side population")
    axes[0].plot(time, flux["predicted_side_population"], "--", label="integrated net flux")
    axes[0].set(xlabel="time (fs)", ylabel="product-side probability")
    axes[0].legend()
    axes[1].plot(time, flux["net_side_current_fs-1"], label="net current")
    axes[1].plot(time, flux["instantaneous_rate_fs-1"], label="current/reactant")
    axes[1].axhline(float(flux["plateau_rate_fs-1"]), color="black", ls=":", label="plateau mean")
    axes[1].set(xlabel="time (fs)", ylabel="flux or rate (fs$^{-1}$)")
    axes[1].legend()
    if spectrum is not None:
        energy = spectrum["energy_cminv"]
        axes[2].plot(energy, spectrum["incident_flux_spectrum"], label="incident")
        axes[2].plot(energy, spectrum["product_flux_spectrum"], label="product")
        axes[2].set(xlabel="energy (cm$^{-1}$)", ylabel="spectral flux (arb. common scale)")
        transmission_axis = axes[2].twinx()
        transmission_axis.plot(
            energy, spectrum["transmission_probability"], color="black", lw=1.0, label="T(E)"
        )
        transmission_axis.set_ylabel("transmission probability")
        handles1, labels1 = axes[2].get_legend_handles_labels()
        handles2, labels2 = transmission_axis.get_legend_handles_labels()
        axes[2].legend(handles1 + handles2, labels1 + labels2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("trajectory")
    result.add_argument("--output", default="reactive_flux.h5")
    result.add_argument("--plot", default=None)
    result.add_argument("--mass", type=float, default=1.00784, help="transferred mass in amu")
    result.add_argument("--surface", type=float, default=0.0, help="dividing surface in angstrom")
    result.add_argument("--plateau-start", type=float, default=None, help="fs")
    result.add_argument("--plateau-end", type=float, default=None, help="fs")
    result.add_argument("--absorber-width", type=float, default=None, help="angstrom")
    result.add_argument("--absorber-rate", type=float, default=0.0, help="fs^-1")
    result.add_argument("--absorber-power", type=int, default=4)
    result.add_argument("--incident-surface", type=float, default=-0.5, help="angstrom")
    result.add_argument("--product-surface", type=float, default=0.5, help="angstrom")
    result.add_argument("--window", choices=("none", "hann", "tukey"), default="tukey")
    result.add_argument(
        "--incident-threshold",
        type=float,
        default=1.0e-3,
        help="minimum incident spectral current relative to its maximum",
    )
    result.add_argument("--no-spectrum", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    data = read_traj_auto(args.trajectory)
    flux = reactive_flux_diagnostics(
        data,
        args.mass,
        args.surface,
        args.plateau_start,
        args.plateau_end,
        args.absorber_width,
        args.absorber_rate,
        args.absorber_power,
    )
    spectrum = None
    if not args.no_spectrum:
        spectrum = energy_resolved_transmission(
            data.x,
            data.t,
            _components(data),
            args.mass,
            args.incident_surface,
            args.product_surface,
            args.window,
            args.incident_threshold,
        )
    output = Path(args.output)
    _write_outputs(output, flux, spectrum, args)
    if args.plot:
        _plot_results(Path(args.plot), flux, spectrum)
    print("Reactive-flux analysis complete")
    print(f"plateau_rate_fs-1: {float(flux['plateau_rate_fs-1']):.10g}")
    print(f"plateau_rate_sem_fs-1: {float(flux['plateau_rate_sem_fs-1']):.10g}")
    print(f"plateau_established: {bool(flux['plateau_established'])}")
    print(f"recrossing_fraction: {float(flux['recrossing_fraction']):.10g}")
    print(f"continuity_rms_fs-1: {float(flux['continuity_rms_fs-1']):.10g}")
    print(f"integrated_flux_side_error: {float(flux['integrated_flux_side_error']):.10g}")
    if spectrum is not None:
        print(f"integrated_transmission: {float(spectrum['integrated_transmission']):.10g}")
        if float(spectrum["maximum_transmission_bound_violation"]) > 0.05:
            print(
                "WARNING: T(E) exceeds unity; finite-window, recrossing, damping, "
                "or driving prevents strict asymptotic scattering interpretation"
            )


if __name__ == "__main__":
    main()
