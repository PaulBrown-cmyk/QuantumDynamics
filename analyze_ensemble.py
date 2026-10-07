#!/usr/bin/env python3
"""Aggregate QuantumDynamics HDF5 trajectories and fit ensemble kinetics."""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import h5py
import numpy as np


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(Path(match) for match in matches)
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise SystemExit("No trajectory files matched")
    return unique


def read_observables(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        names = sorted(name for name in handle if name.startswith("step_"))
        if not names:
            raise ValueError(f"No step groups in {path}")
        time = np.array([float(np.asarray(handle[name]["t"]).reshape(-1)[0]) for name in names])
        pop1 = np.array([float(np.asarray(handle[name]["pop1"]).reshape(-1)[0]) for name in names])
        pop2 = np.array([float(np.asarray(handle[name]["pop2"]).reshape(-1)[0]) for name in names])
    return time, pop1, pop2


def read_observable_records(path: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read legacy per-step file or parallel single-file ensemble layout."""
    with h5py.File(path, "r") as handle:
        if all(name in handle for name in ("time_fs", "pop1", "pop2")):
            time = np.asarray(handle["time_fs"], dtype=float)
            pop1 = np.asarray(handle["pop1"], dtype=float)
            pop2 = np.asarray(handle["pop2"], dtype=float)
            if pop1.ndim == 1:
                return [(time, pop1, pop2)]
            if pop1.ndim == 2 and pop1.shape == pop2.shape:
                return [(time, pop1[index], pop2[index]) for index in range(pop1.shape[0])]
            raise ValueError(f"Unsupported ensemble observable shape in {path}: {pop1.shape}")
    return [read_observables(path)]


def bounded_fit(time: np.ndarray, product: np.ndarray, points: int = 140) -> dict[str, float]:
    if time.size < 5 or np.ptp(product) <= 1.0e-12:
        raise ValueError("Population series lacks enough kinetic variation")
    threshold = float(np.min(product) + 0.01 * np.ptp(product))
    candidates = np.flatnonzero(product >= threshold)
    onset = int(candidates[0]) if candidates.size else 0
    u = time[onset:] - time[onset]
    observed = product[onset:]
    p0 = float(observed[0])
    target = observed - p0
    dt = float(np.median(np.diff(u)))
    rates = np.logspace(np.log10(1.0 / (20.0 * (u[-1] + dt))), np.log10(3.0 / dt), points)

    mono = (np.inf, None)
    for rate in rates:
        basis = 1.0 - np.exp(-rate * u)
        amplitude = float(np.dot(basis, target) / np.dot(basis, basis))
        if 0.0 <= amplitude <= 1.0 - p0:
            prediction = p0 + amplitude * basis
            rss = float(np.sum((observed - prediction) ** 2))
            if rss < mono[0]:
                mono = (rss, (rate, amplitude))

    bi = (np.inf, None)
    for slow_index, slow_rate in enumerate(rates[:-1]):
        slow_basis = 1.0 - np.exp(-slow_rate * u)
        for fast_rate in rates[slow_index + 1:]:
            design = np.column_stack((slow_basis, 1.0 - np.exp(-fast_rate * u)))
            amplitudes = np.linalg.lstsq(design, target, rcond=None)[0]
            if np.any(amplitudes < 0.0) or np.sum(amplitudes) > 1.0 - p0:
                continue
            prediction = p0 + design @ amplitudes
            rss = float(np.sum((observed - prediction) ** 2))
            if rss < bi[0]:
                bi = (rss, (slow_rate, fast_rate, amplitudes))

    if mono[1] is None or bi[1] is None:
        raise ValueError("No bounded kinetic fit found")
    mono_rss, (mono_rate, mono_amplitude) = mono
    bi_rss, (slow_rate, fast_rate, amplitudes) = bi
    count = observed.size
    return {
        "onset_fs": float(time[onset]),
        "p0": p0,
        "mono_rate_fs-1": float(mono_rate),
        "mono_amplitude": float(mono_amplitude),
        "mono_aic": float(count * np.log(max(mono_rss, np.finfo(float).tiny) / count) + 4.0),
        "slow_rate_fs-1": float(slow_rate),
        "fast_rate_fs-1": float(fast_rate),
        "slow_amplitude": float(amplitudes[0]),
        "fast_amplitude": float(amplitudes[1]),
        "bi_aic": float(count * np.log(max(bi_rss, np.finfo(float).tiny) / count) + 8.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="HDF5 paths or quoted glob patterns")
    parser.add_argument("--output", default="ensemble", help="Output prefix")
    parser.add_argument("--bootstrap", type=int, default=0, help="Trajectory bootstrap replicates")
    parser.add_argument("--seed", type=int, default=12345, help="Bootstrap seed")
    parser.add_argument("--skip-fit", action="store_true", help="Aggregate without kinetic fit")
    parser.add_argument("--no-plot", action="store_true", help="Do not create PNG plot")
    args = parser.parse_args()

    paths = expand_paths(args.paths)
    records = [record for path in paths for record in read_observable_records(path)]
    time = records[0][0]
    for record in records[1:]:
        if record[0].shape != time.shape or not np.allclose(record[0], time, rtol=0.0, atol=1.0e-10):
            raise SystemExit("Time grid mismatch among trajectory records")
    pop1 = np.stack([record[1] for record in records])
    pop2 = np.stack([record[2] for record in records])
    count = pop1.shape[0]
    mean1 = np.mean(pop1, axis=0)
    mean2 = np.mean(pop2, axis=0)
    if count > 1:
        sem1 = np.std(pop1, axis=0, ddof=1) / np.sqrt(count)
        sem2 = np.std(pop2, axis=0, ddof=1) / np.sqrt(count)
    else:
        sem1 = np.zeros_like(mean1)
        sem2 = np.zeros_like(mean2)

    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_fs", "reactant_mean", "reactant_sem", "product_mean", "product_sem"))
        writer.writerows(zip(time, mean1, sem1, mean2, sem2))

    fit = None if args.skip_fit else bounded_fit(time, mean2)
    bootstrap_rates: list[tuple[float, float]] = []
    if fit is not None and args.bootstrap > 0 and count > 1:
        rng = np.random.default_rng(args.seed)
        for _ in range(args.bootstrap):
            sample = rng.integers(0, count, count)
            try:
                trial = bounded_fit(time, np.mean(pop2[sample], axis=0), points=100)
                bootstrap_rates.append((trial["slow_rate_fs-1"], trial["fast_rate_fs-1"]))
            except ValueError:
                continue

    lines = [
        "# Ensemble analysis",
        "",
        f"- Trajectories: {count}",
        f"- Frames: {time.size}",
        f"- Time range: {time[0]:.6g} to {time[-1]:.6g} fs",
        f"- Final product population: {mean2[-1]:.8f} +/- {sem2[-1]:.8f} (SEM)",
    ]
    if fit is not None:
        lines += [
            "",
            "## Bounded kinetic fits",
            "",
            f"- Fast rate: {fit['fast_rate_fs-1']:.8g} fs^-1",
            f"- Slow rate: {fit['slow_rate_fs-1']:.8g} fs^-1",
            f"- Delta AIC (mono - bi): {fit['mono_aic'] - fit['bi_aic']:.6g}",
        ]
    if bootstrap_rates:
        rates = np.asarray(bootstrap_rates)
        slow_ci = np.percentile(rates[:, 0], (2.5, 97.5))
        fast_ci = np.percentile(rates[:, 1], (2.5, 97.5))
        lines += [
            f"- Successful bootstrap fits: {rates.shape[0]} / {args.bootstrap}",
            f"- Slow-rate 95% bootstrap interval: {slow_ci[0]:.8g} to {slow_ci[1]:.8g} fs^-1",
            f"- Fast-rate 95% bootstrap interval: {fast_ci[0]:.8g} to {fast_ci[1]:.8g} fs^-1",
        ]
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time, mean1, label="reactant")
        ax.plot(time, mean2, label="product")
        if count > 1:
            ax.fill_between(time, mean2 - sem2, mean2 + sem2, alpha=0.25)
        ax.set(xlabel="time (fs)", ylabel="diabatic population", ylim=(0.0, 1.02))
        ax.legend()
        fig.tight_layout()
        fig.savefig(prefix.with_suffix(".png"), dpi=180)
        plt.close(fig)

    print(f"Wrote {prefix.with_suffix('.csv')} and {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
