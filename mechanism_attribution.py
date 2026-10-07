#!/usr/bin/env python3
"""Controlled campaign for operational over-barrier/tunneling attribution."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess

import numpy as np

from analyze_ensemble import bounded_fit, read_observable_records, read_observables
from convergence_campaign import replace, value
from quantum_fdt_density import (
    AMU_TO_ELECTRON_MASS,
    ANGSTROM_TO_BOHR,
    CMINV_TO_HARTREE,
)


def surfaces(text: str, x_angstrom: np.ndarray, coupling: bool = True):
    k1 = value(text, "k1", float)
    k2 = value(text, "k2", float)
    x1 = value(text, "x1", float)
    x2 = value(text, "x2", float)
    shift1 = value(text, "v1_shift", float)
    shift2 = value(text, "v2_shift", float)
    try:
        c41 = value(text, "c4_1", float)
        c42 = value(text, "c4_2", float)
    except ValueError:
        c41 = c42 = 0.0
    v11 = 0.5 * k1 * (x_angstrom - x1) ** 2 + c41 * (x_angstrom - x1) ** 4 + shift1
    v22 = 0.5 * k2 * (x_angstrom - x2) ** 2 + c42 * (x_angstrom - x2) ** 4 + shift2
    if coupling:
        amplitude = value(text, "v12", float)
        sigma = value(text, "sigma", float)
        center = 0.5 * (x1 + x2)
        try:
            exponential = value(text, "use_exponential", bool)
        except ValueError:
            exponential = False
        if exponential:
            v12 = amplitude * np.exp(-np.abs(x_angstrom - center) / sigma)
        else:
            v12 = amplitude * np.exp(-0.5 * ((x_angstrom - center) / sigma) ** 2)
    else:
        v12 = np.zeros_like(x_angstrom)
    lower = 0.5 * (v11 + v22) - np.sqrt((0.5 * (v11 - v22)) ** 2 + v12**2)
    return v11, v22, v12, lower


def barrier_info(text: str) -> tuple[float, float, float]:
    xmin = value(text, "xmin", float)
    xmax = value(text, "xmax", float)
    x1 = value(text, "x1", float)
    x2 = value(text, "x2", float)
    x = np.linspace(xmin, xmax, 20001)
    _, _, _, lower = surfaces(text, x)
    left, right = sorted((x1, x2))
    between = (x >= left) & (x <= right)
    reactant = x <= 0.5 * (x1 + x2)
    reactant_min = float(np.min(lower[reactant]))
    barrier_absolute = float(np.max(lower[between]))
    return barrier_absolute - reactant_min, barrier_absolute, reactant_min


def classically_allowed_fraction(text: str, p0_a_inv: float, mass_amu: float,
                                 barrier_absolute_cminv: float, samples: int = 200000) -> float:
    rng = np.random.default_rng(41721)
    x0 = value(text, "x0", float)
    sigma = value(text, "sigma0", float)
    # For amplitude exp[-(x-x0)^2/(2 sigma^2)]: Wigner stds sigma/sqrt(2), 1/(sqrt(2)sigma).
    x = rng.normal(x0, sigma / np.sqrt(2.0), samples)
    p_mean = p0_a_inv / ANGSTROM_TO_BOHR
    p_std = 1.0 / (np.sqrt(2.0) * sigma * ANGSTROM_TO_BOHR)
    p = rng.normal(p_mean, p_std, samples)
    _, _, _, lower = surfaces(text, x)
    kinetic = p**2 / (2.0 * mass_amu * AMU_TO_ELECTRON_MASS) / CMINV_TO_HARTREE
    return float(np.mean(lower + kinetic >= barrier_absolute_cminv))


def load_case(case_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    paths = sorted(case_dir.glob("case.traj*.rank*.h5"))
    ensemble = case_dir / "case.ensemble.h5"
    if ensemble.exists():
        records = read_observable_records(ensemble)
    else:
        records = [read_observables(path) for path in paths]
    if not records:
        raise RuntimeError(f"no trajectories for {case_dir.name}")
    time = records[0][0]
    product = np.stack([record[2] for record in records])
    mean = np.mean(product, axis=0)
    sem = (np.std(product, axis=0, ddof=1) / np.sqrt(product.shape[0])
           if product.shape[0] > 1 else np.zeros_like(mean))
    return time, mean, sem, product.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--executable", type=Path, default=Path("./qle_1d"))
    parser.add_argument("--output", type=Path, default=Path("mechanism_campaign"))
    parser.add_argument("--trajectories", type=int, default=0,
                        help="override ntraj; at least four recommended")
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--parallel-hdf5", action="store_true",
                        help="use MPI-IO ensemble files; executable must support it")
    parser.add_argument("--min-trajectories", type=int, default=4)
    args = parser.parse_args()
    source = args.input.resolve().read_text()
    executable = args.executable.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    barrier, barrier_absolute, _ = barrier_info(source)
    mass_h = value(source, "mass_amu", float)
    p0 = value(source, "p0", float)
    sigma0 = value(source, "sigma0", float)
    x0 = value(source, "x0", float)
    lower_x0 = float(surfaces(source, np.array([x0]))[3][0])
    target_kinetic = max(200.0, barrier_absolute - lower_x0 + 200.0)
    high_p_au = np.sqrt(2.0 * mass_h * AMU_TO_ELECTRON_MASS * target_kinetic * CMINV_TO_HARTREE)
    high_p_input = high_p_au * ANGSTROM_TO_BOHR
    sigma_candidates = np.linspace(max(0.15, 0.6 * sigma0), 2.0 * sigma0, 15)
    sigma_fractions = [
        classically_allowed_fraction(replace(source, {"sigma0": float(sigma)}),
                                     0.0, mass_h, barrier_absolute, samples=50000)
        for sigma in sigma_candidates
    ]
    sub_sigma = float(sigma_candidates[int(np.argmin(sigma_fractions))])
    common = {
        "hdf5": True,
        "parallel_hdf5": args.parallel_hdf5,
        "restart_from_checkpoint": False,
        "checkpoint_every": 0,
        "out_prefix": "case",
    }
    if args.trajectories > 0:
        common["ntraj"] = args.trajectories
    definitions = [
        ("base", dict(common)),
        ("coupling_off", dict(common, want_coupling=False)),
        ("subbarrier_H", dict(common, p0=0.0, sigma0=sub_sigma, mass_amu=mass_h)),
        ("subbarrier_D", dict(common, p0=0.0, sigma0=sub_sigma, mass_amu=2.01410177812)),
        ("overbarrier", dict(common, p0=high_p_input, mass_amu=mass_h)),
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OMPI_MCA_btl", "self")
    for name, updates in definitions:
        case_dir = args.output / name
        case_dir.mkdir(parents=True, exist_ok=True)
        input_path = case_dir / "INPUT.nml"
        input_path.write_text(replace(source, updates))
        with (case_dir / "run.log").open("w") as log:
            command = [str(executable), str(input_path)]
            if args.mpi_ranks > 1:
                command = ["mpiexec", "-n", str(args.mpi_ranks), *command]
                env["OMPI_MCA_btl"] = "self,sm"
            subprocess.run(command, cwd=case_dir, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    rows = []
    series = {}
    for name, updates in definitions:
        time, mean, sem, trajectory_count = load_case(args.output / name)
        series[name] = (time, mean, sem)
        case_p0 = float(updates.get("p0", p0))
        case_mass = float(updates.get("mass_amu", mass_h))
        case_source = replace(source, {"sigma0": float(updates.get("sigma0", sigma0))})
        allowed = classically_allowed_fraction(case_source, case_p0, case_mass, barrier_absolute)
        early_index = int(np.searchsorted(time, min(50.0, time[-1])))
        early_index = min(early_index, time.size - 1)
        try:
            fit = bounded_fit(time, mean)
            slow_rate, fast_rate = fit["slow_rate_fs-1"], fit["fast_rate_fs-1"]
        except ValueError:
            slow_rate = fast_rate = np.nan
        rows.append((name, trajectory_count, allowed, mean[early_index], mean[-1],
                     sem[-1], slow_rate, fast_rate))

    table = {row[0]: row for row in rows}
    # row: name,count,allowed,early,final,sem,slow,fast
    ensemble_ready = all(row[1] >= args.min_trajectories for row in rows)
    base_signal = max(table["base"][4], np.finfo(float).eps)
    coupling_clean = table["coupling_off"][4] + 2.0 * table["coupling_off"][5] < max(
        1.0e-6, 0.01 * (base_signal - 2.0 * table["base"][5])
    )
    sub_allowed = table["subbarrier_H"][2] < 1.0e-2
    sub_transfer = table["subbarrier_H"][4] - 2.0 * table["subbarrier_H"][5] > max(
        1.0e-5, 5.0 * (table["coupling_off"][4] + 2.0 * table["coupling_off"][5]),
        5.0 * table["subbarrier_H"][2]
    )
    isotope_suppression = (
        table["subbarrier_D"][4] + 2.0 * table["subbarrier_D"][5]
        < 0.8 * (table["subbarrier_H"][4] - 2.0 * table["subbarrier_H"][5])
    )
    over_allowed = table["overbarrier"][2] > 0.5
    over_prompt = (
        table["overbarrier"][3] - 2.0 * table["overbarrier"][5]
        > 1.25 * (table["subbarrier_H"][3] + 2.0 * table["subbarrier_H"][5])
    )
    tunneling = ensemble_ready and coupling_clean and sub_allowed and sub_transfer and isotope_suppression
    overbarrier = ensemble_ready and coupling_clean and over_allowed and over_prompt
    if tunneling and overbarrier:
        conclusion = "mixed over-barrier and tunneling channels supported"
    elif tunneling:
        conclusion = "tunneling channel supported"
    elif overbarrier:
        conclusion = "over-barrier channel supported"
    else:
        conclusion = "inconclusive: extend duration/ensemble or revise controls"

    with (args.output / "evidence.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("case", "trajectories", "classically_allowed_fraction", "product_at_50fs",
                         "final_product", "final_sem", "slow_rate_fs-1", "fast_rate_fs-1"))
        writer.writerows(rows)
    report = f"""# Mechanism attribution

- Adiabatic barrier above reactant minimum: {barrier:.8g} cm^-1
- Preselected sub-barrier packet width: {sub_sigma:.8g} angstrom
- Ensemble threshold ({args.min_trajectories} trajectories per control) met: {ensemble_ready}
- Coupling-off background clean: {coupling_clean}
- Sub-barrier classical fraction below 1e-2: {sub_allowed}
- Sub-barrier transfer exceeds background and classical contamination by 5x: {sub_transfer}
- Deuterium suppression present: {isotope_suppression}
- Over-barrier classical fraction above 0.5: {over_allowed}
- Over-barrier prompt enhancement present: {over_prompt}

## Conclusion

**{conclusion}.**

This label is definitive only under this numerical model and these preregistered
controls. It is not a model-independent experimental mechanism claim.
"""
    (args.output / "REPORT.md").write_text(report)
    print(conclusion)
    print(f"Wrote {args.output / 'evidence.csv'} and {args.output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
