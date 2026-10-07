#!/usr/bin/env python3
"""Run reproducible dt/grid/box/absorber/ensemble convergence campaign."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from analyze_ensemble import read_observables


def value(text: str, key: str, cast):
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*([^,!/]+)", text)
    if not match:
        raise ValueError(f"missing namelist key: {key}")
    raw = match.group(1).strip()
    if cast is bool:
        return raw.lower() in (".true.", "true", "t")
    return cast(raw)


def replace(text: str, updates: dict[str, object]) -> str:
    result = text
    for key, new_value in updates.items():
        if isinstance(new_value, bool):
            rendered = ".true." if new_value else ".false."
        elif isinstance(new_value, str):
            rendered = f"'{new_value}'"
        else:
            rendered = str(new_value)
        pattern = rf"(?im)^(\s*{re.escape(key)}\s*=\s*)[^,!/]+"
        result, count = re.subn(pattern, rf"\g<1>{rendered}", result, count=1)
        if count == 0:
            slash = result.rfind("/")
            result = result[:slash] + f"  {key} = {rendered},\n" + result[slash:]
    return result


def cases(text: str) -> list[tuple[str, dict[str, object]]]:
    nx = value(text, "nx", int)
    xmin = value(text, "xmin", float)
    xmax = value(text, "xmax", float)
    dt = value(text, "dt", float)
    nsteps = value(text, "nsteps", int)
    save_every = value(text, "save_every", int)
    ntraj = value(text, "ntraj", int)
    width = xmax - xmin
    center = 0.5 * (xmin + xmax)
    common = {
        "hdf5": True,
        "parallel_hdf5": False,
        "restart_from_checkpoint": False,
        "checkpoint_every": 0,
    }
    result = [
        ("baseline", dict(common)),
        ("dt_half", dict(common, dt=0.5 * dt, nsteps=2 * nsteps,
                         save_every=2 * save_every)),
        ("grid_double", dict(common, nx=2 * nx)),
        ("box_expand", dict(common, xmin=center - 0.625 * width,
                            xmax=center + 0.625 * width,
                            nx=max(nx + 2, int(round(1.25 * nx))))),
        ("ensemble_double", dict(common, ntraj=2 * ntraj)),
    ]
    try:
        if value(text, "use_absorber", bool):
            absorber_rate = value(text, "absorber_rate", float)
            absorber_width = value(text, "absorber_width", float)
            result.extend([
                ("absorber_half_rate", dict(common, absorber_rate=0.5 * absorber_rate)),
                ("absorber_wide", dict(common, absorber_width=1.25 * absorber_width)),
            ])
    except ValueError:
        pass
    return result


def load_mean(case_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted(case_dir.glob("case.traj*.rank*.h5"))
    if not paths:
        raise RuntimeError(f"no HDF5 trajectories in {case_dir}")
    records = [read_observables(path) for path in paths]
    time = records[0][0]
    product = np.stack([record[2] for record in records])
    if any(record[0].shape != time.shape or not np.allclose(record[0], time) for record in records):
        raise RuntimeError(f"time-grid mismatch in {case_dir}")
    mean = np.mean(product, axis=0)
    sem = (np.std(product, axis=0, ddof=1) / np.sqrt(product.shape[0])
           if product.shape[0] > 1 else np.zeros_like(mean))
    return time, mean, sem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--executable", type=Path, default=Path("./qle_1d"))
    parser.add_argument("--output", type=Path, default=Path("convergence"))
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="maximum absolute product-population difference")
    args = parser.parse_args()

    source = args.input.resolve().read_text()
    executable = args.executable.resolve()
    if not executable.is_file():
        raise SystemExit(f"Executable not found: {executable}")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    definitions = cases(source)
    for name, updates in definitions:
        case_dir = args.output / name
        case_dir.mkdir(parents=True, exist_ok=True)
        updates["out_prefix"] = "case"
        input_path = case_dir / "INPUT.nml"
        input_path.write_text(replace(source, updates))
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OMPI_MCA_btl", "self")
        with (case_dir / "run.log").open("w") as log:
            subprocess.run([str(executable), str(input_path)], cwd=case_dir, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    base_t, base_mean, base_sem = load_mean(args.output / "baseline")
    rows = []
    for name, _ in definitions:
        time, mean, sem = load_mean(args.output / name)
        aligned = np.interp(base_t, time, mean)
        max_difference = float(np.max(np.abs(aligned - base_mean)))
        final_difference = float(abs(aligned[-1] - base_mean[-1]))
        rows.append((name, len(list((args.output / name).glob("case.traj*.rank*.h5"))),
                     max_difference, final_difference, float(sem[-1]),
                     max_difference <= args.tolerance))

    with (args.output / "convergence.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("case", "trajectories", "max_abs_product_difference",
                         "final_abs_product_difference", "final_sem", "within_tolerance"))
        writer.writerows(rows)
    lines = ["# Convergence campaign", "", f"Tolerance: {args.tolerance:.6g}", "",
             "| Case | Trajectories | Max difference | Final difference | Final SEM | Pass |",
             "|---|---:|---:|---:|---:|:---:|"]
    for row in rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]:.6g} | {row[3]:.6g} | "
                     f"{row[4]:.6g} | {'yes' if row[5] else 'no'} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output / 'convergence.csv'} and {args.output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
