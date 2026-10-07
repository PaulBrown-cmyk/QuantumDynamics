#!/usr/bin/env python3
"""Automated hierarchy, bath-pole, and system-basis convergence for HEOM."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from heom_nonmarkovian import HEOMConfig, simulate_model
from quantum_fdt_density import Model


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=Path("heom_convergence"))
    result.add_argument("--nx", type=int, default=10)
    result.add_argument("--states", type=int, default=8)
    result.add_argument("--depth", type=int, default=3)
    result.add_argument("--matsubara", type=int, default=2)
    result.add_argument("--temperature", type=float, default=300.0)
    result.add_argument("--reorganization", type=float, default=500.0, help="cm^-1")
    result.add_argument("--cutoff", type=float, default=0.04, help="fs^-1")
    result.add_argument("--duration", type=float, default=60.0, help="fs")
    result.add_argument("--frames", type=int, default=61)
    result.add_argument("--tolerance", type=float, default=0.02)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.states + 2 > 2 * args.nx:
        raise ValueError("states+2 must not exceed full finite-grid basis")
    model = Model(nx=args.nx)
    base = HEOMConfig(
        temperature_k=args.temperature,
        reorganization_cminv=args.reorganization,
        cutoff_fs_inv=args.cutoff,
        matsubara_terms=args.matsubara,
        hierarchy_depth=args.depth,
        system_states=args.states,
    )
    variants = {
        "baseline": base,
        "depth_plus_one": replace(base, hierarchy_depth=base.hierarchy_depth + 1),
        "matsubara_plus_one": replace(base, matsubara_terms=base.matsubara_terms + 1),
        "states_plus_two": replace(base, system_states=base.system_states + 2),
    }
    results = {
        name: simulate_model(model, config, args.duration, args.frames)
        for name, config in variants.items()
    }
    reference = np.asarray(results["baseline"]["pop2"])
    rows: list[dict[str, object]] = []
    all_pass = True
    for name, config in variants.items():
        result = results[name]
        difference = float(np.max(np.abs(np.asarray(result["pop2"]) - reference)))
        invariant_ok = (
            float(result["max_trace_error"]) < 2.0e-7
            and float(result["max_hermiticity_error"]) < 2.0e-7
            and float(result["min_density_eigenvalue"]) > -2.0e-5
        )
        convergence_ok = name == "baseline" or difference <= args.tolerance
        passed = invariant_ok and convergence_ok
        all_pass = all_pass and passed
        rows.append(
            {
                "case": name,
                "depth": config.hierarchy_depth,
                "matsubara_terms": config.matsubara_terms,
                "system_states": config.system_states,
                "ado_count": result["ado_count"],
                "max_product_difference": difference,
                "final_product": float(np.asarray(result["pop2"])[-1]),
                "max_trace_error": result["max_trace_error"],
                "min_density_eigenvalue": result["min_density_eigenvalue"],
                "passed": passed,
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "convergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "REPORT.md").open("w") as stream:
        stream.write("# HEOM convergence\n\n")
        stream.write(f"- Overall pass: **{all_pass}**\n")
        stream.write(f"- Population tolerance: `{args.tolerance}`\n")
        stream.write(f"- Bath memory time: `{1.0 / args.cutoff:.8g} fs`\n")
        stream.write(
            f"- Baseline reorganization/gap ratio: "
            f"`{results['baseline']['reorganization_to_gap_ratio']:.8g}`\n"
        )
        stream.write(
            f"- Baseline maximum auxiliary norm: "
            f"`{results['baseline']['max_auxiliary_norm']:.8g}`\n\n"
        )
        stream.write("| Case | Depth | Matsubara | States | ADOs | Max dP | Pass |\n")
        stream.write("|---|---:|---:|---:|---:|---:|:---:|\n")
        for row in rows:
            stream.write(
                f"| {row['case']} | {row['depth']} | {row['matsubara_terms']} | "
                f"{row['system_states']} | {row['ado_count']} | "
                f"{row['max_product_difference']:.8g} | {row['passed']} |\n"
            )
    print(f"HEOM convergence pass: {all_pass}")
    print(f"Wrote {args.output / 'convergence.csv'} and {args.output / 'REPORT.md'}")
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
