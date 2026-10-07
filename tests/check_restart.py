#!/usr/bin/env python3
"""Require bitwise-identical uninterrupted and checkpointed trajectories."""

import sys

import h5py
import numpy as np


with h5py.File(sys.argv[1], "r") as full, h5py.File(sys.argv[2], "r") as resumed:
    assert sorted(full) == sorted(resumed) == [
        "step_000000", "step_000001", "step_000002"
    ]
    for step in sorted(full):
        for name in ("t", "psi1", "psi2", "pop1", "pop2", "xavg1", "xavg2"):
            assert np.array_equal(full[step][name][...], resumed[step][name][...]), (
                step, name
            )

with h5py.File(sys.argv[3], "r") as checkpoint:
    assert int(checkpoint["schema_version"][0]) == 1
    assert int(checkpoint["tstep"][0]) == 8
    assert "rng_state" in checkpoint

print("PASS exact checkpoint/restart continuation")
