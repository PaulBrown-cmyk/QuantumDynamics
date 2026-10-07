#!/usr/bin/env python3
"""Validate MPI-IO single-file ensemble layout and physical units."""

import sys

import h5py
import numpy as np


with h5py.File(sys.argv[1], "r") as handle:
    assert handle["psi1_real"].shape == (3, 3, 32)
    assert handle["psi1_imag"].shape == (3, 3, 32)
    assert handle["pop1"].shape == (3, 3)
    assert handle["time_fs"].shape == (3,)
    assert np.allclose(handle["time_fs"][...], (0.0, 0.01, 0.02), atol=1.0e-12)
    assert handle["x"].shape == (32,)
    assert np.isclose(handle["x"][0], -2.90625, atol=1.0e-12)
    assert "V11_cm-1" in handle and handle["V11_cm-1"].shape == (32,)
    total = handle["pop1"][...] + handle["pop2"][...]
    assert np.allclose(total, 1.0, rtol=0.0, atol=2.0e-11)
    assert np.all(np.isfinite(handle["psi1_real"][...]))

print("PASS parallel single-file HDF5 MPI-IO output")
