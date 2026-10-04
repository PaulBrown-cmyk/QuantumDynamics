#!/usr/bin/env python3
import sys

import h5py
import numpy as np


def as_complex(array):
    array = np.asarray(array)
    if np.iscomplexobj(array):
        return array
    if array.ndim == 2 and array.shape[-1] == 2:
        return array[:, 0] + 1j * array[:, 1]
    if array.ndim == 2 and array.shape[0] == 2:
        return array[0, :] + 1j * array[1, :]
    raise AssertionError(f"unexpected complex-array shape {array.shape}")


with h5py.File(sys.argv[1], "r") as handle:
    assert sorted(handle) == ["step_000001", "step_000002"]
    for index, name in enumerate(sorted(handle), start=1):
        group = handle[name]
        x = np.asarray(group["x"])
        psi1 = as_complex(group["psi1"])
        psi2 = as_complex(group["psi2"])
        time_fs = float(np.asarray(group["t"]).reshape(-1)[0])
        assert np.isclose(time_fs, 0.01 * index, rtol=0.0, atol=1.0e-12)
        assert np.isclose(x[0], -3.96875, rtol=0.0, atol=1.0e-12)
        assert np.isclose(x[-1], 3.96875, rtol=0.0, atol=1.0e-12)
        norm = np.sum(np.abs(psi1) ** 2 + np.abs(psi2) ** 2) * (x[1] - x[0])
        assert np.isclose(norm, 1.0, rtol=0.0, atol=2.0e-11)
        for dataset in ("V11", "V22", "V12", "V_lower", "V_upper"):
            assert np.all(np.isfinite(group[dataset][...]))

print("PASS HDF5 physical-unit output")
