#!/usr/bin/env python3
"""Regression check for the bounded two-timescale population fit."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_ensemble import bounded_fit  # noqa: E402


time = np.linspace(0.0, 1000.0, 301)
product = 0.2 * (1.0 - np.exp(-0.05 * time))
product += 0.6 * (1.0 - np.exp(-0.002 * time))
fit = bounded_fit(time, product, points=180)

assert np.isclose(fit["fast_rate_fs-1"], 0.05, rtol=0.1)
assert np.isclose(fit["slow_rate_fs-1"], 0.002, rtol=0.1)
assert fit["bi_aic"] < fit["mono_aic"]

print("PASS bounded biexponential ensemble fit")
