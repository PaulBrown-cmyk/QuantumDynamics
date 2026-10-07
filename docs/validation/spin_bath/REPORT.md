# Exact finite spin-bath validation

Command:

```sh
python spin_bath_non_gaussian.py --spins 4 --duration 300 --frames 301 \
  --output advanced_models_20261007/spin_bath/exact_spin_bath.h5
```

| Metric | Value |
|---|---:|
| Bath spins | 4 |
| Maximum trace error | 4.4433516e-16 |
| Minimum density eigenvalue | 5.5219793e-18 |
| Trace-distance information backflow | 0.049377609 |
| Final product population | 0.54518814 |

Positive information backflow verifies finite-bath memory. It is not a continuum-
bath convergence claim. Raw HDF5 output remains local and is ignored by Git.
