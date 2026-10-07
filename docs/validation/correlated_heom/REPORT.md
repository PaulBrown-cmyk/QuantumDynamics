# Correlated HEOM validation

Command:

```sh
python heom_nonmarkovian.py --nx 8 --states 6 --depth 2 --matsubara 1 \
  --duration 10 --frames 11 --initial-preparation correlated-projected \
  --output advanced_models_20261007/correlated_heom/correlated_projected.h5
```

| Metric | Value |
|---|---:|
| ADO count | 6 |
| Retained initial norm | 0.9998879536 |
| Stationary hierarchy residual | 1.4595641e-18 |
| Initial correlated auxiliary norm | 1.0901055 |
| Maximum trace error | 2.2205775e-16 |
| Maximum Hermiticity error | 2.7097247e-17 |
| Minimum density eigenvalue | -3.4189106e-20 |

Negative eigenvalue is roundoff-scale. Raw HDF5 output remains local and is ignored
by Git.
