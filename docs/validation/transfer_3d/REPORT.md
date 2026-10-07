# Three-coordinate transfer validation

Command:

```sh
python multidimensional_transfer_3d.py --nx 20 --ny 10 --nz 8 \
  --steps 50 --save-every 10 \
  --output advanced_models_20261007/transfer_3d/transfer_3d.h5
```

| Metric | Value |
|---|---:|
| Propagation time | 0.5 fs |
| Maximum norm drift | 1.2212453e-14 |
| Final product population | 3.8554462e-6 |

Regression additionally round-trips analytic surfaces through the external HDF5 PES
schema and reproduces V11, V22, and complex V12 to 2e-15 hartree. Incorrect unit
metadata is rejected. Raw HDF5 output remains local and is ignored by Git.
