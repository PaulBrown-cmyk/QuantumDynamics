# Advanced-model validation

Generated October 7, 2026 with final source tree.

## Quantum-FDT reduced density matrix

Command:

```sh
python quantum_fdt_density.py --nx 24 --temperature 100 --gamma 0.02 \
  --cutoff 0.2 --duration 200 --frames 101 \
  --output advanced_models_20261007/quantum_density.h5
```

- Maximum trace error: `1.110223025e-15`
- Minimum density-matrix eigenvalue: `-8.220011932e-16` (roundoff)
- Gibbs/KMS stationarity residual: `1.240770919e-24 au^-1`
- Final product population: `0.006214745247`

## Two-coordinate wavepacket

Command:

```sh
python multidimensional_transfer.py --nx 96 --ny 48 --dt 0.01 \
  --steps 1000 --save-every 20 \
  --output advanced_models_20261007/transfer_2d.h5
```

- Maximum norm drift: `1.381117443e-13`
- Final product population: `0.0001358328967`

These runs validate numerical invariants, not physical parameter convergence.
Use `convergence_campaign.py` before production interpretation.

## Automated convergence smoke campaign

Fifty-femtosecond product-damped campaign passed the requested `0.02` absolute
population tolerance:

- Half time step: maximum difference `0.00444228`
- Double grid: maximum difference `4.42e-14`
- Expanded box: maximum difference `3.57e-14`
- Double ensemble: maximum difference `0.00323362`; final SEM `0.00250407`

See [`convergence/REPORT.md`](convergence/REPORT.md).

## Controlled mechanism campaign

Four trajectories per control were run for 1.5 ps with four MPI ranks and
single-file Parallel HDF5. Over-barrier transfer passed all gates. Tunneling did
not: sub-barrier transfer failed the contamination threshold, and D transfer was
not suppressed relative to H. Therefore the old slow-rate=tunneling assignment
is not supported for this parameter set. See
[`mechanism/REPORT.md`](mechanism/REPORT.md).
