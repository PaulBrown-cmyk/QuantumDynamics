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

## Strong-coupling non-Markovian HEOM

Command:

```sh
python heom_nonmarkovian.py --nx 12 --states 10 --depth 4 --matsubara 2 \
  --temperature 300 --reorganization 500 --cutoff 0.04 \
  --duration 100 --frames 101 \
  --output advanced_models_20261007/heom/nonmarkovian.h5
```

- Drude bath memory time: `25 fs`
- Reorganization/median-gap ratio: `4.4742674` (strong coupling)
- Auxiliary density operators: `35`
- Maximum trace error: `4.4431660e-16`
- Maximum Hermiticity error: `3.4694470e-16`
- Minimum root-density eigenvalue: `-1.2427030e-16` (roundoff)
- Maximum auxiliary-memory norm: `0.96639610`
- Final product population: `0.0011382594`

Depth, Matsubara-pole, and system-basis convergence passed `0.02` population
tolerance. Largest change was `1.6993067e-4` from adding two retained system states;
depth and bath-pole changes were below `4.1e-8`. See
[`heom/convergence/REPORT.md`](heom/convergence/REPORT.md).

A separate 10 K run used eight Matsubara terms, hierarchy depth three, eight
retained system states, and 220 ADOs for 60 fs. Maximum trace error was
`3.3307014e-16`, minimum root-density eigenvalue was `-2.9908618e-11`, maximum
auxiliary norm was `0.91704674`, and final product population was `0.00015677036`.
Depth four, a ninth Matsubara pole, and two extra system states all passed the
`0.02` tolerance; largest population change was `3.0453647e-4`. See
[`heom/convergence_10K/REPORT.md`](heom/convergence_10K/REPORT.md).

## Correlated HEOM preparation

```sh
python heom_nonmarkovian.py --nx 8 --states 6 --depth 2 --matsubara 1 \
  --duration 10 --frames 11 --initial-preparation correlated-projected \
  --output advanced_models_20261007/correlated_heom/correlated_projected.h5
```

- Stationary hierarchy residual: `1.4595641e-18`
- Initial correlated auxiliary norm: `1.0901055`
- Maximum trace error: `2.2205775e-16`
- Minimum density eigenvalue: `-3.4189106e-20` (roundoff)

See [`correlated_heom/REPORT.md`](correlated_heom/REPORT.md).

## Exact non-Gaussian spin bath

Four-spin, 300 fs exact propagation:

- Maximum trace error: `4.4433516e-16`
- Minimum density eigenvalue: `5.5219793e-18`
- Trace-distance information backflow: `0.049377609`
- Final product population: `0.54518814`

See [`spin_bath/REPORT.md`](spin_bath/REPORT.md). Positive information backflow
verifies finite-bath memory; it is not a continuum-bath convergence claim.

## Three-coordinate transfer and PES ingestion

A 20 by 10 by 8 grid propagated for 0.5 fs with maximum norm drift
`1.2212453e-14` and final product population `3.8554462e-6`. See
[`transfer_3d/REPORT.md`](transfer_3d/REPORT.md). Regression also round-trips an analytic surface through
the unit-annotated external PES schema and reproduces all three diabatic surfaces to
`2e-15` hartree.
