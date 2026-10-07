# QuantumDynamics

One-dimensional quantum dynamics for H-atom transfer between two coupled
diabatic potential wells. Wavepackets propagate with a split-operator method.
Reactant and product wells support stochastic coordinate-shift, curvature, and
enthalpy modulation with independent or correlated noise. A Markovian or
Lorentz-colored Langevin/GLE bath thermostats the nuclear wavepacket centroid
using classical fluctuation-dissipation statistics. `damp_reactant` and
`damp_product` select which diabatic state receives momentum damping; enabling
both retains shared-centroid behavior.

Framework equations, controls, verified capabilities, production guidance, and
remaining limitations are collected in
[`docs/framework_capabilities.tex`](docs/framework_capabilities.tex) and the
compiled [`docs/QuantumDynamics_Framework_Notes.pdf`](docs/QuantumDynamics_Framework_Notes.pdf).

Input from `INPUT.nml` uses chemistry-facing units: lengths in angstrom, times
in femtoseconds, energies/couplings in cm^-1, rates in fs^-1, temperature in
kelvin, mass in u/amu, and wave numbers in 1/angstrom. Parser validates these
values and converts them to atomic units immediately. ASCII and HDF5 writers
convert coordinates, time, wavefunctions/densities, expectation values, and
potentials back to physical units.

Quartic inputs `c4_1` and `c4_2` use cm^-1/angstrom^4. Energy shifts remain
independent cm^-1 values; they are no longer overloaded as quartic coefficients.
`k1` and `k2` define well stiffness; obsolete no-op `w0` input was removed.

`temperature_k` and `mass_amu` are physical inputs. `beta` and atomic-unit mass
are derived internally and must not appear in `INPUT.nml`. This replaces old
internal-unit `beta` input and fixed electron-mass propagation.

Build with `make`; disable HDF5 support with `make USE_HDF5=0`. Run with
`./qle_1d [INPUT.nml]`; `--help` never starts a simulation. At runtime,
`hdf5=.false.` selects ASCII output even in HDF5-enabled builds. Run unit,
FFTW, Langevin, and output smoke tests with `make test`.

`write_initial=.true.` records an exact `t=0` frame. Optional smooth edge
absorption uses `use_absorber`, `absorber_width` (angstrom), `absorber_rate`
(fs^-1), and `absorber_power`; absorbed norm represents outgoing flux.

Random streams depend on trajectory number, not MPI rank, so changing rank
count does not change a trajectory's stochastic sequence. Static ASCII PES
uses `*.pes.dat`; stochastic PES snapshots use `*.sNNNNNN.pes.dat`.

Aggregate MPI/rank trajectory files with `analyze_ensemble.py`. It writes
population means/SEM and optionally bounded mono-/biexponential fits with
trajectory-bootstrap rate intervals.

HDF5 writes static potential surfaces only in first frame; stochastic surfaces
remain frame-resolved. `ShowDyn.py` reuses single stored static surface for all frames.

## Checkpoint/restart

Set `checkpoint_every` to a positive step count. Each trajectory gets an atomically
replaced, schema-versioned HDF5 checkpoint containing both wavefunctions, intrinsic
Fortran RNG state, Langevin auxiliary memory, stochastic-potential state, time, and
output indices. Resume with the same grid and time step using
`restart_from_checkpoint=.true.`; `nsteps` is the desired final step. Regression
tests require bitwise identity with uninterrupted propagation. Deterministic
`FFTW_ESTIMATE` plans make restart results reproducible under the same numerical stack.

## Parallel single-file HDF5

True MPI-IO output requires Parallel HDF5:

```sh
make USE_PARALLEL_HDF5=1 HDF5_PREFIX="$(brew --prefix hdf5-mpi)"
make test-parallel
```

Set `parallel_hdf5=.true.`. All MPI ranks write nonoverlapping trajectory/frame
hyperslabs to `PREFIX.ensemble.h5`; no rank-local trajectory files are produced.
`analyze_ensemble.py` reads both legacy per-trajectory files and this ensemble layout.
Checkpoint restart remains a per-trajectory-file mode so crash recovery never depends
on partially committed collective metadata.

## Quantum bath and density matrices

`quantum_fdt_density.py` propagates the complete finite-grid reduced density matrix
with a completely positive secular Davies generator. Drude-Ohmic absorption and
emission rates satisfy the KMS quantum detailed-balance relation at every resolved
Bohr frequency and converge to the Gibbs state. This is the controlled weak-coupling,
Markovian quantum-FDT model; the Langevin executable remains the classical-FDT
centroid model.

```sh
python quantum_fdt_density.py --temperature 100 --gamma 0.02 \
  --cutoff 0.2 --duration 200 --output quantum_density.h5
```

## Multidimensional transfer

`multidimensional_transfer.py` supplies a two-coordinate, two-diabatic-state FFT
split-operator solver. Coordinate one is H transfer; coordinate two is a coupled
promoting/bath mode with independent mass, shifted wells, and reaction-path coupling.
It writes physical-unit 2D densities, surfaces, populations, and time to HDF5.

## Automated evidence campaigns

`convergence_campaign.py` runs time-step, grid, box, absorber, and ensemble-size
variants and writes machine-readable and Markdown pass/fail summaries.

`mechanism_attribution.py` preregisters coupling-off, sub-barrier H, sub-barrier D,
and over-barrier controls. It reports classical Wigner above-barrier fractions,
isotope suppression, prompt transfer, bounded rates, SEM, and coupling background.
Mechanism labels are issued only when all evidence thresholds pass; otherwise result
is explicitly inconclusive.
