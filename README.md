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
