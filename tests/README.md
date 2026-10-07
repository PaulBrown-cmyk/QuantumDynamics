Dissipation regressions
======================

Run `./tests/run.sh` with gfortran and FFTW3 installed, or `make test` with
MPIFC set to a compatible compiler. Stochastic regressions use a fixed seed,
checked bounds, and one OpenMP thread; output smoke tests use two threads.
They exercise the production bath, split propagator, and writers:

* gamma=0: two-component, unrenormalized norm after 100 free split steps.
* White deterministic wavepacket mean: exp(-gamma*t).
* Product-only damping: unchanged reactant momentum and relaxed product momentum.
* Colored deterministic wavepacket mean: analytic exponential-memory solution.
* Independent white and colored centroid ensembles: zero equilibrium mean and
  variance mass/beta (12,000 independent trajectories, 0.025 absolute tolerance).
* Colored stochastic-potential initialization: stationary mean and variance.
* Odd-sized FFT grids: correct positive/negative frequency ordering.
* Physical-unit conversion for length, time, energy, rate, mass, temperature,
  wave number, force constants, and every potential-bath amplitude mode.
* CLI help safety, explicit input path, dynamic ASCII PES output, and HDF5 smoke
  output when HDF5 and Python h5py are available.

The implemented model is a classical-FDT centroid thermostat. It preserves each
pure-state trajectory's norm. With both damping switches enabled, one shared kick
acts on both amplitudes. Selective mode uses the chosen diabatic component's
conditional centroid momentum and applies its kick only to that component. It
does not thermalize intrinsic wavepacket momentum variance or implement
quantum-frequency-dependent FDT or a density-matrix master equation.

White noise uses the exact OU centroid update. For colored noise, with
r=y-z and omega^2=gamma/tau, a half rotation of (p,r), an exact OU step,
and a half rotation integrate dp=-r, dy=(gamma*p-y)/tau, dz=-z/tau+noise.
Both subflows preserve the stationary covariance
Var(p)=mass/beta, Var(r)=mass*gamma/(beta*tau), Cov(p,r)=0.
The colored deterministic response is second-order in dt; white free relaxation
is exact. y starts at zero; z starts in its stationary Gaussian distribution.
The outer split is bath(dt/2), Hamiltonian(dt), bath(dt/2).

Momentum translations exp(i*delta_p*x) assume the packet is resolved in momentum
and negligible at the periodic box edges. Increase box/grid size if it reaches
edges or Nyquist momentum; these kicks cannot avoid finite-grid aliasing.

Build fixes included: .f08 compiler language selection, serial module ordering,
module-scoped FFT thread wrappers, and dedicated aligned FFTW plans for
both wavefunction components. HDF5-off build: make USE_HDF5=0.

Validation on 2026-10-04
------------------------
GNU Fortran 16.2.0, FFTW 3.3.11, OpenMPI 5.0.10, and HDF5 2.2.0
(macOS/Apple Silicon). Full HDF5 and HDF5-disabled executables compiled and
linked. Strict `-Wall -Wextra -Wimplicit-interface` build completed without
warnings. ASCII and HDF5 physical-unit round trips passed. Checked results:

    gamma=0 norm              1.0000000000000029
    white mean momentum       0.1353352832366140 (target 0.1353352832366127)
    colored mean momentum     0.0695923160846613 (target 0.0696104692927771)
    white ensemble mean/var   0.0019632990 / 0.4975866070
    colored ensemble mean/var 0.0067755961 / 0.5104215296
    coupling switches         pass
    quartic/shift separation  pass
    ASCII/HDF5 output units   pass

Audit: the former main call omitted pbar, suppressing both white friction and
colored y driving. Scalar exp(-gamma*dt) damping was exactly removed by
normalization. The former y update also contained an extra 1/tau factor:
holding p constant gives y_new=a*y+gamma*(1-a)*p, not
(a*y+gamma/tau*(1-a)*p). Endpoint-only colored-force impulses and Euler white
friction also lacked a consistent finite-step equilibrium covariance.
