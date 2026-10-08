# Future development roadmap

Items 1-4 are preferred starting candidates. They strengthen quantitative rate
and mechanism analysis without replacing the present model architecture.

## Near-term priorities

- [ ] **1. Reactive-flux rate theory.** Compute state-resolved probability current
  at configurable dividing surfaces, flux-side and flux-flux correlations, plateau
  rates, recrossing diagnostics, and absorber-resolved outgoing yield. Validate
  current continuity and agreement between integrated flux and population change.

- [ ] **2. Energy-resolved transmission.** Fourier transform windowed outgoing
  flux or wavepacket correlations to obtain transmission probabilities, cumulative
  reaction probability, resonance energies, and tunneling windows. Test against
  analytically soluble barriers and energy-integrated flux.

- [ ] **3. Thermal initial states.** Support canonical mixtures of stationary
  states, stochastic thermal wavepackets, and density-matrix initialization.
  Report effective sample size and thermal trace error; recover zero-temperature
  and high-temperature limits.

- [ ] **4. Automated H/D/T isotope campaigns.** Generate mass-matched input sets,
  propagate common physical controls, bootstrap kinetic isotope effects, and test
  stability against grid, time-step, duration, ensemble size, and classical
  over-barrier contamination.

## Additional physics and analysis

- [ ] **5. Phase-space observables.** Add integrated Wigner negativity,
  phase-space current, conditional momentum, coherence lifetimes, and positive
  Husimi-Q visualization.

- [ ] **6. Quantum-jump trajectories.** Unravel Davies/Lindblad dynamics into
  stochastic pure-state trajectories and verify ensemble agreement with the
  density-matrix solver.

- [ ] **7. Nonsecular Redfield dynamics.** Retain near-degenerate population-
  coherence coupling, while reporting trace, Hermiticity, equilibrium error, and
  any positivity violations. Compare with Davies and HEOM regimes.

- [ ] **8. Structured spectral densities.** Add multiple Drude poles,
  underdamped Brownian modes, and unit-checked tabulated spectral densities.

- [ ] **9. Time-dependent driving.** Support pulsed or periodic bias and coupling
  modulation for pump-probe, Floquet-assisted transfer, and coherent-control tests.

- [ ] **10. Richer diabatic Hamiltonians.** Add coordinate-dependent non-Condon
  coupling, Morse and multi-barrier surfaces, asymmetric effective masses, and
  three or more coupled electronic basis states.

- [ ] **11. Proton-coupled electron transfer.** Couple proton motion to an
  explicit donor-acceptor or solvent coordinate and electronic-state manifold.

- [ ] **12. Conical intersections and geometric phase.** Extend multidimensional
  complex couplings to wavepacket branching, Berry phase, and seam-crossing tests.

- [ ] **13. Alternative propagators.** Add Chebyshev, Krylov, DVR, and higher-order
  split methods for independent accuracy checks and reduced periodic-box artifacts.

- [ ] **14. Uncertainty quantification.** Automate global sensitivity, parameter
  ensembles, bootstrap or Bayesian rate intervals, and surrogate-assisted fitting.

- [ ] **15. Embedded provenance.** Store source revision, compiler and dependency
  versions, build flags, input, seeds, convergence evidence, and movie metadata in
  HDF5 outputs.

## Meaning of electronic states

Here, an electronic state means one basis channel in the coupled nuclear
Hamiltonian. Current one-dimensional model propagates a two-component nuclear
wavepacket over the 2 x 2 diabatic matrix formed by `V11`, `V22`, and `V12`.
Adding more electronic states means propagating more nuclear components over an
N x N potential/coupling matrix.

Electronic-structure theory is optional but provides a physical route to those
matrix elements. Surfaces and couplings may remain analytic model functions, or
they may be generated externally with DFT, multireference, coupled-cluster, or
other quantum-chemistry methods and imported through the existing unit-checked
PES interface. Reliable import requires consistent state tracking, phase/gauge,
diabatization, interpolation, and uncertainty metadata.

This repository does not currently perform SCF, DFT, correlated electronic-
structure, or on-the-fly force calculations. Recommended progression is:

1. Offline import of precomputed adiabatic or diabatic surfaces and couplings.
2. Reproducible adapters to external electronic-structure packages.
3. Only later, on-the-fly nonadiabatic dynamics; this is a much larger framework.
