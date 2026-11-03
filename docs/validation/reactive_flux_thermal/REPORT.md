# Reactive-flux, transmission, and thermal-initial-state validation

Validation date: 2026-11-03.

## Deterministic analytic checks

- Periodic plane-wave current matches `k/(m L)` to relative tolerance 2e-12.
- Exactly propagated free Gaussian gives forward integrated flux above 0.98.
- Backward recrossing remains below 1e-3.
- Flux-integrated and directly integrated side populations agree within 4e-3.
- Continuity RMS is below 2e-4 fs^-1.
- Incident/product surface Fourier analysis recovers integrated free-particle
  transmission within 0.08 of unity; median resolved `T(E)` lies from 0.85 to 1.15.

## Thermal initial states

- 1 K canonical populations recover ground-state occupation to 1e-12.
- 1e12 K populations recover the uniform finite-basis limit to 1e-8.
- A 12000-sample stochastic thermal-pure estimate at 300 K has Frobenius error
  below 0.025 and maximum population error below 0.015.
- Exact Gibbs and localized pure-state density matrices retain unit trace and
  positive semidefiniteness.
- A 4000-sample, 100 K validation produced effective sample size 2230.65,
  trace-estimator error 0.00687, density Frobenius error 0.01061, and maximum
  population error 0.000321.

## Dissipative-trajectory rejection test

Applying energy-resolved analysis to the 5 ps product-damped example produced
integrated apparent transmission 2.06, recrossing fraction 0.915, and no accepted
rate plateau. This intentionally demonstrates validity gating: such a damped,
recurrent trajectory is not an asymptotic scattering experiment, so `T(E)` is not
interpreted as a physical transmission probability. The tool emits a warning.
