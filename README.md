# QuantumDynamics
Codebase for the simulation of quantum dynamics within the harmonic oscillator approximation. 
Two diabatic potentials are dynamically coupled are set the reatant and product potential. 
The wavepacket propagation is handled with the Split-operator. Both reactant and product 
states can be perturbed in three ways: 1) force modulation, 2) enthalpic modulation, and 3) equilibrium 
shifts. Each perturbative effect is driven with random oscillations and modulate the 
barrier height between both states. Both states also contain Quantum Langevin equations (QGLE)
that handle include disspiation in both states. 

NOTE: Units should be Angstrom and wavenumbers, which needs to be fixed.
