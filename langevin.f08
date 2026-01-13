!=============================== langevin.f90 =================================
MODULE langevin
  !
  ! Generalized Langevin (memory-friction) helpers.
  !
  ! Target (per diabatic component j = 1,2):
  !
  !   dp_j/dt = -dV_j/dx  - ∫_0^t gamma_j(t-t') p_j(t') dt'  +  R_j(t)
  !
  ! with fluctuation–dissipation:
  !
  !   <R_j(t) R_j(t')> = (1/beta) * gamma_j(|t-t'|)
  !
  ! Exponential (Drude/Lorentz/Debye) memory kernel:
  !
  !   gamma(t) = (gamma/tau_c) * EXP(-t/tau_c),  t >= 0
  !
  ! Define auxiliary memory integral:
  !
  !   y(t) = ∫_0^t gamma(t-t') p(t') dt'
  !   dy/dt = (gamma/tau_c) p - y/tau_c
  !
  ! We propagate an OU random force R(t)=z(t) with
  !   <z(t) z(t')> = (1/beta) * gamma(|t-t'|)
  !
  USE kinds
  USE params
  USE rng
  IMPLICIT NONE

  TYPE:: LangevinState
    LOGICAL :: enabled = .false.
    LOGICAL :: colored = .false.
    REAL(dp):: gamma = 0.0_dp
    REAL(dp):: tau_c = 0.0_dp  ! correlation time (a.u.) from FWHM
    REAL(dp):: a = 0.0_dp      ! decay per step = EXP(-dt/tau_c)
    REAL(dp):: s = 0.0_dp      ! OU noise scale per step for z(t)
    REAL(dp):: z = 0.0_dp      ! OU random force z(t)
    REAL(dp):: y = 0.0_dp      ! memory friction integral y(t)
  END TYPE LangevinState

CONTAINS

  SUBROUTINE init_langevin(state, ctrl, dt, gamma_in, fwhm_in)
    TYPE(LangevinState), INTENT(INOUT):: state
    TYPE(SimCtrl),        INTENT(IN)  :: ctrl
    REAL(dp),             INTENT(IN)  :: dt
    REAL(dp), OPTIONAL,   INTENT(IN)  :: gamma_in
    REAL(dp), OPTIONAL,   INTENT(IN)  :: fwhm_in

    REAL(dp) :: var_z, gamma_eff, fwhm_eff

    gamma_eff = ctrl%gamma
    IF (PRESENT(gamma_in)) gamma_eff = gamma_in

    fwhm_eff = ctrl%fwhm
    IF (PRESENT(fwhm_in)) fwhm_eff = fwhm_in

    state%gamma = gamma_eff
    state%enabled = (gamma_eff > 0.0_dp)

    IF (.NOT. state%enabled) THEN
      state%colored = .false.
      state%tau_c = 0.0_dp
      state%a = 0.0_dp
      state%s = 0.0_dp
      state%z = 0.0_dp
      state%y = 0.0_dp
      RETURN
    END IF

    state%colored = (ctrl%use_colored .and. TRIM(ctrl%kernel) == 'lorentz' .and. fwhm_eff > 0.0_dp)

    IF (state%colored) THEN
      ! Lorentzian FWHM -> correlation time: tau_c = 2/FWHM
      state%tau_c = 2.0_dp/fwhm_eff
      state%a = EXP(-dt/state%tau_c)

      ! Exponential kernel: gamma(t) = (gamma/tau_c) EXP(-t/tau_c)
      ! FDT => Var(z) = gamma/(beta*tau_c)
      var_z = state%gamma/(ctrl%beta*state%tau_c)

      ! Discrete OU: z_{n+1} = a z_n + s N(0,1), Var(z)=var_z
      state%s = SQRT(MAX(1.0e-30_dp, var_z*(1.0_dp - state%a**2)))

      state%z = 0.0_dp
      state%y = 0.0_dp
    ELSE
      ! Markovian (white) case handled in next_kick()
      state%tau_c = 0.0_dp
      state%a = 0.0_dp
      state%s = 0.0_dp
      state%z = 0.0_dp
      state%y = 0.0_dp
    END IF
  END SUBROUTINE init_langevin


  SUBROUTINE white_kick(beta, gamma, dt, xi)
    ! Markovian impulse (momentum kick) over dt:
    !   Δp_noise = SQRT(2 gamma dt / beta) * N(0,1)
    REAL(dp), INTENT(IN) :: beta, gamma, dt
    REAL(dp), INTENT(OUT):: xi
    REAL(dp) :: z1, z2
    CALL randn_gauss(z1, z2)
    xi = SQRT(2.0_dp*gamma*dt/beta) * z1
  END SUBROUTINE white_kick


  SUBROUTINE next_kick(state, ctrl, dt, xi, pbar)
    ! Returns total momentum impulse Δp over dt (friction + stochastic).
    ! If state%enabled=.false. then xi=0.
    TYPE(LangevinState), INTENT(INOUT):: state
    TYPE(SimCtrl),        INTENT(IN)  :: ctrl
    REAL(dp),             INTENT(IN)  :: dt
    REAL(dp),             INTENT(OUT) :: xi
    REAL(dp), OPTIONAL,   INTENT(IN)  :: pbar

    REAL(dp) :: z1, z2, p_loc

    IF (.NOT. state%enabled) THEN
      xi = 0.0_dp
      RETURN
    END IF

    p_loc = 0.0_dp
    IF (PRESENT(pbar)) p_loc = pbar

    IF (state%colored) THEN
      ! y update (exact discrete update if p is held constant over dt)
      state%y = state%a*state%y + (state%gamma/state%tau_c)*(1.0_dp - state%a)*p_loc

      ! OU random force z(t)
      CALL randn_gauss(z1, z2)
      state%z = state%a*state%z + state%s*z1

      ! Impulse over dt
      xi = (state%z - state%y) * dt
    ELSE
      ! Markovian: dp = (-gamma p + R) dt
      CALL white_kick(ctrl%beta, state%gamma, dt, xi)
      xi = xi - state%gamma*p_loc*dt
    END IF
  END SUBROUTINE next_kick

END MODULE langevin
