!=============================== langevin.f90 =================================
MODULE langevin
  !
  ! Generalized Langevin (memory-friction) helpers.
  !
  ! Target: shared nuclear wavepacket centroid (atomic units):
  !
  !   dp/dt = < -dV/dx > - ∫_0^t gamma(t-t') p(t') dt' + R(t)
  !
  ! with fluctuation–dissipation:
  !
  !   <R(t) R(t')> = (mass/beta) * gamma(|t-t'|)
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
  !   <z(t) z(t')> = (mass/beta) * gamma(|t-t'|)
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

    REAL(dp) :: var_z, gamma_eff, fwhm_eff, z1, z2

    gamma_eff = ctrl%gamma
    IF (PRESENT(gamma_in)) gamma_eff = gamma_in

    fwhm_eff = ctrl%fwhm
    IF (PRESENT(fwhm_in)) fwhm_eff = fwhm_in

    IF (gamma_eff < 0.0_dp .OR. dt < 0.0_dp) &
      ERROR STOP "Invalid Langevin gamma or dt"
    state%gamma = gamma_eff
    state%enabled = (gamma_eff > 0.0_dp) .AND. &
                    (ctrl%damp_reactant .OR. ctrl%damp_product)

    IF (state%enabled .AND. (ctrl%beta <= 0.0_dp .OR. ctrl%mass <= 0.0_dp)) &
      ERROR STOP "Invalid Langevin beta or mass"

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
      ! FDT => Var(z) = mass*gamma/(beta*tau_c)
      var_z = ctrl%mass*state%gamma/(ctrl%beta*state%tau_c)

      ! Discrete OU: z_{n+1} = a z_n + s N(0,1), Var(z)=var_z
      state%s = SQRT(MAX(0.0_dp, var_z*(1.0_dp - state%a**2)))

      CALL randn_gauss(z1, z2)
      state%z = SQRT(var_z)*z1  ! stationary random force; y(0)=0
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


  SUBROUTINE white_kick(beta, mass, gamma, dt, xi)
    ! Markovian impulse (momentum kick) over dt:
    !   Δp_noise = SQRT(mass*(1-exp(-2 gamma dt))/beta) * N(0,1)
    REAL(dp), INTENT(IN) :: beta, mass, gamma, dt
    REAL(dp), INTENT(OUT):: xi
    REAL(dp) :: z1, z2
    CALL randn_gauss(z1, z2)
    xi = SQRT(mass*(1.0_dp-EXP(-2.0_dp*gamma*dt))/beta) * z1
  END SUBROUTINE white_kick


  SUBROUTINE next_kick(state, ctrl, dt, xi, pbar)
    ! Returns total momentum impulse Δp over dt (friction + stochastic).
    ! If state%enabled=.false. then xi=0.
    TYPE(LangevinState), INTENT(INOUT):: state
    TYPE(SimCtrl),        INTENT(IN)  :: ctrl
    REAL(dp),             INTENT(IN)  :: dt
    REAL(dp),             INTENT(OUT) :: xi
    REAL(dp),             INTENT(IN)  :: pbar

    REAL(dp) :: z1, z2, p_loc, r, rnew, omega, c, sn, decay

    IF (dt < 0.0_dp .OR. ctrl%beta <= 0.0_dp .OR. ctrl%mass <= 0.0_dp) &
      ERROR STOP "Invalid bath step"
    xi = 0.0_dp
    IF (.NOT. state%enabled .OR. dt <= 0.0_dp) RETURN
    p_loc = pbar
    IF (state%colored) THEN
      ! Strang splitting of dp=-r dt, dr=(gamma/tau)p dt-r/tau dt+noise,
      ! r=y-z. Rotation preserves p^2 + r^2/omega^2; OU preserves FDT.
      ! Keep y and z separately: dy=(gamma*p-y)/tau, dz=-z/tau+noise.
      omega = SQRT(state%gamma/state%tau_c)
      c = COS(0.5_dp*dt*omega)
      sn = SIN(0.5_dp*dt*omega)
      r = state%y-state%z
      rnew = c*r + omega*sn*p_loc
      p_loc = c*p_loc - sn*r/omega
      state%y = rnew + state%z
      decay = EXP(-dt/state%tau_c)
      CALL randn_gauss(z1, z2)
      state%y = decay*state%y
      state%z = decay*state%z + &
        SQRT(ctrl%mass*state%gamma/(ctrl%beta*state%tau_c)*(1.0_dp-decay**2))*z1
      r = state%y-state%z
      rnew = c*r + omega*sn*p_loc
      p_loc = c*p_loc - sn*r/omega
      state%y = rnew + state%z
      xi = p_loc-pbar
    ELSE
      ! Exact free OU impulse: equilibrium Var(pbar)=mass/beta for any dt.
      CALL white_kick(ctrl%beta, ctrl%mass, state%gamma, dt, xi)
      xi = xi + (EXP(-state%gamma*dt)-1.0_dp)*pbar
    END IF
  END SUBROUTINE next_kick

END MODULE langevin
