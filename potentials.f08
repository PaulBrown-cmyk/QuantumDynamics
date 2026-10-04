!=============================== potentials.f08 ===============================
! Two-surface (diabatic) potentials used by the dynamics code.
!
! Convention implemented here (per project discussion):
!   * The base model is ALWAYS two harmonic wells (one per diabatic surface):
!       V11_h(x) = 1/2 k1 (x-x1)^2
!       V22_h(x) = 1/2 k2 (x-x2)^2
!   * The "anharmonic" option means: add a quartic term to either (or both)
!     harmonic wells:
!       Vii(x) = Vii_h(x) + c4_i (x-xi)^4
!   * Optional diabatic coupling (Gaussian or exponential) and constant energy
!     shifts are controlled independently.
!
MODULE potentials
  USE kinds
  USE params
  USE rng, ONLY: randn_gauss
  IMPLICIT NONE

  ! Stochastic potential modulation state (shared across grid evaluations).
  ! Updated once per time step via potentials_bath_next().
  TYPE :: BathPotState
     LOGICAL  :: enabled = .false.
     LOGICAL  :: colored = .false.
     LOGICAL  :: coupled = .true.
     REAL(dp) :: a = 0.0_dp
     REAL(dp) :: s = 0.0_dp
     REAL(dp) :: rt(2) = 0.0_dp   ! (1)=reactant, (2)=product
  END TYPE BathPotState

  TYPE(BathPotState), SAVE :: bath_pot

CONTAINS


  SUBROUTINE potentials_bath_init(ctrl, dt)
    TYPE(SimCtrl), INTENT(IN) :: ctrl
    REAL(dp),      INTENT(IN) :: dt
    REAL(dp) :: tau

    bath_pot%enabled = (ctrl%bath_pot_mode /= 0) .AND. (ctrl%bath_pot_sigma > 0.0_dp) .AND. &
                      (ctrl%bath_pot_reactant .OR. ctrl%bath_pot_product)
    bath_pot%colored = ctrl%bath_pot_colored
    bath_pot%coupled = ctrl%bath_pot_coupled
    bath_pot%rt = 0.0_dp
    bath_pot%a  = 0.0_dp
    bath_pot%s  = 0.0_dp

    IF (.NOT. bath_pot%enabled) RETURN

    IF (bath_pot%colored) THEN
      IF (ctrl%bath_pot_fwhm > 0.0_dp) THEN
        tau = 2.0_dp/ctrl%bath_pot_fwhm
        bath_pot%a = EXP(-dt/tau)
        bath_pot%s = ctrl%bath_pot_sigma*SQRT(MAX(0.0_dp, 1.0_dp - bath_pot%a*bath_pot%a))
      ELSE
        bath_pot%colored = .FALSE.
      END IF
    END IF

  END SUBROUTINE potentials_bath_init

  SUBROUTINE potentials_bath_next(ctrl)
    TYPE(SimCtrl), INTENT(IN) :: ctrl
    REAL(dp) :: z1, z2, z3, z4

    IF (.NOT. bath_pot%enabled) THEN
      bath_pot%rt = 0.0_dp
      RETURN
    END IF

    IF (bath_pot%colored) THEN
      CALL randn_gauss(z1, z2)
      bath_pot%rt(1) = bath_pot%a*bath_pot%rt(1) + bath_pot%s*z1

      IF (bath_pot%coupled) THEN
        bath_pot%rt(2) = bath_pot%rt(1)
      ELSE
        CALL randn_gauss(z3, z4)
        bath_pot%rt(2) = bath_pot%a*bath_pot%rt(2) + bath_pot%s*z3
      END IF
    ELSE
      CALL randn_gauss(z1, z2)
      bath_pot%rt(1) = ctrl%bath_pot_sigma*z1
      IF (bath_pot%coupled) THEN
        bath_pot%rt(2) = bath_pot%rt(1)
      ELSE
        bath_pot%rt(2) = ctrl%bath_pot_sigma*z2
      END IF
    END IF

    IF (.NOT. ctrl%bath_pot_reactant) bath_pot%rt(1) = 0.0_dp
    IF (.NOT. ctrl%bath_pot_product ) bath_pot%rt(2) = 0.0_dp

  END SUBROUTINE potentials_bath_next

  SUBROUTINE v_two_surface(ctrl, x, v11, v22, v12)
    TYPE(SimCtrl), INTENT(IN)  :: ctrl
    REAL(dp),      INTENT(IN)  :: x
    REAL(dp),      INTENT(OUT) :: v11, v22, v12

    REAL(dp) :: dx1, dx2
    REAL(dp) :: base11, base22
    REAL(dp) :: xc
    REAL(dp) :: c4_1, c4_2
    LOGICAL  :: add_cubic1, add_cubic2
    LOGICAL  :: want_coupling
    LOGICAL  :: use_exponential
    CHARACTER(LEN=:), ALLOCATABLE :: model

    REAL(dp) :: rt1, rt2
    REAL(dp) :: k_eff1, k_eff2

    REAL(dp), PARAMETER :: TINY = 1.0e-30_dp

    model = ADJUSTL(TRIM(ctrl%pot_model))

    rt1 = 0.0_dp
    rt2 = 0.0_dp
    IF (bath_pot%enabled) THEN
      rt1 = bath_pot%rt(1)
      rt2 = bath_pot%rt(2)
    END IF

    ! Base: two harmonic wells (always) + optional stochastic modulation
    SELECT CASE (ctrl%bath_pot_mode)
    CASE (1)
      ! Coordinate shift: Vii = 1/2 ki (x - xi - R_t)^2
      dx1 = x - ctrl%x1 - rt1
      dx2 = x - ctrl%x2 - rt2
      base11 = 0.5_dp*ctrl%k1*dx1*dx1
      base22 = 0.5_dp*ctrl%k2*dx2*dx2

    CASE (2)
      ! Curvature modulation: Vii = 1/2 (ki + R_t) (x - xi)^2
      dx1 = x - ctrl%x1
      dx2 = x - ctrl%x2
      k_eff1 = MAX(0.0_dp, ctrl%k1 + rt1)
      k_eff2 = MAX(0.0_dp, ctrl%k2 + rt2)
      base11 = 0.5_dp*k_eff1*dx1*dx1
      base22 = 0.5_dp*k_eff2*dx2*dx2

    CASE DEFAULT
      dx1 = x - ctrl%x1
      dx2 = x - ctrl%x2
      base11 = 0.5_dp*ctrl%k1*dx1*dx1
      base22 = 0.5_dp*ctrl%k2*dx2*dx2
    END SELECT

    v11 = base11
    v22 = base22
    v12 = 0.0_dp

    ! --- Model selection -----------------------------------------------------
    ! We treat the following families (case-insensitive, substring matching):
    !   * 'harmonic'            : just the two harmonic wells (plus optional coupling)
    !   * 'anharmonic1'         : quartic term added to surface 1 only
    !   * 'anharmonic2'         : quartic term added to surface 2 only
    !   * 'anharmonic12'        : quartic term added to both surfaces
    !
    ! Coupling control:
    !   - Enabled if |ctrl%v12|>0 and ctrl%sigma>0, unless model contains 'nocpl'
    !   - Default shape is Gaussian; if model contains 'exp' use exponential
    !
    ! Energy shifts are independent of explicit quartic coefficients c4_1/c4_2.

    add_cubic1 = .FALSE.
    add_cubic2 = .FALSE.

    IF (INDEX(model,'anharmonic12') > 0 .OR. INDEX(model,'cubic12') > 0) THEN
      add_cubic1 = .TRUE.
      add_cubic2 = .TRUE.

    ELSE IF (INDEX(model,'anharmonic1') > 0 .OR. INDEX(model,'cubic1') > 0) THEN
      add_cubic1 = .TRUE.

    ELSE IF (INDEX(model,'anharmonic2') > 0 .OR. INDEX(model,'cubic2') > 0) THEN
      add_cubic2 = .TRUE.
    END IF

    c4_1 = ctrl%c4_1
    c4_2 = ctrl%c4_2

    IF (add_cubic1) v11 = v11 + c4_1*dx1*dx1*dx1*dx1
    IF (add_cubic2) v22 = v22 + c4_2*dx2*dx2*dx2*dx2

    ! --- Constant shifts (optional) ------------------------------------------
    v11 = v11 + ctrl%v1_shift
    v22 = v22 + ctrl%v2_shift

    ! Enthalpy modulation: Vii = Vii + R_t
    IF (ctrl%bath_pot_mode == 3) THEN
      v11 = v11 + rt1
      v22 = v22 + rt2
    END IF

    ! --- Diabatic coupling V12(x) (optional) ---------------------------------
    want_coupling = ctrl%want_coupling
    IF (INDEX(model,'nocpl') > 0 .OR. INDEX(model,'nocouple') > 0 .OR. INDEX(model,'decouple') > 0) THEN
      want_coupling = .FALSE.
    ELSE
      want_coupling = want_coupling .AND. ABS(ctrl%v12) > TINY .AND. ctrl%sigma > 0.0_dp
    END IF

    use_exponential = ctrl%use_exponential .OR. INDEX(model,'exponential') > 0 .OR. &
                      INDEX(model,'exp') > 0

    IF (want_coupling) THEN
      ! Center coupling at the midpoint of the diabatic well minima by default.
      xc = x - 0.5_dp*(ctrl%x1 + ctrl%x2)

      IF (use_exponential) THEN
        ! V12(x) = v12 * exp(-|x-xc|/sigma)
        v12 = ctrl%v12*EXP(-ABS(xc)/ctrl%sigma)
      ELSE
        ! V12(x) = v12 * exp(-(x-xc)^2/(2*sigma^2))
        v12 = ctrl%v12*EXP(-0.5_dp*(xc/ctrl%sigma)**2)
      END IF
    END IF

  END SUBROUTINE v_two_surface



!===========================================================================
! Convenience: build diabatic + adiabatic PES arrays on a supplied x-grid.
! This is useful for output/plotting and for any precomputation steps.
!
! Outputs:
!   v11(x), v22(x), v12(x) : diabatic matrix elements
!   v_lower(x), v_upper(x) : adiabatic eigenvalues of [[v11,v12],[v12,v22]]
!===========================================================================
SUBROUTINE pes_on_grid(ctrl, x, v11, v22, v12, v_lower, v_upper)
  TYPE(SimCtrl), INTENT(IN)  :: ctrl
  REAL(dp),      INTENT(IN)  :: x(:)
  REAL(dp),      INTENT(OUT) :: v11(:), v22(:), v12(:), v_lower(:), v_upper(:)

  INTEGER  :: i, nx
  REAL(dp) :: vavg, dlt, rad

  nx = SIZE(x)
  IF (SIZE(v11) /= nx .OR. SIZE(v22) /= nx .OR. SIZE(v12) /= nx .OR. &
      SIZE(v_lower) /= nx .OR. SIZE(v_upper) /= nx) THEN
    ERROR STOP 'pes_on_grid: output arrays must match SIZE(x)'
  END IF

  DO i = 1, nx
    CALL v_two_surface(ctrl, x(i), v11(i), v22(i), v12(i))
    vavg = 0.5_dp*(v11(i) + v22(i))
    dlt  = 0.5_dp*(v11(i) - v22(i))
    rad  = SQRT(dlt*dlt + v12(i)*v12(i))
    v_lower(i) = vavg - rad
    v_upper(i) = vavg + rad
  END DO
END SUBROUTINE pes_on_grid

END MODULE potentials
