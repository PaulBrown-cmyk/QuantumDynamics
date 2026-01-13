!=============================== propagator.f90 ===============================
MODULE propagator
  USE kinds
  USE params
  USE grid
  USE potentials, ONLY: v_two_surface, potentials_bath_next
  USE omp_lib
  USE ISO_C_BINDING, ONLY: c_ptr, c_associated
  USE fftwrap, ONLY: fftw_plan_dft_1d, fftw_execute_dft, fftw_destroy_plan, &
                     FFTW_FORWARD, FFTW_BACKWARD, FFTW_MEASURE
  IMPLICIT NONE

  TYPE:: SOProp
     TYPE(RealGrid) :: g
     COMPLEX(dp), ALLOCATABLE :: psi1(:), psi2(:)   ! two-surface wavefunction
     COMPLEX(dp), ALLOCATABLE :: buf1(:), buf2(:)
     TYPE(C_PTR), ALLOCATABLE :: p_f(:)             ! FFT plans (1: forward, 2: backward)
  END TYPE SOProp

CONTAINS

  SUBROUTINE init_prop(prop, g)
    TYPE(SOProp),  INTENT(INOUT):: prop
    TYPE(RealGrid),INTENT(IN)   :: g
    INTEGER :: nx

    prop%g = g
    nx = g%nx

    ALLOCATE(prop%psi1(nx), prop%psi2(nx), prop%buf1(nx), prop%buf2(nx))
    prop%psi1 = (0.0_dp, 0.0_dp)
    prop%psi2 = (0.0_dp, 0.0_dp)
    prop%buf1 = (0.0_dp, 0.0_dp)
    prop%buf2 = (0.0_dp, 0.0_dp)

    ALLOCATE(prop%p_f(2))
    prop%p_f(1) = fftw_plan_dft_1d(nx, prop%psi1, prop%buf1, FFTW_FORWARD,  FFTW_MEASURE)
    prop%p_f(2) = fftw_plan_dft_1d(nx, prop%buf1, prop%psi1, FFTW_BACKWARD, FFTW_MEASURE)
  END SUBROUTINE init_prop

  SUBROUTINE destroy_prop(prop)
    TYPE(SOProp), INTENT(INOUT):: prop
    IF (ALLOCATED(prop%p_f)) THEN
      IF (c_associated(prop%p_f(1))) CALL fftw_destroy_plan(prop%p_f(1))
      IF (c_associated(prop%p_f(2))) CALL fftw_destroy_plan(prop%p_f(2))
      DEALLOCATE(prop%p_f)
    END IF
    IF (ALLOCATED(prop%psi1)) DEALLOCATE(prop%psi1)
    IF (ALLOCATED(prop%psi2)) DEALLOCATE(prop%psi2)
    IF (ALLOCATED(prop%buf1)) DEALLOCATE(prop%buf1)
    IF (ALLOCATED(prop%buf2)) DEALLOCATE(prop%buf2)
  END SUBROUTINE destroy_prop

  SUBROUTINE set_gaussian_packet(prop, ctrl)
    TYPE(SOProp),  INTENT(INOUT):: prop
    TYPE(SimCtrl), INTENT(IN)   :: ctrl
    INTEGER :: i, nx
    REAL(dp) :: amp, s2, x0, p0

    nx = prop%g%nx
    s2 = ctrl%sigma0**2
    x0 = ctrl%x0
    p0 = ctrl%p0

    !$omp parallel do default(shared) private(i, amp)
    DO i = 1, nx
      amp = EXP(-0.5_dp*((prop%g%x(i)-x0)**2)/s2)
      prop%psi1(i) = CMPLX(amp*COS(p0*prop%g%x(i)), amp*SIN(p0*prop%g%x(i)), dp)
      prop%psi2(i) = (0.0_dp, 0.0_dp)
    END DO
    !$omp end parallel do

    CALL normalize_two(prop%psi1, prop%psi2, prop%g%dx)
  END SUBROUTINE set_gaussian_packet

  SUBROUTINE normalize_two(psi1, psi2, dx)
    COMPLEX(dp), INTENT(INOUT):: psi1(:), psi2(:)
    REAL(dp),    INTENT(IN)   :: dx
    REAL(dp) :: nrm
    nrm = ( SUM(REAL(CONJG(psi1)*psi1, dp)) + SUM(REAL(CONJG(psi2)*psi2, dp)) ) * dx
    IF (nrm > 0.0_dp) THEN
      psi1 = psi1/SQRT(nrm)
      psi2 = psi2/SQRT(nrm)
    END IF
  END SUBROUTINE normalize_two

  SUBROUTINE step_split_na(ctrl, prop, dt, gamma, xi)
    ! Nonadiabatic split step:
    !   exp(-i V dt/2) -> FFT -> exp(-i T dt)*exp(-gamma dt) -> iFFT -> exp(+i xi x) -> exp(-i V dt/2)
    TYPE(SimCtrl), INTENT(IN)    :: ctrl
    TYPE(SOProp),  INTENT(INOUT) :: prop
    REAL(dp),      INTENT(IN)    :: dt, gamma, xi

    INTEGER :: i, nx
    REAL(dp) :: v11, v22, v12, x
    REAL(dp) :: k2, phase
    COMPLEX(dp) :: u11, u12, u21, u22
    COMPLEX(dp) :: a0, b0

    nx = prop%g%nx

    ! Update stochastic potential modulation (if enabled) once per step.
    CALL potentials_bath_next(ctrl)

    ! --- V half-step (local 2x2 at each x) with Gaussian coupling ---
    !$omp parallel do default(shared) private(i, x, v11, v22, v12, u11, u12, u21, u22, a0, b0)
    DO i = 1, nx
      x = prop%g%x(i)
      CALL v_two_surface(ctrl, x, v11, v22, v12)
      CALL unitary_exp_2x2(CMPLX(v11, 0.0_dp, dp), CMPLX(v12, 0.0_dp, dp), &
                           CMPLX(v12, 0.0_dp, dp), CMPLX(v22, 0.0_dp, dp), 0.5_dp*dt, &
                           u11, u12, u21, u22)
      a0 = prop%psi1(i); b0 = prop%psi2(i)
      prop%psi1(i) = u11*a0 + u12*b0
      prop%psi2(i) = u21*a0 + u22*b0
    END DO
    !$omp end parallel do

    ! --- FFT to k-space ---
    CALL fftw_execute_dft(prop%p_f(1), prop%psi1, prop%buf1)
    CALL fftw_execute_dft(prop%p_f(1), prop%psi2, prop%buf2)

    ! --- Kinetic step with damping ---
    !$omp parallel do default(shared) private(i, k2, phase)
    DO i = 1, nx
      k2 = (prop%g%k(i)**2)/(2.0_dp*me)
      phase = -dt*k2
      prop%buf1(i) = prop%buf1(i) * EXP(CMPLX(0.0_dp, phase, dp)) * EXP(-gamma*dt)
      prop%buf2(i) = prop%buf2(i) * EXP(CMPLX(0.0_dp, phase, dp)) * EXP(-gamma*dt)
    END DO
    !$omp end parallel do

    ! --- inverse FFT back to real-space ---
    CALL fftw_execute_dft(prop%p_f(2), prop%buf1, prop%psi1)
    CALL fftw_execute_dft(prop%p_f(2), prop%buf2, prop%psi2)
    prop%psi1 = prop%psi1/REAL(nx, dp)
    prop%psi2 = prop%psi2/REAL(nx, dp)

    ! Random kick applied as a real-space phase: exp(i xi x)
    !$omp parallel do default(shared) private(i)
    DO i = 1, nx
      prop%psi1(i) = prop%psi1(i) * EXP(CMPLX(0.0_dp, xi*prop%g%x(i), dp))
      prop%psi2(i) = prop%psi2(i) * EXP(CMPLX(0.0_dp, xi*prop%g%x(i), dp))
    END DO
    !$omp end parallel do

    ! --- V half-step again ---
    !$omp parallel do default(shared) private(i, x, v11, v22, v12, u11, u12, u21, u22, a0, b0)
    DO i = 1, nx
      x = prop%g%x(i)
      CALL v_two_surface(ctrl, x, v11, v22, v12)
      CALL unitary_exp_2x2(CMPLX(v11, 0.0_dp, dp), CMPLX(v12, 0.0_dp, dp), &
                           CMPLX(v12, 0.0_dp, dp), CMPLX(v22, 0.0_dp, dp), 0.5_dp*dt, &
                           u11, u12, u21, u22)
      a0 = prop%psi1(i); b0 = prop%psi2(i)
      prop%psi1(i) = u11*a0 + u12*b0
      prop%psi2(i) = u21*a0 + u22*b0
    END DO
    !$omp end parallel do

    CALL normalize_two(prop%psi1, prop%psi2, prop%g%dx)
  END SUBROUTINE step_split_na

  SUBROUTINE unitary_exp_2x2(a, b, C, d, tau, u11, u12, u21, u22)
    ! Robust SU(2)-style exponential for a 2x2 Hermitian matrix:
    !   H = [[a, b],[c, d]] with d,a real and c=conjg(b) (in practice).
    COMPLEX(dp), INTENT(IN)  :: a, b, C, d
    REAL(dp),    INTENT(IN)  :: tau
    COMPLEX(dp), INTENT(OUT) :: u11, u12, u21, u22

    REAL(dp) :: tr, dz, bx, by, om, c0, s0
    COMPLEX(dp) :: ph, minus_i

    tr = 0.5_dp*(REAL(a,dp) + REAL(d,dp))
    dz = 0.5_dp*(REAL(a,dp) - REAL(d,dp))
    bx = REAL(b,dp)
    by = AIMAG(b)
    om = SQRT(dz*dz + bx*bx + by*by)

    ph = EXP(CMPLX(0.0_dp, -tau*tr, dp))
    minus_i = CMPLX(0.0_dp, -1.0_dp, dp)

    IF (om < 1.0e-14_dp) THEN
      u11 = ph
      u22 = ph
      u12 = (0.0_dp, 0.0_dp)
      u21 = (0.0_dp, 0.0_dp)
      RETURN
    END IF

    c0 = COS(tau*om)
    s0 = SIN(tau*om)/om

    ! U = ph*( c0*I - i*s0*(vec·sigma) )
    u11 = ph*( CMPLX(c0,0.0_dp,dp) + minus_i*CMPLX(s0*dz,0.0_dp,dp) )
    u22 = ph*( CMPLX(c0,0.0_dp,dp) - minus_i*CMPLX(s0*dz,0.0_dp,dp) )
    u12 = ph*( minus_i*CMPLX(s0*bx, -s0*by, dp) )
    u21 = ph*( minus_i*CMPLX(s0*bx, +s0*by, dp) )
  END SUBROUTINE unitary_exp_2x2

END MODULE propagator