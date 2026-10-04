!=============================== fftwrap.f08 =================================
!> Minimal FFTW3 Fortran interface (FFTW3 + threads).
!!
!! This replaces FFTW's large `fftw3.f03` interface include with a tiny,
!! explicit ISO_C_BINDING interface for only the routines used by this codebase.
!!
!! Exposed C interfaces:
!!   - fftw_plan_dft_1d / fftw_execute_dft / fftw_destroy_plan
!!   - fftw_init_threads / fftw_plan_with_nthreads / fftw_cleanup_threads
!!
!! In addition, this module provides two Fortran convenience wrappers:
!!   - CALL init_fft_threads()
!!   - CALL cleanup_fft_threads()
!!
!! These wrappers set the FFTW planner thread count to the OpenMP max threads
!! (when built with -fopenmp), otherwise to 1.
!===============================================================================
MODULE fftwrap
  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT, C_PTR, C_DOUBLE_COMPLEX
  IMPLICIT NONE
  PRIVATE

  ! Public FFTW constants (subset).
  INTEGER(c_int), PARAMETER, PUBLIC :: FFTW_FORWARD  = -1_C_INT
  INTEGER(c_int), PARAMETER, PUBLIC :: FFTW_BACKWARD =  1_C_INT
  INTEGER(c_int), PARAMETER, PUBLIC :: FFTW_MEASURE  =  0_C_INT
  INTEGER(c_int), PARAMETER, PUBLIC :: FFTW_ESTIMATE = 64_C_INT


  ! Public C-bindings used by the code.
  PUBLIC :: fftw_plan_dft_1d, fftw_execute_dft, fftw_destroy_plan
  PUBLIC :: init_fft_threads, cleanup_fft_threads

  INTERFACE
    INTEGER(C_INT) FUNCTION fftw_init_threads() BIND(C, name="fftw_init_threads")
      IMPORT :: C_INT
    END FUNCTION fftw_init_threads

    SUBROUTINE fftw_cleanup_threads() BIND(C, name="fftw_cleanup_threads")
      ! No dummy arguments.
    END SUBROUTINE fftw_cleanup_threads

    SUBROUTINE fftw_plan_with_nthreads(nthreads) BIND(C, name="fftw_plan_with_nthreads")
      IMPORT :: C_INT
      INTEGER(C_INT), VALUE :: nthreads
    END SUBROUTINE fftw_plan_with_nthreads

    TYPE(C_PTR) FUNCTION fftw_plan_dft_1d(n, IN, OUT, sign, flags) BIND(C, name="fftw_plan_dft_1d")
      IMPORT :: C_INT, C_PTR, C_DOUBLE_COMPLEX
      INTEGER(C_INT), VALUE :: n
      COMPLEX(C_DOUBLE_COMPLEX) :: IN(*)
      COMPLEX(C_DOUBLE_COMPLEX) :: OUT(*)
      INTEGER(C_INT), VALUE :: sign
      INTEGER(C_INT), VALUE :: flags
    END FUNCTION fftw_plan_dft_1d

    SUBROUTINE fftw_execute_dft(p, IN, OUT) BIND(C, name="fftw_execute_dft")
      IMPORT :: C_PTR, C_DOUBLE_COMPLEX
      TYPE(C_PTR), VALUE :: p
      COMPLEX(C_DOUBLE_COMPLEX) :: IN(*)
      COMPLEX(C_DOUBLE_COMPLEX) :: OUT(*)
    END SUBROUTINE fftw_execute_dft

    SUBROUTINE fftw_destroy_plan(p) BIND(C, name="fftw_destroy_plan")
      IMPORT :: C_PTR
      TYPE(C_PTR), VALUE :: p
    END SUBROUTINE fftw_destroy_plan
  END INTERFACE

CONTAINS

SUBROUTINE init_fft_threads()
#ifdef _OPENMP
  USE omp_lib, ONLY: omp_get_max_threads
#endif
  IMPLICIT NONE

  INTEGER(C_INT) :: ok, nthreads

  ok = fftw_init_threads()

  ! If FFTW threads cannot be initialized, continue in single-thread mode.
  ! (FFTW is still usable; only threaded planning/execution is disabled.)
  IF (ok == 0_C_INT) THEN
    RETURN
  END IF

#ifdef _OPENMP
  nthreads = INT(omp_get_max_threads(), C_INT)
#else
  nthreads = 1_C_INT
#endif

  IF (nthreads < 1_C_INT) nthreads = 1_C_INT
  CALL fftw_plan_with_nthreads(nthreads)
END SUBROUTINE init_fft_threads

SUBROUTINE cleanup_fft_threads()
  IMPLICIT NONE
  CALL fftw_cleanup_threads()
END SUBROUTINE cleanup_fft_threads

END MODULE fftwrap
