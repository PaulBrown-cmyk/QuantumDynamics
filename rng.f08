!=============================== rng.f08 =====================================
MODULE rng
  USE kinds
  USE iso_fortran_env, ONLY: int64
  IMPLICIT NONE
CONTAINS

  SUBROUTINE seed_stream(seed)
    INTEGER, INTENT(IN):: seed
    INTEGER :: n, i
    INTEGER, ALLOCATABLE :: s(:)

    CALL random_seed(size=n)
    ALLOCATE(s(n))
    DO i = 1, n
      ! Deterministic per-trajectory scrambling in 64-bit arithmetic; avoid undefined
      ! default-integer overflow before reducing to random_seed's kind.
      s(i) = INT(MODULO(1103515245_int64*(INT(seed, int64) + INT(i, int64)) + &
                        12345_int64, 2147483647_int64), KIND(s))
      IF (s(i) == 0) s(i) = i
    END DO
    CALL random_seed(put=s)
    DEALLOCATE(s)
  END SUBROUTINE seed_stream

  SUBROUTINE randn_gauss(z1, z2)
    REAL(dp), INTENT(OUT):: z1, z2
    REAL(dp) :: u1, u2, r, f
    CALL random_number(u1)
    CALL random_number(u2)
    u1 = MAX(u1, 1.0e-12_dp)
    r  = SQRT(-2.0_dp*LOG(u1))
    f  = 2.0_dp*ACOS(-1.0_dp)*u2
    z1 = r*COS(f)
    z2 = r*SIN(f)
  END SUBROUTINE randn_gauss

END MODULE rng
