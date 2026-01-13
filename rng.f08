!=============================== rng.f08 =====================================
MODULE rng
  USE kinds
  IMPLICIT NONE
CONTAINS

  SUBROUTINE seed_stream(seed)
    INTEGER, INTENT(IN):: seed
    INTEGER :: n, i
    INTEGER, ALLOCATABLE :: s(:)

    CALL random_seed(size=n)
    ALLOCATE(s(n))
    DO i = 1, n
      ! Simple deterministic scrambling for per-rank/per-traj streams.
      ! NOTE: keep within default integer range.
      s(i) = MOD(1103515245*(seed+i) + 12345, 2147483647)
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
