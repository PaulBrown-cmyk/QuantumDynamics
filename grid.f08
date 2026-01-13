!=============================== grid.f08 ====================================
MODULE grid
  USE kinds
  IMPLICIT NONE
  TYPE:: RealGrid
     INTEGER :: nx
     REAL(dp):: xmin, xmax, dx
     REAL(dp), ALLOCATABLE :: x(:)
     REAL(dp), ALLOCATABLE :: k(:)
  END TYPE
CONTAINS
  SUBROUTINE build_grid(g, nx, xmin, xmax)
    TYPE(RealGrid), INTENT(OUT):: g
    INTEGER, INTENT(IN) :: nx
    REAL(dp), INTENT(IN):: xmin, xmax
    INTEGER :: I 
    REAL(dp) :: L

    g%nx   = nx
    g%xmin = xmin
    g%xmax = xmax
    g%dx   = (xmax-xmin)/REAL(nx, dp)
    L      = xmax-xmin

    ALLOCATE(g%x(nx), g%k(nx))

    DO I = 1, nx
      g%x(I) = xmin + (REAL(I-1,dp) + 0.5_dp)*g%dx
    END DO

    ! k-grid (FFT frequency convention, 2π periodicity over box length L)
    DO I = 1, nx
      IF (I <= nx/2) THEN
        g%k(I) = 2.0_dp*ACOS(-1.0_dp)*REAL(I-1, dp)/L
      ELSE
        g%k(I) = 2.0_dp*ACOS(-1.0_dp)*REAL(I-1-nx, dp)/L
      END IF
    END DO
  END SUBROUTINE build_grid
END MODULE grid
