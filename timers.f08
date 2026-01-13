!=============================== timers.f08 ==================================
MODULE timers
  USE kinds
  IMPLICIT NONE
CONTAINS
  FUNCTION walltime() RESULT(t)
    REAL(dp) :: t
    INTEGER :: count, rate
    CALL system_clock(count, rate)
    IF (rate > 0) THEN
      t = REAL(count, dp) / REAL(rate, dp)
    ELSE
      t = 0.0_dp
    END IF
  END FUNCTION walltime
END MODULE timers
