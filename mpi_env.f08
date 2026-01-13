
!=============================== mpi_env.f08 =================================
MODULE mpi_env
  USE kinds
  USE iso_fortran_env, ONLY: error_unit
  USE mpi, ONLY: MPI_Init, MPI_Finalize, MPI_Comm_rank, MPI_Comm_size, MPI_COMM_WORLD
  IMPLICIT NONE
  INTEGER:: comm, nprocs, rank
CONTAINS

  SUBROUTINE mpi_start()
    INTEGER:: ierr
    CALL MPI_INIT(ierr)
    comm = MPI_COMM_WORLD
    CALL MPI_COMM_SIZE(comm, nprocs, ierr)
    CALL MPI_COMM_RANK(comm, rank,  ierr)
  END SUBROUTINE

  SUBROUTINE mpi_finish()
    INTEGER:: ierr
    CALL MPI_FINALIZE(ierr)
  END SUBROUTINE

  SUBROUTINE abort_here(msg)
    CHARACTER(*), INTENT(IN):: msg
    INTEGER:: ierr
    IF (rank == 0) WRITE(error_unit, *) 'FATAL: ', trim(msg)
    CALL MPI_ABORT(comm, 1, ierr)
  END SUBROUTINE


END MODULE mpi_env
