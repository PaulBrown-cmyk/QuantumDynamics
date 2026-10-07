!=============================== main.f08 ====================================
PROGRAM qle_1d
  USE kinds
  USE mpi_env
  USE timers
  USE params
  USE grid
  USE rng
  USE fftwrap
  USE propagator
  USE langevin
  USE potentials, ONLY: potentials_bath_init
  USE io_hdf5
  USE omp_lib
  USE iso_fortran_env, ONLY: output_unit, error_unit
  IMPLICIT NONE

  TYPE(SimCtrl)       :: ctrl, ctrlT
  TYPE(RealGrid)      :: g
  TYPE(SOProp)        :: prop
  TYPE(LangevinState) :: L

  INTEGER :: tstep, isave, my_first, my_last, my_count, itraj
  REAL(dp):: t0, t1, t
  INTEGER :: nth
  CHARACTER(256) :: traj_prefix
  CHARACTER(512) :: input_path, arg
  INTEGER :: nargs

  nargs = COMMAND_ARGUMENT_COUNT()
  input_path = 'INPUT.nml'
  IF (nargs == 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    IF (TRIM(arg) == '-h' .OR. TRIM(arg) == '--help') THEN
      CALL print_usage(output_unit)
      STOP
    ELSE IF (LEN_TRIM(arg) == 0 .OR. arg(1:1) == '-') THEN
      WRITE(error_unit,'(2a)') 'Unknown option: ', TRIM(arg)
      CALL print_usage(error_unit)
      ERROR STOP 'bad command line'
    END IF
    input_path = TRIM(arg)
  ELSE IF (nargs > 1) THEN
    CALL print_usage(error_unit)
    ERROR STOP 'too many command-line arguments'
  END IF

  CALL mpi_start()
  CALL read_input(ctrl, TRIM(input_path))

  ! Divide trajectories across ranks
  my_first = (ctrl%ntraj*rank)/nprocs + 1
  my_last  = (ctrl%ntraj*(rank+1))/nprocs
  my_count = MAX(0, my_last - my_first + 1)

  ! Grid & FFT
  CALL build_grid(g, ctrl%nx, ctrl%xmin, ctrl%xmax)
  nth = MAX(1, omp_get_max_threads())
  CALL init_fft_threads()

  IF (rank == 0) THEN
    WRITE(*,*) 'Welcome to the Quantum Dynamics world of Chemistry!'
    WRITE(*,*) '-------------------------------------------------------------------------------------------'
    WRITE(*,*) '                             by Dr. Paul A. Brown                 '
    WRITE(*,*) 'This code simulates quantum dynamics of H-atom transfer with a '
    WRITE(*,*) 'quantum Langevin equation (QGLE). It models transfer in a dissipative '
    WRITE(*,*) 'environment between two diabatic potential-energy surfaces.'
    WRITE(*,*) '-------------------------------------------------------------------------------------------'
  END IF

  IF (rank == 0) THEN
    WRITE(*,'(a, i0, a, i0)') 'MPI ranks: ', nprocs, ', OMP threads: ', nth
    WRITE(*,'(a, i0)')       'Trajectories total: ', ctrl%ntraj
    WRITE(*,'(2a)')           'Input: ', TRIM(input_path)
  END IF
  WRITE(*,'(a, i0, a, i0, a, i0)') 'Rank ', rank, ' handles traj ', my_first, ' .. ', my_last

  t0 = walltime()

  IF (my_count > 0) CALL init_prop(prop, g)

  DO itraj = my_first, my_last
    ! Make a per-trajectory control copy so outputs don't collide
    ctrlT = ctrl
    WRITE(traj_prefix, '(a, ".traj", i6.6)') TRIM(ctrl%out_prefix), itraj
    ctrlT%out_prefix = TRIM(traj_prefix)

    ! Stream depends only on trajectory index, not MPI decomposition.
    CALL seed_stream(ctrlT%seed0 + itraj)

    CALL set_gaussian_packet(prop, ctrlT)
    CALL init_langevin(L, ctrlT, ctrlT%dt)
    CALL potentials_bath_init(ctrlT, ctrlT%dt)

    t = 0.0_dp
    isave = 0
    IF (ctrlT%write_initial) THEN
      CALL write_snapshot(ctrlT, g, t, prop%psi1, prop%psi2, isave, rank, .TRUE.)
    END IF

    DO tstep = 1, ctrlT%nsteps
      CALL step_langevin(ctrlT, prop, L, ctrlT%dt)
      t = t + ctrlT%dt
      IF (MOD(tstep, ctrlT%save_every) == 0) THEN
        isave = isave + 1
        CALL write_snapshot(ctrlT, g, t, prop%psi1, prop%psi2, isave, rank, &
                            isave == 1 .AND. .NOT. ctrlT%write_initial)
      END IF
    END DO

  END DO

  IF (my_count > 0) CALL destroy_prop(prop)

  t1 = walltime()
  IF (rank == 0) WRITE(*,'(a, f10.3)') 'Wall time (s): ', REAL(t1 - t0, dp)

  CALL cleanup_fft_threads()
  CALL mpi_finish()

CONTAINS

  SUBROUTINE print_usage(unit)
    INTEGER, INTENT(IN) :: unit
    WRITE(unit,'(a)') 'Usage: qle_1d [INPUT.nml]'
    WRITE(unit,'(a)') '       qle_1d --help'
  END SUBROUTINE print_usage
END PROGRAM qle_1d
