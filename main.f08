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
  IMPLICIT NONE

  TYPE(SimCtrl)       :: ctrl, ctrlT
  TYPE(RealGrid)      :: g
  TYPE(SOProp)        :: prop
  TYPE(LangevinState) :: L

  INTEGER :: tstep, isave, my_first, my_last, my_count, itraj
  REAL(dp):: t0, t1, t, xi
  INTEGER :: nth
  CHARACTER(256) :: traj_prefix

  CALL mpi_start()
  CALL read_input(ctrl)

  ! Divide trajectories across ranks
  my_first = (ctrl%ntraj*rank)/nprocs + 1
  my_last  = (ctrl%ntraj*(rank+1))/nprocs
  my_count = MAX(0, my_last - my_first + 1)

  ! Grid & FFT
  CALL build_grid(g, ctrl%nx, ctrl%xmin, ctrl%xmax)
  nth = MAX(1, omp_get_max_threads())
  CALL fft_init_threads(nth)

  IF (rank == 0) THEN
    WRITE(*,*) 'Welcome to the Quantum Dynamics world of Chemistry!'
    WRITE(*,*) '-------------------------------------------------------------------------------------------'
    WRITE(*,*) '                             by Dr. Paul A. Brown                 '
    WRITE(*,*) 'This  code simulates the quantum dynamics of H-atom transfer with a ' 
    WRITE(*,*) 'quantum Langevin eqation (QGLE). We model the dynamics of transfer '
    WRITE(*,*) 'within a dissipative environment within the harmonic approximation '
    WRITE(*,*) 'between two dibatic potential energy surfaces.'
    WRITE(*,*) '-------------------------------------------------------------------------------------------'
  END IF

  IF (rank == 0) THEN
    WRITE(*,'(a, i0, a, i0)') 'MPI ranks: ', nprocs, ', OMP threads: ', nth
    WRITE(*,'(a, i0)')       'Trajectories total: ', ctrl%ntraj
  END IF
  WRITE(*,'(a, i0, a, i0, a, i0)') 'Rank ', rank, ' handles traj ', my_first, ' .. ', my_last

  t0 = walltime()

  DO itraj = my_first, my_last
    ! Make a per-trajectory control copy so outputs don't collide
    ctrlT = ctrl
    WRITE(traj_prefix, '(a, ".traj", i6.6)') TRIM(ctrl%out_prefix), itraj
    ctrlT%out_prefix = TRIM(traj_prefix)

    CALL seed_stream(ctrlT%seed0 + 100000*rank + itraj)

    CALL init_prop(prop, g)
    CALL set_gaussian_packet(prop, ctrlT)
    CALL init_langevin(L, ctrlT, ctrlT%dt)
    CALL potentials_bath_init(ctrlT, ctrlT%dt)

    t = 0.0_dp
    isave = 0

    DO tstep = 1, ctrlT%nsteps
      CALL next_kick(L, ctrlT, ctrlT%dt, xi)
      CALL step_split_na(ctrlT, prop, ctrlT%dt, ctrlT%gamma, xi)
      t = t + ctrlT%dt
      IF (MOD(tstep, ctrlT%save_every) == 0) THEN
        isave = isave + 1
        CALL write_snapshot(ctrlT, g, t, prop%psi1, prop%psi2, isave, rank)
      END IF
    END DO

    CALL destroy_prop(prop)
  END DO

  t1 = walltime()
  IF (rank == 0) WRITE(*,'(a, f10.3)') 'Wall time (s): ', REAL(t1 - t0, dp)

  CALL fft_cleanup_threads()
  CALL mpi_finish()
END PROGRAM qle_1d
