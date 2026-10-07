! MPI-IO-backed single-file ensemble output. Enabled with USE_PARALLEL_HDF5=1.
MODULE io_parallel_hdf5
  USE kinds
  USE params, ONLY: SimCtrl
  USE grid, ONLY: RealGrid
  USE potentials, ONLY: pes_on_grid
  USE constants, ONLY: AU_TO_ANGSTROM, AU_TO_CMINV, AU_TO_FS, ANGSTROM_TO_AU
  USE iso_fortran_env, ONLY: error_unit
#ifdef USE_PARALLEL_HDF5
  USE hdf5
  USE mpi, ONLY: MPI_INFO_NULL, MPI_BARRIER
  USE mpi_env, ONLY: comm, rank
#endif
  IMPLICIT NONE

CONTAINS

  SUBROUTINE parallel_h5_initialize(ctrl, g)
    TYPE(SimCtrl), INTENT(IN) :: ctrl
    TYPE(RealGrid), INTENT(IN) :: g
#ifdef USE_PARALLEL_HDF5
    INTEGER(hid_t) :: file_id, fapl_id
    INTEGER :: ierr, mpierr, nframes
    CHARACTER(512) :: filename
    REAL(dp), ALLOCATABLE :: v11(:), v22(:), v12(:), vlower(:), vupper(:)

    nframes = ctrl%nsteps/ctrl%save_every + MERGE(1, 0, ctrl%write_initial)
    WRITE(filename, '(a,".ensemble.h5")') TRIM(ctrl%out_prefix)
    CALL h5open_f(ierr); CALL require_h5(ierr, 'initialize HDF5')
    CALL h5pcreate_f(H5P_FILE_ACCESS_F, fapl_id, ierr); CALL require_h5(ierr, 'create file access list')
    CALL h5pset_fapl_mpio_f(fapl_id, comm, MPI_INFO_NULL, ierr); CALL require_h5(ierr, 'set MPI-IO driver')
    CALL h5fcreate_f(TRIM(filename), H5F_ACC_TRUNC_F, file_id, ierr, access_prp=fapl_id)
    CALL require_h5(ierr, 'create parallel output')
    CALL h5pclose_f(fapl_id, ierr); CALL require_h5(ierr, 'close file access list')

    CALL create_real_dataset(file_id, 'x', (/INT(g%nx,hsize_t)/))
    CALL create_real_dataset(file_id, 'time_fs', (/INT(nframes,hsize_t)/))
    CALL create_real_dataset(file_id, 'psi1_real', &
      (/INT(g%nx,hsize_t), INT(nframes,hsize_t), INT(ctrl%ntraj,hsize_t)/))
    CALL create_real_dataset(file_id, 'psi1_imag', &
      (/INT(g%nx,hsize_t), INT(nframes,hsize_t), INT(ctrl%ntraj,hsize_t)/))
    CALL create_real_dataset(file_id, 'psi2_real', &
      (/INT(g%nx,hsize_t), INT(nframes,hsize_t), INT(ctrl%ntraj,hsize_t)/))
    CALL create_real_dataset(file_id, 'psi2_imag', &
      (/INT(g%nx,hsize_t), INT(nframes,hsize_t), INT(ctrl%ntraj,hsize_t)/))
    CALL create_real_dataset(file_id, 'pop1', (/INT(nframes,hsize_t), INT(ctrl%ntraj,hsize_t)/))
    CALL create_real_dataset(file_id, 'pop2', (/INT(nframes,hsize_t), INT(ctrl%ntraj,hsize_t)/))
    CALL create_real_dataset(file_id, 'xavg1_angstrom', (/INT(nframes,hsize_t), INT(ctrl%ntraj,hsize_t)/))
    CALL create_real_dataset(file_id, 'xavg2_angstrom', (/INT(nframes,hsize_t), INT(ctrl%ntraj,hsize_t)/))

    IF (dynamic_pes(ctrl)) THEN
      CALL create_real_dataset(file_id, 'V11_cm-1', &
        (/INT(g%nx,hsize_t), INT(nframes,hsize_t), INT(ctrl%ntraj,hsize_t)/))
      CALL create_real_dataset(file_id, 'V22_cm-1', &
        (/INT(g%nx,hsize_t), INT(nframes,hsize_t), INT(ctrl%ntraj,hsize_t)/))
      CALL create_real_dataset(file_id, 'V12_cm-1', &
        (/INT(g%nx,hsize_t), INT(nframes,hsize_t), INT(ctrl%ntraj,hsize_t)/))
    ELSE
      CALL create_real_dataset(file_id, 'V11_cm-1', (/INT(g%nx,hsize_t)/))
      CALL create_real_dataset(file_id, 'V22_cm-1', (/INT(g%nx,hsize_t)/))
      CALL create_real_dataset(file_id, 'V12_cm-1', (/INT(g%nx,hsize_t)/))
    END IF

    IF (rank == 0) THEN
      CALL write_full_1d(file_id, 'x', g%x*AU_TO_ANGSTROM)
      IF (.NOT. dynamic_pes(ctrl)) THEN
        ALLOCATE(v11(g%nx), v22(g%nx), v12(g%nx), vlower(g%nx), vupper(g%nx))
        CALL pes_on_grid(ctrl, g%x, v11, v22, v12, vlower, vupper)
        CALL write_full_1d(file_id, 'V11_cm-1', v11*AU_TO_CMINV)
        CALL write_full_1d(file_id, 'V22_cm-1', v22*AU_TO_CMINV)
        CALL write_full_1d(file_id, 'V12_cm-1', v12*AU_TO_CMINV)
        DEALLOCATE(v11, v22, v12, vlower, vupper)
      END IF
    END IF
    CALL MPI_BARRIER(comm, mpierr)
    IF (mpierr /= 0) ERROR STOP 'MPI barrier failed during parallel HDF5 initialization'
    CALL h5fclose_f(file_id, ierr); CALL require_h5(ierr, 'close parallel output')
#else
    WRITE(error_unit, *) 'parallel HDF5 initialize request:', TRIM(ctrl%out_prefix), g%nx
    ERROR STOP 'parallel_hdf5 requires build with USE_PARALLEL_HDF5=1 and parallel HDF5'
#endif
  END SUBROUTINE parallel_h5_initialize

  SUBROUTINE parallel_h5_write(base_ctrl, traj_ctrl, g, psi1, psi2, trajectory, frame, time, active)
    TYPE(SimCtrl), INTENT(IN) :: base_ctrl, traj_ctrl
    TYPE(RealGrid), INTENT(IN) :: g
    COMPLEX(dp), INTENT(IN) :: psi1(:), psi2(:)
    INTEGER, INTENT(IN) :: trajectory, frame
    REAL(dp), INTENT(IN) :: time
    LOGICAL, INTENT(IN) :: active
#ifdef USE_PARALLEL_HDF5
    INTEGER(hid_t) :: file_id, fapl_id
    INTEGER :: ierr, mpierr
    CHARACTER(512) :: filename
    REAL(dp) :: pop1, pop2, xavg1, xavg2
    REAL(dp), ALLOCATABLE :: v11(:), v22(:), v12(:), vlower(:), vupper(:)

    WRITE(filename, '(a,".ensemble.h5")') TRIM(base_ctrl%out_prefix)
    CALL h5pcreate_f(H5P_FILE_ACCESS_F, fapl_id, ierr); CALL require_h5(ierr, 'create file access list')
    CALL h5pset_fapl_mpio_f(fapl_id, comm, MPI_INFO_NULL, ierr); CALL require_h5(ierr, 'set MPI-IO driver')
    CALL h5fopen_f(TRIM(filename), H5F_ACC_RDWR_F, file_id, ierr, access_prp=fapl_id)
    CALL require_h5(ierr, 'open parallel output')
    CALL h5pclose_f(fapl_id, ierr); CALL require_h5(ierr, 'close file access list')

    IF (active) THEN
      CALL observables(g, psi1, pop1, xavg1)
      CALL observables(g, psi2, pop2, xavg2)
      CALL write_slice_3d(file_id, 'psi1_real', REAL(psi1,dp)*SQRT(ANGSTROM_TO_AU), trajectory, frame)
      CALL write_slice_3d(file_id, 'psi1_imag', AIMAG(psi1)*SQRT(ANGSTROM_TO_AU), trajectory, frame)
      CALL write_slice_3d(file_id, 'psi2_real', REAL(psi2,dp)*SQRT(ANGSTROM_TO_AU), trajectory, frame)
      CALL write_slice_3d(file_id, 'psi2_imag', AIMAG(psi2)*SQRT(ANGSTROM_TO_AU), trajectory, frame)
      CALL write_scalar_2d(file_id, 'pop1', pop1, trajectory, frame)
      CALL write_scalar_2d(file_id, 'pop2', pop2, trajectory, frame)
      CALL write_scalar_2d(file_id, 'xavg1_angstrom', xavg1*AU_TO_ANGSTROM, trajectory, frame)
      CALL write_scalar_2d(file_id, 'xavg2_angstrom', xavg2*AU_TO_ANGSTROM, trajectory, frame)
      IF (rank == 0) CALL write_time(file_id, frame, time*AU_TO_FS)
      IF (dynamic_pes(traj_ctrl)) THEN
        ALLOCATE(v11(g%nx), v22(g%nx), v12(g%nx), vlower(g%nx), vupper(g%nx))
        CALL pes_on_grid(traj_ctrl, g%x, v11, v22, v12, vlower, vupper)
        CALL write_slice_3d(file_id, 'V11_cm-1', v11*AU_TO_CMINV, trajectory, frame)
        CALL write_slice_3d(file_id, 'V22_cm-1', v22*AU_TO_CMINV, trajectory, frame)
        CALL write_slice_3d(file_id, 'V12_cm-1', v12*AU_TO_CMINV, trajectory, frame)
        DEALLOCATE(v11, v22, v12, vlower, vupper)
      END IF
    END IF
    CALL MPI_BARRIER(comm, mpierr)
    IF (mpierr /= 0) ERROR STOP 'MPI barrier failed during parallel HDF5 write'
    CALL h5fclose_f(file_id, ierr); CALL require_h5(ierr, 'close parallel output')
#else
    WRITE(error_unit, *) 'parallel HDF5 write request:', TRIM(base_ctrl%out_prefix), &
      TRIM(traj_ctrl%out_prefix), g%nx, SIZE(psi1), SIZE(psi2), trajectory, frame, time, active
    ERROR STOP 'parallel_hdf5 requires build with USE_PARALLEL_HDF5=1 and parallel HDF5'
#endif
  END SUBROUTINE parallel_h5_write

  PURE LOGICAL FUNCTION dynamic_pes(ctrl)
    TYPE(SimCtrl), INTENT(IN) :: ctrl
    dynamic_pes = ctrl%bath_pot_mode /= 0 .AND. ctrl%bath_pot_sigma > 0.0_dp .AND. &
                  (ctrl%bath_pot_reactant .OR. ctrl%bath_pot_product)
  END FUNCTION dynamic_pes

#ifdef USE_PARALLEL_HDF5
  SUBROUTINE observables(g, psi, pop, xavg)
    TYPE(RealGrid), INTENT(IN) :: g
    COMPLEX(dp), INTENT(IN) :: psi(:)
    REAL(dp), INTENT(OUT) :: pop, xavg
    REAL(dp) :: density(SIZE(psi))
    density = ABS(psi)**2
    pop = SUM(density)*g%dx
    IF (pop > TINY(1.0_dp)) THEN
      xavg = SUM(g%x*density)*g%dx/pop
    ELSE
      xavg = 0.0_dp
    END IF
  END SUBROUTINE observables

  SUBROUTINE require_h5(status, action)
    INTEGER, INTENT(IN) :: status
    CHARACTER(*), INTENT(IN) :: action
    IF (status /= 0) THEN
      WRITE(error_unit,'(3a,1x,i0)') 'Parallel HDF5: ', TRIM(action), ' failed', status
      ERROR STOP 'parallel HDF5 failure'
    END IF
  END SUBROUTINE require_h5

  SUBROUTINE create_real_dataset(file_id, name, dims)
    INTEGER(hid_t), INTENT(IN) :: file_id
    CHARACTER(*), INTENT(IN) :: name
    INTEGER(hsize_t), INTENT(IN) :: dims(:)
    INTEGER(hid_t) :: space_id, dataset_id
    INTEGER :: ierr
    CALL h5screate_simple_f(SIZE(dims), dims, space_id, ierr); CALL require_h5(ierr, 'create dataspace')
    CALL h5dcreate_f(file_id, TRIM(name), H5T_NATIVE_DOUBLE, space_id, dataset_id, ierr)
    CALL require_h5(ierr, 'create '//TRIM(name))
    CALL h5dclose_f(dataset_id, ierr); CALL require_h5(ierr, 'close '//TRIM(name))
    CALL h5sclose_f(space_id, ierr); CALL require_h5(ierr, 'close dataspace')
  END SUBROUTINE create_real_dataset

  SUBROUTINE write_full_1d(file_id, name, values)
    INTEGER(hid_t), INTENT(IN) :: file_id
    CHARACTER(*), INTENT(IN) :: name
    REAL(dp), INTENT(IN) :: values(:)
    INTEGER(hid_t) :: dataset_id
    INTEGER(hsize_t) :: dims(1)
    INTEGER :: ierr
    dims = INT(SIZE(values), hsize_t)
    CALL h5dopen_f(file_id, TRIM(name), dataset_id, ierr); CALL require_h5(ierr, 'open '//TRIM(name))
    CALL h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, values, dims, ierr)
    CALL require_h5(ierr, 'write '//TRIM(name))
    CALL h5dclose_f(dataset_id, ierr); CALL require_h5(ierr, 'close '//TRIM(name))
  END SUBROUTINE write_full_1d

  SUBROUTINE write_slice_3d(file_id, name, values, trajectory, frame)
    INTEGER(hid_t), INTENT(IN) :: file_id
    CHARACTER(*), INTENT(IN) :: name
    REAL(dp), INTENT(IN) :: values(:)
    INTEGER, INTENT(IN) :: trajectory, frame
    INTEGER(hid_t) :: dataset_id, file_space, mem_space
    INTEGER(hsize_t) :: count(3), offset(3)
    REAL(dp) :: buffer(SIZE(values),1,1)
    INTEGER :: ierr
    buffer(:,1,1) = values
    count = (/INT(SIZE(values),hsize_t), 1_hsize_t, 1_hsize_t/)
    offset = (/0_hsize_t, INT(frame-1,hsize_t), INT(trajectory-1,hsize_t)/)
    CALL h5dopen_f(file_id, TRIM(name), dataset_id, ierr); CALL require_h5(ierr, 'open '//TRIM(name))
    CALL h5dget_space_f(dataset_id, file_space, ierr); CALL require_h5(ierr, 'get file dataspace')
    CALL h5sselect_hyperslab_f(file_space, H5S_SELECT_SET_F, offset, count, ierr)
    CALL require_h5(ierr, 'select '//TRIM(name))
    CALL h5screate_simple_f(3, count, mem_space, ierr); CALL require_h5(ierr, 'create memory dataspace')
    CALL h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, buffer, count, ierr, &
                    mem_space_id=mem_space, file_space_id=file_space)
    CALL require_h5(ierr, 'write '//TRIM(name))
    CALL h5sclose_f(mem_space, ierr); CALL require_h5(ierr, 'close memory dataspace')
    CALL h5sclose_f(file_space, ierr); CALL require_h5(ierr, 'close file dataspace')
    CALL h5dclose_f(dataset_id, ierr); CALL require_h5(ierr, 'close '//TRIM(name))
  END SUBROUTINE write_slice_3d

  SUBROUTINE write_scalar_2d(file_id, name, value, trajectory, frame)
    INTEGER(hid_t), INTENT(IN) :: file_id
    CHARACTER(*), INTENT(IN) :: name
    REAL(dp), INTENT(IN) :: value
    INTEGER, INTENT(IN) :: trajectory, frame
    INTEGER(hid_t) :: dataset_id, file_space, mem_space
    INTEGER(hsize_t) :: count(2), offset(2)
    REAL(dp) :: buffer(1,1)
    INTEGER :: ierr
    buffer(1,1) = value
    count = 1_hsize_t
    offset = (/INT(frame-1,hsize_t), INT(trajectory-1,hsize_t)/)
    CALL h5dopen_f(file_id, TRIM(name), dataset_id, ierr); CALL require_h5(ierr, 'open '//TRIM(name))
    CALL h5dget_space_f(dataset_id, file_space, ierr); CALL require_h5(ierr, 'get file dataspace')
    CALL h5sselect_hyperslab_f(file_space, H5S_SELECT_SET_F, offset, count, ierr)
    CALL require_h5(ierr, 'select '//TRIM(name))
    CALL h5screate_simple_f(2, count, mem_space, ierr); CALL require_h5(ierr, 'create memory dataspace')
    CALL h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, buffer, count, ierr, &
                    mem_space_id=mem_space, file_space_id=file_space)
    CALL require_h5(ierr, 'write '//TRIM(name))
    CALL h5sclose_f(mem_space, ierr); CALL require_h5(ierr, 'close memory dataspace')
    CALL h5sclose_f(file_space, ierr); CALL require_h5(ierr, 'close file dataspace')
    CALL h5dclose_f(dataset_id, ierr); CALL require_h5(ierr, 'close '//TRIM(name))
  END SUBROUTINE write_scalar_2d

  SUBROUTINE write_time(file_id, frame, value)
    INTEGER(hid_t), INTENT(IN) :: file_id
    INTEGER, INTENT(IN) :: frame
    REAL(dp), INTENT(IN) :: value
    INTEGER(hid_t) :: dataset_id, file_space, mem_space
    INTEGER(hsize_t) :: count(1), offset(1)
    REAL(dp) :: buffer(1)
    INTEGER :: ierr
    buffer = value; count = 1_hsize_t; offset = INT(frame-1,hsize_t)
    CALL h5dopen_f(file_id, 'time_fs', dataset_id, ierr); CALL require_h5(ierr, 'open time')
    CALL h5dget_space_f(dataset_id, file_space, ierr); CALL require_h5(ierr, 'get time dataspace')
    CALL h5sselect_hyperslab_f(file_space, H5S_SELECT_SET_F, offset, count, ierr)
    CALL require_h5(ierr, 'select time')
    CALL h5screate_simple_f(1, count, mem_space, ierr); CALL require_h5(ierr, 'create time memory space')
    CALL h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, buffer, count, ierr, &
                    mem_space_id=mem_space, file_space_id=file_space)
    CALL require_h5(ierr, 'write time')
    CALL h5sclose_f(mem_space, ierr); CALL require_h5(ierr, 'close time memory space')
    CALL h5sclose_f(file_space, ierr); CALL require_h5(ierr, 'close time file space')
    CALL h5dclose_f(dataset_id, ierr); CALL require_h5(ierr, 'close time')
  END SUBROUTINE write_time
#endif

END MODULE io_parallel_hdf5
