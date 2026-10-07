! Versioned, exact-continuation checkpoints for one trajectory.
MODULE checkpoint
  USE kinds
  USE params, ONLY: SimCtrl
  USE propagator, ONLY: SOProp
  USE langevin, ONLY: LangevinState
  USE potentials, ONLY: bath_pot
  USE rng, ONLY: get_rng_state, set_rng_state
  USE iso_fortran_env, ONLY: error_unit
  USE iso_c_binding, ONLY: c_char, c_int, c_null_char
#ifdef USE_HDF5
  USE hdf5
#endif
  IMPLICIT NONE
  INTEGER, PARAMETER :: CHECKPOINT_SCHEMA = 1

  INTERFACE
    FUNCTION c_rename(old_path, new_path) BIND(C, NAME='rename') RESULT(status)
      IMPORT :: c_char, c_int
      CHARACTER(KIND=c_char), INTENT(IN) :: old_path(*), new_path(*)
      INTEGER(c_int) :: status
    END FUNCTION c_rename
  END INTERFACE

CONTAINS

  SUBROUTINE checkpoint_filename(ctrl, rank, filename)
    TYPE(SimCtrl), INTENT(IN) :: ctrl
    INTEGER, INTENT(IN) :: rank
    CHARACTER(*), INTENT(OUT) :: filename
    WRITE(filename, '(a,".rank",i0,".checkpoint.h5")') TRIM(ctrl%out_prefix), rank
  END SUBROUTINE checkpoint_filename

  LOGICAL FUNCTION checkpoint_exists(ctrl, rank)
    TYPE(SimCtrl), INTENT(IN) :: ctrl
    INTEGER, INTENT(IN) :: rank
    CHARACTER(512) :: filename
    CALL checkpoint_filename(ctrl, rank, filename)
    INQUIRE(FILE=TRIM(filename), EXIST=checkpoint_exists)
  END FUNCTION checkpoint_exists

  SUBROUTINE write_checkpoint(ctrl, prop, bath, rank, tstep, isave, time)
    TYPE(SimCtrl), INTENT(IN) :: ctrl
    TYPE(SOProp), INTENT(IN) :: prop
    TYPE(LangevinState), INTENT(IN) :: bath
    INTEGER, INTENT(IN) :: rank, tstep, isave
    REAL(dp), INTENT(IN) :: time
#ifdef USE_HDF5
    INTEGER(hid_t) :: file_id
    INTEGER :: ierr
    INTEGER, ALLOCATABLE :: rng_state(:)
    CHARACTER(512) :: filename, tmpfile
    CHARACTER(KIND=c_char, LEN=:), ALLOCATABLE :: old_c, new_c
    INTEGER(c_int) :: rename_status

    CALL checkpoint_filename(ctrl, rank, filename)
    tmpfile = TRIM(filename)//'.tmp'
    CALL h5open_f(ierr)
    CALL require_h5(ierr, 'initialize HDF5')
    CALL h5fcreate_f(TRIM(tmpfile), H5F_ACC_TRUNC_F, file_id, ierr)
    CALL require_h5(ierr, 'create checkpoint')

    CALL write_int_scalar(file_id, 'schema_version', CHECKPOINT_SCHEMA)
    CALL write_int_scalar(file_id, 'nx', prop%g%nx)
    CALL write_int_scalar(file_id, 'tstep', tstep)
    CALL write_int_scalar(file_id, 'isave', isave)
    CALL write_real_scalar(file_id, 'time_au', time)
    CALL write_real_scalar(file_id, 'dt_au', ctrl%dt)
    CALL write_real_1d(file_id, 'psi1_real', REAL(prop%psi1, dp))
    CALL write_real_1d(file_id, 'psi1_imag', AIMAG(prop%psi1))
    CALL write_real_1d(file_id, 'psi2_real', REAL(prop%psi2, dp))
    CALL write_real_1d(file_id, 'psi2_imag', AIMAG(prop%psi2))

    CALL write_int_scalar(file_id, 'langevin_enabled', MERGE(1, 0, bath%enabled))
    CALL write_int_scalar(file_id, 'langevin_colored', MERGE(1, 0, bath%colored))
    CALL write_real_scalar(file_id, 'langevin_gamma', bath%gamma)
    CALL write_real_scalar(file_id, 'langevin_tau_c', bath%tau_c)
    CALL write_real_scalar(file_id, 'langevin_a', bath%a)
    CALL write_real_scalar(file_id, 'langevin_s', bath%s)
    CALL write_real_scalar(file_id, 'langevin_z', bath%z)
    CALL write_real_scalar(file_id, 'langevin_y', bath%y)

    CALL write_int_scalar(file_id, 'bath_pot_enabled', MERGE(1, 0, bath_pot%enabled))
    CALL write_int_scalar(file_id, 'bath_pot_colored', MERGE(1, 0, bath_pot%colored))
    CALL write_int_scalar(file_id, 'bath_pot_coupled', MERGE(1, 0, bath_pot%coupled))
    CALL write_real_scalar(file_id, 'bath_pot_a', bath_pot%a)
    CALL write_real_scalar(file_id, 'bath_pot_s', bath_pot%s)
    CALL write_real_1d(file_id, 'bath_pot_rt', bath_pot%rt)

    CALL get_rng_state(rng_state)
    CALL write_int_1d(file_id, 'rng_state', rng_state)
    DEALLOCATE(rng_state)

    CALL h5fclose_f(file_id, ierr)
    CALL require_h5(ierr, 'close checkpoint')
    old_c = TRIM(tmpfile)//c_null_char
    new_c = TRIM(filename)//c_null_char
    rename_status = c_rename(old_c, new_c)
    IF (rename_status /= 0) ERROR STOP 'atomic checkpoint rename failed'
#else
    WRITE(error_unit, *) 'checkpoint request:', TRIM(ctrl%out_prefix), prop%g%nx, &
      bath%enabled, rank, tstep, isave, time
    ERROR STOP 'checkpointing requires an HDF5-enabled build'
#endif
  END SUBROUTINE write_checkpoint

  SUBROUTINE read_checkpoint(ctrl, prop, bath, rank, tstep, isave, time)
    TYPE(SimCtrl), INTENT(IN) :: ctrl
    TYPE(SOProp), INTENT(INOUT) :: prop
    TYPE(LangevinState), INTENT(INOUT) :: bath
    INTEGER, INTENT(IN) :: rank
    INTEGER, INTENT(OUT) :: tstep, isave
    REAL(dp), INTENT(OUT) :: time
#ifdef USE_HDF5
    INTEGER(hid_t) :: file_id
    INTEGER :: ierr, schema, nx, flag
    INTEGER, ALLOCATABLE :: rng_state(:)
    REAL(dp), ALLOCATABLE :: re(:), im(:)
    REAL(dp) :: saved_dt
    CHARACTER(512) :: filename

    CALL checkpoint_filename(ctrl, rank, filename)
    CALL h5open_f(ierr)
    CALL require_h5(ierr, 'initialize HDF5')
    CALL h5fopen_f(TRIM(filename), H5F_ACC_RDONLY_F, file_id, ierr)
    CALL require_h5(ierr, 'open checkpoint')

    CALL read_int_scalar(file_id, 'schema_version', schema)
    IF (schema /= CHECKPOINT_SCHEMA) ERROR STOP 'unsupported checkpoint schema'
    CALL read_int_scalar(file_id, 'nx', nx)
    IF (nx /= prop%g%nx) ERROR STOP 'checkpoint grid-size mismatch'
    CALL read_real_scalar(file_id, 'dt_au', saved_dt)
    IF (ABS(saved_dt-ctrl%dt) > 32.0_dp*EPSILON(ctrl%dt)*MAX(1.0_dp, ABS(ctrl%dt))) &
      ERROR STOP 'checkpoint time-step mismatch'
    CALL read_int_scalar(file_id, 'tstep', tstep)
    CALL read_int_scalar(file_id, 'isave', isave)
    CALL read_real_scalar(file_id, 'time_au', time)

    CALL read_real_1d(file_id, 'psi1_real', re)
    CALL read_real_1d(file_id, 'psi1_imag', im)
    IF (SIZE(re) /= nx .OR. SIZE(im) /= nx) ERROR STOP 'checkpoint psi1 size mismatch'
    prop%psi1 = CMPLX(re, im, dp)
    DEALLOCATE(re, im)
    CALL read_real_1d(file_id, 'psi2_real', re)
    CALL read_real_1d(file_id, 'psi2_imag', im)
    IF (SIZE(re) /= nx .OR. SIZE(im) /= nx) ERROR STOP 'checkpoint psi2 size mismatch'
    prop%psi2 = CMPLX(re, im, dp)
    DEALLOCATE(re, im)

    CALL read_int_scalar(file_id, 'langevin_enabled', flag); bath%enabled = flag /= 0
    CALL read_int_scalar(file_id, 'langevin_colored', flag); bath%colored = flag /= 0
    CALL read_real_scalar(file_id, 'langevin_gamma', bath%gamma)
    CALL read_real_scalar(file_id, 'langevin_tau_c', bath%tau_c)
    CALL read_real_scalar(file_id, 'langevin_a', bath%a)
    CALL read_real_scalar(file_id, 'langevin_s', bath%s)
    CALL read_real_scalar(file_id, 'langevin_z', bath%z)
    CALL read_real_scalar(file_id, 'langevin_y', bath%y)

    CALL read_int_scalar(file_id, 'bath_pot_enabled', flag); bath_pot%enabled = flag /= 0
    CALL read_int_scalar(file_id, 'bath_pot_colored', flag); bath_pot%colored = flag /= 0
    CALL read_int_scalar(file_id, 'bath_pot_coupled', flag); bath_pot%coupled = flag /= 0
    CALL read_real_scalar(file_id, 'bath_pot_a', bath_pot%a)
    CALL read_real_scalar(file_id, 'bath_pot_s', bath_pot%s)
    CALL read_real_1d(file_id, 'bath_pot_rt', re)
    IF (SIZE(re) /= 2) ERROR STOP 'checkpoint potential-bath size mismatch'
    bath_pot%rt = re
    DEALLOCATE(re)

    CALL read_int_1d(file_id, 'rng_state', rng_state)
    CALL set_rng_state(rng_state)
    DEALLOCATE(rng_state)
    CALL h5fclose_f(file_id, ierr)
    CALL require_h5(ierr, 'close checkpoint')
#else
    tstep = 0
    isave = 0
    time = 0.0_dp
    WRITE(error_unit, *) 'restart request:', TRIM(ctrl%out_prefix), prop%g%nx, &
      bath%enabled, rank
    ERROR STOP 'restart requires an HDF5-enabled build'
#endif
  END SUBROUTINE read_checkpoint

#ifdef USE_HDF5
  SUBROUTINE require_h5(status, action)
    INTEGER, INTENT(IN) :: status
    CHARACTER(*), INTENT(IN) :: action
    IF (status /= 0) THEN
      WRITE(error_unit, '(3a,1x,i0)') 'HDF5 checkpoint: ', TRIM(action), ' failed', status
      ERROR STOP 'checkpoint HDF5 failure'
    END IF
  END SUBROUTINE require_h5

  SUBROUTINE write_real_scalar(loc, name, value)
    INTEGER(hid_t), INTENT(IN) :: loc
    CHARACTER(*), INTENT(IN) :: name
    REAL(dp), INTENT(IN) :: value
    CALL write_real_1d(loc, name, (/value/))
  END SUBROUTINE write_real_scalar

  SUBROUTINE write_int_scalar(loc, name, value)
    INTEGER(hid_t), INTENT(IN) :: loc
    CHARACTER(*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: value
    CALL write_int_1d(loc, name, (/value/))
  END SUBROUTINE write_int_scalar

  SUBROUTINE write_real_1d(loc, name, values)
    INTEGER(hid_t), INTENT(IN) :: loc
    CHARACTER(*), INTENT(IN) :: name
    REAL(dp), INTENT(IN) :: values(:)
    INTEGER(hid_t) :: space_id, dataset_id
    INTEGER(hsize_t) :: dims(1)
    INTEGER :: ierr
    dims = INT(SIZE(values), hsize_t)
    CALL h5screate_simple_f(1, dims, space_id, ierr); CALL require_h5(ierr, 'create real dataspace')
    CALL h5dcreate_f(loc, TRIM(name), H5T_NATIVE_DOUBLE, space_id, dataset_id, ierr)
    CALL require_h5(ierr, 'create '//TRIM(name))
    CALL h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, values, dims, ierr)
    CALL require_h5(ierr, 'write '//TRIM(name))
    CALL h5dclose_f(dataset_id, ierr); CALL require_h5(ierr, 'close '//TRIM(name))
    CALL h5sclose_f(space_id, ierr); CALL require_h5(ierr, 'close dataspace')
  END SUBROUTINE write_real_1d

  SUBROUTINE write_int_1d(loc, name, values)
    INTEGER(hid_t), INTENT(IN) :: loc
    CHARACTER(*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: values(:)
    INTEGER(hid_t) :: space_id, dataset_id
    INTEGER(hsize_t) :: dims(1)
    INTEGER :: ierr
    dims = INT(SIZE(values), hsize_t)
    CALL h5screate_simple_f(1, dims, space_id, ierr); CALL require_h5(ierr, 'create integer dataspace')
    CALL h5dcreate_f(loc, TRIM(name), H5T_NATIVE_INTEGER, space_id, dataset_id, ierr)
    CALL require_h5(ierr, 'create '//TRIM(name))
    CALL h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, values, dims, ierr)
    CALL require_h5(ierr, 'write '//TRIM(name))
    CALL h5dclose_f(dataset_id, ierr); CALL require_h5(ierr, 'close '//TRIM(name))
    CALL h5sclose_f(space_id, ierr); CALL require_h5(ierr, 'close dataspace')
  END SUBROUTINE write_int_1d

  SUBROUTINE read_real_scalar(loc, name, value)
    INTEGER(hid_t), INTENT(IN) :: loc
    CHARACTER(*), INTENT(IN) :: name
    REAL(dp), INTENT(OUT) :: value
    REAL(dp), ALLOCATABLE :: values(:)
    CALL read_real_1d(loc, name, values)
    IF (SIZE(values) /= 1) ERROR STOP 'checkpoint scalar shape mismatch'
    value = values(1)
    DEALLOCATE(values)
  END SUBROUTINE read_real_scalar

  SUBROUTINE read_int_scalar(loc, name, value)
    INTEGER(hid_t), INTENT(IN) :: loc
    CHARACTER(*), INTENT(IN) :: name
    INTEGER, INTENT(OUT) :: value
    INTEGER, ALLOCATABLE :: values(:)
    CALL read_int_1d(loc, name, values)
    IF (SIZE(values) /= 1) ERROR STOP 'checkpoint scalar shape mismatch'
    value = values(1)
    DEALLOCATE(values)
  END SUBROUTINE read_int_scalar

  SUBROUTINE read_real_1d(loc, name, values)
    INTEGER(hid_t), INTENT(IN) :: loc
    CHARACTER(*), INTENT(IN) :: name
    REAL(dp), ALLOCATABLE, INTENT(OUT) :: values(:)
    INTEGER(hid_t) :: dataset_id, space_id
    INTEGER(hsize_t) :: dims(1), maxdims(1)
    INTEGER :: ierr
    CALL h5dopen_f(loc, TRIM(name), dataset_id, ierr); CALL require_h5(ierr, 'open '//TRIM(name))
    CALL h5dget_space_f(dataset_id, space_id, ierr); CALL require_h5(ierr, 'get dataspace')
    CALL h5sget_simple_extent_dims_f(space_id, dims, maxdims, ierr)
    IF (ierr < 0) CALL require_h5(ierr, 'get dimensions')
    ALLOCATE(values(INT(dims(1))))
    CALL h5dread_f(dataset_id, H5T_NATIVE_DOUBLE, values, dims, ierr)
    CALL require_h5(ierr, 'read '//TRIM(name))
    CALL h5sclose_f(space_id, ierr); CALL require_h5(ierr, 'close dataspace')
    CALL h5dclose_f(dataset_id, ierr); CALL require_h5(ierr, 'close '//TRIM(name))
  END SUBROUTINE read_real_1d

  SUBROUTINE read_int_1d(loc, name, values)
    INTEGER(hid_t), INTENT(IN) :: loc
    CHARACTER(*), INTENT(IN) :: name
    INTEGER, ALLOCATABLE, INTENT(OUT) :: values(:)
    INTEGER(hid_t) :: dataset_id, space_id
    INTEGER(hsize_t) :: dims(1), maxdims(1)
    INTEGER :: ierr
    CALL h5dopen_f(loc, TRIM(name), dataset_id, ierr); CALL require_h5(ierr, 'open '//TRIM(name))
    CALL h5dget_space_f(dataset_id, space_id, ierr); CALL require_h5(ierr, 'get dataspace')
    CALL h5sget_simple_extent_dims_f(space_id, dims, maxdims, ierr)
    IF (ierr < 0) CALL require_h5(ierr, 'get dimensions')
    ALLOCATE(values(INT(dims(1))))
    CALL h5dread_f(dataset_id, H5T_NATIVE_INTEGER, values, dims, ierr)
    CALL require_h5(ierr, 'read '//TRIM(name))
    CALL h5sclose_f(space_id, ierr); CALL require_h5(ierr, 'close dataspace')
    CALL h5dclose_f(dataset_id, ierr); CALL require_h5(ierr, 'close '//TRIM(name))
  END SUBROUTINE read_int_1d
#endif

END MODULE checkpoint
