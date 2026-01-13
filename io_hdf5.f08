!=============================== io_hdf5.f08 =================================
MODULE io_hdf5
  USE kinds
  USE params
  USE grid
  USE potentials, ONLY: v_two_surface, pes_on_grid
  USE iso_fortran_env, ONLY: output_unit
  USE omp_lib
#ifdef USE_HDF5
  USE hdf5
#ENDIF
  IMPLICIT NONE
CONTAINS

  SUBROUTINE x_expect_1d(x, dx, psi, pop, xavg)
    ! Expectation of position for a (possibly unnormalized) single-surface density.
    !
    !   pop  = \int |psi(x)|^2 dx
    !   xavg = (\int x |psi(x)|^2 dx) / pop
    !
    REAL(dp),    INTENT(IN)  :: x(:)
    REAL(dp),    INTENT(IN)  :: dx
    COMPLEX(dp), INTENT(IN)  :: psi(:)
    REAL(dp),    INTENT(OUT) :: pop
    REAL(dp),    INTENT(OUT) :: xavg

    INTEGER :: i, nx
    REAL(dp) :: rho, xsum

    nx = SIZE(x)
    pop = 0.0_dp
    xsum = 0.0_dp

    !$omp parallel do default(shared) private(i, rho) reduction(+:pop, xsum)
    DO i = 1, nx
      rho = REAL(CONJG(psi(i))*psi(i), dp)
      pop = pop + rho
      xsum = xsum + x(i)*rho
    END DO
    !$omp end parallel do

    pop = pop*dx
    xsum = xsum*dx

    IF (pop > 1.0e-30_dp) THEN
      xavg = xsum/pop
    ELSE
      xavg = 0.0_dp
    END IF
  END SUBROUTINE x_expect_1d

  SUBROUTINE write_snapshot(ctrl, g, t, psi1, psi2, step, rank)
    TYPE(SimCtrl),  INTENT(IN) :: ctrl
    TYPE(RealGrid), INTENT(IN) :: g
    REAL(dp),       INTENT(IN) :: t
    COMPLEX(dp),    INTENT(IN) :: psi1(:), psi2(:)
    INTEGER,        INTENT(IN) :: step, rank
    CHARACTER(256) :: fname

    WRITE(fname, '(a, ".rank",i0, ".h5")') TRIM(ctrl%out_prefix), rank
#ifdef USE_HDF5
    CALL h5_write(ctrl, fname, g, t, psi1, psi2, step)
#ELSE
    CALL ascii_write(ctrl, g, t, psi1, psi2, step, rank)
#ENDIF
  END SUBROUTINE write_snapshot

  SUBROUTINE ascii_write(ctrl, g, t, psi1, psi2, step, rank)
    TYPE(SimCtrl),  INTENT(IN) :: ctrl
    TYPE(RealGrid), INTENT(IN) :: g
    REAL(dp),       INTENT(IN) :: t
    COMPLEX(dp),    INTENT(IN) :: psi1(:), psi2(:)
    INTEGER,        INTENT(IN) :: step, rank
    INTEGER :: i, iu, iu_obs
    CHARACTER(256) :: fname
    CHARACTER(256) :: fobs
    REAL(dp) :: pop1, pop2, xavg1, xavg2

    WRITE(fname, '(a, ".rank",i0, ".s",i6.6, ".dat")') TRIM(ctrl%out_prefix), rank, step
    OPEN(newunit=iu, FILE=fname, ACTION='write', STATUS='replace')
    WRITE(iu, '(a, 1x, f12.6)') '# t =', t

    ! Append observables time series (ASCII/CSV-friendly): step, t, <x>_1, <x>_2
    CALL x_expect_1d(g%x, g%dx, psi1, pop1, xavg1)
    CALL x_expect_1d(g%x, g%dx, psi2, pop2, xavg2)
    WRITE(fobs, '(a, ".rank",i0, ".obs.dat")') TRIM(ctrl%out_prefix), rank
    IF (step == 1) THEN
      OPEN(newunit=iu_obs, FILE=fobs, ACTION='write', STATUS='replace')
      WRITE(iu_obs,'(a)') '# step   t        xavg1       xavg2       pop1        pop2'
    ELSE
      OPEN(newunit=iu_obs, FILE=fobs, ACTION='write', POSITION='append', STATUS='unknown')
    END IF
    WRITE(iu_obs,'(i10,1x,es20.10,1x,es20.10,1x,es20.10,1x,es20.10,1x,es20.10)') &
         step, t, xavg1, xavg2, pop1, pop2
    CLOSE(iu_obs)


    ! Also write the PES once (static) so plotting is possible even in ASCII mode.
    IF (step == 1) THEN
      CALL ascii_write_pes(ctrl, g, rank)
    END IF
    DO i = 1, g%nx
      WRITE(iu, '(f12.6, 1x, es20.10, 1x, es20.10)') g%x(i), REAL(psi1(i)*CONJG(psi1(i))), REAL(psi2(i)*CONJG(psi2(i)))
    END DO
    CLOSE(iu)
  END SUBROUTINE ascii_write

SUBROUTINE ascii_write_pes(ctrl, g, rank)
  TYPE(SimCtrl),  INTENT(IN) :: ctrl
  TYPE(RealGrid), INTENT(IN) :: g
  INTEGER,        INTENT(IN) :: rank
  INTEGER :: i, iu
  CHARACTER(256) :: fname
  REAL(dp) :: v11, v22, v12, vavg, dlt, rad, v_lower, v_upper

  WRITE(fname, '(a, ".rank",i0, ".pes.dat")') TRIM(ctrl%out_prefix), rank
  OPEN(newunit=iu, FILE=fname, ACTION='write', STATUS='replace')
  WRITE(iu,'(a)') '# x  V11  V22  V12  V_lower  V_upper'
  DO i = 1, g%nx
    CALL v_two_surface(ctrl, g%x(i), v11, v22, v12)
    vavg = 0.5_dp*(v11 + v22)
    dlt  = 0.5_dp*(v11 - v22)
    rad  = SQRT(dlt*dlt + v12*v12)
    v_lower = vavg - rad
    v_upper = vavg + rad
    WRITE(iu,'(f12.6, 1x, es20.10, 1x, es20.10, 1x, es20.10, 1x, es20.10, 1x, es20.10)') &
         g%x(i), v11, v22, v12, v_lower, v_upper
  END DO
  CLOSE(iu)
END SUBROUTINE ascii_write_pes


#ifdef USE_HDF5
  SUBROUTINE h5_write(ctrl, fname, g, t, psi1, psi2, step)
    TYPE(SimCtrl),  INTENT(IN) :: ctrl
    CHARACTER(*),   INTENT(IN) :: fname
    TYPE(RealGrid), INTENT(IN) :: g
    REAL(dp),       INTENT(IN) :: t
    COMPLEX(dp),    INTENT(IN) :: psi1(:), psi2(:)
    INTEGER,        INTENT(IN) :: step

    INTEGER(hid_t) :: f, gid
    INTEGER :: ierr
    CHARACTER(64) :: gname

    LOGICAL :: exists


    REAL(dp), ALLOCATABLE :: v11(:), v22(:), v12(:), v_lower(:), v_upper(:)
    REAL(dp), ALLOCATABLE :: vdiab(:,:), vadiab(:,:)
    REAL(dp) :: vavg, dlt, rad
    INTEGER  :: ix, nx

    REAL(dp) :: pop1, pop2, xavg1, xavg2



    ! One-time init of the HDF5 Fortran interface (per-process).
    LOGICAL, SAVE :: hdf5_inited = .FALSE.
    IF (.NOT. hdf5_inited) THEN
      CALL h5open_f(ierr)
      hdf5_inited = .TRUE.
    END IF

    ! If this is the first step of a run, start a fresh file to avoid
    ! collisions with previous runs that used the same filename.
    IF (step == 1) THEN
      CALL h5fcreate_f(fname, H5F_ACC_TRUNC_F, f, ierr)
    ELSE
      CALL h5fopen_f(fname, H5F_ACC_RDWR_F, f, ierr)
      IF (ierr /= 0) CALL h5fcreate_f(fname, H5F_ACC_TRUNC_F, f, ierr)
    END IF
    IF (ierr /= 0) THEN
      WRITE(output_unit,'(a,1x,a,1x,i0)') 'HDF5: failed to open/create file', TRIM(fname), ierr
      RETURN
    END IF

    ! Use a relative group name (root-level). A leading '/' is fine too, but
    ! this makes existence checks a bit more predictable across HDF5 versions.
    WRITE(gname, '("step_",i6.6)') step
    CALL h5lexists_f(f, TRIM(gname), exists, ierr)
    IF (ierr == 0 .AND. exists) THEN
      CALL h5gopen_f(f, TRIM(gname), gid, ierr)
    ELSE
      CALL h5gcreate_f(f, TRIM(gname), gid, ierr)
    END IF
    IF (ierr /= 0) THEN
      WRITE(output_unit,'(a,1x,a,1x,i0)') 'HDF5: failed to open/create group', TRIM(gname), ierr
      CALL h5fclose_f(f, ierr)
      RETURN
    END IF
! --- Static potential energy surfaces on the real-space grid -----------
! We write both diabatic surfaces (reactant/product) and the adiabatic
! eigenvalues of the 2x2 diabatic potential matrix:
!   V_d(x) = [[V11(x), V12(x)],
!            [V12(x), V22(x)]]
!   V_adiab±(x) = 0.5*(V11+V22) ± sqrt( (0.5*(V11-V22))^2 + V12^2 )
!
! Dataset naming:
!   V11, V22, V12     : 1D arrays on x-grid
!   Vdiab(2,nx)       : row 1=V11, row 2=V22
!   Vadiab(2,nx)      : row 1=V_lower, row 2=V_upper
!
    nx = g%nx
    ALLOCATE(v11(nx), v22(nx), v12(nx), v_lower(nx), v_upper(nx))
    ALLOCATE(vdiab(2,nx), vadiab(2,nx))

    CALL pes_on_grid(ctrl, g%x, v11, v22, v12, v_lower, v_upper)

    vdiab(1,:)  = v11
    vdiab(2,:)  = v22
    vadiab(1,:) = v_lower
    vadiab(2,:) = v_upper

    ! ----------------------------------------------------------------------
    CALL write_1d(gid, 'x', g%x)
    CALL write_scalar(gid, 't', t)
    CALL write_c1d(gid, 'psi1', psi1)
    CALL write_c1d(gid, 'psi2', psi2)

    ! ------------------------------------------------------------------
    ! Position expectation values for each diabatic component:
    !   <x>_j = (\int x |psi_j(x)|^2 dx) / (\int |psi_j(x)|^2 dx)
    ! Stored as scalars per step so a time series can be built by reading
    ! step_*/t, step_*/xavg1, step_*/xavg2.
    CALL x_expect_1d(g%x, g%dx, psi1, pop1, xavg1)
    CALL x_expect_1d(g%x, g%dx, psi2, pop2, xavg2)
    CALL write_scalar(gid, 'xavg1', xavg1)
    CALL write_scalar(gid, 'xavg2', xavg2)
    CALL write_scalar(gid, 'pop1', pop1)
    CALL write_scalar(gid, 'pop2', pop2)

    CALL write_1d(gid, 'V11', v11)
    CALL write_1d(gid, 'V22', v22)
    CALL write_1d(gid, 'V12', v12)
    CALL write_1d(gid, 'V_lower', v_lower)
    CALL write_1d(gid, 'V_upper', v_upper)
    CALL write_2d(gid, 'Vdiab', vdiab)
    CALL write_2d(gid, 'Vadiab', vadiab)

    DEALLOCATE(v11, v22, v12, v_lower, v_upper, vdiab, vadiab)

    CALL h5gclose_f(gid, ierr)
    CALL h5fclose_f(f, ierr)

  CONTAINS

    SUBROUTINE write_1d(loc, name, arr)
      INTEGER(hid_t), INTENT(IN) :: loc
      CHARACTER(*),   INTENT(IN) :: name
      REAL(dp),       INTENT(IN) :: arr(:)
      INTEGER(hid_t) :: sid, did
      INTEGER(HSIZE_T) :: dims(1)
      INTEGER :: ierr
      LOGICAL :: exists
      dims(1) = INT(SIZE(arr, 1), KIND=HSIZE_T)
      CALL h5screate_simple_f(1, dims, sid, ierr)
      IF (ierr /= 0) RETURN

      CALL h5lexists_f(loc, TRIM(name), exists, ierr)
      IF (ierr == 0 .AND. exists) THEN
        CALL h5dopen_f(loc, TRIM(name), did, ierr)
      ELSE
        CALL h5dcreate_f(loc, TRIM(name), H5T_NATIVE_DOUBLE, sid, did, ierr)
      END IF
      IF (ierr == 0) THEN
        CALL h5dwrite_f(did, H5T_NATIVE_DOUBLE, arr, dims, ierr)
        CALL h5dclose_f(did, ierr)
      END IF
      CALL h5sclose_f(sid, ierr)
    END SUBROUTINE write_1d

    SUBROUTINE write_2d(loc, name, arr)
      INTEGER(HID_T),   INTENT(IN) :: loc
      CHARACTER(*),     INTENT(IN) :: name
      REAL(dp),         INTENT(IN) :: arr(:,:)

      INTEGER(HSIZE_T) :: dims(2)
      INTEGER(HID_T)   :: sid, did
      INTEGER          :: ierr
      LOGICAL          :: exists

      dims(1) = INT(SIZE(arr, 1), KIND=HSIZE_T)
      dims(2) = INT(SIZE(arr, 2), KIND=HSIZE_T)

      CALL h5screate_simple_f(2, dims, sid, ierr)
      IF (ierr /= 0) RETURN

      CALL h5lexists_f(loc, TRIM(name), exists, ierr)
      IF (ierr == 0 .AND. exists) THEN
        CALL h5dopen_f(loc, TRIM(name), did, ierr)
      ELSE
        CALL h5dcreate_f(loc, TRIM(name), H5T_NATIVE_DOUBLE, sid, did, ierr)
      END IF
      IF (ierr == 0) THEN
        CALL h5dwrite_f(did, H5T_NATIVE_DOUBLE, arr, dims, ierr)
        CALL h5dclose_f(did, ierr)
      END IF

      CALL h5sclose_f(sid, ierr)
    END SUBROUTINE write_2d

    SUBROUTINE write_c1d(loc, name, arr)
      INTEGER(HID_T),   INTENT(IN) :: loc
      CHARACTER(*),     INTENT(IN) :: name
      COMPLEX(dp),      INTENT(IN) :: arr(:)

      REAL(dp), ALLOCATABLE :: tmp(:,:)

      ALLOCATE(tmp(2, SIZE(arr)))
      tmp(1,:) = REAL(arr, KIND=dp)
      tmp(2,:) = AIMAG(arr)

      CALL write_2d(loc, name, tmp)

      DEALLOCATE(tmp)
    END SUBROUTINE write_c1d

    SUBROUTINE write_scalar(loc, name, val)
      INTEGER(hid_t), INTENT(IN) :: loc
      CHARACTER(*),   INTENT(IN) :: name
      REAL(dp),       INTENT(IN) :: val
      INTEGER(hid_t) :: sid, did
      INTEGER(HSIZE_T) :: dims(1)
      INTEGER :: ierr
      LOGICAL :: exists
      dims(1) = 1_HSIZE_T
      CALL h5screate_simple_f(1, dims, sid, ierr)
      IF (ierr /= 0) RETURN

      CALL h5lexists_f(loc, TRIM(name), exists, ierr)
      IF (ierr == 0 .AND. exists) THEN
        CALL h5dopen_f(loc, TRIM(name), did, ierr)
      ELSE
        CALL h5dcreate_f(loc, TRIM(name), H5T_NATIVE_DOUBLE, sid, did, ierr)
      END IF
      IF (ierr == 0) THEN
        CALL h5dwrite_f(did, H5T_NATIVE_DOUBLE, (/val/), dims, ierr)
        CALL h5dclose_f(did, ierr)
      END IF
      CALL h5sclose_f(sid, ierr)
    END SUBROUTINE write_scalar

  END SUBROUTINE h5_write
#ENDIF

END MODULE io_hdf5
