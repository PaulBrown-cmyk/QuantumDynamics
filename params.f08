!=============================== params.f08 ==================================
MODULE params
  USE kinds
  IMPLICIT NONE
  ! Physical constants (atomic units by default)
  REAL(dp), PARAMETER:: hbar = 1.0_dp, me = 1.0_dp

  ! Input container
  TYPE:: SimCtrl
     INTEGER            :: nx = 2048
     REAL(dp)           :: xmin = -20.0_dp, xmax = 20.0_dp
     REAL(dp)           :: dt   = 0.05_dp
     INTEGER            :: nsteps = 4000
     INTEGER            :: save_every = 10
     INTEGER            :: nsave = 0         ! computed
     INTEGER            :: ntraj = 16        ! independent Langevin realizations (MPI-distributed)

     REAL(dp)           :: beta = 5.0_dp     ! inverse temperature (1/E_h)
     REAL(dp)           :: gamma = 0.02_dp   ! friction strength (1/time)
     INTEGER            :: seed0 = 13579

     CHARACTER(16)      :: pot_model = 'harmonic' ! 'harmonic' or 'anharmonic'

     ! Harmonic parameters
     REAL(dp)           :: w0   = 0.02_dp    ! frequency (a.u.) for harmonic test

     ! Two-parabola + Gaussian coupling (anharmonic HAT)
     REAL(dp)           :: k1 = 0.02_dp, k2 = 0.02_dp
     REAL(dp)           :: x1 = -4.0_dp, x2 = +4.0_dp
     REAL(dp)           :: v1_shift = 0.0_dp, v2_shift = 0.0_dp
     REAL(dp)           :: v12 = 0.01_dp     ! coupling amplitude
     REAL(dp)           :: sigma = 1.0_dp    ! coupling width

     ! Initial wavepacket
     REAL(dp)           :: x0 = -8.0_dp, p0 = 1.2_dp, sigma0 = 1.0_dp

     ! Langevin noise controls (optional; defaults keep old INPUT.nml working)
     LOGICAL            :: use_colored = .false.
     CHARACTER(16)      :: kernel      = 'white'     ! 'white' or 'lorentz'
     REAL(dp)           :: fwhm        = 0.0_dp       ! for colored noise (Lorentz)
 
     ! Stochastic potential modulation (bath acting on diabatic wells)
     ! bath_pot_mode: 0 none, 1 coordinate shift, 2 curvature modulation, 3 enthalpy modulation
     INTEGER            :: bath_pot_mode      = 0
     LOGICAL            :: bath_pot_reactant  = .false.
     LOGICAL            :: bath_pot_product   = .false.
     LOGICAL            :: bath_pot_coupled   = .true.   ! same R_t for both surfaces
     LOGICAL            :: bath_pot_colored   = .false.  ! OU colored vs white
     REAL(dp)           :: bath_pot_sigma     = 0.0_dp   ! std dev of R_t (units depend on mode)
     REAL(dp)           :: bath_pot_fwhm      = 0.0_dp   ! if colored: tau_c = 2/fwhm

     ! IO
     CHARACTER(128)     :: out_prefix = 'run'
     LOGICAL            :: hdf5 = .true.
     LOGICAL            :: want_coupling=.false.
     LOGICAL            :: use_exponential=.false.
  END TYPE

CONTAINS

  SUBROUTINE read_input(ctrl)
    TYPE(SimCtrl), INTENT(INOUT) :: ctrl
  
    ! Local mirrors for NAMELIST
    INTEGER :: nx, nsteps, save_every, ntraj, seed0
    REAL(dp) :: xmin, xmax, dt, beta, gamma, w0
    REAL(dp) :: k1, k2, x1, x2, v1_shift, v2_shift, v12, sigma
    REAL(dp) :: x0, p0, sigma0, fwhm
    INTEGER :: bath_pot_mode
    REAL(dp) :: bath_pot_sigma, bath_pot_fwhm
    LOGICAL :: bath_pot_reactant, bath_pot_product, bath_pot_coupled, bath_pot_colored
    LOGICAL :: use_colored, hdf5, want_coupling, use_exponential
    CHARACTER(16)  :: pot_model
    CHARACTER(16)  :: kernel
    CHARACTER(128) :: out_prefix
  
    namelist /qle/ nx, xmin, xmax, dt, nsteps, save_every, ntraj, beta, gamma, &
                   seed0, pot_model, w0, k1, k2, x1, x2, v1_shift, v2_shift, &
                   v12, sigma, x0, p0, sigma0, use_colored, kernel, fwhm, out_prefix, hdf5, &
                   bath_pot_mode, bath_pot_sigma, bath_pot_fwhm, bath_pot_reactant, bath_pot_product, &
                   bath_pot_coupled, bath_pot_colored, want_coupling, use_exponential
  
    INTEGER :: iu, ios
  
    ! Initialize locals from ctrl defaults
    nx         = ctrl%nx
    xmin       = ctrl%xmin
    xmax       = ctrl%xmax
    dt         = ctrl%dt
    nsteps     = ctrl%nsteps
    save_every = ctrl%save_every
    ntraj      = ctrl%ntraj
    beta       = ctrl%beta
    gamma      = ctrl%gamma
    seed0      = ctrl%seed0
    pot_model  = ctrl%pot_model
    w0         = ctrl%w0
    k1         = ctrl%k1
    k2         = ctrl%k2
    x1         = ctrl%x1
    x2         = ctrl%x2
    v1_shift   = ctrl%v1_shift
    v2_shift   = ctrl%v2_shift
    v12        = ctrl%v12
    sigma      = ctrl%sigma
    x0         = ctrl%x0
    p0         = ctrl%p0
    sigma0     = ctrl%sigma0
    use_colored= ctrl%use_colored
    kernel     = ctrl%kernel
    fwhm       = ctrl%fwhm
    bath_pot_mode     = ctrl%bath_pot_mode
    bath_pot_sigma    = ctrl%bath_pot_sigma
    bath_pot_fwhm     = ctrl%bath_pot_fwhm
    bath_pot_reactant = ctrl%bath_pot_reactant
    bath_pot_product  = ctrl%bath_pot_product
    bath_pot_coupled  = ctrl%bath_pot_coupled
    bath_pot_colored  = ctrl%bath_pot_colored
    out_prefix = ctrl%out_prefix
    hdf5       = ctrl%hdf5
    want_coupling = ctrl%want_coupling
    use_exponential = ctrl%use_exponential
  
    OPEN(newunit=iu, FILE='INPUT.nml', STATUS='old', ACTION='read', IOSTAT=ios)
    IF (ios == 0) THEN
      READ(iu, nml=qle, IOSTAT=ios)
      CLOSE(iu)
    END IF
  
    ! Copy locals back into ctrl
    ctrl%nx         = nx
    ctrl%xmin       = xmin
    ctrl%xmax       = xmax
    ctrl%dt         = dt
    ctrl%nsteps     = nsteps
    ctrl%save_every = save_every
    ctrl%ntraj      = ntraj
    ctrl%beta       = beta
    ctrl%gamma      = gamma
    ctrl%seed0      = seed0
    ctrl%pot_model  = pot_model
    ctrl%w0         = w0
    ctrl%k1         = k1
    ctrl%k2         = k2
    ctrl%x1         = x1
    ctrl%x2         = x2
    ctrl%v1_shift   = v1_shift
    ctrl%v2_shift   = v2_shift
    ctrl%v12        = v12
    ctrl%sigma      = sigma
    ctrl%x0         = x0
    ctrl%p0         = p0
    ctrl%sigma0     = sigma0
    ctrl%use_colored= use_colored
    ctrl%kernel     = kernel
    ctrl%fwhm       = fwhm
    ctrl%bath_pot_mode     = bath_pot_mode
    ctrl%bath_pot_sigma    = bath_pot_sigma
    ctrl%bath_pot_fwhm     = bath_pot_fwhm
    ctrl%bath_pot_reactant = bath_pot_reactant
    ctrl%bath_pot_product  = bath_pot_product
    ctrl%bath_pot_coupled  = bath_pot_coupled
    ctrl%bath_pot_colored  = bath_pot_colored
    ctrl%out_prefix = out_prefix
    ctrl%hdf5       = hdf5
    ctrl%want_coupling  = want_coupling 
    ctrl%use_exponential  = use_exponential 
  
    ctrl%nsave = 1 + ctrl%nsteps / ctrl%save_every
  END SUBROUTINE read_input


END MODULE params
