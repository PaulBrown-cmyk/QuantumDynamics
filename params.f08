!=============================== params.f08 ==================================
MODULE params
  USE kinds
  USE constants, ONLY: ANGSTROM_TO_AU, CMINV_TO_AU, FS_TO_AU, AMU_TO_AU, &
                       BOLTZMANN_AU_PER_K
  USE iso_fortran_env, ONLY: error_unit
  IMPLICIT NONE

  ! Input container.
  ! read_input() ingests chemistry-style units from INPUT.nml and converts them
  ! immediately to internal atomic units.
  TYPE:: SimCtrl
     INTEGER            :: nx = 2048
     REAL(dp)           :: xmin = -10.0_dp, xmax = 10.0_dp
     REAL(dp)           :: dt   = 0.05_dp
     INTEGER            :: nsteps = 4000
     INTEGER            :: save_every = 10
     INTEGER            :: nsave = 0         ! computed
     INTEGER            :: ntraj = 16        ! independent Langevin realizations (MPI-distributed)

     REAL(dp)           :: temperature_k = 300.0_dp ! physical input, kelvin
     REAL(dp)           :: mass_amu = 1.00784_dp     ! physical input, unified atomic mass units
     REAL(dp)           :: beta = 0.0_dp             ! derived internal inverse energy, 1/E_h
     REAL(dp)           :: mass = 0.0_dp             ! derived internal mass, electron masses
     REAL(dp)           :: gamma = 0.02_dp           ! input fs^-1; internal 1/atomic-time after read
     LOGICAL            :: damp_reactant = .true.    ! apply momentum bath to diabatic state 1
     LOGICAL            :: damp_product  = .true.    ! apply momentum bath to diabatic state 2
     INTEGER            :: seed0 = 13579

     CHARACTER(16)      :: pot_model = 'harmonic' ! 'harmonic' or 'anharmonic'

     ! Two-parabola + Gaussian coupling (anharmonic HAT)
     ! k1/k2 are force constants in cm^-1 / Å^2 in INPUT.nml and are converted on read.
     REAL(dp)           :: k1 = 0.02_dp, k2 = 0.02_dp
     REAL(dp)           :: x1 = -4.0_dp, x2 = +4.0_dp
     REAL(dp)           :: v1_shift = 0.0_dp, v2_shift = 0.0_dp
     REAL(dp)           :: c4_1 = 0.0_dp, c4_2 = 0.0_dp ! input cm^-1/angstrom^4
     REAL(dp)           :: v12 = 0.01_dp     ! coupling amplitude
     REAL(dp)           :: sigma = 1.0_dp    ! coupling width

     ! Initial wavepacket
     REAL(dp)           :: x0 = -8.0_dp, p0 = 1.2_dp, sigma0 = 1.0_dp

     ! Optional smooth absorbing layers at both periodic-grid edges.
     LOGICAL            :: use_absorber = .false.
     REAL(dp)           :: absorber_width = 0.5_dp ! input angstrom
     REAL(dp)           :: absorber_rate = 1.0_dp  ! input fs^-1
     INTEGER            :: absorber_power = 4

     ! Langevin noise controls
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
     LOGICAL            :: write_initial = .true.
     INTEGER            :: checkpoint_every = 0 ! steps; 0 disables checkpoints
     LOGICAL            :: restart_from_checkpoint = .false.
     LOGICAL            :: parallel_hdf5 = .false.
     LOGICAL            :: want_coupling=.false.
     LOGICAL            :: use_exponential=.false.
     LOGICAL            :: units_converted=.false.
  END TYPE

CONTAINS

  SUBROUTINE read_input(ctrl, filename)
    TYPE(SimCtrl), INTENT(INOUT) :: ctrl
    CHARACTER(*), OPTIONAL, INTENT(IN) :: filename
  
    ! Local mirrors for NAMELIST
    INTEGER :: nx, nsteps, save_every, ntraj, seed0, absorber_power, checkpoint_every
    REAL(dp) :: xmin, xmax, dt, temperature_k, mass_amu, gamma
    REAL(dp) :: k1, k2, x1, x2, v1_shift, v2_shift, c4_1, c4_2, v12, sigma
    REAL(dp) :: x0, p0, sigma0, fwhm, absorber_width, absorber_rate
    INTEGER :: bath_pot_mode
    REAL(dp) :: bath_pot_sigma, bath_pot_fwhm
    LOGICAL :: bath_pot_reactant, bath_pot_product, bath_pot_coupled, bath_pot_colored
    LOGICAL :: use_colored, hdf5, want_coupling, use_exponential, write_initial, use_absorber
    LOGICAL :: restart_from_checkpoint, parallel_hdf5
    LOGICAL :: damp_reactant, damp_product
    CHARACTER(16)  :: pot_model
    CHARACTER(16)  :: kernel
    CHARACTER(128) :: out_prefix
  
    namelist /qle/ nx, xmin, xmax, dt, nsteps, save_every, ntraj, temperature_k, mass_amu, gamma, &
                   damp_reactant, damp_product, &
                   seed0, pot_model, k1, k2, x1, x2, v1_shift, v2_shift, c4_1, c4_2, &
                   v12, sigma, x0, p0, sigma0, use_absorber, absorber_width, absorber_rate, &
                   absorber_power, use_colored, kernel, fwhm, out_prefix, hdf5, write_initial, &
                   checkpoint_every, restart_from_checkpoint, parallel_hdf5, &
                   bath_pot_mode, bath_pot_sigma, bath_pot_fwhm, bath_pot_reactant, bath_pot_product, &
                   bath_pot_coupled, bath_pot_colored, want_coupling, use_exponential
  
    INTEGER :: iu, ios
    CHARACTER(512) :: iomsg, input_file

    input_file = 'INPUT.nml'
    IF (PRESENT(filename)) input_file = TRIM(filename)
  
    ! Initialize locals from ctrl defaults
    nx         = ctrl%nx
    xmin       = ctrl%xmin
    xmax       = ctrl%xmax
    dt         = ctrl%dt
    nsteps     = ctrl%nsteps
    save_every = ctrl%save_every
    ntraj      = ctrl%ntraj
    temperature_k = ctrl%temperature_k
    mass_amu   = ctrl%mass_amu
    gamma      = ctrl%gamma
    damp_reactant = ctrl%damp_reactant
    damp_product  = ctrl%damp_product
    seed0      = ctrl%seed0
    pot_model  = ctrl%pot_model
    k1         = ctrl%k1
    k2         = ctrl%k2
    x1         = ctrl%x1
    x2         = ctrl%x2
    v1_shift   = ctrl%v1_shift
    v2_shift   = ctrl%v2_shift
    c4_1       = ctrl%c4_1
    c4_2       = ctrl%c4_2
    v12        = ctrl%v12
    sigma      = ctrl%sigma
    x0         = ctrl%x0
    p0         = ctrl%p0
    sigma0     = ctrl%sigma0
    use_absorber = ctrl%use_absorber
    absorber_width = ctrl%absorber_width
    absorber_rate = ctrl%absorber_rate
    absorber_power = ctrl%absorber_power
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
    write_initial = ctrl%write_initial
    checkpoint_every = ctrl%checkpoint_every
    restart_from_checkpoint = ctrl%restart_from_checkpoint
    parallel_hdf5 = ctrl%parallel_hdf5
    want_coupling = ctrl%want_coupling
    use_exponential = ctrl%use_exponential
  
    OPEN(newunit=iu, FILE=TRIM(input_file), STATUS='old', ACTION='read', IOSTAT=ios, IOMSG=iomsg)
    IF (ios /= 0) THEN
      WRITE(error_unit,'(a,1x,a,2a)') 'Cannot open input file', TRIM(input_file), ': ', TRIM(iomsg)
      ERROR STOP 'input open failed'
    END IF
    READ(iu, nml=qle, IOSTAT=ios, IOMSG=iomsg)
    CLOSE(iu)
    IF (ios /= 0) THEN
      WRITE(error_unit,'(a,1x,a,2a)') 'Cannot parse input file', TRIM(input_file), ': ', TRIM(iomsg)
      ERROR STOP 'input parse failed'
    END IF
  
    ! Copy locals back into ctrl
    ctrl%nx         = nx
    ctrl%xmin       = xmin
    ctrl%xmax       = xmax
    ctrl%dt         = dt
    ctrl%nsteps     = nsteps
    ctrl%save_every = save_every
    ctrl%ntraj      = ntraj
    ctrl%temperature_k = temperature_k
    ctrl%mass_amu   = mass_amu
    ctrl%gamma      = gamma
    ctrl%damp_reactant = damp_reactant
    ctrl%damp_product  = damp_product
    ctrl%seed0      = seed0
    ctrl%pot_model  = lower_ascii(ADJUSTL(TRIM(pot_model)))
    ctrl%k1         = k1
    ctrl%k2         = k2
    ctrl%x1         = x1
    ctrl%x2         = x2
    ctrl%v1_shift   = v1_shift
    ctrl%v2_shift   = v2_shift
    ctrl%c4_1       = c4_1
    ctrl%c4_2       = c4_2
    ctrl%v12        = v12
    ctrl%sigma      = sigma
    ctrl%x0         = x0
    ctrl%p0         = p0
    ctrl%sigma0     = sigma0
    ctrl%use_absorber = use_absorber
    ctrl%absorber_width = absorber_width
    ctrl%absorber_rate = absorber_rate
    ctrl%absorber_power = absorber_power
    ctrl%use_colored= use_colored
    ctrl%kernel     = lower_ascii(ADJUSTL(TRIM(kernel)))
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
    ctrl%write_initial = write_initial
    ctrl%checkpoint_every = checkpoint_every
    ctrl%restart_from_checkpoint = restart_from_checkpoint
    ctrl%parallel_hdf5 = parallel_hdf5
    ctrl%want_coupling  = want_coupling 
    ctrl%use_exponential  = use_exponential

    CALL validate_physical_input(ctrl)
    CALL convert_input_units(ctrl)
    ctrl%nsave = ctrl%nsteps / ctrl%save_every
  END SUBROUTINE read_input


  SUBROUTINE validate_physical_input(ctrl)
    TYPE(SimCtrl), INTENT(IN) :: ctrl

    IF (ctrl%nx < 2) ERROR STOP 'nx must be at least 2'
    IF (ctrl%xmax <= ctrl%xmin) ERROR STOP 'xmax must exceed xmin'
    IF (ctrl%dt <= 0.0_dp) ERROR STOP 'dt must be positive'
    IF (ctrl%nsteps < 0) ERROR STOP 'nsteps must be nonnegative'
    IF (ctrl%save_every <= 0) ERROR STOP 'save_every must be positive'
    IF (ctrl%ntraj <= 0) ERROR STOP 'ntraj must be positive'
    IF (ctrl%checkpoint_every < 0) ERROR STOP 'checkpoint_every must be nonnegative'
    IF (ctrl%restart_from_checkpoint .AND. ctrl%checkpoint_every <= 0) &
      ERROR STOP 'restart_from_checkpoint requires checkpoint_every > 0'
    IF (ctrl%parallel_hdf5 .AND. .NOT. ctrl%hdf5) &
      ERROR STOP 'parallel_hdf5 requires hdf5=.true.'
    IF (ctrl%parallel_hdf5 .AND. ctrl%restart_from_checkpoint) &
      ERROR STOP 'parallel_hdf5 and per-trajectory restart are mutually exclusive modes'
    IF (ctrl%temperature_k <= 0.0_dp) ERROR STOP 'temperature_k must be positive'
    IF (ctrl%mass_amu <= 0.0_dp) ERROR STOP 'mass_amu must be positive'
    IF (ctrl%gamma < 0.0_dp) ERROR STOP 'gamma must be nonnegative'
    IF (ctrl%sigma0 <= 0.0_dp) ERROR STOP 'sigma0 must be positive'
    IF (ctrl%use_absorber) THEN
      IF (ctrl%absorber_width <= 0.0_dp) ERROR STOP 'absorber_width must be positive'
      IF (ctrl%absorber_width >= 0.5_dp*(ctrl%xmax-ctrl%xmin)) &
        ERROR STOP 'absorber_width must be less than half the box width'
      IF (ctrl%absorber_rate <= 0.0_dp) ERROR STOP 'absorber_rate must be positive'
      IF (ctrl%absorber_power < 1) ERROR STOP 'absorber_power must be at least 1'
    END IF
    IF (ctrl%want_coupling .AND. ABS(ctrl%v12) > 0.0_dp .AND. ctrl%sigma <= 0.0_dp) &
      ERROR STOP 'coupling sigma must be positive'
    IF (ctrl%k1 < 0.0_dp .OR. ctrl%k2 < 0.0_dp) &
      ERROR STOP 'k1 and k2 must be nonnegative'
    IF (ctrl%c4_1 < 0.0_dp .OR. ctrl%c4_2 < 0.0_dp) &
      ERROR STOP 'c4_1 and c4_2 must be nonnegative'
    IF (ctrl%bath_pot_mode < 0 .OR. ctrl%bath_pot_mode > 3) &
      ERROR STOP 'bath_pot_mode must be 0, 1, 2, or 3'
    IF (ctrl%bath_pot_sigma < 0.0_dp) ERROR STOP 'bath_pot_sigma must be nonnegative'
    IF (ctrl%use_colored .AND. ctrl%gamma > 0.0_dp .AND. &
        (ctrl%damp_reactant .OR. ctrl%damp_product)) THEN
      IF (TRIM(ctrl%kernel) /= 'lorentz') ERROR STOP 'colored kernel must be lorentz'
      IF (ctrl%fwhm <= 0.0_dp) ERROR STOP 'colored fwhm must be positive'
    END IF
    IF (ctrl%bath_pot_mode /= 0 .AND. ctrl%bath_pot_colored .AND. &
        (ctrl%bath_pot_reactant .OR. ctrl%bath_pot_product) .AND. &
        ctrl%bath_pot_sigma > 0.0_dp .AND. &
        ctrl%bath_pot_fwhm <= 0.0_dp) ERROR STOP 'colored bath_pot_fwhm must be positive'
  END SUBROUTINE validate_physical_input


  SUBROUTINE convert_input_units(ctrl)
    TYPE(SimCtrl), INTENT(INOUT) :: ctrl

    IF (ctrl%units_converted) ERROR STOP 'input units already converted'

    ! Thermodynamic inputs: kelvin and unified atomic mass -> atomic units.
    ctrl%beta = 1.0_dp/(BOLTZMANN_AU_PER_K*ctrl%temperature_k)
    ctrl%mass = ctrl%mass_amu*AMU_TO_AU

    ! Lengths: angstrom -> bohr
    ctrl%xmin = ctrl%xmin * ANGSTROM_TO_AU
    ctrl%xmax = ctrl%xmax * ANGSTROM_TO_AU
    ctrl%x1   = ctrl%x1   * ANGSTROM_TO_AU
    ctrl%x2   = ctrl%x2   * ANGSTROM_TO_AU
    ctrl%x0   = ctrl%x0   * ANGSTROM_TO_AU
    ctrl%sigma = ctrl%sigma * ANGSTROM_TO_AU
    ctrl%sigma0 = ctrl%sigma0 * ANGSTROM_TO_AU
    ctrl%absorber_width = ctrl%absorber_width * ANGSTROM_TO_AU

    ! Time / rates: fs -> atomic time, fs^-1 -> atomic inverse time
    ctrl%dt    = ctrl%dt * FS_TO_AU
    ctrl%gamma = ctrl%gamma / FS_TO_AU
    ctrl%fwhm  = ctrl%fwhm / FS_TO_AU
    ctrl%bath_pot_fwhm = ctrl%bath_pot_fwhm / FS_TO_AU
    ctrl%absorber_rate = ctrl%absorber_rate / FS_TO_AU

    ! Frequencies / energies: cm^-1 -> Hartree
    ctrl%v1_shift  = ctrl%v1_shift * CMINV_TO_AU
    ctrl%v2_shift  = ctrl%v2_shift * CMINV_TO_AU
    ctrl%v12       = ctrl%v12 * CMINV_TO_AU
    ctrl%k1        = ctrl%k1 * CMINV_TO_AU / (ANGSTROM_TO_AU * ANGSTROM_TO_AU)
    ctrl%k2        = ctrl%k2 * CMINV_TO_AU / (ANGSTROM_TO_AU * ANGSTROM_TO_AU)
    ctrl%c4_1      = ctrl%c4_1 * CMINV_TO_AU / ANGSTROM_TO_AU**4
    ctrl%c4_2      = ctrl%c4_2 * CMINV_TO_AU / ANGSTROM_TO_AU**4

    ! Momentum / wave number: 1/angstrom -> 1/bohr
    ctrl%p0 = ctrl%p0 / ANGSTROM_TO_AU

    ! Bath modulation amplitudes inherit the mode-specific chemistry unit.
    SELECT CASE (ctrl%bath_pot_mode)
    CASE (1)
      ctrl%bath_pot_sigma = ctrl%bath_pot_sigma * ANGSTROM_TO_AU
    CASE (2)
      ctrl%bath_pot_sigma = ctrl%bath_pot_sigma * CMINV_TO_AU / (ANGSTROM_TO_AU * ANGSTROM_TO_AU)
    CASE (3)
      ctrl%bath_pot_sigma = ctrl%bath_pot_sigma * CMINV_TO_AU
    CASE DEFAULT
      ! no conversion needed
    END SELECT
    ctrl%units_converted = .TRUE.
  END SUBROUTINE convert_input_units


  PURE FUNCTION lower_ascii(text) RESULT(lower)
    CHARACTER(*), INTENT(IN) :: text
    CHARACTER(LEN(text)) :: lower
    INTEGER :: i, code

    lower = text
    DO i = 1, LEN(text)
      code = IACHAR(text(i:i))
      IF (code >= IACHAR('A') .AND. code <= IACHAR('Z')) &
        lower(i:i) = ACHAR(code + IACHAR('a') - IACHAR('A'))
    END DO
  END FUNCTION lower_ascii


END MODULE params
