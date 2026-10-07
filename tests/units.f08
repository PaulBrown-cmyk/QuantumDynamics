program unit_conversions
  use kinds
  use constants, only: ANGSTROM_TO_AU, AU_TO_ANGSTROM, CMINV_TO_AU, &
                       AU_TO_CMINV, FS_TO_AU, AU_TO_FS, AMU_TO_AU, &
                       BOLTZMANN_AU_PER_K
  use params
  implicit none

  type(SimCtrl) :: ctrl, mode2, mode3

  ctrl%xmin = -1.25_dp
  ctrl%xmax = 2.75_dp
  ctrl%dt = 0.125_dp
  ctrl%temperature_k = 300.0_dp
  ctrl%mass_amu = 1.00784_dp
  ctrl%gamma = 0.75_dp
  ctrl%fwhm = 1.5_dp
  ctrl%k1 = 2200.0_dp
  ctrl%k2 = 3300.0_dp
  ctrl%x1 = -0.8_dp
  ctrl%x2 = 0.9_dp
  ctrl%v1_shift = 100.0_dp
  ctrl%v2_shift = 200.0_dp
  ctrl%c4_1 = 12.0_dp
  ctrl%c4_2 = 13.0_dp
  ctrl%v12 = 350.0_dp
  ctrl%sigma = 0.4_dp
  ctrl%x0 = -0.7_dp
  ctrl%p0 = 2.5_dp
  ctrl%sigma0 = 0.2_dp
  ctrl%use_absorber = .true.
  ctrl%absorber_width = 0.4_dp
  ctrl%absorber_rate = 0.6_dp
  ctrl%absorber_power = 4
  ctrl%bath_pot_mode = 1
  ctrl%bath_pot_sigma = 0.3_dp
  ctrl%bath_pot_fwhm = 2.0_dp

  call validate_physical_input(ctrl)
  call convert_input_units(ctrl)

  call check('angstrom input', ctrl%xmin*AU_TO_ANGSTROM, -1.25_dp)
  call check('femtosecond input', ctrl%dt*AU_TO_FS, 0.125_dp)
  call check('energy input', ctrl%v12*AU_TO_CMINV, 350.0_dp)
  call check('rate input', ctrl%gamma*FS_TO_AU, 0.75_dp)
  call check('kernel rate input', ctrl%fwhm*FS_TO_AU, 1.5_dp)
  call check('mass input', ctrl%mass/AMU_TO_AU, 1.00784_dp)
  call check('temperature input', 1.0_dp/(ctrl%beta*BOLTZMANN_AU_PER_K), 300.0_dp)
  call check('wave number input', ctrl%p0*ANGSTROM_TO_AU, 2.5_dp)
  call check('absorber width input', ctrl%absorber_width*AU_TO_ANGSTROM, 0.4_dp)
  call check('absorber rate input', ctrl%absorber_rate*FS_TO_AU, 0.6_dp)
  call check('force constant input', &
    ctrl%k1/Cminv_to_au*ANGSTROM_TO_AU**2, 2200.0_dp)
  call check('quartic input', ctrl%c4_1/CMINV_TO_AU*ANGSTROM_TO_AU**4, 12.0_dp)
  call check('coordinate bath amplitude', ctrl%bath_pot_sigma*AU_TO_ANGSTROM, 0.3_dp)

  mode2%bath_pot_mode = 2
  mode2%bath_pot_sigma = 17.0_dp
  call convert_input_units(mode2)
  call check('curvature bath amplitude', &
    mode2%bath_pot_sigma/CMINV_TO_AU*ANGSTROM_TO_AU**2, 17.0_dp)

  mode3%bath_pot_mode = 3
  mode3%bath_pot_sigma = 42.0_dp
  call convert_input_units(mode3)
  call check('enthalpy bath amplitude', mode3%bath_pot_sigma*AU_TO_CMINV, 42.0_dp)

  print *, 'PASS physical-unit conversions'

contains

  subroutine check(label, actual, expected)
    character(*), intent(in) :: label
    real(dp), intent(in) :: actual, expected
    real(dp) :: scale
    scale = max(1.0_dp, abs(expected))
    if (abs(actual-expected) > 2.0e-12_dp*scale) then
      print *, 'FAIL ', trim(label), actual, expected
      error stop 'unit conversion regression'
    end if
  end subroutine check

end program unit_conversions
