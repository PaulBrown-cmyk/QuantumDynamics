! constants.f03
!
! Fortran 2003+ module containing conversion factors from common chemistry units
! to atomic units (hartree atomic units).
!
! Values are based on CODATA 2018 / 2019 recommendations (commonly used in quantum chemistry)
! Last checked/updated: approximate values as of 2024-2025 literature
!
! Usage:
!   use constants_atomic
!   real(dp) :: energy_eV = 5.0
!   real(dp) :: energy_au = energy_eV * EV_TO_AU

module constants
  use kinds, only: dp
  implicit none
  private

  ! =================== Fundamental atomic units (by definition = 1) ===================
  ! Length        : bohr (a0)         = 1
  ! Energy        : hartree (Eh)      = 1
  ! Mass          : electron mass     = 1
  ! Charge        : elementary charge = 1
  ! Action        : hbar              = 1
  ! Electric field: Eh/(e·a0)         = 1

  ! ====================== Conversion factors  → atomic units ======================

  ! Length
  real(dp), parameter, public :: ANGSTROM_TO_AU  = 1.889726125836928_dp   ! 1 Å     → bohr
  real(dp), parameter, public :: PM_TO_AU        = 0.01889726125836928_dp ! 1 pm    → bohr
  real(dp), parameter, public :: NM_TO_AU        = 18.89726125836928_dp   ! 1 nm    → bohr

  ! Mass
  real(dp), parameter, public :: AMU_TO_AU       = 1822.883900484256_dp   ! 1 u (amu) → m_e

  ! Temperature
  real(dp), parameter, public :: BOLTZMANN_AU_PER_K = 3.166811563455557e-6_dp ! k_B in Eh/K

  ! Energy
  real(dp), parameter, public :: EV_TO_AU        = 0.036749324069291_dp   ! 1 eV    → Eh
  real(dp), parameter, public :: HARTREE_TO_EV   = 27.211386245988_dp     ! 1 Eh    → eV   (inverse)

  real(dp), parameter, public :: KJMOL_TO_AU     = 0.000380879140359_dp   ! 1 kJ/mol → Eh (per molecule)
  real(dp), parameter, public :: KJMOL_TO_HARTREE = KJMOL_TO_AU

  real(dp), parameter, public :: KCALMOL_TO_AU   = 0.001593601438640_dp   ! 1 kcal/mol → Eh (per molecule)
  real(dp), parameter, public :: KCALMOL_TO_HARTREE = KCALMOL_TO_AU

  ! Spectroscopic (wavenumber)
  real(dp), parameter, public :: CMINV_TO_AU     = 4.556335252769455e-6_dp ! 1 cm⁻¹ → Eh

  ! Time
  real(dp), parameter, public :: FS_TO_AU        = 41.3413733365614_dp    ! 1 fs → atomic time (ℏ/Eh)
  real(dp), parameter, public :: PS_TO_AU        = 41341.3733365614_dp    ! 1 ps → atomic time

  ! Electric field
  real(dp), parameter, public :: V_PER_M_TO_AU   = 1.944690142257e-12_dp  ! V/m   → au
  real(dp), parameter, public :: V_PER_CM_TO_AU  = 1.944690142257e-14_dp  ! V/cm  → au

  ! Electric dipole moment
  real(dp), parameter, public :: DEBYE_TO_AU     = 0.393430307_dp         ! 1 D → e·bohr

  ! ====================== Convenient inverse conversions (→ common units) ======================

  real(dp), parameter, public :: AU_TO_ANGSTROM  = 1.0_dp / ANGSTROM_TO_AU
  real(dp), parameter, public :: AU_TO_PM        = 1.0_dp / PM_TO_AU
  real(dp), parameter, public :: AU_TO_NM        = 1.0_dp / NM_TO_AU

  real(dp), parameter, public :: AU_TO_AMU       = 1.0_dp / AMU_TO_AU
  real(dp), parameter, public :: AU_TO_EV        = 1.0_dp / EV_TO_AU

  real(dp), parameter, public :: AU_TO_KJMOL     = 1.0_dp / KJMOL_TO_AU
  real(dp), parameter, public :: AU_TO_KCALMOL   = 1.0_dp / KCALMOL_TO_AU

  real(dp), parameter, public :: AU_TO_CMINV     = 1.0_dp / CMINV_TO_AU

  real(dp), parameter, public :: AU_TO_FS        = 1.0_dp / FS_TO_AU
  real(dp), parameter, public :: AU_TO_PS        = 1.0_dp / PS_TO_AU

  real(dp), parameter, public :: AU_TO_DEBYE     = 1.0_dp / DEBYE_TO_AU

  ! ==================== Physical constants often used together with a.u. ====================
  real(dp), parameter, public :: HARTREE_IN_WAVENUMBER = 219474.631363_dp   ! Eh → cm⁻¹
  real(dp), parameter, public :: BOHR_RADIUS_IN_PM     = 52.9177210903_dp   ! a0 in pm

contains

  ! Optional: small utility function
  pure elemental function to_au_energy_kcalmol(kcal) result(au)
    real(dp), intent(in) :: kcal
    real(dp) :: au
    au = kcal * KCALMOL_TO_AU
  end function

end module constants
