program regression
  use kinds
  use params
  use grid
  use propagator
  use langevin
  use potentials, only: v_two_surface, potentials_bath_init, bath_pot
  use rng
  implicit none
  type(SimCtrl) :: ctrl
  type(RealGrid) :: g
  type(SOProp) :: prop
  type(LangevinState) :: bath
  integer :: i,j,mode
  real(dp) :: p,xi,norm,avg,var,avg2,var2,expected,v11,v22,v12,p1,p2
  integer, parameter :: samples=12000
  call seed_stream(4321)
  ctrl%k1=0; ctrl%k2=0; ctrl%v12=0
  ctrl%gamma=0; ctrl%dt=0.05_dp; ctrl%mass=1.0_dp
  ctrl%x0=0; ctrl%p0=1.0_dp; ctrl%sigma0=3
  call build_grid(g,512,-80.0_dp,80.0_dp)
  call init_prop(prop,g)
  call set_gaussian_packet(prop,ctrl)
  prop%psi2=prop%psi1/sqrt(2.0_dp)
  prop%psi1=prop%psi1/sqrt(2.0_dp)
  call init_langevin(bath,ctrl,ctrl%dt)
  do i=1,100
    call step_langevin(ctrl,prop,bath,ctrl%dt)
  end do
  norm=sum(abs(prop%psi1)**2+abs(prop%psi2)**2)*g%dx
  if(abs(norm-1.0_dp)>1.e-11_dp) error stop 'gamma=0 norm'
  print *, 'PASS gamma=0 norm',norm
  ctrl%gamma=0.4_dp; ctrl%beta=huge(1.0_dp)
  call set_gaussian_packet(prop,ctrl)
  call init_langevin(bath,ctrl,ctrl%dt)
  do i=1,100
    call step_langevin(ctrl,prop,bath,ctrl%dt)
  end do
  expected=ctrl%p0*exp(-ctrl%gamma*100*ctrl%dt)
  p=mean_momentum(prop)
  if(abs(p-expected)>1.e-9_dp) error stop 'white relaxation'
  print *, 'PASS white packet relaxation',p,expected

  ! Product-only damping must leave reactant momentum unchanged.
  call set_gaussian_packet(prop,ctrl)
  prop%psi2=prop%psi1/sqrt(2.0_dp)
  prop%psi1=prop%psi1/sqrt(2.0_dp)
  ctrl%damp_reactant=.false.; ctrl%damp_product=.true.
  call init_langevin(bath,ctrl,ctrl%dt)
  p1=component_momentum(prop,1)
  p2=component_momentum(prop,2)
  call bath_kick(ctrl,prop,bath,ctrl%dt)
  if(abs(component_momentum(prop,1)-p1)>1.e-9_dp) &
    error stop 'product-only bath changed reactant momentum'
  expected=p2*exp(-ctrl%gamma*ctrl%dt)
  if(abs(component_momentum(prop,2)-expected)>1.e-9_dp) &
    error stop 'product-only bath failed to relax product momentum'
  print *, 'PASS product-only momentum damping'
  ctrl%damp_reactant=.true.; ctrl%damp_product=.true.
  ! Colored deterministic reference: p''+p'/tau+gamma*p/tau=0, y(0)=0.
  ctrl%use_colored=.true.; ctrl%kernel='lorentz'; ctrl%fwhm=2
  call set_gaussian_packet(prop,ctrl)
  call init_langevin(bath,ctrl,ctrl%dt)
  bath%z=0
  do i=1,100
    call step_langevin(ctrl,prop,bath,ctrl%dt)
  end do
  expected=exp(-2.5_dp)*(cos(sqrt(0.15_dp)*5)+ &
    0.5_dp/sqrt(0.15_dp)*sin(sqrt(0.15_dp)*5))
  p=mean_momentum(prop)
  if(abs(p-expected)>2.e-4_dp) error stop 'colored relaxation'
  print *, 'PASS colored packet relaxation',p,expected
  call destroy_prop(prop)
  ! Independent free-centroid ensembles; covariance is classical mass/beta.
  ctrl%beta=2; ctrl%dt=0.2_dp
  do mode=1,2
    ctrl%use_colored=(mode==2)
    avg=0; var=0
    do j=1,samples
      call init_langevin(bath,ctrl,ctrl%dt)
      p=0
      do i=1,200
        call next_kick(bath,ctrl,ctrl%dt,xi,p)
        p=p+xi
      end do
      avg=avg+p; var=var+p*p
    end do
    avg=avg/samples; var=var/samples-avg*avg
    if(abs(avg)>0.025_dp) error stop 'ensemble mean'
    if(abs(var-ctrl%mass/ctrl%beta)>0.025_dp) error stop 'ensemble FDT'
    print *, 'PASS ensemble FDT mode, mean, variance',mode,avg,var
  end do

  ! Colored potential modulation must start in its stationary distribution.
  call seed_stream(8765)
  ctrl%bath_pot_mode=3; ctrl%bath_pot_sigma=2.0_dp
  ctrl%bath_pot_colored=.true.; ctrl%bath_pot_fwhm=0.5_dp
  ctrl%bath_pot_reactant=.true.; ctrl%bath_pot_product=.true.
  ctrl%bath_pot_coupled=.false.
  avg=0; var=0; avg2=0; var2=0
  do i=1,samples
    call potentials_bath_init(ctrl,ctrl%dt)
    avg=avg+bath_pot%rt(1); var=var+bath_pot%rt(1)**2
    avg2=avg2+bath_pot%rt(2); var2=var2+bath_pot%rt(2)**2
  end do
  avg=avg/samples; var=var/samples-avg*avg
  avg2=avg2/samples; var2=var2/samples-avg2*avg2
  if(abs(avg)>0.06_dp .or. abs(avg2)>0.06_dp) error stop 'potential OU initial mean'
  if(abs(var-4.0_dp)>0.16_dp .or. abs(var2-4.0_dp)>0.16_dp) &
    error stop 'potential OU initial variance'
  print *, 'PASS stationary potential OU initialization',avg,var,avg2,var2
  ctrl%bath_pot_mode=0

  ctrl%k1=0; ctrl%k2=0; ctrl%x1=-1; ctrl%x2=1
  ctrl%v12=2; ctrl%sigma=1; ctrl%want_coupling=.false.
  call v_two_surface(ctrl,0.5_dp,v11,v22,v12)
  if(abs(v12)>tiny(1.0_dp)) error stop 'coupling disable switch'
  ctrl%want_coupling=.true.; ctrl%use_exponential=.false.
  call v_two_surface(ctrl,0.5_dp,v11,v22,v12)
  expected=2.0_dp*exp(-0.125_dp)
  if(abs(v12-expected)>1.e-14_dp) error stop 'Gaussian coupling switch'
  ctrl%use_exponential=.true.
  call v_two_surface(ctrl,0.5_dp,v11,v22,v12)
  expected=2.0_dp*exp(-0.5_dp)
  if(abs(v12-expected)>1.e-14_dp) error stop 'exponential coupling switch'
  print *, 'PASS coupling input switches'

  ctrl%want_coupling=.false.; ctrl%pot_model='anharmonic1'
  ctrl%c4_1=3.0_dp; ctrl%v1_shift=5.0_dp
  call v_two_surface(ctrl,0.5_dp,v11,v22,v12)
  expected=3.0_dp*1.5_dp**4+5.0_dp
  if(abs(v11-expected)>1.e-14_dp) error stop 'quartic and shift independence'
  print *, 'PASS quartic coefficient and energy shift'

  call build_grid(g,5,0.0_dp,5.0_dp)
  if(abs(g%k(3)-4.0_dp*acos(-1.0_dp)/5.0_dp)>1.e-14_dp .or. &
     abs(g%k(4)+4.0_dp*acos(-1.0_dp)/5.0_dp)>1.e-14_dp) &
    error stop 'odd FFT grid ordering'
  print *, 'PASS odd-sized FFT grid ordering'
end program
