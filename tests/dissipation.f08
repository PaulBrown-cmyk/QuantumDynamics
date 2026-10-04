program regression
  use kinds
  use params
  use grid
  use propagator
  use langevin
  use potentials, only: v_two_surface
  use rng
  implicit none
  type(SimCtrl) :: ctrl
  type(RealGrid) :: g
  type(SOProp) :: prop
  type(LangevinState) :: bath
  integer :: i,j,mode
  real(dp) :: p,xi,norm,avg,var,expected,v11,v22,v12
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
end program
