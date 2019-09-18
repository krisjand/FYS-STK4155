program project1
  use RNGesus
  use problem1
  use common_flags

  implicit none
    
  character(len=256) :: filename, partxt
  integer(kind=4)    :: n_arg, ind, i, j, n, p, seed
  real(8)            :: sigma, det
  
  n_arg=iargc()
  if (n_arg > 0)  call getarg(1,partxt)

  if (n_arg < 1 .or. partxt == 'help' ) then
     write(*,*) ' '
     write(*,*) 'Syntax: project1 [n_samples] + options '
     write(*,*) ' '
     write(*,*) 'n_samples (int): number of values in x- and y-direction on the grid'
     write(*,*) ' '
     write(*,*) 'Options: '
     write(*,*) ' <flag>  [value]      (<type>, default value)'
     write(*,*) '----------------------------------------------------------- '
     write(*,*) ' -sigma [sigma]    :: (dp, 0.0) STD of sampled data (Franke function)'
     write(*,*) ' -p [order]        :: (int, 0) polynomial order'
     write(*,*) ' -seed [seed]      :: (int, 20198935) seed value for the RNG'
     write(*,*) ' -verb [verbocity] :: (int, 0) different amount of terminal output '
     write(*,*) ' -rnd              :: (logical, .false.) random uniform (x,y) values'
     write(*,*) ' -print            :: (logical, .false.) include to write results to file'
     write(*,*) ' -debug            :: (logical, .false.) include to run code in debug mode'
     write(*,*) ' -label [label]    :: (string, empty) label to add to end of output files'
     write(*,*)
     stop
  end if

  !  default values
  sigma=0.0d0
  p = 0
  seed = 20198935
  verbocity = 0
  rnd = .false.
  print_data = .false.
  debug = .false.
  label = ''
  
  ind=1
  call getarg(ind,partxt)
  read(partxt,*) n

  !reed in options
  do while (ind < n_arg)
     ind=ind+1
     call getarg(ind,partxt)
     if (partxt=='-p') then
        ind=ind+1 
        call getarg(ind,partxt)
        read(partxt,*) p
     elseif (partxt=='-sigma') then
        ind=ind+1 
        call getarg(ind,partxt)
        read(partxt,*) sigma
     elseif (partxt=='-seed') then
        ind=ind+1 
        call getarg(ind,partxt)
        read(partxt,*) seed
     elseif (partxt=='-verb') then
        ind=ind+1 
        call getarg(ind,partxt)
        read(partxt,*) verbocity
     elseif (partxt=='-label') then
        ind=ind+1 
        call getarg(ind,partxt)
        label=trim(partxt)
     elseif (partxt=='-rnd') then
        rnd=.true.
     elseif (partxt=='-print') then
        print_data=.true.
     elseif (partxt=='-debug') then
        debug=.true.
     end if
  end do
     
  if (iargc() > 0) then
     call getarg(1,partxt)
     read(partxt,*) seed
  else

  end if

  ! Initialize random number generators
  call initialize_RNG(seed)

  ! debugging codes
  if (debug) call test_linalg
  if (debug) call test_rng

  !Problem 1
  if (verbocity > 0) then
     write(*,*) 'Problem 1'
     write(*,*) '-----------------------'
  end if
  call initialize_problem1(n,p,sigma)
  call solve_problem1
  if (verbocity > 0) then
     write(*,*) '------------------------------------'
     write(*,*) 'beta values:'
     call print_beta_values(p)
     write(*,*) '------------------------------------'
     write(*,*) ''
  end if

  ! Output to file any desired parameters:
  
  if (print_data) then ! Output milestone 2 parameters
     filename='xy.dat'
     OPEN(UNIT=1,FILE=filename)
     do i = 1,n
        write(1,*) x_n(i,1),y_n(i,1)
     end do
     CLOSE(1)

     filename='beta.dat'
     OPEN(UNIT=1,FILE=filename)
     do i = 1,p+1
        write(1,*) b_p(i,1)
     end do
     CLOSE(1)
  end if

end program project1
