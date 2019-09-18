program project1
  use RNGesus
  use problem1
  use common_flags

  implicit none
    
  character(len=256) :: filename, partxt
  integer(kind=4)    :: n_arg, ind, i, j, n, p, seed
  real(8)            :: sigma, det
  
  sigma=0.1d0
  p = 4
  n = 21


  n_arg=iargc()
  if (iargc() > 0)  call getarg(1,partxt)

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
  end if

  ind=0
  do while (ind <= n_arg)
     ind=ind+1
     call getarg(ind,partxt)
     if (partxt=='-p') then

     end if

  end do
     
  if (iargc() > 0) then
     call getarg(1,partxt)
     read(partxt,*) seed
  else
     seed = 20198935
  end if

  ! Initialize time grids
  call initialize_RNG(seed)

  if (debug) call test_linalg
  if (debug) call test_rng

  write(*,*) 0.d0**0
  
  call initialize_problem1(n,p,sigma)

  call solve_problem1

  write(*,*) b_p
  ! Output to file any desired parameters:
  
  if (print) then ! Output milestone 2 parameters
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
