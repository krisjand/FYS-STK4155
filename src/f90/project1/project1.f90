program hw1
  use RNGesus
  use problem1

  implicit none
    
  character(len=256) :: filename, partxt
  integer(kind=4)    :: i, j, n, p, seed
  real(8)            :: sigma, det
  logical            :: debug, plot
  
  sigma=0.0d0
  p = 4
  n = 21

  debug=.false.
  plot=.true.
  
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

  !!!! stalling on invertion, why?????? !!!!
  call solve_problem1

  write(*,*) b_p
  ! Output to file any desired parameters:
  
  if (plot) then ! Output milestone 2 parameters
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

end program hw1
