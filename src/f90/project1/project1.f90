program project1
  use RNGesus
  use problem1
  use common_flags

  implicit none
  ! fixed (should be initialized, not changed)
  integer(kind=4)    :: n, p, seed          
  real(8)            :: sigma, det


  ! variable
  character(len=256) :: filename, partxt
  integer(kind=4)    :: n_arg, ind, i, j, l 
  
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
     write(*,*) ' -write            :: (logical, .false.) include to write results to file'
     write(*,*) ' -debug            :: (logical, .false.) include to run code in debug mode'
     write(*,*) ' -label [label]    :: (string, empty) label to add to end of output files'
     write(*,*) ' -rnd [version]    :: (int, 0) distribution of x and y. (0) equidist. grid'
     write(*,*) '                            (1) two uniformly distributed sets for x & y'
     write(*,*) '                            (2) a uniformly distributed set of (x,y) pairs'
     write(*,*) ''
     stop
  end if

  !  default values
  sigma=0.0d0
  p = 0
  seed = 20198935
  verbocity = 0
  rnd = 0
  write_data = .false.
  debug = .false.
  label = ''

  !get n (number of x- and y-values)
  ind=1
  call getarg(ind,partxt)
  read(partxt,*) n

  !read in options
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
        label='_'//trim(partxt)
     elseif (partxt=='-rnd') then
        ind=ind+1 
        call getarg(ind,partxt)
        read(partxt,*) rnd
        if (rnd < 0 .or. rnd > 2) then
           write(*,*) 'ERROR: unvalid (x,y) distribution [rnd]'
           stop
        end if
     elseif (partxt=='-write') then
        write_data=.true.
     elseif (partxt=='-debug') then
        debug=.true.
     end if
  end do
     
  ! Initialize random number generators
  call initialize_RNG(seed)

  ! debugging codes
  if (debug) then
     call test_linalg
     call test_rng
     stop !stop after debug
  end if
  
  !Problem 1
  if (verbocity > 0) then
     write(*,*) 'Problem 1'
     write(*,*) '-----------------------'
  end if
  call initialize_problem1(n,p,sigma)
  call solve_problem1
  if (verbocity > 0) call print_problem1

  
  ! Output to file any desired parameters:
  
  if (write_data) call write_to_disk


end program project1
