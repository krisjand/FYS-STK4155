module common_flags
  implicit none
  logical            :: print_data ! flag: print data to files
  logical            :: debug      ! flag: run debug of code
  logical            :: rnd        ! flag: use uniformly distributed values for x and y
  integer(4)         :: verbocity  ! amount of text written to terminal 
  character(len=256) :: label      ! label to add at the end of filenames
contains
  
end module common_flags
