module common_flags
  implicit none
  logical            :: write_data ! flag: print data to files
  logical            :: debug      ! flag: run debug of code
  integer(4)         :: rnd        ! int : (0) equidist. spacing (grid)
                                   !       (1) uniform distr. x set and y set (grid)
                                   !       (2) uniform distr. (x,y) pairs
  integer(4)         :: verbocity  ! amount of text written to terminal 
  character(len=256) :: label      ! label to add at the end of filenames
contains
  
end module common_flags
