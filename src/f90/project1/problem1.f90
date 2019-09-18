module problem1
  use RNGesus
  use linalg
  use common_flags
  implicit none


  real(8),    allocatable, dimension(:,:) :: x_n      ! vector of relevant x-values (Nx1)
  real(8),    allocatable, dimension(:,:) :: y_n      ! vector of relevant y-values (Nx1)
  real(8),    allocatable, dimension(:,:) :: f_n      ! vector of relevant f-values (N**2x1)
  real(8),    allocatable, dimension(:,:) :: b_p      ! Grid of beta values

  
  integer(kind=4), private                      :: poly     ! Degree of polynomial
  integer(kind=4), private                      :: n_p      ! Number of betas to fit (P)
  integer(kind=4), private                      :: n_x      ! Number of x-values (N)
  integer(kind=4), private                      :: n_xy     ! Number of xy-values (N**2)

  real(8), private, allocatable, dimension(:,:) :: XY       ! matrix of x*y polynomials (N**2xP)
  real(8), private, allocatable, dimension(:,:) :: XYt      ! transpose of XY (PxN)
  real(8), private, allocatable, dimension(:,:) :: XYtXY    ! XYt*XY (PxP)
  real(8), private, allocatable, dimension(:,:) :: XYtXYi   ! inverse of XYt*XY (PxP)
  real(8), private, allocatable, dimension(:,:) :: XYf      ! XY_t * f (P x 1)
  real(8), private :: MSE    ! Mean square error
  real(8), private :: R2     ! R square value
  real(8), private :: sig    ! sigma
  
contains

  subroutine initialize_problem1(n,p,s)
    implicit none
    integer(kind=4), intent(in) :: n, p    ! number of x-values (n) and polynomial degree (p)
    real(8),     intent(in) :: s       ! sigma of noise in y (0.1 in problem2)

    integer(kind=4)       :: i, j, k, l, m, ind  ! Integers for looping and indexing
    real(8)               :: dx

    if (verbocity > 0) write(*,*) 'Initializing problem 1'
    sig = s
    n_x  = n
    poly = p
    n_xy = n*n
    n_p  = 0
    do i = 0,p
       n_p = n_p + i + 1
    end do
    
    allocate(XY(n_xy,n_p),f_n(n_xy,1),b_p(1:n_p,1))
    allocate(x_n(1:n_xy,1),y_n(1:n_xy,1))

    if (rnd<2) then !init random positions from uniform distributions
       if (rnd==0) then !equidistant grid
          dx = 1.d0/(n_x-1)
          do i = 1,n_x
             x_n(n_x*(i-1)+1:n_x*i,1) = (i-1)*dx
             y_n(i:n_xy:n_x,1) = (i-1)*dx
          end do
       else ! x and y have uniform distr. but they create a grid.
          do i = 1,n_x
             x_n((n_x-1)*i+1:n_x*i,1) = ran1()
             y_n(i:n_xy:n_x,1) = ran1()
          end do
       end if
    else ! x anf y create n_x**2 (x,y) pairs (uniform prob. dist.)
       do i = 1,n_xy
          x_n(i,1) = ran1()
          y_n(i,1) = ran1()
       end do
    end if

    !calculate the different polynomials x^i * y^j
    do i = 1,n_xy
       f_n(i,1)=franke(x_n(i,1),y_n(i,1)) + sig*random_normal(0.d0,1.d0)
       l = 0
       do k = 0,p
          do m = 0,k
             l=l+1
             XY(i,l) = x_n(i,1)**(k-m) * y_n(i,1)**m
          end do
       end do
    end do
    
  end subroutine initialize_problem1

  
  subroutine solve_problem1
    implicit none
    real(8) :: term1, term2, mean_f
    integer(kind=4)       :: i, j          ! Integers for looping 

    allocate(XYt(n_p,n_x**2))
    allocate(XYtXY(n_p,n_p))
    allocate(XYf(n_p,1))
    allocate(XYtXYi(n_p,n_p))
    if (verbocity > 0) write(*,*) 'Solving problem 1'
    if (verbocity > 1) write(*,*) 'transposing XY'
    call matrix_T2D(XY,XYt)
    if (verbocity > 1) write(*,*) 'XYt*XY'
    call matrix_mult2D(XYt,XY,XYtXY)
    if (verbocity > 1) write(*,*) 'XYt*f'
    call matrix_mult2D(XYt,f_n,XYf)
    if (verbocity > 1) write(*,*) 'inv(XYt*XY)'
    call matrix_inv2D(XYtXY,XYtXYi)
    if (verbocity > 1) write(*,*) '(XYt XY)^-1 * (XYt*f)'
    call matrix_mult2D(XYtXYi,XYf,b_p)
    if (verbocity > 0) write(*,*) '' 

    !check errors
    MSE=0.d0
    mean_f=0.d0
    do i = 1,n_xy
       MSE=MSE+(f_n(i,1)-polynom_xy(XY(i,:)))**2
       mean_f=mean_f+f_n(i,1)
    end do
    term1=MSE
    MSE=MSE/n_xy
    mean_f=mean_f/n_xy
    term2=0.d0
    do i = 1,n_xy
       term2=term2+(f_n(i,1)-mean_f)**2
    end do
    R2 = 1.d0 - term1/term2
    
    
  end subroutine solve_problem1

  function franke(x,y)
    real(8), intent(in) :: x,y
    real(8) :: franke, term1, term2, term3, term4

    term1 = 0.75d0*exp(-(9.d0*x-2.d0)**2/4.d0 - (9.d0*y-2.d0)**2/4.d0 ) 
    term2 = 0.75d0*exp(-(9.d0*x+1.d0)**2/49.d0 - (9.d0*y+1.d0)/10.d0 ) 
    term3 = 0.5d0*exp(-(9.d0*x-7.d0)**2/4.d0 - (9.d0*y-3.d0)**2/4.d0 ) 
    term4 = -0.2d0*exp(-(9.d0*x-4.d0)**2 - (9.d0*y-7.d0)**2 ) 

    franke = term1 + term2 + term3 + term4
  end function franke

  function polynom_xy(xy_vec)
    implicit none
    real(8), dimension(:),   intent(in)  :: xy_vec !the different polynomials of x and y
    real(8)    :: polynom_xy
    integer(4) :: i

    polynom_xy=0.d0    
    do i = 1,n_p
       polynom_xy = polynom_xy + xy_vec(i)*b_p(i,1)
    end do
    
  end function polynom_xy

  subroutine print_problem1
    implicit none
    
    write(*,*) '------------------------------------'
    write(*,*) 'beta values:'
    call print_beta_values(poly)
    write(*,*) '------------------------------------'
    write(*,*) 'Mean Square Error:'
    write(*,*) 'MSE =',MSE
    write(*,*) '------------------------------------'
    write(*,*) 'R squared score'
    write(*,*) 'R^2 =',R2
    write(*,*) '------------------------------------'
    write(*,*) ''

  end subroutine print_problem1
  
  subroutine print_beta_values(p)
    integer(kind=4), intent(in) :: p    ! number of x-values (n) and polynomial degree (p)
    integer(kind=4)       :: i, j, k, l, m, ind  ! Integers for looping and indexing
    character(len=256)    :: str, str2
    character(len=8)      :: partxt

    l=0
    do k = 0,p
       do m = 0,k
          l=l+1
          str='beta_'
          write(partxt,'(I8)') l
          partxt=adjustl(partxt)
          str=trim(str)//partxt//'='

          if (k==0) then
             str2='1'
          elseif (k==1) then
             if (m==0) then
                str2='x'
             else
                str2='y'
             end if
          else
             if (k-m==0) then
                write(partxt,'(I8)') m
                partxt=adjustl(partxt)
                str2='y^'//trim(partxt)
             elseif (k-m==1) then
                str2='x'
                if (m==1) then
                   str2=trim(str2)//' y'
                else
                   write(partxt,'(I8)') m
                   partxt=adjustl(partxt)
                   str2=trim(str2)//' y^'//trim(partxt)
                end if
             else
                write(partxt,'(I8)') k-m
                partxt=adjustl(partxt)
                str2='x^'//trim(partxt)
                if (m==1) then
                   str2=trim(str2)//' y'
                elseif (m>1) then
                   write(partxt,'(I8)') m
                   partxt=adjustl(partxt)
                   str2=trim(str2)//' y^'//trim(partxt)
                end if
             end if
          end if
          write(*,*) trim(str), b_p(l,1), trim(str2)
       end do
    end do
  end subroutine print_beta_values

  subroutine write_to_disk
    character(len=256) :: filename, partxt, endlabel
    integer(kind=4)    :: ind, i, j, l 
        

    write(partxt,'(I8)') poly
    partxt=adjustl(partxt)
    endlabel='_p'//trim(partxt)
    write(partxt,'(I8)') n_x
    partxt=adjustl(partxt)
    endlabel=trim(endlabel)//'_n'//trim(partxt)
    write(partxt,'(I8)') rnd
    partxt=adjustl(partxt)
    endlabel=trim(endlabel)//'_rnd'//trim(partxt)
    write(partxt,'(F6.2)') sig
    partxt=adjustl(partxt)
    endlabel=trim(endlabel)//'_s'//trim(partxt)
    endlabel=trim(endlabel)//trim(label)//'.dat'

    filename='xy'//trim(endlabel)
    OPEN(UNIT=1,FILE=filename)
    do i = 1,n_xy
       write(1,*) x_n(i,1),y_n(i,1)
    end do
    CLOSE(1)

    filename='franke'//trim(endlabel)
    OPEN(UNIT=1,FILE=filename)
    do i = 1,n_xy
       write(1,*) x_n(i,1),y_n(i,1)
    end do
    CLOSE(1)

    filename='beta'//trim(endlabel)
    OPEN(UNIT=1,FILE=filename)
    l=0
    do i = 1,n_p
       write(1,*) i, b_p(i,1)
    end do
    CLOSE(1)
  end subroutine write_to_disk
end module problem1
