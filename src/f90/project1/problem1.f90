module problem1
  use RNGesus
  use linalg
  use common_flags
  implicit none


  real(8),    allocatable, dimension(:,:) :: x_n      ! vector of relevant x-values (Nx1)
  real(8),    allocatable, dimension(:,:) :: y_n      ! vector of relevant y-values (Nx1)
  real(8),    allocatable, dimension(:,:) :: f_n      ! vector of relevant f-values (N**2x1)
  real(8),    allocatable, dimension(:,:) :: b_p      ! Grid of beta values

  
  integer(kind=4), private                      :: n_p      ! Number of betas to fit
  integer(kind=4), private                      :: n_x      ! Number of x-values
  integer(kind=4), private                      :: n_xy     ! Number of xy-values

  real(8), private, allocatable, dimension(:,:) :: XY       ! matrix of x*y polynomials (N**2xP)
  real(8), private, allocatable, dimension(:,:) :: XYt      ! transpose of XY (PxN)
  real(8), private, allocatable, dimension(:,:) :: XYtXY    ! XYt*XY (PxP)
  real(8), private, allocatable, dimension(:,:) :: XYtXYi   ! inverse of XYt*XY (PxP)
  real(8), private, allocatable, dimension(:,:) :: XYf      ! XY_t * f (P x 1)

  
contains

  subroutine initialize_problem1(n,p,s)
    implicit none
    integer(kind=4), intent(in) :: n, p    ! number of x-values (n) and polynomial degree (p)
    real(8),     intent(in) :: s       ! sigma of noise in y (0.1 in problem2)

    integer(kind=4)       :: i, j, k, l, m, ind  ! Integers for looping and indexing
    real(8)               :: dx

    if (verbocity > 0) write(*,*) 'Initializing problem 1'
    n_x  = n
    n_p  = 0
    n_xy = n*n
    do i = 0,p
       n_p = n_p + i + 1
    end do
    
    allocate(x_n(1:n_x,1),y_n(1:n_x,1),b_p(1:n_p,1))
    allocate(XY(n_xy,n_p),f_n(n_xy,1))

    dx = 1.d0/(n_x-1)
    do i = 1,n_x
       x_n(i,1) = (i-1)*dx
    end do
    y_n = x_n

    do i = 1,n_x
       do j = 1,n_x
          ind = i + n_x*(j-1)
          f_n(ind,1)=franke(x_n(i,1),y_n(j,1)) + s*random_normal(0.d0,1.d0)
          l = 0
          do k = 0,p
             do m = 0,k
                l=l+1
                XY(ind,l) = x_n(i,1)**(k-m) * y_n(j,1)**m
             end do
          end do
       end do
    end do
    
  end subroutine initialize_problem1

  subroutine solve_problem1
    implicit none

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
  end subroutine solve_problem1

  function franke(x,y)
    real(8), intent(in) :: x,y
    real(8) :: franke, term1, term2, term3, term4

    term1 = 3.d0/4.d0*exp(-(9.d0*x-2.d0)**2/4.d0 - (9.d0*y-2.d0)**2/4.d0 ) 
    term2 = 3.d0/4.d0*exp(-(9.d0*x+1.d0)**2/49.d0 - (9.d0*y+1.d0)/10.d0 ) 
    term3 = 1.d0/2.d0*exp(-(9.d0*x-7.d0)**2/4.d0 - (9.d0*y-3.d0)**2/4.d0 ) 
    term4 = 1.d0/5.d0*exp(-(9.d0*x-4.d0)**2 - (9.d0*y-7.d0)**2 ) 

    franke = term1 + term2 + term3 - term4
  end function franke

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
end module problem1
