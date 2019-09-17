module problem2
  use RNGesus
  use linalg
  implicit none


  real(8),    allocatable, dimension(:,:) :: x_n      ! vector of relevant x-values (Nx1)
  real(8),    allocatable, dimension(:,:) :: y_n      ! vector of relevant y-values (Nx1)
  real(8),    allocatable, dimension(:,:) :: b_p      ! Grid of beta values

  
  integer(kind=4), private                      :: n_p      ! Number of betas to fit
  integer(kind=4), private                      :: n_x      ! Number of x-values
  real(8), private, allocatable, dimension(:,:) :: X_m      ! Matrix of relevant x-values (NxP)
  real(8), private, allocatable, dimension(:,:) :: X_t      ! transpose of X_m (PxN)
  real(8), private, allocatable, dimension(:,:) :: XtX      ! X_t*X_m (PxP)
  real(8), private, allocatable, dimension(:,:) :: XtX_i    ! inverse of X_t*X_m (PxP)
  real(8), private, allocatable, dimension(:,:) :: Xy       ! X_t*y  (Px1)
  
contains

  subroutine initialize_problem2(n,p,s)
    implicit none
    integer(kind=4), intent(in) :: n, p    ! number of x-values (n) and polynomial degree (p)
    real(8),     intent(in) :: s       ! sigma of noise in y (0.1 in problem2)

    integer(kind=4)       :: i, j          ! Integers for looping 
    real(8)           :: x_rnd, y_rnd  ! random numbers
    
    n_x = n
    n_p = p+1

    allocate(x_n(1:n_x,1),y_n(1:n_x,1),b_p(1:n_p,1))

    do i = 1,n_x
       x_rnd = ran1() !random uniform value over [0,1]
       y_rnd = random_normal(0.d0,1.d0) !random gaussian value over N(0,1)
       x_n(i,1) = x_rnd
       y_n(i,1) = x_rnd*x_rnd + s*y_rnd  ! y = x^2 + noise
    end do

  end subroutine initialize_problem2

  subroutine solve_problem2
    implicit none

    integer(kind=4)       :: i, j          ! Integers for looping 

    allocate(X_m(1:n_x,1:n_p))
    allocate(X_t(1:n_p,1:n_x))
    allocate(XtX(1:n_p,1:n_p))
    allocate(Xy(1:n_p,1))
    allocate(XtX_i(1:n_p,1:n_p))

    do i=1,n_x
       do j=1,n_p
          X_m(i,j)=x_n(i,1)**(j-1)
       end do
    end do

    call matrix_T2D(X_m,X_t)
    call matrix_mult2D(X_t,X_m,XtX)
    call matrix_mult2D(X_t,y_n,Xy)
    call matrix_inv2D(XtX,XtX_i)
    call matrix_mult2D(XtX_i,Xy,b_p)

  end subroutine solve_problem2

  

end module problem2
