module RNGesus
  implicit none

  
  integer(kind=4), private :: iseed      ! seed of the ran0 RNG
  integer(kind=4), private :: iseed1     ! seed of the ran1 RNG
  
contains

  subroutine initialize_RNG(seed)
    implicit none
    integer(kind=4), intent(in) :: seed    ! number of x-values (n) and polynomial degree (p)
    
    iseed=seed
    iseed1=seed
    
  end subroutine initialize_RNG
  
  function ran0()
    implicit none

    real(8) :: ran0
    integer(kind=4),parameter :: IA=16807,IM=2147483647,IQ=127773,IR=2836
    real, save :: am
    integer(kind=4), save :: ix=-1, iy=-1,k

    if (iseed <= 0 .or. iy < 0) then
       am = nearest(1.e0,-1.e0)/IM
       iy = ior(ieor(888889999,abs(iseed)),1)
       ix = ieor(777755555,abs(iseed))
       iseed = abs(iseed)+1
    end if
    ix = ieor(ix,ishft(ix,13))
    ix = ieor(ix,ishft(ix,-17))
    ix = ieor(ix,ishft(ix,5))
    k = iy/IQ
    iy = IA*(iy-k*IQ)-IR*k
    if (iy<0) iy = iy+IM
    ran0 = am*ior(iand(IM,ieor(ix,iy)),1)
  end function ran0

  function ran1()
    implicit none

    real(8) :: ran1
    integer(kind=4),parameter :: IA=16807,IM=2147483647,IQ=127773,IR=2836,IMASK=123459876
    integer(kind=4) :: k
    real(8), save :: am

    am=1.0/IM
    
    k = iseed1/IQ
    iseed1=IA*(iseed1-k*IQ)-IR*k
    if (iseed1 < 0) iseed1=iseed1+IM
    ran1 = am*iseed1
  end function ran1

  !Box-Muller
  function random_normal(m,v)
    implicit none

    real(8), intent(in) :: m,v      !mean and variance
    real(8)             :: random_normal 
    real(8)             :: u1,u2    !uniform numbers
    real(8), parameter  :: pi=3.141592653589793238

    u1 = ran1() 
    u2 = ran1()
    
    random_normal = m + dsqrt(-2.d0*v*log(u1))*cos(2.d0*pi*u2)
  end function random_normal

  !Testing the RNG functions
  subroutine test_rng
    implicit none
    real(8)            :: rnd, dx, x_m(2), x_v(2), means(2), vars(2)
    integer(kind=4)    :: i, j, k
    integer(kind=4), allocatable, dimension(:) :: n
    real(8), allocatable, dimension(:,:)  :: prb, x_n
    real(8), allocatable, dimension(:)    :: x_d
    character(len=256) :: filename, partxt
    
    allocate(prb(100,2),x_d(100))
    allocate(n(4))
    n(1)=1000
    n(2)=10000
    n(3)=100000
    n(4)=1000000

    dx=1.d-2
    do i = 1,100
       x_d(i) = i*dx
    end do

    do k = 1,4
       allocate(x_n(n(k),2))
       prb=0.0
       do i = 1,n(k)
          x_n(i,1) = ran0()
          x_n(i,2) = ran1()
          do j = 1,100 !get distribution
             if (x_n(i,1) < x_d(j)) then
                prb(j,1) = prb(j,1)+1.0
                exit
             end if
          end do
          do j = 1,100 !get distribution
             if (x_n(i,2) < x_d(j)) then
                prb(j,2) = prb(j,2)+1.0
                exit
             end if
          end do
       end do

       !print distribution to file
       prb=prb/n(k)
       write(partxt,'(I10)') n(k)
       partxt=adjustl(partxt)
       filename='prb_ran0_n'//trim(partxt)//'.dat'

       OPEN(UNIT=1,FILE=filename)
       do i=1,100
          write(1,*) x_d(i) , prb(i,1)
       end do
       CLOSE(1)

       filename='prb_ran1_n'//trim(partxt)//'.dat'

       OPEN(UNIT=1,FILE=filename)
       do i=1,100
          write(1,*) x_d(i) , prb(i,2)
       end do
       CLOSE(1)


       !calculate <x>
       x_m=0.d0
       x_v=0.d0
       do i =1,n(k)
          x_m(1) = x_m(1) + x_n(i,1)
          x_m(2) = x_m(2) + x_n(i,2)
       end do
       x_m = x_m/n(k)
       
       !calculate Var(x)
       do i =1,n(k)
          x_v(1) = x_v(1) + (x_n(i,1)-x_m(1))**2
          x_v(2) = x_v(2) + (x_n(i,2)-x_m(2))**2
       end do
       x_v = x_v/n(k)

       write(*,*) 'n =',n(k)
       write(*,*) '   ran0                       ran1'
       write(*,*) '<x> =',x_m
       write(*,*) 'Var(x) =',x_v
       write(*,*) ''
       deallocate(x_n)       
    end do


    dx=0.2d0
    do i = 1,100
       x_d(i) = -10.d0 + i*dx
    end do
    
    do k = 1,4
       allocate(x_n(n(k),2))

       !debug random_normal
       !mean vals
       means(1)=0.d0 
       means(2)=2.d0

       !variance
       vars(1)=1.d0 !std=1.0
       vars(2)=9.d0 !std=3.0

    
       prb=0.0
       do i = 1,n(k)
          x_n(i,1) = random_normal(means(1),vars(1))
          x_n(i,2) = random_normal(means(2),vars(2))
          do j = 1,100 !get distribution
             if (x_n(i,1) < x_d(j)) then
                prb(j,1) = prb(j,1)+1.0
                exit
             end if
          end do
          do j = 1,100 !get distribution
             if (x_n(i,2) < x_d(j)) then
                prb(j,2) = prb(j,2)+1.0
                exit
             end if
          end do
       end do

       !print distribution to file
       prb=prb/n(k)
       write(partxt,'(I10)') n(k)
       partxt=adjustl(partxt)
       filename='prb_normal_0_1_n'//trim(partxt)//'.dat'

       OPEN(UNIT=1,FILE=filename)
       do i=1,100
          write(1,*) x_d(i) , prb(i,1)
       end do
       CLOSE(1)

       filename='prb_normal_2_3_n'//trim(partxt)//'.dat'

       OPEN(UNIT=1,FILE=filename)
       do i=1,100
          write(1,*) x_d(i) , prb(i,2)
       end do
       CLOSE(1)


       !calculate <x>
       x_m=0.d0
       x_v=0.d0
       do i =1,n(k)
          x_m(1) = x_m(1) + x_n(i,1)
          x_m(2) = x_m(2) + x_n(i,2)
       end do
       x_m = x_m/n(k)
       
       !calculate Var(x)
       do i =1,n(k)
          x_v(1) = x_v(1) + (x_n(i,1)-x_m(1))**2
          x_v(2) = x_v(2) + (x_n(i,2)-x_m(2))**2
       end do
       x_v = x_v/n(k)

       write(*,*) 'n =',n(k)
       write(*,*) '   random_normal(0,1)        random_normal(2,9)'
       write(*,*) '<x> =',x_m
       write(*,*) 'Var(x) =',x_v
       write(*,*) ''
       
       deallocate(x_n)       
    end do

    deallocate(n,prb,x_d)
  end subroutine test_rng

  
end module RNGesus
