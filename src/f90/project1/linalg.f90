module linalg
  implicit none



contains

  subroutine matrix_T2D(X_in, X_out)
    real(8), dimension(:,:), intent(in)  :: X_in
    real(8), dimension(:,:), intent(out) :: X_out

    integer(kind=4)   :: shape_X(2), m, n ! Integers for shape of matrices
    integer(kind=4)   :: i, j             ! Integers for looping 

    shape_X=shape(X_in)
    m=shape_X(1)
    n=shape_X(2)

    shape_X=shape(X_out)

    if (m /= shape_X(2)) then
       write(*,*) "1st dim of input matrix and 2nd dim of output matrix doesn't match"
       stop
    elseif (n /= shape_X(1)) then
       write(*,*) "2nd dim of input matrix and 1st dim of output matrix doesn't match"
       stop
    end if

    do i = 1,m
       do j = 1,n
          X_out(j,i) = X_in(i,j)
       end do
    end do

  end subroutine matrix_T2D

  subroutine matrix_mult2D(X1,X2,X_out)
    real(8), dimension(:,:), intent(in)  :: X1,X2
    real(8), dimension(:,:), intent(out) :: X_out

    real(8)           :: temp
    integer(kind=4)   :: i, j, k ! Integers for looping
    integer(kind=4)   :: shape_X(2), l, m, n ! Integers for shape of matrices
    shape_X=shape(X1)
    m = shape_X(1) ! 1st dim of matrix 1 and 1st dim of output matrix
    n = shape_X(2) ! 2nd dim of matrix 1 and 1st dim of matrix 2

    shape_X=shape(X2)

    if (n /= shape_X(1)) then
       write(*,*) "2nd dim of matrix 1 and 1st dim of matrix 2 doesn't match"
       stop
    end if

    l = shape_X(2) ! 2nd dim of matrix 2 and output matrix
    shape_X=shape(X_out)

    if (m /= shape_X(1)) then
       write(*,*) "1st dim of matrix 1 and 1st dim of output matrix doesn't match"
       stop
    elseif (l /= shape_X(2)) then
       write(*,*) "2nd dim of matrix 2 and 2nd dim of output matrix doesn't match"
       stop
    end if

    do i = 1,m
       do j = 1,l
          temp=0.d0
          do k = 1,n
             temp = temp + X1(i,k)*X2(k,j) !multiply row i (X1) with column l (X2)
          end do
          X_out(i,j) = temp
       end do
    end do
    
  end subroutine matrix_mult2D

  subroutine matrix_inv2D(X_in, X_out, check)
    real(8), dimension(:,:), intent(in)  :: X_in
    real(8), dimension(:,:), intent(out) :: X_out
    logical, optional,       intent(in)  :: check
    real(8), dimension(:,:), allocatable :: X
    real(8), dimension(:), allocatable   :: lineX

    real(8)           :: detX
    logical           :: check_det
    integer(kind=4)   :: shape_X(2), m, n ! Integers for shape of matrices
    integer(kind=4)   :: i, j, k, l       ! Integers for looping 

    if (present(check)) then
       check_det=check
    else
       check_det=.false.
    end if

    shape_X=shape(X_in)
    m=shape_X(1)
    n=shape_X(2)

    shape_X=shape(X_out)

    if (m /= n) then
       write(*,*) "Matrix is not diagonal (NxN)"
       stop
    elseif (m /= shape_X(1)) then
       write(*,*) "1st dim of input matrix and output matrix doesn't match"
       stop
    elseif (n /= shape_X(2)) then
       write(*,*) "2nd dim of input matrix and output matrix doesn't match"
       stop
    end if

    if (check_det .and. m < 11) then !determinants take a long time to compute after NxN = 10x10
       ! scaling in operations is N! (10x10 --> 10! ~ 3.6 million )
       call get_det2D(X_in,detX)
       if (detX == 0.d0) then
          X_out = 0.d0
          write(*,*) 'matrix has det = 0.0. Not invertable'
          return
       end if
    end if

    !init matrix to reduce
    allocate(X(m,2*m))
    allocate(lineX(2*m))
    X=0.d0
    X(:,:m)=X_in
    do i = 1,m
       X(i,m+i)=1.d0
    end do

    !reduce
    do i = 1,m !if i = m, we only check if X(i,i)=0.d0
       k=i
       do j=i+1,m
          if (abs(X(j,i)) > abs(X(k,i))) k=j
       end do
       if (k /= i) then !change lines i and k
          lineX=X(k,:)
          X(k,:)=X(i,:)
          X(i,:)=lineX
       end if

       !need to check if we are dividing by 0.d0 if determinant is not checked, or numerical
       !precision makes this happen
       if (X(i,i)==0.d0) then
          !We have put the line with the largest absolute value in column i in row i.
          !If this value is 0.d0, then all values are 0.d0 and we have a zero determinant.
          !then the inverse matrix doesn't exist
          write(*,*) 'matrix has det = 0.0, Matrix is not invertable'
       end if
       do j=i+1,m
          lineX=(X(j,i)/X(i,i))*X(i,:)
          X(j,:)=X(j,:)-lineX
       end do   
    end do

    !reduce opposite way (no switch)
    do i = m,2,-1 !don't need to reduce other lines (i>1) using the first line (i=1)
       do j=i-1,1,-1
          lineX=(X(j,i)/X(i,i))*X(i,:)
          X(j,:)=X(j,:)-lineX
       end do
    end do

    do i=1,m
       !reduce line i so that X(i,i)=1
       X(i,:)=X(i,:)/X(i,i)
    end do

    X_out=X(:,m+1:)
    
    deallocate(X,lineX)
  end subroutine matrix_inv2D

  recursive subroutine get_det2D(X_in, det)
    real(8), dimension(:,:), intent(in)  :: X_in
    real(8),                 intent(out) :: det
    real(8), dimension(:,:), allocatable :: X_temp
    real(8) :: det_i

    integer(kind=4)   :: shape_X(2), m, n ! Integers for shape of matrices
    integer(kind=4)   :: i, j             ! Integers for looping
    character(len=32) :: M_str, partxt

    shape_X=shape(X_in)
    m=shape_X(1)
    n=shape_X(2)

    det=0.d0
    if (m /= n) then
       return
    elseif (m==2) then
       det=X_in(1,1)*X_in(2,2)-X_in(1,2)*X_in(2,1)
    elseif (m==1) then
       det=X_in(1,1)
    else
       if (m>10) then
          write(partxt,'(I10)') m
          partxt=adjustl(partxt)
          M_str = '('//trim(partxt)//'x'//trim(partxt)//')'
          write(*,*) 'WARNING: calculating determinant of a large matrix '//trim(M_str)//', this might take a very long time!'
       end if
       allocate(X_temp(m-1,m-1))
       do i = 1,m
          X_temp(:,1:i-1)=X_in(2:,1:i-1)
          X_temp(:,i:(m-1))=X_in(2:,i+1:m)
          call get_det2D(X_temp,det_i)
          det = det + ((-1)**(i+1))*X_in(1,i)*det_i
       end do
       deallocate(X_temp)
    end if

  end subroutine get_det2D


  subroutine test_linalg
    real(8), allocatable, dimension(:,:) :: X,Xi,XX
    real(8) :: det
    allocate(X(3,3),Xi(3,3),XX(3,3))

    X(1,1)=1.d0
    X(1,2)=0.d0
    X(1,3)=0.d0
    X(2,1)=0.d0
    X(2,2)=2.d0
    X(2,3)=1.d0
    X(3,1)=0.d0
    X(3,2)=2.d0
    X(3,3)=3.d0
    write(*,*) X(1,:)
    write(*,*) X(2,:)
    write(*,*) X(3,:)

    write(*,*) ''

    call get_det2D(X,det)

    write(*,*) ''
    write(*,*) 'det',det

    call matrix_inv2D(X,Xi)
    write(*,*) ''
    write(*,*) Xi(1,:)
    write(*,*) Xi(2,:)
    write(*,*) Xi(3,:)

    call matrix_mult2D(X,Xi,XX)
    write(*,*) ''
    write(*,*) XX(1,:)
    write(*,*) XX(2,:)
    write(*,*) XX(3,:)

    deallocate(X,Xi,XX)
  end subroutine test_linalg

end module linalg
