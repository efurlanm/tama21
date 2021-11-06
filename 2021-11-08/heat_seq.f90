
! computationally intensive core
subroutine kernel(anew, aold, sizeStart, sizeEnd)
    double precision, intent(out) :: anew(:,:)
    double precision, intent(in)  :: aold(:,:)
    integer, intent(in) :: sizeStart, sizeEnd
    integer             :: i, j

    do j = sizeStart, sizeEnd
        do i = sizeStart, sizeEnd
            anew(i,j)= aold(i,j)/2.0  &
                +(aold(i-1,j)+aold(i+1,j)+aold(i,j-1)+aold(i,j+1))/8.0
        enddo
    enddo   
end subroutine

! main routine
program stencil
    implicit none
    interface
        subroutine kernel(anew, aold, sizeStart, sizeEnd)
            double precision, intent(out) :: anew(:,:)
            double precision, intent(in)  :: aold(:,:)
            integer, intent(in) :: sizeStart, sizeEnd
        end subroutine
    end interface
    
    ! parameters for calculation
    integer             :: n=2400     ! n x n grid
    integer             :: energy=1   ! energy to be injected per iteration
    integer             :: niters=250 ! number of iterations

    ! other variables
    integer, parameter  :: nsources=3
    integer, dimension(3, 2)        :: sources
    double precision, allocatable   :: aold(:,:), anew(:,:)
    integer             :: iters, i, j, size, sizeStart, sizeEnd
    double precision    :: heat=0.0           
    
    size = n + 2
    sizeStart = 2
    sizeEnd = n + 1

    allocate(aold(size, size))
    allocate(anew(size, size))
    aold = 0.0
    anew = 0.0

    sources(1,:) = (/ n/2,   n/2   /)
    sources(2,:) = (/ n/3,   n/3   /)
    sources(3,:) = (/ n*4/5, n*8/9 /)

    do iters = 1, niters, 2
        
        ! odd iteration: anew <- stencil(aold)
            
        ! computationally intensive core
        call kernel(anew, aold, sizeStart, sizeEnd)
        
        ! three-point energy insertion
        do i = 1, nsources
            anew(sources(i,1)+1, sources(i,2)+1) =  &
                anew(sources(i,1)+1, sources(i,2)+1) + energy
        enddo

        ! even iteration: aold <- stencil(anew)
           
        ! computationally intensive core
        call kernel(aold, anew, sizeStart, sizeEnd)

        ! three-point energy insertion       
        do i = 1, nsources
            aold(sources(i,1)+1, sources(i,2)+1) =  &
                aold(sources(i,1)+1, sources(i,2)+1) + energy
        enddo

    enddo

    ! sum of grid points to get total heat
    heat = 0.0
    do j = sizeStart, sizeEnd
        do i = sizeStart, sizeEnd
            heat = heat + aold(i,j)
        end do
    end do

    ! show de result
    write(*, "('Heat: ' f0.4)") heat

    deallocate(aold)
    deallocate(anew)

end
