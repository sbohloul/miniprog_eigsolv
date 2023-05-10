program scalapack_helloworld
   use mpi_f08
   implicit none

   integer :: ictxt, nprocs, nprow, npcol
   integer :: myrank, myprow, mypcol, icaller
   integer :: hisprow, hispcol
   integer :: blacs_pnum
   integer :: i, j

   ! Number of processes and my process number
   call blacs_pinfo(myrank, nprocs)

   ! Process grid as close to square as possible
   nprow = int(sqrt(real(nprocs)))
   npcol = nprocs / nprow

   if (myrank .eq. 0) then
      write(*, "('Total number of processe: ', I2)") nprocs
      write(*, "('process grid = {', I2, ',', I2, '}')") nprow, npcol
   endif

   ! Get default system context and define process grid
   call blacs_get(0, 0, ictxt)
   call blacs_gridinit(ictxt, 'Row', nprow, npcol)

   ! Get grid info
   call blacs_gridinfo(ictxt, nprow, npcol, myprow, mypcol)

   ! Only processes in the grid do this
   if (((myprow .ge. 0) .and. (mypcol .ge. 0))) then
      ! Get my process number
      icaller = blacs_pnum(ictxt, myprow, mypcol)

      ! Process {0, 0} recieves check-in messages from all nodes
      if ((myprow .eq. 0) .and. (mypcol .eq. 0)) then
         do i = 0, nprow - 1
            do j = 0, npcol - 1

               ! recieve info sent from {i, j} to {0, 0}
               if ((i .ne. 0) .or. (j .ne. 0)) then
                  call igerv2d(ictxt, 1, 1, icaller, 1, i, j)
               end if

               ! Check info is correct
               call blacs_pcoord(ictxt, icaller, hisprow, hispcol)
               if ((hisprow .ne. i) .and. (hispcol .ne. j)) then
                  error stop "Error in process grid."
               end if

               ! Print info
               write(*, "('icaller = ', I2, ' I = ', I2, ' J = ', I2)") &
                  icaller, i, j

            end do
         end do
         write(*, *) "All processes checked in."
      else
         ! sent info to {0, 0}
         call igesd2d(ictxt, 1, 1, icaller, 1, 0, 0)
      end if
   end if

   ! Release all related parameters
   call blacs_exit(0)

end program scalapack_helloworld
