MODULE Mask_Manager_Mod
    USE openacc  
    ! ! USE Precision_Mod   
    IMPLICIT NONE

    INTEGER(1), ALLOCATABLE, PUBLIC :: mask_packed(:,:,:)

CONTAINS

    SUBROUTINE pack_grid_masks(isize, jsize, ksize, &
                               mask3d, surface_flag, surface_k, &
                               umask, vmask, wmask)
        
        INTEGER, INTENT(IN) :: isize, jsize, ksize
        INTEGER(1), INTENT(IN) :: mask3d(isize, jsize, ksize)
        LOGICAL,    INTENT(IN) :: surface_flag(isize, jsize)
        INTEGER,    INTENT(IN) :: surface_k(isize, jsize)
        INTEGER(1), INTENT(IN) :: umask(isize, jsize, ksize)
        INTEGER(1), INTENT(IN) :: vmask(isize, jsize, ksize)
        INTEGER(1), INTENT(IN) :: wmask(isize, jsize, ksize)

        INTEGER :: i, j, k
        INTEGER(1) :: val

        IF (ALLOCATED(mask_packed)) THEN
            !$acc exit data delete(mask_packed)
            DEALLOCATE(mask_packed)
        END IF
        
        ALLOCATE(mask_packed(-1:isize+2, -1:jsize+2, -1:ksize+2))
        mask_packed = 0_1

        ! --- OpenMP 並行計算 (CPU) ---
        !$OMP PARALLEL DO COLLAPSE(3) PRIVATE(val)
        DO k = 1, ksize
           DO j = 1, jsize
              DO i = 1, isize
                 
                 IF (mask3d(i, j, k) == 0) THEN
                    mask_packed(i, j, k) = 0_1
                    CYCLE
                 END IF

                 val = 0_1
                 
                 ! Bit 7: Active
                 val = IBSET(val, 7)

                 ! Bit 6: Surface
                 IF (surface_flag(i, j) .AND. k == surface_k(i, j)) val = IBSET(val, 6)

                 ! Bit 0-5: Faces
                 IF (umask(i, j, k)   == 1) val = IBSET(val, 0) ! +X
                 IF (umask(i-1, j, k) == 1) val = IBSET(val, 1) ! -X
                 IF (vmask(i, j, k)   == 1) val = IBSET(val, 2) ! +Y
                 IF (vmask(i, j-1, k) == 1) val = IBSET(val, 3) ! -Y
                 IF (wmask(i, j, k)   == 1) val = IBSET(val, 4) ! +Z
                 IF (wmask(i, j, k-1) == 1) val = IBSET(val, 5) ! -Z
                 
                 mask_packed(i, j, k) = val
              END DO
           END DO
        END DO
        !$OMP END PARALLEL DO

        !$acc enter data copyin(mask_packed)
        
    END SUBROUTINE pack_grid_masks


    SUBROUTINE finalize_pack_grid
        IF (ALLOCATED(mask_packed)) THEN
            !$acc exit data delete(mask_packed)
            DEALLOCATE(mask_packed)
        END IF
    END SUBROUTINE finalize_pack_grid

END MODULE Mask_Manager_Mod